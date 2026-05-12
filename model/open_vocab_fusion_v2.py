"""
Open Vocabulary 3D Fusion Model V2 with support for precomputed features.

This model supports two modes:
1. Online extraction: Extract LSeg + ODISE features on-the-fly (slow, for inference)
2. Precomputed features: Use pre-extracted features (fast, for training)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.pc_net import PC_Processor
from model.modeling import ODISEPixelMaskFusionNet, pad_mask_embeddings, pad_mask_tensors


DEFAULT_LOGIT_SCALE = 1.0 / 0.07


@dataclass
class OpenVocabFusionModelV2Config:
    """Model configuration."""
    device: str = "cuda"
    threshold: float = 0.5
    mask_embedding_dim: int = 256
    pixel_embedding_dim: int = 512
    fused_embedding_dim: int = 256
    alpha_max: float = 0.5
    # Optional paths for online extraction (only needed if not using precomputed)
    label_path: Optional[str] = None
    lseg_ckpt_path: Optional[str] = None
    odise_model_config_path: Optional[str] = None
    # Architecture
    pc_arch: str = "MinkUNet34C"
    pc_last_dim: int = 256


class OpenVocab3DFusionModelV2(nn.Module):
    """
    Open Vocabulary 3D Fusion Model.
    
    Fuses mask-level embeddings (ODISE) and pixel-level embeddings (LSeg)
    to produce robust 3D mask predictions.
    """

    def __init__(self, config: OpenVocabFusionModelV2Config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.threshold = config.threshold

        # 3D Point cloud processor
        self.pc_processor = PC_Processor(
            decoder_proj_out_dim=config.fused_embedding_dim,
            last_dim=config.pc_last_dim,
            arch_3d=config.pc_arch,
        )

        # 2D feature fusion network
        self.fuse_embed = ODISEPixelMaskFusionNet(
            pixel_dim=config.pixel_embedding_dim,
            mask_dim=config.mask_embedding_dim,
            out_dim=config.fused_embedding_dim,
            alpha_max=config.alpha_max,
        )

        # Learnable temperature for similarity
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(DEFAULT_LOGIT_SCALE)
        )

        # Optional: online extractors (lazy initialization)
        self._pix_extractor = None
        self._mask_extractor = None

    @property
    def pix_extractor(self):
        """Lazy load LSeg extractor."""
        if self._pix_extractor is None:
            if self.config.label_path is None or self.config.lseg_ckpt_path is None:
                raise RuntimeError(
                    "Online extraction requires label_path and lseg_ckpt_path"
                )
            from lang_seg import lseg_feature as lf
            self._pix_extractor = lf.LSegExtractor(
                self.config.label_path, self.config.lseg_ckpt_path
            )
            self._pix_extractor.eval()
        return self._pix_extractor

    @property
    def mask_extractor(self):
        """Lazy load ODISE extractor."""
        if self._mask_extractor is None:
            if self.config.odise_model_config_path is None:
                raise RuntimeError(
                    "Online extraction requires odise_model_config_path"
                )
            from ODISE import odise_feature as of
            self._mask_extractor = of.ODISEMaskEmbeddingExtractor(
                self.config.odise_model_config_path
            )
        return self._mask_extractor

    def _extract_2d_features_online(
        self, images: torch.Tensor
    ) -> tuple:
        """Extract 2D features on-the-fly."""
        batch_size = images.shape[0]
        pixel_embeddings = []
        mask_embeddings = []
        mask_tensors = []

        for i in range(batch_size):
            with torch.no_grad():
                pixel_embedding = torch.from_numpy(
                    self.pix_extractor(images[i])
                ).to(self.device, non_blocking=True)
                mask_features = self.mask_extractor.extract(images[i])

            pixel_embeddings.append(pixel_embedding)

            if len(mask_features["masks"]) == 0:
                height, width = images[i].shape[:2]
                empty_mask = torch.empty(
                    0, height, width, device=self.device, dtype=torch.bool
                )
                empty_embed = torch.empty(
                    0, self.config.mask_embedding_dim, device=self.device
                )
                mask_tensors.append(empty_mask)
                mask_embeddings.append(empty_embed)
            else:
                mask_tensors.append(torch.stack(mask_features["masks"]))
                mask_embeddings.append(mask_features["mask_embeddings"])

        pixel_embeddings = torch.stack(pixel_embeddings).float()
        mask_tensors, mask_valid_from_masks = pad_mask_tensors(mask_tensors)
        mask_embeddings, mask_valid = pad_mask_embeddings(mask_embeddings)

        return pixel_embeddings, mask_tensors, mask_embeddings, mask_valid, mask_valid_from_masks

    def forward(
        self, batch_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Supports both precomputed features and online extraction.
        
        Args:
            batch_input: Dict with keys:
                - sinput: SparseTensor for 3D points
                - inds_reconstruct: Point reconstruction indices
                - ori_coords_3d: Original 3D coordinates with batch index
                
                For precomputed features:
                - pixel_embeddings: (B, H, W, C) LSeg features
                - masks: (B, K, H, W) ODISE masks
                - mask_embeddings: (B, K, C) ODISE embeddings
                - mask_valid: (B, K) validity mask
                
                For online extraction:
                - img: (B, H, W, 3) RGB images
        """
        # Check if using precomputed features (pixel-level or pre-pooled from npz)
        use_precomputed = (
            ("pixel_embeddings" in batch_input or "pixel_pooled" in batch_input)
            and "mask_embeddings" in batch_input
            and "masks" in batch_input
        )

        if use_precomputed:
            # 确保数据类型一致性，避免 AMP 训练中的类型问题
            mask_tensors = batch_input["masks"].float()
            mask_embeddings = batch_input["mask_embeddings"].float()
            mask_valid = batch_input.get("mask_valid", None)
            mask_valid_from_masks = mask_valid  # Same for precomputed
            # Pixel: (B,H,W,C) from *_lseg.npy, or (B,K,C) pre-pooled from npz pixel_pooled.
            # LSeg remains 512D at input; fuse_embed projects it into ODISE's 256D space.
            if "pixel_pooled" in batch_input:
                pixel_embeddings = batch_input["pixel_pooled"].float()  # (B,K,512)
            else:
                pixel_embeddings = batch_input["pixel_embeddings"].float()  # (B,H,W,512)
            
            # 验证预计算数据的有效性
            if mask_tensors.numel() == 0 or mask_embeddings.numel() == 0:
                raise ValueError("Empty precomputed masks or mask_embeddings")
            if pixel_embeddings.numel() == 0:
                raise ValueError("Empty precomputed pixel_embeddings or pixel_pooled")
        else:
            # Online extraction
            if "img" not in batch_input:
                raise KeyError("Missing 'img' for online feature extraction")
            (
                pixel_embeddings,
                mask_tensors,
                mask_embeddings,
                mask_valid,
                mask_valid_from_masks,
            ) = self._extract_2d_features_online(batch_input["img"])

        # Validate dimensions
        batch_size = pixel_embeddings.shape[0]
        # pixel_embeddings may be (B,H,W,C) or (B,K,C); last dim must be pixel_embedding_dim
        if pixel_embeddings.shape[-1] != self.config.pixel_embedding_dim:
            raise ValueError(
                f"pixel embedding dim mismatch: "
                f"{pixel_embeddings.shape[-1]} != {self.config.pixel_embedding_dim}"
            )
        if mask_embeddings.shape[-1] != self.config.mask_embedding_dim:
            raise ValueError(
                f"mask embedding dim mismatch: "
                f"{mask_embeddings.shape[-1]} != {self.config.mask_embedding_dim}"
            )

        # Fuse 2D features
        fusion_out = self.fuse_embed(
            pixel_embeddings,
            mask_embeddings,
            mask_tensors,
            mask_valid,
            return_aux=True,
        )
        fused_embeddings = fusion_out["fused"]
        pixel_pooled_embeddings = self._pool_pixel_embeddings_for_eval(
            pixel_embeddings, mask_tensors, mask_valid
        )

        # Process 3D points（MinkowskiEngine 体素化后输出点数 M 可能 < 输入点数 N）
        # 🔥 关键修复：正确映射每个输入点到其对应的体素特征
        implicit_condition, pred_3d_voxel, _ = self.pc_processor(batch_input["sinput"])
        sinput = batch_input["sinput"]
        
        # 目标：为每个输入点找到其量化后对应的体素索引
        
        input_coords = batch_input["coords_3d"].int().to(sinput.device)  # (N_total, 4)
        voxel_coords = sinput.C  # (M_voxels, 4)
        
        # 使用向量化哈希映射
        def _encode_coords(coords_4d):
            c = coords_4d.long() + 20000  # 偏移确保非负
            BASE = 40001  # 大于最大坐标范围
            return c[:, 0] * (BASE ** 3) + c[:, 1] * (BASE ** 2) + c[:, 2] * BASE + c[:, 3]
        
        voxel_hash = _encode_coords(voxel_coords)
        input_hash = _encode_coords(input_coords)
        
        # 排序 voxel_hash 以使用 searchsorted
        sort_idx = torch.argsort(voxel_hash)
        sorted_voxel_hash = voxel_hash[sort_idx]
        
        # 查找每个 input 点的 voxel 索引
        pos = torch.searchsorted(sorted_voxel_hash, input_hash)
        pos = pos.clamp(max=sorted_voxel_hash.shape[0] - 1)
        
        # 检查是否真正匹配
        matched = sorted_voxel_hash[pos] == input_hash
        point_to_voxel_idx = sort_idx[pos]
        
        # 验证 point→voxel 映射是否匹配
        matched_ratio = matched.float().mean().item()
        if matched_ratio < 0.999:
            print(f'[FATAL] point→voxel matched_ratio={matched_ratio:.6f}')
            print(f'  input_coords shape: {input_coords.shape}')
            print(f'  voxel_coords shape: {voxel_coords.shape}')
            print(f'  input_coords range: batch={input_coords[:, 0].unique()}, '
                  f'x=[{input_coords[:, 1].min()},{input_coords[:, 1].max()}], '
                  f'y=[{input_coords[:, 2].min()},{input_coords[:, 2].max()}], '
                  f'z=[{input_coords[:, 3].min()},{input_coords[:, 3].max()}]')
            print(f'  voxel_coords range: batch={voxel_coords[:, 0].unique()}, '
                  f'x=[{voxel_coords[:, 1].min()},{voxel_coords[:, 1].max()}], '
                  f'y=[{voxel_coords[:, 2].min()},{voxel_coords[:, 2].max()}], '
                  f'z=[{voxel_coords[:, 3].min()},{voxel_coords[:, 3].max()}]')
            num_unmatched = (~matched).sum().item()
            print(f'  未匹配点数: {num_unmatched}/{input_coords.shape[0]} ({100*(1-matched_ratio):.3f}%)')
            raise RuntimeError('point→voxel mapping mismatch: 未匹配的点会被错误映射到 voxel[0]')
        
        point_to_voxel_idx[~matched] = 0  # 未匹配的用 0 兜底（理论上不会执行到）
        
        # 用正确的映射索引体素特征：每个点 → 其体素索引 → 体素特征
        pred_3d = pred_3d_voxel[point_to_voxel_idx, :].float()

        # Compute per-point, per-mask similarity
        batch_indices = batch_input["ori_coords_3d"][:, 0].long()
        outputs = [[] for _ in range(batch_size)]

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        for b in range(batch_size):
            point_mask = batch_indices == b
            if not torch.any(point_mask):
                continue

            mask_tokens = fused_embeddings[b].float()
            if mask_valid is not None:
                valid_mask = mask_valid[b].to(mask_tokens.device)
                if not valid_mask.any():
                    print(f"Warning: No valid masks for batch {b}, skipping")
                    continue
                mask_tokens = mask_tokens[valid_mask]
            if mask_tokens.numel() == 0:
                print(f"Warning: Empty mask_tokens for batch {b}, skipping")
                continue

            # Match Diff2Scene/text evaluation: optimize angular similarity, not feature norms.
            point_features = F.normalize(pred_3d[point_mask], dim=-1)
            mask_tokens_norm = F.normalize(mask_tokens, dim=-1)
            logits = logit_scale * (point_features @ mask_tokens_norm.t())
            mask_tokens_detached_norm = F.normalize(mask_tokens.detach(), dim=-1)
            logits_detached = logit_scale * (point_features @ mask_tokens_detached_norm.t())

            # Expand to full mask count
            if mask_valid is not None:
                num_masks = fused_embeddings.shape[1]
                full_logits = pred_3d.new_full(
                    (logits.shape[0], num_masks), float("-inf")
                )
                full_logits_detached = pred_3d.new_full(
                    (logits_detached.shape[0], num_masks), float("-inf")
                )
                full_logits[:, valid_mask] = logits.to(full_logits.dtype)
                full_logits_detached[:, valid_mask] = logits_detached.to(full_logits_detached.dtype)
                logits = full_logits
                logits_detached = full_logits_detached

            # 传 logits 给 criterion，便于 AMP 下用 BCEWithLogitsLoss（autocast 安全）
            outputs[b].append({
                "pred_mask_logits": logits,
                "pred_mask_logits_detached_teacher": logits_detached,
            })

        return {
            "outputs": outputs,
            "mask_valid_from_masks": mask_valid_from_masks,
            "mask_masks": mask_tensors,
            "batch_indices": batch_indices,
            "fused_embeddings": fused_embeddings,
            "pixel_pooled_embeddings": batch_input.get("clip_pooled", pixel_pooled_embeddings).float(),
            "pred_3d": pred_3d,
            "fusion_aux": fusion_out,
        }

    def _pool_pixel_embeddings_for_eval(
        self,
        pixel_embeddings: torch.Tensor,
        mask_tensors: torch.Tensor,
        mask_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return raw CLIP/LSeg mask-pooled features for ODISE Eq.10 evaluation."""
        if pixel_embeddings.dim() == 3:
            return pixel_embeddings.float()
        if pixel_embeddings.dim() != 4:
            raise RuntimeError(
                f"pixel_embeddings must be (B,K,C) or (B,H,W,C), got {tuple(pixel_embeddings.shape)}"
            )

        B, H, W, C = pixel_embeddings.shape
        Bm, K, Hm, Wm = mask_tensors.shape
        if B != Bm:
            raise RuntimeError(f"batch mismatch: pixel B={B}, mask B={Bm}")
        masks = mask_tensors.float()
        if H != Hm or W != Wm:
            masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
        if mask_valid is None:
            valid = torch.ones(B, K, dtype=torch.bool, device=masks.device)
        else:
            valid = mask_valid.to(masks.device)
        weights = masks * valid.unsqueeze(-1).unsqueeze(-1).float()
        denom = weights.sum(dim=(-1, -2)).clamp_min(1.0)
        pooled = torch.matmul(weights.view(B, K, H * W), pixel_embeddings.view(B, H * W, C))
        pooled = pooled / denom.unsqueeze(-1)
        return pooled.float()

    def freeze_extractors(self):
        """Freeze 2D feature extractors for training."""
        if self._pix_extractor is not None:
            for param in self._pix_extractor.parameters():
                param.requires_grad = False
        if self._mask_extractor is not None:
            for param in self._mask_extractor.model.parameters():
                param.requires_grad = False
