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
    fused_embedding_dim: int = 768
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
        # Check if using precomputed features
        use_precomputed = (
            "pixel_embeddings" in batch_input
            and "mask_embeddings" in batch_input
            and "masks" in batch_input
        )

        if use_precomputed:
            pixel_embeddings = batch_input["pixel_embeddings"].float()
            mask_tensors = batch_input["masks"].float()
            mask_embeddings = batch_input["mask_embeddings"].float()
            mask_valid = batch_input.get("mask_valid", None)
            mask_valid_from_masks = mask_valid  # Same for precomputed
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
        fused_embeddings = self.fuse_embed(
            pixel_embeddings, mask_embeddings, mask_tensors, mask_valid
        )

        # Process 3D points
        implicit_condition, pred_3d, _ = self.pc_processor(batch_input["sinput"])
        pred_3d = pred_3d[batch_input["inds_reconstruct"], :].float()

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
                mask_tokens = mask_tokens[valid_mask]
            if mask_tokens.numel() == 0:
                continue

            # Normalize and compute similarity
            point_features = F.normalize(pred_3d[point_mask], dim=-1)
            mask_tokens = F.normalize(mask_tokens, dim=-1)
            logits = logit_scale * (point_features @ mask_tokens.t())

            # Expand to full mask count
            if mask_valid is not None:
                num_masks = fused_embeddings.shape[1]
                full_logits = pred_3d.new_full(
                    (logits.shape[0], num_masks), float("-inf")
                )
                full_logits[:, valid_mask] = logits
                logits = full_logits

            mask_logits = torch.sigmoid(logits)
            outputs[b].append({"pred_mask_logits": mask_logits})

        return {
            "outputs": outputs,
            "mask_valid_from_masks": mask_valid_from_masks,
            "mask_masks": mask_tensors,
            "batch_indices": batch_indices,
            "fused_embeddings": fused_embeddings,
            "pred_3d": pred_3d,
        }

    def freeze_extractors(self):
        """Freeze 2D feature extractors for training."""
        if self._pix_extractor is not None:
            for param in self._pix_extractor.parameters():
                param.requires_grad = False
        if self._mask_extractor is not None:
            for param in self._mask_extractor.model.parameters():
                param.requires_grad = False

