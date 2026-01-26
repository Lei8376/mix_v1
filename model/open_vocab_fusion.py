from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.pc_net import PC_Processor
from model.modeling import ODISEPixelMaskFusionNet, pad_mask_embeddings, pad_mask_tensors
from lang_seg import lseg_feature as lf
from ODISE import odise_feature as of


DEFAULT_LOGIT_SCALE = 1.0 / 0.07


@dataclass
class OpenVocabFusionModelConfig:
    label_path: str
    lseg_ckpt_path: str
    odise_model_config_path: str
    device: str = "cuda"
    threshold: float = 0.5
    mask_embedding_dim: int = 256
    pixel_embedding_dim: int = 512
    fused_embedding_dim: int = 768


class OpenVocab3DFusionModel(nn.Module):
    def __init__(self, config: OpenVocabFusionModelConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.threshold = config.threshold

        self.pc_processor = PC_Processor()
        self.pix_extractor = lf.LSegExtractor(
            config.label_path, config.lseg_ckpt_path
        )
        self.mask_extractor = of.ODISEMaskEmbeddingExtractor(
            config.odise_model_config_path
        )
        self.pix_extractor.eval()

        self.fuse_embed = ODISEPixelMaskFusionNet(
            pixel_dim=config.pixel_embedding_dim,
            mask_dim=config.mask_embedding_dim,
            out_dim=config.fused_embedding_dim,
        )
        self.logit_scale = nn.Parameter(
            torch.ones([], device=self.device) * np.log(DEFAULT_LOGIT_SCALE)
        )

    def forward(self, batch_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required_keys = [
            "img",
            "sinput",
            "inds_reconstruct",
            "ori_coords_3d",
        ]
        for key in required_keys:
            if key not in batch_input:
                raise KeyError(f"Missing required batch_input key: {key}")

        batch_size = batch_input["img"].shape[0]
        assert batch_size > 0, "batch_size must be positive."

        pixel_embeddings: List[torch.Tensor] = []
        mask_embeddings: List[torch.Tensor] = []
        mask_tensors: List[torch.Tensor] = []

        for i in range(batch_size):
            with torch.no_grad():
                pixel_embedding = torch.from_numpy(
                    self.pix_extractor(batch_input["img"][i])
                ).to(self.device, non_blocking=True)
                mask_features = self.mask_extractor.extract(batch_input["img"][i])

            pixel_embeddings.append(pixel_embedding)

            if len(mask_features["masks"]) == 0:
                height, width = batch_input["img"][i].shape[:2]
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
        if pixel_embeddings.shape[-1] != self.config.pixel_embedding_dim:
            raise ValueError(
                "pixel embedding dim mismatch: "
                f"{pixel_embeddings.shape[-1]} != {self.config.pixel_embedding_dim}"
            )
        mask_tensors, mask_valid_from_masks = pad_mask_tensors(mask_tensors)
        mask_embeddings, mask_valid = pad_mask_embeddings(mask_embeddings)
        if mask_embeddings.shape[-1] != self.config.mask_embedding_dim:
            raise ValueError(
                "mask embedding dim mismatch: "
                f"{mask_embeddings.shape[-1]} != {self.config.mask_embedding_dim}"
            )

        fused_embeddings = self.fuse_embed(
            pixel_embeddings, mask_embeddings, mask_tensors, mask_valid
        )

        implicit_condition, pred_3d, _ = self.pc_processor(batch_input["sinput"])
        pred_3d = pred_3d[batch_input["inds_reconstruct"], :].float()

        batch_indices = batch_input["ori_coords_3d"][:, 0].long()
        outputs = [[] for _ in range(batch_size)]

        for batch_index in range(batch_size):
            point_mask = batch_indices == batch_index
            if not torch.any(point_mask):
                continue

            mask_tokens = fused_embeddings[batch_index].float()
            if mask_valid is not None:
                valid_mask = mask_valid[batch_index].to(mask_tokens.device)
                mask_tokens = mask_tokens[valid_mask]
            if mask_tokens.numel() == 0:
                continue

            logit_scale = self.logit_scale.exp().clamp(max=100.0)
            pred_3d[point_mask] = F.normalize(pred_3d[point_mask], dim=-1)
            mask_tokens = F.normalize(mask_tokens, dim=-1)
            logits = logit_scale * (pred_3d[point_mask] @ mask_tokens.t())

            if mask_valid is not None:
                num_masks = fused_embeddings.shape[1]
                full_logits = pred_3d.new_full(
                    (logits.shape[0], num_masks), float("-inf")
                )
                full_logits[:, valid_mask] = logits
                logits = full_logits

            mask_logits = torch.sigmoid(logits)
            outputs[batch_index].append({"pred_mask_logits": mask_logits})

        return {
            "outputs": outputs,
            "mask_valid_from_masks": mask_valid_from_masks,
            "mask_masks": mask_tensors,
            "batch_indices": batch_indices,
        }

