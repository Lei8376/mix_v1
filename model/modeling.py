import logging
import numpy as np
import operator
from collections import OrderedDict
from typing import Any, Mapping
#import diffdist.functional as diff_dist
import torch
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.memory import retry_if_cuda_oom
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MLP,
    MultiScaleMaskedTransformerDecoder,
)

from .pc_net import PC_Processor, PC_Binary_Processor

from lang_seg import lseg_feature as lf
# ODISE 仅在 MaskStablePixle 中使用；V2 用预计算 npz 时不加载，避免 ModuleNotFoundError: odise
from torch import nn
from torch.nn import functional as F


class MaskStablePixle(nn.Module): 
    def __init__(self, lsg_label_path, lsg_ckpt_path, odise_model_config_path, threshold=0.5, device="cuda"):
        super(MaskStablePixle, self).__init__()
        from ODISE import odise_feature as of
        self.lsg_label_path = lsg_label_path
        self.lsg_ckpt_path = lsg_ckpt_path
        self.threshold = threshold
        self.odise_model_config_path = odise_model_config_path
        self.pc_processor = PC_Processor()
        self.pix_extractor = lf.LSegExtractor(lsg_label_path, lsg_ckpt_path)
        self.mask_extractor = of.ODISEMaskEmbeddingExtractor(odise_model_config_path)
        self.pix_extractor.eval()
        #self.pixel_reliability = PixelReliability(dim=512)
        #self.mask_token_projector = MaskTokenProjector(in_dim=256, out_dim=512)
        #self.pixel_mask_attention = PixelMaskAttention(dim=512)
        #self.mask_spatial_aggregator = MaskSpatialAggregator()
        #self.adaptive_fusion = AdaptiveFusion(dim=512)
        self.fuse_embed = ODISEPixelMaskFusionNet(pixel_dim=512, mask_dim=256, out_dim=768)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        #print(self.logit_scale)
        self.device = device

    def forward(self, batch_input):
        pix_feats = []
        mask_feats = []
        mask_embeds = []
        mask_masks = []
        for i in range(batch_input["img"].shape[0]):
            #TODO: 1. save the pixel embedding and mask embedding into local
            with torch.no_grad():
                pix_feat = torch.from_numpy(
                    self.pix_extractor(batch_input["img"][i])
                ).to(self.device, non_blocking=True)
                mask_feat = self.mask_extractor.extract(batch_input["img"][i])
            #print("mask_feat keys:", mask_feat.keys())
            #Eprint(mask_feat["results"]) the detail of masks,, including the mask_embedding, category_name, category_id, is_thing, score, area
            pix_feats.append(pix_feat)
            mask_feats.append(mask_feat)
            #print(mask_feat)
            #print(mask_feat["mask_embeddings"].shape)# list type
            #print(mask_feat['masks'][0].shape)
            #print("mask_feat['masks']: ", mask_feat["results"])
            
            # Handle empty mask list (no masks detected for this image)
            if len(mask_feat["masks"]) == 0:
                # Get image dimensions from input
                H, W = batch_input["img"][i].shape[:2]
                mask_ = torch.empty(0, H, W, device=self.device)
            else:
                mask_ = torch.stack(mask_feat["masks"])
            #print("masks genereted:", mask_.shape)
            #print(mask_.shape)
            mask_masks.append(mask_)
            mask_embeds.append(mask_feat['mask_embeddings'])

        #print("xxxxxxxxxxx")
        #mask_masks = torch.cat(mask_masks.unsqueeze(dim=0), dim=0)
        #print("mask_masks: ", mask_masks.shape)
        pix_feats = torch.stack(pix_feats).float()  # Convert to float32 for linear layers
        mask_masks, mask_valid_from_masks = pad_mask_tensors(mask_masks)  # Pad masks to same size batch,k,h,w
        #print("mask_masks: ", mask_masks.shape)
        mask_embeds, mask_valid = pad_mask_embeddings(mask_embeds)
        fused_embed = self.fuse_embed(pix_feats, mask_embeds, mask_masks, mask_valid)#(batch, num_mask, feat_dim)

        #3d point net 
        imp_condition, pred_3d, _idx = self.pc_processor(batch_input["sinput"])
        #pred_3d the point feature
        #print(pred_3d.shape)
        pred_3d = pred_3d[batch_input["inds_reconstruct"], :].float()
        #print("pred_3d:", pred_3d.shape)

        # Compute inner-product similarity logits per point and per mask
        #print("coords_3d:", batch_input["ori_coords_3d"].shape)
        batch_indices = batch_input["ori_coords_3d"][:, 0].long()
      
        #print("batch_indices:", batch_indices.shape)
        batch_size, num_masks, feat_dim = fused_embed.shape
        min_points_per_mask = 10
        results = {}
        outputs = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            point_mask = batch_indices == b
            #print("point_mask:", point_mask)
            #print("point_mask:", point_mask.shape)
            if not torch.any(point_mask):
                continue
            mask_tokens = fused_embed[b].float()  # (K, C)
            if mask_valid is not None:
                valid = mask_valid[b].to(mask_tokens.device)
                mask_tokens = mask_tokens[valid]#(num_valid_mask, feat_dim)
                #print(mask_tokens.shape)
            logit_scale = self.logit_scale.exp().clamp(max=100)
            pred_3d[point_mask] = F.normalize(pred_3d[point_mask], dim=-1)
            mask_tokens = F.normalize(mask_tokens, dim=-1)
            logits = logit_scale * (pred_3d[point_mask] @ mask_tokens.t())
            #print("logits:" , logits.shape)
            if mask_valid is not None:
                full_logits = pred_3d.new_full((logits.shape[0], num_masks), float("-inf"))
                full_logits[:, valid] = logits
                #print("valid:", valid)
                #print(torch.sum(full_logits, dim=0)!=float("-inf"))
                logits = full_logits
            #print(point_mask.shape)
            #here the outputt should be logits

            mask_logits = torch.sigmoid(logits)
            outputs[b].append({"pred_mask_logits": mask_logits})

        results["outputs"] = outputs
        results["mask_valid_from_masks"] = mask_valid_from_masks
        results["mask_masks"] = mask_masks
        results["batch_indices"] = batch_indices
        return results
"""
            mask_pred = (mask_logits > self.threshold).float()
            keep = torch.sum(mask_pred, dim=0) > min_points_per_mask
            #print(keep)
            mask_preds = mask_pred[:,keep].transpose(0,1)# the final mask results
            #print(mask_preds.shape)
            #point_logits[point_mask] = logits
            print("mask_preds:", mask_preds.shape)
            outputs[b].append({"pred_mask_3d": mask_preds})

        print("+++++++")
        #the ground truth of 2d labels and masks - use ODISE masks as ground truth
        targets = [[] for _ in range(batch_size)]
        x_label = batch_input["x_label"]
        y_label = batch_input["y_label"]
        for idx in range(batch_size):
            # Use ODISE masks as ground truth instead of label_2d
            valid = mask_valid_from_masks[idx]  # (num_masks,) boolean tensor
            masks = mask_masks[idx][valid]  # (num_valid_masks, H, W)
            print("masks (ODISE GT): ", masks.shape)
            
            point_mask = batch_indices == idx
            x_idx = x_label[point_mask].long()
            y_idx = y_label[point_mask].long()
            if x_idx.numel() > 0:
                assert x_idx.max().item() < masks.shape[2]
                assert y_idx.max().item() < masks.shape[1]
            masks_3d = masks[:, y_idx, x_idx]  # (K, N_points_in_batch)
            masks_3d = (masks_3d > self.threshold)
            print("masks_3d:", masks_3d.shape)
            #gt_mask_point_counts = masks_3d.to(torch.int64).sum(dim=1)
            #keep_gt_by_points = gt_mask_point_counts > min_points_per_mask
            #masks_3d = masks_3d[keep_gt_by_points]
            #masks = masks[keep_gt_by_points]
            #print("masks_3d (filtered): ", masks_3d.shape)
            #ground truth masks 
            targets[idx].append({
                "masks_2d": masks,
                "masks_3d": masks_3d,
            })

        #return fused_embed#, point_logits, mask_probs
        return outputs, targets
"""

class PixelReliability(nn.Module):
    """
    Estimate per-pixel confidence to supress nosiy pixel embeddings
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(dim//2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: (B,H,W,C)
        """
        B,H,W,C = x.shape
        conf = self.net(x.view(B*H*W,C))
        conf = conf.view(B,H,W,1)
        return conf
        
       
class MaskTokenProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, mask_embed):
        # mask_embed: (B,K,Cm)
        return self.proj(mask_embed)  # (B,K,C_out)


class PixelMaskAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, pixel_feat, mask_tokens, valid_mask):
        """
        pixel_feat: (B,H,W,C)
        mask_tokens:(B,K,C)
        valid_mask: (B,K) ∈ {0,1}
        """
        B, H, W, C = pixel_feat.shape

        Q = self.q(pixel_feat).view(B, H * W, C)
        K = self.k(mask_tokens)
        V = self.v(mask_tokens)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / (C ** 0.5)

        # 🔑 Mask out padded instances
        logits = logits.masked_fill(
            valid_mask.unsqueeze(1) == 0,
            float("-inf")
        )

        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, V)

        return out.view(B, H, W, C)


class MaskSpatialAggregator(nn.Module):
    def forward(self, mask_tokens, masks, valid_mask):
        """
        mask_tokens: (B,K,C)
        masks:       (B,K,H,W)
        valid_mask:  (B,K)
        """
        weights = masks * valid_mask.unsqueeze(-1).unsqueeze(-1)
        tokens = mask_tokens.unsqueeze(2).unsqueeze(2)

        spatial = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        norm = weights.sum(dim=1, keepdim=False) + 1e-6

        return spatial / norm.unsqueeze(-1)


class AdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, pixel_feat, mask_feat, confidence):
        g = self.net(torch.cat([pixel_feat, confidence], dim=-1))
        return g * pixel_feat + (1 - g) * mask_feat


class ODISEPixelMaskFusionNet(nn.Module):
    def __init__(self, pixel_dim, mask_dim=256, out_dim=768):
        super().__init__()

        self.pixel_proj = nn.Linear(pixel_dim, out_dim)
        self.mask_proj = MaskTokenProjector(mask_dim, out_dim)
        self.gate = nn.Linear(out_dim * 2, 1)
        self.refine = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        nn.init.constant_(self.gate.bias, -2.0)  # bias toward mask-dominant fusion

        # ODISE-residual fusion: final = mask_tokens + alpha * refine(mask_tokens + gate * pixel_tokens).
        # Keep alpha learnable/adaptive, matching the original mix2_v1 backup.
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, pixel_embed, mask_embed, masks, valid_mask):
        """
        pixel_embed: either (B, H, W, Cp) pixel-level features, or (B, K, Cp) pre-pooled per-mask features.
        Returns:
            fused_embed: (B, K, C_out)
        """
        assert mask_embed.dim() == 3, "mask_embed must be (B,K,C)"
        assert masks.dim() == 4, "masks must be (B,K,H,W)"
        assert valid_mask.dim() == 2, "valid_mask must be (B,K)"

        Bm, K, Hm, Wm = masks.shape

        mask_tokens = self.mask_proj(mask_embed)  # (B,K,C_out)

        if pixel_embed.dim() == 4:
            # Pixel-level (B,H,W,Cp): aggregate per mask then project
            B, H, W, Cp = pixel_embed.shape
            assert B == Bm, "batch size must match"
            if H != Hm or W != Wm:
                masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
            weights = masks.float() * valid_mask.unsqueeze(-1).unsqueeze(-1).float()
            denom = weights.sum(dim=(-1, -2)).clamp_min(1.0)  # (B,K)
            pixel_flat = pixel_embed.view(B, H * W, Cp)
            weights_flat = weights.view(B, K, H * W)
            pixel_pooled = torch.matmul(weights_flat, pixel_flat) / denom.unsqueeze(-1)  # (B,K,Cp)
        elif pixel_embed.dim() == 3:
            # Pre-pooled (B,K,Cp): skip aggregation, only project (e.g. from run_all_pixel_pooled npz)
            assert pixel_embed.shape[:2] == (Bm, K), "pixel_embed (B,K,Cp) must match masks batch and K"
            pixel_pooled = pixel_embed
        else:
            raise AssertionError("pixel_embed must be (B,H,W,C) or (B,K,C)")

        pixel_tokens = self.pixel_proj(pixel_pooled)  # (B,K,C_out)

        gate = torch.sigmoid(self.gate(torch.cat([mask_tokens, pixel_tokens], dim=-1)))
        delta = self.refine(mask_tokens + gate * pixel_tokens)
        fused = mask_tokens + self.alpha * delta

        fused = fused * valid_mask.unsqueeze(-1).float()
        return fused


def pad_mask_embeddings(mask_embeds_list):
    #pad mask_embeds to the same size
    max_masks = max(m.shape[0] for m in mask_embeds_list) 
    embed_dim = mask_embeds_list[0].shape[-1]

    padded_embeds = []
    mask_valid =[] #track which masks are real vs padding

    for m in mask_embeds_list:
        num_masks = m.shape[0]
        if num_masks<max_masks:
            padding = torch.zeros(max_masks-num_masks, embed_dim, device=m.device)
            padded = torch.cat([m, padding], dim=0)
        else:
            padded = m
        padded_embeds.append(padded)
        valid = torch.zeros(max_masks, dtype=torch.bool, device=m.device)
        valid[:num_masks] = True
        mask_valid.append(valid)
    mask_embeds = torch.stack(padded_embeds, dim=0).float()
    mask_valid = torch.stack(mask_valid)
    return mask_embeds, mask_valid


def pad_mask_tensors(mask_list):
    """Pad binary masks [K, H, W] to same K dimension across batch."""
    max_masks = max(m.shape[0] for m in mask_list)
    H, W = mask_list[0].shape[-2:]
    
    padded_masks = []
    mask_valid = []
    
    for m in mask_list:
        num_masks = m.shape[0]
        if num_masks < max_masks:
            padding = torch.zeros(max_masks - num_masks, H, W, device=m.device, dtype=m.dtype)
            padded = torch.cat([m, padding], dim=0)
        else:
            padded = m
        padded_masks.append(padded)
        valid = torch.zeros(max_masks, dtype=torch.bool, device=m.device)
        valid[:num_masks] = True
        mask_valid.append(valid)
    
    mask_masks = torch.stack(padded_masks).float()  # [B, max_masks, H, W]
    mask_valid = torch.stack(mask_valid)  # [B, max_masks]
    return mask_masks, mask_valid
