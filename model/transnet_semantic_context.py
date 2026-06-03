from typing import Optional

import torch
import torch.nn as nn


class TransNetSemanticContext(nn.Module):
    """
    Lightweight TransNet-style semantic context branch.

    3D point features query ODISE mask-level tokens through cross-attention.
    The output is a residual context feature; it does not replace the 3D
    backbone feature.
    """

    def __init__(
        self,
        point_dim: int,
        odise_dim: int = 256,
        geom_dim: int = 0,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        max_context_masks: int = 64,
        use_geom: bool = False,
    ):
        super().__init__()
        self.point_dim = int(point_dim)
        self.odise_dim = int(odise_dim)
        self.geom_dim = int(geom_dim or 0)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.max_context_masks = int(max_context_masks)
        self.use_geom = bool(use_geom and self.geom_dim > 0)

        self.odise_proj = nn.Sequential(
            nn.Linear(self.odise_dim, self.point_dim),
            nn.LayerNorm(self.point_dim),
            nn.GELU(),
            nn.Linear(self.point_dim, self.point_dim),
        )

        if self.use_geom:
            self.geom_proj = nn.Sequential(
                nn.Linear(self.geom_dim, self.point_dim),
                nn.LayerNorm(self.point_dim),
                nn.GELU(),
                nn.Linear(self.point_dim, self.point_dim),
            )
        else:
            self.geom_proj = None

        self.query_norm = nn.LayerNorm(self.point_dim)
        self.token_norm = nn.LayerNorm(self.point_dim)
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.point_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.point_dim),
                    nn.Linear(self.point_dim, self.point_dim * 2),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.point_dim * 2, self.point_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.point_dim)

    def _select_valid_tokens(
        self,
        mask_emb_b: torch.Tensor,
        valid_b: Optional[torch.Tensor],
        geom_b: Optional[torch.Tensor] = None,
    ):
        if mask_emb_b is None or mask_emb_b.numel() == 0:
            return None, None

        if valid_b is not None:
            keep = valid_b.bool()
            if keep.numel() != mask_emb_b.shape[0]:
                keep = keep[: mask_emb_b.shape[0]]
            if not keep.any():
                return None, None
            mask_emb_b = mask_emb_b[keep]
            if geom_b is not None:
                geom_b = geom_b[keep]

        if mask_emb_b.shape[0] == 0:
            return None, None

        if self.max_context_masks > 0 and mask_emb_b.shape[0] > self.max_context_masks:
            mask_emb_b = mask_emb_b[: self.max_context_masks]
            if geom_b is not None:
                geom_b = geom_b[: self.max_context_masks]

        return mask_emb_b, geom_b

    def forward(
        self,
        point_feat: torch.Tensor,
        mask_embeddings: torch.Tensor,
        mask_valid: Optional[torch.Tensor],
        batch_indices: torch.Tensor,
        mask_geom: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if point_feat is None or point_feat.numel() == 0:
            return torch.zeros_like(point_feat)

        context_feat = torch.zeros_like(point_feat)
        if mask_embeddings is None or mask_embeddings.numel() == 0:
            return context_feat

        device = point_feat.device
        dtype = point_feat.dtype
        batch_indices = batch_indices.to(device=device)
        batch_size = mask_embeddings.shape[0]

        for b in range(batch_size):
            point_mask = batch_indices == b
            if not point_mask.any():
                continue

            mask_emb_b = mask_embeddings[b].to(device=device, dtype=dtype)
            valid_b = mask_valid[b].to(device=device) if mask_valid is not None else None
            geom_b = None
            if self.use_geom and mask_geom is not None:
                geom_b = mask_geom[b].to(device=device, dtype=dtype)

            mask_emb_b, geom_b = self._select_valid_tokens(mask_emb_b, valid_b, geom_b)
            if mask_emb_b is None or mask_emb_b.shape[0] == 0:
                continue

            token_b = self.odise_proj(mask_emb_b)
            if self.use_geom and self.geom_proj is not None and geom_b is not None:
                token_b = token_b + self.geom_proj(geom_b)

            token_b = self.token_norm(token_b)
            query_b = self.query_norm(point_feat[point_mask])

            out = query_b.unsqueeze(0)
            kv = token_b.unsqueeze(0)
            for attn, ffn in zip(self.attn_layers, self.ffn_layers):
                attn_out, _ = attn(out, kv, kv, need_weights=False)
                out = out + attn_out
                out = out + ffn(out)

            context_feat[point_mask] = self.out_norm(out.squeeze(0))

        return context_feat
