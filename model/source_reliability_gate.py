"""Source-aware semantic reliability gate for dual-space MoE fusion."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SourceReliabilityGate(nn.Module):
    """Predict LSeg reliability from teacher/mask evidence.

    gate=0 means prefer ODISE, gate=1 means prefer LSeg.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        init_bias: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        final_linear = self.net[-2]
        nn.init.constant_(final_linear.bias, float(init_bias))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim not in (2, 3):
            raise RuntimeError(f"evidence must be (K,D) or (K,Q,D), got {tuple(evidence.shape)}")
        return self.net(evidence).squeeze(-1)


def _expand_query_feature(feature: torch.Tensor, c: int) -> torch.Tensor:
    return feature.unsqueeze(-1).expand(-1, c)


def _normalized_optional_query_feature(
    value: Optional[torch.Tensor],
    k: int,
    device: torch.device,
    dtype: torch.dtype,
    use_log: bool,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(k, device=device, dtype=dtype)
    out = value.to(device=device, dtype=dtype).view(k)
    if use_log:
        out = torch.log1p(out.clamp_min(0.0))
        max_val = out.max().clamp_min(1e-6)
        out = out / max_val
    else:
        out = out.clamp(0.0, 1.0)
    return out


def normalize_log_feature(value: Optional[torch.Tensor], k: int) -> torch.Tensor:
    if value is None:
        raise ValueError("text-free source gate evidence requires all mask quality features")
    out = value.float().view(k)
    out = torch.log1p(out.clamp_min(0.0))
    return out / out.max().clamp_min(1e-6)


def _margin(probs: torch.Tensor) -> torch.Tensor:
    if probs.shape[-1] < 2:
        return probs.max(dim=-1).values.clamp(0.0, 1.0)
    top2 = probs.topk(k=2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)


def build_source_gate_evidence(
    p_odise: torch.Tensor,
    p_lseg: torch.Tensor,
    logits_odise: Optional[torch.Tensor] = None,
    logits_lseg: Optional[torch.Tensor] = None,
    mask_area: Optional[torch.Tensor] = None,
    lifted_point_count: Optional[torch.Tensor] = None,
    point_mask_conf: Optional[torch.Tensor] = None,
    mv_odise_stability: Optional[torch.Tensor] = None,
    mv_lseg_stability: Optional[torch.Tensor] = None,
    mv_valid: Optional[torch.Tensor] = None,
    input_dim: int = 17,
) -> torch.Tensor:
    """Build source reliability evidence for each query-text pair.

    logits_odise/logits_lseg are accepted for API extensibility but the first
    version intentionally uses only normalized probabilities and quality cues.
    input_dim=14 preserves the original evidence layout for old checkpoints;
    input_dim=17 appends multiview ODISE/LSeg stability and multiview validity.
    """
    del logits_odise, logits_lseg
    if input_dim not in (14, 17):
        raise ValueError(f"source gate evidence supports input_dim 14 or 17, got {input_dim}")
    if p_odise.shape != p_lseg.shape:
        raise RuntimeError(
            f"p_odise and p_lseg shape mismatch: {tuple(p_odise.shape)} vs {tuple(p_lseg.shape)}"
        )
    if p_odise.ndim != 2:
        raise RuntimeError(f"p_odise must be (K,C), got {tuple(p_odise.shape)}")

    p_o = p_odise.float().clamp(1e-6, 1.0)
    p_l = p_lseg.float().clamp(1e-6, 1.0)
    k, c = p_o.shape
    device = p_o.device
    dtype = p_o.dtype

    max_o = p_o.max(dim=-1).values
    max_l = p_l.max(dim=-1).values
    log_c = torch.log(torch.tensor(float(max(c, 2)), device=device, dtype=dtype))
    ent_o = (-(p_o * p_o.log()).sum(dim=-1) / log_c).clamp(0.0, 1.0)
    ent_l = (-(p_l * p_l.log()).sum(dim=-1) / log_c).clamp(0.0, 1.0)
    margin_o = _margin(p_o)
    margin_l = _margin(p_l)
    agreement = (p_o.argmax(dim=-1) == p_l.argmax(dim=-1)).to(dtype)
    area = _normalized_optional_query_feature(mask_area, k, device, dtype, use_log=True)
    lifted = _normalized_optional_query_feature(lifted_point_count, k, device, dtype, use_log=True)
    point_conf = _normalized_optional_query_feature(point_mask_conf, k, device, dtype, use_log=False)
    mv_o = _normalized_optional_query_feature(mv_odise_stability, k, device, dtype, use_log=False)
    mv_l = _normalized_optional_query_feature(mv_lseg_stability, k, device, dtype, use_log=False)
    mv_ok = _normalized_optional_query_feature(mv_valid, k, device, dtype, use_log=False)

    evidence_parts = [
        p_o,
        p_l,
        p_l - p_o,
        (p_l - p_o).abs(),
        _expand_query_feature(max_o, c),
        _expand_query_feature(max_l, c),
        _expand_query_feature(ent_o, c),
        _expand_query_feature(ent_l, c),
        _expand_query_feature(margin_o, c),
        _expand_query_feature(margin_l, c),
        _expand_query_feature(agreement, c),
        _expand_query_feature(area, c),
        _expand_query_feature(lifted, c),
        _expand_query_feature(point_conf, c),
    ]
    if input_dim == 17:
        evidence_parts.extend(
            [
                _expand_query_feature(mv_o, c),
                _expand_query_feature(mv_l, c),
                _expand_query_feature(mv_ok, c),
            ]
        )

    evidence = torch.stack(evidence_parts, dim=-1)
    return evidence


def build_text_free_source_gate_evidence(
    mv_odise_stability: torch.Tensor,
    mv_lseg_stability: torch.Tensor,
    mv_valid: torch.Tensor,
    mask_area: torch.Tensor,
    lifted_point_count: torch.Tensor,
    point_mask_conf: torch.Tensor,
) -> torch.Tensor:
    """Build 6D text-free mask-level source reliability evidence.

    Evidence layout:
      1. ODISE multiview stability
      2. LSeg multiview stability
      3. multiview validity
      4. normalized mask area
      5. normalized lifted point count
      6. point-mask confidence
    """
    k = int(mv_odise_stability.numel())
    if any(int(x.numel()) != k for x in (mv_lseg_stability, mv_valid, mask_area, lifted_point_count, point_mask_conf)):
        raise RuntimeError("text-free source gate evidence inputs must all be shape (K,)")
    device = mv_odise_stability.device
    dtype = mv_odise_stability.dtype
    area = normalize_log_feature(mask_area.to(device=device), k).to(dtype)
    lifted = normalize_log_feature(lifted_point_count.to(device=device), k).to(dtype)
    evidence = torch.stack(
        [
            mv_odise_stability.to(device=device, dtype=dtype).view(k).clamp(0.0, 1.0),
            mv_lseg_stability.to(device=device, dtype=dtype).view(k).clamp(0.0, 1.0),
            mv_valid.to(device=device, dtype=dtype).view(k).clamp(0.0, 1.0),
            area,
            lifted,
            point_mask_conf.to(device=device, dtype=dtype).view(k).clamp(0.0, 1.0),
        ],
        dim=-1,
    )
    return evidence
