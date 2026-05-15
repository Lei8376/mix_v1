

import contextlib
import collections
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from MinkowskiEngine import SparseTensor

from experiment_mask_distill.criterion_mask_distill import MaskDistillCriteria, build_lifted_3d_masks
from experiment_mask_distill.semantic_miou import (
    MaskMIoUTracker, ODISEPCSemanticMIoUTracker,
    build_text_features,
    diff2scene_class_probs_predict,
    diff2scene_mask_feature_predict,
    diff2scene_point_class_probs,
    mask_feature_class_probs,
)
from evaluate.semantic_iou import _SemanticAccumulator
from model.source_reliability_gate import build_source_gate_evidence, build_text_free_source_gate_evidence
from utils.util import AverageMeter
from trainer.open_vocab_trainer_v2 import MetricsTracker


# ============================================================
# Config
# ============================================================

@dataclass
class MaskDistillTrainerConfig:
    """Mask Distillation Trainer 配置。"""
    num_epochs:                 int   = 60
    base_lr:                    float = 1e-4
    weight_decay:               float = 4e-4
    grad_clip_norm:             float = 1.0
    log_dir:                    str   = "runs/mask_distill.1"
    checkpoint_dir:             str   = "checkpoints/mask_distill.1"
    log_every_steps:            int   = 20
    val_every_epochs:           int   = 2
    save_every_epochs:          int   = 5
    # Scheduler
    warmup_epochs:              int   = 1
    scheduler_type:             str   = "cosine"
    scheduler_t0:               int   = 1
    scheduler_t_mult:           int   = 2
    scheduler_eta_min:          float = 1e-6
    # AMP
    use_amp:                    bool  = True
    # Early stopping
    early_stopping_patience:    int   = 15
    early_stopping_min_delta:   float = 1e-4
    # ---- Loss 权重（主损失 mask distillation）----
    mask_distill_weight:        float = 1.0   # L_mask_distill 主损失
    bce_weight:                 float = 0.0   # 辅助 BCE（默认不用）
    dice_weight:                float = 0.0   # 辅助 Dice（默认不用）
    # GT 过滤阈值
    min_points_per_mask:        int   = 10
    # Resume
    resume_checkpoint:          Optional[str] = None
    override_optimizer_hparams_on_resume: bool = True
    reset_scheduler_on_resume:  bool  = True
    # Quick test
    max_batches_per_epoch:      Optional[int] = None
    use_model_half:             bool  = False
    gradient_accumulation_steps: int  = 2
    semantic_clip_model:         str   = "ODISE-256"
    semantic_pixel_clip_model:   str   = "ViT-B/32"
    semantic_prompt_template:    str   = "a photo of a {}"
    semantic_pc_lambda:          float = 0.5
    dual_space_eval:             bool  = True
    dual_space_odise_weight:     float = 0.5
    dual_space_lseg_weight:      float = 0.5
    dual_space_tau_odise:        float = 0.07
    dual_space_tau_lseg:         float = 0.07
    dual_space_use_confidence:   bool  = False
    dual_space_conf_min:         float = 0.2
    dual_space_conf_max:         float = 0.7
    best_monitor:                str   = "semantic_miou_projected_gate"
    lambda_align:                float = 1.0
    semantic_readout_mode:       str   = "projected_gate"
    eval_only:                   bool  = False
    use_lseg_semantic_loss:      bool  = False
    use_odise_semantic_loss:     bool  = False
    enable_verbose_legacy_probes: bool = False
    enable_legacy_source_gate_logs: bool = False
    enable_size_aware_ablation:  bool  = True
    enable_projected_size_gate_ablation: bool = True
    allow_gt_ce_upper_bound:     bool  = False
    # Source-aware Semantic MoE
    source_gate_train: bool = False
    source_gate_loss_weight: float = 0.03
    source_gate_open_loss_weight: float = 0.03
    source_gate_start_epoch: int = 3
    source_gate_detach_teacher_probs: bool = True
    source_gate_detach_pred_logits: bool = False
    source_gate_balance_reg: float = 0.0
    source_gate_entropy_reg: float = 0.0
    source_gate_monitor: str = "semantic_miou_dual_space_gate"
    source_gate_training_target: str = "text_free_mv_stability"
    source_gate_single_weight: float = 1.0
    source_gate_multiview_weight: float = 1.0
    source_gate_conflict_weight: float = 0.5
    source_gate_odise_prior: float = 1.2
    source_gate_lseg_prior: float = 1.0
    source_gate_conflict_safe_min: float = 0.25
    source_gate_mv_iou_threshold: float = 0.15
    source_gate_mv_topk: int = 5
    source_gate_mv_min_pairs: int = 1
    source_gate_mv_min_lifted_points: int = 2
    source_gate_mv_min_valid_masks: int = 2
    source_gate_skip_when_no_mv: bool = True
    source_gate_target_gamma: float = 2.0
    source_gate_mv_margin: float = 0.03
    source_gate_use_margin_filter: bool = True
    source_gate_mv_default_stability: float = 0.5
    source_gate_mask_quality_weight: float = 1.0
    source_gate_point_conf_weight: float = 1.0
    allow_source_gate_gt_ce_upper_bound: bool = False
    source_gate_train_query_file: Optional[str] = None
    source_gate_num_train_queries: int = 64
    dual_branch_probe: bool = False
    dual_branch_probe_weight: float = 0.0
    dual_branch_oracle_margin: float = 0.02
    dual_branch_probe_log_every: int = 20
    projected_sem_probe: bool = False
    projected_sem_probe_min_views: int = 2
    projected_sem_probe_max_points: int = 4096
    projected_sem_probe_region_mode: str = "point"
    projected_sem_probe_iou_weighted: bool = False
    projected_sem_probe_log_every: int = 20
    projected_sem_gate_scale: float = 10.0
    alignment_query_mode: str = "fused"
    semantic_readout_ablation: bool = False
    semantic_size_aware: bool = True
    semantic_small_area_thr: float = 0.01
    semantic_medium_area_thr: float = 0.10
    semantic_small_lseg_weight: float = 0.45
    semantic_medium_lseg_weight: float = 0.65
    semantic_large_lseg_weight: float = 0.80
    semantic_projected_gate: bool = True
    semantic_projected_gate_scale: float = 10.0
    semantic_projected_gate_min: float = 0.45
    semantic_projected_gate_max: float = 0.85
    semantic_projected_gate_default: float = 0.70
    semantic_projected_size_gate: bool = True
    semantic_projected_size_base: float = 0.65
    semantic_projected_size_beta: float = 1.0
    semantic_projected_size_gamma: float = 0.20
    semantic_projected_size_min: float = 0.35
    semantic_projected_size_max: float = 0.85


def _mask_feature_class_probs_tau(
    mask_features: torch.Tensor,
    text_features: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return mask_feature_class_probs(
        mask_features=mask_features,
        text_features=text_features,
        logit_scale=1.0 / float(tau),
    )


def _dual_space_confidence_probs(
    p_odise: torch.Tensor,
    p_lseg: torch.Tensor,
    conf_min: float,
    conf_max: float,
) -> torch.Tensor:
    if p_odise.shape != p_lseg.shape:
        raise RuntimeError(
            f"class-prob shape mismatch: ODISE={tuple(p_odise.shape)} LSeg={tuple(p_lseg.shape)}"
        )
    log_c = math.log(float(p_odise.shape[-1]))
    ent_odise = -(p_odise * p_odise.clamp_min(1e-12).log()).sum(dim=-1) / log_c
    ent_lseg = -(p_lseg * p_lseg.clamp_min(1e-12).log()).sum(dim=-1) / log_c
    conf_odise = 1.0 - ent_odise
    conf_lseg = 1.0 - ent_lseg
    w_lseg = conf_lseg / (conf_lseg + conf_odise + 1e-6)
    w_lseg = w_lseg.clamp(conf_min, conf_max).unsqueeze(-1)
    return (1.0 - w_lseg) * p_odise + w_lseg * p_lseg


def _source_gate_input_dim(model_ref) -> int:
    cfg = getattr(model_ref, "config", None)
    return int(getattr(cfg, "source_gate_input_dim", 6))


def _semantic_query_entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.float().clamp_min(1e-6)
    log_t = math.log(float(max(p.shape[-1], 2)))
    return (-(p * p.log()).sum(dim=-1) / log_t).clamp(0.0, 1.0)


def _semantic_query_margin(probs: torch.Tensor) -> torch.Tensor:
    p = probs.float().clamp(0.0, 1.0)
    if p.shape[-1] < 2:
        return p.max(dim=-1).values.clamp(0.0, 1.0)
    top2 = p.topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _compute_batch_multiview_probe_logs(batch: Dict[str, Any]) -> Dict[str, float]:
    scene_names = batch.get("scene_name") or []
    frame_stems = batch.get("frame_stem") or []
    if not scene_names:
        return {
            "batch_unique_scenes": 0.0,
            "batch_frames_per_scene_mean": 0.0,
            "batch_same_scene_pair_count": 0.0,
            "dual_branch_odise_loss_mean": 0.0,
            "dual_branch_lseg_loss_mean": 0.0,
            "dual_branch_fixed_loss_mean": 0.0,
            "dual_branch_oracle_loss_mean": 0.0,
            "dual_branch_odise_iou_mean": 0.0,
            "dual_branch_lseg_iou_mean": 0.0,
            "dual_branch_fixed_iou_mean": 0.0,
            "dual_branch_oracle_iou_mean": 0.0,
            "dual_branch_delta_loss_mean": 0.0,
            "dual_branch_delta_loss_std": 0.0,
            "dual_branch_odise_win_rate": 0.0,
            "dual_branch_lseg_win_rate": 0.0,
            "dual_branch_clear_win_rate": 0.0,
            "dual_branch_oracle_gain_vs_best_single_loss": 0.0,
            "dual_branch_oracle_gain_vs_fixed_loss": 0.0,
            "dual_branch_oracle_gain_vs_best_single_iou": 0.0,
            "dual_branch_oracle_gain_vs_fixed_iou": 0.0,
        }
    scene_counter = collections.Counter(str(scene) for scene in scene_names)
    same_scene_pair_count = sum(count * (count - 1) // 2 for count in scene_counter.values())
    unique_pairs = {(str(scene), str(frame)) for scene, frame in zip(scene_names, frame_stems)}
    return {
        "batch_unique_scenes": float(len(scene_counter)),
        "batch_frames_per_scene_mean": float(sum(scene_counter.values()) / max(len(scene_counter), 1)),
        "batch_same_scene_pair_count": float(same_scene_pair_count),
        "batch_unique_scene_frame_pairs": float(len(unique_pairs)),
    }


def compute_multiview_mask_stability(
    mask_point_indices,
    odise_feats: torch.Tensor,
    lseg_feats: torch.Tensor,
    scene_ids: Optional[list] = None,
    view_ids: Optional[list] = None,
    iou_threshold: float = 0.15,
    topk: int = 5,
    min_pairs: int = 1,
    default_stability: float = 0.5,
    min_lifted_points: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate per-mask same-teacher embedding stability across overlapping views."""
    device = odise_feats.device
    dtype = odise_feats.dtype
    k = int(odise_feats.shape[0])
    default = torch.full((k,), float(default_stability), device=device, dtype=dtype)
    mv_odise = default.clone()
    mv_lseg = default.clone()
    mv_valid = torch.zeros(k, device=device, dtype=dtype)
    mv_pair_count = torch.zeros(k, device=device, dtype=dtype)
    if k == 0 or scene_ids is None or view_ids is None:
        return mv_odise, mv_lseg, mv_valid, mv_pair_count

    point_sets = []
    for idx in mask_point_indices:
        if idx is None or idx.numel() == 0:
            point_sets.append(set())
        else:
            point_sets.append(set(idx.detach().cpu().long().unique().tolist()))

    odise_norm = F.normalize(odise_feats.float(), dim=-1)
    lseg_norm = F.normalize(lseg_feats.float(), dim=-1)
    for i in range(k):
        if len(point_sets[i]) < int(min_lifted_points):
            continue
        matches = []
        for j in range(k):
            if i == j or len(point_sets[j]) < int(min_lifted_points):
                continue
            if scene_ids[i] != scene_ids[j] or view_ids[i] == view_ids[j]:
                continue
            inter = len(point_sets[i].intersection(point_sets[j]))
            if inter == 0:
                continue
            union = len(point_sets[i]) + len(point_sets[j]) - inter
            iou = float(inter) / max(float(union), 1.0)
            if iou >= iou_threshold:
                matches.append((iou, j))
        if not matches:
            continue
        matches = sorted(matches, key=lambda x: x[0], reverse=True)[: max(1, int(topk))]
        if len(matches) < max(1, int(min_pairs)):
            continue
        js = torch.tensor([m[1] for m in matches], device=device, dtype=torch.long)
        weights = torch.tensor([m[0] for m in matches], device=device, dtype=torch.float32)
        sim_o = ((odise_norm[i].unsqueeze(0) * odise_norm[js]).sum(dim=-1) + 1.0) * 0.5
        sim_l = ((lseg_norm[i].unsqueeze(0) * lseg_norm[js]).sum(dim=-1) + 1.0) * 0.5
        mv_odise[i] = _weighted_mean(sim_o, weights).to(dtype)
        mv_lseg[i] = _weighted_mean(sim_l, weights).to(dtype)
        mv_valid[i] = 1.0
        mv_pair_count[i] = float(len(matches))
    return mv_odise.clamp(0.0, 1.0), mv_lseg.clamp(0.0, 1.0), mv_valid, mv_pair_count


def compute_text_free_mv_mask_stability(
    mask_point_indices,
    odise_feats: torch.Tensor,
    lseg_feats: torch.Tensor,
    scene_ids: Optional[list] = None,
    view_ids: Optional[list] = None,
    iou_threshold: float = 0.15,
    topk: int = 5,
    default_stability: float = 0.5,
    min_lifted_points: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return compute_multiview_mask_stability(
        mask_point_indices=mask_point_indices,
        odise_feats=odise_feats,
        lseg_feats=lseg_feats,
        scene_ids=scene_ids,
        view_ids=view_ids,
        iou_threshold=iou_threshold,
        topk=topk,
        min_pairs=1,
        default_stability=default_stability,
        min_lifted_points=min_lifted_points,
    )


def _normalize_log_vector(value: torch.Tensor) -> torch.Tensor:
    out = torch.log1p(value.float().clamp_min(0.0))
    return out / out.max().clamp_min(1e-6)


def build_text_free_mv_gate_target(
    mv_odise: torch.Tensor,
    mv_lseg: torch.Tensor,
    mv_valid: torch.Tensor,
    mask_area: torch.Tensor,
    lifted_point_count: torch.Tensor,
    point_mask_conf: torch.Tensor,
    odise_prior: float = 1.2,
    lseg_prior: float = 1.0,
    target_gamma: float = 2.0,
    mask_quality_weight: float = 1.0,
    point_conf_weight: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    area_q = _normalize_log_vector(mask_area)
    lift_q = _normalize_log_vector(lifted_point_count)
    point_q = point_mask_conf.float().clamp(0.0, 1.0)
    mask_quality = (area_q + lift_q + float(point_conf_weight) * point_q) / (2.0 + float(point_conf_weight))
    mask_quality = (float(mask_quality_weight) * mask_quality).clamp(0.0, 1.0)

    neutral = torch.full_like(mv_odise, 0.5)
    st_o = torch.where(mv_valid.bool(), mv_odise, neutral).clamp(0.0, 1.0)
    st_l = torch.where(mv_valid.bool(), mv_lseg, neutral).clamp(0.0, 1.0)
    r_o_raw = float(odise_prior) * st_o
    r_l_raw = float(lseg_prior) * st_l
    diff = torch.abs(r_l_raw - r_o_raw)
    gamma = float(target_gamma)
    r_o = (r_o_raw.clamp_min(eps) ** gamma) * mask_quality
    r_l = (r_l_raw.clamp_min(eps) ** gamma) * mask_quality
    target_g = r_l / (r_o + r_l + eps)

    loss_weight = mask_quality
    loss_weight = loss_weight / loss_weight.mean().clamp_min(eps)
    logs = {
        "source_gate_mask_quality_mean": float(mask_quality.detach().mean().cpu()),
        "source_gate_mv_diff_mean": float(diff.detach().mean().cpu()),
    }
    return target_g.detach(), loss_weight.detach(), diff.detach(), logs


def build_open_reliability_gate_target(
    p_odise: torch.Tensor,
    p_lseg: torch.Tensor,
    mv_odise: torch.Tensor,
    mv_lseg: torch.Tensor,
    mv_valid: torch.Tensor,
    odise_prior: float = 1.2,
    lseg_prior: float = 1.0,
    conflict_safe_min: float = 0.25,
    single_weight: float = 1.0,
    multiview_weight: float = 1.0,
    conflict_weight: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Build an open-vocabulary source target without semantic ground truth."""
    p_o = p_odise.float().clamp_min(eps)
    p_l = p_lseg.float().clamp_min(eps)
    ent_o = _semantic_query_entropy(p_o)
    ent_l = _semantic_query_entropy(p_l)
    margin_o = _semantic_query_margin(p_o)
    margin_l = _semantic_query_margin(p_l)
    conf_o = 0.5 * (1.0 - ent_o) + 0.5 * margin_o
    conf_l = 0.5 * (1.0 - ent_l) + 0.5 * margin_l

    single_o = float(single_weight) * p_o * conf_o[:, None]
    single_l = float(single_weight) * p_l * conf_l[:, None]
    neutral = torch.full_like(mv_odise, 0.5)
    st_o = torch.where(mv_valid.bool(), mv_odise, neutral).clamp(0.0, 1.0)
    st_l = torch.where(mv_valid.bool(), mv_lseg, neutral).clamp(0.0, 1.0)
    mv_w = float(multiview_weight)
    st_o_eff = (1.0 - mv_w) * torch.ones_like(st_o) + mv_w * st_o
    st_l_eff = (1.0 - mv_w) * torch.ones_like(st_l) + mv_w * st_l

    r_o = float(odise_prior) * single_o * st_o_eff[:, None]
    r_l = float(lseg_prior) * single_l * st_l_eff[:, None]
    raw_target = r_l / (r_o + r_l + eps)

    top_disagree = (p_o.argmax(dim=-1) != p_l.argmax(dim=-1)).float()
    both_conf = torch.minimum(conf_o, conf_l)
    m = (0.5 * (p_o + p_l)).clamp_min(eps)
    js = 0.5 * (p_o * (p_o / m).log()).sum(dim=-1) + 0.5 * (p_l * (p_l / m).log()).sum(dim=-1)
    js = (js / math.log(2.0)).clamp(0.0, 1.0)
    conflict = (float(conflict_weight) * top_disagree * both_conf * js).clamp(0.0, 1.0)
    safe_factor = (1.0 - conflict).clamp(min=float(conflict_safe_min), max=1.0)
    target_g = 0.5 + safe_factor[:, None] * (raw_target - 0.5)

    base_weight = torch.maximum(r_o, r_l).detach()
    mv_boost = torch.where(mv_valid.bool(), torch.ones_like(mv_valid), torch.full_like(mv_valid, 0.5))
    loss_weight = base_weight * mv_boost[:, None]
    loss_weight = loss_weight / loss_weight.mean().clamp_min(eps)
    logs = {
        "source_gate_conflict_mean": float(conflict.detach().mean().cpu()),
        "source_gate_single_odise_mean": float(single_o.detach().mean().cpu()),
        "source_gate_single_lseg_mean": float(single_l.detach().mean().cpu()),
    }
    return target_g.detach(), loss_weight.detach(), logs


# ============================================================
# Trainer
# ============================================================

class MaskDistillTrainer:
    """
    基于 mask-level cosine distillation 的训练器（Diff2Scene 方案）。
    主损失：L = (1/K) * sum_k [1 - cos(B_k^{3d'}, B_k^{3d})]
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        config: MaskDistillTrainerConfig = None,
        device: str = "cuda",
        rank: int = 0,
        train_sampler=None,
    ):
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.config        = config or MaskDistillTrainerConfig()
        self.device        = device
        self.rank          = rank
        self.train_sampler = train_sampler
        self.is_main       = (rank == 0)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.base_lr,
            weight_decay=self.config.weight_decay,
        )

        self.steps_per_epoch        = len(train_loader)
        self.accum_steps            = getattr(self.config, "gradient_accumulation_steps", 2)
        self.optim_steps_per_epoch  = math.ceil(self.steps_per_epoch / self.accum_steps)
        self.total_steps            = self.optim_steps_per_epoch * self.config.num_epochs
        self.warmup_steps           = self.optim_steps_per_epoch * self.config.warmup_epochs
        self._build_scheduler()

        self.scaler = GradScaler(enabled=self.config.use_amp)

        self.global_step                = 0
        self.current_epoch              = 0
        self.best_loss                  = float("inf")
        self.best_iou                   = 0.0
        self.epochs_without_improvement = 0

        self.writer = None
        if self.is_main:
            os.makedirs(self.config.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.config.log_dir)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self._text_features: Optional[torch.Tensor] = None
        self._pixel_text_features: Optional[torch.Tensor] = None
        self._source_gate_text_features: Optional[torch.Tensor] = None
        self._source_gate_pixel_text_features: Optional[torch.Tensor] = None
        self._warned_pixel_text_dim_mismatch = False
        self._warned_source_gate_gt_ce = False
        self._warned_source_gate_query_fallback = False
        self._source_gate_zero_mv_steps = 0
        self._warned_source_gate_zero_mv = False

        if str(self.config.source_gate_training_target).lower() == "gt_ce_upper_bound":
            if not (self.config.allow_gt_ce_upper_bound and self.config.allow_source_gate_gt_ce_upper_bound):
                raise RuntimeError(
                    "gt_ce_upper_bound uses semantic GT and is not allowed in the open-vocabulary training path."
                )

        if self.config.resume_checkpoint:
            self._load_checkpoint(self.config.resume_checkpoint)

        if self.is_main:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            print(f"[MaskDistillTrainer] Parameters: {trainable:,} trainable / {total:,} total")
            print(f"[Alignment] query_mode={self.config.alignment_query_mode}")
            print(f"  mask_distill_weight={self.config.mask_distill_weight}  "
                  f"bce_weight={self.config.bce_weight}  "
                  f"dice_weight={self.config.dice_weight}")

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------

    def _build_scheduler(self):
        cfg = self.config
        if cfg.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.optim_steps_per_epoch * cfg.scheduler_t0,
                T_mult=cfg.scheduler_t_mult,
                eta_min=cfg.scheduler_eta_min,
            )
        elif cfg.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.optim_steps_per_epoch * 10,
                gamma=0.5,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )

    def _adjust_learning_rate_warmup(self, step: int):
        if step < self.warmup_steps:
            lr = self.config.base_lr * (step + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        tensor_keys = [
            "coords_3d", "feat_3d", "ori_coords_3d",
            "binary_label_3d", "binary_label_2d", "label_2d",
            "img", "x_label", "y_label", "inds_reconstruct",
        ]
        precomputed_keys = [
            "pixel_embeddings", "pixel_pooled", "clip_pooled",
            "masks", "mask_embeddings", "mask_valid",
        ]
        moved = dict(batch)
        for key in tensor_keys + precomputed_keys:
            if key in moved and isinstance(moved[key], torch.Tensor):
                moved[key] = moved[key].to(self.device, non_blocking=True)
        return moved

    def _build_sparse_tensor(self, batch: Dict[str, Any]) -> SparseTensor:
        return SparseTensor(batch["feat_3d"], batch["coords_3d"].int())

    def _make_criteria(self, results, batch) -> MaskDistillCriteria:
        return MaskDistillCriteria(
            results=results,
            batch_input=batch,
            mask_distill_weight=self.config.mask_distill_weight,
            bce_weight=self.config.bce_weight,
            dice_weight=self.config.dice_weight,
            min_points_per_mask=self.config.min_points_per_mask,
        )

    def _compute_source_gate_loss(
        self,
        results: Dict,
        batch: Dict,
        text_feats: Optional[torch.Tensor],
        pixel_text_feats: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        target = str(self.config.source_gate_training_target).lower()
        if target == "none":
            return None, self._empty_source_gate_logs()
        if target == "text_free_mv_stability":
            return self._compute_source_gate_text_free_mv_loss(results, batch)
        if target == "open_reliability":
            return self._compute_source_gate_open_reliability_loss(
                results,
                batch,
                text_feats,
                pixel_text_feats,
            )
        if target == "gt_ce_upper_bound":
            if not self.config.allow_source_gate_gt_ce_upper_bound:
                raise RuntimeError(
                    "gt_ce_upper_bound uses semantic GT and is not open-vocabulary training. "
                    "Set allow_source_gate_gt_ce_upper_bound=true only for upper-bound ablation."
                )
            if self.is_main and not self._warned_source_gate_gt_ce:
                print(
                    "WARNING: SourceGate GT CE uses semantic ground-truth labels. "
                    "This is a closed-set supervised upper-bound ablation, not open-vocabulary training."
                )
                self._warned_source_gate_gt_ce = True
            return self._compute_source_gate_gt_ce_upper_bound_loss(
                results,
                batch,
                text_feats,
                pixel_text_feats,
            )
        raise ValueError(
            "source_gate_training_target must be one of "
            "{'none', 'text_free_mv_stability', 'open_reliability', 'gt_ce_upper_bound'}, "
            f"got {self.config.source_gate_training_target!r}"
        )

    def _empty_source_gate_logs(self) -> Dict[str, float]:
        return {
            "loss_source_gate": 0.0,
            "loss_source_gate_open": 0.0,
            "loss_source_gate_gt_ce_upper_bound": 0.0,
            "source_gate_mean": 0.0,
            "source_gate_std": 0.0,
            "source_gate_min": 0.0,
            "source_gate_max": 0.0,
            "source_gate_target_mean": 0.0,
            "source_gate_target_std": 0.0,
            "source_gate_mv_valid_ratio": 0.0,
            "source_gate_mv_valid_count": 0.0,
            "source_gate_mv_pair_count": 0.0,
            "source_gate_skipped_no_mv": 0.0,
            "source_gate_mv_odise_mean": 0.0,
            "source_gate_mv_lseg_mean": 0.0,
            "source_gate_mv_diff_mean": 0.0,
            "source_gate_mv_diff_valid_mean": 0.0,
            "source_gate_loss_valid_count": 0.0,
            "source_gate_loss_valid_ratio": 0.0,
            "source_gate_gate_mean_valid": 0.0,
            "source_gate_gate_std_valid": 0.0,
            "source_gate_target_mean_valid": 0.0,
            "source_gate_target_std_valid": 0.0,
            "source_gate_mv_odise_mean_valid": 0.0,
            "source_gate_mv_lseg_mean_valid": 0.0,
            "source_gate_conflict_mean": 0.0,
            "source_gate_mask_quality_mean": 0.0,
            "batch_unique_scenes": 0.0,
            "batch_frames_per_scene_mean": 0.0,
            "batch_same_scene_pair_count": 0.0,
        }

    def _source_gate_regularizers(
        self,
        loss_extra: torch.Tensor,
        gate_cat: torch.Tensor,
        logs: Dict[str, float],
    ) -> torch.Tensor:
        if self.config.source_gate_balance_reg > 0:
            loss_balance = (gate_cat.mean() - 0.5) ** 2
            loss_extra = loss_extra + self.config.source_gate_balance_reg * loss_balance
            logs["loss_source_gate_balance"] = float(loss_balance.detach().cpu())
        if self.config.source_gate_entropy_reg > 0:
            gate_entropy = -(
                gate_cat.clamp_min(1e-6).log() * gate_cat
                + (1.0 - gate_cat).clamp_min(1e-6).log() * (1.0 - gate_cat)
            ).mean()
            loss_extra = loss_extra - self.config.source_gate_entropy_reg * gate_entropy
            logs["loss_source_gate_entropy"] = float(gate_entropy.detach().cpu())
        return loss_extra

    def _empty_dual_branch_logs(self) -> Dict[str, float]:
        return {
            "dual_branch_odise_loss_mean": 0.0,
            "dual_branch_lseg_loss_mean": 0.0,
            "dual_branch_fixed_loss_mean": 0.0,
            "dual_branch_oracle_loss_mean": 0.0,
            "dual_branch_odise_iou_mean": 0.0,
            "dual_branch_lseg_iou_mean": 0.0,
            "dual_branch_fixed_iou_mean": 0.0,
            "dual_branch_oracle_iou_mean": 0.0,
            "dual_branch_delta_loss_mean": 0.0,
            "dual_branch_delta_loss_std": 0.0,
            "dual_branch_odise_win_rate": 0.0,
            "dual_branch_lseg_win_rate": 0.0,
            "dual_branch_clear_win_rate": 0.0,
            "dual_branch_oracle_gain_vs_best_single_loss": 0.0,
            "dual_branch_oracle_gain_vs_fixed_loss": 0.0,
            "dual_branch_oracle_gain_vs_best_single_iou": 0.0,
            "dual_branch_oracle_gain_vs_fixed_iou": 0.0,
        }

    def _empty_projected_sem_logs(self) -> Dict[str, float]:
        return {
            "projected_sem_point_count_total": 0.0,
            "projected_sem_point_count_valid": 0.0,
            "projected_sem_valid_ratio": 0.0,
            "projected_sem_view_count_mean": 0.0,
            "projected_sem_view_count_max": 0.0,
            "projected_sem_odise_consistency_mean": 0.0,
            "projected_sem_lseg_consistency_mean": 0.0,
            "projected_sem_odise_consistency_std": 0.0,
            "projected_sem_lseg_consistency_std": 0.0,
            "projected_sem_diff_mean": 0.0,
            "projected_sem_diff_std": 0.0,
            "projected_sem_abs_diff_mean": 0.0,
            "projected_sem_odise_win_rate": 0.0,
            "projected_sem_lseg_win_rate": 0.0,
            "projected_sem_clear_win_rate_001": 0.0,
            "projected_sem_clear_win_rate_003": 0.0,
            "projected_sem_clear_win_rate_005": 0.0,
            "projected_sem_g_sem_rule_mean": 0.0,
            "projected_sem_g_sem_rule_std": 0.0,
            "projected_sem_g_sem_rule_min": 0.0,
            "projected_sem_g_sem_rule_max": 0.0,
        }

    def _semantic_ablation_names(self) -> Tuple[str, ...]:
        return (
            "odise_only",
            "lseg_only",
            "fixed_05",
            "lseg_06",
            "lseg_07",
            "lseg_08",
            "size_aware",
            "projected_gate",
            "projected_size_gate",
        )

    def _semantic_size_group_names(self) -> Tuple[str, ...]:
        return ("small", "medium", "large")

    def _compute_size_aware_lseg_weight(self, mask_area_ratio: torch.Tensor) -> torch.Tensor:
        small_thr = float(self.config.semantic_small_area_thr)
        medium_thr = float(self.config.semantic_medium_area_thr)
        weight = torch.full_like(mask_area_ratio, float(self.config.semantic_large_lseg_weight))
        weight = torch.where(
            mask_area_ratio < medium_thr,
            torch.full_like(weight, float(self.config.semantic_medium_lseg_weight)),
            weight,
        )
        weight = torch.where(
            mask_area_ratio < small_thr,
            torch.full_like(weight, float(self.config.semantic_small_lseg_weight)),
            weight,
        )
        return weight.clamp(0.0, 1.0)

    def _build_eval_semantic_aux(
        self,
        results: Dict,
        batch: Dict,
        lifted: torch.Tensor,
        lifted_valid: torch.Tensor,
        lseg_all,
    ) -> Dict[str, Any]:
        batch_size = len(results["outputs"])
        point_groups = [None] * batch_size
        projected_mask_diff = {}
        projected_mask_valid = {}
        if lseg_all is None:
            return {
                "point_groups": point_groups,
                "projected_mask_diff": projected_mask_diff,
                "projected_mask_valid": projected_mask_valid,
            }

        scene_names = batch.get("scene_name") or []
        point_to_entries = collections.defaultdict(list)
        point_to_views = collections.defaultdict(set)
        mask_stats = collections.defaultdict(list)

        for b in range(batch_size):
            if len(results["outputs"][b]) == 0:
                continue
            valid_k = results["mask_valid_from_masks"][b]
            if not valid_k.any():
                continue
            pt_mask = results["batch_indices"] == b
            if not pt_mask.any():
                continue

            pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()
            lifted_b = lifted[pt_mask][:, valid_k]
            lifted_valid_b = lifted_valid[pt_mask]
            odise_q = batch["mask_embeddings"][b][valid_k].float()
            lseg_q = lseg_all[b][valid_k].float()
            point_coords = batch["ori_coords_3d"][pt_mask][:, 1:4].long()
            mask_area = batch["masks"][b][valid_k].float().sum(dim=(1, 2))
            image_area = float(batch["masks"][b].shape[-1] * batch["masks"][b].shape[-2])
            mask_area_ratio = (mask_area / max(image_area, 1.0)).clamp(0.0, 1.0)
            point_conf = torch.sigmoid(pred_logits)
            point_group = torch.full(
                (point_coords.shape[0],),
                -1,
                dtype=torch.long,
                device=point_coords.device,
            )
            scene_name = (
                str(scene_names[b])
                if isinstance(scene_names, (list, tuple)) and b < len(scene_names)
                else str(b)
            )

            for n in range(point_coords.shape[0]):
                if not bool(lifted_valid_b[n]):
                    continue
                mask_ids = torch.where(lifted_b[n] > 0.5)[0]
                if mask_ids.numel() == 0:
                    continue
                choose_idx = mask_ids[point_conf[n, mask_ids].argmax()]
                area_ratio = float(mask_area_ratio[choose_idx].item())
                if area_ratio < float(self.config.semantic_small_area_thr):
                    point_group[n] = 0
                elif area_ratio < float(self.config.semantic_medium_area_thr):
                    point_group[n] = 1
                else:
                    point_group[n] = 2

                key = (
                    scene_name,
                    int(point_coords[n, 0].item()),
                    int(point_coords[n, 1].item()),
                    int(point_coords[n, 2].item()),
                )
                point_to_entries[key].append((b, int(choose_idx.item()), odise_q[choose_idx], lseg_q[choose_idx]))
                point_to_views[key].add(b)

            point_groups[b] = point_group

        def _avg_pairwise_cos(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] < 2:
                return x.new_tensor(0.0)
            sim = x @ x.t()
            idx = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
            return sim[idx[0], idx[1]].mean()

        for key, entries in point_to_entries.items():
            if len(point_to_views[key]) < int(self.config.projected_sem_probe_min_views):
                continue
            odise_stack = torch.stack([entry[2] for entry in entries], dim=0).float()
            lseg_stack = torch.stack([entry[3] for entry in entries], dim=0).float()
            diff = _avg_pairwise_cos(F.normalize(lseg_stack, dim=-1)) - _avg_pairwise_cos(
                F.normalize(odise_stack, dim=-1)
            )
            diff_val = float(diff.detach().cpu().item())
            for b, local_idx, _, _ in entries:
                mask_stats[(b, local_idx)].append(diff_val)

        for b in range(batch_size):
            if len(results["outputs"][b]) == 0:
                continue
            valid_k = results["mask_valid_from_masks"][b]
            if not valid_k.any():
                continue
            num_valid = int(valid_k.sum().item())
            diff_t = torch.zeros(num_valid, dtype=torch.float32, device=self.device)
            valid_t = torch.zeros(num_valid, dtype=torch.bool, device=self.device)
            for local_idx in range(num_valid):
                values = mask_stats.get((b, local_idx))
                if not values:
                    continue
                diff_t[local_idx] = float(sum(values) / max(len(values), 1))
                valid_t[local_idx] = True
            projected_mask_diff[b] = diff_t
            projected_mask_valid[b] = valid_t

        return {
            "point_groups": point_groups,
            "projected_mask_diff": projected_mask_diff,
            "projected_mask_valid": projected_mask_valid,
        }

    def _compute_semantic_readout_probs(
        self,
        p_odise: torch.Tensor,
        p_lseg: torch.Tensor,
        mask_area_ratio: torch.Tensor,
        projected_diff: Optional[torch.Tensor],
        projected_valid: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        probs = {
            "odise_only": p_odise,
            "lseg_only": p_lseg,
            "fixed_05": 0.5 * p_lseg + 0.5 * p_odise,
            "lseg_06": 0.6 * p_lseg + 0.4 * p_odise,
            "lseg_07": 0.7 * p_lseg + 0.3 * p_odise,
            "lseg_08": 0.8 * p_lseg + 0.2 * p_odise,
        }
        size_weight = self._compute_size_aware_lseg_weight(mask_area_ratio)
        probs["size_aware"] = size_weight[:, None] * p_lseg + (1.0 - size_weight[:, None]) * p_odise

        if projected_diff is None or projected_valid is None:
            projected_diff = torch.zeros_like(mask_area_ratio)
            projected_valid = torch.zeros_like(mask_area_ratio, dtype=torch.bool)

        raw_proj = torch.sigmoid(float(self.config.semantic_projected_gate_scale) * projected_diff)
        proj_weight = raw_proj.clamp(
            float(self.config.semantic_projected_gate_min),
            float(self.config.semantic_projected_gate_max),
        )
        proj_weight = torch.where(
            projected_valid,
            proj_weight,
            torch.full_like(proj_weight, float(self.config.semantic_projected_gate_default)),
        )
        probs["projected_gate"] = proj_weight[:, None] * p_lseg + (1.0 - proj_weight[:, None]) * p_odise

        small_score = (
            (float(self.config.semantic_small_area_thr) - mask_area_ratio)
            / max(float(self.config.semantic_small_area_thr), 1e-6)
        ).clamp(0.0, 1.0)
        proj_size_weight = (
            float(self.config.semantic_projected_size_base)
            + float(self.config.semantic_projected_size_beta) * projected_diff
            - float(self.config.semantic_projected_size_gamma) * small_score
        ).clamp(
            float(self.config.semantic_projected_size_min),
            float(self.config.semantic_projected_size_max),
        )
        proj_size_weight = torch.where(projected_valid, proj_size_weight, size_weight)
        probs["projected_size_gate"] = (
            proj_size_weight[:, None] * p_lseg + (1.0 - proj_size_weight[:, None]) * p_odise
        )
        return probs

    def _collect_text_free_gate_items(self, results: Dict, batch: Dict):
        lseg_all = results.get("pixel_pooled_embeddings", batch.get("pixel_pooled"))
        if lseg_all is None:
            return None
        with torch.no_grad():
            lifted, lifted_valid = build_lifted_3d_masks(
                batch["masks"],
                batch["mask_valid"],
                batch["x_label"],
                batch["y_label"],
                results["batch_indices"],
            )

        global_mask_points = []
        global_odise = []
        global_lseg = []
        global_scene_ids = []
        global_view_ids = []
        item_records = []
        scene_names = batch.get("scene_name")
        frame_stems = batch.get("frame_stem")

        for b in range(len(results["outputs"])):
            if len(results["outputs"][b]) == 0:
                continue
            valid_k = results["mask_valid_from_masks"][b]
            if not valid_k.any():
                continue
            odise_q = batch["mask_embeddings"][b][valid_k].float()
            lseg_q = lseg_all[b][valid_k].float()
            pt_mask = results["batch_indices"] == b
            lifted_b = lifted[pt_mask][:, valid_k]
            lifted_valid_b = lifted_valid[pt_mask]
            mask_points = [
                torch.where((lifted_b[:, k] > 0.5) & lifted_valid_b)[0]
                for k in range(lifted_b.shape[1])
            ]
            start = len(global_mask_points)
            global_mask_points.extend(mask_points)
            global_odise.append(odise_q)
            global_lseg.append(lseg_q)
            scene_id = scene_names[b] if isinstance(scene_names, (list, tuple)) and b < len(scene_names) else str(b)
            view_id = frame_stems[b] if isinstance(frame_stems, (list, tuple)) and b < len(frame_stems) else str(b)
            global_scene_ids.extend([scene_id] * int(valid_k.sum().item()))
            global_view_ids.extend([view_id] * int(valid_k.sum().item()))
            item_records.append((b, valid_k, start, len(mask_points), mask_points))

        if not item_records:
            return None
        global_odise_t = torch.cat(global_odise, dim=0)
        global_lseg_t = torch.cat(global_lseg, dim=0)
        mv_odise_all, mv_lseg_all, mv_valid_all, mv_pair_count_all = compute_text_free_mv_mask_stability(
            global_mask_points,
            global_odise_t,
            global_lseg_t,
            scene_ids=global_scene_ids,
            view_ids=global_view_ids,
            iou_threshold=self.config.source_gate_mv_iou_threshold,
            topk=self.config.source_gate_mv_topk,
            default_stability=self.config.source_gate_mv_default_stability,
            min_lifted_points=self.config.source_gate_mv_min_lifted_points,
        )
        return lseg_all, item_records, mv_odise_all, mv_lseg_all, mv_valid_all, mv_pair_count_all

    def _compute_source_gate_text_free_mv_loss(
        self,
        results: Dict,
        batch: Dict,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        source_gate = getattr(model_ref, "source_gate", None)
        empty_logs = self._empty_source_gate_logs()
        if source_gate is None:
            return None, empty_logs

        collected = self._collect_text_free_gate_items(results, batch)
        batch_probe_logs = _compute_batch_multiview_probe_logs(batch)
        if collected is None:
            empty_logs.update(batch_probe_logs)
            return None, empty_logs
        lseg_all, item_records, mv_odise_all, mv_lseg_all, mv_valid_all, mv_pair_count_all = collected

        loss_gate_total = None
        num_gate_items = 0
        gate_values_for_log = []
        target_values_for_log = []
        mv_valid_values = []
        mv_odise_values = []
        mv_lseg_values = []
        mask_quality_values = []
        gate_valid_values = []
        target_valid_values = []
        mv_odise_valid_values = []
        mv_lseg_valid_values = []
        mv_diff_values = []
        mv_diff_valid_values = []
        total_valid_count = 0
        total_pair_count = 0
        total_loss_valid_count = 0
        skipped_no_mv = 0

        for b, valid_k, start, count, mask_points in item_records:
            pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()
            pred_logits_for_gate = pred_logits.detach() if self.config.source_gate_detach_pred_logits else pred_logits
            point_mask_conf = torch.sigmoid(pred_logits_for_gate).mean(dim=0).detach()
            lifted_point_count = torch.tensor(
                [float(idx.numel()) for idx in mask_points],
                device=pred_logits.device,
                dtype=pred_logits.dtype,
            )
            mask_area = batch["masks"][b][valid_k].float().sum(dim=(1, 2)).detach()
            mv_slice = slice(start, start + count)
            mv_odise = mv_odise_all[mv_slice].to(pred_logits.device)
            mv_lseg = mv_lseg_all[mv_slice].to(pred_logits.device)
            mv_valid = mv_valid_all[mv_slice].to(pred_logits.device)
            mv_pair_count = mv_pair_count_all[mv_slice].to(pred_logits.device)

            target_g, loss_weight, diff, target_logs = build_text_free_mv_gate_target(
                mv_odise,
                mv_lseg,
                mv_valid,
                mask_area,
                lifted_point_count,
                point_mask_conf,
                odise_prior=self.config.source_gate_odise_prior,
                lseg_prior=self.config.source_gate_lseg_prior,
                target_gamma=self.config.source_gate_target_gamma,
                mask_quality_weight=self.config.source_gate_mask_quality_weight,
                point_conf_weight=self.config.source_gate_point_conf_weight,
            )
            evidence = build_text_free_source_gate_evidence(
                mv_odise,
                mv_lseg,
                mv_valid,
                mask_area,
                lifted_point_count,
                point_mask_conf,
            )
            gate = source_gate(evidence)
            valid = mv_valid.bool()
            mv_valid_count = int(valid.sum().item())
            total_valid_count += mv_valid_count
            total_pair_count += int(mv_pair_count[valid].sum().item())
            if mv_valid_count == 0:
                skipped_no_mv += 1
                continue
            if self.config.source_gate_use_margin_filter:
                valid = valid & (diff > float(self.config.source_gate_mv_margin))
            loss_valid_count = int(valid.sum().item())
            total_loss_valid_count += loss_valid_count
            if self.config.source_gate_skip_when_no_mv and loss_valid_count < int(self.config.source_gate_mv_min_valid_masks):
                skipped_no_mv += 1
                continue
            loss_gate = (
                ((gate[valid] - target_g[valid].detach()) ** 2) * loss_weight[valid].detach()
            ).sum() / loss_weight[valid].sum().clamp_min(1.0)
            if torch.isnan(loss_gate) or torch.isinf(loss_gate):
                continue
            loss_gate_total = loss_gate if loss_gate_total is None else loss_gate_total + loss_gate
            num_gate_items += 1
            gate_values_for_log.append(gate.detach())
            target_values_for_log.append(target_g.detach())
            mv_valid_values.append(mv_valid.detach())
            mv_odise_values.append(mv_odise.detach())
            mv_lseg_values.append(mv_lseg.detach())
            mv_diff_values.append(diff.detach())
            mask_quality_values.append(torch.tensor(target_logs["source_gate_mask_quality_mean"], device=gate.device))
            gate_valid_values.append(gate[valid].detach())
            target_valid_values.append(target_g[valid].detach())
            mv_odise_valid_values.append(mv_odise[valid].detach())
            mv_lseg_valid_values.append(mv_lseg[valid].detach())
            mv_diff_valid_values.append(diff[valid].detach())

        if num_gate_items == 0 or loss_gate_total is None:
            empty_logs["source_gate_mv_valid_count"] = float(total_valid_count)
            empty_logs["source_gate_mv_pair_count"] = float(total_pair_count)
            empty_logs["source_gate_loss_valid_count"] = float(total_loss_valid_count)
            empty_logs["source_gate_loss_valid_ratio"] = float(
                total_loss_valid_count / max(total_valid_count, 1)
            )
            empty_logs["source_gate_skipped_no_mv"] = float(skipped_no_mv)
            empty_logs.update(batch_probe_logs)
            return None, empty_logs

        loss_gate_total = loss_gate_total / num_gate_items
        loss_extra = self.config.source_gate_open_loss_weight * loss_gate_total
        gate_cat = torch.cat([g.reshape(-1) for g in gate_values_for_log])
        gate_valid_cat = torch.cat([g.reshape(-1) for g in gate_valid_values])
        target_valid_cat = torch.cat([t.reshape(-1) for t in target_valid_values])
        mv_odise_valid_cat = torch.cat([v.reshape(-1) for v in mv_odise_valid_values])
        mv_lseg_valid_cat = torch.cat([v.reshape(-1) for v in mv_lseg_valid_values])
        mv_diff_cat = torch.cat([v.reshape(-1) for v in mv_diff_values])
        mv_diff_valid_cat = torch.cat([v.reshape(-1) for v in mv_diff_valid_values])
        logs = {
            "loss_source_gate": float(loss_gate_total.detach().cpu()),
            "loss_source_gate_open": float(loss_gate_total.detach().cpu()),
            "loss_source_gate_gt_ce_upper_bound": 0.0,
            "source_gate_mean": float(gate_cat.mean().detach().cpu()),
            "source_gate_std": float(gate_cat.std(unbiased=False).detach().cpu()),
            "source_gate_min": float(gate_cat.min().detach().cpu()),
            "source_gate_max": float(gate_cat.max().detach().cpu()),
            "source_gate_target_mean": float(torch.cat(target_values_for_log).mean().detach().cpu()),
            "source_gate_target_std": float(torch.cat(target_values_for_log).std(unbiased=False).detach().cpu()),
            "source_gate_mv_valid_ratio": float(torch.cat(mv_valid_values).float().mean().detach().cpu()),
            "source_gate_mv_valid_count": float(total_valid_count),
            "source_gate_mv_pair_count": float(total_pair_count),
            "source_gate_mv_diff_mean": float(mv_diff_cat.mean().detach().cpu()),
            "source_gate_mv_diff_valid_mean": float(mv_diff_valid_cat.mean().detach().cpu()),
            "source_gate_loss_valid_count": float(total_loss_valid_count),
            "source_gate_loss_valid_ratio": float(total_loss_valid_count / max(total_valid_count, 1)),
            "source_gate_skipped_no_mv": float(skipped_no_mv),
            "source_gate_mv_odise_mean": float(torch.cat(mv_odise_values).mean().detach().cpu()),
            "source_gate_mv_lseg_mean": float(torch.cat(mv_lseg_values).mean().detach().cpu()),
            "source_gate_gate_mean_valid": float(gate_valid_cat.mean().detach().cpu()),
            "source_gate_gate_std_valid": float(gate_valid_cat.std(unbiased=False).detach().cpu()),
            "source_gate_target_mean_valid": float(target_valid_cat.mean().detach().cpu()),
            "source_gate_target_std_valid": float(target_valid_cat.std(unbiased=False).detach().cpu()),
            "source_gate_mv_odise_mean_valid": float(mv_odise_valid_cat.mean().detach().cpu()),
            "source_gate_mv_lseg_mean_valid": float(mv_lseg_valid_cat.mean().detach().cpu()),
            "source_gate_mask_quality_mean": float(torch.stack(mask_quality_values).mean().detach().cpu()),
        }
        logs.update(batch_probe_logs)
        loss_extra = self._source_gate_regularizers(loss_extra, gate_cat, logs)
        return loss_extra, logs

    def _compute_source_gate_open_reliability_loss(
        self,
        results: Dict,
        batch: Dict,
        text_feats: Optional[torch.Tensor],
        pixel_text_feats: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        source_gate = getattr(model_ref, "source_gate", None)
        empty_logs = self._empty_source_gate_logs()
        if source_gate is None or text_feats is None or pixel_text_feats is None:
            return None, empty_logs

        lseg_all = results.get("pixel_pooled_embeddings", batch.get("pixel_pooled"))
        if lseg_all is None:
            return None, empty_logs

        with torch.no_grad():
            lifted, lifted_valid = build_lifted_3d_masks(
                batch["masks"],
                batch["mask_valid"],
                batch["x_label"],
                batch["y_label"],
                results["batch_indices"],
            )

        global_mask_points = []
        global_odise = []
        global_lseg = []
        global_scene_ids = []
        global_view_ids = []
        item_records = []
        scene_names = batch.get("scene_name")
        frame_stems = batch.get("frame_stem")

        for b in range(len(results["outputs"])):
            if len(results["outputs"][b]) == 0:
                continue
            valid_k = results["mask_valid_from_masks"][b]
            if not valid_k.any():
                continue
            odise_q = batch["mask_embeddings"][b][valid_k].float()
            lseg_q = lseg_all[b][valid_k].float()
            if odise_q.shape[-1] != text_feats.shape[-1] or lseg_q.shape[-1] != pixel_text_feats.shape[-1]:
                continue
            pt_mask = results["batch_indices"] == b
            lifted_b = lifted[pt_mask][:, valid_k]
            lifted_valid_b = lifted_valid[pt_mask]
            mask_points = [
                torch.where((lifted_b[:, k] > 0.5) & lifted_valid_b)[0]
                for k in range(lifted_b.shape[1])
            ]
            start = len(global_mask_points)
            global_mask_points.extend(mask_points)
            global_odise.append(odise_q)
            global_lseg.append(lseg_q)
            scene_id = scene_names[b] if isinstance(scene_names, (list, tuple)) and b < len(scene_names) else str(b)
            view_id = frame_stems[b] if isinstance(frame_stems, (list, tuple)) and b < len(frame_stems) else str(b)
            global_scene_ids.extend([scene_id] * int(valid_k.sum().item()))
            global_view_ids.extend([view_id] * int(valid_k.sum().item()))
            item_records.append((b, valid_k, start, len(mask_points), mask_points))

        if not item_records:
            return None, empty_logs

        global_odise_t = torch.cat(global_odise, dim=0)
        global_lseg_t = torch.cat(global_lseg, dim=0)
        mv_odise_all, mv_lseg_all, mv_valid_all, _mv_pair_count_all = compute_multiview_mask_stability(
            global_mask_points,
            global_odise_t,
            global_lseg_t,
            scene_ids=global_scene_ids,
            view_ids=global_view_ids,
            iou_threshold=self.config.source_gate_mv_iou_threshold,
            topk=self.config.source_gate_mv_topk,
            min_pairs=self.config.source_gate_mv_min_pairs,
            default_stability=self.config.source_gate_mv_default_stability,
        )

        loss_gate_total = None
        num_gate_items = 0
        gate_values_for_log = []
        target_values_for_log = []
        mv_valid_values = []
        mv_odise_values = []
        mv_lseg_values = []
        conflict_values = []
        input_dim = _source_gate_input_dim(model_ref)

        for b, valid_k, start, count, mask_points in item_records:
            pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()
            pred_logits_for_gate = (
                pred_logits.detach()
                if self.config.source_gate_detach_pred_logits
                else pred_logits
            )
            odise_q = batch["mask_embeddings"][b][valid_k].float()
            lseg_q = lseg_all[b][valid_k].float()

            p_odise = _mask_feature_class_probs_tau(
                odise_q,
                text_feats,
                self.config.dual_space_tau_odise,
            )
            p_lseg = _mask_feature_class_probs_tau(
                lseg_q,
                pixel_text_feats,
                self.config.dual_space_tau_lseg,
            )
            p_odise_gate = p_odise.detach() if self.config.source_gate_detach_teacher_probs else p_odise
            p_lseg_gate = p_lseg.detach() if self.config.source_gate_detach_teacher_probs else p_lseg

            point_mask_conf = torch.sigmoid(pred_logits_for_gate).mean(dim=0).detach()
            lifted_point_count = torch.tensor(
                [float(idx.numel()) for idx in mask_points],
                device=pred_logits.device,
                dtype=pred_logits.dtype,
            )
            mask_area = batch["masks"][b][valid_k].float().sum(dim=(1, 2)).detach()
            mv_slice = slice(start, start + count)
            mv_odise = mv_odise_all[mv_slice].to(pred_logits.device)
            mv_lseg = mv_lseg_all[mv_slice].to(pred_logits.device)
            mv_valid = mv_valid_all[mv_slice].to(pred_logits.device)
            evidence = build_source_gate_evidence(
                p_odise_gate,
                p_lseg_gate,
                mask_area=mask_area,
                lifted_point_count=lifted_point_count,
                point_mask_conf=point_mask_conf,
                mv_odise_stability=mv_odise,
                mv_lseg_stability=mv_lseg,
                mv_valid=mv_valid,
                input_dim=input_dim,
            )
            gate = source_gate(evidence)
            target_g, loss_weight, target_logs = build_open_reliability_gate_target(
                p_odise_gate.detach(),
                p_lseg_gate.detach(),
                mv_odise,
                mv_lseg,
                mv_valid,
                odise_prior=self.config.source_gate_odise_prior,
                lseg_prior=self.config.source_gate_lseg_prior,
                conflict_safe_min=self.config.source_gate_conflict_safe_min,
                single_weight=self.config.source_gate_single_weight,
                multiview_weight=self.config.source_gate_multiview_weight,
                conflict_weight=self.config.source_gate_conflict_weight,
            )
            loss_gate = ((gate - target_g) ** 2 * loss_weight).sum() / loss_weight.sum().clamp_min(1.0)
            if torch.isnan(loss_gate) or torch.isinf(loss_gate):
                continue
            loss_gate_total = loss_gate if loss_gate_total is None else loss_gate_total + loss_gate
            num_gate_items += 1
            gate_values_for_log.append(gate.detach())
            target_values_for_log.append(target_g.detach())
            mv_valid_values.append(mv_valid.detach())
            mv_odise_values.append(mv_odise.detach())
            mv_lseg_values.append(mv_lseg.detach())
            conflict_values.append(torch.tensor(target_logs["source_gate_conflict_mean"], device=gate.device))

        if num_gate_items == 0 or loss_gate_total is None:
            return None, empty_logs

        loss_gate_total = loss_gate_total / num_gate_items
        loss_extra = self.config.source_gate_loss_weight * loss_gate_total
        gate_cat = torch.cat([g.reshape(-1) for g in gate_values_for_log])
        logs = {
            "loss_source_gate": float(loss_gate_total.detach().cpu()),
            "loss_source_gate_open": float(loss_gate_total.detach().cpu()),
            "loss_source_gate_gt_ce_upper_bound": 0.0,
            "source_gate_mean": float(gate_cat.mean().detach().cpu()),
            "source_gate_std": float(gate_cat.std(unbiased=False).detach().cpu()),
            "source_gate_min": float(gate_cat.min().detach().cpu()),
            "source_gate_max": float(gate_cat.max().detach().cpu()),
            "source_gate_target_mean": float(torch.cat([t.reshape(-1) for t in target_values_for_log]).mean().detach().cpu()),
            "source_gate_target_std": float(torch.cat([t.reshape(-1) for t in target_values_for_log]).std(unbiased=False).detach().cpu()),
            "source_gate_mv_valid_ratio": float(torch.cat(mv_valid_values).float().mean().detach().cpu()),
            "source_gate_mv_odise_mean": float(torch.cat(mv_odise_values).mean().detach().cpu()),
            "source_gate_mv_lseg_mean": float(torch.cat(mv_lseg_values).mean().detach().cpu()),
            "source_gate_conflict_mean": float(torch.stack(conflict_values).mean().detach().cpu()),
        }
        loss_extra = self._source_gate_regularizers(loss_extra, gate_cat, logs)
        return loss_extra, logs

    def _compute_source_gate_gt_ce_upper_bound_loss(
        self,
        results: Dict,
        batch: Dict,
        text_feats: Optional[torch.Tensor],
        pixel_text_feats: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        raise RuntimeError(
            "gt_ce_upper_bound uses semantic GT and is not allowed in the open-vocabulary training path."
        )
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        source_gate = getattr(model_ref, "source_gate", None)
        empty_logs = self._empty_source_gate_logs()
        if source_gate is None or text_feats is None or pixel_text_feats is None:
            return None, empty_logs

        lseg_all = results.get("pixel_pooled_embeddings", batch.get("pixel_pooled"))
        if lseg_all is None:
            return None, empty_logs

        loss_gate_total = None
        num_gate_items = 0
        gate_values_for_log = []
        input_dim = _source_gate_input_dim(model_ref)
        for b in range(len(results["outputs"])):
            if len(results["outputs"][b]) == 0:
                continue
            valid_k = results["mask_valid_from_masks"][b]
            if not valid_k.any():
                continue
            pt_mask = results["batch_indices"] == b
            gt_b = batch["binary_label_3d"][pt_mask].long()
            if not ((gt_b >= 0) & (gt_b != 255)).any():
                continue

            pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()
            pred_logits_for_gate = pred_logits.detach() if self.config.source_gate_detach_pred_logits else pred_logits
            odise_q = batch["mask_embeddings"][b][valid_k].float()
            lseg_q = lseg_all[b][valid_k].float()
            if odise_q.shape[-1] != text_feats.shape[-1] or lseg_q.shape[-1] != pixel_text_feats.shape[-1]:
                continue

            p_odise = _mask_feature_class_probs_tau(odise_q, text_feats, self.config.dual_space_tau_odise)
            p_lseg = _mask_feature_class_probs_tau(lseg_q, pixel_text_feats, self.config.dual_space_tau_lseg)
            p_odise_gate = p_odise.detach() if self.config.source_gate_detach_teacher_probs else p_odise
            p_lseg_gate = p_lseg.detach() if self.config.source_gate_detach_teacher_probs else p_lseg
            point_mask_conf = torch.sigmoid(pred_logits_for_gate).mean(dim=0).detach()
            if input_dim == 6:
                k = int(point_mask_conf.numel())
                mv_default = torch.full((k,), self.config.source_gate_mv_default_stability, device=pred_logits.device)
                mv_valid = torch.zeros(k, device=pred_logits.device)
                mask_area = batch["masks"][b][valid_k].float().sum(dim=(1, 2)).detach()
                lifted_count = torch.ones(k, device=pred_logits.device)
                evidence = build_text_free_source_gate_evidence(
                    mv_default,
                    mv_default,
                    mv_valid,
                    mask_area,
                    lifted_count,
                    point_mask_conf,
                )
                gate = source_gate(evidence)
                p_gate = (1.0 - gate[:, None]) * p_odise_gate + gate[:, None] * p_lseg_gate
            else:
                evidence = build_source_gate_evidence(
                    p_odise_gate,
                    p_lseg_gate,
                    point_mask_conf=point_mask_conf,
                    input_dim=input_dim,
                )
                gate = source_gate(evidence)
                p_gate = (1.0 - gate) * p_odise_gate + gate * p_lseg_gate
            point_probs = diff2scene_point_class_probs(pred_logits_for_gate, p_gate)
            loss_gate = F.nll_loss(torch.log(point_probs.clamp_min(1e-8)), gt_b, ignore_index=255)
            if torch.isnan(loss_gate) or torch.isinf(loss_gate):
                continue
            loss_gate_total = loss_gate if loss_gate_total is None else loss_gate_total + loss_gate
            num_gate_items += 1
            gate_values_for_log.append(gate.detach())

        if num_gate_items == 0 or loss_gate_total is None:
            return None, empty_logs

        loss_gate_total = loss_gate_total / num_gate_items
        loss_extra = self.config.source_gate_loss_weight * loss_gate_total
        gate_cat = torch.cat([g.reshape(-1) for g in gate_values_for_log])
        logs = {
            "loss_source_gate": float(loss_gate_total.detach().cpu()),
            "loss_source_gate_open": 0.0,
            "loss_source_gate_gt_ce_upper_bound": float(loss_gate_total.detach().cpu()),
            "source_gate_mean": float(gate_cat.mean().detach().cpu()),
            "source_gate_std": float(gate_cat.std(unbiased=False).detach().cpu()),
            "source_gate_min": float(gate_cat.min().detach().cpu()),
            "source_gate_max": float(gate_cat.max().detach().cpu()),
        }
        loss_extra = self._source_gate_regularizers(loss_extra, gate_cat, logs)
        return loss_extra, logs

    def _compute_dual_branch_mask_probe(
        self,
        results: Dict,
        batch: Dict,
    ) -> Dict[str, float]:
        empty_logs = self._empty_dual_branch_logs()
        outputs = results.get("outputs", [])
        if not outputs:
            return empty_logs

        first_output = None
        for batch_outputs in outputs:
            if batch_outputs:
                first_output = batch_outputs[0]
                break
        if first_output is None:
            return empty_logs
        if (
            "pred_mask_logits_odise_branch" not in first_output
            or "pred_mask_logits_lseg_branch" not in first_output
        ):
            return empty_logs

        with torch.no_grad():
            lifted, lifted_valid = build_lifted_3d_masks(
                batch["masks"],
                batch["mask_valid"],
                batch["x_label"],
                batch["y_label"],
                results["batch_indices"],
            )

        odise_losses = []
        lseg_losses = []
        fixed_losses = []
        oracle_losses = []
        odise_ious = []
        lseg_ious = []
        fixed_ious = []
        oracle_ious = []
        delta_losses = []
        odise_wins = []
        lseg_wins = []
        clear_wins = []

        for b in range(len(outputs)):
            if len(outputs[b]) == 0:
                continue
            valid_k = results["mask_valid_from_masks"][b]
            if not valid_k.any():
                continue

            odise_logits_full = outputs[b][0]["pred_mask_logits_odise_branch"]
            lseg_logits_full = outputs[b][0]["pred_mask_logits_lseg_branch"]
            odise_logits = odise_logits_full[:, valid_k].float()
            lseg_logits = lseg_logits_full[:, valid_k].float()

            pt_mask = results["batch_indices"] == b
            gt_masks = lifted[pt_mask][:, valid_k].float()
            pt_valid = lifted_valid[pt_mask]
            if not pt_valid.any():
                continue

            odise_logits = odise_logits[pt_valid]
            lseg_logits = lseg_logits[pt_valid]
            gt_masks = gt_masks[pt_valid]
            pos_cnt = (gt_masks > 0.5).float().sum(dim=0)
            keep = pos_cnt >= self.config.min_points_per_mask
            if not keep.any():
                continue

            odise_logits = odise_logits[:, keep]
            lseg_logits = lseg_logits[:, keep]
            gt_masks = gt_masks[:, keep]
            fixed_logits = 0.5 * (odise_logits + lseg_logits)
            gt_binary = (gt_masks > 0.5).float()

            loss_o = F.binary_cross_entropy_with_logits(
                odise_logits, gt_binary, reduction="none"
            ).mean(dim=0)
            loss_l = F.binary_cross_entropy_with_logits(
                lseg_logits, gt_binary, reduction="none"
            ).mean(dim=0)
            loss_f = F.binary_cross_entropy_with_logits(
                fixed_logits, gt_binary, reduction="none"
            ).mean(dim=0)

            prob_o = torch.sigmoid(odise_logits) > 0.5
            prob_l = torch.sigmoid(lseg_logits) > 0.5
            prob_f = torch.sigmoid(fixed_logits) > 0.5
            gt_bool = gt_binary > 0.5

            def _mask_iou(pred_bool: torch.Tensor, gt_bool_local: torch.Tensor) -> torch.Tensor:
                inter = (pred_bool & gt_bool_local).float().sum(dim=0)
                union = (pred_bool | gt_bool_local).float().sum(dim=0)
                return inter / union.clamp_min(1.0)

            iou_o = _mask_iou(prob_o, gt_bool)
            iou_l = _mask_iou(prob_l, gt_bool)
            iou_f = _mask_iou(prob_f, gt_bool)

            choose_odise = loss_o < loss_l
            loss_oracle = torch.minimum(loss_o, loss_l)
            iou_oracle = torch.where(choose_odise, iou_o, iou_l)
            delta = loss_l - loss_o
            clear = delta.abs() > float(self.config.dual_branch_oracle_margin)

            odise_losses.append(loss_o.detach())
            lseg_losses.append(loss_l.detach())
            fixed_losses.append(loss_f.detach())
            oracle_losses.append(loss_oracle.detach())
            odise_ious.append(iou_o.detach())
            lseg_ious.append(iou_l.detach())
            fixed_ious.append(iou_f.detach())
            oracle_ious.append(iou_oracle.detach())
            delta_losses.append(delta.detach())
            odise_wins.append((loss_o < loss_l).float().detach())
            lseg_wins.append((loss_l < loss_o).float().detach())
            clear_wins.append(clear.float().detach())

        if not odise_losses:
            return empty_logs

        loss_o_cat = torch.cat([v.reshape(-1) for v in odise_losses])
        loss_l_cat = torch.cat([v.reshape(-1) for v in lseg_losses])
        loss_f_cat = torch.cat([v.reshape(-1) for v in fixed_losses])
        loss_oracle_cat = torch.cat([v.reshape(-1) for v in oracle_losses])
        iou_o_cat = torch.cat([v.reshape(-1) for v in odise_ious])
        iou_l_cat = torch.cat([v.reshape(-1) for v in lseg_ious])
        iou_f_cat = torch.cat([v.reshape(-1) for v in fixed_ious])
        iou_oracle_cat = torch.cat([v.reshape(-1) for v in oracle_ious])
        delta_cat = torch.cat([v.reshape(-1) for v in delta_losses])
        odise_win_cat = torch.cat([v.reshape(-1) for v in odise_wins])
        lseg_win_cat = torch.cat([v.reshape(-1) for v in lseg_wins])
        clear_win_cat = torch.cat([v.reshape(-1) for v in clear_wins])
        best_single_loss = torch.minimum(loss_o_cat, loss_l_cat)
        best_single_iou = torch.maximum(iou_o_cat, iou_l_cat)

        return {
            "dual_branch_odise_loss_mean": float(loss_o_cat.mean().detach().cpu()),
            "dual_branch_lseg_loss_mean": float(loss_l_cat.mean().detach().cpu()),
            "dual_branch_fixed_loss_mean": float(loss_f_cat.mean().detach().cpu()),
            "dual_branch_oracle_loss_mean": float(loss_oracle_cat.mean().detach().cpu()),
            "dual_branch_odise_iou_mean": float(iou_o_cat.mean().detach().cpu()),
            "dual_branch_lseg_iou_mean": float(iou_l_cat.mean().detach().cpu()),
            "dual_branch_fixed_iou_mean": float(iou_f_cat.mean().detach().cpu()),
            "dual_branch_oracle_iou_mean": float(iou_oracle_cat.mean().detach().cpu()),
            "dual_branch_delta_loss_mean": float(delta_cat.mean().detach().cpu()),
            "dual_branch_delta_loss_std": float(delta_cat.std(unbiased=False).detach().cpu()),
            "dual_branch_odise_win_rate": float(odise_win_cat.mean().detach().cpu()),
            "dual_branch_lseg_win_rate": float(lseg_win_cat.mean().detach().cpu()),
            "dual_branch_clear_win_rate": float(clear_win_cat.mean().detach().cpu()),
            "dual_branch_oracle_gain_vs_best_single_loss": float(
                (best_single_loss.mean() - loss_oracle_cat.mean()).detach().cpu()
            ),
            "dual_branch_oracle_gain_vs_fixed_loss": float(
                (loss_f_cat.mean() - loss_oracle_cat.mean()).detach().cpu()
            ),
            "dual_branch_oracle_gain_vs_best_single_iou": float(
                (iou_oracle_cat.mean() - best_single_iou.mean()).detach().cpu()
            ),
            "dual_branch_oracle_gain_vs_fixed_iou": float(
                (iou_oracle_cat.mean() - iou_f_cat.mean()).detach().cpu()
            ),
        }

    def _compute_projected_semantic_consistency_probe(
        self,
        results: Dict,
        batch: Dict,
    ) -> Dict[str, float]:
        if str(self.config.projected_sem_probe_region_mode).lower() != "point":
            raise ValueError(
                "Only projected_sem_probe_region_mode='point' is supported in the first version, "
                f"got {self.config.projected_sem_probe_region_mode!r}"
            )

        empty_logs = self._empty_projected_sem_logs()
        lseg_all = results.get("pixel_pooled_embeddings", batch.get("pixel_pooled"))
        if lseg_all is None:
            return empty_logs

        scene_names = batch.get("scene_name") or []
        with torch.no_grad():
            lifted, lifted_valid = build_lifted_3d_masks(
                batch["masks"],
                batch["mask_valid"],
                batch["x_label"],
                batch["y_label"],
                results["batch_indices"],
            )

            point_to_odise = collections.defaultdict(list)
            point_to_lseg = collections.defaultdict(list)
            point_to_views = collections.defaultdict(set)
            total_points = 0

            for b in range(len(results["outputs"])):
                if len(results["outputs"][b]) == 0:
                    continue
                valid_k = results["mask_valid_from_masks"][b]
                if not valid_k.any():
                    continue
                pt_mask = results["batch_indices"] == b
                if not pt_mask.any():
                    continue

                scene_name = (
                    str(scene_names[b])
                    if isinstance(scene_names, (list, tuple)) and b < len(scene_names)
                    else str(b)
                )
                odise_q = batch["mask_embeddings"][b][valid_k].float()
                lseg_q = lseg_all[b][valid_k].float()
                lifted_b = lifted[pt_mask][:, valid_k]
                lifted_valid_b = lifted_valid[pt_mask]
                pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()
                point_coords = batch["ori_coords_3d"][pt_mask][:, 1:4].long()
                mask_area = batch["masks"][b][valid_k].float().sum(dim=(1, 2))

                point_conf = torch.sigmoid(pred_logits)
                total_points += int(point_coords.shape[0])

                for n in range(point_coords.shape[0]):
                    if not bool(lifted_valid_b[n]):
                        continue
                    mask_ids = torch.where(lifted_b[n] > 0.5)[0]
                    if mask_ids.numel() == 0:
                        continue
                    if point_conf.shape[0] > n:
                        choose_idx = mask_ids[point_conf[n, mask_ids].argmax()]
                    else:
                        choose_idx = mask_ids[mask_area[mask_ids].argmin()]
                    key = (
                        scene_name,
                        int(point_coords[n, 0].item()),
                        int(point_coords[n, 1].item()),
                        int(point_coords[n, 2].item()),
                    )
                    point_to_odise[key].append(odise_q[choose_idx].detach())
                    point_to_lseg[key].append(lseg_q[choose_idx].detach())
                    point_to_views[key].add(b)

            valid_keys = [
                key for key, views in point_to_views.items()
                if len(views) >= int(self.config.projected_sem_probe_min_views)
            ]
            if not valid_keys:
                empty_logs["projected_sem_point_count_total"] = float(total_points)
                return empty_logs

            max_points = int(self.config.projected_sem_probe_max_points)
            if len(valid_keys) > max_points:
                perm = torch.randperm(len(valid_keys))[:max_points].tolist()
                valid_keys = [valid_keys[i] for i in perm]

            def _avg_pairwise_cos(x: torch.Tensor) -> torch.Tensor:
                if x.shape[0] < 2:
                    return x.new_tensor(0.0)
                sim = x @ x.t()
                idx = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
                return sim[idx[0], idx[1]].mean()

            odise_cons = []
            lseg_cons = []
            view_counts = []
            for key in valid_keys:
                odise_stack = torch.stack(point_to_odise[key], dim=0).float()
                lseg_stack = torch.stack(point_to_lseg[key], dim=0).float()
                odise_norm = F.normalize(odise_stack, dim=-1)
                lseg_norm = F.normalize(lseg_stack, dim=-1)
                odise_cons.append(_avg_pairwise_cos(odise_norm))
                lseg_cons.append(_avg_pairwise_cos(lseg_norm))
                view_counts.append(float(len(point_to_views[key])))

            odise_cons_t = torch.stack(odise_cons)
            lseg_cons_t = torch.stack(lseg_cons)
            diff_t = lseg_cons_t - odise_cons_t
            abs_diff_t = diff_t.abs()
            g_sem_rule = torch.sigmoid(float(self.config.projected_sem_gate_scale) * diff_t)
            view_count_t = torch.tensor(view_counts, dtype=torch.float32, device=odise_cons_t.device)

            return {
                "projected_sem_point_count_total": float(total_points),
                "projected_sem_point_count_valid": float(len(valid_keys)),
                "projected_sem_valid_ratio": float(len(valid_keys) / max(total_points, 1)),
                "projected_sem_view_count_mean": float(view_count_t.mean().detach().cpu()),
                "projected_sem_view_count_max": float(view_count_t.max().detach().cpu()),
                "projected_sem_odise_consistency_mean": float(odise_cons_t.mean().detach().cpu()),
                "projected_sem_lseg_consistency_mean": float(lseg_cons_t.mean().detach().cpu()),
                "projected_sem_odise_consistency_std": float(odise_cons_t.std(unbiased=False).detach().cpu()),
                "projected_sem_lseg_consistency_std": float(lseg_cons_t.std(unbiased=False).detach().cpu()),
                "projected_sem_diff_mean": float(diff_t.mean().detach().cpu()),
                "projected_sem_diff_std": float(diff_t.std(unbiased=False).detach().cpu()),
                "projected_sem_abs_diff_mean": float(abs_diff_t.mean().detach().cpu()),
                "projected_sem_odise_win_rate": float((odise_cons_t > lseg_cons_t).float().mean().detach().cpu()),
                "projected_sem_lseg_win_rate": float((lseg_cons_t > odise_cons_t).float().mean().detach().cpu()),
                "projected_sem_clear_win_rate_001": float((abs_diff_t > 0.01).float().mean().detach().cpu()),
                "projected_sem_clear_win_rate_003": float((abs_diff_t > 0.03).float().mean().detach().cpu()),
                "projected_sem_clear_win_rate_005": float((abs_diff_t > 0.05).float().mean().detach().cpu()),
                "projected_sem_g_sem_rule_mean": float(g_sem_rule.mean().detach().cpu()),
                "projected_sem_g_sem_rule_std": float(g_sem_rule.std(unbiased=False).detach().cpu()),
                "projected_sem_g_sem_rule_min": float(g_sem_rule.min().detach().cpu()),
                "projected_sem_g_sem_rule_max": float(g_sem_rule.max().detach().cpu()),
            }

    # ----------------------------------------------------------
    # 训练 epoch
    # ----------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss        = AverageMeter()
        epoch_distill     = AverageMeter()
        epoch_aux         = AverageMeter()
        epoch_gate        = AverageMeter()
        batch_time        = AverageMeter()
        verbose_legacy = bool(self.config.enable_verbose_legacy_probes)
        legacy_gate_logs = bool(self.config.enable_legacy_source_gate_logs)
        gate_train_enabled = (
            self.config.source_gate_train
            and epoch >= self.config.source_gate_start_epoch
        )
        gate_target = str(self.config.source_gate_training_target).lower()
        if gate_train_enabled and gate_target in {"open_reliability", "gt_ce_upper_bound"}:
            gate_text_feats, gate_pixel_text_feats = self._get_source_gate_text_features()
        else:
            gate_text_feats, gate_pixel_text_feats = None, None

        accum_steps    = self.accum_steps
        is_distributed = hasattr(self.model, "no_sync")

        self.optimizer.zero_grad(set_to_none=True)
        end_time        = time.time()
        micro_done      = 0
        max_steps       = self.config.max_batches_per_epoch

        for step, batch in enumerate(self.train_loader):
            if max_steps is not None and step >= max_steps:
                break
            micro_done += 1

            batch = self._move_batch_to_device(batch)
            batch["sinput"] = self._build_sparse_tensor(batch)

            is_accum = ((step + 1) % accum_steps != 0)
            sync_ctx = (
                self.model.no_sync()
                if (is_accum and is_distributed)
                else contextlib.nullcontext()
            )

            with sync_ctx:
                with autocast(enabled=self.config.use_amp):
                    try:
                        results  = self.model(batch)
                        criteria = self._make_criteria(results, batch)
                        loss, loss_dict = criteria.compute_loss()
                        gate_extra, gate_logs = (
                            self._compute_source_gate_loss(
                                results,
                                batch,
                                gate_text_feats,
                                gate_pixel_text_feats,
                            )
                            if gate_train_enabled
                            else (
                                None,
                                self._empty_source_gate_logs(),
                            )
                        )
                        if gate_extra is not None:
                            loss = loss + gate_extra
                        loss_dict.update(gate_logs)
                        if self.config.dual_branch_probe:
                            probe_logs = self._compute_dual_branch_mask_probe(results, batch)
                            loss_dict.update(probe_logs)
                        if self.config.projected_sem_probe:
                            projected_logs = self._compute_projected_semantic_consistency_probe(results, batch)
                            loss_dict.update(projected_logs)
                        if gate_train_enabled and gate_target == "text_free_mv_stability":
                            mv_valid_ratio = float(loss_dict.get("source_gate_mv_valid_ratio", 0.0))
                            if mv_valid_ratio <= 0.0:
                                self._source_gate_zero_mv_steps += 1
                            else:
                                self._source_gate_zero_mv_steps = 0
                                self._warned_source_gate_zero_mv = False
                            if (
                                self.is_main
                                and self._source_gate_zero_mv_steps >= 100
                                and not self._warned_source_gate_zero_mv
                            ):
                                print(
                                    "[SourceGate/TextFreeMV] mv_valid_ratio is 0 for many steps. "
                                    "Check whether batch contains same-scene multi-view samples, "
                                    "scene_name/frame_stem metadata, lifted mask point sets, and iou_threshold."
                                )
                                self._warned_source_gate_zero_mv = True
                        if torch.isnan(loss) or torch.isinf(loss):
                            raise ValueError(f"Invalid loss: {loss}")
                    except Exception as e:
                        print(f"\n[MaskDistillTrainer] Forward error: {e}")
                        raise

                scaled = loss / accum_steps
                self.scaler.scale(scaled).backward()

            epoch_loss.update(loss.item())
            epoch_distill.update(loss_dict["loss_mask_distill"])
            epoch_aux.update(loss_dict["loss_aux"])
            epoch_gate.update(loss_dict.get("loss_source_gate", 0.0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (step + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.config.grad_clip_norm,
                )

                # ---- logit_scale 审查：backward 后、step 前 ----
                if self.global_step % self.config.log_every_steps == 0 and self.is_main:
                    ls = self.model.logit_scale if hasattr(self.model, "logit_scale") else None
                    if ls is None and hasattr(self.model, "module"):
                        ls = getattr(self.model.module, "logit_scale", None)
                    if ls is not None:
                        ls_val   = ls.item()
                        ls_exp   = ls.exp().item()
                        clamped  = ls_exp >= 100.0
                        if ls.grad is not None:
                            g_max  = ls.grad.abs().max().item()
                            g_mean = ls.grad.abs().mean().item()
                            print(
                                f"[logit_scale] raw={ls_val:.6e}  "
                                f"exp={ls_exp:.6e}{'  *** CLAMPED ***' if clamped else ''}  "
                                f"grad_max={g_max:.6e}  grad_mean={g_mean:.6e}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[logit_scale] raw={ls_val:.6e}  "
                                f"exp={ls_exp:.6e}{'  *** CLAMPED ***' if clamped else ''}  "
                                f"grad=None (no gradient)",
                                flush=True,
                            )
                # ---- end logit_scale 审查 ----

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.global_step >= self.warmup_steps:
                    if self.config.scheduler_type != "plateau":
                        self.scheduler.step()

                if self.writer is not None:
                    self.writer.add_scalar("Loss/Train_Step",         loss.item(),                        self.global_step)
                    self.writer.add_scalar("Loss/Train_Alignment",    loss_dict["loss_mask_distill"],     self.global_step)
                    self.writer.add_scalar("Loss/Train_MaskDistill",  loss_dict["loss_mask_distill"],     self.global_step)
                    self.writer.add_scalar("Loss/Train_Aux",          loss_dict["loss_aux"],              self.global_step)
                    if legacy_gate_logs:
                        self.writer.add_scalar("Loss/Train_SourceGate",   loss_dict.get("loss_source_gate", 0.0), self.global_step)
                    if verbose_legacy:
                        self.writer.add_scalar("Loss/Train_SourceGate_Open", loss_dict.get("loss_source_gate_open", 0.0), self.global_step)
                        self.writer.add_scalar("Loss/Train_SourceGate_TextFreeMV", loss_dict.get("loss_source_gate_open", 0.0), self.global_step)
                        self.writer.add_scalar("Loss/Train_SourceGate_OpenReliability", loss_dict.get("loss_source_gate_open", 0.0), self.global_step)
                        self.writer.add_scalar("Loss/Train_SourceGate_GTCE_UpperBound", loss_dict.get("loss_source_gate_gt_ce_upper_bound", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/train_mean",   loss_dict.get("source_gate_mean", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/train_std",    loss_dict.get("source_gate_std", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/train_min",    loss_dict.get("source_gate_min", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/train_max",    loss_dict.get("source_gate_max", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/target_mean",  loss_dict.get("source_gate_target_mean", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/target_std",   loss_dict.get("source_gate_target_std", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/mv_valid_ratio", loss_dict.get("source_gate_mv_valid_ratio", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/mv_valid_count", loss_dict.get("source_gate_mv_valid_count", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/mv_pair_count", loss_dict.get("source_gate_mv_pair_count", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/mv_diff_mean", loss_dict.get("source_gate_mv_diff_mean", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/mv_diff_valid_mean", loss_dict.get("source_gate_mv_diff_valid_mean", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/loss_valid_count", loss_dict.get("source_gate_loss_valid_count", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/loss_valid_ratio", loss_dict.get("source_gate_loss_valid_ratio", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/skipped_no_mv", loss_dict.get("source_gate_skipped_no_mv", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/mv_odise_mean", loss_dict.get("source_gate_mv_odise_mean", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/mv_lseg_mean", loss_dict.get("source_gate_mv_lseg_mean", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/gate_mean_valid", loss_dict.get("source_gate_gate_mean_valid", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/gate_std_valid", loss_dict.get("source_gate_gate_std_valid", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/target_mean_valid", loss_dict.get("source_gate_target_mean_valid", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/target_std_valid", loss_dict.get("source_gate_target_std_valid", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/mv_odise_mean_valid", loss_dict.get("source_gate_mv_odise_mean_valid", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/mv_lseg_mean_valid", loss_dict.get("source_gate_mv_lseg_mean_valid", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/batch_unique_scenes", loss_dict.get("batch_unique_scenes", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/batch_frames_per_scene_mean", loss_dict.get("batch_frames_per_scene_mean", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/batch_same_scene_pair_count", loss_dict.get("batch_same_scene_pair_count", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/conflict_mean", loss_dict.get("source_gate_conflict_mean", 0.0), self.global_step)
                        self.writer.add_scalar("SourceGate/mask_quality_mean", loss_dict.get("source_gate_mask_quality_mean", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/odise_loss_mean", loss_dict.get("dual_branch_odise_loss_mean", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/lseg_loss_mean", loss_dict.get("dual_branch_lseg_loss_mean", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/fixed_loss_mean", loss_dict.get("dual_branch_fixed_loss_mean", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/oracle_loss_mean", loss_dict.get("dual_branch_oracle_loss_mean", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/odise_iou_mean", loss_dict.get("dual_branch_odise_iou_mean", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/lseg_iou_mean", loss_dict.get("dual_branch_lseg_iou_mean", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/fixed_iou_mean", loss_dict.get("dual_branch_fixed_iou_mean", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/oracle_iou_mean", loss_dict.get("dual_branch_oracle_iou_mean", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/delta_loss_mean", loss_dict.get("dual_branch_delta_loss_mean", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/delta_loss_std", loss_dict.get("dual_branch_delta_loss_std", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/odise_win_rate", loss_dict.get("dual_branch_odise_win_rate", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/lseg_win_rate", loss_dict.get("dual_branch_lseg_win_rate", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/clear_win_rate", loss_dict.get("dual_branch_clear_win_rate", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/oracle_gain_vs_best_single_loss", loss_dict.get("dual_branch_oracle_gain_vs_best_single_loss", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/oracle_gain_vs_fixed_loss", loss_dict.get("dual_branch_oracle_gain_vs_fixed_loss", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/oracle_gain_vs_best_single_iou", loss_dict.get("dual_branch_oracle_gain_vs_best_single_iou", 0.0), self.global_step)
                        self.writer.add_scalar("DualBranch/oracle_gain_vs_fixed_iou", loss_dict.get("dual_branch_oracle_gain_vs_fixed_iou", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/point_count_total", loss_dict.get("projected_sem_point_count_total", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/point_count_valid", loss_dict.get("projected_sem_point_count_valid", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/valid_ratio", loss_dict.get("projected_sem_valid_ratio", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/view_count_mean", loss_dict.get("projected_sem_view_count_mean", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/view_count_max", loss_dict.get("projected_sem_view_count_max", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/odise_consistency_mean", loss_dict.get("projected_sem_odise_consistency_mean", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/lseg_consistency_mean", loss_dict.get("projected_sem_lseg_consistency_mean", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/odise_consistency_std", loss_dict.get("projected_sem_odise_consistency_std", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/lseg_consistency_std", loss_dict.get("projected_sem_lseg_consistency_std", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/diff_mean", loss_dict.get("projected_sem_diff_mean", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/diff_std", loss_dict.get("projected_sem_diff_std", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/abs_diff_mean", loss_dict.get("projected_sem_abs_diff_mean", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/odise_win_rate", loss_dict.get("projected_sem_odise_win_rate", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/lseg_win_rate", loss_dict.get("projected_sem_lseg_win_rate", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/clear_win_rate_001", loss_dict.get("projected_sem_clear_win_rate_001", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/clear_win_rate_003", loss_dict.get("projected_sem_clear_win_rate_003", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/clear_win_rate_005", loss_dict.get("projected_sem_clear_win_rate_005", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/g_sem_rule_mean", loss_dict.get("projected_sem_g_sem_rule_mean", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/g_sem_rule_std", loss_dict.get("projected_sem_g_sem_rule_std", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/g_sem_rule_min", loss_dict.get("projected_sem_g_sem_rule_min", 0.0), self.global_step)
                        self.writer.add_scalar("ProjectedSem/g_sem_rule_max", loss_dict.get("projected_sem_g_sem_rule_max", 0.0), self.global_step)
                    self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], self.global_step)

                    model_ref = self.model.module if hasattr(self.model, "module") else self.model
                    # Log fusion alpha (ODISE-residual fusion mixing weight)
                    fuse = getattr(model_ref, "fuse_embed", None)
                    if fuse is not None and hasattr(fuse, "alpha"):
                        self.writer.add_scalar("Fusion/alpha", fuse.alpha.item(), self.global_step)
                    if hasattr(model_ref, "_get_semantic_fusion_weights"):
                        w_o, w_l = model_ref._get_semantic_fusion_weights()
                        self.writer.add_scalar(
                            "FusionSemantic/w_odise",
                            float(w_o.detach().cpu()),
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "FusionSemantic/w_lseg",
                            float(w_l.detach().cpu()),
                            self.global_step,
                        )

                self.global_step += 1

            self._adjust_learning_rate_warmup(self.global_step)

            if step % self.config.log_every_steps == 0 and self.is_main:
                lr  = self.optimizer.param_groups[0]["lr"]
                eta = batch_time.avg * (len(self.train_loader) - step)
                print(
                    f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                    f"Step [{step}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(alignment={loss_dict['loss_mask_distill']:.4f} aux={loss_dict['loss_aux']:.4f}) "
                    f"avg={epoch_loss.avg:.4f}  LR: {lr:.2e}  ETA: {eta:.0f}s"
                )
                if gate_train_enabled and legacy_gate_logs:
                    print(
                        "  [OpenVocab Gate Training] "
                        f"source_gate_target={self.config.source_gate_training_target} "
                        f"loss_open={loss_dict.get('loss_source_gate_open', 0.0):.4f} "
                        f"mv_valid={loss_dict.get('source_gate_mv_valid_ratio', 0.0):.3f} "
                        f"batch_unique_scenes={loss_dict.get('batch_unique_scenes', 0.0):.0f} "
                        f"batch_frames_per_scene_mean={loss_dict.get('batch_frames_per_scene_mean', 0.0):.2f} "
                        f"batch_same_scene_pair_count={loss_dict.get('batch_same_scene_pair_count', 0.0):.0f} "
                        f"mv_valid_count={loss_dict.get('source_gate_mv_valid_count', 0.0):.0f} "
                        f"mv_pair_count={loss_dict.get('source_gate_mv_pair_count', 0.0):.0f} "
                        f"mv_diff={loss_dict.get('source_gate_mv_diff_mean', 0.0):.4f} "
                        f"mv_diff_valid={loss_dict.get('source_gate_mv_diff_valid_mean', 0.0):.4f} "
                        f"loss_valid_count={loss_dict.get('source_gate_loss_valid_count', 0.0):.0f} "
                        f"loss_valid_ratio={loss_dict.get('source_gate_loss_valid_ratio', 0.0):.3f} "
                        f"skipped_no_mv={loss_dict.get('source_gate_skipped_no_mv', 0.0):.0f} "
                        f"mv_odise={loss_dict.get('source_gate_mv_odise_mean', 0.0):.3f} "
                        f"mv_lseg={loss_dict.get('source_gate_mv_lseg_mean', 0.0):.3f} "
                        f"gate_valid={loss_dict.get('source_gate_gate_mean_valid', 0.0):.4f} "
                        f"gate_std_valid={loss_dict.get('source_gate_gate_std_valid', 0.0):.4f} "
                        f"target_valid={loss_dict.get('source_gate_target_mean_valid', 0.0):.4f} "
                        f"target_std_valid={loss_dict.get('source_gate_target_std_valid', 0.0):.4f} "
                        f"mask_quality={loss_dict.get('source_gate_mask_quality_mean', 0.0):.3f} "
                        f"gate_mean={loss_dict.get('source_gate_mean', 0.0):.4f} "
                        f"gate_std={loss_dict.get('source_gate_std', 0.0):.4f}"
                    )
                if (
                    verbose_legacy
                    and self.config.dual_branch_probe
                    and step % max(int(self.config.dual_branch_probe_log_every), 1) == 0
                    and self.is_main
                ):
                    print(
                        "  [DualBranch Probe] "
                        f"odise_loss={loss_dict.get('dual_branch_odise_loss_mean', 0.0):.4f} "
                        f"lseg_loss={loss_dict.get('dual_branch_lseg_loss_mean', 0.0):.4f} "
                        f"fixed_loss={loss_dict.get('dual_branch_fixed_loss_mean', 0.0):.4f} "
                        f"oracle_loss={loss_dict.get('dual_branch_oracle_loss_mean', 0.0):.4f} "
                        f"odise_iou={loss_dict.get('dual_branch_odise_iou_mean', 0.0):.4f} "
                        f"lseg_iou={loss_dict.get('dual_branch_lseg_iou_mean', 0.0):.4f} "
                        f"fixed_iou={loss_dict.get('dual_branch_fixed_iou_mean', 0.0):.4f} "
                        f"oracle_iou={loss_dict.get('dual_branch_oracle_iou_mean', 0.0):.4f} "
                        f"clear_win={loss_dict.get('dual_branch_clear_win_rate', 0.0):.3f} "
                        f"oracle_gain_fixed_loss={loss_dict.get('dual_branch_oracle_gain_vs_fixed_loss', 0.0):.4f}"
                    )
                if (
                    verbose_legacy
                    and self.config.projected_sem_probe
                    and step % max(int(self.config.projected_sem_probe_log_every), 1) == 0
                    and self.is_main
                ):
                    print(
                        "  [ProjectedSem Probe] "
                        f"valid_points={loss_dict.get('projected_sem_point_count_valid', 0.0):.0f} "
                        f"view_mean={loss_dict.get('projected_sem_view_count_mean', 0.0):.2f} "
                        f"odise_cons={loss_dict.get('projected_sem_odise_consistency_mean', 0.0):.4f} "
                        f"lseg_cons={loss_dict.get('projected_sem_lseg_consistency_mean', 0.0):.4f} "
                        f"diff={loss_dict.get('projected_sem_diff_mean', 0.0):.4f} "
                        f"abs_diff={loss_dict.get('projected_sem_abs_diff_mean', 0.0):.4f} "
                        f"odise_win={loss_dict.get('projected_sem_odise_win_rate', 0.0):.3f} "
                        f"lseg_win={loss_dict.get('projected_sem_lseg_win_rate', 0.0):.3f} "
                        f"clear@0.03={loss_dict.get('projected_sem_clear_win_rate_003', 0.0):.3f} "
                        f"g_sem_mean={loss_dict.get('projected_sem_g_sem_rule_mean', 0.0):.4f} "
                        f"g_sem_std={loss_dict.get('projected_sem_g_sem_rule_std', 0.0):.4f}"
                    )

        # 处理 epoch 末尾剩余 micro-step
        remaining = micro_done % accum_steps
        if remaining > 0 and max_steps is None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.config.grad_clip_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.global_step >= self.warmup_steps:
                if self.config.scheduler_type != "plateau":
                    self.scheduler.step()
            self.global_step += 1

        if self.writer is not None:
            self.writer.add_scalar("Loss/Train_MaskDistill_Epoch", epoch_distill.avg, epoch)
            self.writer.add_scalar("Loss/Train_Aux_Epoch",         epoch_aux.avg,     epoch)
            if self.config.enable_legacy_source_gate_logs:
                self.writer.add_scalar("Loss/Train_SourceGate_Epoch",  epoch_gate.avg,    epoch)

        return epoch_loss.avg

    # ----------------------------------------------------------
    # 验证 epoch
    # ----------------------------------------------------------

    def _get_text_features(self) -> Optional[torch.Tensor]:
        if self._text_features is None:
            if self.is_main:
                print("[MaskDistillTrainer] Building hybrid/text features for semantic mIoU ...")
            try:
                self._text_features = build_text_features(
                    device=self.device,
                    clip_model=self.config.semantic_clip_model,
                    prompt_template=self.config.semantic_prompt_template,
                )
                if self.is_main:
                    print(f"  text_features shape: {self._text_features.shape}")
            except Exception as e:
                if self.is_main:
                    print(f"  [WARNING] Failed to build text features: {e}")
                    print("  Semantic mIoU will be skipped.")
                self._text_features = None
        return self._text_features

    def _get_pixel_text_features(self) -> Optional[torch.Tensor]:
        if self._pixel_text_features is None:
            if self.is_main:
                print("[MaskDistillTrainer] Building CLIP/text features for ODISE PC mIoU ...")
            try:
                self._pixel_text_features = build_text_features(
                    device=self.device,
                    clip_model=self.config.semantic_pixel_clip_model,
                    prompt_template=self.config.semantic_prompt_template,
                )
                if self.is_main:
                    print(f"  pixel_text_features shape: {self._pixel_text_features.shape}")
            except Exception as e:
                if self.is_main:
                    print(f"  [WARNING] Failed to build CLIP/text features: {e}")
                    print("  ODISE PC mIoU will be skipped.")
                self._pixel_text_features = None
        return self._pixel_text_features

    def _load_source_gate_train_queries(self) -> Optional[list]:
        query_file = self.config.source_gate_train_query_file
        if not query_file:
            return None
        if not os.path.exists(query_file):
            raise FileNotFoundError(f"source_gate_train_query_file not found: {query_file}")
        with open(query_file, "r") as f:
            queries = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if self.config.source_gate_num_train_queries > 0:
            queries = queries[: self.config.source_gate_num_train_queries]
        if not queries:
            raise ValueError(f"source_gate_train_query_file has no usable queries: {query_file}")
        return queries

    def _get_source_gate_text_features(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        queries = self._load_source_gate_train_queries()
        if queries is None:
            if self.is_main and not self._warned_source_gate_query_fallback:
                print(
                    "[SourceGate] No source_gate_train_query_file set; using configured semantic "
                    "queries only as unsupervised teacher-response probes, not as GT labels."
                )
                self._warned_source_gate_query_fallback = True
            return self._get_text_features(), self._get_pixel_text_features()

        if self._source_gate_text_features is None or self._source_gate_pixel_text_features is None:
            if self.is_main:
                print(f"[SourceGate] Building open semantic query features from {len(queries)} phrases ...")
            self._source_gate_text_features = build_text_features(
                class_names=queries,
                device=self.device,
                clip_model=self.config.semantic_clip_model,
                prompt_template=self.config.semantic_prompt_template,
            )
            self._source_gate_pixel_text_features = build_text_features(
                class_names=queries,
                device=self.device,
                clip_model=self.config.semantic_pixel_clip_model,
                prompt_template=self.config.semantic_prompt_template,
            )
        return self._source_gate_text_features, self._source_gate_pixel_text_features

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict:
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss      = AverageMeter()
        val_distill   = AverageMeter()
        val_aux       = AverageMeter()
        verbose_legacy = bool(self.config.enable_verbose_legacy_probes)

        text_feats = self._get_text_features()
        pixel_text_feats = self._get_pixel_text_features()
        pc_tracker = (
            ODISEPCSemanticMIoUTracker(pc_lambda=self.config.semantic_pc_lambda)
            if verbose_legacy and text_feats is not None and pixel_text_feats is not None
            else None
        )
        odise256_probe_accs = (
            {
                "hybrid_odise256": _SemanticAccumulator(),
                "clip_odise256": _SemanticAccumulator(),
                "odise_odise256": _SemanticAccumulator(),
                "base_odise256": _SemanticAccumulator(),
                "refine_odise256": _SemanticAccumulator(),
                "lseg_semproj_odise256": _SemanticAccumulator(),
                "semantic_query_odise256": _SemanticAccumulator(),
            }
            if verbose_legacy and text_feats is not None
            else None
        )
        dual_space_accs = (
            {
                "odise_only_text256": _SemanticAccumulator(),
                "lseg_only_text512": _SemanticAccumulator(),
                "current_fused_text256": _SemanticAccumulator(),
                "dual_space_fixed": _SemanticAccumulator(),
                "dual_space_confidence": _SemanticAccumulator(),
                "dual_space_gate": _SemanticAccumulator(),
                "odise_only": _SemanticAccumulator(),
                "lseg_only": _SemanticAccumulator(),
                "fixed_05": _SemanticAccumulator(),
                "lseg_06": _SemanticAccumulator(),
                "lseg_07": _SemanticAccumulator(),
                "lseg_08": _SemanticAccumulator(),
                "size_aware": _SemanticAccumulator(),
                "projected_gate": _SemanticAccumulator(),
                "projected_size_gate": _SemanticAccumulator(),
            }
            if self.config.dual_space_eval and text_feats is not None and pixel_text_feats is not None
            else None
        )
        dual_space_group_accs = (
            {
                group: {name: _SemanticAccumulator() for name in self._semantic_ablation_names()}
                for group in self._semantic_size_group_names()
            }
            if dual_space_accs is not None and self.config.semantic_readout_ablation
            else None
        )
        source_gate_val_values = []
        dual_weight_sum = self.config.dual_space_odise_weight + self.config.dual_space_lseg_weight
        if dual_weight_sum <= 0:
            raise ValueError("dual_space_odise_weight + dual_space_lseg_weight must be positive")
        dual_w_odise = self.config.dual_space_odise_weight / dual_weight_sum
        dual_w_lseg = self.config.dual_space_lseg_weight / dual_weight_sum

        # Mask-level mIoU
        mask_tracker  = MaskMIoUTracker(threshold=0.5)

        for batch_idx, batch in enumerate(self.val_loader):
            batch = self._move_batch_to_device(batch)
            batch["sinput"] = self._build_sparse_tensor(batch)

            with autocast(enabled=self.config.use_amp):
                results          = self.model(batch)
                criteria         = self._make_criteria(results, batch)
                loss, loss_dict  = criteria.compute_loss()

            val_loss.update(loss.item())
            val_distill.update(loss_dict["loss_mask_distill"])
            val_aux.update(loss_dict["loss_aux"])

            from experiment_mask_distill.criterion_mask_distill import build_lifted_3d_masks
            mask_valid = results["mask_valid_from_masks"]    # (B, K_max)
            mask_masks = results["mask_masks"]               # (B, K_max, H, W)

            lifted, lifted_valid = build_lifted_3d_masks(
                masks         = mask_masks,
                mask_valid    = mask_valid,
                x_label       = batch["x_label"],
                y_label       = batch["y_label"],
                batch_indices = results["batch_indices"],
            )
            # lifted: (N_total, K_max)
            eval_lseg_all = results.get("pixel_pooled_embeddings", batch.get("pixel_pooled"))
            semantic_eval_aux = self._build_eval_semantic_aux(
                results=results,
                batch=batch,
                lifted=lifted,
                lifted_valid=lifted_valid,
                lseg_all=eval_lseg_all,
            )

            # ---- 语义 mIoU：Hybrid / CLIP / Final-PC ----
            if pc_tracker is not None:
                fused_all = results["fused_embeddings"]
                pixel_all = results.get("pixel_pooled_embeddings", None)
                mask_valid_for_sem = results["mask_valid_from_masks"]
                for b in range(len(results["outputs"])):
                    if len(results["outputs"][b]) == 0:
                        continue
                    valid_k = mask_valid_for_sem[b]
                    if not valid_k.any():
                        continue
                    pt_mask = results["batch_indices"] == b
                    pred_logits_full = results["outputs"][b][0]["pred_mask_logits"]
                    pred_logits = pred_logits_full[:, valid_k].float()
                    gt_b = batch["binary_label_3d"][pt_mask]
                    fused_b = fused_all[b][valid_k]
                    if pred_logits.shape[0] != gt_b.numel():
                        raise RuntimeError(
                            f"Semantic eval batch={batch_idx} item={b}: "
                            f"logit rows {pred_logits.shape[0]} != gt labels {gt_b.numel()}"
                        )
                    if pixel_all is not None:
                        pixel_b = pixel_all[b][valid_k]
                        if pixel_b.shape[-1] != pixel_text_feats.shape[-1]:
                            if self.is_main and not self._warned_pixel_text_dim_mismatch:
                                print(
                                    "[WARNING] Skip semantic Hybrid/CLIP/Final-PC mIoU: "
                                    f"pixel feature dim={pixel_b.shape[-1]} but "
                                    f"text dim={pixel_text_feats.shape[-1]}. "
                                    "Regenerate mask-pooled CLIP features with the same CLIP model "
                                    "as semantic_pixel_clip_model."
                                )
                                self._warned_pixel_text_dim_mismatch = True
                        else:
                            pc_tracker.update(
                                gt_labels=gt_b,
                                hybrid_features=fused_b,
                                clip_features=pixel_b,
                                pred_mask_logits=pred_logits,
                                hybrid_text_features=text_feats,
                                clip_text_features=pixel_text_feats,
                                salient_masks=lifted[pt_mask][:, valid_k].float(),
                            )

            # ---- ODISE-256 probe metrics for fusion components ----
            # These expose the two learned branches and the pre/post-refine states:
            #   odise_odise256: ODISE mask branch in the 256D ODISE text space
            #   clip_odise256:  learned LSeg/CLIP projection into the ODISE 256D text space
            #   base_odise256:  mask_tokens + gate * clip_tokens, before refine()
            #   refine_odise256/hybrid_odise256: final fused token after refine residual
            if odise256_probe_accs is not None:
                component_map = {
                    "hybrid_odise256": results["fused_embeddings"],
                    "clip_odise256": results["clip_projected_embeddings"],
                    "odise_odise256": results["odise_projected_embeddings"],
                    "base_odise256": results["fusion_base_embeddings"],
                    "refine_odise256": results["fused_embeddings"],
                    "lseg_semproj_odise256": results.get("lseg_semantic_embeddings"),
                    "semantic_query_odise256": results.get("semantic_embeddings"),
                }
                mask_valid_for_sem = results["mask_valid_from_masks"]
                for b in range(len(results["outputs"])):
                    if len(results["outputs"][b]) == 0:
                        continue
                    valid_k = mask_valid_for_sem[b]
                    if not valid_k.any():
                        continue
                    pt_mask = results["batch_indices"] == b
                    gt_b = batch["binary_label_3d"][pt_mask]
                    pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()
                    for name, features_all in component_map.items():
                        if features_all is None:
                            continue
                        pred = diff2scene_mask_feature_predict(
                            point_mask_logits=pred_logits,
                            mask_features=features_all[b][valid_k],
                            text_features=text_feats,
                        )
                        odise256_probe_accs[name].update_labels(
                            pred,
                            gt_b.detach().cpu().long(),
                        )

            # ---- Dual-space semantic probability fusion ----
            # ODISE raw256 is read by ODISE text256. LSeg raw512 is read by
            # CLIP text512. No LSeg->ODISE projection is used here.
            if dual_space_accs is not None:
                mask_valid_for_sem = results["mask_valid_from_masks"]
                lseg_all = eval_lseg_all
                if lseg_all is None:
                    raise RuntimeError("Dual-space eval requires raw LSeg pixel_pooled features.")
                for b in range(len(results["outputs"])):
                    if len(results["outputs"][b]) == 0:
                        continue
                    valid_k = mask_valid_for_sem[b]
                    if not valid_k.any():
                        continue
                    pt_mask = results["batch_indices"] == b
                    gt_b = batch["binary_label_3d"][pt_mask]
                    pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()
                    odise_q = batch["mask_embeddings"][b][valid_k].float()
                    lseg_q = lseg_all[b][valid_k].float()
                    fused_q = results["fused_embeddings"][b][valid_k].float()
                    image_area = float(batch["masks"][b].shape[-1] * batch["masks"][b].shape[-2])
                    mask_area = batch["masks"][b][valid_k].float().sum(dim=(1, 2))
                    mask_area_ratio = (mask_area / max(image_area, 1.0)).clamp(0.0, 1.0)
                    projected_diff = semantic_eval_aux["projected_mask_diff"].get(b)
                    projected_valid = semantic_eval_aux["projected_mask_valid"].get(b)

                    if odise_q.shape[-1] != text_feats.shape[-1]:
                        raise RuntimeError(
                            f"Dual-space ODISE dim mismatch: mask={odise_q.shape[-1]} text={text_feats.shape[-1]}"
                        )
                    if lseg_q.shape[-1] != pixel_text_feats.shape[-1]:
                        raise RuntimeError(
                            f"Dual-space LSeg dim mismatch: mask={lseg_q.shape[-1]} text={pixel_text_feats.shape[-1]}"
                        )
                    if fused_q.shape[-1] != text_feats.shape[-1]:
                        raise RuntimeError(
                            f"Dual-space fused dim mismatch: mask={fused_q.shape[-1]} text={text_feats.shape[-1]}"
                        )

                    p_odise = _mask_feature_class_probs_tau(
                        odise_q,
                        text_feats,
                        self.config.dual_space_tau_odise,
                    )
                    p_lseg = _mask_feature_class_probs_tau(
                        lseg_q,
                        pixel_text_feats,
                        self.config.dual_space_tau_lseg,
                    )
                    p_fixed = dual_w_odise * p_odise + dual_w_lseg * p_lseg
                    p_conf = _dual_space_confidence_probs(
                        p_odise,
                        p_lseg,
                        self.config.dual_space_conf_min,
                        self.config.dual_space_conf_max,
                    )
                    semantic_probs = self._compute_semantic_readout_probs(
                        p_odise=p_odise,
                        p_lseg=p_lseg,
                        mask_area_ratio=mask_area_ratio,
                        projected_diff=projected_diff,
                        projected_valid=projected_valid,
                    )

                    dual_preds = {
                        "odise_only_text256": diff2scene_mask_feature_predict(
                            pred_logits,
                            odise_q,
                            text_feats,
                        ),
                        "lseg_only_text512": diff2scene_mask_feature_predict(
                            pred_logits,
                            lseg_q,
                            pixel_text_feats,
                        ),
                        "current_fused_text256": diff2scene_mask_feature_predict(
                            pred_logits,
                            fused_q,
                            text_feats,
                        ),
                        "dual_space_fixed": diff2scene_class_probs_predict(
                            pred_logits,
                            p_fixed,
                        ),
                        "dual_space_confidence": diff2scene_class_probs_predict(
                            pred_logits,
                            p_conf,
                        ),
                    }
                    for name, class_probs in semantic_probs.items():
                        dual_preds[name] = diff2scene_class_probs_predict(
                            pred_logits,
                            class_probs,
                        )
                    model_ref = self.model.module if hasattr(self.model, "module") else self.model
                    source_gate = getattr(model_ref, "source_gate", None)
                    if source_gate is not None:
                        point_mask_conf = torch.sigmoid(pred_logits).mean(dim=0).detach()
                        input_dim = _source_gate_input_dim(model_ref)
                        if input_dim == 6:
                            lifted_b = lifted[pt_mask][:, valid_k]
                            lifted_point_count = (lifted_b > 0.5).float().sum(dim=0)
                            mask_area = batch["masks"][b][valid_k].float().sum(dim=(1, 2))
                            mv_default = torch.full_like(point_mask_conf, self.config.source_gate_mv_default_stability)
                            mv_valid = torch.zeros_like(point_mask_conf)
                            evidence = build_text_free_source_gate_evidence(
                                mv_default,
                                mv_default,
                                mv_valid,
                                mask_area,
                                lifted_point_count,
                                point_mask_conf,
                            )
                            gate = source_gate(evidence)
                            p_gate = (1.0 - gate[:, None]) * p_odise + gate[:, None] * p_lseg
                        else:
                            evidence = build_source_gate_evidence(
                                p_odise,
                                p_lseg,
                                point_mask_conf=point_mask_conf,
                                input_dim=input_dim,
                            )
                            gate = source_gate(evidence)
                            p_gate = (1.0 - gate) * p_odise + gate * p_lseg
                        dual_preds["dual_space_gate"] = diff2scene_class_probs_predict(
                            pred_logits,
                            p_gate,
                        )
                        source_gate_val_values.append(gate.detach().reshape(-1).cpu())
                    for name, pred in dual_preds.items():
                        dual_space_accs[name].update_labels(
                            pred,
                            gt_b.detach().cpu().long(),
                        )
                        if dual_space_group_accs is None or name not in self._semantic_ablation_names():
                            continue
                        point_group = semantic_eval_aux["point_groups"][b]
                        if point_group is None:
                            continue
                        gt_cpu = gt_b.detach().cpu().long()
                        for group_idx, group_name in enumerate(self._semantic_size_group_names()):
                            group_mask = (point_group == group_idx).detach().cpu()
                            if bool(group_mask.any()):
                                dual_space_group_accs[group_name][name].update_labels(
                                    pred[group_mask],
                                    gt_cpu[group_mask],
                                )

            # ---- Mask-level mIoU ----
            B      = mask_valid.shape[0]
            K_max  = mask_valid.shape[1]
            fused_embeddings_all = results["fused_embeddings"]   # (B, K_max, 256)

            for b in range(B):
                if len(results["outputs"][b]) == 0:
                    continue
                valid_k = mask_valid[b]
                if not valid_k.any():
                    continue

                # 与训练 _mask_distill_loss 完全一致的过滤链路
                pred_logits_full = results["outputs"][b][0]["pred_mask_logits"]  # (N_b, K_max)
                pred_logits = pred_logits_full[:, valid_k].float()               # (N_b, K_valid)

                pt_mask  = (results["batch_indices"] == b)
                B3d_gt   = lifted[pt_mask][:, valid_k].float()                   # (N_b, K_valid)
                pt_valid = lifted_valid[pt_mask]                                  # (N_b,)
                if not pt_valid.any():
                    continue

                pred_logits = pred_logits[pt_valid]   # (N_v, K_valid)
                B3d_gt      = B3d_gt[pt_valid]        # (N_v, K_valid)

                # min_points_per_mask 过滤（与训练一致）
                pos_cnt = (B3d_gt > 0.5).float().sum(dim=0)          # (K_valid,)
                keep    = pos_cnt >= self.config.min_points_per_mask  # (K_valid,)
                if not keep.any():
                    continue

                keep_idx  = valid_k.nonzero(as_tuple=True)[0][keep]
                pred_prob = torch.sigmoid(pred_logits[:, keep])       # (N_v, K_keep)
                B3d_gt_k  = B3d_gt[:, keep]                           # (N_v, K_keep)

                # ---- mask mIoU ----
                pred_full = torch.zeros(pred_prob.shape[0], K_max, device=pred_prob.device)
                gt_full   = torch.zeros(B3d_gt_k.shape[0], K_max, device=B3d_gt_k.device)
                pred_full[:, keep_idx] = pred_prob
                gt_full[:, keep_idx]   = B3d_gt_k
                mask_tracker.update(pred_full, gt_full, keep_idx)

            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

        val_metrics = {
            "loss":                      val_loss.avg,
            "loss_mask_distill":         val_distill.avg,
            "loss_aux":                  val_aux.avg,
            "semantic_miou":             0.0,  # final semantic metric used for best_model
            "semantic_miou_hybrid_text": 0.0,
            "semantic_miou_clip_text":   0.0,
            "semantic_miou_final":       0.0,
            "semantic_macc_hybrid_text": 0.0,
            "semantic_macc_clip_text":   0.0,
            "semantic_macc_final":       0.0,
            "n_valid_classes_hybrid":    0,
            "n_valid_classes_clip":      0,
            "n_valid_classes_final":     0,
            "mask_miou":                 0.0,
            "n_masks":                   0,
            "semantic_miou_dual_space_fixed": 0.0,
            "semantic_macc_dual_space_fixed": 0.0,
            "semantic_miou_dual_space_confidence": 0.0,
            "semantic_macc_dual_space_confidence": 0.0,
            "semantic_miou_dual_space_gate": 0.0,
            "semantic_macc_dual_space_gate": 0.0,
            "source_gate_val_mean": 0.0,
            "source_gate_val_std": 0.0,
            "source_gate_val_min": 0.0,
            "source_gate_val_max": 0.0,
            "semantic_miou_odise_only_text256": 0.0,
            "semantic_miou_lseg_only_text512": 0.0,
            "semantic_miou_current_fused_text256": 0.0,
            "semantic_miou_odise_only": 0.0,
            "semantic_miou_lseg_only": 0.0,
            "semantic_miou_fixed_05": 0.0,
            "semantic_miou_lseg_06": 0.0,
            "semantic_miou_lseg_07": 0.0,
            "semantic_miou_lseg_08": 0.0,
            "semantic_miou_size_aware": 0.0,
            "semantic_miou_projected_gate": 0.0,
            "semantic_miou_projected_size_gate": 0.0,
        }

        if pc_tracker is not None:
            pc_res = pc_tracker.compute()
            val_metrics["semantic_miou_hybrid_text"] = pc_res["semantic_miou_hybrid_text"]
            val_metrics["semantic_miou_clip_text"] = pc_res["semantic_miou_clip_text"]
            val_metrics["semantic_miou_final"] = pc_res["semantic_miou_pc"]
            val_metrics["semantic_miou"] = pc_res["semantic_miou_pc"]
            val_metrics["semantic_macc_hybrid_text"] = pc_res["semantic_macc_hybrid_text"]
            val_metrics["semantic_macc_clip_text"] = pc_res["semantic_macc_clip_text"]
            val_metrics["semantic_macc_final"] = pc_res["semantic_macc_pc"]
            val_metrics["n_valid_classes_hybrid"] = pc_res["n_valid_classes_hybrid_text"]
            val_metrics["n_valid_classes_clip"] = pc_res["n_valid_classes_clip_text"]
            val_metrics["n_valid_classes_final"] = pc_res["n_valid_classes_pc"]
            val_metrics["per_class_iou_hybrid_text"] = pc_res.get("per_class_iou_hybrid_text", {})
            val_metrics["per_class_iou_clip_text"] = pc_res.get("per_class_iou_clip_text", {})
            val_metrics["per_class_iou_final"] = pc_res.get("per_class_iou_pc", {})
            val_metrics["per_class_acc_hybrid_text"] = pc_res.get("per_class_acc_hybrid_text", {})
            val_metrics["per_class_acc_clip_text"] = pc_res.get("per_class_acc_clip_text", {})
            val_metrics["per_class_acc_final"] = pc_res.get("per_class_acc_pc", {})

        if odise256_probe_accs is not None:
            for name, acc in odise256_probe_accs.items():
                res = acc.compute(f"semantic_miou_{name}")
                val_metrics[f"semantic_miou_{name}"] = res[f"semantic_miou_{name}"]
                val_metrics[f"semantic_macc_{name}"] = res[f"semantic_macc_{name}"]
                val_metrics[f"n_valid_classes_{name}"] = res[f"n_valid_classes_semantic_miou_{name}"]
                val_metrics[f"per_class_iou_{name}"] = res[f"per_class_iou_semantic_miou_{name}"]
                val_metrics[f"per_class_acc_{name}"] = res[f"per_class_acc_semantic_miou_{name}"]
            if source_gate_val_values:
                gate_cat = torch.cat(source_gate_val_values)
                val_metrics["source_gate_val_mean"] = float(gate_cat.mean().item())
                val_metrics["source_gate_val_std"] = float(gate_cat.std(unbiased=False).item())
                val_metrics["source_gate_val_min"] = float(gate_cat.min().item())
                val_metrics["source_gate_val_max"] = float(gate_cat.max().item())

        if dual_space_accs is not None:
            for name, acc in dual_space_accs.items():
                res = acc.compute(f"semantic_miou_{name}")
                val_metrics[f"semantic_miou_{name}"] = res[f"semantic_miou_{name}"]
                val_metrics[f"semantic_macc_{name}"] = res[f"semantic_macc_{name}"]
                val_metrics[f"n_valid_classes_{name}"] = res[f"n_valid_classes_semantic_miou_{name}"]
                val_metrics[f"per_class_iou_{name}"] = res[f"per_class_iou_semantic_miou_{name}"]
                val_metrics[f"per_class_acc_{name}"] = res[f"per_class_acc_semantic_miou_{name}"]
            readout_mode = str(self.config.semantic_readout_mode).lower()
            metric_key = f"semantic_miou_{readout_mode}"
            metric_acc_key = f"semantic_macc_{readout_mode}"
            if metric_key in val_metrics:
                val_metrics["semantic_miou"] = val_metrics[metric_key]
            if metric_acc_key in val_metrics:
                val_metrics["semantic_macc_final"] = val_metrics[metric_acc_key]
        if dual_space_group_accs is not None:
            for group_name, group_accs in dual_space_group_accs.items():
                for name, acc in group_accs.items():
                    res = acc.compute(f"semantic_miou_{group_name}_{name}")
                    val_metrics[f"semantic_miou_{group_name}_{name}"] = res[f"semantic_miou_{group_name}_{name}"]
                    val_metrics[f"semantic_macc_{group_name}_{name}"] = res[f"semantic_macc_{group_name}_{name}"]
                    val_metrics[f"per_class_iou_{group_name}_{name}"] = res[
                        f"per_class_iou_semantic_miou_{group_name}_{name}"
                    ]
                    val_metrics[f"per_class_acc_{group_name}_{name}"] = res[
                        f"per_class_acc_semantic_miou_{group_name}_{name}"
                    ]

        mask_res = mask_tracker.compute()
        val_metrics["mask_miou"] = mask_res["mask_miou"]
        val_metrics["n_masks"]   = mask_res["n_masks"]

        return val_metrics

    # ----------------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool = False, suffix: str = ""):
        ckpt = {
            "epoch":                epoch,
            "global_step":          self.global_step,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "best_loss":            self.best_loss,
            "best_iou":             self.best_iou,
            "best_monitor":         self.config.best_monitor,
            "config":               self.config,
        }
        if suffix:
            torch.save(ckpt, f"{self.config.checkpoint_dir}/checkpoint_{suffix}.pth")
        if is_best:
            torch.save(ckpt, f"{self.config.checkpoint_dir}/best_model.pth")
            print(f"  -> Saved best model (metric={metric:.4f})")

    def _load_checkpoint(self, path: str):
        print(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=self.device)
        sd   = ckpt["model_state_dict"]
        is_model_ddp = list(self.model.state_dict().keys())[0].startswith("module.")
        is_ckpt_ddp  = list(sd.keys())[0].startswith("module.")
        if is_model_ddp and not is_ckpt_ddp:
            sd = {"module." + k: v for k, v in sd.items()}
        elif not is_model_ddp and is_ckpt_ddp:
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        current_sd = self.model.state_dict()
        mismatched = []
        filtered_sd = {}
        for k, v in sd.items():
            if k in current_sd and tuple(current_sd[k].shape) != tuple(v.shape):
                mismatched.append((k, tuple(v.shape), tuple(current_sd[k].shape)))
                continue
            filtered_sd[k] = v
        sd = filtered_sd
        # strict=False tolerates newly added params (e.g. fuse_embed.alpha)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if mismatched and self.is_main:
            print(f"[resume] skipped shape-mismatched keys: {mismatched}")
        if missing and self.is_main:
            print(f"[resume] missing keys (will use init values): {missing}")
        if unexpected and self.is_main:
            print(f"[resume] unexpected keys (ignored): {unexpected}")
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        semantic_proj_path = getattr(getattr(model_ref, "config", None), "semantic_proj_path", None)
        if semantic_proj_path:
            model_ref._load_semantic_projection(semantic_proj_path)
            if self.is_main:
                print(f"[resume] reloaded semantic projection from {semantic_proj_path}")

        if "optimizer_state_dict" in ckpt and not mismatched:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as exc:
                if self.is_main:
                    print(
                        "[resume] skipped optimizer state because parameter groups changed: "
                        f"{exc}"
                    )
        elif "optimizer_state_dict" in ckpt and mismatched and self.is_main:
            print("[resume] skipped optimizer state because model parameter shapes changed")
        if self.config.override_optimizer_hparams_on_resume:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config.base_lr
                pg["weight_decay"] = self.config.weight_decay

        if self.config.reset_scheduler_on_resume:
            self._build_scheduler()
        elif "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.current_epoch = ckpt["epoch"] + 1
        self.global_step   = ckpt["global_step"]
        self.best_loss     = ckpt.get("best_loss", float("inf"))
        if ckpt.get("best_monitor") == self.config.best_monitor:
            self.best_iou = ckpt.get("best_iou", 0.0)
        else:
            # Older checkpoints may have used a different validation monitor,
            # so reset the baseline for the current run's monitor.
            self.best_iou = 0.0
            if self.is_main:
                print(
                    "[resume] reset best_iou: checkpoint used old monitor, "
                    f"now using {self.config.best_monitor}"
                )
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    @torch.no_grad()
    def evaluate_only(self) -> Dict:
        if self.is_main:
            print("[EvalOnly] Running validation only")
        return self._validate(max(self.current_epoch - 1, 0))

    # ----------------------------------------------------------
    # 主训练入口
    # ----------------------------------------------------------

    def train(self) -> Dict:
        if self.is_main:
            print(f"[MaskDistillTrainer] Starting {self.config.num_epochs} epochs")

        final_epoch = self.current_epoch
        for epoch in range(self.current_epoch, self.config.num_epochs):
            final_epoch = epoch + 1
            t0 = time.time()

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            if self.is_main:
                print(f"\n>> Epoch [{epoch+1}/{self.config.num_epochs}]")

            train_loss = self._train_epoch(epoch)
            if self.writer is not None:
                self.writer.add_scalar("Loss/Train_Epoch", train_loss, epoch)

            if self.is_main:
                print(f"  Train Loss: {train_loss:.4f}  Time: {time.time()-t0:.1f}s")

            val_metrics = None
            if (epoch + 1) % self.config.val_every_epochs == 0:
                val_metrics = self._validate(epoch)
                if val_metrics and self.is_main:
                    if self.writer is not None:
                        self.writer.add_scalar("Loss/Val",                   val_metrics["loss"],                       epoch)
                        self.writer.add_scalar("Loss/Val_Alignment",         val_metrics["loss_mask_distill"],          epoch)
                        self.writer.add_scalar("Loss/Val_MaskDistill",       val_metrics["loss_mask_distill"],          epoch)
                        self.writer.add_scalar("Loss/Val_Aux",               val_metrics["loss_aux"],                   epoch)
                        self.writer.add_scalar("SemanticReadout/odise_only",          val_metrics["semantic_miou_odise_only"],          epoch)
                        self.writer.add_scalar("SemanticReadout/lseg_only",           val_metrics["semantic_miou_lseg_only"],           epoch)
                        self.writer.add_scalar("SemanticReadout/fixed_05",            val_metrics["semantic_miou_fixed_05"],            epoch)
                        self.writer.add_scalar("SemanticReadout/lseg_06",             val_metrics["semantic_miou_lseg_06"],             epoch)
                        self.writer.add_scalar("SemanticReadout/lseg_07",             val_metrics["semantic_miou_lseg_07"],             epoch)
                        self.writer.add_scalar("SemanticReadout/lseg_08",             val_metrics["semantic_miou_lseg_08"],             epoch)
                        self.writer.add_scalar("SemanticReadout/size_aware",          val_metrics["semantic_miou_size_aware"],          epoch)
                        self.writer.add_scalar("SemanticReadout/projected_gate",      val_metrics["semantic_miou_projected_gate"],      epoch)
                        self.writer.add_scalar("SemanticReadout/projected_size_gate", val_metrics["semantic_miou_projected_size_gate"], epoch)
                        self.writer.add_scalar("Metrics/Mask_mIoU",               val_metrics["mask_miou"],                epoch)
                        self.writer.add_scalar("Metrics/N_Masks",                  val_metrics["n_masks"],                  epoch)
                        if self.config.enable_verbose_legacy_probes:
                            self.writer.add_scalar("Metrics/Semantic_mIoU_HybridText", val_metrics["semantic_miou_hybrid_text"], epoch)
                            self.writer.add_scalar("Metrics/Semantic_mIoU_CLIPText",   val_metrics["semantic_miou_clip_text"],   epoch)
                            self.writer.add_scalar("Metrics/Semantic_mIoU_FinalPC",    val_metrics["semantic_miou_final"],       epoch)
                            self.writer.add_scalar("Metrics/Semantic_mAcc_HybridText", val_metrics["semantic_macc_hybrid_text"], epoch)
                            self.writer.add_scalar("Metrics/Semantic_mAcc_CLIPText",   val_metrics["semantic_macc_clip_text"],   epoch)
                            self.writer.add_scalar("Metrics/Semantic_mAcc_FinalPC",    val_metrics["semantic_macc_final"],       epoch)
                            self.writer.add_scalar("Metrics/Semantic_mIoU_DualSpaceFixed",      val_metrics["semantic_miou_dual_space_fixed"],      epoch)
                            self.writer.add_scalar("Metrics/Semantic_mIoU_DualSpaceConfidence", val_metrics["semantic_miou_dual_space_confidence"], epoch)
                            self.writer.add_scalar("Metrics/Semantic_mIoU_DualSpaceGate",       val_metrics["semantic_miou_dual_space_gate"],       epoch)
                            self.writer.add_scalar("Metrics/Semantic_mAcc_DualSpaceGate",       val_metrics["semantic_macc_dual_space_gate"],       epoch)
                            self.writer.add_scalar("Metrics/Semantic_mIoU_ODISEOnlyText256",    val_metrics["semantic_miou_odise_only_text256"],    epoch)
                            self.writer.add_scalar("Metrics/Semantic_mIoU_LSegOnlyText512",     val_metrics["semantic_miou_lseg_only_text512"],     epoch)
                            self.writer.add_scalar("Metrics/Semantic_mIoU_CurrentFusedText256", val_metrics["semantic_miou_current_fused_text256"], epoch)
                            self.writer.add_scalar("SourceGate/val_mean", val_metrics["source_gate_val_mean"], epoch)
                            self.writer.add_scalar("SourceGate/val_std",  val_metrics["source_gate_val_std"],  epoch)
                            self.writer.add_scalar("SourceGate/val_min",  val_metrics["source_gate_val_min"],  epoch)
                            self.writer.add_scalar("SourceGate/val_max",  val_metrics["source_gate_val_max"],  epoch)
                        per_class_groups = {
                            "PerClass_IoU_Readout_LSegOnly": val_metrics.get("per_class_iou_lseg_only", {}),
                            "PerClass_IoU_Readout_ODISEOnly": val_metrics.get("per_class_iou_odise_only", {}),
                            "PerClass_IoU_Readout_LSeg07": val_metrics.get("per_class_iou_lseg_07", {}),
                            "PerClass_IoU_Readout_SizeAware": val_metrics.get("per_class_iou_size_aware", {}),
                            "PerClass_IoU_Readout_ProjectedSizeGate": val_metrics.get("per_class_iou_projected_size_gate", {}),
                            "PerClass_IoU_Readout_ProjectedGate": val_metrics.get("per_class_iou_projected_gate", {}),
                        }
                        for tag_prefix, per_class in per_class_groups.items():
                            for cls_name, iou_val in per_class.items():
                                self.writer.add_scalar(
                                    f"{tag_prefix}/{cls_name}", iou_val, epoch
                                )
                        for group_name in self._semantic_size_group_names():
                            for metric_name in self._semantic_ablation_names():
                                metric_key = f"semantic_miou_{group_name}_{metric_name}"
                                if metric_key in val_metrics:
                                    self.writer.add_scalar(
                                        f"SemanticReadout_{group_name}/{metric_name}",
                                        val_metrics[metric_key],
                                        epoch,
                                    )

                    print(
                        "  [Main Metrics] "
                        f"val_loss={val_metrics['loss']:.4f}  "
                        f"alignment_loss={val_metrics['loss_mask_distill']:.4f}  "
                        f"mask_iou={val_metrics['mask_miou']:.4f}  "
                        f"semantic_miou_projected_gate={val_metrics.get('semantic_miou_projected_gate', 0.0):.4f}  "
                        f"semantic_macc_projected_gate={val_metrics.get('semantic_macc_projected_gate', 0.0):.4f}  "
                        f"semantic_miou_fixed_05={val_metrics.get('semantic_miou_fixed_05', 0.0):.4f}  "
                        f"semantic_miou_lseg_only={val_metrics.get('semantic_miou_lseg_only', 0.0):.4f}  "
                        f"semantic_miou_odise_only={val_metrics.get('semantic_miou_odise_only', 0.0):.4f}"
                    )
                    if self.config.enable_verbose_legacy_probes and "semantic_miou_dual_space_fixed" in val_metrics:
                        print(
                            "  [Dual-Space] "
                            f"odise={val_metrics['semantic_miou_odise_only_text256']:.4f}  "
                            f"lseg={val_metrics['semantic_miou_lseg_only_text512']:.4f}  "
                            f"fused={val_metrics['semantic_miou_current_fused_text256']:.4f}  "
                            f"fixed={val_metrics['semantic_miou_dual_space_fixed']:.4f}  "
                            f"conf={val_metrics['semantic_miou_dual_space_confidence']:.4f}  "
                            f"gate={val_metrics['semantic_miou_dual_space_gate']:.4f}"
                        )
                        if self.config.source_gate_train:
                            print(
                                "  [SourceGate] "
                                f"mean={val_metrics['source_gate_val_mean']:.4f}  "
                                f"std={val_metrics['source_gate_val_std']:.4f}  "
                                f"min={val_metrics['source_gate_val_min']:.4f}  "
                                f"max={val_metrics['source_gate_val_max']:.4f}"
                            )
                    print(
                        "  [Alignment Query Mode] "
                        f"mode={self.config.alignment_query_mode}  "
                        f"val_loss={val_metrics['loss']:.4f}  "
                        f"mask_iou={val_metrics['mask_miou']:.4f}  "
                        f"semantic_miou_projected_gate={val_metrics.get('semantic_miou_projected_gate', 0.0):.4f}  "
                        f"semantic_miou_fixed_05={val_metrics.get('semantic_miou_fixed_05', 0.0):.4f}  "
                        f"semantic_miou_lseg_only={val_metrics.get('semantic_miou_lseg_only', 0.0):.4f}  "
                        f"semantic_miou_odise_only={val_metrics.get('semantic_miou_odise_only', 0.0):.4f}"
                    )
                    if self.config.semantic_readout_ablation:
                        print(
                            "  [Semantic Readout Ablation] "
                            f"odise_only={val_metrics['semantic_miou_odise_only']:.4f}  "
                            f"lseg_only={val_metrics['semantic_miou_lseg_only']:.4f}  "
                            f"fixed_0.5={val_metrics['semantic_miou_fixed_05']:.4f}  "
                            f"lseg_0.6={val_metrics['semantic_miou_lseg_06']:.4f}  "
                            f"lseg_0.7={val_metrics['semantic_miou_lseg_07']:.4f}  "
                            f"lseg_0.8={val_metrics['semantic_miou_lseg_08']:.4f}  "
                            f"size_aware={val_metrics['semantic_miou_size_aware']:.4f}  "
                            f"projected_gate={val_metrics['semantic_miou_projected_gate']:.4f}  "
                            f"projected_size_gate={val_metrics['semantic_miou_projected_size_gate']:.4f}"
                        )
                        print("  [Region Size Ablation]")
                        for group_name in self._semantic_size_group_names():
                            print(
                                f"    {group_name}: "
                                f"odise={val_metrics.get(f'semantic_miou_{group_name}_odise_only', 0.0):.4f}  "
                                f"lseg={val_metrics.get(f'semantic_miou_{group_name}_lseg_only', 0.0):.4f}  "
                                f"projected={val_metrics.get(f'semantic_miou_{group_name}_projected_gate', 0.0):.4f}"
                            )
                    if self.config.enable_verbose_legacy_probes and "semantic_miou_hybrid_odise256" in val_metrics:
                        print(
                            "  [ODISE-256 probes] "
                            f"hybrid={val_metrics['semantic_miou_hybrid_odise256']:.4f}  "
                            f"clip_proj={val_metrics['semantic_miou_clip_odise256']:.4f}  "
                            f"odise={val_metrics['semantic_miou_odise_odise256']:.4f}  "
                            f"base={val_metrics['semantic_miou_base_odise256']:.4f}  "
                            f"refine={val_metrics['semantic_miou_refine_odise256']:.4f}  "
                            f"lseg_semproj={val_metrics.get('semantic_miou_lseg_semproj_odise256', 0.0):.4f}  "
                            f"semantic_query={val_metrics.get('semantic_miou_semantic_query_odise256', 0.0):.4f}"
                        )
                    if self.is_main and self.config.enable_verbose_legacy_probes:
                        for name, key in (
                            ("Hybrid/Text", "per_class_iou_hybrid_text"),
                            ("CLIP/Text", "per_class_iou_clip_text"),
                            ("Final-PC", "per_class_iou_final"),
                            ("Hybrid@ODISE256", "per_class_iou_hybrid_odise256"),
                            ("CLIPProj@ODISE256", "per_class_iou_clip_odise256"),
                            ("ODISE@ODISE256", "per_class_iou_odise_odise256"),
                            ("Base@ODISE256", "per_class_iou_base_odise256"),
                            ("Refine@ODISE256", "per_class_iou_refine_odise256"),
                            ("LSegSemProj@ODISE256", "per_class_iou_lseg_semproj_odise256"),
                            ("SemanticQuery@ODISE256", "per_class_iou_semantic_query_odise256"),
                            ("DualSpaceFixed", "per_class_iou_dual_space_fixed"),
                            ("DualSpaceConfidence", "per_class_iou_dual_space_confidence"),
                            ("DualSpaceGate", "per_class_iou_dual_space_gate"),
                        ):
                            per_cls = val_metrics.get(key, {})
                            if not per_cls:
                                continue
                            top10 = "  ".join(
                                f"{k}:{v:.3f}" for k, v in
                                sorted(per_cls.items(), key=lambda x: -x[1])[:10]
                            )
                            print(f"  Top-10 classes ({name}): {top10}")
                        if self.config.semantic_readout_ablation:
                            print("  [Top-K Per-Class Comparison]")
                            baseline = val_metrics.get("per_class_iou_lseg_only", {})
                            for compare_name, metric_key in (
                                ("ODISEOnly", "per_class_iou_odise_only"),
                                ("LSeg07", "per_class_iou_lseg_07"),
                                ("SizeAware", "per_class_iou_size_aware"),
                                ("ProjectedSizeGate", "per_class_iou_projected_size_gate"),
                            ):
                                per_cls = val_metrics.get(metric_key, {})
                                if not baseline or not per_cls:
                                    continue
                                deltas = [
                                    (cls_name, float(per_cls.get(cls_name, 0.0) - baseline.get(cls_name, 0.0)))
                                    for cls_name in baseline.keys()
                                ]
                                gains = "  ".join(
                                    f"{k}:{v:+.3f}" for k, v in sorted(deltas, key=lambda x: -x[1])[:5]
                                )
                                drops = "  ".join(
                                    f"{k}:{v:+.3f}" for k, v in sorted(deltas, key=lambda x: x[1])[:5]
                                )
                                print(f"    {compare_name} gains: {gains}")
                                print(f"    {compare_name} drops: {drops}")

            is_best = False
            if val_metrics is not None:
                monitor_name = self.config.best_monitor
                monitor_value = val_metrics.get(monitor_name, None)
                if monitor_value is None:
                    monitor_name = "semantic_miou_dual_space_fixed"
                    monitor_value = val_metrics.get(monitor_name, None)
                if monitor_value is None:
                    monitor_name = "semantic_miou_final"
                    monitor_value = val_metrics.get(monitor_name, None)
                if monitor_value is None:
                    monitor_name = "val_loss"
                    monitor_value = -val_metrics["loss"]
                monitored = float(monitor_value)
                if monitored > self.best_iou + self.config.early_stopping_min_delta:
                    prev = self.best_iou
                    self.best_iou = monitored
                    self.epochs_without_improvement = 0
                    is_best = True
                    if self.is_main:
                        print(
                            f"  New best {monitor_name}: "
                            f"{monitored:.4f} (prev: {prev:.4f})"
                        )
                else:
                    self.epochs_without_improvement += 1
                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
                monitored_value = monitored
            else:
                monitored_value = train_loss

            if self.is_main:
                if (epoch + 1) % self.config.save_every_epochs == 0:
                    self._save_checkpoint(epoch, monitored_value, is_best=False, suffix=f"epoch_{epoch+1}")
                if is_best:
                    self._save_checkpoint(epoch, monitored_value, is_best=True)

            if val_metrics is not None and self.config.scheduler_type == "plateau":
                self.scheduler.step(val_metrics["loss"])

            if (
                val_metrics is not None
                and self.epochs_without_improvement >= self.config.early_stopping_patience
            ):
                if self.is_main:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        if self.writer is not None:
            self.writer.close()
        if self.is_main:
            print(f"Done. Best mIoU={self.best_iou:.4f}  Best loss={self.best_loss:.4f}")

        return {
            "best_loss":   self.best_loss,
            "best_iou":    self.best_iou,
            "final_epoch": final_epoch,
        }
