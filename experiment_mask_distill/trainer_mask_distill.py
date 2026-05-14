

import contextlib
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from MinkowskiEngine import SparseTensor

from experiment_mask_distill.criterion_mask_distill import MaskDistillCriteria
from experiment_mask_distill.semantic_miou import (
    MaskMIoUTracker, ODISEPCSemanticMIoUTracker,
    build_text_features,
    diff2scene_class_probs_predict,
    diff2scene_mask_feature_predict,
    diff2scene_point_class_probs,
    mask_feature_class_probs,
)
from evaluate.semantic_iou import _SemanticAccumulator
from model.source_reliability_gate import build_source_gate_evidence
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
    best_monitor:                str   = "semantic_miou_dual_space_fixed"
    # Source-aware Semantic MoE
    source_gate_train: bool = False
    source_gate_loss_weight: float = 0.03
    source_gate_start_epoch: int = 3
    source_gate_detach_teacher_probs: bool = True
    source_gate_detach_pred_logits: bool = False
    source_gate_balance_reg: float = 0.0
    source_gate_entropy_reg: float = 0.0
    source_gate_monitor: str = "semantic_miou_dual_space_gate"


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
        self._warned_pixel_text_dim_mismatch = False

        if self.config.resume_checkpoint:
            self._load_checkpoint(self.config.resume_checkpoint)

        if self.is_main:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            print(f"[MaskDistillTrainer] Parameters: {trainable:,} trainable / {total:,} total")
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
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        source_gate = getattr(model_ref, "source_gate", None)
        empty_logs = {
            "loss_source_gate": 0.0,
            "source_gate_mean": 0.0,
            "source_gate_std": 0.0,
            "source_gate_min": 0.0,
            "source_gate_max": 0.0,
        }
        if source_gate is None or text_feats is None or pixel_text_feats is None:
            return None, empty_logs

        lseg_all = results.get("pixel_pooled_embeddings", batch.get("pixel_pooled"))
        if lseg_all is None:
            return None, empty_logs

        loss_gate_total = None
        num_gate_items = 0
        gate_values_for_log = []
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
            pred_logits_for_gate = (
                pred_logits.detach()
                if self.config.source_gate_detach_pred_logits
                else pred_logits
            )
            odise_q = batch["mask_embeddings"][b][valid_k].float()
            lseg_q = lseg_all[b][valid_k].float()
            if odise_q.shape[-1] != text_feats.shape[-1] or lseg_q.shape[-1] != pixel_text_feats.shape[-1]:
                continue

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
            evidence = build_source_gate_evidence(
                p_odise_gate,
                p_lseg_gate,
                point_mask_conf=point_mask_conf,
            )
            gate = source_gate(evidence)
            p_gate = (1.0 - gate) * p_odise_gate + gate * p_lseg_gate
            point_probs = diff2scene_point_class_probs(
                pred_logits_for_gate,
                p_gate,
            )
            loss_gate = F.nll_loss(
                torch.log(point_probs.clamp_min(1e-8)),
                gt_b,
                ignore_index=255,
            )
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
            "source_gate_mean": float(gate_cat.mean().detach().cpu()),
            "source_gate_std": float(gate_cat.std(unbiased=False).detach().cpu()),
            "source_gate_min": float(gate_cat.min().detach().cpu()),
            "source_gate_max": float(gate_cat.max().detach().cpu()),
        }
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
        return loss_extra, logs

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
        gate_train_enabled = (
            self.config.source_gate_train
            and epoch >= self.config.source_gate_start_epoch
        )
        gate_text_feats = self._get_text_features() if gate_train_enabled else None
        gate_pixel_text_feats = self._get_pixel_text_features() if gate_train_enabled else None

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
                                {
                                    "loss_source_gate": 0.0,
                                    "source_gate_mean": 0.0,
                                    "source_gate_std": 0.0,
                                    "source_gate_min": 0.0,
                                    "source_gate_max": 0.0,
                                },
                            )
                        )
                        if gate_extra is not None:
                            loss = loss + gate_extra
                        loss_dict.update(gate_logs)
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
                    self.writer.add_scalar("Loss/Train_MaskDistill",  loss_dict["loss_mask_distill"],     self.global_step)
                    self.writer.add_scalar("Loss/Train_Aux",          loss_dict["loss_aux"],              self.global_step)
                    self.writer.add_scalar("Loss/Train_SourceGate",   loss_dict.get("loss_source_gate", 0.0), self.global_step)
                    self.writer.add_scalar("SourceGate/train_mean",   loss_dict.get("source_gate_mean", 0.0), self.global_step)
                    self.writer.add_scalar("SourceGate/train_std",    loss_dict.get("source_gate_std", 0.0), self.global_step)
                    self.writer.add_scalar("SourceGate/train_min",    loss_dict.get("source_gate_min", 0.0), self.global_step)
                    self.writer.add_scalar("SourceGate/train_max",    loss_dict.get("source_gate_max", 0.0), self.global_step)
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
                    f"(distill={loss_dict['loss_mask_distill']:.4f} aux={loss_dict['loss_aux']:.4f} "
                    f"source_gate={loss_dict.get('loss_source_gate', 0.0):.4f}) "
                    f"gate_mean={loss_dict.get('source_gate_mean', 0.0):.4f} "
                    f"gate_std={loss_dict.get('source_gate_std', 0.0):.4f} "
                    f"avg={epoch_loss.avg:.4f}  LR: {lr:.2e}  ETA: {eta:.0f}s"
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

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict:
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss      = AverageMeter()
        val_distill   = AverageMeter()
        val_aux       = AverageMeter()

        # 语义 mIoU：只保留三项：
        # 1) Hybrid/Text: fused 256D vs ODISE text256
        # 2) CLIP/Text: raw LSeg/CLIP 512D vs CLIP-B text512
        # 3) Final-PC: geometric fused final result
        text_feats = self._get_text_features()
        pixel_text_feats = self._get_pixel_text_features()
        pc_tracker = (
            ODISEPCSemanticMIoUTracker(pc_lambda=self.config.semantic_pc_lambda)
            if text_feats is not None and pixel_text_feats is not None
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
            if text_feats is not None
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
            }
            if self.config.dual_space_eval and text_feats is not None and pixel_text_feats is not None
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
                lseg_all = results.get("pixel_pooled_embeddings", batch.get("pixel_pooled"))
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
                    model_ref = self.model.module if hasattr(self.model, "module") else self.model
                    source_gate = getattr(model_ref, "source_gate", None)
                    if source_gate is not None:
                        point_mask_conf = torch.sigmoid(pred_logits).mean(dim=0).detach()
                        evidence = build_source_gate_evidence(
                            p_odise,
                            p_lseg,
                            point_mask_conf=point_mask_conf,
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
        # strict=False tolerates newly added params (e.g. fuse_embed.alpha)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
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

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
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
                        self.writer.add_scalar("Loss/Val_MaskDistill",       val_metrics["loss_mask_distill"],          epoch)
                        self.writer.add_scalar("Loss/Val_Aux",               val_metrics["loss_aux"],                   epoch)
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
                        self.writer.add_scalar("Metrics/N_Valid_Classes_Hybrid",   val_metrics["n_valid_classes_hybrid"],   epoch)
                        self.writer.add_scalar("Metrics/N_Valid_Classes_CLIP",     val_metrics["n_valid_classes_clip"],     epoch)
                        self.writer.add_scalar("Metrics/N_Valid_Classes_Final",    val_metrics["n_valid_classes_final"],    epoch)
                        for tag_name, metric_key in (
                            ("Hybrid", "semantic_miou_hybrid_odise256"),
                            ("CLIPProj", "semantic_miou_clip_odise256"),
                            ("ODISE", "semantic_miou_odise_odise256"),
                            ("Base", "semantic_miou_base_odise256"),
                            ("Refine", "semantic_miou_refine_odise256"),
                            ("LSegSemProj", "semantic_miou_lseg_semproj_odise256"),
                            ("SemanticQuery", "semantic_miou_semantic_query_odise256"),
                        ):
                            if metric_key in val_metrics:
                                self.writer.add_scalar(
                                    f"Metrics_ODISE256/{tag_name}",
                                    val_metrics[metric_key],
                                    epoch,
                                )
                        self.writer.add_scalar("Metrics/Mask_mIoU",               val_metrics["mask_miou"],                epoch)
                        self.writer.add_scalar("Metrics/N_Masks",                  val_metrics["n_masks"],                  epoch)
                        # 每类 IoU 写入 TensorBoard
                        per_class_groups = {
                            "PerClass_IoU_HybridText": val_metrics.get("per_class_iou_hybrid_text", {}),
                            "PerClass_IoU_CLIPText": val_metrics.get("per_class_iou_clip_text", {}),
                            "PerClass_IoU_FinalPC": val_metrics.get("per_class_iou_final", {}),
                            "PerClass_Acc_HybridText": val_metrics.get("per_class_acc_hybrid_text", {}),
                            "PerClass_Acc_CLIPText": val_metrics.get("per_class_acc_clip_text", {}),
                            "PerClass_Acc_FinalPC": val_metrics.get("per_class_acc_final", {}),
                            "PerClass_IoU_DualSpaceFixed": val_metrics.get("per_class_iou_dual_space_fixed", {}),
                            "PerClass_IoU_DualSpaceConfidence": val_metrics.get("per_class_iou_dual_space_confidence", {}),
                            "PerClass_IoU_DualSpaceGate": val_metrics.get("per_class_iou_dual_space_gate", {}),
                        }
                        for tag_prefix, per_class in per_class_groups.items():
                            for cls_name, iou_val in per_class.items():
                                self.writer.add_scalar(
                                    f"{tag_prefix}/{cls_name}", iou_val, epoch
                                )

                    sem_miou_h = val_metrics["semantic_miou_hybrid_text"]
                    sem_miou_c = val_metrics["semantic_miou_clip_text"]
                    sem_miou_final = val_metrics["semantic_miou_final"]
                    sem_macc_h = val_metrics["semantic_macc_hybrid_text"]
                    sem_macc_c = val_metrics["semantic_macc_clip_text"]
                    sem_macc_final = val_metrics["semantic_macc_final"]
                    mask_miou  = val_metrics["mask_miou"]
                    n_cls      = val_metrics["n_valid_classes_final"]
                    print(
                        f"  Val Loss: {val_metrics['loss']:.4f} "
                        f"(distill={val_metrics['loss_mask_distill']:.4f})  "
                        f"[Hybrid/Text] mIoU={sem_miou_h:.4f} mAcc={sem_macc_h:.4f}  "
                        f"[CLIP/Text] mIoU={sem_miou_c:.4f} mAcc={sem_macc_c:.4f}  "
                        f"[Final-PC] mIoU={sem_miou_final:.4f} mAcc={sem_macc_final:.4f} ({n_cls} classes)  "
                        f"[MaskIoU] {mask_miou:.4f} ({val_metrics['n_masks']} masks)"
                    )
                    if "semantic_miou_dual_space_fixed" in val_metrics:
                        print(
                            "  [Dual-Space] "
                            f"odise={val_metrics['semantic_miou_odise_only_text256']:.4f}  "
                            f"lseg={val_metrics['semantic_miou_lseg_only_text512']:.4f}  "
                            f"fused={val_metrics['semantic_miou_current_fused_text256']:.4f}  "
                            f"fixed={val_metrics['semantic_miou_dual_space_fixed']:.4f}  "
                            f"conf={val_metrics['semantic_miou_dual_space_confidence']:.4f}  "
                            f"gate={val_metrics['semantic_miou_dual_space_gate']:.4f}"
                        )
                        print(
                            "  [SourceGate] "
                            f"mean={val_metrics['source_gate_val_mean']:.4f}  "
                            f"std={val_metrics['source_gate_val_std']:.4f}  "
                            f"min={val_metrics['source_gate_val_min']:.4f}  "
                            f"max={val_metrics['source_gate_val_max']:.4f}"
                        )
                    if "semantic_miou_hybrid_odise256" in val_metrics:
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
                    if self.is_main:
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
