

import contextlib
import json
import math
import os
import subprocess
import sys
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
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
)
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
    validation_log_every_batches: int   = 25
    validation_subprocess:        bool  = False
    validation_config_path:       str   = "config/train_scannet_v2_full_multi_gpu.yaml"
    validation_device:            str   = "cuda"


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

    # ----------------------------------------------------------
    # 训练 epoch
    # ----------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss        = AverageMeter()
        epoch_distill     = AverageMeter()
        epoch_aux         = AverageMeter()
        batch_time        = AverageMeter()

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
                    self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], self.global_step)

                    # Log fusion alpha (ODISE-residual fusion mixing weight)
                    fuse = self.model.fuse_embed if hasattr(self.model, "fuse_embed") \
                        else getattr(self.model, "module", None) and self.model.module.fuse_embed
                    if fuse is not None and hasattr(fuse, "alpha"):
                        self.writer.add_scalar("Fusion/alpha", fuse.alpha.item(), self.global_step)

                self.global_step += 1

            self._adjust_learning_rate_warmup(self.global_step)

            if step % self.config.log_every_steps == 0 and self.is_main:
                lr  = self.optimizer.param_groups[0]["lr"]
                eta = batch_time.avg * (len(self.train_loader) - step)
                print(
                    f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                    f"Step [{step}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(distill={loss_dict['loss_mask_distill']:.4f} aux={loss_dict['loss_aux']:.4f}) "
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
        validate_start = time.time()
        total_val_batches = len(self.val_loader)
        if self.is_main:
            val_batch_size = getattr(self.val_loader, "batch_size", "unknown")
            val_num_workers = getattr(self.val_loader, "num_workers", "unknown")
            val_persistent_workers = getattr(self.val_loader, "persistent_workers", "unknown")
            val_prefetch_factor = getattr(self.val_loader, "prefetch_factor", None)
            val_samples = (
                len(self.val_loader.dataset)
                if getattr(self.val_loader, "dataset", None) is not None
                else "unknown"
            )
            print("  +---------------- Validation ----------------+", flush=True)
            print(f"  | samples            : {val_samples}", flush=True)
            print(f"  | batches            : {total_val_batches}", flush=True)
            print(f"  | batch_size         : {val_batch_size}", flush=True)
            print(f"  | num_workers        : {val_num_workers}", flush=True)
            print(f"  | persistent_workers : {val_persistent_workers}", flush=True)
            print(f"  | prefetch_factor    : {val_prefetch_factor}", flush=True)
            print("  +--------------------------------------------+", flush=True)

        # 语义 mIoU：保留三项，其中 best_model 只监控 Hybrid/Text：
        # 1) Hybrid/Text: fused 256D vs ODISE text256
        # 2) CLIP/Text: raw LSeg/CLIP 512D vs CLIP-B text512
        # 3) Final-PC: geometric fused final result, only for reference
        text_feats = self._get_text_features()
        pixel_text_feats = self._get_pixel_text_features()
        pc_tracker = (
            ODISEPCSemanticMIoUTracker(pc_lambda=self.config.semantic_pc_lambda)
            if text_feats is not None and pixel_text_feats is not None
            else None
        )

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

            if (
                self.is_main
                and self.config.validation_log_every_batches > 0
                and (
                    (batch_idx + 1) % self.config.validation_log_every_batches == 0
                    or (batch_idx + 1) == total_val_batches
                )
            ):
                elapsed = time.time() - validate_start
                done = batch_idx + 1
                seconds_per_batch = elapsed / max(done, 1)
                eta = seconds_per_batch * max(total_val_batches - done, 0)
                pct = done / max(total_val_batches, 1)
                bar_width = 28
                filled = int(bar_width * pct)
                bar = "#" * filled + "-" * (bar_width - filled)
                print(
                    f"  Validation [{bar}] "
                    f"{done}/{total_val_batches} ({pct * 100:5.1f}%) "
                    f"elapsed={elapsed:.1f}s ETA={eta:.1f}s "
                    f"loss={val_loss.avg:.4f}",
                    flush=True,
                )

            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

        val_metrics = {
            "loss":                      val_loss.avg,
            "loss_mask_distill":         val_distill.avg,
            "loss_aux":                  val_aux.avg,
            "semantic_miou":             0.0,  # main semantic metric used for best_model: Hybrid/Text
            "semantic_miou_hybrid_text": 0.0,
            "semantic_miou_clip_text":   0.0,
            "semantic_miou_final":       0.0,
            "semantic_acc":              0.0,
            "semantic_acc_hybrid_text":  0.0,
            "semantic_acc_clip_text":    0.0,
            "semantic_acc_final":        0.0,
            "semantic_mean_acc_hybrid_text": 0.0,
            "semantic_mean_acc_clip_text":   0.0,
            "semantic_mean_acc_final":       0.0,
            "n_valid_classes_hybrid":    0,
            "n_valid_classes_clip":      0,
            "n_valid_classes_final":     0,
            "mask_miou":                 0.0,
            "n_masks":                   0,
        }

        if pc_tracker is not None:
            pc_res = pc_tracker.compute()
            val_metrics["semantic_miou_hybrid_text"] = pc_res["semantic_miou_hybrid_text"]
            val_metrics["semantic_miou_clip_text"] = pc_res["semantic_miou_clip_text"]
            val_metrics["semantic_miou_final"] = pc_res["semantic_miou_pc"]
            val_metrics["semantic_miou"] = pc_res["semantic_miou_hybrid_text"]
            val_metrics["semantic_acc_hybrid_text"] = pc_res["semantic_acc_hybrid_text"]
            val_metrics["semantic_acc_clip_text"] = pc_res["semantic_acc_clip_text"]
            val_metrics["semantic_acc_final"] = pc_res["semantic_acc_pc"]
            val_metrics["semantic_acc"] = pc_res["semantic_acc_hybrid_text"]
            val_metrics["semantic_mean_acc_hybrid_text"] = pc_res["semantic_mean_acc_hybrid_text"]
            val_metrics["semantic_mean_acc_clip_text"] = pc_res["semantic_mean_acc_clip_text"]
            val_metrics["semantic_mean_acc_final"] = pc_res["semantic_mean_acc_pc"]
            val_metrics["n_valid_classes_hybrid"] = pc_res["n_valid_classes_hybrid_text"]
            val_metrics["n_valid_classes_clip"] = pc_res["n_valid_classes_clip_text"]
            val_metrics["n_valid_classes_final"] = pc_res["n_valid_classes_pc"]
            val_metrics["per_class_iou_hybrid_text"] = pc_res.get("per_class_iou_hybrid_text", {})
            val_metrics["per_class_iou_clip_text"] = pc_res.get("per_class_iou_clip_text", {})
            val_metrics["per_class_iou_final"] = pc_res.get("per_class_iou_pc", {})
            val_metrics["per_class_acc_hybrid_text"] = pc_res.get("per_class_acc_hybrid_text", {})
            val_metrics["per_class_acc_clip_text"] = pc_res.get("per_class_acc_clip_text", {})
            val_metrics["per_class_acc_final"] = pc_res.get("per_class_acc_pc", {})

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
            "best_monitor":         "semantic_miou_hybrid_text",
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
        if ckpt.get("best_monitor") == "semantic_miou_hybrid_text":
            self.best_iou = ckpt.get("best_iou", 0.0)
        else:
            # Older checkpoints stored mask_mIoU or Final-PC in best_iou. The
            # current best criterion is pure Hybrid/Text semantic mIoU, so reset
            # the baseline instead of comparing against a different metric.
            self.best_iou = 0.0
            if self.is_main:
                print("[resume] reset best_iou: checkpoint used old monitor, now using semantic_miou_hybrid_text")
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def _validate_subprocess(self, epoch: int, checkpoint_path: str) -> Optional[Dict]:
        repo_root = Path(__file__).resolve().parents[1]
        eval_script = repo_root / "evaluate" / "eval_mask_distill_checkpoint.py"
        metrics_path = (
            Path(self.config.checkpoint_dir)
            / f"eval_metrics_epoch_{epoch + 1}.json"
        )
        cmd = [
            sys.executable,
            str(eval_script),
            "--checkpoint",
            checkpoint_path,
            "--config",
            self.config.validation_config_path,
            "--split",
            "val",
            "--device",
            self.config.validation_device,
            "--metrics-json",
            str(metrics_path),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        py_paths = [
            str(repo_root),
            str(repo_root / "ODISE"),
            str(repo_root / "ODISE" / "third_party" / "Mask2Former"),
        ]
        env["PYTHONPATH"] = os.pathsep.join(py_paths + [env.get("PYTHONPATH", "")])
        env.setdefault("CLIP_CACHE_DIR", str(repo_root / "checkpoints" / "pretrained" / "clip"))
        env.setdefault("TORCH_HOME", str(repo_root / "checkpoints" / "pretrained" / "torch"))

        if self.is_main:
            print("  Running validation in a subprocess:", flush=True)
            print(f"    checkpoint: {checkpoint_path}", flush=True)
            print(f"    metrics_json: {metrics_path}", flush=True)

        result = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if result.returncode != 0:
            if self.is_main:
                print(
                    f"  [ERROR] Validation subprocess failed with code {result.returncode}",
                    flush=True,
                )
            return None
        if not metrics_path.exists():
            if self.is_main:
                print(f"  [ERROR] Validation subprocess did not write {metrics_path}", flush=True)
            return None
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)

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
                epoch_checkpoint_path = (
                    f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
                )
                if self.is_main:
                    self._save_checkpoint(epoch, train_loss, is_best=False, suffix=f"epoch_{epoch+1}")
                    self._save_checkpoint(epoch, train_loss, is_best=False, suffix="before_val")
                    print(
                        f"  Saved checkpoint_epoch_{epoch+1}.pth and "
                        "checkpoint_before_val.pth before validation"
                    )
                try:
                    if self.config.validation_subprocess:
                        val_metrics = self._validate_subprocess(epoch, epoch_checkpoint_path)
                    else:
                        val_metrics = self._validate(epoch)
                except Exception as exc:
                    val_metrics = None
                    if self.is_main:
                        print(f"  [ERROR] Validation failed at epoch {epoch+1}: {exc}", flush=True)
                        traceback.print_exc()
                        print(
                            "  Continue training from the checkpoint saved before validation.",
                            flush=True,
                        )
                if val_metrics and self.is_main:
                    if self.writer is not None:
                        self.writer.add_scalar("Loss/Val",                   val_metrics["loss"],                       epoch)
                        self.writer.add_scalar("Loss/Val_MaskDistill",       val_metrics["loss_mask_distill"],          epoch)
                        self.writer.add_scalar("Loss/Val_Aux",               val_metrics["loss_aux"],                   epoch)
                        self.writer.add_scalar("Metrics/Semantic_mIoU_HybridText", val_metrics["semantic_miou_hybrid_text"], epoch)
                        self.writer.add_scalar("Metrics/Semantic_mIoU_CLIPText",   val_metrics["semantic_miou_clip_text"],   epoch)
                        self.writer.add_scalar("Metrics/Semantic_mIoU_FinalPC",    val_metrics["semantic_miou_final"],       epoch)
                        self.writer.add_scalar("Metrics/Semantic_Acc_HybridText",  val_metrics["semantic_acc_hybrid_text"],  epoch)
                        self.writer.add_scalar("Metrics/Semantic_Acc_CLIPText",    val_metrics["semantic_acc_clip_text"],    epoch)
                        self.writer.add_scalar("Metrics/Semantic_Acc_FinalPC",     val_metrics["semantic_acc_final"],        epoch)
                        self.writer.add_scalar("Metrics/Semantic_MeanAcc_HybridText", val_metrics["semantic_mean_acc_hybrid_text"], epoch)
                        self.writer.add_scalar("Metrics/Semantic_MeanAcc_CLIPText",   val_metrics["semantic_mean_acc_clip_text"],   epoch)
                        self.writer.add_scalar("Metrics/Semantic_MeanAcc_FinalPC",    val_metrics["semantic_mean_acc_final"],       epoch)
                        self.writer.add_scalar("Metrics/N_Valid_Classes_Hybrid",   val_metrics["n_valid_classes_hybrid"],   epoch)
                        self.writer.add_scalar("Metrics/N_Valid_Classes_CLIP",     val_metrics["n_valid_classes_clip"],     epoch)
                        self.writer.add_scalar("Metrics/N_Valid_Classes_Final",    val_metrics["n_valid_classes_final"],    epoch)
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
                        }
                        for tag_prefix, per_class in per_class_groups.items():
                            for cls_name, iou_val in per_class.items():
                                self.writer.add_scalar(
                                    f"{tag_prefix}/{cls_name}", iou_val, epoch
                                )

                    sem_miou_h = val_metrics["semantic_miou_hybrid_text"]
                    sem_miou_c = val_metrics["semantic_miou_clip_text"]
                    sem_miou_final = val_metrics["semantic_miou_final"]
                    sem_acc_h = val_metrics["semantic_acc_hybrid_text"]
                    sem_acc_c = val_metrics["semantic_acc_clip_text"]
                    sem_acc_final = val_metrics["semantic_acc_final"]
                    mask_miou  = val_metrics["mask_miou"]
                    n_cls      = val_metrics["n_valid_classes_final"]
                    print(
                        f"  Val Loss: {val_metrics['loss']:.4f} "
                        f"(distill={val_metrics['loss_mask_distill']:.4f})  "
                        f"[Hybrid/Text] {sem_miou_h:.4f}  "
                        f"[CLIP/Text] {sem_miou_c:.4f}  "
                        f"[Final-PC] {sem_miou_final:.4f} ({n_cls} classes)  "
                        f"[Acc H/C/F] {sem_acc_h:.4f}/{sem_acc_c:.4f}/{sem_acc_final:.4f}  "
                        f"[MaskIoU] {mask_miou:.4f} ({val_metrics['n_masks']} masks)"
                    )
                    if self.is_main:
                        for name, key in (
                            ("Hybrid/Text", "per_class_iou_hybrid_text"),
                            ("CLIP/Text", "per_class_iou_clip_text"),
                            ("Final-PC", "per_class_iou_final"),
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
                monitored = val_metrics.get("semantic_miou_hybrid_text", 0.0)
                if monitored > self.best_iou + self.config.early_stopping_min_delta:
                    prev = self.best_iou
                    self.best_iou = monitored
                    self.epochs_without_improvement = 0
                    is_best = True
                    if self.is_main:
                        print(f"  New best Hybrid/Text semantic mIoU: {monitored:.4f} (prev: {prev:.4f})")
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
