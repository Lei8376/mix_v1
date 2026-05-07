"""
新版 Trainer：使用 DistillCriteria 替换旧的 Criteria。

相比 open_vocab_trainer_v2.py 的改动：
  1. _train_epoch: loss 改为 DistillCriteria.compute_loss()，同时记录 loss_feat / loss_mask
  2. _validate:    loss 同样改为 DistillCriteria.compute_loss()
  3. TrainerConfig 增加 feat_loss_weight / mask_loss_weight 两个超参
  4. 其余逻辑（AMP、梯度累积、DDP、checkpoint、scheduler）完全不变

使用方式：
  在 train_open_vocab_v2_ddp.py 里把
    from trainer.open_vocab_trainer_v2 import OpenVocabTrainerV2, OpenVocabTrainerV2Config
  改为
    from experiment_distill.trainer_distill import DistillTrainer, DistillTrainerConfig
  并把 OpenVocabTrainerV2Config → DistillTrainerConfig，OpenVocabTrainerV2 → DistillTrainer。
"""

import contextlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from MinkowskiEngine import SparseTensor

# 新版 criterion
from experiment_distill.criterion_distill import DistillCriteria
# 语义 mIoU
from experiment_distill.semantic_miou import SemanticMIoUTracker, build_text_features
# 旧版 util，直接复用
from utils.util import AverageMeter

# MetricsTracker 直接从旧 trainer 里复用（不做修改）
from trainer.open_vocab_trainer_v2 import MetricsTracker


# ============================================================
# Config
# ============================================================

@dataclass
class DistillTrainerConfig:
    """新版 trainer 配置。新增 feat_loss_weight / mask_loss_weight。"""
    num_epochs:                 int   = 50
    base_lr:                    float = 1e-4
    weight_decay:               float = 1e-4
    grad_clip_norm:             float = 1.0
    log_dir:                    str   = "runs/distill"
    checkpoint_dir:             str   = "checkpoints/distill"
    log_every_steps:            int   = 20
    val_every_epochs:           int   = 1
    save_every_epochs:          int   = 5
    # Scheduler
    warmup_epochs:              int   = 2
    scheduler_type:             str   = "cosine"
    scheduler_t0:               int   = 1
    scheduler_t_mult:           int   = 2
    scheduler_eta_min:          float = 1e-6
    # AMP
    use_amp:                    bool  = True
    # Early stopping
    early_stopping_patience:    int   = 10
    early_stopping_min_delta:   float = 1e-4
    # ---- 新版 loss 权重 ----
    feat_loss_weight:           float = 1.0   # L_feat 主损失
    mask_loss_weight:           float = 0.1   # L_mask 辅助项（原 BCE+Dice 降权）
    bce_weight:                 float = 1.0   # L_mask 内部 BCE 权重
    dice_weight:                float = 1.0   # L_mask 内部 Dice 权重
    # GT 过滤阈值
    min_points_per_mask:        int   = 10
    # Resume
    resume_checkpoint:          Optional[str] = None
    override_optimizer_hparams_on_resume: bool = True
    reset_scheduler_on_resume:  bool  = True
    # Quick test
    max_batches_per_epoch:      Optional[int] = None
    use_model_half:             bool  = False
    gradient_accumulation_steps: int  = 1
    semantic_clip_model:         str   = "ViT-L/14@336px"
    semantic_prompt_template:    str   = "a {} in a scene"


# ============================================================
# Trainer
# ============================================================

class DistillTrainer:
    """
    基于 point-level hybrid teacher distillation 的训练器。
    除 loss 部分外，其余逻辑与 OpenVocabTrainerV2 完全一致。
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        config: DistillTrainerConfig = None,
        device: str = "cuda",
        rank: int = 0,
        train_sampler=None,
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.config         = config or DistillTrainerConfig()
        self.device         = device
        self.rank           = rank
        self.train_sampler  = train_sampler
        self.is_main_process = (rank == 0)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.base_lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler & warmup
        self.steps_per_epoch = len(train_loader)
        self.accum_steps     = getattr(self.config, "gradient_accumulation_steps", 1)
        self.optim_steps_per_epoch = math.ceil(self.steps_per_epoch / self.accum_steps)
        self.total_steps     = self.optim_steps_per_epoch * self.config.num_epochs
        self.warmup_steps    = self.optim_steps_per_epoch * self.config.warmup_epochs
        self._build_scheduler()

        # AMP
        self.scaler = GradScaler(enabled=self.config.use_amp)

        # Tracking
        self.global_step             = 0
        self.current_epoch           = 0
        self.best_loss               = float("inf")
        self.best_iou                = 0.0
        self.epochs_without_improvement = 0

        # TensorBoard
        self.writer = None
        if self.is_main_process:
            os.makedirs(self.config.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.config.log_dir)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # 语义 mIoU：在第一次验证时懒加载文本特征（避免启动时加载 CLIP 拖慢速度）
        self._text_features: Optional[torch.Tensor] = None

        # Resume
        if self.config.resume_checkpoint:
            self._load_checkpoint(self.config.resume_checkpoint)

        if self.is_main_process:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            print(f"[DistillTrainer] Parameters: {trainable:,} trainable / {total:,} total")
            print(f"  feat_loss_weight={self.config.feat_loss_weight}  "
                  f"mask_loss_weight={self.config.mask_loss_weight}")

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

    def _get_warmup_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.config.base_lr * (step + 1) / self.warmup_steps
        return self.config.base_lr

    def _adjust_learning_rate_warmup(self, step: int):
        if step < self.warmup_steps:
            lr = self._get_warmup_lr(step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        tensor_keys = [
            "coords_3d", "feat_3d", "ori_coords_3d",
            "binary_label_3d", "binary_label_2d", "label_2d",
            "img", "x_label", "y_label", "inds_reconstruct",
        ]
        precomputed_keys = [
            "pixel_embeddings", "pixel_pooled", "masks", "mask_embeddings", "mask_valid",
        ]
        moved = dict(batch)
        for key in tensor_keys + precomputed_keys:
            if key in moved and isinstance(moved[key], torch.Tensor):
                moved[key] = moved[key].to(self.device, non_blocking=True)
        return moved

    def _build_sparse_tensor(self, batch: Dict[str, Any]) -> SparseTensor:
        return SparseTensor(batch["feat_3d"], batch["coords_3d"].int())

    def _make_criteria(self, results, batch) -> DistillCriteria:
        """构造 DistillCriteria 实例。"""
        return DistillCriteria(
            results=results,
            batch_input=batch,
            feat_loss_weight=self.config.feat_loss_weight,
            mask_loss_weight=self.config.mask_loss_weight,
            bce_weight=self.config.bce_weight,
            dice_weight=self.config.dice_weight,
            min_points_per_mask=self.config.min_points_per_mask,
        )

    # ----------------------------------------------------------
    # 训练 epoch
    # ----------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss      = AverageMeter()
        epoch_feat_loss = AverageMeter()
        epoch_mask_loss = AverageMeter()
        batch_time      = AverageMeter()

        accum_steps   = self.accum_steps
        is_distributed = hasattr(self.model, "no_sync")

        self.optimizer.zero_grad(set_to_none=True)
        end_time = time.time()
        micro_steps_done = 0
        max_steps = self.config.max_batches_per_epoch

        for step, batch in enumerate(self.train_loader):
            if max_steps is not None and step >= max_steps:
                break
            micro_steps_done += 1

            batch = self._move_batch_to_device(batch)
            batch["sinput"] = self._build_sparse_tensor(batch)

            is_accum_step  = ((step + 1) % accum_steps != 0)
            sync_context   = (
                self.model.no_sync()
                if (is_accum_step and is_distributed)
                else contextlib.nullcontext()
            )

            with sync_context:
                with autocast(enabled=self.config.use_amp):
                    try:
                        results  = self.model(batch)
                        criteria = self._make_criteria(results, batch)
                        loss, loss_dict = criteria.compute_loss()

                        if torch.isnan(loss) or torch.isinf(loss):
                            raise ValueError(f"Invalid loss: {loss}")
                    except Exception as e:
                        print(f"\n❌ Forward error: {e}")
                        raise

                scaled_loss = loss / accum_steps
                self.scaler.scale(scaled_loss).backward()

            epoch_loss.update(loss.item())
            epoch_feat_loss.update(loss_dict["loss_feat"])
            epoch_mask_loss.update(loss_dict["loss_mask"])

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # optimizer step（累积边界）
            if (step + 1) % accum_steps == 0:
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

                if self.writer is not None:
                    self.writer.add_scalar("Loss/Train_Step",      loss.item(),                  self.global_step)
                    self.writer.add_scalar("Loss/Train_Feat_Step",  loss_dict["loss_feat"],       self.global_step)
                    self.writer.add_scalar("Loss/Train_Mask_Step",  loss_dict["loss_mask"],       self.global_step)
                    self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], self.global_step)

                    # Log fusion alpha (ODISE-residual fusion mixing weight)
                    fuse = self.model.fuse_embed if hasattr(self.model, "fuse_embed") \
                        else getattr(self.model, "module", None) and self.model.module.fuse_embed
                    if fuse is not None and hasattr(fuse, "alpha"):
                        self.writer.add_scalar("Fusion/alpha", fuse.alpha.item(), self.global_step)

                self.global_step += 1

            self._adjust_learning_rate_warmup(self.global_step)

            if step % self.config.log_every_steps == 0 and self.is_main_process:
                lr  = self.optimizer.param_groups[0]["lr"]
                eta = batch_time.avg * (len(self.train_loader) - step)
                print(
                    f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                    f"Step [{step}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(feat={loss_dict['loss_feat']:.4f} mask={loss_dict['loss_mask']:.4f}) "
                    f"avg={epoch_loss.avg:.4f} "
                    f"LR: {lr:.2e}  ETA: {eta:.0f}s"
                )

        # 处理 epoch 末尾剩余 micro-step
        remaining = micro_steps_done % accum_steps
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
            self.writer.add_scalar("Loss/Train_Feat_Epoch", epoch_feat_loss.avg, epoch)
            self.writer.add_scalar("Loss/Train_Mask_Epoch", epoch_mask_loss.avg, epoch)

        return epoch_loss.avg

    # ----------------------------------------------------------
    # 验证 epoch
    # ----------------------------------------------------------

    def _get_text_features(self) -> torch.Tensor:
        """懒加载 CLIP 文本特征（只在第一次验证时构建）。"""
        if self._text_features is None:
            if self.is_main_process:
                print("[DistillTrainer] Building CLIP text features for semantic mIoU ...")
            try:
                self._text_features = build_text_features(
                    device=self.device,
                    clip_model=self.config.semantic_clip_model,
                    prompt_template=self.config.semantic_prompt_template,
                )
                if self.is_main_process:
                    print(f"  text_features shape: {self._text_features.shape}")
            except Exception as e:
                if self.is_main_process:
                    print(f"  [WARNING] Failed to build text features: {e}")
                    print("  Semantic mIoU will be skipped.")
                self._text_features = None
        return self._text_features

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss      = AverageMeter()
        val_feat_loss = AverageMeter()
        val_mask_loss = AverageMeter()
        # binary mask IoU（保留旧指标，用于和原版对比）
        binary_metrics = MetricsTracker()
        # 语义 mIoU
        text_features = self._get_text_features()
        sem_tracker   = SemanticMIoUTracker(text_features) if text_features is not None else None

        for batch_idx, batch in enumerate(self.val_loader):
            batch = self._move_batch_to_device(batch)
            batch["sinput"] = self._build_sparse_tensor(batch)

            with autocast(enabled=self.config.use_amp):
                results  = self.model(batch)
                criteria = self._make_criteria(results, batch)
                loss, loss_dict = criteria.compute_loss()

            val_loss.update(loss.item())
            val_feat_loss.update(loss_dict["loss_feat"])
            val_mask_loss.update(loss_dict["loss_mask"])

            # ---- 语义 mIoU：直接用 pred_3d 和 GT 3D 标签 ----
            if sem_tracker is not None:
                pred_3d    = results["pred_3d"]                  # (N_total, 768)
                gt_labels  = batch["binary_label_3d"]            # (N_total,) nyu40 id
                sem_tracker.update(pred_3d, gt_labels)

            # ---- binary mask IoU（保留旧指标） ----
            for b in range(len(results["outputs"])):
                if len(results["outputs"][b]) == 0:
                    continue
                pred_logits = results["outputs"][b][0]["pred_mask_logits"]
                valid       = results["mask_valid_from_masks"][b]

                mask_2d    = results["mask_masks"][b][valid]
                point_mask = results["batch_indices"] == b
                x_idx      = batch["x_label"][point_mask].float()
                y_idx      = batch["y_label"][point_mask].float()

                if x_idx.numel() == 0:
                    continue

                H, W = mask_2d.shape[1], mask_2d.shape[2]
                x_max = x_idx.max().item()
                y_max = y_idx.max().item()
                if (x_max > W + 20) or (y_max > H + 20):
                    x_idx = (x_idx * W / max(640, x_max + 10)).long()
                    y_idx = (y_idx * H / max(480, y_max + 10)).long()
                else:
                    x_idx = x_idx.long()
                    y_idx = y_idx.long()

                inbounds = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
                if not inbounds.any():
                    continue

                x_idx = x_idx[inbounds]
                y_idx = y_idx[inbounds]
                pred_filtered = pred_logits[inbounds]

                gt_3d      = mask_2d[:, y_idx, x_idx]
                gt_3d      = (gt_3d > 0.5).float().t()           # (N_valid, K_valid)
                pred_valid = pred_filtered[:, valid]               # (N_valid, K_valid)

                gt_pos  = gt_3d.sum(dim=0)
                keep_gt = gt_pos >= self.config.min_points_per_mask
                if keep_gt.any():
                    binary_metrics.update(
                        torch.sigmoid(pred_valid[:, keep_gt]).float(),
                        gt_3d[:, keep_gt],
                    )

            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

        # 汇总
        binary_result = binary_metrics.compute()
        val_metrics = {
            "loss":           val_loss.avg,
            "loss_feat":      val_feat_loss.avg,
            "loss_mask":      val_mask_loss.avg,
            # 语义 mIoU（主要指标）
            "semantic_miou":  0.0,
            "n_valid_classes": 0,
            # binary mask IoU（旧指标，保留用于对比）
            "iou":            binary_result["iou"],
            "miou":           binary_result["miou"],
            "accuracy":       binary_result["accuracy"],
            "macc":           binary_result["macc"],
        }
        if sem_tracker is not None:
            sem_result = sem_tracker.compute()
            val_metrics["semantic_miou"]   = sem_result["semantic_miou"]
            val_metrics["n_valid_classes"] = sem_result["n_valid_classes"]
            val_metrics["per_class_iou"]   = sem_result["per_class_iou"]

        return val_metrics

    # ----------------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool = False, suffix: str = ""):
        checkpoint = {
            "epoch":                epoch,
            "global_step":          self.global_step,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "best_loss":            self.best_loss,
            "best_iou":             self.best_iou,
            "config":               self.config,
        }
        if suffix:
            torch.save(checkpoint, f"{self.config.checkpoint_dir}/checkpoint_{suffix}.pth")
        if is_best:
            best_path = f"{self.config.checkpoint_dir}/best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  -> Saved best model (metric={metric:.4f})")

    def _load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        model_keys = list(self.model.state_dict().keys())
        ckpt_keys  = list(state_dict.keys())
        is_model_ddp = model_keys[0].startswith("module.")
        is_ckpt_ddp  = ckpt_keys[0].startswith("module.")
        if is_model_ddp and not is_ckpt_ddp:
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif not is_model_ddp and is_ckpt_ddp:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # strict=False tolerates newly added params (e.g. fuse_embed.alpha)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[resume] missing keys (will use init values): {missing}")
        if unexpected:
            print(f"[resume] unexpected keys (ignored): {unexpected}")

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.config.override_optimizer_hparams_on_resume:
            for pg in self.optimizer.param_groups:
                pg["lr"]           = self.config.base_lr
                pg["weight_decay"] = self.config.weight_decay

        if self.config.scheduler_type == "plateau":
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            if self.config.reset_scheduler_on_resume:
                self._build_scheduler()
            elif "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step   = checkpoint["global_step"]
        self.best_loss     = checkpoint.get("best_loss", float("inf"))
        self.best_iou      = checkpoint.get("best_iou",  0.0)
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    # ----------------------------------------------------------
    # 主训练入口
    # ----------------------------------------------------------

    def train(self) -> Dict[str, float]:
        if self.is_main_process:
            print(f"[DistillTrainer] Starting {self.config.num_epochs} epochs")

        final_epoch = self.current_epoch
        for epoch in range(self.current_epoch, self.config.num_epochs):
            final_epoch = epoch + 1
            epoch_start = time.time()

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            if self.is_main_process:
                print(f"\n>> Epoch [{epoch+1}/{self.config.num_epochs}]")

            train_loss = self._train_epoch(epoch)
            if self.writer is not None:
                self.writer.add_scalar("Loss/Train_Epoch", train_loss, epoch)

            epoch_time = time.time() - epoch_start
            if self.is_main_process:
                print(f"  Train Loss: {train_loss:.4f}  Time: {epoch_time:.1f}s")

            # Validation
            val_metrics = None
            if (epoch + 1) % self.config.val_every_epochs == 0:
                val_metrics = self._validate(epoch)
                if val_metrics and self.is_main_process:
                    if self.writer is not None:
                        self.writer.add_scalar("Loss/Val",                   val_metrics["loss"],                        epoch)
                        self.writer.add_scalar("Loss/Val_Feat",              val_metrics.get("loss_feat", 0),            epoch)
                        self.writer.add_scalar("Loss/Val_Mask",              val_metrics.get("loss_mask", 0),            epoch)
                        # 语义 mIoU（主要指标）
                        self.writer.add_scalar("Metrics/Semantic_mIoU",     val_metrics.get("semantic_miou", 0),        epoch)
                        self.writer.add_scalar("Metrics/N_Valid_Classes",    val_metrics.get("n_valid_classes", 0),      epoch)
                        # binary mask IoU（旧指标，保留对比用）
                        self.writer.add_scalar("Metrics/Binary_IoU",        val_metrics["iou"],                         epoch)
                        self.writer.add_scalar("Metrics/Binary_mIoU",       val_metrics.get("miou", 0),                 epoch)
                        self.writer.add_scalar("Metrics/Accuracy",          val_metrics["accuracy"],                    epoch)
                        self.writer.add_scalar("Metrics/mAcc",              val_metrics.get("macc", 0),                 epoch)
                    sem_miou = val_metrics.get("semantic_miou", 0)
                    n_cls    = val_metrics.get("n_valid_classes", 0)
                    print(
                        f"  Val Loss: {val_metrics['loss']:.4f} "
                        f"(feat={val_metrics.get('loss_feat',0):.4f} "
                        f"mask={val_metrics.get('loss_mask',0):.4f})  "
                        f"[主] Semantic mIoU: {sem_miou:.4f} ({n_cls} classes)  "
                        f"[参] Binary mIoU: {val_metrics.get('miou',0):.4f}"
                    )
                    # 打印每类 IoU（方便观察哪类学得好/差）
                    if "per_class_iou" in val_metrics and self.is_main_process:
                        per_cls = val_metrics["per_class_iou"]
                        cls_str = "  ".join(
                            f"{k}:{v:.3f}" for k, v in sorted(per_cls.items(), key=lambda x: -x[1])[:10]
                        )
                        print(f"  Top-10 classes: {cls_str}")

            # is_best 判断：以语义 mIoU 为主，没有时 fallback 到 binary mIoU
            is_best = False
            if val_metrics is not None:
                monitored = val_metrics.get("semantic_miou") or val_metrics.get("miou", val_metrics.get("iou", 0))
                if monitored > self.best_iou + self.config.early_stopping_min_delta:
                    prev = self.best_iou
                    self.best_iou = monitored
                    self.epochs_without_improvement = 0
                    is_best = True
                    if self.is_main_process:
                        print(f"  🎯 New best mIoU: {monitored:.4f} (prev: {prev:.4f})")
                else:
                    self.epochs_without_improvement += 1
                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
                monitored_value = monitored
            else:
                monitored_value = train_loss

            # Checkpoint
            if self.is_main_process:
                if (epoch + 1) % self.config.save_every_epochs == 0:
                    self._save_checkpoint(epoch, monitored_value, is_best=False, suffix=f"epoch_{epoch+1}")
                if is_best:
                    self._save_checkpoint(epoch, monitored_value, is_best=True)

            # Plateau scheduler
            if val_metrics is not None and self.config.scheduler_type == "plateau":
                self.scheduler.step(val_metrics["loss"])

            # Early stopping
            if (
                val_metrics is not None
                and self.epochs_without_improvement >= self.config.early_stopping_patience
            ):
                if self.is_main_process:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        if self.writer is not None:
            self.writer.close()
        if self.is_main_process:
            print(f"Done. Best mIoU={self.best_iou:.4f}  Best loss={self.best_loss:.4f}")

        return {
            "best_loss":   self.best_loss,
            "best_iou":    self.best_iou,
            "final_epoch": final_epoch,
        }
