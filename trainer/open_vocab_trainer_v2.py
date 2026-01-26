"""
Enhanced Open Vocabulary 3D Trainer with:
- Mixed precision training (AMP)
- Validation loop with metrics (IoU, mIoU)
- Checkpoint resumption
- Learning rate warmup
- Detailed logging
- Early stopping
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import os
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from MinkowskiEngine import SparseTensor

from model.criterion import Criteria, dice_loss
from utils.util import AverageMeter


@dataclass
class OpenVocabTrainerV2Config:
    """Enhanced trainer configuration."""
    num_epochs: int = 50
    base_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    log_dir: str = "runs/open_vocab_3d_v2"
    checkpoint_dir: str = "checkpoints"
    log_every_steps: int = 20
    val_every_epochs: int = 1
    save_every_epochs: int = 5
    # Scheduler
    warmup_epochs: int = 2
    scheduler_type: str = "cosine"  # cosine, step, plateau
    scheduler_t0: int = 1
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 1e-6
    # AMP
    use_amp: bool = True
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    # Loss weights
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    # Resume
    resume_checkpoint: Optional[str] = None


class MetricsTracker:
    """Track and compute evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_intersection = 0.0
        self.total_union = 0.0
        self.total_correct = 0
        self.total_points = 0
        self.per_class_intersection = {}
        self.per_class_union = {}

    def update(
        self,
        pred_masks: torch.Tensor,  # (N, K) probabilities
        gt_masks: torch.Tensor,    # (N, K) binary
        threshold: float = 0.5,
    ):
        """Update metrics with batch predictions."""
        pred_binary = (pred_masks > threshold).float()
        gt_binary = gt_masks.float()

        # IoU computation
        intersection = (pred_binary * gt_binary).sum()
        union = ((pred_binary + gt_binary) > 0).float().sum()

        self.total_intersection += intersection.item()
        self.total_union += union.item()

        # Accuracy
        correct = (pred_binary == gt_binary).sum()
        total = pred_binary.numel()
        self.total_correct += correct.item()
        self.total_points += total

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        iou = (
            self.total_intersection / (self.total_union + 1e-6)
            if self.total_union > 0
            else 0.0
        )
        accuracy = (
            self.total_correct / (self.total_points + 1e-6)
            if self.total_points > 0
            else 0.0
        )
        return {
            "iou": iou,
            "accuracy": accuracy,
        }


class OpenVocabTrainerV2:
    """Enhanced trainer for open vocabulary 3D segmentation."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: OpenVocabTrainerV2Config,
        device: str = "cuda",
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )

        # Scheduler with warmup
        self.steps_per_epoch = max(1, len(train_loader))
        self.total_steps = self.steps_per_epoch * config.num_epochs
        self.warmup_steps = self.steps_per_epoch * config.warmup_epochs

        if config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.steps_per_epoch * config.scheduler_t0,
                T_mult=config.scheduler_t_mult,
                eta_min=config.scheduler_eta_min,
            )
        elif config.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.steps_per_epoch * 10,
                gamma=0.5,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )

        # AMP scaler
        self.scaler = GradScaler(enabled=config.use_amp)

        # Logging
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.best_iou = 0.0
        self.epochs_without_improvement = 0

        # Resume if specified
        if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
            self._load_checkpoint(config.resume_checkpoint)

        # Count parameters
        trainable_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_count = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {trainable_count:,} trainable / {total_count:,} total")

    def _get_warmup_lr(self, step: int) -> float:
        """Compute learning rate with linear warmup."""
        if step < self.warmup_steps:
            return self.config.base_lr * (step + 1) / self.warmup_steps
        return self.config.base_lr

    def _adjust_learning_rate_warmup(self, step: int):
        """Apply warmup to learning rate."""
        if step < self.warmup_steps:
            lr = self._get_warmup_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def _move_batch_to_device(
        self, batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Move batch tensors to device."""
        tensor_keys = [
            "coords_3d", "feat_3d", "ori_coords_3d",
            "binary_label_3d", "binary_label_2d", "label_2d",
            "img", "x_label", "y_label", "inds_reconstruct",
        ]
        # Also move precomputed features if present
        precomputed_keys = [
            "pixel_embeddings", "masks", "mask_embeddings", "mask_valid",
        ]
        moved = dict(batch)
        for key in tensor_keys + precomputed_keys:
            if key in moved and isinstance(moved[key], torch.Tensor):
                moved[key] = moved[key].to(self.device, non_blocking=True)
        return moved

    def _build_sparse_tensor(self, batch: Dict[str, Any]) -> SparseTensor:
        """Build MinkowskiEngine SparseTensor from batch."""
        return SparseTensor(batch["feat_3d"], batch["coords_3d"])

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        epoch_loss = AverageMeter()
        epoch_bce = AverageMeter()
        epoch_dice = AverageMeter()
        batch_time = AverageMeter()

        end_time = time.time()

        for step, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)
            batch["sinput"] = self._build_sparse_tensor(batch)

            # Apply warmup
            self._adjust_learning_rate_warmup(self.global_step)

            # Forward pass with AMP
            with autocast(enabled=self.config.use_amp):
                results = self.model(batch)
                criteria = Criteria(
                    results,
                    batch,
                    bce_weight=self.config.bce_weight,
                    dice_weight=self.config.dice_weight,
                )
                loss = criteria.loss_pt()

            if loss.item() == 0:
                continue

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.config.grad_clip_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update scheduler (after warmup)
            if self.global_step >= self.warmup_steps:
                if self.config.scheduler_type != "plateau":
                    self.scheduler.step()

            # Logging
            epoch_loss.update(loss.item())
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            self.writer.add_scalar("Loss/Train_Step", loss.item(), self.global_step)
            self.writer.add_scalar(
                "LR", self.optimizer.param_groups[0]["lr"], self.global_step
            )
            self.global_step += 1

            if step % self.config.log_every_steps == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                eta = batch_time.avg * (len(self.train_loader) - step)
                print(
                    f"Epoch [{epoch + 1}/{self.config.num_epochs}] "
                    f"Step [{step}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} ({epoch_loss.avg:.4f}) "
                    f"LR: {current_lr:.2e} "
                    f"ETA: {eta:.0f}s"
                )

        return epoch_loss.avg

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss = AverageMeter()
        metrics = MetricsTracker()

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)
            batch["sinput"] = self._build_sparse_tensor(batch)

            with autocast(enabled=self.config.use_amp):
                results = self.model(batch)
                criteria = Criteria(
                    results, batch,
                    bce_weight=self.config.bce_weight,
                    dice_weight=self.config.dice_weight,
                )
                loss = criteria.loss_pt()

            if loss.item() > 0:
                val_loss.update(loss.item())

            # Compute metrics for each batch item
            for b in range(len(results["outputs"])):
                if len(results["outputs"][b]) == 0:
                    continue
                pred_logits = results["outputs"][b][0]["pred_mask_logits"]
                valid = results["mask_valid_from_masks"][b]

                # Get GT masks
                mask_2d = results["mask_masks"][b][valid]
                point_mask = results["batch_indices"] == b
                x_idx = batch["x_label"][point_mask].long()
                y_idx = batch["y_label"][point_mask].long()

                if x_idx.numel() == 0:
                    continue
                if (
                    x_idx.max().item() >= mask_2d.shape[2]
                    or y_idx.max().item() >= mask_2d.shape[1]
                ):
                    continue

                gt_3d = mask_2d[:, y_idx, x_idx]
                gt_3d = (gt_3d > 0.5).float().transpose(0, 1)

                pred_valid = pred_logits[:, valid]
                metrics.update(pred_valid, gt_3d)

        val_metrics = metrics.compute()
        val_metrics["loss"] = val_loss.avg

        return val_metrics

    def _save_checkpoint(
        self, epoch: int, loss: float, is_best: bool = False, suffix: str = ""
    ):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "best_iou": self.best_iou,
            "config": self.config,
        }

        if suffix:
            path = f"{self.config.checkpoint_dir}/checkpoint_{suffix}.pth"
        else:
            path = f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth"

        torch.save(checkpoint, path)

        if is_best:
            best_path = f"{self.config.checkpoint_dir}/best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  -> Saved best model (loss: {loss:.4f})")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.best_iou = checkpoint.get("best_iou", 0.0)

        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def train(self) -> Dict[str, float]:
        """Run full training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Total steps: {self.total_steps}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"AMP enabled: {self.config.use_amp}")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            epoch_start = time.time()

            # Training
            train_loss = self._train_epoch(epoch)
            self.writer.add_scalar("Loss/Train_Epoch", train_loss, epoch)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch [{epoch + 1}/{self.config.num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Time: {epoch_time:.1f}s"
            )

            # Validation
            if (epoch + 1) % self.config.val_every_epochs == 0:
                val_metrics = self._validate(epoch)
                if val_metrics:
                    self.writer.add_scalar("Loss/Val", val_metrics["loss"], epoch)
                    self.writer.add_scalar("Metrics/IoU", val_metrics["iou"], epoch)
                    self.writer.add_scalar(
                        "Metrics/Accuracy", val_metrics["accuracy"], epoch
                    )
                    print(
                        f"  Val Loss: {val_metrics['loss']:.4f} "
                        f"IoU: {val_metrics['iou']:.4f} "
                        f"Acc: {val_metrics['accuracy']:.4f}"
                    )

                    # Update plateau scheduler with validation loss
                    if self.config.scheduler_type == "plateau":
                        self.scheduler.step(val_metrics["loss"])

            # Save best model
            is_best = train_loss < self.best_loss - self.config.early_stopping_min_delta
            if is_best:
                self.best_loss = train_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self._save_checkpoint(epoch, train_loss, is_best=is_best)

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_epochs == 0:
                self._save_checkpoint(
                    epoch, train_loss, suffix=f"epoch_{epoch + 1}"
                )

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"({self.epochs_without_improvement} epochs without improvement)"
                )
                break

        self.writer.close()
        print(f"Training complete! Best loss: {self.best_loss:.4f}")

        return {
            "best_loss": self.best_loss,
            "best_iou": self.best_iou,
            "final_epoch": epoch + 1,
        }

