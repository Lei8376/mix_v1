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
import contextlib

import os
import time
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from MinkowskiEngine import SparseTensor

from model.criterion import Criteria, dice_loss
from utils.util import AverageMeter
from experiment_mask_distill.semantic_miou import (
    Diff2SceneSemanticMIoUTracker, build_text_features,
)


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
    # GT 过滤阈值（与 Criteria 一致）
    min_points_per_mask: int = 10
    # Resume
    resume_checkpoint: Optional[str] = None
    # Resume behavior
    # NOTE: optimizer/scheduler state in checkpoint will override YAML hyperparams unless we re-apply them.
    override_optimizer_hparams_on_resume: bool = True
    # For iteration-stepped schedulers, changing batch_size changes steps_per_epoch and breaks old scheduler state.
    reset_scheduler_on_resume: bool = True
    # Quick run: limit batches per epoch (e.g. 100 for smoke test)
    max_batches_per_epoch: Optional[int] = None
    # Use float16 for model parameters (reduces VRAM; use with AMP)
    use_model_half: bool = False
    # 🔥 梯度累积：用于降低显存峰值，同时保持有效 batch size
    # 例：batch_size=16, gradient_accumulation_steps=2 → 有效 batch=32
    gradient_accumulation_steps: int = 1
    # Open-vocabulary semantic evaluation.
    semantic_clip_model: str = "ViT-L/14@336px"
    semantic_prompt_template: str = "a {} in a scene"


class MetricsTracker:
    """Track and compute evaluation metrics with per-mask mIoU and mAcc."""

    def __init__(self):
        self.reset()

    def reset(self):
        # 全局统计（用于计算全局 IoU）
        self.total_intersection = 0.0
        self.total_union = 0.0
        self.total_correct = 0
        self.total_points = 0
        # 🔥 OOM 修复：Per-mask 统计改用增量算法（O(1) 显存）而非 list 累积（O(N) 显存）
        # 计算逻辑完全一致，只是存储方式从 list.append → sum/count 累加
        self.per_mask_iou_sum = 0.0
        self.per_mask_iou_count = 0
        self.per_mask_acc_sum = 0.0
        self.per_mask_acc_count = 0

    def update(
        self,
        pred_masks: torch.Tensor,  # (N, K) probabilities
        gt_masks: torch.Tensor,    # (N, K) binary
        threshold: float = 0.5,
    ):
        """Update metrics with batch predictions."""
        pred_binary = (pred_masks > threshold).float()
        gt_binary = gt_masks.float()

        # 全局 IoU（所有点加在一起）
        intersection = (pred_binary * gt_binary).sum()
        union = ((pred_binary + gt_binary) > 0).float().sum()
        self.total_intersection += intersection.item()
        self.total_union += union.item()

        # 全局 Accuracy
        correct = (pred_binary == gt_binary).sum()
        total = pred_binary.numel()
        self.total_correct += correct.item()
        self.total_points += total

        # Per-mask IoU 和 Acc（每个 mask 单独计算，然后平均 = mIoU/mAcc）
        # 🔥 关键修复 C: 只对 GT 中有正样本的 mask 计算 IoU/Acc
        # 否则 GT=0 + pred>0 会导致 IoU=0，严重拉低 mIoU
        K = pred_binary.shape[1] if pred_binary.dim() > 1 else 1
        for k in range(K):
            if pred_binary.dim() > 1:
                pred_k = pred_binary[:, k]
                gt_k = gt_binary[:, k]
            else:
                pred_k = pred_binary
                gt_k = gt_binary
            
            gt_pos = gt_k.sum().item()
            
            # 🔥 只有当 GT 中有正样本（gt_pos > 0）才计算该 mask 的 IoU
            # 这与 mIoU 的标准定义一致：只对"存在的类别"计算平均
            if gt_pos > 0:
                inter_k = (pred_k * gt_k).sum().item()
                union_k = ((pred_k + gt_k) > 0).float().sum().item()
                if union_k > 0:
                    iou_k = inter_k / union_k
                    # 🔥 OOM 修复：改用增量累加（而非 list.append）
                    self.per_mask_iou_sum += iou_k
                    self.per_mask_iou_count += 1
                
                # Per-mask Accuracy（也只对 GT 有正样本的 mask 计入）
                correct_k = (pred_k == gt_k).sum().item()
                total_k = pred_k.numel()
                if total_k > 0:
                    acc_k = correct_k / total_k
                    # 🔥 OOM 修复：改用增量累加（而非 list.append）
                    self.per_mask_acc_sum += acc_k
                    self.per_mask_acc_count += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        # 全局 IoU（所有点）
        global_iou = (
            self.total_intersection / (self.total_union + 1e-6)
            if self.total_union > 0
            else 0.0
        )
        # 全局 Accuracy
        global_acc = (
            self.total_correct / (self.total_points + 1e-6)
            if self.total_points > 0
            else 0.0
        )
        # mIoU（per-mask IoU 的平均值）
        # 🔥 OOM 修复：从 sum(list)/len(list) 改为 sum/count（计算结果完全一致）
        miou = (
            self.per_mask_iou_sum / self.per_mask_iou_count
            if self.per_mask_iou_count > 0
            else 0.0
        )
        # mAcc（per-mask Accuracy 的平均值）
        macc = (
            self.per_mask_acc_sum / self.per_mask_acc_count
            if self.per_mask_acc_count > 0
            else 0.0
        )
        return {
            "iou": global_iou,      # 全局 IoU（向后兼容）
            "miou": miou,           # mIoU（per-mask 平均）
            "accuracy": global_acc, # 全局 Accuracy（向后兼容）
            "macc": macc,           # mAcc（per-mask 平均）
        }


class OpenVocabTrainerV2:
    """Enhanced trainer for open vocabulary 3D segmentation with DDP support."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: OpenVocabTrainerV2Config,
        device: str = "cuda",
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        train_sampler=None,  # DistributedSampler for DDP
        is_distributed: bool = False,
        is_main_process: bool = True,
    ):
        if config.use_model_half and device != "cpu":
            model = model.half()
        # Don't call .to(device) if model is already DDP-wrapped
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.to(device)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.train_sampler = train_sampler
        self.is_distributed = is_distributed
        self.is_main_process = is_main_process
        self._text_features = None   # 懒加载 CLIP 文本特征（语义 mIoU 用）

        # Optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )

        # Scheduler with warmup (cap steps if max_batches_per_epoch set for quick run)
        full_steps = len(train_loader)
        self.steps_per_epoch = (
            min(full_steps, config.max_batches_per_epoch)
            if config.max_batches_per_epoch is not None
            else full_steps
        )
        self.steps_per_epoch = max(1, self.steps_per_epoch)
        # 🔥 梯度累积后，scheduler/warmup/global_step 都应该以“优化步数(optimizer.step 次数)”为单位
        # 否则会出现：T_0 用 micro-step 计算，但 scheduler.step() 只在 optimizer.step() 调用 → LR 周期错误（你日志里 epoch12 LR 重置就是这个）
        self.accum_steps = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
        self.optim_steps_per_epoch = int(math.ceil(self.steps_per_epoch / self.accum_steps))
        self.total_steps = self.optim_steps_per_epoch * config.num_epochs
        self.warmup_steps = self.optim_steps_per_epoch * config.warmup_epochs

        if config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.optim_steps_per_epoch * config.scheduler_t0,
                T_mult=config.scheduler_t_mult,
                eta_min=config.scheduler_eta_min,
            )
        elif config.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.optim_steps_per_epoch * 10,
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

        # Logging (only on main process)
        if self.is_main_process:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            # 只有当 log_dir 非空时才创建 TensorBoard
            if config.log_dir and config.log_dir.strip():
                os.makedirs(config.log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=config.log_dir)
            else:
                self.writer = None
        else:
            self.writer = None

        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.best_iou = 0.0
        self.epochs_without_improvement = 0

        # Resume if specified
        if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
            self._load_checkpoint(config.resume_checkpoint)

        # Count parameters (only print on main process)
        trainable_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_count = sum(p.numel() for p in self.model.parameters())
        if self.is_main_process:
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
        # Also move precomputed features if present (pixel_pooled = pre-pooled from npz)
        precomputed_keys = [
            "pixel_embeddings", "pixel_pooled", "masks", "mask_embeddings", "mask_valid",
        ]
        moved = dict(batch)
        for key in tensor_keys + precomputed_keys:
            if key in moved and isinstance(moved[key], torch.Tensor):
                moved[key] = moved[key].to(self.device, non_blocking=True)
        return moved

    def _build_sparse_tensor(self, batch: Dict[str, Any]) -> SparseTensor:
        """Build MinkowskiEngine SparseTensor from batch."""
        # 使用 .int() 避免 MinkowskiEngine 的 "coordinates implicitly converted to torch.IntTensor" 警告
        return SparseTensor(batch["feat_3d"], batch["coords_3d"].int())

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        epoch_loss = AverageMeter()
        epoch_bce = AverageMeter()
        epoch_dice = AverageMeter()
        batch_time = AverageMeter()

        end_time = time.time()

        # 🔥 梯度累积配置
        accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
        is_distributed = hasattr(self.model, "no_sync")  # DDP 模型有 no_sync 方法
        
        # 在 epoch 开始时清零梯度
        self.optimizer.zero_grad(set_to_none=True)

        max_steps = self.config.max_batches_per_epoch
        micro_steps_done = 0
        for step, batch in enumerate(self.train_loader):
            if max_steps is not None and step >= max_steps:
                break
            micro_steps_done += 1
            batch = self._move_batch_to_device(batch)
            batch["sinput"] = self._build_sparse_tensor(batch)

            # 🔥 梯度累积：判断是否为累积中间步（不需要同步梯度）
            is_accum_step = ((step + 1) % accum_steps != 0)
            
            # 🔥 DDP 优化：累积中间步使用 no_sync() 避免每次都 all-reduce
            sync_context = self.model.no_sync() if (is_accum_step and is_distributed) else contextlib.nullcontext()

            with sync_context:
                # Forward pass with AMP
                with autocast(enabled=self.config.use_amp):
                    try:
                        results = self.model(batch)
                        criteria = Criteria(
                            results,
                            batch,
                            bce_weight=self.config.bce_weight,
                            dice_weight=self.config.dice_weight,
                            min_points_per_mask=self.config.min_points_per_mask,
                            use_pos_weight=True,
                            use_per_mask_dice=True,
                        )
                        loss = criteria.loss_pt()
                        
                        # 检查 loss 是否有效
                        if torch.isnan(loss) or torch.isinf(loss):
                            raise ValueError(f"Invalid loss detected: {loss}. This indicates a training instability issue.")
                            
                    except Exception as e:
                        # 打印诊断信息，然后重新抛出错误（不跳过 batch）
                        print(f"\n❌ Error in forward pass: {e}")
                        print(f"Batch keys: {list(batch.keys())}")
                        if "x_label" in batch and "y_label" in batch:
                            print(f"x_label range: [{batch['x_label'].min()}, {batch['x_label'].max()}]")
                            print(f"y_label range: [{batch['y_label'].min()}, {batch['y_label'].max()}]")
                        raise

                # 🔥 梯度累积：loss 除以累积步数，保证累积后的梯度量级与原来一致
                scaled_loss = loss / accum_steps
                
                # Backward（累积梯度，不清零）
                self.scaler.scale(scaled_loss).backward()

            # 记录原始 loss（用于日志，不是 scaled_loss）
            epoch_loss.update(loss.item())
            
            # Logging（每个 micro-step 都记录 loss）
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # 🔥 只在累积边界（每 accum_steps 个 step）执行 optimizer.step()
            if (step + 1) % accum_steps == 0:
                # Unscale + Clip + Step + Update
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.config.grad_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # 🔥 scheduler 只在 optimizer.step() 时更新（与优化步数对齐）
                if self.global_step >= self.warmup_steps:
                    if self.config.scheduler_type != "plateau":
                        self.scheduler.step()

                # 🔥 global_step 按优化步数更新（而不是 micro-step）
                if self.writer is not None:
                    self.writer.add_scalar("Loss/Train_Step", loss.item(), self.global_step)
                    self.writer.add_scalar(
                        "LR", self.optimizer.param_groups[0]["lr"], self.global_step
                    )

                    # Log fusion alpha (ODISE-residual fusion mixing weight)
                    fuse = self.model.fuse_embed if hasattr(self.model, "fuse_embed") \
                        else getattr(self.model, "module", None) and self.model.module.fuse_embed
                    if fuse is not None and hasattr(fuse, "alpha"):
                        self.writer.add_scalar("Fusion/alpha", fuse.alpha.item(), self.global_step)
                self.global_step += 1

            # Apply warmup（基于 global_step，即优化步数）
            self._adjust_learning_rate_warmup(self.global_step)

            # 日志打印（每 log_every_steps 个 micro-step 打印一次）
            if step % self.config.log_every_steps == 0 and self.is_main_process:
                current_lr = self.optimizer.param_groups[0]["lr"]
                eta = batch_time.avg * (len(self.train_loader) - step)
                accum_info = f" (accum {accum_steps})" if accum_steps > 1 else ""
                print(
                    f"Epoch [{epoch + 1}/{self.config.num_epochs}] "
                    f"Step [{step}/{len(self.train_loader)}]{accum_info} "
                    f"Loss: {loss.item():.4f} ({epoch_loss.avg:.4f}) "
                    f"LR: {current_lr:.2e} "
                    f"ETA: {eta:.0f}s"
                )

        # 🔥 处理 epoch 结束时剩余的未累积完的梯度
        # 如果 total_steps % accum_steps != 0，最后几个 step 的梯度还没 step
        remaining_steps = micro_steps_done % accum_steps
        if remaining_steps > 0 and max_steps is None:
            # 执行最后一次 optimizer.step()
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

        return epoch_loss.avg

    def _select_item_rows(
        self,
        pred_mask_values: torch.Tensor,
        pt_mask: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        """Return rows for one batch item, accepting item-local or batch-global tensors."""
        n_rows = pred_mask_values.shape[0]
        n_total = pt_mask.numel()
        n_item = int(pt_mask.sum().item())

        if n_rows == n_item:
            return pred_mask_values
        if n_rows == n_total:
            return pred_mask_values[pt_mask]

        raise RuntimeError(
            f"{context}: pred_mask rows do not match item or batch points "
            f"(rows={n_rows}, item_points={n_item}, total_points={n_total})."
        )

    def _select_valid_mask_columns(
        self,
        pred_mask_values: torch.Tensor,
        valid_k: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        """Return valid mask columns, accepting full K_max or already-filtered K_valid tensors."""
        valid_k = valid_k.to(device=pred_mask_values.device, dtype=torch.bool)
        n_cols = pred_mask_values.shape[1]
        n_full = valid_k.numel()
        n_valid = int(valid_k.sum().item())

        if n_cols == n_full:
            return pred_mask_values[:, valid_k]
        if n_cols == n_valid:
            return pred_mask_values

        raise RuntimeError(
            f"{context}: pred_mask columns do not match full or valid mask slots "
            f"(cols={n_cols}, full_masks={n_full}, valid_masks={n_valid})."
        )

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss = AverageMeter()
        metrics = MetricsTracker()

        # 语义 mIoU：懒加载 CLIP 文本特征（只加载一次）
        if self._text_features is None:
            try:
                self._text_features = build_text_features(
                    device=self.device,
                    clip_model=self.config.semantic_clip_model,
                    prompt_template=self.config.semantic_prompt_template,
                )
            except Exception as e:
                if self.is_main_process:
                    print(f"[SemanticMIoU] Failed to build text features: {e}")
                self._text_features = None

        text_feats = self._text_features
        # Diff2Scene Eq.3：mask label probabilities are assigned to points
        # through predicted 3D mask probabilities.
        sem_tracker = Diff2SceneSemanticMIoUTracker() if text_feats is not None else None

        for batch_idx, batch in enumerate(self.val_loader):
            batch = self._move_batch_to_device(batch)
            batch["sinput"] = self._build_sparse_tensor(batch)

            with autocast(enabled=self.config.use_amp):
                results = self.model(batch)
                criteria = Criteria(
                    results, batch,
                    bce_weight=self.config.bce_weight,
                    dice_weight=self.config.dice_weight,
                    min_points_per_mask=self.config.min_points_per_mask,  # 🔥 GT 过滤阈值
                    use_pos_weight=True,  # 🔥 启用 pos_weight 平衡正负样本
                    use_per_mask_dice=True,  # 🔥 使用 per-mask dice loss
                )
                loss = criteria.loss_pt()

            val_loss.update(loss.item())

            # Compute metrics for each batch item
            for b in range(len(results["outputs"])):
                if len(results["outputs"][b]) == 0:
                    continue
                pred_logits = results["outputs"][b][0]["pred_mask_logits"]
                valid = results["mask_valid_from_masks"][b]
                point_mask = results["batch_indices"] == b
                pred_logits = self._select_item_rows(
                    pred_logits,
                    point_mask,
                    context=f"val mask metric batch={batch_idx} item={b}",
                )

                # Get GT masks
                mask_2d = results["mask_masks"][b][valid]
                x_idx = batch["x_label"][point_mask].float()  # 先转 float 以便缩放
                y_idx = batch["y_label"][point_mask].float()

                if x_idx.numel() == 0:
                    continue

                # mask 尺寸
                H, W = mask_2d.shape[1], mask_2d.shape[2]
                
                # 🔥 与训练时一致的缩放逻辑
                x_max = x_idx.max().item() if x_idx.numel() > 0 else 0
                y_max = y_idx.max().item() if y_idx.numel() > 0 else 0
                
                # 判断是否需要缩放（容差 20 像素）
                need_scale = (x_max > W + 20) or (y_max > H + 20)
                
                if need_scale:
                    # 原图尺寸投影，需要缩放到 mask 尺寸
                    orig_W = max(640, x_max + 10)
                    orig_H = max(480, y_max + 10)
                    scale_x = W / orig_W
                    scale_y = H / orig_H
                    x_idx = (x_idx * scale_x).long()
                    y_idx = (y_idx * scale_y).long()
                else:
                    # 已经是 mask 尺寸，直接转 long
                    x_idx = x_idx.long()
                    y_idx = y_idx.long()
                
                # 越界过滤（与训练时一致）
                valid_mask = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
                if valid_mask.sum() == 0:
                    continue
                
                x_idx = x_idx[valid_mask]
                y_idx = y_idx[valid_mask]
                pred_logits_filtered = pred_logits[valid_mask, :]

                gt_3d = mask_2d[:, y_idx, x_idx]
                gt_3d = (gt_3d > 0.5).float().transpose(0, 1)  # (N_valid, K_valid)

                pred_valid = pred_logits_filtered[:, valid]  # (N_valid, K_valid)
                
                # 🔥 验证时与训练一致：过滤 GT 正样本数不足的 mask
                # 这确保训练指标和验证指标使用相同的 mask 定义
                gt_pos = gt_3d.sum(dim=0)  # (K_valid,)
                min_points = self.config.min_points_per_mask
                keep_gt = gt_pos >= min_points
                
                if keep_gt.any():
                    pred_valid = pred_valid[:, keep_gt]
                    gt_3d = gt_3d[:, keep_gt]
                    # MetricsTracker 期望概率；pred_logits 为 logits，需先 sigmoid
                    metrics.update(torch.sigmoid(pred_valid).float(), gt_3d)

            # ---- 语义 mIoU：Diff2Scene Eq.3，fused_embed × pred_mask ----
            if sem_tracker is not None:
                fused_all   = results["fused_embeddings"]   # (B, K_max, 768)
                mask_valid  = results["mask_valid_from_masks"]  # (B, K_max)
                for b in range(len(results["outputs"])):
                    if len(results["outputs"][b]) == 0:
                        continue
                    valid_k = mask_valid[b]
                    if not valid_k.any():
                        continue
                    pred_logits_b = results["outputs"][b][0]["pred_mask_logits"]
                    pt_mask = results["batch_indices"] == b
                    fused_b = fused_all[b][valid_k]                       # (K_valid, 768)
                    pred_logits_b = self._select_item_rows(
                        pred_logits_b,
                        pt_mask,
                        context=f"D2S semantic mIoU batch={batch_idx} item={b}",
                    )
                    logits_b = self._select_valid_mask_columns(
                        pred_logits_b,
                        valid_k,
                        context=f"D2S semantic mIoU batch={batch_idx} item={b}",
                    ).float()                                             # (N_b, K_valid)
                    gt_b = batch["binary_label_3d"][pt_mask]              # (N_b,)
                    if logits_b.shape[0] != gt_b.numel():
                        raise RuntimeError(
                            f"D2S semantic mIoU batch={batch_idx} item={b}: "
                            f"logits rows ({logits_b.shape[0]}) != gt labels ({gt_b.numel()})."
                        )
                    if logits_b.shape[1] != fused_b.shape[0]:
                        raise RuntimeError(
                            f"D2S semantic mIoU batch={batch_idx} item={b}: "
                            f"logits masks ({logits_b.shape[1]}) != fused masks ({fused_b.shape[0]})."
                        )
                    sem_tracker.update(gt_b, fused_b, logits_b, text_feats)

            # 🔥 OOM 修复：每 50 个 batch 清理一次 CUDA 缓存（不影响计算结果）
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

        val_metrics = metrics.compute()
        val_metrics["loss"] = val_loss.avg

        # 语义 mIoU：Diff2Scene Eq.3
        val_metrics["semantic_miou"]   = 0.0
        val_metrics["semantic_miou_diff2scene"] = 0.0
        val_metrics["n_valid_classes"] = 0
        if sem_tracker is not None:
            sem_res = sem_tracker.compute()
            val_metrics["semantic_miou"] = sem_res["semantic_miou_diff2scene"]
            val_metrics["semantic_miou_diff2scene"] = sem_res["semantic_miou_diff2scene"]
            val_metrics["n_valid_classes"] = sem_res["n_valid_classes"]
            val_metrics["per_class_iou_diff2scene"] = sem_res.get("per_class_iou_diff2scene", {})

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
            # 定期保存：checkpoint_epoch_5.pth, checkpoint_epoch_10.pth, ...
            path = f"{self.config.checkpoint_dir}/checkpoint_{suffix}.pth"
            torch.save(checkpoint, path)
        
        if is_best:
            # 最佳模型：best_model.pth
            best_path = f"{self.config.checkpoint_dir}/best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  -> Saved best model (mIoU/loss: {loss:.4f})")

    def _reset_scheduler_to_current_config(self):
        """Recreate scheduler with current config & steps_per_epoch.

        This avoids broken LR schedules when resuming with different batch_size
        (steps_per_epoch changes) or when you intentionally change schedule hyperparams.
        """
        cfg = self.config
        if cfg.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.optim_steps_per_epoch * cfg.scheduler_t0,
                T_mult=cfg.scheduler_t_mult,
                eta_min=cfg.scheduler_eta_min,
            )
            # We call scheduler.step() once per optimizer.step, so treat global_step as the scheduler "epoch".
            if self.global_step > 0:
                self.scheduler.step(self.global_step)
        elif cfg.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.optim_steps_per_epoch * 10,
                gamma=0.5,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle DDP vs non-DDP checkpoint loading
        state_dict = checkpoint["model_state_dict"]
        model_state_dict = self.model.state_dict()
        
        # Check if we need to add or remove 'module.' prefix
        model_keys = list(model_state_dict.keys())
        ckpt_keys = list(state_dict.keys())
        
        is_model_ddp = model_keys[0].startswith('module.')
        is_ckpt_ddp = ckpt_keys[0].startswith('module.')
        
        if is_model_ddp and not is_ckpt_ddp:
            # Model is DDP but checkpoint is not -> add 'module.' prefix
            print("Adding 'module.' prefix to checkpoint keys for DDP model")
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not is_model_ddp and is_ckpt_ddp:
            # Model is not DDP but checkpoint is -> remove 'module.' prefix
            print("Removing 'module.' prefix from checkpoint keys")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # strict=False so newly added params (e.g. fuse_embed.alpha for ODISE-residual fusion)
        # default to their init values when resuming from older checkpoints
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[resume] missing keys (will use init values): {missing}")
        if unexpected:
            print(f"[resume] unexpected keys (ignored): {unexpected}")

        # Optimizer state (will override lr/weight_decay in param_groups)
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Re-apply YAML hyperparams if requested (otherwise config changes won't take effect on resume)
        if getattr(self.config, "override_optimizer_hparams_on_resume", True):
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config.base_lr
                pg["weight_decay"] = self.config.weight_decay

        # Scheduler
        if self.config.scheduler_type == "plateau":
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            if getattr(self.config, "reset_scheduler_on_resume", True):
                self._reset_scheduler_to_current_config()
            else:
                if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.best_iou = checkpoint.get("best_iou", 0.0)

        if self.is_main_process:
            lr_now = self.optimizer.param_groups[0].get("lr", None)
            wd_now = self.optimizer.param_groups[0].get("weight_decay", None)
            print(
                f"After resume: lr={lr_now} weight_decay={wd_now} "
                f"steps_per_epoch={self.steps_per_epoch} warmup_steps={self.warmup_steps}"
            )

        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def train(self) -> Dict[str, float]:
        """Run full training loop with DDP support."""
        if self.is_main_process:
            print(f"Starting training for {self.config.num_epochs} epochs")
            print(f"Steps per epoch: {self.steps_per_epoch}")
            if getattr(self.config, "gradient_accumulation_steps", 1) > 1:
                print(
                    f"Optimizer steps/epoch: {self.optim_steps_per_epoch} "
                    f"(accum={self.accum_steps})"
                )
            print(f"Total steps: {self.total_steps}")
            print(f"Warmup steps: {self.warmup_steps}")
            print(f"AMP enabled: {self.config.use_amp}")

        final_epoch = self.current_epoch
        for epoch in range(self.current_epoch, self.config.num_epochs):
            final_epoch = epoch + 1
            epoch_start = time.time()
            
            # Set epoch for distributed sampler (required for proper shuffling)
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            if self.is_main_process:
                print(f"\n>> Starting Epoch [{epoch + 1}/{self.config.num_epochs}] ...")

            # Training
            train_loss = self._train_epoch(epoch)
            if self.writer is not None:
                self.writer.add_scalar("Loss/Train_Epoch", train_loss, epoch)

            epoch_time = time.time() - epoch_start
            if self.is_main_process:
                print(
                    f"Epoch [{epoch + 1}/{self.config.num_epochs}] "
                    f"Train Loss: {train_loss:.4f} "
                    f"Time: {epoch_time:.1f}s"
                )

            # Validation
            val_metrics = None
            if (epoch + 1) % self.config.val_every_epochs == 0:
                val_metrics = self._validate(epoch)
                if val_metrics and self.is_main_process:
                    if self.writer is not None:
                        self.writer.add_scalar("Loss/Val",                          val_metrics["loss"],                            epoch)
                        self.writer.add_scalar("Metrics/IoU",                       val_metrics["iou"],                             epoch)
                        self.writer.add_scalar("Metrics/mIoU",                      val_metrics.get("miou", 0),                     epoch)
                        self.writer.add_scalar("Metrics/Accuracy",                  val_metrics["accuracy"],                        epoch)
                        self.writer.add_scalar("Metrics/mAcc",                      val_metrics.get("macc", 0),                     epoch)
                        self.writer.add_scalar("Metrics/Semantic_mIoU_Diff2Scene",  val_metrics.get("semantic_miou_diff2scene", 0), epoch)
                        if "per_class_iou_diff2scene" in val_metrics:
                            for cls_name, iou_val in val_metrics["per_class_iou_diff2scene"].items():
                                self.writer.add_scalar(f"PerClass_IoU_D2S/{cls_name}", iou_val, epoch)
                    print(
                        f"  Val Loss: {val_metrics['loss']:.4f} "
                        f"[MaskIoU] {val_metrics.get('miou', 0):.4f}  "
                        f"[语义mIoU-D2S] {val_metrics.get('semantic_miou_diff2scene', 0):.4f} "
                        f"({val_metrics.get('n_valid_classes', 0)} classes)  "
                        f"Acc: {val_metrics['accuracy']:.4f}"
                    )
                    if "per_class_iou_diff2scene" in val_metrics and val_metrics["per_class_iou_diff2scene"]:
                        per_cls = val_metrics["per_class_iou_diff2scene"]
                        top10 = "  ".join(
                            f"{k}:{v:.3f}" for k, v in
                            sorted(per_cls.items(), key=lambda x: -x[1])[:10]
                        )
                        print(f"  Top-10 (D2S): {top10}")
                    
                    # Update best IoU
                    current_iou = val_metrics.get('miou', val_metrics['iou'])
                    if current_iou > self.best_iou:
                        self.best_iou = current_iou

                    # Update plateau scheduler with validation loss
                    if self.config.scheduler_type == "plateau":
                        self.scheduler.step(val_metrics["loss"])

            # Save best model (only on main process)
            # 🔥 关键修复：按 mIoU 而非 loss 保存 best model
            # 只在有验证结果时判断 is_best，避免用 train_loss 误判过拟合
            is_best = False
            if val_metrics is not None and isinstance(val_metrics, dict):
                # 优先监控 mIoU
                monitored_metric = val_metrics.get('miou', val_metrics.get('iou', 0))
                is_best = monitored_metric > self.best_iou + self.config.early_stopping_min_delta
                
                if is_best:
                    prev_best = self.best_iou
                    self.best_iou = monitored_metric
                    self.epochs_without_improvement = 0
                    if self.is_main_process:
                        print(f"  🎯 New best mIoU: {monitored_metric:.4f} (prev: {prev_best:.4f})")
                else:
                    self.epochs_without_improvement += 1
                
                # 同时更新 best_loss（用于日志）
                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
                
                monitored_value = monitored_metric  # 用于 checkpoint 文件名
            else:
                # 没有 val_metrics 时，不判断 is_best（避免过拟合）
                monitored_value = train_loss
                # 注意：不更新 epochs_without_improvement（只在验证时更新）

            # Save checkpoints (only on main process)
            if self.is_main_process:
                # 🔥 修复：只在两种情况下保存 checkpoint
                # 1. 定期保存：每 save_every_epochs 个 epoch
                # 2. 最佳模型：is_best=True 时保存 best_model.pth
                
                # 定期保存（每 N 个 epoch）
                if (epoch + 1) % self.config.save_every_epochs == 0:
                    self._save_checkpoint(
                        epoch, monitored_value, 
                        is_best=False,  # 定期保存不标记为 best
                        suffix=f"epoch_{epoch + 1}"
                    )
                
                # 最佳模型保存（仅当 is_best=True）
                if is_best:
                    self._save_checkpoint(
                        epoch, monitored_value, 
                        is_best=True,  # 保存为 best_model.pth
                        suffix=""  # 空 suffix 表示保存为 best_model.pth
                    )

            # Early stopping
            # 🔥 修复：只在有验证结果时检查早停（避免误判）
            if (
                val_metrics is not None 
                and self.epochs_without_improvement >= self.config.early_stopping_patience
            ):
                if self.is_main_process:
                    print(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"({self.epochs_without_improvement} epochs without improvement)"
                    )
                break

        if self.writer is not None:
            self.writer.close()
        if self.is_main_process:
            print(f"Training complete! Best loss: {self.best_loss:.4f}")

        return {
            "best_loss": self.best_loss,
            "best_iou": self.best_iou,
            "final_epoch": final_epoch,
        }
