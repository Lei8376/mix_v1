import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 🔥 模块级统计变量：跨 Criteria 实例持久化，便于统计多个 step 的 GT 分布
_gt_stats_global = {
    'count': 0,
    'total_masks': 0,
    'zero_masks': 0,
    'kept_masks': 0,
    'pos_sum': 0,
}


def dice_loss(pred, target, smooth=1.0):
    """
    Compute Dice loss for binary masks (legacy global version).
    pred: (N, K) - soft predictions (sigmoid output)
    target: (N, K) - binary targets
    """
    pred = pred.contiguous().reshape(-1)
    target = target.contiguous().reshape(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def dice_loss_per_mask(pred, target, smooth=1.0):
    """
    Compute per-mask Dice loss then average (recommended for training).
    pred: (N, K) - soft predictions (sigmoid output)
    target: (N, K) - binary targets
    
    修复：改为 per-mask 计算再平均，更贴近 mIoU 指标
    """
    # Per-mask intersection and union
    inter = (pred * target).sum(dim=0)  # (K,)
    denom = pred.sum(dim=0) + target.sum(dim=0)  # (K,)
    
    # Per-mask dice
    dice_per = (2. * inter + smooth) / (denom + smooth)  # (K,)
    
    # Average across masks
    return 1 - dice_per.mean()


def sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance in mask prediction.
    """
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


class Criteria(nn.Module):
    def __init__(self, results, batch_input, threshold=0.5, min_points_per_mask=10,
                 bce_weight=1.0, dice_weight=1.0, use_keep_filter=False,
                 use_pos_weight=True, use_per_mask_dice=True):
        """
        Args:
            use_keep_filter: 是否使用硬阈值过滤 mask。
                - False (默认，推荐训练时): 所有 valid mask 都参与 loss，避免训练初期 loss=0
                - True (推荐推理时): 只计算有足够 >0.5 点的 mask
            use_pos_weight: 是否使用 pos_weight 平衡正负样本（强烈推荐 True）
            use_per_mask_dice: 是否使用 per-mask dice loss（推荐 True）
        """
        super(Criteria, self).__init__()
        self.outputs = results["outputs"] 
        self.mask_valid_from_masks = results["mask_valid_from_masks"]
        self.min_points_per_mask = min_points_per_mask
        self.mask_masks = results["mask_masks"]
        self.batch_indices = results["batch_indices"]
        self.x_label = batch_input["x_label"]
        self.y_label = batch_input["y_label"]
        self.threshold = threshold
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.use_keep_filter = use_keep_filter
        self.use_pos_weight = use_pos_weight
        self.use_per_mask_dice = use_per_mask_dice

    def loss_pt(self):
        """Compute loss using BCE + Dice loss for mask prediction."""
        batch = len(self.outputs)
        losses = []
        
        for i in range(batch):
            # Skip if no outputs for this batch item
            if len(self.outputs[i]) == 0:
                continue
                
            # Get soft prediction logits (differentiable)
            mask_logit = self.outputs[i][0]["pred_mask_logits"]  # (N_points, num_masks_padded)
            valid = self.mask_valid_from_masks[i]  # Boolean mask for valid ODISE masks
            
            # Skip if no valid masks
            if not valid.any():
                continue
            
            # Filter to only valid masks（pred_mask_logits 为 logits，未 sigmoid）
            mask_logit_valid = mask_logit[:, valid]  # (N_points, K_valid)
            
            # 训练时不使用硬阈值过滤，让所有 valid mask 参与 loss（避免训练初期 loss=0）
            # 推理时可启用 use_keep_filter=True 只保留有足够点的 mask
            if self.use_keep_filter:
                pred_probs = torch.sigmoid(mask_logit_valid)
                mask_hard = (pred_probs > self.threshold).float()
                keep = torch.sum(mask_hard, dim=0) > self.min_points_per_mask  # (K_valid,)
                if not keep.any():
                    continue
                pred_logits = mask_logit_valid[:, keep]  # (N_points, K_kept)
            else:
                # 训练时：所有 valid mask 都参与 loss
                keep = torch.ones(mask_logit_valid.shape[1], dtype=torch.bool, device=mask_logit_valid.device)
                pred_logits = mask_logit_valid  # (N_points, K_valid)
            
            # Get GT masks
            mask_2d = self.mask_masks[i][valid]  # (K_valid, H, W)
            point_mask = self.batch_indices == i
            x_idx = self.x_label[point_mask].float()  # 先转 float 以便缩放
            y_idx = self.y_label[point_mask].float()
            
            if x_idx.numel() == 0:
                continue
            
            # mask 尺寸
            H, W = mask_2d.shape[1], mask_2d.shape[2]
            
            # 🔥 关键修复：检查 x_label/y_label 是否需要缩放
            # 如果 x_label 的最大值接近 W，说明已经是 mask 尺寸，不需要缩放
            # 如果 x_label 的最大值接近 640，说明是原图尺寸，需要缩放
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
            
            # 检查越界情况
            valid_mask = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
            num_valid = valid_mask.sum().item()
            total_points = x_idx.numel()
            num_out_of_bounds = total_points - num_valid
            
            # 统计越界占比（每 100 个 batch 报告一次）
            if not hasattr(self, '_oob_stats'):
                self._oob_stats = {'total_oob': 0, 'total_pts': 0, 'total_valid': 0, 'report_count': 0, 'need_scale_count': 0}
            self._oob_stats['total_oob'] += num_out_of_bounds
            self._oob_stats['total_pts'] += total_points
            self._oob_stats['total_valid'] += num_valid
            self._oob_stats['report_count'] += 1
            if need_scale:
                self._oob_stats['need_scale_count'] += 1
            
            if self._oob_stats['report_count'] % 100 == 0:
                oob_ratio = self._oob_stats['total_oob'] / self._oob_stats['total_pts'] * 100
                valid_ratio = self._oob_stats['total_valid'] / self._oob_stats['total_pts'] * 100
                scale_ratio = self._oob_stats['need_scale_count'] / self._oob_stats['report_count'] * 100
                print(f"[OOB Stats] 越界: {oob_ratio:.2f}%, 有效: {valid_ratio:.2f}% "
                      f"({self._oob_stats['total_valid']}/{self._oob_stats['total_pts']} points), "
                      f"需要缩放: {scale_ratio:.1f}%, mask: {H}x{W})")
            
            # 如果所有点都越界或无效，跳过这个 batch
            if num_valid == 0:
                if (x_idx == 0).all() and (y_idx == 0).all():
                    print(f"Warning: All x_label/y_label are 0 for batch {i}, skipping")
                else:
                    print(f"Warning: All points out of bounds for batch {i} after scaling, skipping")
                continue
            
            # 只保留有效点（不使用 clamp）
            x_idx = x_idx[valid_mask]
            y_idx = y_idx[valid_mask]
            pred_logits = pred_logits[valid_mask, :]  # 同步过滤 pred_logits (N_valid, K_kept)
            
            gt_3d = mask_2d[:, y_idx, x_idx]  # (K_valid, N_valid)
            gt_3d = (gt_3d > self.threshold).float()
            gt_3d = gt_3d[keep]  # (K_kept, N_valid)
            gt_3d = gt_3d.transpose(0, 1)  # (N_valid, K_kept)
            
            # 🔥 关键修复 A: 过滤 GT 中正样本数不足的 mask
            # 这是 mIoU 低的主要原因：GT=0 的 mask 会让 BCE 鼓励"全预测 0"
            gt_pos = gt_3d.sum(dim=0)  # (K_kept,) - 每个 mask 的正样本数
            keep_gt = gt_pos >= self.min_points_per_mask
            
            # # 🔥 使用模块级全局变量统计 GT 正样本分布（跨 step 持久化）
            # global _gt_stats_global
            # _gt_stats_global['count'] += 1
            # _gt_stats_global['total_masks'] += gt_pos.numel()
            # _gt_stats_global['zero_masks'] += (gt_pos == 0).sum().item()
            # _gt_stats_global['kept_masks'] += keep_gt.sum().item()
            # _gt_stats_global['pos_sum'] += gt_pos.sum().item()
            
            # 每 200 个 batch item 打印一次（跨 step 累积）
            # if _gt_stats_global['count'] % 200 == 0:
            #     total = _gt_stats_global['total_masks']
            #     zero = _gt_stats_global['zero_masks']
            #     kept = _gt_stats_global['kept_masks']
            #     avg_pos = _gt_stats_global['pos_sum'] / max(total, 1)
            #     zero_ratio = zero / max(total, 1) * 100
            #     kept_ratio = kept / max(total, 1) * 100
            #     print(f"\n[GT Stats] After {_gt_stats_global['count']} batch items: "
            #           f"zero_mask_ratio={zero_ratio:.1f}%, "
            #           f"kept_ratio={kept_ratio:.1f}%, "
            #           f"avg_pos_per_mask={avg_pos:.1f}\n")
            #     # 重置累积统计（开始新一轮 200）
            #     _gt_stats_global = {
            #         'count': 0, 'total_masks': 0, 'zero_masks': 0,
            #         'kept_masks': 0, 'pos_sum': 0
            #     }
            
            # 如果没有任何有效的 GT mask，跳过这个 batch
            if not keep_gt.any():
                continue
            
            # 只保留 GT 正样本数足够的 mask
            pred_logits = pred_logits[:, keep_gt]
            gt_3d = gt_3d[:, keep_gt]
            
            # Compute loss if we have valid masks（BCEWithLogits 兼容 AMP autocast）
            if pred_logits.numel() > 0 and gt_3d.numel() > 0 and pred_logits.shape[1] > 0:
                # 🔥 修复 1: 使用 pos_weight 平衡正负样本
                if self.use_pos_weight:
                    # 重新计算过滤后的正负样本比例
                    pos = gt_3d.sum(dim=0)  # (K_filtered,) - 每个 mask 的正样本数
                    neg = gt_3d.shape[0] - pos  # 负样本数
                    # 此时 pos >= min_points_per_mask，不会出现 pos=0 的情况
                    pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=50.0)
                    bce = F.binary_cross_entropy_with_logits(
                        pred_logits, gt_3d,
                        pos_weight=pos_weight,
                        reduction="mean"
                    )
                else:
                    # 原始 BCE（不推荐用于类别不均衡任务）
                    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_3d, reduction='mean')
                
                # 🔥 修复 2: 使用 per-mask dice loss
                pred_probs_kept = torch.sigmoid(pred_logits)
                if self.use_per_mask_dice:
                    dice = dice_loss_per_mask(pred_probs_kept, gt_3d)
                else:
                    dice = dice_loss(pred_probs_kept, gt_3d)
                
                # Combined loss
                loss = self.bce_weight * bce + self.dice_weight * dice
                losses.append(loss)
        
        if len(losses) == 0:
            # 所有 batch 都被跳过（如 x_label/y_label 全 0、无有效 mask 等），无法计算 loss
            import warnings
            warnings.warn(
                "Criteria: no valid loss (all batches skipped). "
                "Check x_label/y_label in 3D .pth and mask_valid. "
                "Training step will not backward.",
                UserWarning,
                stacklevel=1,
            )
            return torch.tensor(0.0, requires_grad=True, device=self.mask_masks.device)
        return torch.stack(losses).mean()


def loss_pt(outputs, targets):
    """Legacy function for backward compatibility."""
    losses = []
    for batch in range(len(outputs)):
        pred_mask = outputs[batch][0]["pred_mask_3d"]
        gt_mask = targets[batch][0]["masks_3d"]
        bce = F.binary_cross_entropy(pred_mask, gt_mask.float(), reduction='mean')
        dice = dice_loss(pred_mask, gt_mask.float())
        loss = bce + dice
        losses.append(loss)
    return torch.stack(losses).mean()
