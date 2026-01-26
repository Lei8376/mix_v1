import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.0):
    """
    Compute Dice loss for binary masks.
    pred: (N, K) - soft predictions (sigmoid output)
    target: (N, K) - binary targets
    """
    pred = pred.contiguous().reshape(-1)
    target = target.contiguous().reshape(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


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
                 bce_weight=1.0, dice_weight=1.0):
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
            
            # Filter to only valid masks
            mask_logit_valid = mask_logit[:, valid]  # (N_points, K_valid)
            
            # Compute keep mask based on hard threshold (for filtering)
            mask_hard = (mask_logit_valid > self.threshold).float()
            keep = torch.sum(mask_hard, dim=0) > self.min_points_per_mask  # (K_valid,)
            
            # Skip if no masks pass the filter
            if not keep.any():
                continue
            
            # Get soft predictions for kept masks (DIFFERENTIABLE)
            pred_soft = mask_logit_valid[:, keep]  # (N_points, K_kept)
            
            # Get GT masks
            mask_2d = self.mask_masks[i][valid]  # (K_valid, H, W)
            point_mask = self.batch_indices == i
            x_idx = self.x_label[point_mask].long()
            y_idx = self.y_label[point_mask].long()
            
            if x_idx.numel() == 0:
                continue
                
            if x_idx.max().item() >= mask_2d.shape[2] or y_idx.max().item() >= mask_2d.shape[1]:
                continue
            
            gt_3d = mask_2d[:, y_idx, x_idx]  # (K_valid, N_points)
            gt_3d = (gt_3d > self.threshold).float()
            gt_3d = gt_3d[keep]  # (K_kept, N_points)
            gt_3d = gt_3d.transpose(0, 1)  # (N_points, K_kept)
            
            # Compute loss if we have valid masks
            if pred_soft.numel() > 0 and gt_3d.numel() > 0 and pred_soft.shape[1] > 0:
                # BCE Loss
                bce = F.binary_cross_entropy(pred_soft, gt_3d, reduction='mean')
                
                # Dice Loss (per mask, then average)
                dice = dice_loss(pred_soft, gt_3d)
                
                # Combined loss
                loss = self.bce_weight * bce + self.dice_weight * dice
                losses.append(loss)
        
        if len(losses) == 0:
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
