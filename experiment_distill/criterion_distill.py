"""
Point-level Hybrid Teacher Distillation

核心思路：
  旧版：让 pred_3d 学会判断"点属于哪个 mask-slot"（BCE+Dice on pred_mask_logits）
  新版：让 pred_3d 直接逼近 point-level hybrid teacher 向量（cosine distillation）
        + 小权重的 mask BCE+Dice 作为几何辅助约束

主损失：
  L_feat = mean(1 - cos(pred_3d[i], teacher[i]))  对所有有 teacher 的点

辅助损失（旧 BCE+Dice 降权）：
  L_mask = bce_weight * BCE + dice_weight * Dice
  权重由 mask_loss_weight 控制，默认 0.1

总损失：
  L = L_feat + mask_loss_weight * L_mask


"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 辅助函数
# ============================================================

def dice_loss_per_mask(pred, target, smooth=1.0):
    """
    Per-mask Dice loss，再对 mask 维度取均值。
    pred:   (N_points, K) sigmoid 后的概率
    target: (N_points, K) 0/1 二值 GT
    """
    inter = (pred * target).sum(dim=0)           # (K,)
    denom = pred.sum(dim=0) + target.sum(dim=0)  # (K,)
    dice_per = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice_per.mean()


def build_point_teacher(
    fused_embeddings: torch.Tensor,   # (B, K_max, C)
    mask_valid: torch.Tensor,         # (B, K_max) bool
    masks: torch.Tensor,              # (B, K_max, H, W) float
    x_label: torch.Tensor,            # (N_total,) long — 点的列坐标
    y_label: torch.Tensor,            # (N_total,) long — 点的行坐标
    batch_indices: torch.Tensor,      # (N_total,) long — 点所属 batch
    threshold: float = 0.5,
    orig_img_W: int = 640,
    orig_img_H: int = 480,
):
    """
    为每个 3D 点构造 point-level hybrid teacher 向量。

    对点 i（属于 batch b），找出所有它落入的有效 mask k，
    把对应的 fused_embeddings[b, k] 做加权平均（等权），
    得到该点的 teacher 向量 t_i。

    返回：
        teacher:      (N_total, C) float32，无有效 mask 的点为全零
        teacher_valid:(N_total,) bool，True 表示该点有至少一个 mask 命中
    """
    B, K_max, C = fused_embeddings.shape
    N_total = batch_indices.shape[0]
    device  = fused_embeddings.device

    teacher       = torch.zeros(N_total, C, dtype=torch.float32, device=device)
    teacher_valid = torch.zeros(N_total, dtype=torch.bool,       device=device)

    H = masks.shape[2]
    W = masks.shape[3]

    for b in range(B):
        point_mask = (batch_indices == b)
        if not point_mask.any():
            continue

        xi = x_label[point_mask].float()
        yi = y_label[point_mask].float()

        # 判断是否需要从原图尺寸缩放到 mask 尺寸
        x_max = xi.max().item() if xi.numel() > 0 else 0
        y_max = yi.max().item() if yi.numel() > 0 else 0
        if (x_max > W + 20) or (y_max > H + 20):
            xi = (xi * W / max(orig_img_W, x_max + 10)).long()
            yi = (yi * H / max(orig_img_H, y_max + 10)).long()
        else:
            xi = xi.long()
            yi = yi.long()

        # 越界过滤
        inbounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if not inbounds.any():
            continue

        xi_valid = xi[inbounds]
        yi_valid = yi[inbounds]

        # 有效 mask 的 index
        valid_k = mask_valid[b]         # (K_max,) bool
        if not valid_k.any():
            continue

        # masks_b: (K_valid, H, W)
        masks_b  = masks[b][valid_k]    # (K_valid, H, W)
        embeds_b = fused_embeddings[b][valid_k]  # (K_valid, C)

        # 查询每个点落在哪些 mask 里
        # membership: (K_valid, N_inbounds) bool
        membership = masks_b[:, yi_valid, xi_valid] > threshold  # (K_valid, N_inbounds)

        # 对有命中的点做加权平均（等权）
        # hit_count: (N_inbounds,)
        hit_count = membership.float().sum(dim=0)  # (N_inbounds,)
        has_hit   = hit_count > 0                  # (N_inbounds,) bool

        if not has_hit.any():
            continue

        # weighted sum: (N_inbounds, C)  = membership.T (N_in, K_v) @ embeds_b (K_v, C)
        weighted_sum = membership.float().t() @ embeds_b  # (N_inbounds, C)
        t_i = weighted_sum / hit_count.unsqueeze(-1).clamp(min=1.0)  # (N_inbounds, C)

        # 写回
        # point_mask 里 inbounds 子集
        point_mask_idx  = point_mask.nonzero(as_tuple=True)[0]  # (Nb,)
        inbounds_idx    = point_mask_idx[inbounds]               # (N_inbounds,)
        has_hit_idx     = inbounds_idx[has_hit]                  # (N_hit,)

        teacher[has_hit_idx]       = t_i[has_hit].to(torch.float32)
        teacher_valid[has_hit_idx] = True

    return teacher, teacher_valid


# ============================================================
# 主类：DistillCriteria
# ============================================================

class DistillCriteria(nn.Module):
    """
    新版损失：point-level hybrid teacher distillation。

    参数：
        feat_loss_weight:  L_feat 的权重（主损失，默认 1.0）
        mask_loss_weight:  L_mask 的权重（辅助，默认 0.1）
        bce_weight:        L_mask 内部 BCE 权重
        dice_weight:       L_mask 内部 Dice 权重
        min_points_per_mask: GT mask 最少需要多少个正样本点才参与 mask loss
        threshold:         mask 二值化阈值
        use_pos_weight:    mask loss 是否用 pos_weight 平衡正负样本
    """

    def __init__(
        self,
        results: dict,
        batch_input: dict,
        feat_loss_weight:  float = 1.0,
        mask_loss_weight:  float = 0.1,
        bce_weight:        float = 1.0,
        dice_weight:       float = 1.0,
        min_points_per_mask: int = 10,
        threshold:         float = 0.5,
        use_pos_weight:    bool  = True,
    ):
        super().__init__()

        self.feat_loss_weight   = feat_loss_weight
        self.mask_loss_weight   = mask_loss_weight
        self.bce_weight         = bce_weight
        self.dice_weight        = dice_weight
        self.min_points_per_mask = min_points_per_mask
        self.threshold          = threshold
        self.use_pos_weight     = use_pos_weight

        # 从 results 取出模型输出
        self.outputs              = results["outputs"]
        self.mask_valid_from_masks = results["mask_valid_from_masks"]
        self.mask_masks           = results["mask_masks"]
        self.batch_indices        = results["batch_indices"]
        self.fused_embeddings     = results["fused_embeddings"]
        self.pred_3d              = results["pred_3d"]

        # 从 batch_input 取投影坐标
        self.x_label = batch_input["x_label"]
        self.y_label = batch_input["y_label"]

    # ----------------------------------------------------------
    # 主损失：feature distillation
    # ----------------------------------------------------------
    def _feat_loss(self):
        """
        L_feat = mean(1 - cos(pred_3d[i], teacher[i]))
        只对 teacher_valid=True 的点计算。
        """
        teacher, teacher_valid = build_point_teacher(
            fused_embeddings = self.fused_embeddings,
            mask_valid       = self.mask_valid_from_masks,
            masks            = self.mask_masks,
            x_label          = self.x_label,
            y_label          = self.y_label,
            batch_indices    = self.batch_indices,
            threshold        = self.threshold,
        )

        if not teacher_valid.any():
            warnings.warn(
                "DistillCriteria: no points have a valid teacher. "
                "Check mask projection (x_label/y_label) and mask_valid.",
                UserWarning,
            )
            return torch.tensor(0.0, requires_grad=True,
                                device=self.pred_3d.device)

        s = self.pred_3d[teacher_valid]   # (N_valid, C)
        t = teacher[teacher_valid]         # (N_valid, C)

        # cosine distillation: L = 1 - cos(s, t)
        loss = 1.0 - F.cosine_similarity(s, t, dim=-1)  # (N_valid,)
        return loss.mean()

    # ----------------------------------------------------------
    # 辅助损失：mask BCE + Dice（旧 criterion 逻辑，权重降低）
    # ----------------------------------------------------------
    def _mask_loss(self):
        """
        保留旧的 BCE+Dice，作为几何辅助约束。
        逻辑与原 Criteria.loss_pt() 相同，不赘述。
        """
        batch  = len(self.outputs)
        losses = []

        H_full = self.mask_masks.shape[2]
        W_full = self.mask_masks.shape[3]

        for i in range(batch):
            if len(self.outputs[i]) == 0:
                continue

            mask_logit = self.outputs[i][0]["pred_mask_logits"]  # (N_pts, K_padded)
            valid = self.mask_valid_from_masks[i]
            if not valid.any():
                continue

            mask_logit_valid = mask_logit[:, valid]  # (N_pts, K_valid)

            mask_2d   = self.mask_masks[i][valid]    # (K_valid, H, W)
            H, W = mask_2d.shape[1], mask_2d.shape[2]

            point_mask = self.batch_indices == i
            xi = self.x_label[point_mask].float()
            yi = self.y_label[point_mask].float()
            if xi.numel() == 0:
                continue

            x_max = xi.max().item()
            y_max = yi.max().item()
            if (x_max > W + 20) or (y_max > H + 20):
                xi = (xi * W / max(640, x_max + 10)).long()
                yi = (yi * H / max(480, y_max + 10)).long()
            else:
                xi = xi.long()
                yi = yi.long()

            inbounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            if not inbounds.any():
                continue

            xi = xi[inbounds]
            yi = yi[inbounds]
            pred_logits = mask_logit_valid[inbounds, :]  # (N_valid, K_valid)

            gt_3d = mask_2d[:, yi, xi]                   # (K_valid, N_valid)
            gt_3d = (gt_3d > self.threshold).float().t() # (N_valid, K_valid)

            gt_pos   = gt_3d.sum(dim=0)                  # (K_valid,)
            keep_gt  = gt_pos >= self.min_points_per_mask
            if not keep_gt.any():
                continue

            pred_logits = pred_logits[:, keep_gt]
            gt_3d       = gt_3d[:, keep_gt]

            if pred_logits.numel() == 0:
                continue

            if self.use_pos_weight:
                pos = gt_3d.sum(dim=0)
                neg = gt_3d.shape[0] - pos
                pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=50.0)
                bce = F.binary_cross_entropy_with_logits(
                    pred_logits, gt_3d, pos_weight=pos_weight, reduction="mean"
                )
            else:
                bce = F.binary_cross_entropy_with_logits(
                    pred_logits, gt_3d, reduction="mean"
                )

            pred_probs = torch.sigmoid(pred_logits)
            dice = dice_loss_per_mask(pred_probs, gt_3d)

            losses.append(self.bce_weight * bce + self.dice_weight * dice)

        if not losses:
            return torch.tensor(0.0, requires_grad=True,
                                device=self.pred_3d.device)
        return torch.stack(losses).mean()

    # ----------------------------------------------------------
    # 对外接口：compute_loss()
    # ----------------------------------------------------------
    def compute_loss(self):
        """
        总损失：
            L = feat_loss_weight * L_feat + mask_loss_weight * L_mask

        返回 (total_loss, loss_dict)
        loss_dict 包含各项 loss 的标量值，用于 TensorBoard 记录。
        """
        l_feat = self._feat_loss()
        l_mask = self._mask_loss()

        total = self.feat_loss_weight * l_feat + self.mask_loss_weight * l_mask

        loss_dict = {
            "loss_feat": l_feat.item(),
            "loss_mask": l_mask.item(),
            "loss_total": total.item(),
        }
        return total, loss_dict
