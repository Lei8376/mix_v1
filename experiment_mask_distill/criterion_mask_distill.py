"""
Mask Distillation Loss（Diff2Scene 方案）

核心思路（对应论文 Eq.1 & Eq.2）：
  1. 2D hybrid token f_k^{2d} = fused_embeddings[k]
  2. 3D 特征 F^{3d} = pred_3d  (N_points, C)
  3. Logits  S_k = logit_scale * <normalize(F^{3d}), normalize(f_k^{2d})>
     → 直接复用模型前向已算好的 pred_mask_logits，保持归一化和温度一致
  4. 预测 3D 概率 mask  B_k^{3d'} = sigmoid(S_k)
  5. 伪 GT 3D mask B_k^{3d} 由 2D mask 投影到 3D 点
     → masks 实际为 bool 类型，float 后为 {0,1} 的 hard lifted mask
  6. 主损失（按总有效 mask 数平均，对齐论文 Eq.2）:
     L_mask_distill = (1 / K_total) * sum_k (1 - cos(B_k^{3d'}, B_k^{3d}))

只使用 L_mask_distill，不额外加 BCE/Dice（可按需开启辅助项）。

与 experiment_distill/criterion_distill.py 的区别：
  - 旧版：让 pred_3d 逼近 point-level teacher 向量（特征蒸馏）
  - 本版：让 pred_3d 生成的 3D mask 与 lifted 2D mask 保持一致（mask 蒸馏）
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 辅助：构建 lifted 3D mask（soft，直接取 2D 概率值）
# ============================================================

def build_lifted_3d_masks(
    masks: torch.Tensor,          # (B, K_max, H, W) float — 2D soft mask
    mask_valid: torch.Tensor,     # (B, K_max) bool
    x_label: torch.Tensor,        # (N_total,) long — 点的列坐标
    y_label: torch.Tensor,        # (N_total,) long — 点的行坐标
    batch_indices: torch.Tensor,  # (N_total,) long — 点所属 batch
    orig_img_W: int = 640,
    orig_img_H: int = 480,
):
    """
    把 2D soft mask 提升到 3D，对每个点直接读取其投影位置的 2D mask 概率值。

    返回:
        lifted:       (N_total, K_max) float — 每个点对应每个 mask 的软概率
        lifted_valid: (N_total,) bool  — 该点至少有一个有效 mask 命中（坐标在界内）
    """
    B, K_max, H_mask, W_mask = masks.shape
    N_total = batch_indices.shape[0]
    device  = masks.device

    lifted       = torch.zeros(N_total, K_max, dtype=torch.float32, device=device)
    lifted_valid = torch.zeros(N_total, dtype=torch.bool, device=device)

    for b in range(B):
        pt_mask = (batch_indices == b)
        if not pt_mask.any():
            continue

        xi = x_label[pt_mask].float()
        yi = y_label[pt_mask].float()

        # 判断是否需要从原图尺寸缩放到 mask 尺寸
        x_max = xi.max().item() if xi.numel() > 0 else 0
        y_max = yi.max().item() if yi.numel() > 0 else 0
        if (x_max > W_mask + 20) or (y_max > H_mask + 20):
            xi = (xi * W_mask / max(orig_img_W, x_max + 10)).long()
            yi = (yi * H_mask / max(orig_img_H, y_max + 10)).long()
        else:
            xi = xi.long()
            yi = yi.long()

        inbounds = (xi >= 0) & (xi < W_mask) & (yi >= 0) & (yi < H_mask)
        if not inbounds.any():
            continue

        xi_v = xi[inbounds]
        yi_v = yi[inbounds]

        # 有效 mask 切片：(K_valid, H, W)
        valid_k = mask_valid[b]           # (K_max,) bool
        if not valid_k.any():
            continue

        # 对所有 K_max 个 slot 都取值（无效 slot 留 0），方便后续对齐 pred_mask_logits
        # masks_b: (K_max, H, W)
        masks_b = masks[b]                # (K_max, H, W)

        # 逐点采样: (K_max, N_inbounds)
        sampled = masks_b[:, yi_v, xi_v]  # (K_max, N_inbounds)
        sampled = sampled.t()             # (N_inbounds, K_max)

        # 写回
        pt_idx   = pt_mask.nonzero(as_tuple=True)[0]  # (Nb,)
        ib_idx   = pt_idx[inbounds]                    # (N_inbounds,)

        lifted[ib_idx]       = sampled.to(torch.float32)
        lifted_valid[ib_idx] = True

    return lifted, lifted_valid


# ============================================================
# 主类：MaskDistillCriteria
# ============================================================

class MaskDistillCriteria(nn.Module):
    """
    Mask Distillation Loss（Diff2Scene Eq.2）

    主损失：
        L = (1/K) * sum_k [1 - cos(B_k^{3d'}, B_k^{3d})]

    其中：
        B_k^{3d}  = lifted soft mask（2D mask 投影到点云）
        B_k^{3d'} = sigmoid(<F^{3d}, f_k^{2d}>)

    参数：
        mask_distill_weight:  主损失权重（默认 1.0）
        bce_weight:           辅助 BCE 权重（默认 0，不使用）
        dice_weight:          辅助 Dice 权重（默认 0，不使用）
        min_points_per_mask:  GT mask 最少需要多少个正样本才参与 mask loss
        threshold:            辅助 BCE/Dice 里二值化 GT 的阈值
        use_pos_weight:       辅助 BCE 是否用 pos_weight 平衡
    """

    def __init__(
        self,
        results:    dict,
        batch_input: dict,
        mask_distill_weight: float = 1.0,
        bce_weight:          float = 0.0,
        dice_weight:         float = 0.0,
        min_points_per_mask: int   = 10,
        threshold:           float = 0.5,
        use_pos_weight:      bool  = True,
    ):
        super().__init__()

        self.mask_distill_weight = mask_distill_weight
        self.bce_weight          = bce_weight
        self.dice_weight         = dice_weight
        self.min_points_per_mask = min_points_per_mask
        self.threshold           = threshold
        self.use_pos_weight      = use_pos_weight

        # 从 results 取模型输出
        self.outputs               = results["outputs"]
        self.mask_valid_from_masks = results["mask_valid_from_masks"]  # (B, K_max) bool
        self.mask_masks            = results["mask_masks"]              # (B, K_max, H, W)
        self.batch_indices         = results["batch_indices"]           # (N_total,)
        self.fused_embeddings      = results["fused_embeddings"]        # (B, K_max, C)
        self.pred_3d               = results["pred_3d"]                 # (N_total, C)

        # 从 batch_input 取投影坐标
        self.x_label = batch_input["x_label"]   # (N_total,)
        self.y_label = batch_input["y_label"]    # (N_total,)

    # ----------------------------------------------------------
    # 主损失：mask-level cosine distillation
    # ----------------------------------------------------------
    def _mask_distill_loss(self):
        """
        逐 batch item 计算 mask distillation loss。

        核心计算（论文 Eq.1&2，对齐模型前向）：
          S_k = logit_scale * <normalize(F^{3d}), normalize(f_k^{2d})>
              → 直接复用 outputs[b][0]["pred_mask_logits"]，与模型定义完全一致
          B_k^{3d'} = sigmoid(S_k)
          B_k^{3d}  = lifted hard mask（masks 为 bool，float 后 {0,1}）
          L_k = 1 - cos(B_k^{3d'}, B_k^{3d})

        averaging：按所有有效 mask 总数平均（对齐论文 Eq.2 的 sum/N 语义）：
          L = (1 / K_total_valid) * sum_{b,k} L_k
        """
        B = len(self.outputs)
        device = self.pred_3d.device

        # 先用 build_lifted_3d_masks 统一构建所有点的 lifted mask
        lifted, lifted_valid = build_lifted_3d_masks(
            masks        = self.mask_masks,
            mask_valid   = self.mask_valid_from_masks,
            x_label      = self.x_label,
            y_label      = self.y_label,
            batch_indices= self.batch_indices,
        )
        # lifted: (N_total, K_max) float {0,1}

        all_loss  = []   # 收集每个有效 mask 的标量 loss（用于 stack 后统一梯度）

        for b in range(B):
            if len(self.outputs[b]) == 0:
                continue

            valid_k = self.mask_valid_from_masks[b]   # (K_max,) bool
            if not valid_k.any():
                continue

            # 直接取模型前向算好的 logits（已含 normalize + logit_scale）
            # shape: (N_pts_b, K_max)，无效 slot 为 -inf
            pred_logits_full = self.outputs[b][0]["pred_mask_logits"]   # (N_b, K_max)

            # 只取有效 slot
            pred_logits = pred_logits_full[:, valid_k].float()  # (N_b, K_valid)

            # lifted GT mask（仅本 batch item 的点 + 有效 slot）
            pt_mask   = (self.batch_indices == b)
            B3d_gt    = lifted[pt_mask][:, valid_k].float()     # (N_b, K_valid)

            # 过滤在界内的点（inbounds 由 build_lifted_3d_masks 保证有值的点 lifted_valid=True）
            pt_valid  = lifted_valid[pt_mask]                   # (N_b,) bool
            if not pt_valid.any():
                continue

            pred_logits = pred_logits[pt_valid]     # (N_v, K_valid)
            B3d_gt      = B3d_gt[pt_valid]          # (N_v, K_valid)

            # 预测 3D mask
            B3d_pred = torch.sigmoid(pred_logits)   # (N_v, K_valid)

            # 过滤正样本点太少的 mask（GT 全零时 cosine 无意义）
            pos_cnt = (B3d_gt > self.threshold).float().sum(dim=0)  # (K_valid,)
            keep    = pos_cnt >= self.min_points_per_mask            # (K_valid,)
            if not keep.any():
                continue

            B3d_pred_k = B3d_pred[:, keep].t()   # (K_keep, N_v)
            B3d_gt_k   = B3d_gt[:, keep].t()     # (K_keep, N_v)

            # L2 归一化 → cosine（数值稳定）
            pred_norm = F.normalize(B3d_pred_k, dim=1, eps=1e-8)
            gt_norm   = F.normalize(B3d_gt_k,   dim=1, eps=1e-8)
            cos_sim   = (pred_norm * gt_norm).sum(dim=1)   # (K_keep,)

            # 每个有效 mask 对应一个标量 loss，单独追加（不提前 .mean()）
            for v in (1.0 - cos_sim):
                all_loss.append(v)

        if len(all_loss) == 0:
            warnings.warn(
                "MaskDistillCriteria: no valid masks found. "
                "Check mask projections (x_label/y_label/mask_masks).",
                UserWarning,
            )
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 按总有效 mask 数平均（对齐论文 Eq.2）
        return torch.stack(all_loss).mean()

    # ----------------------------------------------------------
    # 辅助损失：BCE + Dice（可选）
    # ----------------------------------------------------------
    def _aux_loss(self):
        """
        可选辅助项，使用 pred_mask_logits 对 lifted 二值 GT 做 BCE+Dice。
        当 bce_weight == 0 且 dice_weight == 0 时直接返回 0。
        """
        if self.bce_weight == 0.0 and self.dice_weight == 0.0:
            return torch.tensor(0.0, device=self.pred_3d.device, requires_grad=False)

        B = len(self.outputs)
        losses = []
        H_full = self.mask_masks.shape[2]
        W_full = self.mask_masks.shape[3]

        for i in range(B):
            if len(self.outputs[i]) == 0:
                continue
            mask_logit = self.outputs[i][0]["pred_mask_logits"]   # (N_pts, K_padded)
            valid      = self.mask_valid_from_masks[i]
            if not valid.any():
                continue

            mask_logit_v = mask_logit[:, valid]                   # (N_pts, K_valid)
            mask_2d      = self.mask_masks[i][valid]              # (K_valid, H, W)
            H, W = mask_2d.shape[1], mask_2d.shape[2]

            pt_mask = self.batch_indices == i
            xi = self.x_label[pt_mask].float()
            yi = self.y_label[pt_mask].float()
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

            xi = xi[inbounds]; yi = yi[inbounds]
            pred_logits = mask_logit_v[inbounds, :]                # (N_v, K_valid)

            gt_3d = mask_2d[:, yi, xi]                             # (K_valid, N_v)
            gt_3d = (gt_3d > self.threshold).float().t()           # (N_v, K_valid)

            gt_pos  = gt_3d.sum(dim=0)
            keep_gt = gt_pos >= self.min_points_per_mask
            if not keep_gt.any():
                continue

            pred_logits = pred_logits[:, keep_gt]
            gt_3d       = gt_3d[:, keep_gt]
            if pred_logits.numel() == 0:
                continue

            bce = torch.tensor(0.0, device=pred_logits.device)
            if self.bce_weight > 0:
                if self.use_pos_weight:
                    pos = gt_3d.sum(dim=0)
                    neg = gt_3d.shape[0] - pos
                    pw  = (neg / (pos + 1e-6)).clamp(min=1.0, max=50.0)
                    bce = F.binary_cross_entropy_with_logits(
                        pred_logits, gt_3d, pos_weight=pw, reduction="mean"
                    )
                else:
                    bce = F.binary_cross_entropy_with_logits(
                        pred_logits, gt_3d, reduction="mean"
                    )

            dice = torch.tensor(0.0, device=pred_logits.device)
            if self.dice_weight > 0:
                prob = torch.sigmoid(pred_logits)
                inter  = (prob * gt_3d).sum(dim=0)
                denom  = prob.sum(dim=0) + gt_3d.sum(dim=0)
                dice_k = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
                dice   = dice_k.mean()

            losses.append(self.bce_weight * bce + self.dice_weight * dice)

        if not losses:
            return torch.tensor(0.0, device=self.pred_3d.device, requires_grad=True)
        return torch.stack(losses).mean()
        # return torch.stack(losses).sum()

    # ----------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------
    def compute_loss(self):
        """
        总损失：
            L = mask_distill_weight * L_mask_distill
              + bce_weight * L_bce  (默认 0)
              + dice_weight * L_dice (默认 0)

        返回 (total_loss, loss_dict)
        """
        l_mask_distill = self._mask_distill_loss()
        l_aux          = self._aux_loss()

        total = self.mask_distill_weight * l_mask_distill + l_aux

        loss_dict = {
            "loss_mask_distill": l_mask_distill.item(),
            "loss_aux":          l_aux.item() if isinstance(l_aux, torch.Tensor) else float(l_aux),
            "loss_total":        total.item(),
        }
        return total, loss_dict
