"""
语义 mIoU 计算模块（OpenScene 风格）

流程：
  1. 用冻结的 CLIP ViT-L/14 对 20 个 ScanNet 类别编码成文本特征 T (20, 768)
  2. 对每个 3D 点：pred_class = argmax(normalize(pred_3d[i]) @ T.T)
  3. 和 GT label 对比，按类别计算 IoU，取平均

注意：
  - GT label 是 nyu40id（1-40），需要映射到 0-19 的 20 类索引
  - ignore_label=0（unlabeled/unknown）的点不计入评估
  - 文本特征在第一次调用时构建，之后缓存（避免重复 encode）
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# ScanNet 20 类标签名（按 nyu40 类别顺序）
SCANNET_LABELS_20 = (
    'wall', 'floor', 'cabinet', 'bed', 'chair',
    'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator',
    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture',
)

# nyu40id → 0-based 20类索引
# ScanNet 用的 nyu40 label，有效的 20 个类别 id：
NYU40_VALID_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
# 构建映射表：nyu40id → 0-19；其余 → -1 (ignore)
_NYU40_TO_20 = {nid: i for i, nid in enumerate(NYU40_VALID_IDS)}


def nyu40_to_20(labels: torch.Tensor) -> torch.Tensor:
    """
    把 nyu40 label tensor 映射到 0-19 的 20 类索引。
    不在 20 类里的（包括 0=unlabeled）映射为 -1。
    labels: (N,) long
    return: (N,) long，值域 [-1, 19]
    """
    out = torch.full_like(labels, -1)
    for nid, idx in _NYU40_TO_20.items():
        out[labels == nid] = idx
    return out


def build_text_features(device: str = "cuda", clip_model: str = "ViT-L/14") -> torch.Tensor:
    """
    用 CLIP 编码 ScanNet 20 类文本，返回 (20, 768) 归一化 float32 tensor。
    需要能 import clip（已在 mix 环境中安装）。
    """
    import clip
    model, _ = clip.load(clip_model, device=device)
    model.eval()

    prompts = [f"a {label} in a room" for label in SCANNET_LABELS_20]
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        text_feats = model.encode_text(tokens).float()          # (20, 768)
        text_feats = F.normalize(text_feats, dim=-1)

    del model
    torch.cuda.empty_cache()
    return text_feats  # (20, 768)


class SemanticMIoUTracker:
    """
    增量式语义 mIoU 统计，支持多 batch 累积后统一 compute()。

    使用方式：
        tracker = SemanticMIoUTracker(text_features)
        for batch in val_loader:
            tracker.update(pred_3d, gt_labels)   # pred_3d: (N, 768), gt_labels: (N,) nyu40
        result = tracker.compute()
        print(result["semantic_miou"])
    """

    NUM_CLASSES = 20

    def __init__(self, text_features: torch.Tensor):
        """
        text_features: (20, 768) 归一化 CLIP 文本特征，在 GPU/CPU 上均可
        """
        self.text_features = text_features   # (20, 768)
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.NUM_CLASSES, dtype=np.float64)
        self.union        = np.zeros(self.NUM_CLASSES, dtype=np.float64)
        self.n_points_per_class = np.zeros(self.NUM_CLASSES, dtype=np.int64)

    @torch.no_grad()
    def update(
        self,
        pred_3d: torch.Tensor,    # (N, 768) — 模型输出，未归一化也可
        gt_labels: torch.Tensor,  # (N,) — nyu40 label（整数）
    ):
        """
        累积一个 batch 的统计。
        pred_3d 和 gt_labels 必须对应同一组点。
        """
        # 把 gt 映射到 0-19
        gt_20 = nyu40_to_20(gt_labels.cpu())     # (N,) 值域 [-1, 19]
        valid = gt_20 >= 0                        # 过滤 ignore 点
        if not valid.any():
            return

        pred_valid = pred_3d[valid].float()       # (N_valid, 768)
        gt_valid   = gt_20[valid]                 # (N_valid,) 0-19

        # cos 相似度分类
        pred_norm = F.normalize(pred_valid, dim=-1)
        tf = self.text_features.to(pred_norm.device)
        sim = pred_norm @ tf.T                    # (N_valid, 20)
        pred_class = sim.argmax(dim=-1).cpu()     # (N_valid,)

        # 按类别统计
        for c in range(self.NUM_CLASSES):
            gt_c   = (gt_valid   == c)
            pred_c = (pred_class == c)
            inter  = (gt_c & pred_c).sum().item()
            union  = (gt_c | pred_c).sum().item()
            self.intersection[c] += inter
            self.union[c]        += union
            self.n_points_per_class[c] += gt_c.sum().item()

    def compute(self) -> Dict[str, float]:
        """
        计算最终语义 mIoU 和每类 IoU。
        只对出现过（GT 有点）的类别取平均。
        """
        ious = {}
        for c in range(self.NUM_CLASSES):
            if self.union[c] > 0:
                ious[SCANNET_LABELS_20[c]] = float(
                    self.intersection[c] / (self.union[c] + 1e-10)
                )
        miou = float(np.mean(list(ious.values()))) if ious else 0.0
        return {
            "semantic_miou":  miou,
            "per_class_iou":  ious,
            "n_valid_classes": len(ious),
        }
