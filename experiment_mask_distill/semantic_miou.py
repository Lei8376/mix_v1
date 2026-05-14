"""Compatibility entry for semantic IoU evaluation.

The implementation lives in evaluate.semantic_iou so training, validation, and
standalone evaluation use the same Diff2Scene Eq.3 metric code.
"""

from evaluate.semantic_iou import (
    IGNORE_LABEL,
    SCANNET_LABELS_20,
    AverageMeter,
    Diff2SceneSemanticEvaluator,
    Diff2SceneSemanticMIoUTracker,
    ODISEPCSemanticMIoUTracker,
    SemanticMIoUTracker,
    XMask3DSemanticEvaluator,
    build_text_features,
    canonical_prompt_label,
    compute_iou,
    diff2scene_class_probs_predict,
    diff2scene_dual_mask_class_probs_predict,
    mask_feature_class_probs,
    diff2scene_mask_feature_predict,
    diff2scene_point_class_probs,
    odise_geometric_mask_feature_predict,
    odise_geometric_dual_mask_predict,
    diff2scene_point_class_scores,
    intersectionAndUnion,
    intersectionAndUnionGPU,
)


class MaskMIoUTracker:
    """Mask-level IoU helper. This is not semantic IoU."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._iou_sum = 0.0
        self._iou_count = 0

    def update(self, pred_mask_prob, gt_mask_soft, valid_k):
        import torch

        if not valid_k.any():
            return

        with torch.no_grad():
            pred = (pred_mask_prob[:, valid_k] > self.threshold).float()
            gt = (gt_mask_soft[:, valid_k] > self.threshold).float()
            inter = (pred * gt).sum(dim=0)
            union = (pred + gt - pred * gt).sum(dim=0)
            has_gt = gt.sum(dim=0) > 0
            if not has_gt.any():
                return

            iou = inter[has_gt] / (union[has_gt] + 1e-6)
            self._iou_sum += iou.sum().item()
            self._iou_count += has_gt.sum().item()

    def compute(self):
        if self._iou_count == 0:
            return {"mask_miou": 0.0, "n_masks": 0}
        return {
            "mask_miou": self._iou_sum / self._iou_count,
            "n_masks": self._iou_count,
        }
