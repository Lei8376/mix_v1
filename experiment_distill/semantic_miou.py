"""Compatibility entry for Diff2Scene-style semantic IoU evaluation."""

from evaluate.semantic_iou import (
    IGNORE_LABEL,
    SCANNET_LABELS_20,
    AverageMeter,
    Diff2SceneSemanticEvaluator,
    Diff2SceneSemanticMIoUTracker,
    SemanticMIoUTracker,
    XMask3DSemanticEvaluator,
    build_text_features,
    canonical_prompt_label,
    compute_iou,
    diff2scene_mask_feature_predict,
    diff2scene_point_class_scores,
    intersectionAndUnion,
    intersectionAndUnionGPU,
)
