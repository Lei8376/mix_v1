"""Diff2Scene-style open-vocabulary semantic IoU evaluation.

The semantic prediction follows Zhu et al., "Open-Vocabulary 3D Semantic
Segmentation with Text-to-Image Diffusion Models":
  1. Use each 2D mask embedding as a semantic query/classifier.
  2. Predict 3D mask probabilities B_i^3d for every point and mask i.
  3. Compute each mask's label probability p_i^c against the requested label set.
  4. Assign point label probabilities with p^c(point) = sum_i B_i^3d(point) p_i^c.

This file intentionally does not use the old OpenScene-style semantic metric
that classifies pred_3d directly with text embeddings; this project predicts
semantic labels through mask queries.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


SCANNET_LABELS_20: Tuple[str, ...] = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

IGNORE_LABEL = 255
CLIP_LOGIT_SCALE = 100.0


def canonical_prompt_label(label: str) -> str:
    """Match OpenScene/XMask3D ScanNet prompt handling."""
    return "other" if label == "otherfurniture" else label


def build_text_features(
    class_names: Sequence[str] = SCANNET_LABELS_20,
    prompt_template: str = "a {} in a scene",
    clip_model: str = "ViT-B/32",
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Extract normalized CLIP text features using OpenScene-style prompts."""
    if clip_model in {"ODISE-256", "ODISE", "ODISE-ViT-L/14"}:
        return build_odise_256_text_features(
            class_names=class_names,
            prompt_template=prompt_template,
            device=device,
        )

    prompts = [
        prompt_template.format(canonical_prompt_label(label))
        for label in class_names
    ]

    try:
        import clip

        kwargs = {"device": device, "jit": False}
        cache_dir = os.environ.get("CLIP_CACHE_DIR")
        if cache_dir:
            try:
                model, _ = clip.load(clip_model, download_root=cache_dir, **kwargs)
            except TypeError:
                model, _ = clip.load(clip_model, **kwargs)
        else:
            model, _ = clip.load(clip_model, **kwargs)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            text = clip.tokenize(prompts).to(device)
            text_features = model.encode_text(text).float()
            text_features = F.normalize(text_features, dim=-1)
        del model
        return text_features
    except ImportError:
        import open_clip

        name_map = {
            "ViT-L/14": "ViT-L-14",
            "ViT-L/14@336px": "ViT-L-14-336",
            "ViT-B/32": "ViT-B-32",
            "ViT-B/16": "ViT-B-16",
        }
        model_name = name_map.get(clip_model, clip_model)
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai",
            device=device,
        )
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            text = open_clip.tokenize(prompts).to(device)
            text_features = model.encode_text(text).float()
            text_features = F.normalize(text_features, dim=-1)
        del model
        return text_features


def _resolve_odise_checkpoint(config_name: str = "Panoptic/odise_caption_coco_50e") -> Path:
    """Resolve the local ODISE checkpoint cache used for word_head.text_proj."""
    config_name = config_name.replace(".py", "").replace(".yaml", "")
    checkpoint_names = {
        "Panoptic/odise_caption_coco_50e": "odise_caption_coco_50e-853cc971.pth",
        "Panoptic/odise_label_coco_50e": "odise_label_coco_50e-b67d2efc.pth",
    }
    if config_name not in checkpoint_names:
        raise RuntimeError(f"Unsupported ODISE text-head config: {config_name}")
    filename = checkpoint_names[config_name]
    candidates = [
        Path(__file__).resolve().parents[1] / "checkpoints" / "pretrained" / filename,
        Path(__file__).resolve().parents[1] / "checkpoints" / "pretrained" / "odise" / filename,
        Path.home()
        / ".torch"
        / "iopath_cache"
        / "NVlabs"
        / "ODISE"
        / "releases"
        / "download"
        / "v1.0.0"
        / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_odise_256_text_features(
    class_names: Sequence[str] = SCANNET_LABELS_20,
    prompt_template: str = "a photo of a {}",
    device: str | torch.device = "cuda",
    odise_model_config: str = "Panoptic/odise_caption_coco_50e",
) -> torch.Tensor:
    """Build normalized 256D text features with ODISE word_head.text_proj.

    This is the matching text reader for ODISE raw 256D mask embeddings and the
    new 256D fused space.
    """
    import open_clip

    device = torch.device(device)
    checkpoint_path = _resolve_odise_checkpoint(odise_model_config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"ODISE checkpoint not found in local cache: {checkpoint_path}. "
            "Run an ODISE script once or place the checkpoint there before validation."
        )
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    weight = state_dict["word_head.text_proj.weight"].float().to(device)
    bias = state_dict["word_head.text_proj.bias"].float().to(device)

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device=device,
    )
    model = model.to(device)
    model.eval()

    prompts = [
        prompt_template.format(canonical_prompt_label(label))
        for label in class_names
    ]
    with torch.no_grad():
        tokens = open_clip.tokenize(prompts).to(device)
        text_features = model.encode_text(tokens).float()
        text_features = text_features @ weight.t() + bias
        text_features = F.normalize(text_features, dim=-1)
    del model
    return text_features


class AverageMeter:
    """Same accumulator pattern used by OpenScene/XMask3D."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _as_ignore_list(ignore_index: int | Iterable[int]) -> Tuple[int, ...]:
    if isinstance(ignore_index, int):
        return (ignore_index,)
    return tuple(int(v) for v in ignore_index)


def intersectionAndUnion(
    output: np.ndarray,
    target: np.ndarray,
    K: int,
    ignore_index: int | Iterable[int] = IGNORE_LABEL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenScene/XMask3D numpy IoU histogram.

    Values outside [0, K - 1], including unsupported label 255, do not
    contribute to predicted area. If their GT is valid, the GT class remains in
    the union, so unsupported predictions are penalized correctly.
    """
    assert output.ndim in [1, 2, 3, 4]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    for ignore in _as_ignore_list(ignore_index):
        output[np.where(target == ignore)[0]] = ignore
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(
    output: torch.Tensor,
    target: torch.Tensor,
    K: int,
    ignore_index: int | Iterable[int] = IGNORE_LABEL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """OpenScene/XMask3D torch IoU histogram without forcing CUDA output."""
    assert output.dim() in [1, 2, 3, 4]
    assert output.shape == target.shape
    output = output.view(-1).clone()
    target = target.view(-1).clone()
    for ignore in _as_ignore_list(ignore_index):
        output[target == ignore] = ignore
    intersection = output[output == target]
    area_intersection = torch.histc(
        intersection.float().cpu(), bins=K, min=0, max=K - 1
    )
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def compute_iou(
    intersection: np.ndarray | torch.Tensor,
    union: np.ndarray | torch.Tensor,
    target: Optional[np.ndarray | torch.Tensor] = None,
    class_names: Sequence[str] = SCANNET_LABELS_20,
    eps: float = 1e-10,
) -> Dict:
    """Convert accumulated OpenScene histograms to mIoU/mAcc and per-class stats."""
    inter = torch.as_tensor(intersection, dtype=torch.float64).cpu().numpy()
    uni = torch.as_tensor(union, dtype=torch.float64).cpu().numpy()
    tgt = None if target is None else torch.as_tensor(target).cpu().numpy()

    valid = uni > 0
    per_class_iou = {
        class_names[i]: float(inter[i] / (uni[i] + eps))
        for i in range(min(len(class_names), len(uni)))
        if valid[i]
    }
    if tgt is not None:
        valid_acc = tgt > 0
        per_class_acc = {
            class_names[i]: float(inter[i] / (tgt[i] + eps))
            for i in range(min(len(class_names), len(tgt)))
            if valid_acc[i]
        }
    else:
        valid_acc = np.zeros_like(valid, dtype=bool)
        per_class_acc = {}
    result = {
        "miou": float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0,
        "macc": float(np.mean(list(per_class_acc.values()))) if per_class_acc else 0.0,
        "per_class_iou": per_class_iou,
        "per_class_acc": per_class_acc,
        "n_valid_classes": int(valid.sum()),
        "n_valid_classes_acc": int(valid_acc.sum()),
    }
    if tgt is not None:
        result["target"] = {
            class_names[i]: int(tgt[i])
            for i in range(min(len(class_names), len(tgt)))
            if tgt[i] > 0
        }
    return result


@torch.no_grad()
def diff2scene_point_class_scores(
    point_mask_logits: torch.Tensor,
    mask_class_probs: torch.Tensor,
    chunk_size: int = 50000,
) -> torch.Tensor:
    """Compute Eq.3 point class probabilities.

    Args:
        point_mask_logits: (N, I) logits for I predicted 3D masks.
        mask_class_probs: (I, C) per-mask label probabilities over C labels.

    Returns:
        (N, C) point label scores p^c(point) = sum_i sigmoid(S_i) p_i^c.
    """
    if point_mask_logits.ndim != 2:
        raise RuntimeError(
            f"point_mask_logits must be (N,I), got {tuple(point_mask_logits.shape)}"
        )
    if mask_class_probs.ndim != 2:
        raise RuntimeError(
            f"mask_class_probs must be (I,C), got {tuple(mask_class_probs.shape)}"
        )
    if point_mask_logits.shape[1] != mask_class_probs.shape[0]:
        raise RuntimeError(
            f"mask count mismatch: logits I={point_mask_logits.shape[1]}, "
            f"mask_class_probs I={mask_class_probs.shape[0]}"
        )

    if point_mask_logits.shape[0] == 0:
        return torch.empty(
            0,
            mask_class_probs.shape[1],
            dtype=mask_class_probs.dtype,
            device=point_mask_logits.device,
        )
    if mask_class_probs.shape[0] == 0:
        return torch.zeros(
            point_mask_logits.shape[0],
            mask_class_probs.shape[1],
            dtype=mask_class_probs.dtype,
            device=point_mask_logits.device,
        )

    device = point_mask_logits.device
    probs = mask_class_probs.float().to(device)
    scores = []
    for start in range(0, point_mask_logits.shape[0], chunk_size):
        end = min(start + chunk_size, point_mask_logits.shape[0])
        point_to_masks = torch.sigmoid(point_mask_logits[start:end].float())
        scores.append(point_to_masks @ probs)
    return torch.cat(scores, dim=0)


def diff2scene_point_class_probs(
    point_mask_logits: torch.Tensor,
    mask_class_probs: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Eq.3 point class probabilities from mask class probs."""
    if point_mask_logits.ndim != 2:
        raise RuntimeError(
            f"point_mask_logits must be (N,K), got {tuple(point_mask_logits.shape)}"
        )
    if mask_class_probs.ndim != 2:
        raise RuntimeError(
            f"mask_class_probs must be (K,C), got {tuple(mask_class_probs.shape)}"
        )
    if point_mask_logits.shape[1] != mask_class_probs.shape[0]:
        raise RuntimeError(
            f"mask count mismatch: logits K={point_mask_logits.shape[1]}, "
            f"mask_class_probs K={mask_class_probs.shape[0]}"
        )
    point_mask_prob = torch.sigmoid(point_mask_logits.float())
    point_class_scores = point_mask_prob @ mask_class_probs.float()
    point_class_probs = point_class_scores / point_class_scores.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-6)
    return point_class_probs


@torch.no_grad()
def mask_feature_class_probs(
    mask_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float = CLIP_LOGIT_SCALE,
) -> torch.Tensor:
    """Per-mask text probabilities p(z_i, C) from ODISE Eq.6."""
    if mask_features.ndim != 2:
        raise RuntimeError(f"mask_features must be (K,D), got {tuple(mask_features.shape)}")
    if text_features.ndim != 2:
        raise RuntimeError(f"text_features must be (C,D), got {tuple(text_features.shape)}")
    if mask_features.shape[1] != text_features.shape[1]:
        raise RuntimeError(
            f"feature dim mismatch: mask={mask_features.shape[1]}, text={text_features.shape[1]}"
        )
    if mask_features.shape[0] == 0:
        return torch.empty(
            0,
            text_features.shape[0],
            dtype=text_features.dtype,
            device=text_features.device,
        )

    device = mask_features.device
    logits = logit_scale * (
        F.normalize(mask_features.float(), dim=-1)
        @ F.normalize(text_features.float().to(device), dim=-1).t()
    )
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def diff2scene_class_probs_predict(
    point_mask_logits: torch.Tensor,
    mask_class_probs: torch.Tensor,
    support_threshold: float = 1e-6,
    unsupported_label: int = IGNORE_LABEL,
    chunk_size: int = 50000,
) -> torch.Tensor:
    """Diff2Scene Eq.3 semantic labels from precomputed per-mask class probabilities."""
    point_scores = diff2scene_point_class_scores(
        point_mask_logits=point_mask_logits,
        mask_class_probs=mask_class_probs,
        chunk_size=chunk_size,
    )
    if point_scores.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)
    pred = torch.max(point_scores, 1)[1].long()
    support = torch.zeros(point_mask_logits.shape[0], device=point_mask_logits.device)
    for start in range(0, point_mask_logits.shape[0], chunk_size):
        end = min(start + chunk_size, point_mask_logits.shape[0])
        support[start:end] = torch.sigmoid(point_mask_logits[start:end].float()).sum(dim=1)
    pred = torch.where(
        support > support_threshold,
        pred,
        torch.full_like(pred, unsupported_label),
    )
    return pred.cpu()


@torch.no_grad()
def diff2scene_dual_mask_class_probs_predict(
    salient_masks: torch.Tensor,
    geometric_mask_logits: torch.Tensor,
    mask_class_probs: torch.Tensor,
    lambda_weight: float = 0.5,
    support_threshold: float = 1e-6,
    unsupported_label: int = IGNORE_LABEL,
    chunk_size: int = 50000,
) -> torch.Tensor:
    """Diff2Scene Eq.3 labels from lifted 2D masks and predicted 3D masks."""
    if salient_masks.ndim != 2:
        raise RuntimeError(f"salient_masks must be (N,K), got {tuple(salient_masks.shape)}")
    if geometric_mask_logits.ndim != 2:
        raise RuntimeError(
            f"geometric_mask_logits must be (N,K), got {tuple(geometric_mask_logits.shape)}"
        )
    if mask_class_probs.ndim != 2:
        raise RuntimeError(f"mask_class_probs must be (K,C), got {tuple(mask_class_probs.shape)}")
    if salient_masks.shape != geometric_mask_logits.shape:
        raise RuntimeError(
            f"mask shape mismatch: salient={tuple(salient_masks.shape)}, "
            f"geometric={tuple(geometric_mask_logits.shape)}"
        )
    if salient_masks.shape[1] != mask_class_probs.shape[0]:
        raise RuntimeError(
            f"mask count mismatch: masks K={salient_masks.shape[1]}, "
            f"class_probs K={mask_class_probs.shape[0]}"
        )
    lam = float(lambda_weight)
    if lam < 0.0 or lam > 1.0:
        raise RuntimeError(f"lambda_weight must be in [0,1], got {lambda_weight}")
    if salient_masks.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)

    device = geometric_mask_logits.device
    probs = mask_class_probs.float().to(device)
    scores = []
    support = torch.zeros(geometric_mask_logits.shape[0], device=device)
    for start in range(0, geometric_mask_logits.shape[0], chunk_size):
        end = min(start + chunk_size, geometric_mask_logits.shape[0])
        salient = salient_masks[start:end].float().to(device)
        geometric = torch.sigmoid(geometric_mask_logits[start:end].float())
        point_to_masks = lam * salient + (1.0 - lam) * geometric
        support[start:end] = point_to_masks.sum(dim=1)
        scores.append(point_to_masks @ probs)

    point_scores = torch.cat(scores, dim=0)
    pred = torch.max(point_scores, 1)[1].long()
    pred = torch.where(
        support > support_threshold,
        pred,
        torch.full_like(pred, unsupported_label),
    )
    return pred.cpu()


@torch.no_grad()
def diff2scene_mask_feature_predict(
    point_mask_logits: torch.Tensor,
    mask_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float = CLIP_LOGIT_SCALE,
    support_threshold: float = 1e-6,
    unsupported_label: int = IGNORE_LABEL,
    chunk_size: int = 50000,
) -> torch.Tensor:
    """Diff2Scene Eq.3 semantic prediction for mask-query outputs.

    `mask_features` are the semantically-rich 2D mask embeddings. They are
    converted to per-mask label probabilities p_i^c over the provided class set
    (ScanNet labels if the caller passes ScanNet class names/text features).
    `point_mask_logits` has I columns, where I is the number of masks. The point
    class score is then:
        p^c(point) = sum_i sigmoid(point_mask_logits[:, i]) * p_i^c
    """
    if point_mask_logits.ndim != 2:
        raise RuntimeError(
            f"point_mask_logits must be (N,K), got {tuple(point_mask_logits.shape)}"
        )
    if mask_features.ndim != 2:
        raise RuntimeError(f"mask_features must be (K,D), got {tuple(mask_features.shape)}")
    if text_features.ndim != 2:
        raise RuntimeError(f"text_features must be (C,D), got {tuple(text_features.shape)}")
    if point_mask_logits.shape[1] != mask_features.shape[0]:
        raise RuntimeError(
            f"mask count mismatch: logits K={point_mask_logits.shape[1]}, features K={mask_features.shape[0]}"
        )
    if mask_features.shape[1] != text_features.shape[1]:
        raise RuntimeError(
            f"feature dim mismatch: mask={mask_features.shape[1]}, text={text_features.shape[1]}"
        )
    if point_mask_logits.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)
    if mask_features.shape[0] == 0:
        return torch.full((point_mask_logits.shape[0],), unsupported_label, dtype=torch.long)

    masks_to_classes = mask_feature_class_probs(
        mask_features=mask_features.float().to(point_mask_logits.device),
        text_features=text_features.float().to(point_mask_logits.device),
        logit_scale=logit_scale,
    )

    return diff2scene_class_probs_predict(
        point_mask_logits=point_mask_logits,
        mask_class_probs=masks_to_classes,
        support_threshold=support_threshold,
        unsupported_label=unsupported_label,
        chunk_size=chunk_size,
    )


@torch.no_grad()
def odise_geometric_mask_feature_predict(
    point_mask_logits: torch.Tensor,
    hybrid_features: torch.Tensor,
    hybrid_text_features: torch.Tensor,
    clip_features: torch.Tensor,
    clip_text_features: torch.Tensor,
    lambda_weight: float = 0.5,
    hybrid_logit_scale: float = CLIP_LOGIT_SCALE,
    clip_logit_scale: float = CLIP_LOGIT_SCALE,
    support_threshold: float = 1e-6,
    unsupported_label: int = IGNORE_LABEL,
    chunk_size: int = 50000,
    eps: float = 1e-12,
) -> torch.Tensor:
    """ODISE Eq.10: geometric mean of hybrid/diffusion and CLIP mask predictions."""
    if hybrid_features.shape[0] != clip_features.shape[0]:
        raise RuntimeError(
            f"mask count mismatch: hybrid K={hybrid_features.shape[0]}, clip K={clip_features.shape[0]}"
        )
    lam = float(lambda_weight)
    if lam < 0.0 or lam > 1.0:
        raise RuntimeError(f"lambda_weight must be in [0,1], got {lambda_weight}")

    hybrid_probs = mask_feature_class_probs(
        hybrid_features.float().to(point_mask_logits.device),
        hybrid_text_features.float().to(point_mask_logits.device),
        logit_scale=hybrid_logit_scale,
    )
    clip_probs = mask_feature_class_probs(
        clip_features.float().to(point_mask_logits.device),
        clip_text_features.float().to(point_mask_logits.device),
        logit_scale=clip_logit_scale,
    )
    if hybrid_probs.shape != clip_probs.shape:
        raise RuntimeError(
            f"class prob shape mismatch: hybrid={tuple(hybrid_probs.shape)}, clip={tuple(clip_probs.shape)}"
        )

    log_probs = lam * torch.log(hybrid_probs.clamp_min(eps))
    log_probs = log_probs + (1.0 - lam) * torch.log(clip_probs.clamp_min(eps))
    geom_probs = torch.softmax(log_probs, dim=-1)

    return diff2scene_class_probs_predict(
        point_mask_logits=point_mask_logits,
        mask_class_probs=geom_probs,
        support_threshold=support_threshold,
        unsupported_label=unsupported_label,
        chunk_size=chunk_size,
    )


@torch.no_grad()
def odise_geometric_dual_mask_predict(
    salient_masks: torch.Tensor,
    geometric_mask_logits: torch.Tensor,
    hybrid_features: torch.Tensor,
    hybrid_text_features: torch.Tensor,
    clip_features: torch.Tensor,
    clip_text_features: torch.Tensor,
    class_lambda: float = 0.5,
    mask_lambda: float = 0.5,
    hybrid_logit_scale: float = CLIP_LOGIT_SCALE,
    clip_logit_scale: float = CLIP_LOGIT_SCALE,
    support_threshold: float = 1e-6,
    unsupported_label: int = IGNORE_LABEL,
    chunk_size: int = 50000,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Zhu/Diff2Scene eval: ODISE Eq.10 class PC + Diff2Scene Eq.3 mask fusion."""
    hybrid_probs = mask_feature_class_probs(
        hybrid_features.float().to(geometric_mask_logits.device),
        hybrid_text_features.float().to(geometric_mask_logits.device),
        logit_scale=hybrid_logit_scale,
    )
    clip_probs = mask_feature_class_probs(
        clip_features.float().to(geometric_mask_logits.device),
        clip_text_features.float().to(geometric_mask_logits.device),
        logit_scale=clip_logit_scale,
    )
    log_probs = float(class_lambda) * torch.log(hybrid_probs.clamp_min(eps))
    log_probs = log_probs + (1.0 - float(class_lambda)) * torch.log(clip_probs.clamp_min(eps))
    geom_probs = torch.softmax(log_probs, dim=-1)
    return diff2scene_dual_mask_class_probs_predict(
        salient_masks=salient_masks,
        geometric_mask_logits=geometric_mask_logits,
        mask_class_probs=geom_probs,
        lambda_weight=mask_lambda,
        support_threshold=support_threshold,
        unsupported_label=unsupported_label,
        chunk_size=chunk_size,
    )


@dataclass
class _SemanticAccumulator:
    class_names: Sequence[str] = SCANNET_LABELS_20
    ignore_index: int | Iterable[int] = IGNORE_LABEL

    def __post_init__(self):
        self.class_names = tuple(self.class_names)
        self.K = len(self.class_names)
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.K, dtype=torch.float64)
        self.union = torch.zeros(self.K, dtype=torch.float64)
        self.target = torch.zeros(self.K, dtype=torch.float64)

    def update_labels(self, pred_labels: torch.Tensor, gt_labels: torch.Tensor):
        if pred_labels.shape != gt_labels.shape:
            raise RuntimeError(
                f"pred/gt shape mismatch: pred={tuple(pred_labels.shape)}, gt={tuple(gt_labels.shape)}"
            )
        inter, union, target = intersectionAndUnionGPU(
            pred_labels.long(),
            gt_labels.long(),
            self.K,
            self.ignore_index,
        )
        self.intersection += inter.double()
        self.union += union.double()
        self.target += target.double()

    def compute(self, prefix: str) -> Dict:
        result = compute_iou(self.intersection, self.union, self.target, self.class_names)
        return {
            prefix: result["miou"],
            prefix.replace("miou", "macc"): result["macc"],
            f"per_class_iou_{prefix}": result["per_class_iou"],
            f"per_class_acc_{prefix}": result["per_class_acc"],
            f"n_valid_classes_{prefix}": result["n_valid_classes"],
            f"n_valid_classes_acc_{prefix}": result["n_valid_classes_acc"],
            f"target_{prefix}": result["target"],
        }


class Diff2SceneSemanticEvaluator:
    """Mask-query evaluator using Diff2Scene Eq.3 and standard IoU accounting."""

    def __init__(
        self,
        text_features: torch.Tensor,
        class_names: Sequence[str] = SCANNET_LABELS_20,
        ignore_index: int | Iterable[int] = IGNORE_LABEL,
        logit_scale: float = CLIP_LOGIT_SCALE,
        support_threshold: float = 1e-6,
        unsupported_label: int = IGNORE_LABEL,
        chunk_size: int = 50000,
    ):
        self.text_features = text_features
        self.logit_scale = logit_scale
        self.support_threshold = support_threshold
        self.unsupported_label = unsupported_label
        self.chunk_size = chunk_size
        self.acc = _SemanticAccumulator(class_names, ignore_index)

    def reset(self):
        self.acc.reset()

    @torch.no_grad()
    def update(
        self,
        gt_labels: torch.Tensor,
        mask_features: torch.Tensor,
        point_mask_logits: torch.Tensor,
    ):
        pred = diff2scene_mask_feature_predict(
            point_mask_logits=point_mask_logits,
            mask_features=mask_features,
            text_features=self.text_features,
            logit_scale=self.logit_scale,
            support_threshold=self.support_threshold,
            unsupported_label=self.unsupported_label,
            chunk_size=self.chunk_size,
        )
        self.acc.update_labels(pred, gt_labels.detach().cpu().long())

    def compute(self) -> Dict:
        result = self.acc.compute("semantic_miou_diff2scene")
        return {
            "semantic_miou_diff2scene": result["semantic_miou_diff2scene"],
            "semantic_macc_diff2scene": result["semantic_macc_diff2scene"],
            "per_class_iou_diff2scene": result["per_class_iou_semantic_miou_diff2scene"],
            "per_class_acc_diff2scene": result["per_class_acc_semantic_miou_diff2scene"],
            "n_valid_classes": result["n_valid_classes_semantic_miou_diff2scene"],
            "n_valid_classes_acc": result["n_valid_classes_acc_semantic_miou_diff2scene"],
            "target": result["target_semantic_miou_diff2scene"],
        }


SemanticMIoUTracker = Diff2SceneSemanticEvaluator
XMask3DSemanticEvaluator = Diff2SceneSemanticEvaluator


class Diff2SceneSemanticMIoUTracker:
    """Backward-compatible mask-query tracker used by existing trainers."""

    def __init__(
        self,
        class_names: Sequence[str] = SCANNET_LABELS_20,
        text_temperature: float = CLIP_LOGIT_SCALE,
        chunk_size: int = 50000,
        mask_support_threshold: float = 1e-6,
        unsupported_label: int = IGNORE_LABEL,
        ignore_index: int | Iterable[int] = IGNORE_LABEL,
    ):
        self.class_names = tuple(class_names)
        self.text_temperature = text_temperature
        self.chunk_size = chunk_size
        self.mask_support_threshold = mask_support_threshold
        self.unsupported_label = unsupported_label
        self.acc = _SemanticAccumulator(self.class_names, ignore_index)

    def reset(self):
        self.acc.reset()

    @torch.no_grad()
    def predict_labels(
        self,
        fused_embeddings: torch.Tensor,
        pred_mask_logits: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        return diff2scene_mask_feature_predict(
            point_mask_logits=pred_mask_logits,
            mask_features=fused_embeddings,
            text_features=text_features,
            logit_scale=self.text_temperature,
            support_threshold=self.mask_support_threshold,
            unsupported_label=self.unsupported_label,
            chunk_size=self.chunk_size,
        )

    @torch.no_grad()
    def update(
        self,
        gt_labels: torch.Tensor,
        fused_embeddings: torch.Tensor,
        pred_mask_logits: torch.Tensor,
        text_features: torch.Tensor,
    ):
        pred = self.predict_labels(fused_embeddings, pred_mask_logits, text_features)
        self.acc.update_labels(pred, gt_labels.detach().cpu().long())

    def compute(self) -> Dict:
        result = self.acc.compute("semantic_miou_diff2scene")
        return {
            "semantic_miou_diff2scene": result["semantic_miou_diff2scene"],
            "semantic_macc_diff2scene": result["semantic_macc_diff2scene"],
            "per_class_iou_diff2scene": result["per_class_iou_semantic_miou_diff2scene"],
            "per_class_acc_diff2scene": result["per_class_acc_semantic_miou_diff2scene"],
            "n_valid_classes": result["n_valid_classes_semantic_miou_diff2scene"],
            "n_valid_classes_acc": result["n_valid_classes_acc_semantic_miou_diff2scene"],
            "target": result["target_semantic_miou_diff2scene"],
        }


class ODISEPCSemanticMIoUTracker:
    """Track ODISE-style hybrid/text, CLIP/text, and Eq.10 geometric-PC mIoU."""

    def __init__(
        self,
        class_names: Sequence[str] = SCANNET_LABELS_20,
        text_temperature: float = CLIP_LOGIT_SCALE,
        clip_text_temperature: float = CLIP_LOGIT_SCALE,
        pc_lambda: float = 0.5,
        chunk_size: int = 50000,
        mask_support_threshold: float = 1e-6,
        unsupported_label: int = IGNORE_LABEL,
        ignore_index: int | Iterable[int] = IGNORE_LABEL,
    ):
        self.class_names = tuple(class_names)
        self.text_temperature = text_temperature
        self.clip_text_temperature = clip_text_temperature
        self.pc_lambda = pc_lambda
        self.chunk_size = chunk_size
        self.mask_support_threshold = mask_support_threshold
        self.unsupported_label = unsupported_label
        self.hybrid_acc = _SemanticAccumulator(self.class_names, ignore_index)
        self.clip_acc = _SemanticAccumulator(self.class_names, ignore_index)
        self.pc_acc = _SemanticAccumulator(self.class_names, ignore_index)

    def reset(self):
        self.hybrid_acc.reset()
        self.clip_acc.reset()
        self.pc_acc.reset()

    @torch.no_grad()
    def update(
        self,
        gt_labels: torch.Tensor,
        hybrid_features: torch.Tensor,
        clip_features: torch.Tensor,
        pred_mask_logits: torch.Tensor,
        hybrid_text_features: torch.Tensor,
        clip_text_features: torch.Tensor,
        salient_masks: Optional[torch.Tensor] = None,
    ):
        gt_cpu = gt_labels.detach().cpu().long()
        pred_hybrid = diff2scene_mask_feature_predict(
            point_mask_logits=pred_mask_logits,
            mask_features=hybrid_features,
            text_features=hybrid_text_features,
            logit_scale=self.text_temperature,
            support_threshold=self.mask_support_threshold,
            unsupported_label=self.unsupported_label,
            chunk_size=self.chunk_size,
        )
        pred_clip = diff2scene_mask_feature_predict(
            point_mask_logits=pred_mask_logits,
            mask_features=clip_features,
            text_features=clip_text_features,
            logit_scale=self.clip_text_temperature,
            support_threshold=self.mask_support_threshold,
            unsupported_label=self.unsupported_label,
            chunk_size=self.chunk_size,
        )
        if salient_masks is None:
            pred_pc = odise_geometric_mask_feature_predict(
                point_mask_logits=pred_mask_logits,
                hybrid_features=hybrid_features,
                hybrid_text_features=hybrid_text_features,
                clip_features=clip_features,
                clip_text_features=clip_text_features,
                lambda_weight=self.pc_lambda,
                hybrid_logit_scale=self.text_temperature,
                clip_logit_scale=self.clip_text_temperature,
                support_threshold=self.mask_support_threshold,
                unsupported_label=self.unsupported_label,
                chunk_size=self.chunk_size,
            )
        else:
            pred_pc = odise_geometric_dual_mask_predict(
                salient_masks=salient_masks,
                geometric_mask_logits=pred_mask_logits,
                hybrid_features=hybrid_features,
                hybrid_text_features=hybrid_text_features,
                clip_features=clip_features,
                clip_text_features=clip_text_features,
                class_lambda=self.pc_lambda,
                mask_lambda=0.5,
                hybrid_logit_scale=self.text_temperature,
                clip_logit_scale=self.clip_text_temperature,
                support_threshold=self.mask_support_threshold,
                unsupported_label=self.unsupported_label,
                chunk_size=self.chunk_size,
            )
        self.hybrid_acc.update_labels(pred_hybrid, gt_cpu)
        self.clip_acc.update_labels(pred_clip, gt_cpu)
        self.pc_acc.update_labels(pred_pc, gt_cpu)

    def compute(self) -> Dict:
        hybrid = self.hybrid_acc.compute("semantic_miou_hybrid_text")
        clip = self.clip_acc.compute("semantic_miou_clip_text")
        pc = self.pc_acc.compute("semantic_miou_pc")
        return {
            "semantic_miou_hybrid_text": hybrid["semantic_miou_hybrid_text"],
            "semantic_miou_clip_text": clip["semantic_miou_clip_text"],
            "semantic_miou_pc": pc["semantic_miou_pc"],
            "semantic_macc_hybrid_text": hybrid["semantic_macc_hybrid_text"],
            "semantic_macc_clip_text": clip["semantic_macc_clip_text"],
            "semantic_macc_pc": pc["semantic_macc_pc"],
            "per_class_iou_hybrid_text": hybrid["per_class_iou_semantic_miou_hybrid_text"],
            "per_class_iou_clip_text": clip["per_class_iou_semantic_miou_clip_text"],
            "per_class_iou_pc": pc["per_class_iou_semantic_miou_pc"],
            "per_class_acc_hybrid_text": hybrid["per_class_acc_semantic_miou_hybrid_text"],
            "per_class_acc_clip_text": clip["per_class_acc_semantic_miou_clip_text"],
            "per_class_acc_pc": pc["per_class_acc_semantic_miou_pc"],
            "n_valid_classes_hybrid_text": hybrid["n_valid_classes_semantic_miou_hybrid_text"],
            "n_valid_classes_clip_text": clip["n_valid_classes_semantic_miou_clip_text"],
            "n_valid_classes_pc": pc["n_valid_classes_semantic_miou_pc"],
            "n_valid_classes_acc_hybrid_text": hybrid["n_valid_classes_acc_semantic_miou_hybrid_text"],
            "n_valid_classes_acc_clip_text": clip["n_valid_classes_acc_semantic_miou_clip_text"],
            "n_valid_classes_acc_pc": pc["n_valid_classes_acc_semantic_miou_pc"],
            "target": pc["target_semantic_miou_pc"],
        }
