#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_open_vocab_v2 import (  # noqa: E402
    DataLoaderConfig,
    create_data_loaders,
    load_yaml_config,
    resolve_odise_config_path,
    set_seed,
)
from dataset.open_vocab_dataset_v2 import OpenVocabDatasetV2Config  # noqa: E402
from model.open_vocab_fusion_v2 import (  # noqa: E402
    OpenVocab3DFusionModelV2,
    OpenVocabFusionModelV2Config,
)
from experiment_mask_distill.trainer_mask_distill import (  # noqa: E402
    MaskDistillTrainer,
    MaskDistillTrainerConfig,
)
from experiment_mask_distill.criterion_mask_distill import build_lifted_3d_masks  # noqa: E402


def _save_histogram(path: Path, values: List[float], title: str, xlabel: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Probe] Skip histogram {path.name}: matplotlib unavailable ({exc})")
        return
    arr = np.asarray(values, dtype=np.float64)
    plt.figure(figsize=(6, 4))
    if arr.size > 0:
        plt.hist(arr, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if float(x.std()) < 1e-8 or float(y.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    return _pearson(_rankdata(x), _rankdata(y))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _safe_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_mean": 0.0,
            "clear_ratio_003": 0.0,
            "clear_ratio_005": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "abs_mean": float(np.abs(arr).mean()),
        "clear_ratio_001": float((np.abs(arr) > 0.01).mean()),
        "clear_ratio_003": float((np.abs(arr) > 0.03).mean()),
        "clear_ratio_005": float((np.abs(arr) > 0.05).mean()),
    }


def _avg_pairwise_cos(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2:
        return x.new_tensor(0.0)
    sim = x @ x.t()
    idx = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    return sim[idx[0], idx[1]].mean()


def _binary_entropy(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    ent = -(x * np.log(x) + (1.0 - x) * np.log(1.0 - x))
    return float(ent.mean())


def _train_probe_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    seed: int,
    max_steps: int = 2000,
    batch_size: int = 256,
) -> Dict[str, float]:
    if features.shape[0] < 8:
        return {
            "train_mse": 0.0,
            "val_mse": 0.0,
            "pearson_corr": 0.0,
            "spearman_corr": 0.0,
            "r2_score": 0.0,
            "pred_mean": 0.0,
            "pred_std": 0.0,
            "target_mean": float(targets.mean()) if targets.size else 0.0,
            "target_std": float(targets.std()) if targets.size else 0.0,
        }

    rng = np.random.default_rng(seed)
    perm = rng.permutation(features.shape[0])
    split = max(int(features.shape[0] * 0.8), 1)
    tr_idx = perm[:split]
    va_idx = perm[split:] if split < features.shape[0] else perm[:0]
    if va_idx.size == 0:
        va_idx = tr_idx[-max(1, len(tr_idx) // 5):]

    x_train = features[tr_idx]
    y_train = targets[tr_idx]
    x_val = features[va_idx]
    y_val = targets[va_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    x_val_t = torch.from_numpy(x_val).float()
    y_val_t = torch.from_numpy(y_val).float()

    steps = 0
    best_state = None
    best_val = float("inf")
    for _ in range(3):
        order = torch.randperm(x_train_t.shape[0])
        for start in range(0, x_train_t.shape[0], batch_size):
            idx = order[start:start + batch_size]
            pred = model(x_train_t[idx]).squeeze(-1)
            loss = F.mse_loss(pred, y_train_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            if steps >= max_steps:
                break
        with torch.no_grad():
            val_pred = model(x_val_t).squeeze(-1)
            val_loss = F.mse_loss(val_pred, y_val_t).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if steps >= max_steps:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        train_pred = model(x_train_t).squeeze(-1).cpu().numpy()
        val_pred = model(x_val_t).squeeze(-1).cpu().numpy()

    return {
        "train_mse": float(np.mean((train_pred - y_train) ** 2)),
        "val_mse": float(np.mean((val_pred - y_val) ** 2)),
        "pearson_corr": _pearson(val_pred, y_val),
        "spearman_corr": _spearman(val_pred, y_val),
        "r2_score": _r2_score(y_val, val_pred),
        "pred_mean": float(val_pred.mean()),
        "pred_std": float(val_pred.std()),
        "target_mean": float(y_val.mean()),
        "target_std": float(y_val.std()),
    }


def _build_probe_model_and_trainer(
    config_path: str,
    device: str,
    output_dir: str,
    args_resume: str,
) -> Tuple[OpenVocab3DFusionModelV2, MaskDistillTrainer, Any, Dict[str, Any]]:
    yaml_config = load_yaml_config(config_path) or {}
    repo_root = str(REPO_ROOT)
    _dataset = yaml_config.get("dataset") or {}
    _dataloader = yaml_config.get("dataloader") or {}
    _model = yaml_config.get("model") or {}
    _trainer = yaml_config.get("trainer") or {}

    data_config_path = _dataset.get("data_config_path", "config/data_scannet_3d.yaml")
    if not os.path.isabs(data_config_path):
        data_config_path = os.path.join(repo_root, data_config_path)
    precomputed_dir = _dataset.get("precomputed_dir")
    if precomputed_dir and not os.path.isabs(precomputed_dir):
        precomputed_dir = os.path.join(repo_root, precomputed_dir)
    projection_dir = _dataset.get("projection_dir")
    if projection_dir and not os.path.isabs(projection_dir):
        projection_dir = os.path.join(repo_root, projection_dir)

    dataset_config = OpenVocabDatasetV2Config(
        data_config_path=data_config_path,
        precomputed_dir=precomputed_dir,
        projection_dir=projection_dir,
        split=_dataset.get("split", "train"),
        scannet200=_dataset.get("scannet200", False),
        voxel_size=_dataset.get("voxel_size", 0.05),
        aug=_dataset.get("aug", True),
        max_samples=_dataset.get("max_samples") or None,
        max_samples_ratio=_dataset.get("max_samples_ratio") or None,
    )
    setattr(dataset_config, "val_max_samples", _dataset.get("val_max_samples") or None)
    setattr(dataset_config, "val_max_samples_ratio", _dataset.get("val_max_samples_ratio") or None)

    raw_batch_size = int(_dataloader.get("batch_size", 4))
    multiview_batch = bool(_dataloader.get("multiview_batch", False))
    scenes_per_batch = int(_dataloader.get("scenes_per_batch", 1))
    views_per_scene = int(_dataloader.get("views_per_scene", 4))
    if multiview_batch:
        raw_batch_size = scenes_per_batch * views_per_scene
    dataloader_config = DataLoaderConfig(
        batch_size=raw_batch_size,
        num_workers=int(_dataloader.get("num_workers", 4)),
        val_batch_size=_dataloader.get("val_batch_size"),
        val_num_workers=_dataloader.get("val_num_workers"),
        drop_last=bool(_dataloader.get("drop_last", True)),
        multiview_batch=multiview_batch,
        scenes_per_batch=scenes_per_batch,
        views_per_scene=views_per_scene,
        seed=int(yaml_config.get("seed", 1342)),
    )
    train_loader, _ = create_data_loaders(dataset_config, dataloader_config)

    label_path = _model.get("label_path", "")
    lseg_ckpt_path = _model.get("lseg_ckpt_path", "")
    odise_config_path = _model.get("odise_model_config_path", "")
    if label_path and not os.path.isabs(label_path):
        label_path = os.path.join(repo_root, label_path)
    if lseg_ckpt_path and not os.path.isabs(lseg_ckpt_path):
        lseg_ckpt_path = os.path.join(repo_root, lseg_ckpt_path)
    if odise_config_path:
        odise_config_path = resolve_odise_config_path(odise_config_path, repo_root)
    semantic_proj_path = _model.get("semantic_proj_path")
    if semantic_proj_path and not os.path.isabs(semantic_proj_path):
        semantic_proj_path = os.path.join(repo_root, semantic_proj_path)

    model_config = OpenVocabFusionModelV2Config(
        device=device,
        label_path=label_path if label_path and os.path.exists(label_path) else None,
        lseg_ckpt_path=lseg_ckpt_path if lseg_ckpt_path and os.path.exists(lseg_ckpt_path) else None,
        odise_model_config_path=odise_config_path,
        pc_arch=_model.get("pc_arch", "MinkUNet34C"),
        pixel_embedding_dim=_model.get("pixel_embedding_dim", 512),
        mask_embedding_dim=_model.get("mask_embedding_dim", 256),
        fused_embedding_dim=_model.get("fused_embedding_dim", 256),
        pc_last_dim=_model.get("pc_last_dim", 256),
        alpha_mode=_model.get("alpha_mode", "fixed"),
        alpha_init=float(_model.get("alpha_init", 1.25)),
        alpha_max=_model.get("alpha_max", 2.0),
        use_semantic_query=bool(_model.get("use_semantic_query", False)),
        semantic_fusion_mode=_model.get("semantic_fusion_mode", "fixed"),
        semantic_odise_weight=float(_model.get("semantic_odise_weight", 0.5)),
        semantic_lseg_weight=float(_model.get("semantic_lseg_weight", 0.5)),
        semantic_init_odise_weight=float(_model.get("semantic_init_odise_weight", 0.3)),
        semantic_init_lseg_weight=float(_model.get("semantic_init_lseg_weight", 0.7)),
        semantic_proj_path=semantic_proj_path,
        freeze_semantic_proj=bool(_model.get("freeze_semantic_proj", True)),
        use_source_reliability_gate=False,
        use_point_semantic_gate=bool(_model.get("use_point_semantic_gate", False)),
        point_sem_gate_hidden_dim=int(_model.get("point_sem_gate_hidden_dim", 128)),
        point_sem_gate_init_bias=float(_model.get("point_sem_gate_init_bias", 0.85)),
        alignment_query_mode=str(_model.get("alignment_query_mode", "fused")),
    )
    model = OpenVocab3DFusionModelV2(model_config).to(device)

    resume_checkpoint = args_resume or _trainer.get("resume")
    if resume_checkpoint and not os.path.isabs(resume_checkpoint):
        resume_checkpoint = os.path.join(repo_root, resume_checkpoint)
    if resume_checkpoint and not os.path.exists(resume_checkpoint):
        resume_checkpoint = None

    trainer_config = MaskDistillTrainerConfig(
        num_epochs=1,
        base_lr=float(_trainer.get("base_lr", 5e-5)),
        weight_decay=float(_trainer.get("weight_decay", 4e-4)),
        grad_clip_norm=float(_trainer.get("grad_clip_norm", 1.0)),
        log_dir=os.path.join(output_dir, "tb"),
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        warmup_epochs=int(_trainer.get("warmup_epochs", 1)),
        scheduler_type=str(_trainer.get("scheduler_type", "cosine")),
        mask_distill_weight=float(_trainer.get("mask_distill_weight", 1.0)),
        bce_weight=float(_trainer.get("bce_weight", 0.0)),
        dice_weight=float(_trainer.get("dice_weight", 0.0)),
        min_points_per_mask=int(_trainer.get("min_points_per_mask", 6)),
        resume_checkpoint=resume_checkpoint,
        use_amp=bool(_trainer.get("use_amp", True)),
        use_model_half=bool(_trainer.get("use_model_half", False)),
        alignment_query_mode=str(_trainer.get("alignment_query_mode", "fused")),
        multiview_batch=multiview_batch,
        scenes_per_batch=scenes_per_batch,
        views_per_scene=views_per_scene,
        source_gate_train=False,
        source_gate_training_target="none",
        enable_verbose_legacy_probes=False,
        enable_legacy_source_gate_logs=False,
    )
    trainer = MaskDistillTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        config=trainer_config,
        device=device,
    )
    return model, trainer, train_loader, yaml_config


def _row_hash(coords_xyz: torch.Tensor) -> torch.Tensor:
    c = coords_xyz.long() + 20000
    base = 40001
    return c[:, 0] * (base ** 2) + c[:, 1] * base + c[:, 2]


def _compute_purity_lseg(
    pixel_embeddings: Optional[torch.Tensor],
    masks_b: torch.Tensor,
    valid_k: torch.Tensor,
    lseg_q: torch.Tensor,
) -> Optional[torch.Tensor]:
    if pixel_embeddings is None or pixel_embeddings.dim() != 4:
        return None
    feats = pixel_embeddings.float()
    H, W = feats.shape[0], feats.shape[1]
    masks = masks_b[valid_k].float()
    if masks.shape[-2:] != (H, W):
        masks = F.interpolate(
            masks.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    feat_flat = feats.reshape(H * W, feats.shape[-1])
    feat_flat = F.normalize(feat_flat, dim=-1)
    pooled = F.normalize(lseg_q.float(), dim=-1)
    purity = pooled.new_zeros(pooled.shape[0])
    for idx in range(masks.shape[0]):
        pix = masks[idx].reshape(-1) > 0.5
        if not bool(pix.any()):
            continue
        purity[idx] = (feat_flat[pix] @ pooled[idx]).mean()
    return purity


def _compute_scene_region_signals(
    scene_items: List[Dict[str, Any]],
    iou_thr: float,
    max_pairs: int,
) -> Tuple[int, int]:
    valid_pairs = 0
    num_items = len(scene_items)
    for item in scene_items:
        k = item["num_masks"]
        item["mv_num_lseg"] = item["odise_q"].new_zeros(k)
        item["mv_den_lseg"] = item["odise_q"].new_zeros(k)
        item["mv_num_odise"] = item["odise_q"].new_zeros(k)
        item["mv_den_odise"] = item["odise_q"].new_zeros(k)
        item["mv_pair_count"] = item["odise_q"].new_zeros(k)
        item["mv_iou_sum"] = item["odise_q"].new_zeros(k)
        item["mv_iou_max"] = item["odise_q"].new_zeros(k)

    for i in range(num_items):
        item_i = scene_items[i]
        hash_i = item_i["coord_hash"]
        sort_j_cache = {}
        for j in range(num_items):
            if i == j:
                continue
            item_j = scene_items[j]
            if item_i["view_id"] == item_j["view_id"]:
                continue
            if j not in sort_j_cache:
                sort_j_cache[j] = torch.sort(item_j["coord_hash"])
            sorted_hash_j, order_j = sort_j_cache[j]
            pos = torch.searchsorted(sorted_hash_j, hash_i)
            valid = pos < sorted_hash_j.shape[0]
            pos = pos.clamp_max(sorted_hash_j.shape[0] - 1)
            valid = valid & (sorted_hash_j[pos] == hash_i)
            if not bool(valid.any()):
                continue
            idx_i = torch.nonzero(valid, as_tuple=True)[0]
            idx_j = order_j[pos[valid]]
            A = item_i["lifted_bool"][idx_i].float()
            B = item_j["lifted_bool"][idx_j].float()
            inter = A.t() @ B
            cnt_i = A.sum(dim=0)
            cnt_j = B.sum(dim=0)
            union = cnt_i[:, None] + cnt_j[None, :] - inter
            iou = inter / union.clamp_min(1.0)
            sim_lseg = F.normalize(item_i["lseg_q"], dim=-1) @ F.normalize(item_j["lseg_q"], dim=-1).t()
            sim_odise = F.normalize(item_i["odise_q"], dim=-1) @ F.normalize(item_j["odise_q"], dim=-1).t()
            topv, topidx = torch.topk(iou, k=min(max_pairs, iou.shape[1]), dim=1)
            for local_i in range(iou.shape[0]):
                keep = topv[local_i] > iou_thr
                if not bool(keep.any()):
                    continue
                weights = topv[local_i, keep]
                nbrs = topidx[local_i, keep]
                item_i["mv_num_lseg"][local_i] += (sim_lseg[local_i, nbrs] * weights).sum()
                item_i["mv_den_lseg"][local_i] += weights.sum()
                item_i["mv_num_odise"][local_i] += (sim_odise[local_i, nbrs] * weights).sum()
                item_i["mv_den_odise"][local_i] += weights.sum()
                item_i["mv_pair_count"][local_i] += float(keep.sum().item())
                item_i["mv_iou_sum"][local_i] += weights.sum()
                item_i["mv_iou_max"][local_i] = torch.maximum(
                    item_i["mv_iou_max"][local_i],
                    weights.max(),
                )
                valid_pairs += int(keep.sum().item())

    for item in scene_items:
        den = item["mv_den_lseg"]
        item["c_valid"] = den > 0
        item["C_lseg"] = torch.where(item["c_valid"], item["mv_num_lseg"] / den.clamp_min(1e-6), torch.zeros_like(den))
        item["C_odise"] = torch.where(item["c_valid"], item["mv_num_odise"] / den.clamp_min(1e-6), torch.zeros_like(den))
        item["overlap_iou_mean"] = torch.where(
            item["mv_pair_count"] > 0,
            item["mv_iou_sum"] / item["mv_pair_count"].clamp_min(1.0),
            torch.zeros_like(den),
        )
        item["overlap_iou_max"] = item["mv_iou_max"]
        item["view_pair_count"] = item["mv_pair_count"]

    total_regions = sum(item["num_masks"] for item in scene_items)
    return valid_pairs, total_regions


def _normalize_robust(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, 10.0))
    hi = float(np.percentile(arr, 90.0))
    if hi - lo < 1e-6:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _g_stats_from_array(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mid_ratio": 0.0,
            "lseg_ratio_06": 0.0,
            "odise_ratio_04": 0.0,
            "entropy": 0.0,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mid_ratio": float(((arr >= 0.45) & (arr <= 0.55)).mean()),
        "lseg_ratio_06": float((arr > 0.6).mean()),
        "odise_ratio_04": float((arr < 0.4).mean()),
        "entropy": _binary_entropy(arr),
    }


def _corr_summary(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if x.size == 0 or y.size == 0:
        return {"pearson": 0.0, "spearman": 0.0}
    return {
        "pearson": _pearson(x, y),
        "spearman": _spearman(x, y),
    }


def _size_bucket(mask_area_ratio: float) -> str:
    if mask_area_ratio < 0.01:
        return "small"
    if mask_area_ratio < 0.10:
        return "medium"
    return "large"


def _rows_to_arrays(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not rows:
        return {}
    scalar_keys = [
        "C_lseg",
        "C_odise",
        "C_diff",
        "sharp_lseg",
        "sharp_odise",
        "sharp_diff",
        "purity_lseg",
        "response_margin",
        "response_conf",
        "inside_mean",
        "outside_mean",
        "lifted_point_count",
        "lifted_point_ratio",
        "mask_area_ratio",
        "view_pair_count",
        "overlap_iou_mean",
        "overlap_iou_max",
        "pred_conf_mean",
        "pred_conf_std",
    ]
    arrays: Dict[str, np.ndarray] = {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in scalar_keys
    }
    arrays["fused_query"] = np.stack([row["fused_query"] for row in rows], axis=0).astype(np.float32)
    arrays["size_bucket"] = np.asarray([row["size_bucket"] for row in rows], dtype=object)
    return arrays


def _build_target_specs(
    arrays: Dict[str, np.ndarray],
    scale: float,
    target_min: float,
    target_max: float,
    sharp_weight: float,
) -> Dict[str, Dict[str, Any]]:
    if not arrays:
        return {}
    diff_mv = arrays["C_diff"]
    diff_sharp = arrays["sharp_diff"]
    diff_mv_sharp = diff_mv + sharp_weight * diff_sharp
    quality_weight = _normalize_robust(arrays["response_margin"]) * _normalize_robust(arrays["overlap_iou_mean"])
    raw_defs = {
        "mv": diff_mv,
        "sharp": diff_sharp,
        "mv+sharp": diff_mv_sharp,
        "mv+sharp+quality_input": diff_mv_sharp,
        "quality_weighted": quality_weight * diff_mv_sharp,
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, r_diff in raw_defs.items():
        g_target = 1.0 / (1.0 + np.exp(-scale * r_diff))
        g_target = np.clip(g_target, target_min, target_max)
        out[name] = {
            "R_diff": r_diff,
            "g_target": g_target,
            "R_stats": _safe_stats(r_diff.tolist()),
            "g_stats": _g_stats_from_array(g_target),
        }
    return out


def _build_feature_matrix(arrays: Dict[str, np.ndarray], feature_set: str) -> np.ndarray:
    fused = arrays["fused_query"]
    if feature_set == "fused":
        return fused
    if feature_set == "fused+mv":
        scalars = np.stack(
            [
                arrays["C_lseg"],
                arrays["C_odise"],
                arrays["C_diff"],
            ],
            axis=1,
        ).astype(np.float32)
        return np.concatenate([fused, scalars], axis=1)
    if feature_set == "fused+all":
        purity = np.nan_to_num(arrays["purity_lseg"], nan=0.0)
        scalars = np.stack(
            [
                arrays["C_lseg"],
                arrays["C_odise"],
                arrays["C_diff"],
                arrays["sharp_lseg"],
                arrays["sharp_odise"],
                arrays["sharp_diff"],
                purity,
                arrays["response_margin"],
                arrays["response_conf"],
                arrays["mask_area_ratio"],
                np.log1p(np.clip(arrays["lifted_point_count"], a_min=0.0, a_max=None)),
                arrays["lifted_point_ratio"],
                arrays["view_pair_count"],
                arrays["overlap_iou_mean"],
                arrays["overlap_iou_max"],
                arrays["pred_conf_mean"],
                arrays["pred_conf_std"],
            ],
            axis=1,
        ).astype(np.float32)
        return np.concatenate([fused, scalars], axis=1)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _signal_stats_by_size(arrays: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    if not arrays:
        return []
    rows: List[Dict[str, Any]] = []
    for size_name in ("small", "medium", "large"):
        keep = arrays["size_bucket"] == size_name
        if not bool(np.any(keep)):
            rows.append(
                {
                    "size_group": size_name,
                    "count": 0,
                    "C_diff_std": 0.0,
                    "C_diff_lseg_win_ratio": 0.0,
                    "C_diff_odise_win_ratio": 0.0,
                    "sharp_diff_std": 0.0,
                    "sharp_lseg_win_ratio": 0.0,
                    "sharp_odise_win_ratio": 0.0,
                }
            )
            continue
        c_diff = arrays["C_diff"][keep]
        sharp_diff = arrays["sharp_diff"][keep]
        rows.append(
            {
                "size_group": size_name,
                "count": int(keep.sum()),
                "C_diff_std": float(c_diff.std()),
                "C_diff_lseg_win_ratio": float((c_diff > 0.03).mean()),
                "C_diff_odise_win_ratio": float((c_diff < -0.03).mean()),
                "sharp_diff_std": float(sharp_diff.std()),
                "sharp_lseg_win_ratio": float((sharp_diff > 0.03).mean()),
                "sharp_odise_win_ratio": float((sharp_diff < -0.03).mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="No-text region-level gate probe")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max-batches", type=int, default=300)
    parser.add_argument("--max-regions", type=int, default=20000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save-per-region", action="store_true")
    parser.add_argument("--ablate-signals", action="store_true")
    parser.add_argument("--probe-learnability", action="store_true")
    parser.add_argument("--region-probe-mv-iou-thr", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    parser.add_argument("--region-probe-max-pairs-per-mask", type=int, default=10)
    parser.add_argument("--region-probe-min-lifted-points", type=int, default=5)
    parser.add_argument("--region-probe-max-pairs", type=int, default=50000)
    parser.add_argument("--region-probe-mv-weight", type=float, default=1.0)
    parser.add_argument("--region-probe-sharp-weight", type=float, default=0.5)
    parser.add_argument("--region-probe-purity-weight", type=float, default=0.5)
    parser.add_argument("--region-probe-target-scale", type=float, default=5.0)
    parser.add_argument("--region-probe-target-min", type=float, default=0.35)
    parser.add_argument("--region-probe-target-max", type=float, default=0.85)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_config = load_yaml_config(args.config) or {}
    seed = int(yaml_config.get("seed", 1342))
    set_seed(seed)

    device = yaml_config.get("device", "cuda")
    gpu_ids = yaml_config.get("gpu_ids", [0])
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    if device == "cuda" and torch.cuda.is_available() and gpu_ids:
        torch.cuda.set_device(gpu_ids[0])
        device = f"cuda:{gpu_ids[0]}"
        print(f"Using GPU: {gpu_ids[0]}")
    elif device == "cuda":
        device = "cpu"
        print("CUDA not available, using CPU")

    model, trainer, train_loader, _ = _build_probe_model_and_trainer(
        config_path=args.config,
        device=device,
        output_dir=str(out_dir),
        args_resume=args.resume,
    )
    model.eval()

    iou_thresholds = sorted({round(float(v), 4) for v in args.region_probe_mv_iou_thr})
    base_thr = 0.05 if 0.05 in iou_thresholds else iou_thresholds[0]
    threshold_rows: Dict[float, List[Dict[str, Any]]] = {thr: [] for thr in iou_thresholds}
    threshold_pair_count: Dict[float, int] = {thr: 0 for thr in iou_thresholds}
    threshold_total_regions: Dict[float, int] = {thr: 0 for thr in iou_thresholds}
    purity_lseg_available = False

    time_per_batch: List[float] = []
    time_lift_masks: List[float] = []
    time_region_pairing: List[float] = []
    time_signal_compute: List[float] = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= args.max_batches:
            break
        if all(len(rows) >= args.max_regions for rows in threshold_rows.values()):
            break
        t0 = time.perf_counter()
        batch = trainer._move_batch_to_device(batch)
        batch["sinput"] = trainer._build_sparse_tensor(batch)

        with autocast(enabled=trainer.config.use_amp):
            results = model(batch)
        scene_names = batch.get("scene_name") or []
        frame_stems = batch.get("frame_stem") or []
        pixel_embeddings_full = batch.get("pixel_embeddings")

        t1 = time.perf_counter()
        lifted, lifted_valid = build_lifted_3d_masks(
            batch["masks"],
            batch["mask_valid"],
            batch["x_label"],
            batch["y_label"],
            results["batch_indices"],
        )
        time_lift_masks.append(time.perf_counter() - t1)

        lseg_all = results.get("pixel_pooled_embeddings", batch.get("pixel_pooled"))
        if lseg_all is None:
            continue

        scene_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for b in range(len(results["outputs"])):
            if len(results["outputs"][b]) == 0:
                continue
            valid_k = results["mask_valid_from_masks"][b]
            if not valid_k.any():
                continue
            pt_mask = results["batch_indices"] == b
            if not pt_mask.any():
                continue
            pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()
            lifted_b = lifted[pt_mask][:, valid_k]
            lifted_valid_b = lifted_valid[pt_mask]
            if not bool(lifted_valid_b.any()):
                continue
            odise_q = batch["mask_embeddings"][b][valid_k].float()
            lseg_q = lseg_all[b][valid_k].float()
            fused_q = results["fused_embeddings"][b][valid_k].float()
            coords_xyz = batch["ori_coords_3d"][pt_mask][:, 1:4].long()[lifted_valid_b]
            coord_hash = _row_hash(coords_xyz)
            lifted_bool = (lifted_b[lifted_valid_b] > 0.5)
            visible_point_count = max(int(lifted_bool.shape[0]), 1)
            lifted_point_count = lifted_bool.float().sum(dim=0)
            keep = lifted_point_count >= float(args.region_probe_min_lifted_points)
            if not bool(keep.any()):
                continue
            pred_logits = pred_logits[:, keep]
            odise_q = odise_q[keep]
            lseg_q = lseg_q[keep]
            fused_q = fused_q[keep]
            lifted_bool = lifted_bool[:, keep]
            lifted_point_count = lifted_point_count[keep]
            masks_b = batch["masks"][b].float()
            mask_area = masks_b[valid_k][keep].sum(dim=(1, 2))
            image_area = float(masks_b.shape[-1] * masks_b.shape[-2])
            mask_area_ratio = (mask_area / max(image_area, 1.0)).clamp(0.0, 1.0)
            pred = torch.sigmoid(pred_logits)
            target = lifted_bool.float()
            inside_count = target.sum(dim=0)
            outside_count = (1.0 - target).sum(dim=0)
            inside_mean = (pred * target).sum(dim=0) / inside_count.clamp_min(1.0)
            outside_mean = (pred * (1.0 - target)).sum(dim=0) / outside_count.clamp_min(1.0)
            response_margin = inside_mean - outside_mean
            response_conf = (pred - 0.5).abs().mean(dim=0) * 2.0
            pred_conf_mean = pred.mean(dim=0)
            pred_conf_std = pred.std(dim=0, unbiased=False)
            lifted_point_ratio = lifted_point_count / float(visible_point_count)

            purity_lseg = None
            if pixel_embeddings_full is not None:
                purity_lseg = _compute_purity_lseg(
                    pixel_embeddings_full[b],
                    masks_b,
                    valid_k,
                    lseg_all[b][valid_k].float(),
                )
                if purity_lseg is not None:
                    purity_lseg = purity_lseg[keep]
                    purity_lseg_available = True

            k = int(keep.sum().item())
            if k < 2:
                sharp_lseg = odise_q.new_zeros(k)
                sharp_odise = odise_q.new_zeros(k)
            else:
                lseg_sim = F.normalize(lseg_q, dim=-1) @ F.normalize(lseg_q, dim=-1).t()
                odise_sim = F.normalize(odise_q, dim=-1) @ F.normalize(odise_q, dim=-1).t()
                lseg_sim.fill_diagonal_(-1.0)
                odise_sim.fill_diagonal_(-1.0)
                top_k = min(2, k - 1)
                top_l = torch.topk(lseg_sim, k=top_k, dim=1).values
                top_o = torch.topk(odise_sim, k=top_k, dim=1).values
                if top_k == 1:
                    sharp_lseg = top_l[:, 0]
                    sharp_odise = top_o[:, 0]
                else:
                    sharp_lseg = top_l[:, 0] - top_l[:, 1]
                    sharp_odise = top_o[:, 0] - top_o[:, 1]

            scene_name = str(scene_names[b]) if isinstance(scene_names, (list, tuple)) and b < len(scene_names) else str(b)
            view_id = str(frame_stems[b]) if isinstance(frame_stems, (list, tuple)) and b < len(frame_stems) else str(b)
            scene_groups[scene_name].append(
                {
                    "scene_name": scene_name,
                    "view_id": view_id,
                    "num_masks": int(keep.sum().item()),
                    "odise_q": odise_q,
                    "lseg_q": lseg_q,
                    "fused_q": fused_q,
                    "lifted_bool": lifted_bool,
                    "lifted_point_count": lifted_point_count,
                    "lifted_point_ratio": lifted_point_ratio,
                    "mask_area_ratio": mask_area_ratio,
                    "coord_hash": coord_hash,
                    "purity_lseg": purity_lseg,
                    "inside_mean": inside_mean,
                    "outside_mean": outside_mean,
                    "response_margin": response_margin,
                    "response_conf": response_conf,
                    "pred_conf_mean": pred_conf_mean,
                    "pred_conf_std": pred_conf_std,
                    "sharp_lseg": sharp_lseg,
                    "sharp_odise": sharp_odise,
                }
            )

        t2 = time.perf_counter()
        per_thr_items: Dict[float, Dict[str, List[Dict[str, Any]]]] = {}
        for thr in iou_thresholds:
            cloned_groups: Dict[str, List[Dict[str, Any]]] = {}
            for scene_name, items in scene_groups.items():
                cloned_groups[scene_name] = [dict(item) for item in items]
            per_thr_items[thr] = cloned_groups
            for scene_name, items in cloned_groups.items():
                valid_pairs, total_regions = _compute_scene_region_signals(
                    items,
                    iou_thr=float(thr),
                    max_pairs=int(args.region_probe_max_pairs_per_mask),
                )
                threshold_pair_count[thr] += valid_pairs
                threshold_total_regions[thr] += total_regions
        time_region_pairing.append(time.perf_counter() - t2)

        t3 = time.perf_counter()
        for thr, scene_groups_thr in per_thr_items.items():
            for scene_name, items in scene_groups_thr.items():
                for item in items:
                    valid_mask = item["c_valid"]
                    for idx in range(item["num_masks"]):
                        if len(threshold_rows[thr]) >= args.max_regions:
                            break
                        if not bool(valid_mask[idx]):
                            continue
                        purity_val = float(item["purity_lseg"][idx].detach().cpu()) if item["purity_lseg"] is not None else float("nan")
                        row = {
                            "threshold": float(thr),
                            "scene_name": scene_name,
                            "view_id": item["view_id"],
                            "region_index": idx,
                            "size_bucket": _size_bucket(float(item["mask_area_ratio"][idx].detach().cpu())),
                            "C_lseg": float(item["C_lseg"][idx].detach().cpu()),
                            "C_odise": float(item["C_odise"][idx].detach().cpu()),
                            "C_diff": float((item["C_lseg"][idx] - item["C_odise"][idx]).detach().cpu()),
                            "sharp_lseg": float(item["sharp_lseg"][idx].detach().cpu()),
                            "sharp_odise": float(item["sharp_odise"][idx].detach().cpu()),
                            "sharp_diff": float((item["sharp_lseg"][idx] - item["sharp_odise"][idx]).detach().cpu()),
                            "purity_lseg": purity_val,
                            "response_margin": float(item["response_margin"][idx].detach().cpu()),
                            "response_conf": float(item["response_conf"][idx].detach().cpu()),
                            "inside_mean": float(item["inside_mean"][idx].detach().cpu()),
                            "outside_mean": float(item["outside_mean"][idx].detach().cpu()),
                            "lifted_point_count": float(item["lifted_point_count"][idx].detach().cpu()),
                            "lifted_point_ratio": float(item["lifted_point_ratio"][idx].detach().cpu()),
                            "mask_area_ratio": float(item["mask_area_ratio"][idx].detach().cpu()),
                            "view_pair_count": float(item["view_pair_count"][idx].detach().cpu()),
                            "overlap_iou_mean": float(item["overlap_iou_mean"][idx].detach().cpu()),
                            "overlap_iou_max": float(item["overlap_iou_max"][idx].detach().cpu()),
                            "pred_conf_mean": float(item["pred_conf_mean"][idx].detach().cpu()),
                            "pred_conf_std": float(item["pred_conf_std"][idx].detach().cpu()),
                            "fused_query": item["fused_q"][idx].detach().cpu().numpy().astype(np.float32),
                        }
                        threshold_rows[thr].append(row)
        time_signal_compute.append(time.perf_counter() - t3)
        time_per_batch.append(time.perf_counter() - t0)

    threshold_summaries: Dict[str, Any] = {}
    signal_rows_all: List[Dict[str, Any]] = []
    learn_rows_all: List[Dict[str, Any]] = []
    size_rows_all: List[Dict[str, Any]] = []

    for thr in iou_thresholds:
        rows = threshold_rows[thr]
        arrays = _rows_to_arrays(rows)
        c_stats = _safe_stats([row["C_diff"] for row in rows])
        sharp_stats = _safe_stats([row["sharp_diff"] for row in rows])
        purity_vals = [row["purity_lseg"] for row in rows if not math.isnan(row["purity_lseg"])]
        purity_stats = _safe_stats(purity_vals)
        response_margin = arrays.get("response_margin", np.asarray([], dtype=np.float64))
        response_conf = arrays.get("response_conf", np.asarray([], dtype=np.float64))
        c_diff = arrays.get("C_diff", np.asarray([], dtype=np.float64))
        mv_sharp_proxy = c_diff + float(args.region_probe_sharp_weight) * arrays.get("sharp_diff", np.asarray([], dtype=np.float64))
        g_proxy = np.clip(
            1.0 / (1.0 + np.exp(-float(args.region_probe_target_scale) * mv_sharp_proxy)),
            float(args.region_probe_target_min),
            float(args.region_probe_target_max),
        ) if rows else np.asarray([], dtype=np.float64)
        target_specs = _build_target_specs(
            arrays,
            scale=float(args.region_probe_target_scale),
            target_min=float(args.region_probe_target_min),
            target_max=float(args.region_probe_target_max),
            sharp_weight=float(args.region_probe_sharp_weight),
        )

        learnability_rows_thr: List[Dict[str, Any]] = []
        if args.probe_learnability and rows:
            feature_sets = ["fused", "fused+mv", "fused+all"]
            feature_cache = {name: _build_feature_matrix(arrays, name) for name in feature_sets}
            target_names = list(target_specs.keys()) if args.ablate_signals else ["mv+sharp"]
            for target_name in target_names:
                targets = target_specs[target_name]["g_target"].astype(np.float32)
                for feature_name in feature_sets:
                    metrics = _train_probe_mlp(feature_cache[feature_name], targets, seed)
                    row = {"threshold": thr, "target_name": target_name, "feature_set": feature_name}
                    row.update(metrics)
                    learnability_rows_thr.append(row)
                    learn_rows_all.append(dict(row))

        target_names = list(target_specs.keys()) if args.ablate_signals else ["mv+sharp"]
        signal_rows_thr = []
        for target_name in target_names:
            spec = target_specs[target_name]
            row = {
                "threshold": thr,
                "target_name": target_name,
                "R_diff_mean": spec["R_stats"]["mean"],
                "R_diff_std": spec["R_stats"]["std"],
                "R_diff_abs_mean": spec["R_stats"]["abs_mean"],
                "R_diff_clear_003": spec["R_stats"]["clear_ratio_003"],
                "R_diff_clear_005": spec["R_stats"]["clear_ratio_005"],
                "g_target_mean": spec["g_stats"]["mean"],
                "g_target_std": spec["g_stats"]["std"],
                "g_target_mid_045_055": spec["g_stats"]["mid_ratio"],
                "g_target_lseg_ratio_06": spec["g_stats"]["lseg_ratio_06"],
                "g_target_odise_ratio_04": spec["g_stats"]["odise_ratio_04"],
            }
            signal_rows_thr.append(row)
            signal_rows_all.append(dict(row))

        size_rows_thr = _signal_stats_by_size(arrays)
        for row in size_rows_thr:
            out = {"threshold": thr}
            out.update(row)
            size_rows_all.append(out)

        threshold_summaries[str(thr)] = {
            "valid_region_count": len(rows),
            "valid_region_ratio": float(len(rows) / max(threshold_total_regions[thr], 1)),
            "valid_pair_count": threshold_pair_count[thr],
            "C_lseg_mean": float(np.mean([row["C_lseg"] for row in rows])) if rows else 0.0,
            "C_lseg_std": float(np.std([row["C_lseg"] for row in rows])) if rows else 0.0,
            "C_odise_mean": float(np.mean([row["C_odise"] for row in rows])) if rows else 0.0,
            "C_odise_std": float(np.std([row["C_odise"] for row in rows])) if rows else 0.0,
            "C_diff_mean": c_stats["mean"],
            "C_diff_std": c_stats["std"],
            "C_diff_clear_003": c_stats["clear_ratio_003"],
            "C_diff_clear_005": c_stats["clear_ratio_005"],
            "C_diff_lseg_win_ratio": float((c_diff > 0.03).mean()) if c_diff.size else 0.0,
            "C_diff_odise_win_ratio": float((c_diff < -0.03).mean()) if c_diff.size else 0.0,
            "sharp_lseg_mean": float(np.mean([row["sharp_lseg"] for row in rows])) if rows else 0.0,
            "sharp_lseg_std": float(np.std([row["sharp_lseg"] for row in rows])) if rows else 0.0,
            "sharp_odise_mean": float(np.mean([row["sharp_odise"] for row in rows])) if rows else 0.0,
            "sharp_odise_std": float(np.std([row["sharp_odise"] for row in rows])) if rows else 0.0,
            "sharp_diff_std": sharp_stats["std"],
            "sharp_diff_clear_003": sharp_stats["clear_ratio_003"],
            "purity_lseg_mean": purity_stats["mean"],
            "purity_lseg_std": purity_stats["std"],
            "purity_lseg_available": bool(purity_vals),
            "purity_lseg_clear_low_ratio": float((np.asarray(purity_vals) < 0.5).mean()) if purity_vals else 0.0,
            "response_margin_mean": float(response_margin.mean()) if response_margin.size else 0.0,
            "response_margin_std": float(response_margin.std()) if response_margin.size else 0.0,
            "response_margin_clear_01": float((np.abs(response_margin) > 0.1).mean()) if response_margin.size else 0.0,
            "response_conf_mean": float(response_conf.mean()) if response_conf.size else 0.0,
            "response_conf_std": float(response_conf.std()) if response_conf.size else 0.0,
            "inside_mean_mean": float(np.mean([row["inside_mean"] for row in rows])) if rows else 0.0,
            "inside_mean_std": float(np.std([row["inside_mean"] for row in rows])) if rows else 0.0,
            "outside_mean_mean": float(np.mean([row["outside_mean"] for row in rows])) if rows else 0.0,
            "outside_mean_std": float(np.std([row["outside_mean"] for row in rows])) if rows else 0.0,
            "mask_area_ratio_mean": float(np.mean([row["mask_area_ratio"] for row in rows])) if rows else 0.0,
            "mask_area_ratio_std": float(np.std([row["mask_area_ratio"] for row in rows])) if rows else 0.0,
            "lifted_point_count_mean": float(np.mean([row["lifted_point_count"] for row in rows])) if rows else 0.0,
            "lifted_point_count_std": float(np.std([row["lifted_point_count"] for row in rows])) if rows else 0.0,
            "view_pair_count_mean": float(np.mean([row["view_pair_count"] for row in rows])) if rows else 0.0,
            "view_pair_count_std": float(np.std([row["view_pair_count"] for row in rows])) if rows else 0.0,
            "overlap_iou_mean": float(np.mean([row["overlap_iou_mean"] for row in rows])) if rows else 0.0,
            "overlap_iou_std": float(np.std([row["overlap_iou_mean"] for row in rows])) if rows else 0.0,
            "corr_response_margin_C_diff": _pearson(response_margin, c_diff) if response_margin.size else 0.0,
            "corr_response_conf_abs_C_diff": _pearson(response_conf, np.abs(c_diff)) if response_conf.size else 0.0,
            "corr_response_margin_g_target": _pearson(response_margin, g_proxy) if response_margin.size else 0.0,
            "corr_mask_area_ratio_C_diff": _pearson(arrays.get("mask_area_ratio", np.asarray([], dtype=np.float64)), c_diff) if rows else 0.0,
            "corr_lifted_point_count_C_diff": _pearson(arrays.get("lifted_point_count", np.asarray([], dtype=np.float64)), c_diff) if rows else 0.0,
            "corr_view_pair_count_abs_C_diff": _pearson(arrays.get("view_pair_count", np.asarray([], dtype=np.float64)), np.abs(c_diff)) if rows else 0.0,
            "corr_overlap_iou_mean_abs_C_diff": _pearson(arrays.get("overlap_iou_mean", np.asarray([], dtype=np.float64)), np.abs(c_diff)) if rows else 0.0,
            "target_ablation": signal_rows_thr,
            "learnability_corr": max([row["pearson_corr"] for row in learnability_rows_thr], default=0.0),
            "region_size_breakdown": size_rows_thr,
        }

    base_summary = threshold_summaries[str(base_thr)]
    base_targets = threshold_summaries[str(base_thr)]["target_ablation"]
    base_learn = [row for row in learn_rows_all if abs(float(row["threshold"]) - float(base_thr)) < 1e-6]
    best_target = max(base_targets, key=lambda row: (row["g_target_std"], row["R_diff_clear_003"])) if base_targets else {}
    best_feature = max(base_learn, key=lambda row: (row["pearson_corr"], row["r2_score"])) if base_learn else {}
    signal_ranking = []
    for row in base_targets:
        signal_ranking.append(
            {
                "name": row["target_name"],
                "diff_std": row["R_diff_std"],
                "clear_003": row["R_diff_clear_003"],
                "learn_corr": max([x["pearson_corr"] for x in base_learn if x["target_name"] == row["target_name"]], default=0.0),
            }
        )
    signal_ranking.sort(key=lambda x: (x["learn_corr"], x["diff_std"], x["clear_003"]), reverse=True)

    recommendation = "keep_rule_based_projected_gate"
    if best_target:
        if best_target["g_target_std"] >= 0.08 and best_target["R_diff_clear_003"] >= 0.5:
            if best_feature and best_feature.get("pearson_corr", 0.0) >= 0.4 and best_feature.get("r2_score", 0.0) > 0.1:
                recommendation = "train_learned_region_gate"
            else:
                recommendation = "add_more_region_features_or_keep_rule_gate"
        elif best_feature and (best_feature.get("pearson_corr", 0.0) < 0.2 or best_feature.get("r2_score", 0.0) <= 0.0):
            recommendation = "keep_rule_based_projected_gate"

    summary = {
        "purity_lseg_available": purity_lseg_available,
        "iou_thresholds": iou_thresholds,
        "base_threshold": base_thr,
        "threshold_summaries": threshold_summaries,
        "signal_ranking": signal_ranking,
        "best_target": best_target,
        "best_feature_set": best_feature,
        "speed": {
            "time_per_batch": float(np.mean(time_per_batch)) if time_per_batch else 0.0,
            "time_lift_masks": float(np.mean(time_lift_masks)) if time_lift_masks else 0.0,
            "time_region_pairing": float(np.mean(time_region_pairing)) if time_region_pairing else 0.0,
            "time_signal_compute": float(np.mean(time_signal_compute)) if time_signal_compute else 0.0,
            "estimated_epoch_time": float((np.mean(time_per_batch) if time_per_batch else 0.0) * len(train_loader)),
        },
        "recommendation": recommendation,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.save_per_region:
        with open(out_dir / "region_signal_stats.csv", "w", newline="") as f:
            fieldnames = [
                "threshold", "scene_name", "view_id", "region_index", "size_bucket",
                "C_lseg", "C_odise", "C_diff", "sharp_lseg", "sharp_odise", "sharp_diff",
                "purity_lseg", "response_margin", "response_conf", "inside_mean", "outside_mean",
                "lifted_point_count", "lifted_point_ratio", "mask_area_ratio", "view_pair_count",
                "overlap_iou_mean", "overlap_iou_max", "pred_conf_mean", "pred_conf_std",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for thr in iou_thresholds:
                for row in threshold_rows[thr]:
                    writer.writerow({k: row[k] for k in fieldnames})

    with open(out_dir / "signal_ablation.csv", "w", newline="") as f:
        fieldnames = [
            "threshold", "target_name", "R_diff_mean", "R_diff_std", "R_diff_abs_mean",
            "R_diff_clear_003", "R_diff_clear_005", "g_target_mean", "g_target_std",
            "g_target_mid_045_055", "g_target_lseg_ratio_06", "g_target_odise_ratio_04",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in signal_rows_all:
            writer.writerow(row)

    with open(out_dir / "learnability_ablation.csv", "w", newline="") as f:
        fieldnames = [
            "threshold", "target_name", "feature_set", "train_mse", "val_mse", "pearson_corr",
            "spearman_corr", "r2_score", "pred_mean", "pred_std", "target_mean", "target_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in learn_rows_all:
            writer.writerow(row)

    with open(out_dir / "region_size_breakdown.csv", "w", newline="") as f:
        fieldnames = [
            "threshold", "size_group", "count", "C_diff_std", "C_diff_lseg_win_ratio",
            "C_diff_odise_win_ratio", "sharp_diff_std", "sharp_lseg_win_ratio", "sharp_odise_win_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in size_rows_all:
            writer.writerow(row)

    base_rows = threshold_rows[base_thr]
    base_arrays = _rows_to_arrays(base_rows)
    base_target_specs = _build_target_specs(
        base_arrays,
        scale=float(args.region_probe_target_scale),
        target_min=float(args.region_probe_target_min),
        target_max=float(args.region_probe_target_max),
        sharp_weight=float(args.region_probe_sharp_weight),
    ) if base_rows else {}
    _save_histogram(out_dir / "C_diff_hist.png", [row["C_diff"] for row in base_rows], "C_diff histogram", "C_diff")
    _save_histogram(out_dir / "sharp_diff_hist.png", [row["sharp_diff"] for row in base_rows], "sharp_diff histogram", "sharp_diff")
    _save_histogram(out_dir / "g_target_hist_mv.png", base_target_specs.get("mv", {}).get("g_target", np.asarray([], dtype=np.float64)).tolist(), "g_target mv histogram", "g_target_mv")
    _save_histogram(out_dir / "g_target_hist_mv_sharp.png", base_target_specs.get("mv+sharp", {}).get("g_target", np.asarray([], dtype=np.float64)).tolist(), "g_target mv+sharp histogram", "g_target_mv_sharp")

    print("[NoText Region Gate Probe V2]")
    for thr in iou_thresholds:
        thr_summary = threshold_summaries[str(thr)]
        print(
            f"iou_thr={thr:.2f}  valid_region_count={thr_summary['valid_region_count']}  "
            f"valid_pair_count={thr_summary['valid_pair_count']}  C_diff_std={thr_summary['C_diff_std']:.4f}  "
            f"C_diff_clear@0.03={thr_summary['C_diff_clear_003']:.4f}  "
            f"g_target_std={max((row['g_target_std'] for row in thr_summary['target_ablation']), default=0.0):.4f}  "
            f"learnability_corr={thr_summary['learnability_corr']:.4f}"
        )
    print("[Signal 2: MV Consistency]")
    print(
        f"C_lseg={base_summary['C_lseg_mean']:.4f}/{base_summary['C_lseg_std']:.4f}  "
        f"C_odise={base_summary['C_odise_mean']:.4f}/{base_summary['C_odise_std']:.4f}  "
        f"C_diff_std={base_summary['C_diff_std']:.4f}  C_diff_clear@0.03={base_summary['C_diff_clear_003']:.4f}  "
        f"lseg_win={base_summary['C_diff_lseg_win_ratio']:.4f}  odise_win={base_summary['C_diff_odise_win_ratio']:.4f}"
    )
    print("[Signal 5: Sharpness]")
    print(
        f"sharp_lseg={base_summary['sharp_lseg_mean']:.4f}/{base_summary['sharp_lseg_std']:.4f}  "
        f"sharp_odise={base_summary['sharp_odise_mean']:.4f}/{base_summary['sharp_odise_std']:.4f}  "
        f"sharp_diff_std={base_summary['sharp_diff_std']:.4f}  sharp_diff_clear@0.03={base_summary['sharp_diff_clear_003']:.4f}"
    )
    print("[Signal 3/4: Quality Correlation]")
    print(
        f"corr(response_margin, C_diff)={base_summary['corr_response_margin_C_diff']:.4f}  "
        f"corr(mask_area_ratio, C_diff)={base_summary['corr_mask_area_ratio_C_diff']:.4f}  "
        f"corr(lifted_point_count, C_diff)={base_summary['corr_lifted_point_count_C_diff']:.4f}  "
        f"corr(overlap_iou_mean, abs(C_diff))={base_summary['corr_overlap_iou_mean_abs_C_diff']:.4f}"
    )
    print("[Target Ablation]")
    for row in base_targets:
        print(
            f"target={row['target_name']} R_diff_std={row['R_diff_std']:.4f} clear@0.03={row['R_diff_clear_003']:.4f} "
            f"g_target_std={row['g_target_std']:.4f} g_mid={row['g_target_mid_045_055']:.4f}"
        )
    print("[Learnability Ablation]")
    for row in base_learn:
        print(f"target={row['target_name']} feature={row['feature_set']} corr={row['pearson_corr']:.4f} r2={row['r2_score']:.4f}")
    print("[Decision]")
    print(
        f"best_target={best_target.get('target_name', 'none')}  "
        f"best_feature_set={best_feature.get('feature_set', 'none')}  recommendation={recommendation}"
    )
    print("[Speed]")
    print(
        f"time_per_batch={summary['speed']['time_per_batch']:.4f}s  "
        f"time_lift_masks={summary['speed']['time_lift_masks']:.4f}s  "
        f"time_region_pairing={summary['speed']['time_region_pairing']:.4f}s  "
        f"time_signal_compute={summary['speed']['time_signal_compute']:.4f}s  "
        f"estimated_epoch_time={summary['speed']['estimated_epoch_time']:.1f}s"
    )


if __name__ == "__main__":
    main()
