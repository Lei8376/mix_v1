#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
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
import sys

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
    _mask_feature_class_probs_tau,
)
from experiment_mask_distill.criterion_mask_distill import build_lifted_3d_masks  # noqa: E402
from experiment_mask_distill.semantic_miou import diff2scene_class_probs_predict  # noqa: E402
from evaluate.semantic_iou import _SemanticAccumulator  # noqa: E402


def _avg_pairwise_cos(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2:
        return x.new_tensor(0.0)
    sim = x @ x.t()
    idx = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    return sim[idx[0], idx[1]].mean()


def _safe_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_mean": 0.0,
            "positive_ratio": 0.0,
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
        "positive_ratio": float((arr > 0).mean()),
        "clear_ratio_003": float((np.abs(arr) > 0.03).mean()),
        "clear_ratio_005": float((np.abs(arr) > 0.05).mean()),
    }


def _binary_entropy(x: np.ndarray) -> float:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    ent = -(x * np.log(x) + (1.0 - x) * np.log(1.0 - x))
    return float(ent.mean())


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std < 1e-8 or y_std < 1e-8:
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


def _build_model_and_probe_trainer(
    config_path: str,
    device: str,
    output_dir: str,
    args_resume: str,
) -> Tuple[OpenVocab3DFusionModelV2, MaskDistillTrainer, Any, Any, Dict[str, Any]]:
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
    train_loader, val_loader = create_data_loaders(dataset_config, dataloader_config)

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
        use_source_reliability_gate=bool(_model.get("use_source_reliability_gate", False)),
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
        semantic_clip_model=str(_trainer.get("semantic_clip_model", "ODISE-256")),
        semantic_pixel_clip_model=str(_trainer.get("semantic_pixel_clip_model", "ViT-B/32")),
        semantic_prompt_template=str(_trainer.get("semantic_prompt_template", "a photo of a {}")),
        semantic_pc_lambda=float(_trainer.get("semantic_pc_lambda", 0.5)),
        dual_space_eval=bool(_trainer.get("dual_space_eval", True)),
        dual_space_odise_weight=float(_trainer.get("dual_space_odise_weight", 0.5)),
        dual_space_lseg_weight=float(_trainer.get("dual_space_lseg_weight", 0.5)),
        dual_space_tau_odise=float(_trainer.get("dual_space_tau_odise", 0.07)),
        dual_space_tau_lseg=float(_trainer.get("dual_space_tau_lseg", 0.07)),
        dual_space_conf_min=float(_trainer.get("dual_space_conf_min", 0.2)),
        dual_space_conf_max=float(_trainer.get("dual_space_conf_max", 0.7)),
        semantic_readout_mode=str(_trainer.get("semantic_readout_mode", "projected_gate")),
        semantic_readout_ablation=True,
        semantic_size_aware=bool(_trainer.get("semantic_size_aware", True)),
        semantic_small_area_thr=float(_trainer.get("semantic_small_area_thr", 0.01)),
        semantic_medium_area_thr=float(_trainer.get("semantic_medium_area_thr", 0.10)),
        semantic_small_lseg_weight=float(_trainer.get("semantic_small_lseg_weight", 0.45)),
        semantic_medium_lseg_weight=float(_trainer.get("semantic_medium_lseg_weight", 0.65)),
        semantic_large_lseg_weight=float(_trainer.get("semantic_large_lseg_weight", 0.80)),
        semantic_projected_gate=bool(_trainer.get("semantic_projected_gate", True)),
        semantic_projected_gate_scale=float(_trainer.get("semantic_projected_gate_scale", 10.0)),
        semantic_projected_gate_min=float(_trainer.get("semantic_projected_gate_min", 0.45)),
        semantic_projected_gate_max=float(_trainer.get("semantic_projected_gate_max", 0.85)),
        semantic_projected_gate_default=float(_trainer.get("semantic_projected_gate_default", 0.70)),
        semantic_projected_size_gate=bool(_trainer.get("semantic_projected_size_gate", True)),
        semantic_projected_size_base=float(_trainer.get("semantic_projected_size_base", 0.65)),
        semantic_projected_size_beta=float(_trainer.get("semantic_projected_size_beta", 1.0)),
        semantic_projected_size_gamma=float(_trainer.get("semantic_projected_size_gamma", 0.20)),
        semantic_projected_size_min=float(_trainer.get("semantic_projected_size_min", 0.35)),
        semantic_projected_size_max=float(_trainer.get("semantic_projected_size_max", 0.85)),
        projected_sem_probe_min_views=int(_trainer.get("point_gate_min_views", 2)),
        point_gate_target_scale=float(_trainer.get("point_gate_target_scale", 10.0)),
        point_gate_target_min=float(_trainer.get("point_gate_target_min", 0.45)),
        point_gate_target_max=float(_trainer.get("point_gate_target_max", 0.85)),
        point_gate_target_default=float(_trainer.get("point_gate_target_default", 0.70)),
        point_gate_min_views=int(_trainer.get("point_gate_min_views", 2)),
        point_gate_max_points=int(_trainer.get("point_gate_max_points", 20000)),
        point_gate_loss_type=str(_trainer.get("point_gate_loss_type", "mse")),
        point_gate_detach_target=bool(_trainer.get("point_gate_detach_target", True)),
        multiview_batch=multiview_batch,
        scenes_per_batch=scenes_per_batch,
        views_per_scene=views_per_scene,
        enable_verbose_legacy_probes=False,
        enable_legacy_source_gate_logs=False,
        source_gate_train=False,
        source_gate_training_target="none",
    )
    trainer = MaskDistillTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device,
    )
    return model, trainer, train_loader, val_loader, yaml_config


def _topk_abs_mean(diffs: torch.Tensor, k: int = 8) -> torch.Tensor:
    if diffs.numel() == 0:
        return diffs.new_tensor(0.0)
    k = min(int(k), int(diffs.numel()))
    idx = torch.topk(diffs.abs(), k=k, largest=True).indices
    return diffs[idx].mean()


def _make_learnability_features(
    fused_query: torch.Tensor,
    pooled_3d: torch.Tensor,
    mask_area_ratio: float,
    point_count: float,
    pred_conf_mean: float,
) -> Dict[str, np.ndarray]:
    fused_np = fused_query.detach().cpu().numpy().astype(np.float32)
    pooled_np = pooled_3d.detach().cpu().numpy().astype(np.float32)
    scalars = np.asarray(
        [
            mask_area_ratio,
            math.log1p(max(point_count, 0.0)),
            pred_conf_mean,
        ],
        dtype=np.float32,
    )
    return {
        "fused_query": fused_np,
        "pooled_3d_feature": pooled_np,
        "fused_query+pooled_3d": np.concatenate([fused_np, pooled_np], axis=0),
        "fused_query+pooled_3d+stats": np.concatenate([fused_np, pooled_np, scalars], axis=0),
    }


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
            "corr": 0.0,
            "spearman": 0.0,
            "r2": 0.0,
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
        "corr": _pearson(val_pred, y_val),
        "spearman": _spearman(val_pred, y_val),
        "r2": _r2_score(y_val, val_pred),
        "pred_mean": float(val_pred.mean()),
        "pred_std": float(val_pred.std()),
        "target_mean": float(y_val.mean()),
        "target_std": float(y_val.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe point-derived region gate targets")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
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

    model, trainer, train_loader, val_loader, yaml_config = _build_model_and_probe_trainer(
        config_path=args.config,
        device=device,
        output_dir=args.output,
        args_resume=args.resume,
    )
    probe_loader = val_loader if val_loader is not None else train_loader
    model.eval()

    text_feats = trainer._get_text_features()
    pixel_text_feats = trainer._get_pixel_text_features()

    total_points = 0
    total_valid_points = 0
    total_regions = 0
    total_valid_regions = 0
    point_diffs: List[float] = []
    point_c_odise: List[float] = []
    point_c_lseg: List[float] = []
    region_diff_weighted: List[float] = []
    region_targets: List[float] = []
    readout_accs = {
        "odise_only": _SemanticAccumulator(),
        "lseg_only": _SemanticAccumulator(),
        "fixed_05": _SemanticAccumulator(),
        "projected_gate": _SemanticAccumulator(),
        "point_derived_region_oracle": _SemanticAccumulator(),
    }
    region_feature_rows: Dict[str, List[np.ndarray]] = defaultdict(list)
    region_targets_rows: List[float] = []

    batch_total_times = []
    batch_target_times = []
    batch_region_times = []

    for batch_idx, batch in enumerate(probe_loader):
        if batch_idx >= args.max_batches:
            break
        t0 = time.perf_counter()
        batch = trainer._move_batch_to_device(batch)
        batch["sinput"] = trainer._build_sparse_tensor(batch)

        with autocast(enabled=trainer.config.use_amp):
            results = model(batch)
        outputs = results.get("outputs", [])
        lseg_all = results.get("pixel_pooled_embeddings", batch.get("pixel_pooled"))
        if lseg_all is None:
            continue
        with torch.no_grad():
            lifted, lifted_valid = build_lifted_3d_masks(
                batch["masks"],
                batch["mask_valid"],
                batch["x_label"],
                batch["y_label"],
                results["batch_indices"],
            )
        semantic_eval_aux = trainer._build_eval_semantic_aux(
            results=results,
            batch=batch,
            lifted=lifted,
            lifted_valid=lifted_valid,
            lseg_all=lseg_all,
        )

        t1 = time.perf_counter()
        scene_names = batch.get("scene_name") or []
        frame_stems = batch.get("frame_stem") or []
        point_to_entries = defaultdict(list)
        point_to_views = defaultdict(set)

        for b in range(len(outputs)):
            if len(outputs[b]) == 0:
                continue
            valid_k = results["mask_valid_from_masks"][b]
            if not valid_k.any():
                continue
            pt_mask = results["batch_indices"] == b
            if not pt_mask.any():
                continue
            pred_logits = outputs[b][0]["pred_mask_logits"][:, valid_k].float()
            lifted_b = lifted[pt_mask][:, valid_k]
            lifted_valid_b = lifted_valid[pt_mask]
            odise_q = batch["mask_embeddings"][b][valid_k].float()
            lseg_q = lseg_all[b][valid_k].float()
            point_coords = batch["ori_coords_3d"][pt_mask][:, 1:4].long()
            point_conf = torch.sigmoid(pred_logits)
            global_indices = pt_mask.nonzero(as_tuple=True)[0]
            scene_name = (
                str(scene_names[b])
                if isinstance(scene_names, (list, tuple)) and b < len(scene_names)
                else str(b)
            )
            view_id = (
                str(frame_stems[b])
                if isinstance(frame_stems, (list, tuple)) and b < len(frame_stems)
                else str(b)
            )
            total_points += int(point_coords.shape[0])

            for n in range(point_coords.shape[0]):
                if not bool(lifted_valid_b[n]):
                    continue
                mask_ids = torch.where(lifted_b[n] > 0.5)[0]
                if mask_ids.numel() == 0:
                    continue
                choose_idx = mask_ids[point_conf[n, mask_ids].argmax()]
                key = (
                    scene_name,
                    int(point_coords[n, 0].item()),
                    int(point_coords[n, 1].item()),
                    int(point_coords[n, 2].item()),
                )
                point_to_entries[key].append(
                    (
                        int(global_indices[n].item()),
                        b,
                        int(choose_idx.item()),
                        view_id,
                        odise_q[choose_idx].detach(),
                        lseg_q[choose_idx].detach(),
                    )
                )
                point_to_views[key].add(view_id)

        point_diff_full = torch.zeros(results["pred_3d"].shape[0], dtype=torch.float32, device=results["pred_3d"].device)
        point_valid_full = torch.zeros(results["pred_3d"].shape[0], dtype=torch.bool, device=results["pred_3d"].device)
        region_point_map = defaultdict(list)
        valid_keys = [
            key for key, views in point_to_views.items()
            if len(views) >= int(trainer.config.point_gate_min_views)
        ]
        max_points = int(trainer.config.point_gate_max_points)
        if max_points > 0 and len(valid_keys) > max_points:
            perm = torch.randperm(len(valid_keys), device=results["pred_3d"].device)[:max_points].cpu().tolist()
            valid_keys = [valid_keys[idx] for idx in perm]

        for key in valid_keys:
            entries = point_to_entries[key]
            odise_stack = torch.stack([entry[4] for entry in entries], dim=0).float()
            lseg_stack = torch.stack([entry[5] for entry in entries], dim=0).float()
            c_odise = _avg_pairwise_cos(F.normalize(odise_stack, dim=-1))
            c_lseg = _avg_pairwise_cos(F.normalize(lseg_stack, dim=-1))
            diff = c_lseg - c_odise
            point_diffs.append(float(diff.detach().cpu()))
            point_c_odise.append(float(c_odise.detach().cpu()))
            point_c_lseg.append(float(c_lseg.detach().cpu()))
            unique_entries = {}
            for global_idx, b, local_idx, _, _, _ in entries:
                unique_entries[global_idx] = (b, local_idx)
            for global_idx, (b, local_idx) in unique_entries.items():
                point_diff_full[global_idx] = diff
                point_valid_full[global_idx] = True
                region_point_map[(b, local_idx)].append(global_idx)

        total_valid_points += int(point_valid_full.sum().item())
        t2 = time.perf_counter()

        for b in range(len(outputs)):
            if len(outputs[b]) == 0:
                continue
            valid_k = results["mask_valid_from_masks"][b]
            if not valid_k.any():
                continue
            pt_mask = results["batch_indices"] == b
            pred_logits = outputs[b][0]["pred_mask_logits"][:, valid_k].float()
            gt_b = batch["binary_label_3d"][pt_mask]
            odise_q = batch["mask_embeddings"][b][valid_k].float()
            lseg_q = lseg_all[b][valid_k].float()
            image_area = float(batch["masks"][b].shape[-1] * batch["masks"][b].shape[-2])
            mask_area = batch["masks"][b][valid_k].float().sum(dim=(1, 2))
            mask_area_ratio = (mask_area / max(image_area, 1.0)).clamp(0.0, 1.0)
            projected_diff = semantic_eval_aux["projected_mask_diff"].get(b)
            projected_valid = semantic_eval_aux["projected_mask_valid"].get(b)

            p_odise = _mask_feature_class_probs_tau(
                odise_q,
                text_feats,
                trainer.config.dual_space_tau_odise,
            )
            p_lseg = _mask_feature_class_probs_tau(
                lseg_q,
                pixel_text_feats,
                trainer.config.dual_space_tau_lseg,
            )
            semantic_probs = trainer._compute_semantic_readout_probs(
                p_odise=p_odise,
                p_lseg=p_lseg,
                mask_area_ratio=mask_area_ratio,
                projected_diff=projected_diff,
                projected_valid=projected_valid,
            )

            region_valid_flags = []
            region_weighted_diffs = []
            g_target_region = torch.full(
                (int(valid_k.sum().item()),),
                float(trainer.config.point_gate_target_default),
                device=pred_logits.device,
                dtype=torch.float32,
            )
            total_regions += int(valid_k.sum().item())
            pred_3d_b = results["pred_3d"][pt_mask].float()
            for local_idx in range(int(valid_k.sum().item())):
                point_ids = region_point_map.get((b, local_idx), [])
                if not point_ids:
                    region_valid_flags.append(False)
                    region_weighted_diffs.append(0.0)
                    continue
                local_point_mask = point_valid_full[pt_mask]
                global_indices = pt_mask.nonzero(as_tuple=True)[0]
                selected_local = []
                selected_weights = []
                selected_diffs = []
                for gid in point_ids:
                    local_pos = int((global_indices == gid).nonzero(as_tuple=True)[0][0].item())
                    selected_local.append(local_pos)
                    selected_weights.append(float(torch.sigmoid(pred_logits[local_pos, local_idx]).item()))
                    selected_diffs.append(float(point_diff_full[gid].item()))
                diff_tensor = torch.tensor(selected_diffs, device=pred_logits.device, dtype=torch.float32)
                weight_tensor = torch.tensor(selected_weights, device=pred_logits.device, dtype=torch.float32)
                diff_mean = diff_tensor.mean()
                diff_median = diff_tensor.median()
                diff_weighted = (diff_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)
                diff_topk = _topk_abs_mean(diff_tensor, k=8)
                _ = (diff_mean, diff_median, diff_topk)
                g_region = torch.sigmoid(float(trainer.config.point_gate_target_scale) * diff_weighted)
                g_region = g_region.clamp(
                    float(trainer.config.point_gate_target_min),
                    float(trainer.config.point_gate_target_max),
                )
                g_target_region[local_idx] = g_region
                region_valid_flags.append(True)
                region_weighted_diffs.append(float(diff_weighted.detach().cpu()))
                region_diff_weighted.append(float(diff_weighted.detach().cpu()))
                region_targets.append(float(g_region.detach().cpu()))
                total_valid_regions += 1

                # Learnability features
                local_point_idx = torch.tensor(selected_local, device=pred_logits.device, dtype=torch.long)
                pooled_weights = torch.sigmoid(pred_logits[local_point_idx, local_idx]).float()
                pooled_3d = (
                    pred_3d_b[local_point_idx] * pooled_weights[:, None]
                ).sum(dim=0) / pooled_weights.sum().clamp_min(1e-6)
                pred_conf_mean = float(pooled_weights.mean().detach().cpu())
                feature_dict = _make_learnability_features(
                    fused_query=results["fused_embeddings"][b][valid_k][local_idx].float(),
                    pooled_3d=pooled_3d.float(),
                    mask_area_ratio=float(mask_area_ratio[local_idx].item()),
                    point_count=float(len(selected_local)),
                    pred_conf_mean=pred_conf_mean,
                )
                for feat_name, feat_vec in feature_dict.items():
                    region_feature_rows[feat_name].append(feat_vec)
                region_targets_rows.append(float(g_region.detach().cpu()))

            oracle_probs = g_target_region[:, None] * p_lseg + (1.0 - g_target_region[:, None]) * p_odise
            preds = {
                "odise_only": diff2scene_class_probs_predict(pred_logits, p_odise),
                "lseg_only": diff2scene_class_probs_predict(pred_logits, p_lseg),
                "fixed_05": diff2scene_class_probs_predict(pred_logits, 0.5 * p_lseg + 0.5 * p_odise),
                "projected_gate": diff2scene_class_probs_predict(pred_logits, semantic_probs["projected_gate"]),
                "point_derived_region_oracle": diff2scene_class_probs_predict(pred_logits, oracle_probs),
            }
            gt_cpu = gt_b.detach().cpu().long()
            for name, pred in preds.items():
                readout_accs[name].update_labels(pred, gt_cpu)

        t3 = time.perf_counter()
        batch_total_times.append(t3 - t0)
        batch_target_times.append(t2 - t1)
        batch_region_times.append(t3 - t2)

    point_stats = _safe_stats(point_diffs)
    point_summary = {
        "point_valid_count": float(total_valid_points),
        "point_valid_ratio": float(total_valid_points / max(total_points, 1)),
        "c_lseg_mean": float(np.mean(point_c_lseg)) if point_c_lseg else 0.0,
        "c_lseg_std": float(np.std(point_c_lseg)) if point_c_lseg else 0.0,
        "c_odise_mean": float(np.mean(point_c_odise)) if point_c_odise else 0.0,
        "c_odise_std": float(np.std(point_c_odise)) if point_c_odise else 0.0,
        "diff_point_mean": point_stats["mean"],
        "diff_point_std": point_stats["std"],
        "diff_point_min": point_stats["min"],
        "diff_point_max": point_stats["max"],
        "diff_point_abs_mean": point_stats["abs_mean"],
        "diff_point_positive_ratio": point_stats["positive_ratio"],
        "diff_point_clear_ratio_003": point_stats["clear_ratio_003"],
        "diff_point_clear_ratio_005": point_stats["clear_ratio_005"],
    }

    region_stats = _safe_stats(region_diff_weighted)
    g_target_arr = np.asarray(region_targets, dtype=np.float64)
    region_summary = {
        "region_valid_count": float(total_valid_regions),
        "region_valid_ratio": float(total_valid_regions / max(total_regions, 1)),
        "diff_region_mean": region_stats["mean"],
        "diff_region_std": region_stats["std"],
        "diff_region_min": region_stats["min"],
        "diff_region_max": region_stats["max"],
        "diff_region_abs_mean": region_stats["abs_mean"],
        "diff_region_clear_ratio_003": region_stats["clear_ratio_003"],
        "diff_region_clear_ratio_005": region_stats["clear_ratio_005"],
        "g_target_mean": float(g_target_arr.mean()) if g_target_arr.size else 0.0,
        "g_target_std": float(g_target_arr.std()) if g_target_arr.size else 0.0,
        "g_target_min": float(g_target_arr.min()) if g_target_arr.size else 0.0,
        "g_target_max": float(g_target_arr.max()) if g_target_arr.size else 0.0,
        "g_target_entropy": _binary_entropy(g_target_arr) if g_target_arr.size else 0.0,
        "g_target_lseg_ratio_06": float((g_target_arr > 0.6).mean()) if g_target_arr.size else 0.0,
        "g_target_odise_ratio_04": float((g_target_arr < 0.4).mean()) if g_target_arr.size else 0.0,
        "g_target_mid_ratio_045_055": float(((g_target_arr >= 0.45) & (g_target_arr <= 0.55)).mean()) if g_target_arr.size else 0.0,
    }

    oracle_summary = {}
    for name, acc in readout_accs.items():
        res = acc.compute(f"semantic_miou_{name}")
        oracle_summary[name] = {
            "miou": float(res[f"semantic_miou_{name}"]),
            "macc": float(res[f"semantic_macc_{name}"]),
        }

    t_learn0 = time.perf_counter()
    learnability_rows = []
    learnability_summary = {}
    if region_targets_rows:
        targets_np = np.asarray(region_targets_rows, dtype=np.float32)
        for feat_name, feature_list in region_feature_rows.items():
            feat_np = np.stack(feature_list, axis=0).astype(np.float32)
            metrics = _train_probe_mlp(feat_np, targets_np, seed=seed)
            metrics["feature"] = feat_name
            learnability_rows.append(metrics)
            learnability_summary[feat_name] = metrics
    t_learn1 = time.perf_counter()

    speed_summary = {
        "time_per_batch": float(np.mean(batch_total_times)) if batch_total_times else 0.0,
        "time_build_point_target": float(np.mean(batch_target_times)) if batch_target_times else 0.0,
        "time_aggregate_region": float(np.mean(batch_region_times)) if batch_region_times else 0.0,
        "time_gate_probe_train": float(t_learn1 - t_learn0),
        "estimated_epoch_time": float((np.mean(batch_total_times) if batch_total_times else 0.0) * len(probe_loader)),
        "num_batches_used": int(min(args.max_batches, len(probe_loader))),
    }

    summary = {
        "point": point_summary,
        "region": region_summary,
        "oracle_readout": oracle_summary,
        "learnability": learnability_summary,
        "speed": speed_summary,
    }

    out_dir = Path(args.output)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _save_histogram(out_dir / "diff_region_hist.png", region_diff_weighted, "Region diff histogram", "diff_region")
    _save_histogram(out_dir / "g_target_hist.png", region_targets, "Region gate target histogram", "g_target_region")

    with open(out_dir / "learnability.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature",
                "train_mse",
                "val_mse",
                "corr",
                "spearman",
                "r2",
                "pred_mean",
                "pred_std",
                "target_mean",
                "target_std",
            ],
        )
        writer.writeheader()
        for row in learnability_rows:
            writer.writerow(row)

    print("[PointDerivedRegionGate Probe]")
    print(
        f"point_valid_ratio={point_summary['point_valid_ratio']:.4f}  "
        f"diff_point_mean={point_summary['diff_point_mean']:.4f}  "
        f"diff_point_std={point_summary['diff_point_std']:.4f}  "
        f"diff_point_abs_mean={point_summary['diff_point_abs_mean']:.4f}  "
        f"diff_point_clear@0.03={point_summary['diff_point_clear_ratio_003']:.4f}  "
        f"diff_point_clear@0.05={point_summary['diff_point_clear_ratio_005']:.4f}"
    )
    print("[Region Target]")
    print(
        f"region_valid_ratio={region_summary['region_valid_ratio']:.4f}  "
        f"diff_region_mean={region_summary['diff_region_mean']:.4f}  "
        f"diff_region_std={region_summary['diff_region_std']:.4f}  "
        f"diff_region_abs_mean={region_summary['diff_region_abs_mean']:.4f}  "
        f"diff_region_clear@0.03={region_summary['diff_region_clear_ratio_003']:.4f}  "
        f"g_target_mean={region_summary['g_target_mean']:.4f}  "
        f"g_target_std={region_summary['g_target_std']:.4f}  "
        f"g_target_lseg>0.6={region_summary['g_target_lseg_ratio_06']:.4f}  "
        f"g_target_odise<0.4={region_summary['g_target_odise_ratio_04']:.4f}  "
        f"g_target_mid_0.45_0.55={region_summary['g_target_mid_ratio_045_055']:.4f}"
    )
    print("[Oracle Readout]")
    print(
        f"odise_only={oracle_summary['odise_only']['miou']:.4f}  "
        f"lseg_only={oracle_summary['lseg_only']['miou']:.4f}  "
        f"fixed_0.5={oracle_summary['fixed_05']['miou']:.4f}  "
        f"projected_gate={oracle_summary['projected_gate']['miou']:.4f}  "
        f"point_derived_region_oracle={oracle_summary['point_derived_region_oracle']['miou']:.4f}"
    )
    print("[Learnability Probe]")
    for feat_name, metrics in learnability_summary.items():
        print(
            f"feature={feat_name}  "
            f"train_mse={metrics['train_mse']:.6f}  "
            f"val_mse={metrics['val_mse']:.6f}  "
            f"corr={metrics['corr']:.4f}  "
            f"r2={metrics['r2']:.4f}"
        )
    print("[Speed]")
    print(
        f"time_per_batch={speed_summary['time_per_batch']:.4f}s  "
        f"time_target_build={speed_summary['time_build_point_target']:.4f}s  "
        f"time_aggregate_region={speed_summary['time_aggregate_region']:.4f}s  "
        f"time_gate_probe_train={speed_summary['time_gate_probe_train']:.4f}s  "
        f"estimated_epoch_time={speed_summary['estimated_epoch_time']:.1f}s"
    )


if __name__ == "__main__":
    main()
