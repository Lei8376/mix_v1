"""Eval-only Dual-Space Semantic Probability Fusion.

ODISE and LSeg stay in their native text spaces:

  ODISE raw256 -> ODISE text256 probabilities
  LSeg raw512  -> CLIP text512 probabilities

The 3D student still uses current fused embeddings for point-mask geometry
logits. Semantic fusion happens only at the per-mask class-probability level.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from MinkowskiEngine import SparseTensor

REPO_ROOT = Path(__file__).resolve().parents[1]
MASK2FORMER_ROOT = REPO_ROOT / "ODISE" / "third_party" / "Mask2Former"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if MASK2FORMER_ROOT.exists() and str(MASK2FORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(MASK2FORMER_ROOT))

from dataset.open_vocab_dataset_v2 import (  # noqa: E402
    OpenVocabDatasetV2Config,
    OpenVocabScannetDatasetV2,
    open_vocab_collate_v2,
)
from evaluate.semantic_iou import (  # noqa: E402
    IGNORE_LABEL,
    SCANNET_LABELS_20,
    _SemanticAccumulator,
    build_text_features,
    diff2scene_class_probs_predict,
    diff2scene_mask_feature_predict,
    mask_feature_class_probs,
)
from model.open_vocab_fusion_v2 import (  # noqa: E402
    OpenVocab3DFusionModelV2,
    OpenVocabFusionModelV2Config,
)
from model.source_reliability_gate import build_source_gate_evidence, build_text_free_source_gate_evidence  # noqa: E402


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_repo_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def _load_model_state(model: torch.nn.Module, checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return checkpoint, missing, unexpected


def _build_loader(config: dict, split: str, max_samples: int | None, batch_size: int, num_workers: int, device: torch.device):
    dataset_cfg = config.get("dataset") or {}
    dataloader_cfg = config.get("dataloader") or {}
    val_config = OpenVocabDatasetV2Config(
        data_config_path=_resolve_repo_path(dataset_cfg.get("data_config_path", "config/data_scannet_3d.yaml")),
        precomputed_dir=_resolve_repo_path(dataset_cfg.get("precomputed_dir")),
        projection_dir=_resolve_repo_path(dataset_cfg.get("projection_dir")),
        split=split,
        scannet200=False,
        voxel_size=dataset_cfg.get("voxel_size", 0.05),
        aug=False,
        loop=1,
        eval_all=True,
        max_samples=max_samples,
        max_samples_ratio=None,
    )
    dataset = OpenVocabScannetDatasetV2(val_config)
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = dataloader_cfg.get("val_persistent_workers", False)
        loader_kwargs["prefetch_factor"] = dataloader_cfg.get(
            "val_prefetch_factor",
            dataloader_cfg.get("prefetch_factor", 2),
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type != "cpu"),
        drop_last=False,
        collate_fn=open_vocab_collate_v2,
        **loader_kwargs,
    )
    return dataset, loader


def _build_model(config: dict, checkpoint_path: str, device: torch.device, use_source_gate: bool = False):
    model_cfg = config.get("model") or {}
    alpha_max = model_cfg.get("alpha_max", 2.0)
    enable_source_gate = use_source_gate or bool(model_cfg.get("use_source_reliability_gate", False))
    model = OpenVocab3DFusionModelV2(
        OpenVocabFusionModelV2Config(
            device=str(device),
            pc_arch=model_cfg.get("pc_arch", "MinkUNet34C"),
            pixel_embedding_dim=model_cfg.get("pixel_embedding_dim", 512),
            mask_embedding_dim=model_cfg.get("mask_embedding_dim", 256),
            fused_embedding_dim=model_cfg.get("fused_embedding_dim", 256),
            pc_last_dim=model_cfg.get("pc_last_dim", 256),
            alpha_mode=model_cfg.get("alpha_mode", "learnable"),
            alpha_init=float(model_cfg.get("alpha_init", 1.0)),
            alpha_max=None if alpha_max is None else float(alpha_max),
            use_semantic_query=False,
            semantic_proj_path=None,
            freeze_semantic_proj=True,
            use_source_reliability_gate=enable_source_gate,
            source_gate_input_dim=int(model_cfg.get("source_gate_input_dim", 6)),
            source_gate_hidden_dim=int(model_cfg.get("source_gate_hidden_dim", 64)),
            source_gate_dropout=float(model_cfg.get("source_gate_dropout", 0.1)),
            source_gate_init_bias=float(model_cfg.get("source_gate_init_bias", -0.85)),
        )
    ).to(device)
    checkpoint, missing, unexpected = _load_model_state(model, checkpoint_path, str(device))
    model.eval()
    return model, checkpoint, missing, unexpected


def _class_probs_tau(mask_features: torch.Tensor, text_features: torch.Tensor, tau: float) -> torch.Tensor:
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return mask_feature_class_probs(mask_features, text_features, logit_scale=1.0 / float(tau))


def _dual_space_confidence_probs(
    p_odise: torch.Tensor,
    p_lseg: torch.Tensor,
    conf_min: float = 0.2,
    conf_max: float = 0.7,
) -> torch.Tensor:
    if p_odise.shape != p_lseg.shape:
        raise RuntimeError(f"class-prob shape mismatch: ODISE={tuple(p_odise.shape)} LSeg={tuple(p_lseg.shape)}")
    c = p_odise.shape[-1]
    log_c = math.log(float(c))
    ent_odise = -(p_odise * p_odise.clamp_min(1e-12).log()).sum(dim=-1) / log_c
    ent_lseg = -(p_lseg * p_lseg.clamp_min(1e-12).log()).sum(dim=-1) / log_c
    conf_odise = 1.0 - ent_odise
    conf_lseg = 1.0 - ent_lseg
    w_lseg = conf_lseg / (conf_lseg + conf_odise + 1e-6)
    w_lseg = w_lseg.clamp(conf_min, conf_max).unsqueeze(-1)
    return (1.0 - w_lseg) * p_odise + w_lseg * p_lseg


def _metric(acc: _SemanticAccumulator, prefix: str) -> Dict:
    out = acc.compute(prefix)
    return {
        "miou": out[prefix],
        "macc": out[prefix.replace("miou", "macc")],
        "n_valid_classes": out[f"n_valid_classes_{prefix}"],
        "n_valid_classes_acc": out[f"n_valid_classes_acc_{prefix}"],
        "per_class_iou": out[f"per_class_iou_{prefix}"],
        "per_class_acc": out[f"per_class_acc_{prefix}"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/train_scannet_v2_full_multi_gpu.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tau-odise", type=float, default=0.07)
    parser.add_argument("--tau-lseg", type=float, default=0.07)
    parser.add_argument("--odise-weight", type=float, default=0.5)
    parser.add_argument("--lseg-weight", type=float, default=0.5)
    parser.add_argument("--conf-min", type=float, default=0.2)
    parser.add_argument("--conf-max", type=float, default=0.7)
    parser.add_argument("--use-source-gate", action="store_true")
    parser.add_argument("--save-gate-stats", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--clip-cache-dir", default=str(REPO_ROOT / "checkpoints" / "pretrained" / "clip"))
    args = parser.parse_args()

    os.environ.setdefault("CLIP_CACHE_DIR", args.clip_cache_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[dual-space] CUDA not available; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    weight_sum = float(args.odise_weight) + float(args.lseg_weight)
    if weight_sum <= 0:
        raise ValueError("--odise-weight + --lseg-weight must be positive")
    w_odise = float(args.odise_weight) / weight_sum
    w_lseg = float(args.lseg_weight) / weight_sum

    config = _load_yaml(Path(_resolve_repo_path(args.config)))
    trainer_cfg = config.get("trainer") or {}
    dataset, loader = _build_loader(config, args.split, args.max_samples, args.batch_size, args.num_workers, device)
    model, checkpoint, missing, unexpected = _build_model(
        config,
        str(Path(_resolve_repo_path(args.checkpoint))),
        device,
        use_source_gate=args.use_source_gate,
    )
    model_ref = model.module if hasattr(model, "module") else model
    source_gate = getattr(model_ref, "source_gate", None)
    if args.use_source_gate and source_gate is None:
        raise RuntimeError("--use-source-gate requires model.source_gate in the checkpoint/config.")

    prompt = trainer_cfg.get("semantic_prompt_template", "a photo of a {}")
    odise_text256 = build_text_features(
        SCANNET_LABELS_20,
        prompt_template=prompt,
        clip_model=trainer_cfg.get("semantic_clip_model", "ODISE-256"),
        device=device,
    )
    clip_text512 = build_text_features(
        SCANNET_LABELS_20,
        prompt_template=prompt,
        clip_model=trainer_cfg.get("semantic_pixel_clip_model", "ViT-B/32"),
        device=device,
    )

    accs = {
        "odise_only_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        "lseg_only_text512": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        "current_fused_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        "dual_space_fixed": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        "dual_space_confidence": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
    }
    if args.use_source_gate:
        accs["dual_space_gate"] = _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL)

    total_points = 0
    total_valid_masks = 0
    gate_values = []
    with torch.no_grad():
        for batch in loader:
            if "pixel_pooled" not in batch:
                raise RuntimeError("Dual-space eval requires batch['pixel_pooled'] raw LSeg 512D features.")
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch["sinput"] = SparseTensor(batch["feat_3d"], batch["coords_3d"].int())
            results = model(batch)
            mask_valid = results["mask_valid_from_masks"]

            for b in range(len(results["outputs"])):
                if not results["outputs"][b]:
                    continue
                valid_k = mask_valid[b]
                if not valid_k.any():
                    continue
                pt_mask = results["batch_indices"] == b
                gt_b = batch["binary_label_3d"][pt_mask].detach().cpu().long()
                pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()

                odise_q = batch["mask_embeddings"][b][valid_k].float()
                lseg_q = results.get("pixel_pooled_embeddings", batch["pixel_pooled"])[b][valid_k].float()
                fused_q = results["fused_embeddings"][b][valid_k].float()

                if odise_q.shape[-1] != odise_text256.shape[-1]:
                    raise RuntimeError(f"ODISE dim mismatch: mask={odise_q.shape[-1]} text={odise_text256.shape[-1]}")
                if lseg_q.shape[-1] != clip_text512.shape[-1]:
                    raise RuntimeError(f"LSeg dim mismatch: mask={lseg_q.shape[-1]} text={clip_text512.shape[-1]}")
                if fused_q.shape[-1] != odise_text256.shape[-1]:
                    raise RuntimeError(f"fused dim mismatch: mask={fused_q.shape[-1]} text={odise_text256.shape[-1]}")

                p_odise = _class_probs_tau(odise_q, odise_text256, args.tau_odise)
                p_lseg = _class_probs_tau(lseg_q, clip_text512, args.tau_lseg)
                p_fixed = w_odise * p_odise + w_lseg * p_lseg
                p_conf = _dual_space_confidence_probs(
                    p_odise,
                    p_lseg,
                    args.conf_min,
                    args.conf_max,
                )

                preds = {
                    "odise_only_text256": diff2scene_mask_feature_predict(pred_logits, odise_q, odise_text256),
                    "lseg_only_text512": diff2scene_mask_feature_predict(pred_logits, lseg_q, clip_text512),
                    "current_fused_text256": diff2scene_mask_feature_predict(pred_logits, fused_q, odise_text256),
                    "dual_space_fixed": diff2scene_class_probs_predict(pred_logits, p_fixed),
                    "dual_space_confidence": diff2scene_class_probs_predict(pred_logits, p_conf),
                }
                if args.use_source_gate:
                    point_mask_conf = torch.sigmoid(pred_logits).mean(dim=0).detach()
                    input_dim = int(getattr(getattr(model_ref, "config", None), "source_gate_input_dim", 6))
                    if input_dim == 6:
                        k = int(pred_logits.shape[1])
                        mv_default = torch.full((k,), 0.5, device=pred_logits.device, dtype=pred_logits.dtype)
                        mv_valid = torch.zeros(k, device=pred_logits.device, dtype=pred_logits.dtype)
                        mask_area = torch.zeros(k, device=pred_logits.device, dtype=pred_logits.dtype)
                        lifted_count = torch.zeros(k, device=pred_logits.device, dtype=pred_logits.dtype)
                        evidence = build_text_free_source_gate_evidence(
                            mv_default,
                            mv_default,
                            mv_valid,
                            mask_area,
                            lifted_count,
                            point_mask_conf,
                        )
                        gate = source_gate(evidence)
                        p_gate = (1.0 - gate[:, None]) * p_odise + gate[:, None] * p_lseg
                    else:
                        evidence = build_source_gate_evidence(
                            p_odise,
                            p_lseg,
                            point_mask_conf=point_mask_conf,
                            input_dim=input_dim,
                        )
                        gate = source_gate(evidence)
                        p_gate = (1.0 - gate) * p_odise + gate * p_lseg
                    preds["dual_space_gate"] = diff2scene_class_probs_predict(
                        pred_logits,
                        p_gate,
                    )
                    gate_values.append(gate.detach().reshape(-1).cpu())
                for name, pred in preds.items():
                    accs[name].update_labels(pred, gt_b)
                total_points += int(gt_b.numel())
                total_valid_masks += int(valid_k.sum().item())

    metrics = {name: _metric(acc, f"semantic_miou_{name}") for name, acc in accs.items()}
    setup = {
        "repo": str(REPO_ROOT),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
        "config": args.config,
        "split": args.split,
        "samples": len(dataset),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_samples": args.max_samples,
        "tau_odise": args.tau_odise,
        "tau_lseg": args.tau_lseg,
        "odise_weight": w_odise,
        "lseg_weight": w_lseg,
        "fusion_type": "dual_space_probability",
        "prompt_template": prompt,
        "odise_text": trainer_cfg.get("semantic_clip_model", "ODISE-256"),
        "clip_text512": trainer_cfg.get("semantic_pixel_clip_model", "ViT-B/32"),
        "use_source_gate": args.use_source_gate,
        "save_gate_stats": args.save_gate_stats,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "total_points": total_points,
        "total_valid_masks": total_valid_masks,
    }
    output = {"setup": setup, "metrics": metrics}
    if args.use_source_gate and gate_values:
        gate_cat = torch.cat(gate_values)
        gate_stats = {
            "mean": float(gate_cat.mean().item()),
            "std": float(gate_cat.std(unbiased=False).item()),
            "min": float(gate_cat.min().item()),
            "max": float(gate_cat.max().item()),
        }
        output["source_gate_stats"] = gate_stats

    print("[DUAL-SPACE SEMANTIC FUSION]")
    print(f"[dual-space] checkpoint={args.checkpoint}")
    print(f"[dual-space] checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(
        f"[dual-space] samples={len(dataset)} split={args.split} "
        f"points={total_points} valid_masks={total_valid_masks}"
    )
    print(f"[dual-space] weights: odise={w_odise:.3f} lseg={w_lseg:.3f} tau_odise={args.tau_odise} tau_lseg={args.tau_lseg}")
    print(f"ODISE only @ ODISE text256:     mIoU={metrics['odise_only_text256']['miou']:.6f}")
    print(f"LSeg only @ CLIP text512:       mIoU={metrics['lseg_only_text512']['miou']:.6f}")
    print(f"Current fused @ ODISE text256:  mIoU={metrics['current_fused_text256']['miou']:.6f}")
    print(f"Dual fixed {w_odise:.1f}/{w_lseg:.1f}:             mIoU={metrics['dual_space_fixed']['miou']:.6f}")
    print(f"Dual confidence fusion:         mIoU={metrics['dual_space_confidence']['miou']:.6f}")
    if args.use_source_gate:
        print(f"Dual gate:                      mIoU={metrics['dual_space_gate']['miou']:.6f}")
        if gate_values:
            stats = output["source_gate_stats"]
            print(
                "[SourceGate] "
                f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
                f"min={stats['min']:.6f} max={stats['max']:.6f}"
            )

    if args.output_json:
        out_path = Path(_resolve_repo_path(args.output_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[dual-space] wrote output_json={out_path}")


if __name__ == "__main__":
    main()
