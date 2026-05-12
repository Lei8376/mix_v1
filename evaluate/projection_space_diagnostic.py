"""Diagnose semantic quality before/after mix2_v1 projection layers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from MinkowskiEngine import SparseTensor
import numpy as np

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
)
from experiment_mask_distill.criterion_mask_distill import build_lifted_3d_masks  # noqa: E402
from model.open_vocab_fusion_v2 import (  # noqa: E402
    OpenVocab3DFusionModelV2,
    OpenVocabFusionModelV2Config,
)


ODISE_TO_SCANNET = {
    "wall": "wall",
    "floor": "floor",
    "cabinet": "cabinet",
    "bed": "bed",
    "chair": "chair",
    "sofa": "sofa",
    "couch": "sofa",
    "table": "table",
    "dining table": "table",
    "door": "door",
    "window": "window",
    "bookshelf": "bookshelf",
    "bookcase": "bookshelf",
    "picture": "picture",
    "painting": "picture",
    "counter": "counter",
    "desk": "desk",
    "curtain": "curtain",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "shower curtain": "shower curtain",
    "toilet": "toilet",
    "sink": "sink",
    "bathtub": "bathtub",
}


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


def _class_probs_from_infos(infos, device: torch.device) -> torch.Tensor:
    label_to_idx = {name: i for i, name in enumerate(SCANNET_LABELS_20)}
    probs = torch.zeros(len(infos), len(SCANNET_LABELS_20), device=device)
    matched = 0
    for i, info in enumerate(infos):
        name = ""
        if isinstance(info, dict):
            name = str(info.get("category_name", ""))
        else:
            try:
                item = info.item()
                if isinstance(item, dict):
                    name = str(item.get("category_name", ""))
            except Exception:
                name = str(info)
        mapped = ODISE_TO_SCANNET.get(name.strip().lower())
        if mapped in label_to_idx:
            probs[i, label_to_idx[mapped]] = 1.0
            matched += 1
    return probs, matched


def _metric(acc: _SemanticAccumulator, name: str) -> dict:
    out = acc.compute(name)
    return {
        "miou": out[name],
        "per_class": out[f"per_class_iou_{name}"],
        "n_valid_classes": out[f"n_valid_classes_{name}"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/train_scannet_v2_full_multi_gpu.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--clip-cache-dir", default="/tmp/clip")
    args = parser.parse_args()

    os.environ.setdefault("CLIP_CACHE_DIR", args.clip_cache_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[diag] CUDA not available; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    config = _load_yaml(Path(_resolve_repo_path(args.config)))
    dataset_cfg = config.get("dataset") or {}
    model_cfg = config.get("model") or {}
    trainer_cfg = config.get("trainer") or {}

    val_config = OpenVocabDatasetV2Config(
        data_config_path=_resolve_repo_path(dataset_cfg.get("data_config_path", "config/data_scannet_3d.yaml")),
        precomputed_dir=_resolve_repo_path(dataset_cfg.get("precomputed_dir")),
        projection_dir=_resolve_repo_path(dataset_cfg.get("projection_dir")),
        split=args.split,
        scannet200=False,
        voxel_size=dataset_cfg.get("voxel_size", 0.05),
        aug=False,
        loop=1,
        eval_all=True,
        max_samples=args.max_samples,
        max_samples_ratio=None,
    )
    dataset = OpenVocabScannetDatasetV2(val_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=open_vocab_collate_v2,
    )

    alpha_max = model_cfg.get("alpha_max", 2.0)
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
        )
    ).to(device)
    checkpoint, missing, unexpected = _load_model_state(model, args.checkpoint, str(device))
    model.eval()

    text_hybrid = build_text_features(
        SCANNET_LABELS_20,
        trainer_cfg.get("semantic_prompt_template", "a photo of a {}"),
        trainer_cfg.get("semantic_clip_model", "ODISE-256"),
        device=device,
    )
    text_b = build_text_features(
        SCANNET_LABELS_20,
        trainer_cfg.get("semantic_prompt_template", "a photo of a {}"),
        trainer_cfg.get("semantic_pixel_clip_model", "ViT-B/32"),
        device=device,
    )

    accs = {
        "odise_raw_label": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        "odise_proj_256_text": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        "lseg_raw_512_text": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        "lseg_proj_256_text": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        "fused_256_text": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
    }
    matched_masks = 0
    total_masks = 0

    with torch.no_grad():
        sample_offset = 0
        for batch in loader:
            batch_size_current = batch["pixel_pooled"].shape[0]
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch["sinput"] = SparseTensor(batch["feat_3d"], batch["coords_3d"].int())
            results = model(batch)

            mask_valid = results["mask_valid_from_masks"]
            mask_proj_all = model.fuse_embed.mask_proj(batch["mask_embeddings"].float())
            pixel_proj_all = model.fuse_embed.pixel_proj(batch["pixel_pooled"].float())

            for b in range(len(results["outputs"])):
                if not results["outputs"][b]:
                    continue
                valid_k = mask_valid[b]
                if not valid_k.any():
                    continue
                pt_mask = results["batch_indices"] == b
                gt_b = batch["binary_label_3d"][pt_mask].detach().cpu().long()
                pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()

                scene_name, frame_stem = dataset.samples[sample_offset + b]
                npz_path = Path(dataset.precomputed_dir) / scene_name / f"{frame_stem}_odise.npz"
                with np.load(npz_path, allow_pickle=True) as npz:
                    infos = npz["info"]
                raw_probs, n_matched = _class_probs_from_infos(infos, device)
                if raw_probs.shape[0] < valid_k.shape[0]:
                    pad = raw_probs.new_zeros(valid_k.shape[0] - raw_probs.shape[0], raw_probs.shape[1])
                    raw_probs = torch.cat([raw_probs, pad], dim=0)
                raw_probs = raw_probs[valid_k]
                matched_masks += n_matched
                total_masks += int(valid_k.sum().item())

                preds = {
                    "odise_raw_label": diff2scene_class_probs_predict(pred_logits, raw_probs),
                    "odise_proj_256_text": diff2scene_mask_feature_predict(
                        pred_logits, mask_proj_all[b][valid_k], text_hybrid
                    ),
                    "lseg_raw_512_text": diff2scene_mask_feature_predict(
                        pred_logits, batch["pixel_pooled"][b][valid_k].float(), text_b
                    ),
                    "lseg_proj_256_text": diff2scene_mask_feature_predict(
                        pred_logits, pixel_proj_all[b][valid_k], text_hybrid
                    ),
                    "fused_256_text": diff2scene_mask_feature_predict(
                        pred_logits, results["fused_embeddings"][b][valid_k], text_hybrid
                    ),
                }
                for name, pred in preds.items():
                    accs[name].update_labels(pred, gt_b)
            sample_offset += batch_size_current

    print(f"[diag] checkpoint={args.checkpoint}")
    print(f"[diag] checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"[diag] samples={len(dataset)} split={args.split}")
    print(f"[diag] missing_keys={missing}")
    print(f"[diag] unexpected_keys={unexpected}")
    print(f"[diag] odise_raw_label_matched_masks={matched_masks}/{total_masks}")
    for name, acc in accs.items():
        res = _metric(acc, name)
        print(f"[diag] {name}: miou={res['miou']:.6f} n_valid={res['n_valid_classes']}")
        for cls, val in res["per_class"].items():
            print(f"  {cls}: {val:.6f}")


if __name__ == "__main__":
    main()
