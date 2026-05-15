"""Evaluate the trained shared LSeg pixel_proj in ODISE-256 text space.

This eval-only diagnostic intentionally tests only the linear layer used by the
geometry fusion branch:

    LSeg raw512 -> model.fuse_embed.pixel_proj -> ODISE text256

It does not use ridge probes, pixel_sem_proj, semantic_embeddings, or any
ODISE/LSeg mixing strategy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
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
    diff2scene_mask_feature_predict,
)
from model.open_vocab_fusion_v2 import (  # noqa: E402
    OpenVocab3DFusionModelV2,
    OpenVocabFusionModelV2Config,
)


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


def _build_loader(
    config: dict,
    split: str,
    max_samples: int | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    dataset_cfg = config.get("dataset") or {}
    dataloader_cfg = config.get("dataloader") or {}
    val_config = OpenVocabDatasetV2Config(
        data_config_path=_resolve_repo_path(
            dataset_cfg.get("data_config_path", "config/data_scannet_3d.yaml")
        ),
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


def _build_model(config: dict, checkpoint_path: str, device: torch.device):
    model_cfg = config.get("model") or {}
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
            # Keep this diagnostic isolated from the decoupled semantic head.
            use_semantic_query=False,
            freeze_semantic_proj=True,
        )
    ).to(device)
    checkpoint, missing, unexpected = _load_model_state(model, checkpoint_path, str(device))
    model.eval()
    return model, checkpoint, missing, unexpected


def _compute_metric(acc: _SemanticAccumulator, prefix: str) -> Dict:
    result = acc.compute(prefix)
    return {
        "miou": result[prefix],
        "macc": result[prefix.replace("miou", "macc")],
        "n_valid_classes": result[f"n_valid_classes_{prefix}"],
        "n_valid_classes_acc": result[f"n_valid_classes_acc_{prefix}"],
        "per_class_iou": result[f"per_class_iou_{prefix}"],
        "per_class_acc": result[f"per_class_acc_{prefix}"],
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
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--clip-cache-dir",
        default=str(REPO_ROOT / "checkpoints" / "pretrained" / "clip"),
    )
    args = parser.parse_args()

    os.environ.setdefault("CLIP_CACHE_DIR", args.clip_cache_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[trained-pixel-proj] CUDA not available; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    config = _load_yaml(Path(_resolve_repo_path(args.config)))
    trainer_cfg = config.get("trainer") or {}
    dataset, loader = _build_loader(
        config=config,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    model, checkpoint, missing, unexpected = _build_model(
        config=config,
        checkpoint_path=str(Path(_resolve_repo_path(args.checkpoint))),
        device=device,
    )

    prompt = trainer_cfg.get("semantic_prompt_template", "a photo of a {}")
    text256 = build_text_features(
        class_names=SCANNET_LABELS_20,
        prompt_template=prompt,
        clip_model=trainer_cfg.get("semantic_clip_model", "ODISE-256"),
        device=device,
    )
    text512 = build_text_features(
        class_names=SCANNET_LABELS_20,
        prompt_template=prompt,
        clip_model=trainer_cfg.get("semantic_pixel_clip_model", "ViT-B/32"),
        device=device,
    )

    accs = {
        "semantic_miou_odise_raw256_odise_text256": _SemanticAccumulator(
            SCANNET_LABELS_20, IGNORE_LABEL
        ),
        "semantic_miou_lseg_raw512_clip_text512": _SemanticAccumulator(
            SCANNET_LABELS_20, IGNORE_LABEL
        ),
        "semantic_miou_lseg_model_pixel_proj256": _SemanticAccumulator(
            SCANNET_LABELS_20, IGNORE_LABEL
        ),
        "semantic_miou_current_fused_odise_text256": _SemanticAccumulator(
            SCANNET_LABELS_20, IGNORE_LABEL
        ),
    }

    total_points = 0
    total_valid_masks = 0
    with torch.no_grad():
        for batch in loader:
            if "pixel_pooled" not in batch:
                raise RuntimeError("This diagnostic requires batch['pixel_pooled'] raw LSeg 512D features.")
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
                lseg_raw = batch["pixel_pooled"][b][valid_k].float()
                lseg_model_256 = model.fuse_embed.pixel_proj(lseg_raw)
                fused_q = results["fused_embeddings"][b][valid_k].float()

                preds = {
                    "semantic_miou_odise_raw256_odise_text256": diff2scene_mask_feature_predict(
                        point_mask_logits=pred_logits,
                        mask_features=odise_q,
                        text_features=text256,
                    ),
                    "semantic_miou_lseg_raw512_clip_text512": diff2scene_mask_feature_predict(
                        point_mask_logits=pred_logits,
                        mask_features=lseg_raw,
                        text_features=text512,
                    ),
                    "semantic_miou_lseg_model_pixel_proj256": diff2scene_mask_feature_predict(
                        point_mask_logits=pred_logits,
                        mask_features=lseg_model_256,
                        text_features=text256,
                    ),
                    "semantic_miou_current_fused_odise_text256": diff2scene_mask_feature_predict(
                        point_mask_logits=pred_logits,
                        mask_features=fused_q,
                        text_features=text256,
                    ),
                }
                for name, pred in preds.items():
                    accs[name].update_labels(pred, gt_b)
                total_points += int(gt_b.numel())
                total_valid_masks += int(valid_k.sum().item())

    metrics = {name: _compute_metric(acc, name) for name, acc in accs.items()}
    output = {
        "setup": {
            "repo": str(REPO_ROOT),
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
            "config": args.config,
            "split": args.split,
            "samples": len(dataset),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_samples": args.max_samples,
            "prompt_template": prompt,
            "odise_text": trainer_cfg.get("semantic_clip_model", "ODISE-256"),
            "clip_text512": trainer_cfg.get("semantic_pixel_clip_model", "ViT-B/32"),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "total_points": total_points,
            "total_valid_masks": total_valid_masks,
        },
        "metrics": metrics,
        "summary": [
            {
                "method": name,
                "miou": metric["miou"],
                "macc": metric["macc"],
                "n_valid_classes": metric["n_valid_classes"],
            }
            for name, metric in metrics.items()
        ],
    }

    print("[TRAINED LINEAR TEST]")
    print("LSeg raw512 -> model.fuse_embed.pixel_proj -> ODISE text256")
    print(f"[trained-pixel-proj] checkpoint={args.checkpoint}")
    print(f"[trained-pixel-proj] checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(
        f"[trained-pixel-proj] samples={len(dataset)} split={args.split} "
        f"points={total_points} valid_masks={total_valid_masks}"
    )
    print(f"[trained-pixel-proj] missing_keys={missing}")
    print(f"[trained-pixel-proj] unexpected_keys={unexpected}")
    labels = {
        "semantic_miou_odise_raw256_odise_text256": "ODISE raw256 @ ODISE text256",
        "semantic_miou_lseg_raw512_clip_text512": "LSeg raw512 @ CLIP text512",
        "semantic_miou_lseg_model_pixel_proj256": (
            "LSeg raw512 -> model.fuse_embed.pixel_proj -> ODISE text256"
        ),
        "semantic_miou_current_fused_odise_text256": "current fused_embeddings @ ODISE text256",
    }
    for name, label in labels.items():
        metric = metrics[name]
        print(
            f"[trained-pixel-proj] {label}: "
            f"miou={metric['miou']:.6f} "
            f"macc={metric['macc']:.6f} "
            f"n_valid={metric['n_valid_classes']}"
        )

    if args.output_json:
        out_path = Path(_resolve_repo_path(args.output_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[trained-pixel-proj] wrote output_json={out_path}")


if __name__ == "__main__":
    main()
