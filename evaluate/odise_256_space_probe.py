"""Probe 512D features into ODISE's 256D text-readable space.

This is a diagnostic only. It fits linear ridge probes on the first N records:

  - LSeg raw512 -> ODISE raw256
  - current fused512 -> ODISE raw256

Then it evaluates Diff2Scene-style mIoU with ODISE's own 256D text features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from MinkowskiEngine import SparseTensor

REPO_ROOT = Path(__file__).resolve().parents[1]
ODISE_ROOT = REPO_ROOT / "ODISE"
MASK2FORMER_ROOT = REPO_ROOT / "ODISE" / "third_party" / "Mask2Former"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if ODISE_ROOT.exists() and str(ODISE_ROOT) not in sys.path:
    sys.path.insert(0, str(ODISE_ROOT))
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


def _fit_ridge_probe(
    source: torch.Tensor,
    target: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Fit normalized source -> normalized target with a bias column."""
    x = F.normalize(source.float(), dim=-1)
    y = F.normalize(target.float(), dim=-1)
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    x_aug = torch.cat([x, ones], dim=1)
    reg = ridge * torch.eye(x_aug.shape[1], dtype=x.dtype, device=x.device)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def _apply_probe(source: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    x = F.normalize(source.float(), dim=-1)
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    return F.normalize(torch.cat([x, ones], dim=1) @ weights, dim=-1)


def _metric(acc: _SemanticAccumulator, name: str) -> dict:
    out = acc.compute(name)
    return {
        "miou": out[name],
        "per_class": out[f"per_class_iou_{name}"],
        "target": out.get(f"target_{name}", {}),
        "n_valid_classes": out[f"n_valid_classes_{name}"],
    }


def _build_odise_text_features(
    class_names: Sequence[str],
    prompt_template: str,
    odise_model_config: str,
    device: torch.device,
) -> torch.Tensor:
    """Use ODISE checkpoint word_head.text_proj to create normalized 256D text features."""
    import open_clip

    config_name = odise_model_config.replace(".py", "").replace(".yaml", "")
    checkpoint_names = {
        "Panoptic/odise_caption_coco_50e": "odise_caption_coco_50e-853cc971.pth",
        "Panoptic/odise_label_coco_50e": "odise_label_coco_50e-b67d2efc.pth",
    }
    if config_name not in checkpoint_names:
        raise RuntimeError(f"Unsupported ODISE model config for text head lookup: {odise_model_config}")
    checkpoint_path = (
        Path.home()
        / ".torch"
        / "iopath_cache"
        / "NVlabs"
        / "ODISE"
        / "releases"
        / "download"
        / "v1.0.0"
        / checkpoint_names[config_name]
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ODISE checkpoint not found in local cache: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    weight = state_dict["word_head.text_proj.weight"].float().to(device)
    bias = state_dict["word_head.text_proj.bias"].float().to(device)

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device=device,
    )
    model.eval()
    prompts = [prompt_template.format("other" if label == "otherfurniture" else label) for label in class_names]
    with torch.no_grad():
        tokens = open_clip.tokenize(prompts).to(device)
        text = model.encode_text(tokens).float()
        text = text @ weight.t() + bias
        text = F.normalize(text, dim=-1)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return text


def _make_loader(config: dict, split: str, max_samples: int, batch_size: int, num_workers: int):
    dataset_cfg = config.get("dataset") or {}
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
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=open_vocab_collate_v2,
    )
    return dataset, loader


def _build_model(config: dict, checkpoint_path: str, device: torch.device):
    model_cfg = config.get("model") or {}
    model = OpenVocab3DFusionModelV2(
        OpenVocabFusionModelV2Config(
            device=str(device),
            pc_arch=model_cfg.get("pc_arch", "MinkUNet34C"),
            pixel_embedding_dim=model_cfg.get("pixel_embedding_dim", 512),
            mask_embedding_dim=model_cfg.get("mask_embedding_dim", 256),
            fused_embedding_dim=model_cfg.get("fused_embedding_dim", 256),
            pc_last_dim=model_cfg.get("pc_last_dim", 256),
        )
    ).to(device)
    checkpoint, missing, unexpected = _load_model_state(model, checkpoint_path, str(device))
    model.eval()
    return model, checkpoint, missing, unexpected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/train_scannet_v2_full_multi_gpu.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--probe-train-records", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--odise-model-config", default="Panoptic/odise_caption_coco_50e.py")
    parser.add_argument("--odise-prompt-template", default="a photo of a {}")
    parser.add_argument("--output-json", default="record/odise_256_space_probe_2026-05-07.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[probe] CUDA not available; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    config = _load_yaml(Path(_resolve_repo_path(args.config)))
    dataset, loader = _make_loader(config, args.split, args.max_samples, args.batch_size, args.num_workers)
    model, checkpoint, missing, unexpected = _build_model(
        config,
        str(Path(_resolve_repo_path(args.checkpoint))),
        device,
    )
    text256 = _build_odise_text_features(
        SCANNET_LABELS_20,
        args.odise_prompt_template,
        args.odise_model_config,
        device,
    )

    train_lseg = []
    train_fused = []
    train_odise = []
    records = []

    with torch.no_grad():
        sample_offset = 0
        for batch in loader:
            batch_size_current = batch["pixel_pooled"].shape[0]
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch["sinput"] = SparseTensor(batch["feat_3d"], batch["coords_3d"].int())
            results = model(batch)
            mask_valid = results["mask_valid_from_masks"]

            for b in range(len(results["outputs"])):
                record_idx = sample_offset + b
                if not results["outputs"][b]:
                    continue
                valid_k = mask_valid[b]
                if not valid_k.any():
                    continue

                odise = batch["mask_embeddings"][b][valid_k].float().detach().cpu()
                lseg = batch["pixel_pooled"][b][valid_k].float().detach().cpu()
                fused = results["fused_embeddings"][b][valid_k].float().detach().cpu()
                records.append(
                    {
                        "record_idx": record_idx,
                        "lseg": lseg,
                        "fused": fused,
                        "odise": odise,
                        "batch_index": b,
                    }
                )
                if record_idx < args.probe_train_records:
                    train_lseg.append(lseg)
                    train_fused.append(fused)
                    train_odise.append(odise)
            sample_offset += batch_size_current

    if not train_lseg:
        raise RuntimeError("No training masks collected for probe fitting.")

    lseg_to_odise = _fit_ridge_probe(
        torch.cat(train_lseg, dim=0).to(device),
        torch.cat(train_odise, dim=0).to(device),
        args.ridge,
    )
    fused_to_odise = _fit_ridge_probe(
        torch.cat(train_fused, dim=0).to(device),
        torch.cat(train_odise, dim=0).to(device),
        args.ridge,
    )

    split_accs = {
        "all20": {
            "odise_raw256_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
            "lseg512_to_odise256_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
            "fused512_to_odise256_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        },
        "train10_probe_fit": {
            "odise_raw256_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
            "lseg512_to_odise256_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
            "fused512_to_odise256_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        },
        "test10_probe_eval": {
            "odise_raw256_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
            "lseg512_to_odise256_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
            "fused512_to_odise256_text256": _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL),
        },
    }

    with torch.no_grad():
        sample_offset = 0
        for batch in loader:
            batch_size_current = batch["pixel_pooled"].shape[0]
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch["sinput"] = SparseTensor(batch["feat_3d"], batch["coords_3d"].int())
            results = model(batch)
            mask_valid = results["mask_valid_from_masks"]

            for b in range(len(results["outputs"])):
                record_idx = sample_offset + b
                if not results["outputs"][b]:
                    continue
                valid_k = mask_valid[b]
                if not valid_k.any():
                    continue

                pt_mask = results["batch_indices"] == b
                gt_b = batch["binary_label_3d"][pt_mask].detach().cpu().long()
                pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()
                odise = batch["mask_embeddings"][b][valid_k].float()
                lseg256 = _apply_probe(batch["pixel_pooled"][b][valid_k].float(), lseg_to_odise)
                fused256 = _apply_probe(results["fused_embeddings"][b][valid_k].float(), fused_to_odise)

                preds = {
                    "odise_raw256_text256": diff2scene_mask_feature_predict(pred_logits, odise, text256),
                    "lseg512_to_odise256_text256": diff2scene_mask_feature_predict(pred_logits, lseg256, text256),
                    "fused512_to_odise256_text256": diff2scene_mask_feature_predict(pred_logits, fused256, text256),
                }

                split_names = ["all20"]
                split_names.append("train10_probe_fit" if record_idx < args.probe_train_records else "test10_probe_eval")
                for split_name in split_names:
                    for method, pred in preds.items():
                        split_accs[split_name][method].update_labels(pred, gt_b)
            sample_offset += batch_size_current

    output = {
        "setup": {
            "repo": str(REPO_ROOT),
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
            "config": args.config,
            "split": args.split,
            "max_samples": args.max_samples,
            "records_used": len(dataset),
            "probe_train_records": args.probe_train_records,
            "probe_test_records": max(0, len(dataset) - args.probe_train_records),
            "probe_fit": "ridge least squares on normalized source512 -> normalized ODISE raw256, bias included",
            "ridge": args.ridge,
            "odise_text256": f"{args.odise_model_config} word_head.text_proj, prompt: {args.odise_prompt_template}",
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        },
        "summary": [],
        "metrics": {},
    }
    for split_name, accs in split_accs.items():
        output["metrics"][split_name] = {}
        for method, acc in accs.items():
            metric = _metric(acc, method)
            output["metrics"][split_name][method] = metric
            output["summary"].append(
                {
                    "split": split_name,
                    "method": method,
                    "miou": metric["miou"],
                    "n_valid_classes": metric["n_valid_classes"],
                }
            )

    out_path = Path(_resolve_repo_path(args.output_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[probe] checkpoint={args.checkpoint}")
    print(f"[probe] checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"[probe] output_json={out_path}")
    for item in output["summary"]:
        print(
            f"[probe] {item['split']} {item['method']}: "
            f"miou={item['miou']:.6f} n_valid={item['n_valid_classes']}"
        )


if __name__ == "__main__":
    main()
