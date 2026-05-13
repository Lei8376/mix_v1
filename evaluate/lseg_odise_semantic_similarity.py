"""Compare raw LSeg and raw ODISE 2D semantic distributions.

This diagnostic does not use the learned LSeg->ODISE pixel_proj. It compares
the two teachers in their own text spaces on the same mask slots:

  LSeg raw512  @ CLIP-B text512  -> ScanNet20 mask class probabilities
  ODISE raw256 @ ODISE text256   -> ScanNet20 mask class probabilities

Then it uses the current checkpoint's 3D point-mask logits to project each
teacher's mask class probabilities to 3D semantic predictions and reports mIoU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
    mask_feature_class_probs,
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


def _build_loader(config: dict, split: str, max_samples: int | None, batch_size: int, num_workers: int):
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
    checkpoint, missing, unexpected = _load_model_state(model, checkpoint_path, str(device))
    model.eval()
    return model, checkpoint, missing, unexpected


def _acc_result(acc: _SemanticAccumulator, prefix: str) -> dict:
    res = acc.compute(prefix)
    return {
        "miou": res[prefix],
        "macc": res[prefix.replace("miou", "macc")],
        "n_valid_classes": res[f"n_valid_classes_{prefix}"],
        "per_class_iou": res[f"per_class_iou_{prefix}"],
        "per_class_acc": res[f"per_class_acc_{prefix}"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/train_scannet_v2_full_multi_gpu.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--clip-cache-dir", default=str(REPO_ROOT / "checkpoints" / "pretrained" / "clip"))
    args = parser.parse_args()

    os.environ.setdefault("CLIP_CACHE_DIR", args.clip_cache_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[lseg-odise] CUDA not available; using CPU")
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
    )
    model, checkpoint, missing, unexpected = _build_model(
        config,
        str(Path(_resolve_repo_path(args.checkpoint))),
        device,
    )

    prompt = trainer_cfg.get("semantic_prompt_template", "a photo of a {}")
    lseg_text = build_text_features(
        class_names=SCANNET_LABELS_20,
        prompt_template=prompt,
        clip_model=trainer_cfg.get("semantic_pixel_clip_model", "ViT-B/32"),
        device=device,
    )
    odise_text = build_text_features(
        class_names=SCANNET_LABELS_20,
        prompt_template=prompt,
        clip_model=trainer_cfg.get("semantic_clip_model", "ODISE-256"),
        device=device,
    )

    lseg_acc = _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL)
    odise_acc = _SemanticAccumulator(SCANNET_LABELS_20, IGNORE_LABEL)
    agreement_count = 0
    total_masks = 0
    prob_cos_sum = 0.0
    js_sum = 0.0

    eps = 1e-12
    with torch.no_grad():
        for batch in loader:
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

                lseg_features = batch["pixel_pooled"][b][valid_k].float()
                odise_features = batch["mask_embeddings"][b][valid_k].float()
                lseg_probs = mask_feature_class_probs(lseg_features, lseg_text)
                odise_probs = mask_feature_class_probs(odise_features, odise_text)

                prob_cos = F.cosine_similarity(lseg_probs, odise_probs, dim=-1)
                prob_cos_sum += float(prob_cos.sum().item())
                agreement_count += int((lseg_probs.argmax(dim=-1) == odise_probs.argmax(dim=-1)).sum().item())
                total_masks += int(valid_k.sum().item())

                m = 0.5 * (lseg_probs + odise_probs)
                js = 0.5 * (
                    (lseg_probs * ((lseg_probs + eps).log() - (m + eps).log())).sum(dim=-1)
                    + (odise_probs * ((odise_probs + eps).log() - (m + eps).log())).sum(dim=-1)
                )
                js_sum += float(js.sum().item())

                pt_mask = results["batch_indices"] == b
                gt_b = batch["binary_label_3d"][pt_mask].detach().cpu().long()
                pred_logits = results["outputs"][b][0]["pred_mask_logits"][:, valid_k].float()
                lseg_pred = diff2scene_class_probs_predict(pred_logits, lseg_probs)
                odise_pred = diff2scene_class_probs_predict(pred_logits, odise_probs)
                lseg_acc.update_labels(lseg_pred, gt_b)
                odise_acc.update_labels(odise_pred, gt_b)

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
        "split": args.split,
        "samples": len(dataset),
        "total_masks": total_masks,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mask_prob_cosine_mean": prob_cos_sum / max(total_masks, 1),
        "mask_top1_agreement": agreement_count / max(total_masks, 1),
        "mask_js_divergence_mean": js_sum / max(total_masks, 1),
        "lseg_raw512_clip_text_3d": _acc_result(lseg_acc, "semantic_miou_lseg_raw512_clip_text_3d"),
        "odise_raw256_odise_text_3d": _acc_result(odise_acc, "semantic_miou_odise_raw256_odise_text_3d"),
    }

    print(f"[lseg-odise] checkpoint={args.checkpoint}")
    print(f"[lseg-odise] checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"[lseg-odise] samples={len(dataset)} split={args.split} total_masks={total_masks}")
    print(f"[lseg-odise] mask_prob_cosine_mean={summary['mask_prob_cosine_mean']:.6f}")
    print(f"[lseg-odise] mask_top1_agreement={summary['mask_top1_agreement']:.6f}")
    print(f"[lseg-odise] mask_js_divergence_mean={summary['mask_js_divergence_mean']:.6f}")
    print(
        "[lseg-odise] lseg_raw512_clip_text_3d "
        f"miou={summary['lseg_raw512_clip_text_3d']['miou']:.6f} "
        f"macc={summary['lseg_raw512_clip_text_3d']['macc']:.6f} "
        f"n_valid={summary['lseg_raw512_clip_text_3d']['n_valid_classes']}"
    )
    print(
        "[lseg-odise] odise_raw256_odise_text_3d "
        f"miou={summary['odise_raw256_odise_text_3d']['miou']:.6f} "
        f"macc={summary['odise_raw256_odise_text_3d']['macc']:.6f} "
        f"n_valid={summary['odise_raw256_odise_text_3d']['n_valid_classes']}"
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[lseg-odise] wrote {out_path}")


if __name__ == "__main__":
    main()
