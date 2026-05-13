"""Evaluate a mix2_v1 mask-distill checkpoint on ScanNet20 val split."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
import yaml

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
from experiment_mask_distill.trainer_mask_distill import (  # noqa: E402
    MaskDistillTrainer,
    MaskDistillTrainerConfig,
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
    model_is_ddp = next(iter(model.state_dict())).startswith("module.")
    ckpt_is_ddp = next(iter(state_dict)).startswith("module.")
    if model_is_ddp and not ckpt_is_ddp:
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif not model_is_ddp and ckpt_is_ddp:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model_ref = model.module if hasattr(model, "module") else model
    semantic_proj_path = getattr(getattr(model_ref, "config", None), "semantic_proj_path", None)
    if semantic_proj_path:
        model_ref._load_semantic_projection(semantic_proj_path)
    return checkpoint, missing, unexpected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/train_scannet_v2_full_multi_gpu.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument(
        "--clip-cache-dir",
        default=str(REPO_ROOT / "checkpoints" / "pretrained" / "clip"),
    )
    args = parser.parse_args()

    os.environ.setdefault("CLIP_CACHE_DIR", args.clip_cache_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA not available; using CPU")
        args.device = "cpu"
    device = args.device

    config = _load_yaml(Path(_resolve_repo_path(args.config)))
    dataset_cfg = config.get("dataset") or {}
    model_cfg = config.get("model") or {}
    trainer_cfg = config.get("trainer") or {}
    dataloader_cfg = config.get("dataloader") or {}

    data_config_path = _resolve_repo_path(
        dataset_cfg.get("data_config_path", "config/data_scannet_3d.yaml")
    )
    precomputed_dir = _resolve_repo_path(dataset_cfg.get("precomputed_dir"))
    projection_dir = _resolve_repo_path(dataset_cfg.get("projection_dir"))

    val_config = OpenVocabDatasetV2Config(
        data_config_path=data_config_path,
        precomputed_dir=precomputed_dir,
        projection_dir=projection_dir,
        split=args.split,
        scannet200=False,
        voxel_size=dataset_cfg.get("voxel_size", 0.05),
        aug=False,
        loop=1,
        eval_all=True,
        max_samples=args.max_samples if args.max_samples is not None else dataloader_cfg.get("val_max_samples"),
        max_samples_ratio=None if args.max_samples is not None else dataloader_cfg.get("val_max_samples_ratio"),
    )
    val_dataset = OpenVocabScannetDatasetV2(val_config)
    batch_size = args.batch_size or dataloader_cfg.get("val_batch_size", dataloader_cfg.get("batch_size", 2))
    num_workers = args.num_workers if args.num_workers is not None else dataloader_cfg.get("num_workers", 4)
    if args.num_workers is None:
        num_workers = dataloader_cfg.get("val_num_workers", num_workers)
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = dataloader_cfg.get("val_persistent_workers", False)
        loader_kwargs["prefetch_factor"] = dataloader_cfg.get(
            "val_prefetch_factor",
            dataloader_cfg.get("prefetch_factor", 2),
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        drop_last=False,
        collate_fn=open_vocab_collate_v2,
        **loader_kwargs,
    )

    alpha_max = model_cfg.get("alpha_max", 2.0)
    semantic_proj_path = _resolve_repo_path(model_cfg.get("semantic_proj_path"))
    model_config = OpenVocabFusionModelV2Config(
        device=device,
        pc_arch=model_cfg.get("pc_arch", "MinkUNet34C"),
        pixel_embedding_dim=model_cfg.get("pixel_embedding_dim", 512),
        mask_embedding_dim=model_cfg.get("mask_embedding_dim", 256),
        fused_embedding_dim=model_cfg.get("fused_embedding_dim", 256),
        pc_last_dim=model_cfg.get("pc_last_dim", 256),
        alpha_mode=model_cfg.get("alpha_mode", "learnable"),
        alpha_init=float(model_cfg.get("alpha_init", 1.0)),
        alpha_max=None if alpha_max is None else float(alpha_max),
        use_semantic_query=bool(model_cfg.get("use_semantic_query", True)),
        semantic_fusion_mode=model_cfg.get("semantic_fusion_mode", "fixed"),
        semantic_odise_weight=float(model_cfg.get("semantic_odise_weight", 0.5)),
        semantic_lseg_weight=float(model_cfg.get("semantic_lseg_weight", 0.5)),
        semantic_init_odise_weight=float(model_cfg.get("semantic_init_odise_weight", 0.5)),
        semantic_init_lseg_weight=float(model_cfg.get("semantic_init_lseg_weight", 0.5)),
        semantic_proj_path=semantic_proj_path,
        freeze_semantic_proj=bool(model_cfg.get("freeze_semantic_proj", True)),
    )
    model = OpenVocab3DFusionModelV2(model_config).to(device)
    checkpoint, missing, unexpected = _load_model_state(model, args.checkpoint, device)
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"[eval] val_samples={len(val_dataset)} batch_size={batch_size} split={args.split}")
    if missing:
        print(f"[eval] missing_keys={missing}")
    if unexpected:
        print(f"[eval] unexpected_keys={unexpected}")

    trainer_config_kwargs = {
        "log_dir": trainer_cfg.get("log_dir", "runs/eval_only"),
        "checkpoint_dir": trainer_cfg.get("checkpoint_dir", "checkpoints/eval_only"),
        "use_amp": (not args.no_amp and device != "cpu"),
        "mask_distill_weight": trainer_cfg.get("mask_distill_weight", 1.0),
        "bce_weight": trainer_cfg.get("bce_weight", 0.0),
        "dice_weight": trainer_cfg.get("dice_weight", 0.0),
        "min_points_per_mask": trainer_cfg.get("min_points_per_mask", 10),
        "semantic_clip_model": trainer_cfg.get("semantic_clip_model", "ODISE-256"),
        "semantic_pixel_clip_model": trainer_cfg.get("semantic_pixel_clip_model", "ViT-B/32"),
        "semantic_prompt_template": trainer_cfg.get("semantic_prompt_template", "a photo of a {}"),
        "semantic_pc_lambda": trainer_cfg.get("semantic_pc_lambda", 0.5),
        "validation_log_every_batches": trainer_cfg.get("validation_log_every_batches", 25),
    }
    trainer_config_fields = {field.name for field in dataclasses.fields(MaskDistillTrainerConfig)}
    trainer_config = MaskDistillTrainerConfig(
        **{
            key: value
            for key, value in trainer_config_kwargs.items()
            if key in trainer_config_fields
        }
    )
    print("[eval] runtime config:")
    print(f"  device: {device}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_workers: {num_workers}")
    print(f"  persistent_workers: {loader_kwargs.get('persistent_workers', False)}")
    print(f"  prefetch_factor: {loader_kwargs.get('prefetch_factor', None)}")
    print(f"  val_samples: {len(val_dataset)}")
    print(f"  checkpoint_dir: {trainer_config.checkpoint_dir}")
    print(f"  log_dir: {trainer_config.log_dir}")
    print(
        "  validation_log_every_batches: "
        f"{getattr(trainer_config, 'validation_log_every_batches', 'unsupported')}"
    )
    trainer = MaskDistillTrainer(
        model=model,
        train_loader=val_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device,
        rank=0,
    )
    metrics = trainer._validate(checkpoint.get("epoch", 0))
    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[eval] wrote metrics_json={metrics_path}")
    print("[eval] metrics:")
    if "semantic_miou_hybrid_odise256" in metrics:
        print(
            "  ODISE-256 probes: "
            f"hybrid={metrics['semantic_miou_hybrid_odise256']} "
            f"clip_proj={metrics['semantic_miou_clip_odise256']} "
            f"odise={metrics['semantic_miou_odise_odise256']} "
            f"base={metrics['semantic_miou_base_odise256']} "
            f"refine={metrics['semantic_miou_refine_odise256']} "
            f"lseg_semproj={metrics.get('semantic_miou_lseg_semproj_odise256')} "
            f"semantic_query={metrics.get('semantic_miou_semantic_query_odise256')}"
        )
    for key, value in metrics.items():
        if key.startswith("per_class") or key == "target":
            continue
        print(f"  {key}: {value}")
    for key in (
        "per_class_iou_hybrid_text",
        "per_class_iou_clip_text",
        "per_class_iou_final",
        "per_class_acc_hybrid_text",
        "per_class_acc_clip_text",
        "per_class_acc_final",
        "per_class_iou_hybrid_odise256",
        "per_class_iou_clip_odise256",
        "per_class_iou_odise_odise256",
        "per_class_iou_base_odise256",
        "per_class_iou_refine_odise256",
        "per_class_iou_lseg_semproj_odise256",
        "per_class_iou_semantic_query_odise256",
        "per_class_acc_hybrid_odise256",
        "per_class_acc_clip_odise256",
        "per_class_acc_odise_odise256",
        "per_class_acc_base_odise256",
        "per_class_acc_refine_odise256",
        "per_class_acc_lseg_semproj_odise256",
        "per_class_acc_semantic_query_odise256",
    ):
        if key in metrics:
            print(f"  {key}:")
            for cls, val in metrics[key].items():
                print(f"    {cls}: {val}")


if __name__ == "__main__":
    main()
