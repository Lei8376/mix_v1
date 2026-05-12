"""Evaluate a mix2_v1 mask-distill checkpoint on ScanNet20 val split."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--clip-cache-dir", default="/tmp/clip")
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
        max_samples=args.max_samples,
        max_samples_ratio=None,
    )
    val_dataset = OpenVocabScannetDatasetV2(val_config)
    batch_size = args.batch_size or dataloader_cfg.get("batch_size", 2)
    num_workers = args.num_workers if args.num_workers is not None else dataloader_cfg.get("num_workers", 4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        drop_last=False,
        collate_fn=open_vocab_collate_v2,
    )

    alpha_max = model_cfg.get("alpha_max", 2.0)
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

    trainer_config = MaskDistillTrainerConfig(
        log_dir=trainer_cfg.get("log_dir", "runs/eval_only"),
        checkpoint_dir=trainer_cfg.get("checkpoint_dir", "checkpoints/eval_only"),
        use_amp=(not args.no_amp and device != "cpu"),
        mask_distill_weight=trainer_cfg.get("mask_distill_weight", 1.0),
        bce_weight=trainer_cfg.get("bce_weight", 0.0),
        dice_weight=trainer_cfg.get("dice_weight", 0.0),
        min_points_per_mask=trainer_cfg.get("min_points_per_mask", 10),
        semantic_clip_model=trainer_cfg.get("semantic_clip_model", "ODISE-256"),
        semantic_pixel_clip_model=trainer_cfg.get("semantic_pixel_clip_model", "ViT-B/32"),
        semantic_prompt_template=trainer_cfg.get("semantic_prompt_template", "a photo of a {}"),
        semantic_pc_lambda=trainer_cfg.get("semantic_pc_lambda", 0.5),
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
    print("[eval] metrics:")
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
    ):
        if key in metrics:
            print(f"  {key}:")
            for cls, val in metrics[key].items():
                print(f"    {cls}: {val}")


if __name__ == "__main__":
    main()
