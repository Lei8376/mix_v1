"""
Training script for Open Vocabulary 3D Fusion Model V2.

This script supports:
- Precomputed 2D features for fast training
- Mixed precision training (AMP)
- Validation with metrics
- Checkpoint resumption
- Configurable via YAML or command line

Usage:
    # With precomputed features (recommended for training)
    python train_open_vocab_v2.py \
        --precomputed-dir /path/to/precomputed_2d \
        --num-epochs 50

    # Without precomputed features (online extraction, slower)
    python train_open_vocab_v2.py \
        --label-path lang_seg/label_files/ade20k_objectInfo150.txt \
        --lseg-ckpt-path lang_seg/checkpoints/demo_e200.ckpt \
        --odise-model-config-path Panoptic/odise_caption_coco_50e.py
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from dataset.open_vocab_dataset_v2 import (
    OpenVocabDatasetV2Config,
    OpenVocabScannetDatasetV2,
    open_vocab_collate_v2,
)
from model.open_vocab_fusion_v2 import (
    OpenVocabFusionModelV2Config,
    OpenVocab3DFusionModelV2,
)
from trainer.open_vocab_trainer_v2 import (
    OpenVocabTrainerV2,
    OpenVocabTrainerV2Config,
)


@dataclass
class DataLoaderConfig:
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_odise_config_path(config_path: str, repo_root: str) -> Optional[str]:
    """Resolve ODISE config path."""
    if not config_path:
        return None
    configs_root = os.path.join(repo_root, "ODISE", "configs")
    if os.path.isabs(config_path) and os.path.exists(config_path):
        return os.path.relpath(config_path, configs_root)
    candidate_path = os.path.join(configs_root, config_path)
    if os.path.exists(candidate_path):
        return config_path
    return None


def create_data_loaders(
    dataset_config: OpenVocabDatasetV2Config,
    dataloader_config: DataLoaderConfig,
    val_split: str = "val",
) -> tuple:
    """Create train and validation data loaders."""
    train_dataset = OpenVocabScannetDatasetV2(dataset_config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataloader_config.batch_size,
        shuffle=True,
        num_workers=dataloader_config.num_workers,
        pin_memory=dataloader_config.pin_memory,
        drop_last=dataloader_config.drop_last,
        collate_fn=open_vocab_collate_v2,
    )

    # Create validation loader if validation split exists
    val_loader = None
    try:
        val_config = OpenVocabDatasetV2Config(
            data_config_path=dataset_config.data_config_path,
            precomputed_dir=dataset_config.precomputed_dir,
            split=val_split,
            scannet200=dataset_config.scannet200,
            voxel_size=dataset_config.voxel_size,
            aug=False,  # No augmentation for validation
            memcache_init=dataset_config.memcache_init,
            identifier=dataset_config.identifier + 1,  # Different identifier
            loop=1,
            eval_all=True,
            input_color=dataset_config.input_color,
        )
        val_dataset = OpenVocabScannetDatasetV2(val_config)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=False,
            num_workers=dataloader_config.num_workers,
            pin_memory=dataloader_config.pin_memory,
            drop_last=False,
            collate_fn=open_vocab_collate_v2,
        )
        print(f"Created validation loader with {len(val_dataset)} samples")
    except Exception as e:
        print(f"Could not create validation loader: {e}")

    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Open-Vocabulary 3D Fusion Model V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML config file (overrides defaults)",
    )

    # Dataset
    parser.add_argument(
        "--data-config-path",
        type=str,
        default="/home/featurize/work/XMask3D/config/scannet/xmask3d_scannet_B10N9.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--precomputed-dir",
        type=str,
        default="",
        help="Path to precomputed 2D features (recommended for training)",
    )
    parser.add_argument("--scannet200", action="store_true", help="Use ScanNet200")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size")
    parser.add_argument("--aug", action="store_true", help="Enable data augmentation")

    # Model (only needed for online extraction)
    parser.add_argument(
        "--label-path",
        type=str,
        default="",
        help="LSeg label file (for online extraction)",
    )
    parser.add_argument(
        "--lseg-ckpt-path",
        type=str,
        default="",
        help="LSeg checkpoint (for online extraction)",
    )
    parser.add_argument(
        "--odise-model-config-path",
        type=str,
        default="",
        help="ODISE config name (for online extraction)",
    )

    # Training
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--base-lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clip norm")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau"],
        help="LR scheduler type",
    )
    parser.add_argument("--bce-weight", type=float, default=1.0, help="BCE loss weight")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight")

    # Data loading
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")

    # Checkpointing
    parser.add_argument("--log-dir", type=str, default="runs/open_vocab_3d_v2", help="Log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--save-every-epochs", type=int, default=5, help="Save checkpoint interval")

    # AMP and early stopping
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")

    # Misc
    parser.add_argument("--seed", type=int, default=1342, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Load YAML config and override with command line args
    yaml_config = load_yaml_config(args.config)

    # Set seed
    seed = yaml_config.get("seed", args.seed)
    set_seed(seed)

    # Device
    device = yaml_config.get("device", args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    # Resolve paths
    repo_root = os.path.abspath(os.path.dirname(__file__))

    # Dataset config
    precomputed_dir = yaml_config.get("dataset", {}).get(
        "precomputed_dir", args.precomputed_dir
    )
    if precomputed_dir and not os.path.isabs(precomputed_dir):
        precomputed_dir = os.path.join(repo_root, precomputed_dir)
    if precomputed_dir and not os.path.exists(precomputed_dir):
        print(f"Warning: precomputed_dir not found: {precomputed_dir}")
        precomputed_dir = None

    dataset_config = OpenVocabDatasetV2Config(
        data_config_path=yaml_config.get("dataset", {}).get(
            "data_config_path", args.data_config_path
        ),
        precomputed_dir=precomputed_dir,
        split=yaml_config.get("dataset", {}).get("split", "train"),
        scannet200=yaml_config.get("dataset", {}).get("scannet200", args.scannet200),
        voxel_size=yaml_config.get("dataset", {}).get("voxel_size", args.voxel_size),
        aug=yaml_config.get("dataset", {}).get("aug", args.aug),
    )

    dataloader_config = DataLoaderConfig(
        batch_size=yaml_config.get("dataloader", {}).get("batch_size", args.batch_size),
        num_workers=yaml_config.get("dataloader", {}).get("num_workers", args.num_workers),
    )

    # Model config
    label_path = yaml_config.get("model", {}).get("label_path", args.label_path)
    lseg_ckpt_path = yaml_config.get("model", {}).get("lseg_ckpt_path", args.lseg_ckpt_path)
    odise_config_path = yaml_config.get("model", {}).get(
        "odise_model_config_path", args.odise_model_config_path
    )

    # Resolve relative paths
    if label_path and not os.path.isabs(label_path):
        label_path = os.path.join(repo_root, label_path)
    if lseg_ckpt_path and not os.path.isabs(lseg_ckpt_path):
        lseg_ckpt_path = os.path.join(repo_root, lseg_ckpt_path)
    if odise_config_path:
        odise_config_path = resolve_odise_config_path(odise_config_path, repo_root)

    model_config = OpenVocabFusionModelV2Config(
        device=device,
        label_path=label_path if label_path and os.path.exists(label_path) else None,
        lseg_ckpt_path=lseg_ckpt_path if lseg_ckpt_path and os.path.exists(lseg_ckpt_path) else None,
        odise_model_config_path=odise_config_path,
    )

    # Trainer config
    resume_checkpoint = yaml_config.get("trainer", {}).get("resume", args.resume)
    if resume_checkpoint and not os.path.isabs(resume_checkpoint):
        resume_checkpoint = os.path.join(repo_root, resume_checkpoint)
    if resume_checkpoint and not os.path.exists(resume_checkpoint):
        resume_checkpoint = None

    trainer_config = OpenVocabTrainerV2Config(
        num_epochs=yaml_config.get("trainer", {}).get("num_epochs", args.num_epochs),
        base_lr=yaml_config.get("trainer", {}).get("base_lr", args.base_lr),
        weight_decay=yaml_config.get("trainer", {}).get("weight_decay", args.weight_decay),
        grad_clip_norm=yaml_config.get("trainer", {}).get("grad_clip_norm", args.grad_clip_norm),
        warmup_epochs=yaml_config.get("trainer", {}).get("warmup_epochs", args.warmup_epochs),
        scheduler_type=yaml_config.get("trainer", {}).get("scheduler_type", args.scheduler_type),
        bce_weight=yaml_config.get("trainer", {}).get("bce_weight", args.bce_weight),
        dice_weight=yaml_config.get("trainer", {}).get("dice_weight", args.dice_weight),
        log_dir=yaml_config.get("trainer", {}).get("log_dir", args.log_dir),
        checkpoint_dir=yaml_config.get("trainer", {}).get("checkpoint_dir", args.checkpoint_dir),
        save_every_epochs=yaml_config.get("trainer", {}).get("save_every_epochs", args.save_every_epochs),
        use_amp=not args.no_amp,
        early_stopping_patience=yaml_config.get("trainer", {}).get(
            "early_stopping_patience", args.early_stopping_patience
        ),
        resume_checkpoint=resume_checkpoint,
    )

    # Validate configuration
    if not os.path.exists(dataset_config.data_config_path):
        raise FileNotFoundError(
            f"data_config_path not found: {dataset_config.data_config_path}"
        )

    # Check if we can train (need either precomputed features or online extraction paths)
    use_precomputed = (
        dataset_config.precomputed_dir is not None
        and os.path.exists(dataset_config.precomputed_dir)
    )
    can_extract_online = (
        model_config.label_path is not None
        and model_config.lseg_ckpt_path is not None
        and model_config.odise_model_config_path is not None
    )

    if not use_precomputed and not can_extract_online:
        raise ValueError(
            "Either provide --precomputed-dir with precomputed features, "
            "or provide --label-path, --lseg-ckpt-path, and --odise-model-config-path "
            "for online feature extraction."
        )

    print("=" * 60)
    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Precomputed features: {use_precomputed}")
    print(f"  Online extraction: {can_extract_online}")
    print(f"  Batch size: {dataloader_config.batch_size}")
    print(f"  Epochs: {trainer_config.num_epochs}")
    print(f"  Learning rate: {trainer_config.base_lr}")
    print(f"  AMP enabled: {trainer_config.use_amp}")
    print("=" * 60)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset_config, dataloader_config, val_split="val"
    )
    print(f"Train loader: {len(train_loader)} batches")

    # Create model
    model = OpenVocab3DFusionModelV2(model_config)

    # Create trainer and run
    trainer = OpenVocabTrainerV2(
        model=model,
        train_loader=train_loader,
        config=trainer_config,
        device=device,
        val_loader=val_loader,
    )

    results = trainer.train()
    print(f"Training finished: {results}")


if __name__ == "__main__":
    main()

