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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent
MASK2FORMER_ROOT = REPO_ROOT / "ODISE" / "third_party" / "Mask2Former"
if MASK2FORMER_ROOT.exists() and str(MASK2FORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(MASK2FORMER_ROOT))

import numpy as np
import torch
import yaml

from dataset.open_vocab_dataset_v2 import (
    OpenVocabDatasetV2Config,
    OpenVocabScannetDatasetV2,
    SceneGroupedBatchSampler,
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
try:
    from experiment_mask_distill.trainer_mask_distill import (
        MaskDistillTrainer,
        MaskDistillTrainerConfig,
    )
    _MASK_DISTILL_AVAILABLE = True
except ImportError:
    MaskDistillTrainer = None
    MaskDistillTrainerConfig = None
    _MASK_DISTILL_AVAILABLE = False


@dataclass
class DataLoaderConfig:
    batch_size: int = 2
    num_workers: int = 4
    val_batch_size: Optional[int] = None
    val_num_workers: Optional[int] = None
    pin_memory: bool = True
    drop_last: bool = True
    multiview_batch: bool = False
    scenes_per_batch: int = 1
    views_per_scene: int = 4
    seed: int = 0


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def expand_data_paths(data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Expand DATA paths from a data config YAML.

    If DATA.data_base_dir is set, relative DATA paths are resolved under it.
    This keeps machine-specific data locations in one file.
    """
    data_cfg = dict(data_cfg or {})
    base_dir = data_cfg.get("data_base_dir") or data_cfg.get("base_dir")
    if base_dir:
        base_dir = os.path.expanduser(os.path.expandvars(str(base_dir)))
        data_cfg["data_base_dir"] = base_dir
    for key in (
        "data_root",
        "data_root_2d",
        "data_root_lseg_feat",
        "data_root_odise_feat",
        "projection_dir",
        "data_root_projection",
    ):
        value = data_cfg.get(key)
        if not value:
            continue
        value = os.path.expanduser(os.path.expandvars(str(value)))
        if base_dir:
            value = value.replace("${data_base_dir}", base_dir)
            if not os.path.isabs(value):
                value = os.path.join(base_dir, value)
        data_cfg[key] = value
    return data_cfg


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
    train_loader_kwargs = dict(
        dataset=train_dataset,
        num_workers=dataloader_config.num_workers,
        pin_memory=dataloader_config.pin_memory,
        collate_fn=open_vocab_collate_v2,
    )
    if dataloader_config.multiview_batch:
        train_batch_sampler = SceneGroupedBatchSampler(
            train_dataset,
            scenes_per_batch=dataloader_config.scenes_per_batch,
            views_per_scene=dataloader_config.views_per_scene,
            drop_last=dataloader_config.drop_last,
            shuffle=True,
            seed=dataloader_config.seed,
        )
        train_loader = torch.utils.data.DataLoader(
            batch_sampler=train_batch_sampler,
            **train_loader_kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            batch_size=dataloader_config.batch_size,
            shuffle=True,
            drop_last=dataloader_config.drop_last,
            **train_loader_kwargs,
        )

    # Create validation loader if validation split exists
    val_loader = None
    try:
        val_config = OpenVocabDatasetV2Config(
            data_config_path=dataset_config.data_config_path,
            precomputed_dir=dataset_config.precomputed_dir,
            projection_dir=dataset_config.projection_dir,
            split=val_split,
            scannet200=dataset_config.scannet200,
            voxel_size=dataset_config.voxel_size,
            aug=False,  # No augmentation for validation
            memcache_init=dataset_config.memcache_init,
            identifier=dataset_config.identifier + 1,  # Different identifier
            loop=1,
            eval_all=True,
            input_color=dataset_config.input_color,
            max_samples=getattr(dataset_config, "val_max_samples", None),
            max_samples_ratio=getattr(dataset_config, "val_max_samples_ratio", None),
        )
        val_dataset = OpenVocabScannetDatasetV2(val_config)
        val_batch_size = dataloader_config.val_batch_size or dataloader_config.batch_size
        val_num_workers = (
            dataloader_config.num_workers
            if dataloader_config.val_num_workers is None
            else dataloader_config.val_num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=dataloader_config.pin_memory,
            drop_last=False,
            collate_fn=open_vocab_collate_v2,
        )
        print(
            f"Created validation loader with {len(val_dataset)} samples "
            f"(batch_size={val_batch_size}, num_workers={val_num_workers})"
        )
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
        default="config/data_scannet_3d.yaml",
        help="Path to dataset config YAML (DATA.data_root for 3D); relative to repo root if not abs",
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
    parser.add_argument("--val-every-epochs", type=int, default=1, help="Validation interval")  # 🔥 修复

    # AMP and early stopping
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--model-half", action="store_true", help="Store model in float16 to reduce VRAM (use with AMP)")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--eval-only", action="store_true", help="Run validation only after loading checkpoint")

    # Misc
    parser.add_argument("--seed", type=int, default=1342, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Load YAML config and override with command line args
    yaml_config = load_yaml_config(args.config) or {}

    # Set seed
    seed = yaml_config.get("seed", args.seed)
    set_seed(seed)

    # Device and GPU selection
    device = yaml_config.get("device", args.device)
    gpu_ids = yaml_config.get("gpu_ids", [0])
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    elif device == "cuda" and gpu_ids:
        # Set specific GPU
        torch.cuda.set_device(gpu_ids[0])
        device = f"cuda:{gpu_ids[0]}"
        print(f"Using GPU: {gpu_ids[0]}")

    # Resolve paths
    repo_root = os.path.abspath(os.path.dirname(__file__))

    # Dataset config（YAML 中 dataset/model 等可能为 null，用 or {} 避免 .get 得到 None）
    _dataset = yaml_config.get("dataset") or {}
    data_config_path = _dataset.get(
        "data_config_path", args.data_config_path
    )
    if data_config_path and not os.path.isabs(data_config_path):
        data_config_path = os.path.join(repo_root, data_config_path)
    if not data_config_path or not os.path.exists(data_config_path):
        data_config_path = os.path.join(repo_root, "config/data_scannet_3d.yaml")
    data_path_cfg = expand_data_paths((load_yaml_config(data_config_path) or {}).get("DATA", {}))

    precomputed_dir = _dataset.get("precomputed_dir", args.precomputed_dir)
    if not precomputed_dir:
        precomputed_dir = data_path_cfg.get("data_root_lseg_feat")
    if precomputed_dir and not os.path.isabs(precomputed_dir):
        precomputed_dir = os.path.join(repo_root, precomputed_dir)
    if precomputed_dir and not os.path.exists(precomputed_dir):
        print(f"Warning: precomputed_dir not found: {precomputed_dir}")
        precomputed_dir = None
    # 方案 B: 预计算投影目录
    projection_dir = _dataset.get("projection_dir", None)
    if not projection_dir:
        projection_dir = data_path_cfg.get("projection_dir") or data_path_cfg.get("data_root_projection")
    if projection_dir and not os.path.isabs(projection_dir):
        projection_dir = os.path.join(repo_root, projection_dir)

    dataset_config = OpenVocabDatasetV2Config(
        data_config_path=data_config_path,
        precomputed_dir=precomputed_dir,
        projection_dir=projection_dir,
        split=_dataset.get("split", "train"),
        scannet200=_dataset.get("scannet200", args.scannet200),
        voxel_size=_dataset.get("voxel_size", args.voxel_size),
        aug=_dataset.get("aug", args.aug),
        max_samples=_dataset.get("max_samples") or None,
        max_samples_ratio=_dataset.get("max_samples_ratio") or None,
    )
    setattr(dataset_config, "val_max_samples", _dataset.get("val_max_samples") or None)
    setattr(dataset_config, "val_max_samples_ratio", _dataset.get("val_max_samples_ratio") or None)

    _dataloader = yaml_config.get("dataloader") or {}
    raw_batch_size = _dataloader.get("batch_size", args.batch_size)
    # MinkowskiEngine BatchNorm requires >1 value per channel in training; batch_size 1 causes "Expected more than 1 value per channel".
    if raw_batch_size < 2:
        print(
            f"Warning: batch_size={raw_batch_size} is not supported (MinkowskiEngine BatchNorm). Using batch_size=2."
        )
        raw_batch_size = 2
    multiview_batch = bool(_dataloader.get("multiview_batch", False))
    scenes_per_batch = int(_dataloader.get("scenes_per_batch", 1))
    views_per_scene = int(_dataloader.get("views_per_scene", 4))
    if multiview_batch:
        expected_batch_size = scenes_per_batch * views_per_scene
        if raw_batch_size != expected_batch_size:
            print(
                f"Warning: batch_size={raw_batch_size} does not match "
                f"scenes_per_batch*views_per_scene={expected_batch_size}. "
                f"Using batch_size={expected_batch_size}."
            )
            raw_batch_size = expected_batch_size
    dataloader_config = DataLoaderConfig(
        batch_size=raw_batch_size,
        num_workers=_dataloader.get("num_workers", args.num_workers),
        val_batch_size=_dataloader.get("val_batch_size"),
        val_num_workers=_dataloader.get("val_num_workers"),
        drop_last=bool(_dataloader.get("drop_last", True)),
        multiview_batch=multiview_batch,
        scenes_per_batch=scenes_per_batch,
        views_per_scene=views_per_scene,
        seed=seed,
    )

    # Model config
    _model = yaml_config.get("model") or {}
    label_path = _model.get("label_path", args.label_path)
    lseg_ckpt_path = _model.get("lseg_ckpt_path", args.lseg_ckpt_path)
    odise_config_path = _model.get(
        "odise_model_config_path", args.odise_model_config_path
    )

    # Resolve relative paths
    if label_path and not os.path.isabs(label_path):
        label_path = os.path.join(repo_root, label_path)
    if lseg_ckpt_path and not os.path.isabs(lseg_ckpt_path):
        lseg_ckpt_path = os.path.join(repo_root, lseg_ckpt_path)
    if odise_config_path:
        odise_config_path = resolve_odise_config_path(odise_config_path, repo_root)
    alpha_max = _model.get("alpha_max", 2.0)
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
        alpha_mode=_model.get("alpha_mode", "learnable"),
        alpha_init=float(_model.get("alpha_init", 1.0)),
        alpha_max=None if alpha_max is None else float(alpha_max),
        use_semantic_query=bool(_model.get("use_semantic_query", False)),
        semantic_fusion_mode=_model.get("semantic_fusion_mode", "fixed"),
        semantic_odise_weight=float(_model.get("semantic_odise_weight", 0.5)),
        semantic_lseg_weight=float(_model.get("semantic_lseg_weight", 0.5)),
        semantic_init_odise_weight=float(_model.get("semantic_init_odise_weight", 0.5)),
        semantic_init_lseg_weight=float(_model.get("semantic_init_lseg_weight", 0.5)),
        semantic_proj_path=semantic_proj_path,
        freeze_semantic_proj=bool(_model.get("freeze_semantic_proj", True)),
        use_source_reliability_gate=bool(_model.get("use_source_reliability_gate", False)),
        source_gate_input_dim=int(_model.get("source_gate_input_dim", 6)),
        source_gate_hidden_dim=int(_model.get("source_gate_hidden_dim", 64)),
        source_gate_dropout=float(_model.get("source_gate_dropout", 0.1)),
        source_gate_init_bias=float(_model.get("source_gate_init_bias", -0.85)),
        dual_branch_probe=bool(_model.get("dual_branch_probe", False)),
        dual_branch_lseg_match_dim=int(_model.get("dual_branch_lseg_match_dim", 512)),
        dual_branch_odise_match_dim=int(_model.get("dual_branch_odise_match_dim", 256)),
        use_point_semantic_gate=bool(_model.get("use_point_semantic_gate", False)),
        point_sem_gate_hidden_dim=int(_model.get("point_sem_gate_hidden_dim", 128)),
        point_sem_gate_init_bias=float(_model.get("point_sem_gate_init_bias", 0.85)),
        use_region_reliability_gate=bool(_model.get("use_region_reliability_gate", False)),
        region_gate_hidden_dim=int(_model.get("region_gate_hidden_dim", 128)),
        region_gate_init_bias=float(_model.get("region_gate_init_bias", 0.85)),
        region_gate_signal_dim=int(_model.get("region_gate_signal_dim", 11)),
        alignment_query_mode=str(_model.get("alignment_query_mode", "fused")),
    )

    # Trainer config
    _trainer = yaml_config.get("trainer") or {}
    eval_only = bool(yaml_config.get("eval_only", False) or _trainer.get("eval_only", False) or args.eval_only)
    resume_checkpoint = _trainer.get("resume", args.resume)
    if resume_checkpoint and not os.path.isabs(resume_checkpoint):
        resume_checkpoint = os.path.join(repo_root, resume_checkpoint)
    if resume_checkpoint and not os.path.exists(resume_checkpoint):
        resume_checkpoint = None

    trainer_config = OpenVocabTrainerV2Config(
        num_epochs=_trainer.get("num_epochs", args.num_epochs),
        base_lr=_trainer.get("base_lr", args.base_lr),
        weight_decay=_trainer.get("weight_decay", args.weight_decay),
        grad_clip_norm=_trainer.get("grad_clip_norm", args.grad_clip_norm),
        warmup_epochs=_trainer.get("warmup_epochs", args.warmup_epochs),
        scheduler_type=_trainer.get("scheduler_type", args.scheduler_type),
        bce_weight=_trainer.get("bce_weight", args.bce_weight),
        dice_weight=_trainer.get("dice_weight", args.dice_weight),
        min_points_per_mask=_trainer.get("min_points_per_mask", 10),  # 🔥 从配置读取
        log_dir=_trainer.get("log_dir", args.log_dir),
        checkpoint_dir=_trainer.get("checkpoint_dir", args.checkpoint_dir),
        save_every_epochs=_trainer.get("save_every_epochs", args.save_every_epochs),
        val_every_epochs=_trainer.get("val_every_epochs", args.val_every_epochs),  # 🔥 修复：从配置读取
        use_amp=not args.no_amp,
        early_stopping_patience=_trainer.get(
            "early_stopping_patience", args.early_stopping_patience
        ),
        resume_checkpoint=resume_checkpoint,
        use_model_half=_trainer.get("use_model_half", args.model_half),
        gradient_accumulation_steps=_trainer.get("gradient_accumulation_steps", 1),  # 🔥 梯度累积
        semantic_clip_model=_trainer.get("semantic_clip_model", "ODISE-256"),
        semantic_pixel_clip_model=_trainer.get("semantic_pixel_clip_model", "ViT-B/32"),
        semantic_prompt_template=_trainer.get("semantic_prompt_template", "a photo of a {}"),
        semantic_pc_lambda=_trainer.get("semantic_pc_lambda", 0.5),
    )
    use_mask_distill = _trainer.get("use_mask_distill", False)
    if use_mask_distill:
        if not _MASK_DISTILL_AVAILABLE:
            raise ImportError("experiment_mask_distill not found; cannot use use_mask_distill")
        trainer_config = MaskDistillTrainerConfig(
            num_epochs=_trainer.get("num_epochs", args.num_epochs),
            base_lr=_trainer.get("base_lr", args.base_lr),
            weight_decay=_trainer.get("weight_decay", args.weight_decay),
            grad_clip_norm=_trainer.get("grad_clip_norm", args.grad_clip_norm),
            warmup_epochs=_trainer.get("warmup_epochs", args.warmup_epochs),
            scheduler_type=_trainer.get("scheduler_type", args.scheduler_type),
            mask_distill_weight=_trainer.get("mask_distill_weight", 1.0),
            bce_weight=_trainer.get("bce_weight", 0.0),
            dice_weight=_trainer.get("dice_weight", 0.0),
            min_points_per_mask=_trainer.get("min_points_per_mask", 10),
            log_dir=_trainer.get("log_dir", args.log_dir),
            checkpoint_dir=_trainer.get("checkpoint_dir", args.checkpoint_dir),
            save_every_epochs=_trainer.get("save_every_epochs", args.save_every_epochs),
            val_every_epochs=_trainer.get("val_every_epochs", args.val_every_epochs),
            use_amp=not args.no_amp,
            early_stopping_patience=_trainer.get(
                "early_stopping_patience", args.early_stopping_patience
            ),
            resume_checkpoint=resume_checkpoint,
            max_batches_per_epoch=_trainer.get("max_batches_per_epoch", None),
            use_model_half=_trainer.get("use_model_half", args.model_half),
            gradient_accumulation_steps=_trainer.get("gradient_accumulation_steps", 1),
            semantic_clip_model=_trainer.get("semantic_clip_model", "ODISE-256"),
            semantic_pixel_clip_model=_trainer.get("semantic_pixel_clip_model", "ViT-B/32"),
            semantic_prompt_template=_trainer.get("semantic_prompt_template", "a photo of a {}"),
            semantic_pc_lambda=_trainer.get("semantic_pc_lambda", 0.5),
            dual_space_eval=_trainer.get("dual_space_eval", True),
            dual_space_odise_weight=_trainer.get("dual_space_odise_weight", 0.5),
            dual_space_lseg_weight=_trainer.get("dual_space_lseg_weight", 0.5),
            dual_space_tau_odise=_trainer.get("dual_space_tau_odise", 0.07),
            dual_space_tau_lseg=_trainer.get("dual_space_tau_lseg", 0.07),
            dual_space_use_confidence=_trainer.get("dual_space_use_confidence", False),
            dual_space_conf_min=_trainer.get("dual_space_conf_min", 0.2),
            dual_space_conf_max=_trainer.get("dual_space_conf_max", 0.7),
            best_monitor=_trainer.get("monitor_metric", _trainer.get("best_monitor", "semantic_miou_learned_region_gate")),
            source_gate_train=_trainer.get("source_gate_train", False),
            source_gate_loss_weight=_trainer.get("source_gate_loss_weight", 0.03),
            source_gate_open_loss_weight=_trainer.get("source_gate_open_loss_weight", 0.03),
            source_gate_start_epoch=_trainer.get("source_gate_start_epoch", 3),
            source_gate_detach_teacher_probs=_trainer.get("source_gate_detach_teacher_probs", True),
            source_gate_detach_pred_logits=_trainer.get("source_gate_detach_pred_logits", False),
            source_gate_balance_reg=_trainer.get("source_gate_balance_reg", 0.0),
            source_gate_entropy_reg=_trainer.get("source_gate_entropy_reg", 0.0),
            source_gate_monitor=_trainer.get("source_gate_monitor", "semantic_miou_dual_space_gate"),
            source_gate_training_target=_trainer.get("source_gate_training_target", "none"),
            source_gate_single_weight=_trainer.get("source_gate_single_weight", 1.0),
            source_gate_multiview_weight=_trainer.get("source_gate_multiview_weight", 1.0),
            source_gate_conflict_weight=_trainer.get("source_gate_conflict_weight", 0.5),
            source_gate_odise_prior=_trainer.get("source_gate_odise_prior", 1.2),
            source_gate_lseg_prior=_trainer.get("source_gate_lseg_prior", 1.0),
            source_gate_conflict_safe_min=_trainer.get("source_gate_conflict_safe_min", 0.25),
            source_gate_mv_iou_threshold=_trainer.get("source_gate_mv_iou_threshold", 0.15),
            source_gate_mv_topk=_trainer.get("source_gate_mv_topk", 5),
            source_gate_mv_min_pairs=_trainer.get("source_gate_mv_min_pairs", 1),
            source_gate_mv_min_lifted_points=_trainer.get("source_gate_mv_min_lifted_points", 2),
            source_gate_mv_min_valid_masks=_trainer.get("source_gate_mv_min_valid_masks", 2),
            source_gate_skip_when_no_mv=_trainer.get("source_gate_skip_when_no_mv", True),
            source_gate_target_gamma=_trainer.get("source_gate_target_gamma", 2.0),
            source_gate_mv_margin=_trainer.get("source_gate_mv_margin", 0.03),
            source_gate_use_margin_filter=_trainer.get("source_gate_use_margin_filter", True),
            source_gate_mv_default_stability=_trainer.get("source_gate_mv_default_stability", 0.5),
            source_gate_mask_quality_weight=_trainer.get("source_gate_mask_quality_weight", 1.0),
            source_gate_point_conf_weight=_trainer.get("source_gate_point_conf_weight", 1.0),
            allow_source_gate_gt_ce_upper_bound=_trainer.get("allow_source_gate_gt_ce_upper_bound", False),
            source_gate_train_query_file=_trainer.get("source_gate_train_query_file", None),
            source_gate_num_train_queries=_trainer.get("source_gate_num_train_queries", 64),
            dual_branch_probe=_trainer.get("dual_branch_probe", False),
            dual_branch_probe_weight=_trainer.get("dual_branch_probe_weight", 0.0),
            dual_branch_oracle_margin=_trainer.get("dual_branch_oracle_margin", 0.02),
            dual_branch_probe_log_every=_trainer.get("dual_branch_probe_log_every", 20),
            projected_sem_probe=_trainer.get("projected_sem_probe", False),
            projected_sem_probe_min_views=_trainer.get("projected_sem_probe_min_views", 2),
            projected_sem_probe_max_points=_trainer.get("projected_sem_probe_max_points", 4096),
            projected_sem_probe_region_mode=_trainer.get("projected_sem_probe_region_mode", "point"),
            projected_sem_probe_iou_weighted=_trainer.get("projected_sem_probe_iou_weighted", False),
            projected_sem_probe_log_every=_trainer.get("projected_sem_probe_log_every", 20),
            projected_sem_gate_scale=_trainer.get("projected_sem_gate_scale", 10.0),
            alignment_query_mode=str(_trainer.get("alignment_query_mode", _model.get("alignment_query_mode", "fused"))),
            semantic_readout_ablation=_trainer.get("semantic_readout_ablation", False),
            semantic_size_aware=_trainer.get("semantic_size_aware", True),
            semantic_small_area_thr=_trainer.get("semantic_small_area_thr", 0.01),
            semantic_medium_area_thr=_trainer.get("semantic_medium_area_thr", 0.10),
            semantic_small_lseg_weight=_trainer.get("semantic_small_lseg_weight", 0.45),
            semantic_medium_lseg_weight=_trainer.get("semantic_medium_lseg_weight", 0.65),
            semantic_large_lseg_weight=_trainer.get("semantic_large_lseg_weight", 0.80),
            semantic_projected_gate=_trainer.get("semantic_projected_gate", True),
            semantic_projected_gate_scale=_trainer.get("semantic_projected_gate_scale", 10.0),
            semantic_projected_gate_min=_trainer.get("semantic_projected_gate_min", 0.45),
            semantic_projected_gate_max=_trainer.get("semantic_projected_gate_max", 0.85),
            semantic_projected_gate_default=_trainer.get("semantic_projected_gate_default", 0.70),
            semantic_projected_size_gate=_trainer.get("semantic_projected_size_gate", True),
            semantic_projected_size_base=_trainer.get("semantic_projected_size_base", 0.65),
            semantic_projected_size_beta=_trainer.get("semantic_projected_size_beta", 1.0),
            semantic_projected_size_gamma=_trainer.get("semantic_projected_size_gamma", 0.20),
            semantic_projected_size_min=_trainer.get("semantic_projected_size_min", 0.35),
            semantic_projected_size_max=_trainer.get("semantic_projected_size_max", 0.85),
            use_point_gate_loss=_trainer.get("use_point_gate_loss", False),
            lambda_point_gate=_trainer.get("lambda_point_gate", 0.05),
            point_gate_target_scale=_trainer.get("point_gate_target_scale", 10.0),
            point_gate_target_min=_trainer.get("point_gate_target_min", 0.45),
            point_gate_target_max=_trainer.get("point_gate_target_max", 0.85),
            point_gate_target_default=_trainer.get("point_gate_target_default", 0.70),
            point_gate_min_views=_trainer.get("point_gate_min_views", 2),
            point_gate_max_points=_trainer.get("point_gate_max_points", 20000),
            point_gate_loss_type=_trainer.get("point_gate_loss_type", "mse"),
            point_gate_detach_target=_trainer.get("point_gate_detach_target", True),
            use_region_gate_loss=_trainer.get("use_region_gate_loss", False),
            lambda_region_gate=_trainer.get("lambda_region_gate", 0.05),
            region_gate_input_mode=_trainer.get("region_gate_input_mode", "fused_plus_all_no_text_signals"),
            region_gate_target_mode=_trainer.get("region_gate_target_mode", "mv_plus_sharp"),
            region_gate_mv_weight=_trainer.get("region_gate_mv_weight", 1.0),
            region_gate_sharp_weight=_trainer.get("region_gate_sharp_weight", 0.5),
            region_gate_target_scale=_trainer.get("region_gate_target_scale", 5.0),
            region_gate_target_min=_trainer.get("region_gate_target_min", 0.35),
            region_gate_target_max=_trainer.get("region_gate_target_max", 0.85),
            region_gate_target_default=_trainer.get("region_gate_target_default", 0.70),
            region_gate_mv_iou_thr=_trainer.get("region_gate_mv_iou_thr", 0.05),
            region_gate_max_pairs_per_mask=_trainer.get("region_gate_max_pairs_per_mask", 10),
            region_gate_min_lifted_points=_trainer.get("region_gate_min_lifted_points", 5),
            region_gate_loss_type=_trainer.get("region_gate_loss_type", "mse"),
            region_gate_detach_target=_trainer.get("region_gate_detach_target", True),
            multiview_batch=multiview_batch,
            scenes_per_batch=scenes_per_batch,
            views_per_scene=views_per_scene,
            semantic_readout_mode=_trainer.get("semantic_readout_mode", "projected_gate"),
            validation_log_every_batches=_trainer.get("validation_log_every_batches", 25),
            fast_val=_trainer.get("fast_val", True),
            fast_val_only_main_metric=_trainer.get("fast_val_only_main_metric", True),
            eval_only=eval_only,
            lambda_align=_trainer.get("lambda_align", _trainer.get("mask_distill_weight", 1.0)),
            use_lseg_semantic_loss=_trainer.get("use_lseg_semantic_loss", False),
            use_odise_semantic_loss=_trainer.get("use_odise_semantic_loss", False),
            enable_verbose_legacy_probes=_trainer.get("enable_verbose_legacy_probes", False),
            enable_legacy_source_gate_logs=_trainer.get("enable_legacy_source_gate_logs", False),
            enable_size_aware_ablation=_trainer.get("enable_size_aware_ablation", True),
            enable_projected_size_gate_ablation=_trainer.get("enable_projected_size_gate_ablation", True),
            allow_gt_ce_upper_bound=_trainer.get("allow_gt_ce_upper_bound", False),
        )
        model_config.alignment_query_mode = trainer_config.alignment_query_mode

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
    print(f"  Precomputed projections: {projection_dir if projection_dir and os.path.exists(projection_dir or '') else 'No (runtime)'}")
    print(f"  Online extraction: {can_extract_online}")
    print(f"  Batch size: {dataloader_config.batch_size} (min 2 for MinkowskiEngine BatchNorm)")
    print(f"  3D backbone: {model_config.pc_arch}")
    print(f"  Epochs: {trainer_config.num_epochs}")
    print(f"  Learning rate: {trainer_config.base_lr}")
    print(f"  AMP enabled: {trainer_config.use_amp}")
    print(f"  Model half (float16): {trainer_config.use_model_half}")
    print("=" * 60)

    # Free GPU memory before creating model (helps avoid OOM when model is large)
    if device == "cuda":
        torch.cuda.empty_cache()

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset_config, dataloader_config, val_split="val"
    )
    print(f"Train loader: {len(train_loader)} batches")

    # Create model (smaller backbone via config model.pc_arch, e.g. MinkUNet18A, reduces VRAM)
    model = OpenVocab3DFusionModelV2(model_config)
    model = model.to(device)

    # Create trainer and run
    if use_mask_distill:
        trainer = MaskDistillTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            device=device,
            rank=0,
        )
    else:
        trainer = OpenVocabTrainerV2(
            model=model,
            train_loader=train_loader,
            config=trainer_config,
            device=device,
            val_loader=val_loader,
        )

    if eval_only and use_mask_distill:
        results = trainer.evaluate_only()
        print(
            "[EvalOnly Result] "
            f"val_loss={results.get('loss', 0.0):.4f} "
            f"alignment_loss={results.get('loss_mask_distill', 0.0):.4f} "
            f"mask_iou={results.get('mask_miou', 0.0):.4f} "
            f"semantic_miou_learned_region_gate={results.get('semantic_miou_learned_region_gate', 0.0):.4f}  "
            f"semantic_miou_projected_gate={results.get('semantic_miou_projected_gate', 0.0):.4f}"
        )
    else:
        results = trainer.train()
        print(f"Training finished: {results}")


if __name__ == "__main__":
    main()
