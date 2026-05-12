"""
Multi-GPU Training script for Open Vocabulary 3D Fusion Model V2 using DDP.

Usage:
    # Single GPU (same as train_open_vocab_v2.py)
    python train_open_vocab_v2_ddp.py --config config/train_scannet_v2.yaml

    # Multi-GPU with torchrun (recommended)
    torchrun --nproc_per_node=2 train_open_vocab_v2_ddp.py --config config/train_scannet_v2.yaml

    # Multi-GPU with specific GPUs
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_open_vocab_v2_ddp.py --config config/train_scannet_v2.yaml
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("CLIP_CACHE_DIR", str(REPO_ROOT / "checkpoints" / "pretrained" / "clip"))
os.environ.setdefault("ODISE_MODEL_ZOO", str(REPO_ROOT / "checkpoints" / "pretrained"))
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / "checkpoints" / "pretrained" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "checkpoints" / "pretrained" / "xdg"))
MASK2FORMER_ROOT = REPO_ROOT / "ODISE" / "third_party" / "Mask2Former"
if MASK2FORMER_ROOT.exists() and str(MASK2FORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(MASK2FORMER_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
# 新版蒸馏 trainer（通过 --use-distill 启用）
try:
    from experiment_distill.trainer_distill import DistillTrainer, DistillTrainerConfig
    _DISTILL_AVAILABLE = True
except ImportError:
    _DISTILL_AVAILABLE = False

# Mask Distillation trainer（通过 --use-mask-distill 启用，Diff2Scene 方案）
try:
    from experiment_mask_distill.trainer_mask_distill import (
        MaskDistillTrainer, MaskDistillTrainerConfig,
    )
    _MASK_DISTILL_AVAILABLE = True
except ImportError:
    _MASK_DISTILL_AVAILABLE = False


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def setup_distributed(gpu_ids=None):
    """
    Initialize distributed training if available.
    
    Args:
        gpu_ids: List of GPU IDs from config (e.g., [0, 1]). 
                 If provided and not using torchrun, will set CUDA_VISIBLE_DEVICES.
    """
    # If using torchrun/torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
        
        if rank == 0:
            print(f"Initialized DDP: world_size={world_size}, backend=nccl")
        
        return local_rank, world_size, True
    else:
        # Single GPU mode - use gpu_ids[0] if provided
        if gpu_ids is not None and len(gpu_ids) > 0:
            gpu_id = gpu_ids[0]
            torch.cuda.set_device(gpu_id)
            return gpu_id, 1, False
        return 0, 1, False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class DataLoaderConfig:
    batch_size: int = 2
    val_batch_size: int = 0
    num_workers: int = 4
    val_num_workers: Optional[int] = None
    pin_memory: bool = True
    drop_last: bool = True
    val_max_samples: Optional[int] = None
    val_max_samples_ratio: Optional[float] = None


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int, rank: int = 0) -> None:
    """Set random seeds for reproducibility."""
    seed = seed + rank  # Different seed per rank for data augmentation diversity
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
    use_distributed: bool = False,
) -> tuple:
    """Create train and validation data loaders with optional DDP support."""
    train_dataset = OpenVocabScannetDatasetV2(dataset_config)
    
    # Create sampler for distributed training
    train_sampler = None
    shuffle = True
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
        )
        shuffle = False  # Sampler handles shuffling
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataloader_config.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=dataloader_config.num_workers,
        pin_memory=dataloader_config.pin_memory,
        drop_last=dataloader_config.drop_last,
        collate_fn=open_vocab_collate_v2,
        persistent_workers=dataloader_config.num_workers > 0,
    )

    # Create validation loader (only on main process or all processes)
    val_loader = None
    try:
        val_config = OpenVocabDatasetV2Config(
            data_config_path=dataset_config.data_config_path,
            precomputed_dir=dataset_config.precomputed_dir,
            projection_dir=dataset_config.projection_dir,
            split=val_split,
            scannet200=dataset_config.scannet200,
            voxel_size=dataset_config.voxel_size,
            aug=False,
            identifier=dataset_config.identifier + 1,
            loop=1,
            eval_all=True,
            input_color=dataset_config.input_color,
            max_samples=dataloader_config.val_max_samples,
            max_samples_ratio=dataloader_config.val_max_samples_ratio,
        )
        val_dataset = OpenVocabScannetDatasetV2(val_config)
        
        val_sampler = None
        if use_distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
            )
        
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
            sampler=val_sampler,
            num_workers=val_num_workers,
            pin_memory=dataloader_config.pin_memory,
            drop_last=False,
            collate_fn=open_vocab_collate_v2,
            persistent_workers=val_num_workers > 0,
        )
        if is_main_process():
            print(
                f"Created validation loader with {len(val_dataset)} samples "
                f"(batch_size={val_batch_size}, num_workers={val_num_workers})"
            )
    except Exception as e:
        if is_main_process():
            print(f"Could not create validation loader: {e}")

    return train_loader, val_loader, train_sampler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Open-Vocabulary 3D Fusion Model V2 with Multi-GPU support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file")

    # Dataset
    parser.add_argument("--data-config-path", type=str, default="config/data_scannet_3d.yaml")
    parser.add_argument("--precomputed-dir", type=str, default="")
    parser.add_argument("--scannet200", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--aug", action="store_true")

    # Model
    parser.add_argument("--label-path", type=str, default="")
    parser.add_argument("--lseg-ckpt-path", type=str, default="")
    parser.add_argument("--odise-model-config-path", type=str, default="")

    # Training
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--scheduler-type", type=str, default="cosine")
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)

    # Data loading
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)

    # Checkpointing
    parser.add_argument("--log-dir", type=str, default="runs/open_vocab_3d_v2")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--val-every-epochs", type=int, default=1)  # 🔥 修复：添加验证间隔参数

    # AMP and early stopping
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--model-half", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=10)

    # Misc
    parser.add_argument("--seed", type=int, default=1342)
    parser.add_argument("--device", type=str, default="cuda")
    # 新版蒸馏训练开关
    parser.add_argument("--use-distill", action="store_true",
                        help="Use distillation trainer (experiment_distill)")
    # Mask distillation 训练开关（Diff2Scene 方案）
    parser.add_argument("--use-mask-distill", action="store_true",
                        help="Use mask distillation trainer (experiment_mask_distill, Diff2Scene)")

    args = parser.parse_args()
    
    # Load YAML config first to get gpu_ids
    yaml_config = load_yaml_config(args.config) or {}
    
    # Get GPU IDs from config
    gpu_ids = yaml_config.get("gpu_ids", [0])
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]

    # Setup distributed training
    local_rank, world_size, use_distributed = setup_distributed(gpu_ids)

    # Set seed (different per rank)
    seed = yaml_config.get("seed", args.seed)
    set_seed(seed, get_rank())

    # Device
    if use_distributed:
        device = f"cuda:{local_rank}"
    else:
        device = yaml_config.get("device", args.device)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

    # Resolve paths
    repo_root = os.path.abspath(os.path.dirname(__file__))

    # Dataset config
    _dataset = yaml_config.get("dataset") or {}
    precomputed_dir = _dataset.get("precomputed_dir", args.precomputed_dir)
    if precomputed_dir and not os.path.isabs(precomputed_dir):
        precomputed_dir = os.path.join(repo_root, precomputed_dir)
    if precomputed_dir and not os.path.exists(precomputed_dir):
        if is_main_process():
            print(f"Warning: precomputed_dir not found: {precomputed_dir}")
        precomputed_dir = None

    data_config_path = _dataset.get("data_config_path", args.data_config_path)
    if data_config_path and not os.path.isabs(data_config_path):
        data_config_path = os.path.join(repo_root, data_config_path)
    if not data_config_path or not os.path.exists(data_config_path):
        data_config_path = os.path.join(repo_root, "config/data_scannet_3d.yaml")

    projection_dir = _dataset.get("projection_dir", None)
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

    _dataloader = yaml_config.get("dataloader") or {}
    raw_batch_size = _dataloader.get("batch_size", args.batch_size)
    if raw_batch_size < 2:
        if is_main_process():
            print(f"Warning: batch_size={raw_batch_size} < 2, using 2")
        raw_batch_size = 2
    
    # Scale batch size by world size for effective batch size
    per_gpu_batch_size = raw_batch_size
    effective_batch_size = per_gpu_batch_size * world_size
    
    dataloader_config = DataLoaderConfig(
        batch_size=per_gpu_batch_size,
        val_batch_size=_dataloader.get("val_batch_size", 0),
        num_workers=_dataloader.get("num_workers", args.num_workers),
        val_num_workers=_dataloader.get("val_num_workers"),
        val_max_samples=_dataloader.get("val_max_samples"),
        val_max_samples_ratio=_dataloader.get("val_max_samples_ratio"),
    )

    # Model config
    _model = yaml_config.get("model") or {}
    label_path = _model.get("label_path", args.label_path)
    lseg_ckpt_path = _model.get("lseg_ckpt_path", args.lseg_ckpt_path)
    odise_config_path = _model.get("odise_model_config_path", args.odise_model_config_path)

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
        pc_arch=_model.get("pc_arch", "MinkUNet34C"),
        pixel_embedding_dim=_model.get("pixel_embedding_dim", 512),
        mask_embedding_dim=_model.get("mask_embedding_dim", 256),
        fused_embedding_dim=_model.get("fused_embedding_dim", 256),
        alpha_max=_model.get("alpha_max", 0.2),
        pc_last_dim=_model.get("pc_last_dim", 256),
    )

    # Trainer config
    _trainer = yaml_config.get("trainer") or {}
    resume_checkpoint = _trainer.get("resume", args.resume)
    if resume_checkpoint and not os.path.isabs(resume_checkpoint):
        resume_checkpoint = os.path.join(repo_root, resume_checkpoint)
    if resume_checkpoint and not os.path.exists(resume_checkpoint):
        resume_checkpoint = None

    # Scale learning rate by world size (linear scaling rule)
    base_lr = _trainer.get("base_lr", args.base_lr)
    scaled_lr = base_lr * world_size if world_size > 1 else base_lr

    # 共用的 trainer 超参（旧版和新版都有的字段）
    _common_trainer_kwargs = dict(
        num_epochs=_trainer.get("num_epochs", args.num_epochs),
        base_lr=scaled_lr,
        weight_decay=_trainer.get("weight_decay", args.weight_decay),
        grad_clip_norm=_trainer.get("grad_clip_norm", args.grad_clip_norm),
        warmup_epochs=_trainer.get("warmup_epochs", args.warmup_epochs),
        scheduler_type=_trainer.get("scheduler_type", args.scheduler_type),
        bce_weight=_trainer.get("bce_weight", args.bce_weight),
        dice_weight=_trainer.get("dice_weight", args.dice_weight),
        min_points_per_mask=_trainer.get("min_points_per_mask", 10),
        log_dir=_trainer.get("log_dir", args.log_dir),
        checkpoint_dir=_trainer.get("checkpoint_dir", args.checkpoint_dir),
        save_every_epochs=_trainer.get("save_every_epochs", args.save_every_epochs),
        val_every_epochs=_trainer.get("val_every_epochs", args.val_every_epochs),
        use_amp=not args.no_amp,
        early_stopping_patience=_trainer.get("early_stopping_patience", args.early_stopping_patience),
        resume_checkpoint=resume_checkpoint,
        use_model_half=_trainer.get("use_model_half", args.model_half),
        gradient_accumulation_steps=_trainer.get("gradient_accumulation_steps", 1),
        semantic_clip_model=_trainer.get("semantic_clip_model", "ODISE-256"),
        semantic_pixel_clip_model=_trainer.get("semantic_pixel_clip_model", "ViT-B/32"),
        semantic_prompt_template=_trainer.get("semantic_prompt_template", "a photo of a {}"),
        semantic_pc_lambda=_trainer.get("semantic_pc_lambda", 0.5),
        validation_subprocess=_trainer.get("validation_subprocess", False),
        validation_config_path=args.config or "config/train_scannet_v2_full_multi_gpu.yaml",
        validation_device=device,
    )

    use_distill      = args.use_distill      or _trainer.get("use_distill",      False)
    use_mask_distill = args.use_mask_distill or _trainer.get("use_mask_distill", False)

    if use_mask_distill:
        if not _MASK_DISTILL_AVAILABLE:
            raise ImportError("experiment_mask_distill not found; cannot use --use-mask-distill")
        # _common_trainer_kwargs 已含 bce_weight/dice_weight，mask distill 用独立值覆盖
        _mask_distill_kwargs = {k: v for k, v in _common_trainer_kwargs.items()
                                if k not in ("bce_weight", "dice_weight")}
        trainer_config = MaskDistillTrainerConfig(
            **_mask_distill_kwargs,
            mask_distill_weight=_trainer.get("mask_distill_weight", 1.0),
            mask_student_weight=_trainer.get("mask_student_weight", 1.0),
            mask_joint_weight=_trainer.get("mask_joint_weight", 0.05),
            nce_weight=_trainer.get("nce_weight", 0.5),
            nce_type=_trainer.get("nce_type", "hard"),
            nce_tau=_trainer.get("nce_tau", 0.1),
            nce_tau_teacher=_trainer.get("nce_tau_teacher", 0.2),
            vicreg_weight=_trainer.get("vicreg_weight", 0.01),
            bce_weight=_trainer.get("bce_weight", 0.0),
            dice_weight=_trainer.get("dice_weight", 0.0),
            validation_log_every_batches=_trainer.get("validation_log_every_batches", 25),
            validation_subprocess=_trainer.get("validation_subprocess", False),
            validation_config_path=args.config or "config/train_scannet_v2_full_multi_gpu.yaml",
            validation_device=device,
        )
        if is_main_process():
            print(f"[MaskDistill] mask_student_weight={trainer_config.mask_student_weight}  "
                  f"mask_joint_weight={trainer_config.mask_joint_weight}  "
                  f"nce_weight={trainer_config.nce_weight}  "
                  f"vicreg_weight={trainer_config.vicreg_weight}  "
                  f"bce_weight={trainer_config.bce_weight}  "
                  f"dice_weight={trainer_config.dice_weight}")
    elif use_distill:
        if not _DISTILL_AVAILABLE:
            raise ImportError("experiment_distill not found; cannot use --use-distill")
        trainer_config = DistillTrainerConfig(
            **_common_trainer_kwargs,
            feat_loss_weight=_trainer.get("feat_loss_weight", 1.0),
            mask_loss_weight=_trainer.get("mask_loss_weight", 0.1),
        )
        if is_main_process():
            print(f"[Distill] feat_loss_weight={trainer_config.feat_loss_weight}  "
                  f"mask_loss_weight={trainer_config.mask_loss_weight}")
    else:
        trainer_config = OpenVocabTrainerV2Config(**_common_trainer_kwargs)

    # Validate configuration
    if not os.path.exists(dataset_config.data_config_path):
        raise FileNotFoundError(f"data_config_path not found: {dataset_config.data_config_path}")

    use_precomputed = dataset_config.precomputed_dir is not None and os.path.exists(dataset_config.precomputed_dir)
    can_extract_online = (
        model_config.label_path is not None
        and model_config.lseg_ckpt_path is not None
        and model_config.odise_model_config_path is not None
    )

    if not use_precomputed and not can_extract_online:
        raise ValueError(
            "Either provide --precomputed-dir or provide extraction model paths"
        )

    eval_split = (_trainer.get("eval_split") or "val").strip().lower()
    if eval_split != "val":
        raise ValueError(
            f"Training-time evaluation must use val split, got eval_split={eval_split!r}. "
            "Do not use test for training-time validation or model selection."
        )

    if is_main_process():
        print("=" * 60)
        print("Configuration:")
        print(f"  Device: {device}")
        print(f"  World size: {world_size}")
        print(f"  Distributed: {use_distributed}")
        print(f"  Per-GPU batch size: {per_gpu_batch_size}")
        print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Train num_workers: {dataloader_config.num_workers}")
        print(f"  Val batch size per GPU: {dataloader_config.val_batch_size or dataloader_config.batch_size}")
        print(
            "  Val num_workers: "
            f"{dataloader_config.num_workers if dataloader_config.val_num_workers is None else dataloader_config.val_num_workers}"
        )
        print(f"  Base LR: {base_lr} -> Scaled LR: {scaled_lr}")
        print(f"  Precomputed features: {use_precomputed}")
        print(f"  Precomputed projections: {projection_dir if projection_dir else 'No'}")
        print(f"  Checkpoint dir: {trainer_config.checkpoint_dir}")
        print(f"  Log dir: {trainer_config.log_dir}")
        print(f"  Eval split (val loader): {eval_split}")
        print(f"  3D backbone: {model_config.pc_arch}")
        print(f"  Epochs: {trainer_config.num_epochs}")
        print(f"  AMP enabled: {trainer_config.use_amp}")
        print("=" * 60)

    # Free GPU memory
    torch.cuda.empty_cache()

    # Create data loaders (train uses dataset.split; periodic eval uses trainer.eval_split, default val).
    train_loader, val_loader, train_sampler = create_data_loaders(
        dataset_config,
        dataloader_config,
        val_split=eval_split,
        use_distributed=use_distributed,
    )
    if is_main_process():
        print(f"Train loader: {len(train_loader)} batches")

    # Create model
    model = OpenVocab3DFusionModelV2(model_config)
    model = model.to(device)

    # Wrap model with DDP if distributed
    if use_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # Required for some architectures
        )
        if is_main_process():
            print("Model wrapped with DistributedDataParallel")

    # Create trainer and run
    if use_mask_distill:
        trainer = MaskDistillTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            device=device,
            rank=get_rank(),
            train_sampler=train_sampler,
        )
    elif use_distill:
        trainer = DistillTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            device=device,
            rank=get_rank(),
            train_sampler=train_sampler,
        )
    else:
        trainer = OpenVocabTrainerV2(
            model=model,
            train_loader=train_loader,
            config=trainer_config,
            device=device,
            val_loader=val_loader,
            train_sampler=train_sampler,
            is_distributed=use_distributed,
            is_main_process=is_main_process(),
        )

    try:
        results = trainer.train()
        if is_main_process():
            print(f"Training finished: {results}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
