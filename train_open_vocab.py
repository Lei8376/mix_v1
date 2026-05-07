import argparse
from dataclasses import asdict, dataclass
import os
from typing import Any, Dict

import numpy as np
import torch
import yaml

from dataset.open_vocab_dataset import (
    OpenVocabDatasetConfig,
    OpenVocabScannetDataset,
    open_vocab_collate,
)
from model.open_vocab_fusion import (
    OpenVocabFusionModelConfig,
    OpenVocab3DFusionModel,
)
from trainer.open_vocab_trainer import OpenVocabTrainer, OpenVocabTrainerConfig


@dataclass
class DataLoaderConfig:
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_odise_config_path(config_path: str, repo_root: str) -> str:
    if not config_path:
        raise ValueError("odise_model_config_path is required.")
    configs_root = os.path.join(repo_root, "ODISE", "configs")
    if os.path.isabs(config_path) and os.path.exists(config_path):
        if not os.path.commonpath([configs_root, config_path]) == configs_root:
            raise ValueError(
                "odise_model_config_path must be under ODISE/configs or be a "
                "relative config name like Panoptic/odise_caption_coco_50e.py."
            )
        return os.path.relpath(config_path, configs_root)
    candidate_path = os.path.join(configs_root, config_path)
    if os.path.exists(candidate_path):
        return config_path
    raise FileNotFoundError(
        "odise_model_config_path not found in ODISE/configs: "
        f"{config_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Open-Vocabulary 3D Fusion")
    parser.add_argument("--config", type=str, default="config/data_scannet_3d.yaml")
    parser.add_argument("--data-config-path", type=str, default="config/data_scannet_3d.yaml")
    parser.add_argument("--label-path", type=str, default="lang_seg/label_files/ade20k_objectInfo150.txt")
    parser.add_argument("--lseg-ckpt-path", type=str, default="lang_seg/checkpoints/demo_e200.ckpt")
    parser.add_argument("--odise-model-config-path", type=str, default="Panoptic/odise_caption_coco_50e.py")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", type=bool, default=True)
    parser.add_argument("--drop-last", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1342)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config_from_yaml = _load_yaml_config(args.config)

    dataset_config = OpenVocabDatasetConfig(
        data_config_path=config_from_yaml.get("dataset", {}).get(
            "data_config_path", args.data_config_path
        ),
        split=config_from_yaml.get("dataset", {}).get("split", "train"),
        scannet200=config_from_yaml.get("dataset", {}).get("scannet200", False),
        voxel_size=config_from_yaml.get("dataset", {}).get("voxel_size", 0.05),
        aug=config_from_yaml.get("dataset", {}).get("aug", False),
        memcache_init=config_from_yaml.get("dataset", {}).get("memcache_init", False),
        identifier=config_from_yaml.get("dataset", {}).get("identifier", 7791),
        loop=config_from_yaml.get("dataset", {}).get("loop", 1),
        eval_all=config_from_yaml.get("dataset", {}).get("eval_all", False),
        input_color=config_from_yaml.get("dataset", {}).get("input_color", False),
    )

    dataloader_config = DataLoaderConfig(
        batch_size=config_from_yaml.get("dataloader", {}).get(
            "batch_size", args.batch_size
        ),
        num_workers=config_from_yaml.get("dataloader", {}).get(
            "num_workers", args.num_workers
        ),
        pin_memory=config_from_yaml.get("dataloader", {}).get(
            "pin_memory", args.pin_memory
        ),
        drop_last=config_from_yaml.get("dataloader", {}).get(
            "drop_last", args.drop_last
        ),
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    repo_root = os.path.abspath(os.path.dirname(__file__))
    odise_model_config_path = _resolve_odise_config_path(
        config_from_yaml.get("model", {}).get(
            "odise_model_config_path", args.odise_model_config_path
        ),
        repo_root,
    )

    model_config = OpenVocabFusionModelConfig(
        label_path=config_from_yaml.get("model", {}).get("label_path", args.label_path),
        lseg_ckpt_path=config_from_yaml.get("model", {}).get(
            "lseg_ckpt_path", args.lseg_ckpt_path
        ),
        odise_model_config_path=odise_model_config_path,
        device=device,
    )

    trainer_config = OpenVocabTrainerConfig(
        num_epochs=config_from_yaml.get("trainer", {}).get(
            "num_epochs", args.num_epochs
        ),
        base_lr=config_from_yaml.get("trainer", {}).get("base_lr", args.base_lr),
        weight_decay=config_from_yaml.get("trainer", {}).get(
            "weight_decay", args.weight_decay
        ),
        grad_clip_norm=config_from_yaml.get("trainer", {}).get(
            "grad_clip_norm", args.grad_clip_norm
        ),
    )

    if not os.path.exists(dataset_config.data_config_path):
        raise FileNotFoundError(
            f"data_config_path not found: {dataset_config.data_config_path}"
        )
    if not os.path.exists(model_config.label_path):
        raise FileNotFoundError(f"label_path not found: {model_config.label_path}")
    if not os.path.exists(model_config.lseg_ckpt_path):
        raise FileNotFoundError(
            f"lseg_ckpt_path not found: {model_config.lseg_ckpt_path}"
        )

    _set_seed(args.seed)

    dataset = OpenVocabScannetDataset(dataset_config)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataloader_config.batch_size,
        shuffle=True,
        num_workers=dataloader_config.num_workers,
        pin_memory=dataloader_config.pin_memory,
        drop_last=dataloader_config.drop_last,
        collate_fn=open_vocab_collate,
    )

    model = OpenVocab3DFusionModel(model_config)
    trainer = OpenVocabTrainer(
        model=model,
        train_loader=train_loader,
        config=trainer_config,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
