"""Benchmark mix_v1 dataset loading throughput.

Run from mix_v1:
  python io_cache/benchmark_dataloader.py --config config/train_scannet_v2_full_multi_gpu.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.open_vocab_dataset_v2 import (  # noqa: E402
    OpenVocabDatasetV2Config,
    OpenVocabScannetDatasetV2,
    open_vocab_collate_v2,
)


def _resolve(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_scannet_v2_full_multi_gpu.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    args = parser.parse_args()

    with open(_resolve(args.config), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    dataset_cfg = cfg.get("dataset") or {}
    dataloader_cfg = cfg.get("dataloader") or {}
    batch_size = args.batch_size or dataloader_cfg.get("batch_size", 16)
    num_workers = (
        args.num_workers
        if args.num_workers >= 0
        else dataloader_cfg.get("num_workers", 8)
    )

    ds = OpenVocabScannetDatasetV2(
        OpenVocabDatasetV2Config(
            data_config_path=_resolve(dataset_cfg.get("data_config_path", "config/data_scannet_3d.yaml")),
            precomputed_dir=dataset_cfg.get("precomputed_dir"),
            projection_dir=dataset_cfg.get("projection_dir"),
            split=args.split,
            aug=(args.split == "train" and dataset_cfg.get("aug", False)),
            max_samples=args.max_samples,
            max_samples_ratio=None,
        )
    )

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = dataloader_cfg.get("persistent_workers", True)
        loader_kwargs["prefetch_factor"] = dataloader_cfg.get("prefetch_factor", 4)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=open_vocab_collate_v2,
        **loader_kwargs,
    )

    t0 = time.perf_counter()
    last = t0
    seen = 0
    for batch_idx, batch in enumerate(loader, start=1):
        now = time.perf_counter()
        seen += batch["pixel_pooled"].shape[0]
        print(
            f"batch={batch_idx:03d} dt={now - last:.3f}s "
            f"avg={(now - t0) / batch_idx:.3f}s samples={seen} "
            f"points={batch['coords_3d'].shape[0]} K={batch['pixel_pooled'].shape[1]}",
            flush=True,
        )
        last = now
        if batch_idx >= args.batches:
            break

    total = time.perf_counter() - t0
    print(f"total={total:.3f}s batches={batch_idx} avg_batch={total / batch_idx:.3f}s")


if __name__ == "__main__":
    main()
