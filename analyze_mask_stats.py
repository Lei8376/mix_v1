"""
快速统计训练数据中的空 mask 比例
不需要运行完整训练，只需要跑几个 batch 就能得到统计结果
"""

import sys
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset.open_vocab_dataset_v2 import (
    OpenVocabScannetDatasetV2,
    OpenVocabDatasetV2Config,
    open_vocab_collate_v2,
)
from torch.utils.data import DataLoader


def analyze_mask_statistics(
    config_path: str, num_batches: int = 50, split: str = "train", shuffle: bool = False
):
    """
    统计数据集中的 mask 质量
    
    Args:
        config_path: 配置文件路径
        num_batches: 统计多少个 batch（默认 50，约 800 个样本）
        split: 'train' 或 'val'
        shuffle: 是否打乱样本顺序（更容易抓到“稀有空 mask”样本）
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建 dataset config
    dataset_cfg = config['dataset']
    dataloader_config = config['dataloader']
    
    dataset_config = OpenVocabDatasetV2Config(
        data_config_path=dataset_cfg['data_config_path'],
        precomputed_dir=dataset_cfg.get('precomputed_dir'),
        projection_dir=dataset_cfg.get('projection_dir'),
        split=split,
        max_samples_ratio=dataset_cfg.get('max_samples_ratio'),
    )
    
    # 创建 dataset
    dataset = OpenVocabScannetDatasetV2(
        config=dataset_config,
    )
    
    actual_batch_size = int(dataloader_config["batch_size"])

    # 创建 dataloader（使用与训练一致的 collate）
    dataloader = DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=shuffle,
        num_workers=min(4, int(dataloader_config.get("num_workers", 4))),
        pin_memory=True,
        collate_fn=open_vocab_collate_v2,
        drop_last=False,
    )

    total_samples_to_process = num_batches * actual_batch_size
    
    # 统计变量
    total_samples = 0
    empty_mask_count = 0  # ODISE 没给出任何有效 mask
    no_valid_gt_count = 0  # GT 过滤后无有效 mask
    
    # 统计每个样本的 mask 数量分布
    mask_count_distribution = []
    valid_mask_count_distribution = []
    
    # 统计 GT 过滤情况（使用不同的 min_points 阈值）
    min_points_thresholds = [1, 5, 7, 10, 15, 20]
    filtered_by_threshold = {th: 0 for th in min_points_thresholds}
    
    print(f"\n{'='*60}")
    print(f"📊 Analyzing {split} dataset from: {config_path}")
    print(f"   Processing {total_samples_to_process} samples (~{num_batches} batches of size {actual_batch_size})")
    print(f"{'='*60}\n")
    
    # 遍历数据（以“训练时的 batch”为单位）
    with torch.no_grad():
        pbar = tqdm(dataloader, total=num_batches, desc="Processing batches")
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= num_batches:
                break

            masks = batch["masks"]  # (B, Kmax, H, W) float32 0/1
            mask_valid = batch["mask_valid"]  # (B, Kmax) bool
            coords_3d = batch["coords_3d"]  # (N_total, 4), 第一列是 batch 索引
            x_all = batch["x_label"]
            y_all = batch["y_label"]

            B = masks.shape[0]
            for b in range(B):
                total_samples += 1

                masks_b = masks[b]
                valid_b = mask_valid[b]

                num_total_masks = masks_b.shape[0]
                num_valid_masks = int(valid_b.sum().item())

                mask_count_distribution.append(num_total_masks)
                valid_mask_count_distribution.append(num_valid_masks)

                if num_valid_masks == 0:
                    empty_mask_count += 1
                    continue

                point_mask = coords_3d[:, 0] == b
                if not point_mask.any():
                    continue

                x_idx = x_all[point_mask]
                y_idx = y_all[point_mask]
                if x_idx.numel() == 0:
                    continue

                valid_masks = masks_b[valid_b]  # (K_valid, H, W)
                H, W = valid_masks.shape[1], valid_masks.shape[2]

                x_max = x_idx.max().item() if x_idx.numel() > 0 else 0
                y_max = y_idx.max().item() if y_idx.numel() > 0 else 0
                need_scale = (x_max > W + 20) or (y_max > H + 20)

                if need_scale:
                    # 这里假设原图是 640x480；若你的投影原始分辨率不同，改这里即可
                    x_scaled = (x_idx.float() / 640.0 * W).long().clamp(0, W - 1)
                    y_scaled = (y_idx.float() / 480.0 * H).long().clamp(0, H - 1)
                else:
                    x_scaled = x_idx.long().clamp(0, W - 1)
                    y_scaled = y_idx.long().clamp(0, H - 1)

                # (K_valid, N_points) -> (N_points, K_valid)
                gt_3d = valid_masks[:, y_scaled, x_scaled].transpose(0, 1).float()
                gt_pos = gt_3d.sum(dim=0)

                for th in min_points_thresholds:
                    keep_gt = gt_pos >= th
                    if not keep_gt.any():
                        filtered_by_threshold[th] += 1
                        break
    
    # 打印统计结果
    print(f"\n{'='*60}")
    print(f"📊 统计结果 ({split} dataset)")
    print(f"{'='*60}\n")
    
    print(f"总样本数: {total_samples}")
    print(f"\n1️⃣  Empty Masks (ODISE 没给出有效 mask):")
    print(f"   数量: {empty_mask_count}")
    print(f"   比例: {empty_mask_count / total_samples * 100:.2f}%")
    
    print(f"\n2️⃣  Mask 数量分布:")
    print(f"   平均 mask 数量: {sum(mask_count_distribution) / len(mask_count_distribution):.1f}")
    print(f"   平均 valid mask 数量: {sum(valid_mask_count_distribution) / len(valid_mask_count_distribution):.1f}")
    
    print(f"\n3️⃣  GT 过滤统计 (不同 min_points_per_mask 阈值):")
    print(f"   {'阈值':<6} {'过滤数量':<10} {'过滤比例':<10} {'有效样本比例':<15}")
    print(f"   {'-'*50}")
    for th in min_points_thresholds:
        filtered = filtered_by_threshold[th]
        filtered_ratio = filtered / total_samples * 100
        effective_ratio = (total_samples - empty_mask_count - filtered) / total_samples * 100
        print(f"   {th:<6} {filtered:<10} {filtered_ratio:>6.2f}%     {effective_ratio:>6.2f}%")
    
    print(f"\n{'='*60}")
    print(f"💡 建议:")
    print(f"{'='*60}\n")
    
    # 给出建议
    empty_ratio = empty_mask_count / total_samples * 100
    
    if empty_ratio > 5:
        print(f"⚠️  Empty mask 比例较高 ({empty_ratio:.2f}%)，可能需要检查 ODISE 质量")
    else:
        print(f"✅ Empty mask 比例正常 ({empty_ratio:.2f}%)")
    
    print(f"\n推荐的 min_points_per_mask 阈值:")
    
    # 找到最优阈值（保持 > 90% 有效样本）
    best_threshold = 1
    for th in min_points_thresholds:
        filtered = filtered_by_threshold[th]
        effective_ratio = (total_samples - empty_mask_count - filtered) / total_samples * 100
        if effective_ratio >= 90:
            best_threshold = th
        else:
            break
    
    current_threshold = config.get('trainer', {}).get('min_points_per_mask', 10)
    current_filtered = filtered_by_threshold.get(current_threshold, 0)
    current_effective = (total_samples - empty_mask_count - current_filtered) / total_samples * 100
    
    print(f"\n   当前配置: min_points_per_mask = {current_threshold}")
    print(f"   当前有效样本比例: {current_effective:.2f}%")
    
    if current_effective < 90:
        print(f"\n   ⚠️  有效样本比例过低！建议降低到 {best_threshold}")
    elif current_effective >= 95:
        print(f"\n   ✅ 有效样本比例良好")
    else:
        print(f"\n   ⚙️  可以考虑降低到 {best_threshold} 以提高数据利用率")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze mask statistics in training data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-batches", type=int, default=50, help="Number of batches to analyze")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Dataset split")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before analysis")
    
    args = parser.parse_args()
    
    analyze_mask_statistics(args.config, args.num_batches, args.split, shuffle=args.shuffle)
