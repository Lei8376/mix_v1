#!/usr/bin/env python3
"""
诊断脚本：检查训练中 "No valid masks" / "skip" 的根因和频率。

检查项：
  1. NPZ 文件中 mask 数量分布（K=0？K 极少？）
  2. 模拟 collate 后 mask_valid 全 False 的频率（padding 导致？）
  3. 3D 数据中 x_label/y_label 全零 / 越界问题
  4. 模拟 Criteria.loss_pt 中各类 skip 的频率

用法:
  cd /home/featurize/work/mix
  python tmp_diagnostics/diagnose_mask_valid.py
"""

import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict

# ──────────────────────────────────────────────────────
# 配置（与训练保持一致）
# ──────────────────────────────────────────────────────
PRECOMPUTED_DIR = Path("/home/featurize/data/pixel_pooled")
DATA_ROOT = Path("/home/featurize/data/scannet_3d")
DATA_ROOT_2D = Path("/home/featurize/data/scannet_2d")
SPLIT = "train"
MAX_SAMPLES_RATIO = 0.1  # 和训练配置一致
BATCH_SIZE = 32
NUM_BATCHES_TO_SIMULATE = 20  # 模拟多少个 batch 做深度诊断
THRESHOLD = 0.5
MIN_POINTS_PER_MASK = 10

SEPARATOR = "=" * 70


def scan_npz_files():
    """检查1: 扫描所有 NPZ 文件, 统计 mask 数量分布。"""
    print(f"\n{SEPARATOR}")
    print("【检查 1】NPZ 文件中 mask 数量 (K) 分布")
    print(SEPARATOR)

    if not PRECOMPUTED_DIR.exists():
        print(f"  ❌ precomputed_dir 不存在: {PRECOMPUTED_DIR}")
        return [], {}

    mask_counts = []
    zero_mask_files = []
    scene_mask_stats = defaultdict(list)
    total_files = 0

    for scene_dir in sorted(PRECOMPUTED_DIR.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene"):
            continue
        for npz_path in sorted(scene_dir.glob("*_odise.npz")):
            total_files += 1
            try:
                with np.load(npz_path, allow_pickle=True) as f:
                    masks = f["masks"]
                    if masks.dtype == object:
                        masks = np.stack(masks, axis=0)
                    K = masks.shape[0]
                    mask_counts.append(K)
                    scene_mask_stats[scene_dir.name].append(K)
                    if K == 0:
                        zero_mask_files.append(str(npz_path))
            except Exception as e:
                print(f"  ⚠️  读取失败: {npz_path} -> {e}")
                mask_counts.append(-1)

    if not mask_counts:
        print("  ❌ 没有找到 NPZ 文件!")
        return [], {}

    valid_counts = [c for c in mask_counts if c >= 0]
    print(f"  总 NPZ 文件数: {total_files}")
    print(f"  K=0 (无 mask) 的文件数: {len(zero_mask_files)}")
    print(f"  K>0 的文件数: {len([c for c in valid_counts if c > 0])}")
    print(f"  K 值分布:")
    print(f"    min={min(valid_counts)}, max={max(valid_counts)}, "
          f"mean={np.mean(valid_counts):.1f}, median={np.median(valid_counts):.0f}")

    # 分位数
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(valid_counts, p)
        print(f"    P{p:>2d} = {val:.0f}")

    # K 值频率直方图 (前15个最常见值)
    counter = Counter(valid_counts)
    print(f"\n  K 值频率 (Top 15):")
    for k, cnt in counter.most_common(15):
        pct = cnt / len(valid_counts) * 100
        bar = "█" * int(pct)
        print(f"    K={k:>3d}: {cnt:>6d} ({pct:5.1f}%) {bar}")

    if zero_mask_files:
        print(f"\n  ⚠️  K=0 的文件列表 (前 20):")
        for f in zero_mask_files[:20]:
            print(f"    {f}")
        if len(zero_mask_files) > 20:
            print(f"    ... 还有 {len(zero_mask_files) - 20} 个")

    return mask_counts, scene_mask_stats


def simulate_collate_mask_valid(mask_counts):
    """检查2: 模拟 collate 过程, 看 mask_valid 全 False 的频率。"""
    print(f"\n{SEPARATOR}")
    print("【检查 2】模拟 collate 后 mask_valid 全 False 的频率")
    print(SEPARATOR)

    if not mask_counts:
        print("  跳过 (无 NPZ 数据)")
        return

    # 模拟使用 max_samples_ratio 过滤
    valid_counts = [c for c in mask_counts if c >= 0]
    n_total = len(valid_counts)
    n_used = max(1, int(n_total * MAX_SAMPLES_RATIO))
    used_counts = valid_counts[:n_used]

    print(f"  总样本: {n_total}, 使用 {MAX_SAMPLES_RATIO*100:.0f}%: {n_used}")

    # 模拟多个 batch
    num_batches = min(NUM_BATCHES_TO_SIMULATE, n_used // BATCH_SIZE)
    if num_batches == 0:
        print(f"  样本不足一个 batch ({n_used} < {BATCH_SIZE})")
        return

    total_items = 0
    all_false_count = 0
    max_k_per_batch = []

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_used)
        batch_ks = used_counts[start:end]
        B = len(batch_ks)
        max_k = max(batch_ks)
        max_k_per_batch.append(max_k)

        # 模拟 padding: mask_valid[b, :k] = True, 其余 False
        for b_idx, k in enumerate(batch_ks):
            total_items += 1
            if k == 0:
                all_false_count += 1

    pct = all_false_count / total_items * 100 if total_items > 0 else 0
    print(f"  模拟 {num_batches} 个 batch (共 {total_items} 个 batch item)")
    print(f"  mask_valid 全 False 的 item 数: {all_false_count} ({pct:.2f}%)")
    print(f"  每 batch 的 max_k: min={min(max_k_per_batch)}, max={max(max_k_per_batch)}, "
          f"mean={np.mean(max_k_per_batch):.1f}")

    # 额外：K 差异大 → padding 比例高 → 浪费显存
    if max_k_per_batch:
        batch_start = 0
        padding_ratios = []
        for batch_idx in range(num_batches):
            end = min(batch_start + BATCH_SIZE, n_used)
            batch_ks = used_counts[batch_start:end]
            max_k = max(batch_ks)
            if max_k > 0:
                actual_total = sum(batch_ks)
                padded_total = max_k * len(batch_ks)
                padding_ratios.append(1 - actual_total / padded_total)
            batch_start = end
        if padding_ratios:
            print(f"\n  padding 浪费比 (mask 维度上):")
            print(f"    mean={np.mean(padding_ratios)*100:.1f}%, "
                  f"max={max(padding_ratios)*100:.1f}%")


def check_3d_data_labels():
    """检查3: 抽样检查 3D .pth 文件中 x_label / y_label 的情况。"""
    print(f"\n{SEPARATOR}")
    print("【检查 3】3D 数据中 x_label / y_label 情况 (需要运行时投影)")
    print(SEPARATOR)

    # 因为 x_label/y_label 是运行时通过 _load_3d_with_projection 计算的，
    # 这里我们直接走一遍 Dataset 的 __getitem__ 来抽样检查。
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from dataset.open_vocab_dataset_v2 import (
            OpenVocabDatasetV2Config,
            OpenVocabScannetDatasetV2,
        )
    except ImportError:
        print("  ❌ 无法导入 dataset 模块, 跳过此检查")
        return None

    cfg = OpenVocabDatasetV2Config(
        data_config_path="config/data_scannet_3d.yaml",
        precomputed_dir=str(PRECOMPUTED_DIR),
        split=SPLIT,
        max_samples_ratio=MAX_SAMPLES_RATIO,
    )
    try:
        ds = OpenVocabScannetDatasetV2(cfg)
    except Exception as e:
        print(f"  ❌ 创建 Dataset 失败: {e}")
        return None

    n_samples = len(ds)
    n_check = min(200, n_samples)
    indices = sorted(random.sample(range(n_samples), n_check))

    stats = {
        "total": n_check,
        "xy_all_zero": 0,        # x_label 和 y_label 全部为 0
        "x_all_zero": 0,         # 只有 x_label 全 0
        "y_all_zero": 0,         # 只有 y_label 全 0
        "mask_k_zero": 0,        # mask 数量 K=0
        "mask_valid_all_false": 0,
        "x_max_values": [],
        "y_max_values": [],
        "mask_k_values": [],
        "point_counts": [],
        "errors": 0,
    }

    print(f"  抽样检查 {n_check}/{n_samples} 个样本...")

    for i, idx in enumerate(indices):
        if (i + 1) % 50 == 0:
            print(f"    进度: {i+1}/{n_check}")
        try:
            sample = ds[idx]
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"    ⚠️  样本 {idx} 加载失败: {e}")
            continue

        x_label = sample["x_label"]
        y_label = sample["y_label"]
        mask_valid = sample["mask_valid"]
        masks = sample["masks"]

        K = masks.shape[0]
        stats["mask_k_values"].append(K)
        stats["point_counts"].append(x_label.shape[0])

        if K == 0:
            stats["mask_k_zero"] += 1
        if not mask_valid.any():
            stats["mask_valid_all_false"] += 1

        x_is_zero = (x_label == 0).all().item()
        y_is_zero = (y_label == 0).all().item()
        if x_is_zero and y_is_zero:
            stats["xy_all_zero"] += 1
        if x_is_zero:
            stats["x_all_zero"] += 1
        if y_is_zero:
            stats["y_all_zero"] += 1

        if x_label.numel() > 0:
            stats["x_max_values"].append(x_label.max().item())
            stats["y_max_values"].append(y_label.max().item())

    print(f"\n  === 结果 ({n_check} 个样本) ===")
    print(f"  加载失败: {stats['errors']}")
    print(f"  K=0 (无 mask): {stats['mask_k_zero']} ({stats['mask_k_zero']/n_check*100:.1f}%)")
    print(f"  mask_valid 全 False: {stats['mask_valid_all_false']} ({stats['mask_valid_all_false']/n_check*100:.1f}%)")
    print(f"  x_label 全 0: {stats['x_all_zero']} ({stats['x_all_zero']/n_check*100:.1f}%)")
    print(f"  y_label 全 0: {stats['y_all_zero']} ({stats['y_all_zero']/n_check*100:.1f}%)")
    print(f"  x+y 都全 0: {stats['xy_all_zero']} ({stats['xy_all_zero']/n_check*100:.1f}%)")

    if stats["x_max_values"]:
        print(f"\n  x_label.max() 分布: min={min(stats['x_max_values'])}, "
              f"max={max(stats['x_max_values'])}, mean={np.mean(stats['x_max_values']):.1f}")
        print(f"  y_label.max() 分布: min={min(stats['y_max_values'])}, "
              f"max={max(stats['y_max_values'])}, mean={np.mean(stats['y_max_values']):.1f}")
    if stats["mask_k_values"]:
        print(f"  mask K 分布: min={min(stats['mask_k_values'])}, "
              f"max={max(stats['mask_k_values'])}, mean={np.mean(stats['mask_k_values']):.1f}")
    if stats["point_counts"]:
        print(f"  点数分布: min={min(stats['point_counts'])}, "
              f"max={max(stats['point_counts'])}, mean={np.mean(stats['point_counts']):.1f}")

    return stats


def simulate_criterion_skips():
    """检查4: 模拟完整的 DataLoader + Criterion 流程，统计各类 skip 频率。"""
    print(f"\n{SEPARATOR}")
    print("【检查 4】模拟 Criterion.loss_pt 中各类 skip 的频率")
    print(SEPARATOR)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from dataset.open_vocab_dataset_v2 import (
            OpenVocabDatasetV2Config,
            OpenVocabScannetDatasetV2,
            open_vocab_collate_v2,
        )
    except ImportError:
        print("  ❌ 无法导入 dataset 模块, 跳过此检查")
        return

    cfg = OpenVocabDatasetV2Config(
        data_config_path="config/data_scannet_3d.yaml",
        precomputed_dir=str(PRECOMPUTED_DIR),
        split=SPLIT,
        max_samples_ratio=MAX_SAMPLES_RATIO,
    )
    try:
        ds = OpenVocabScannetDatasetV2(cfg)
    except Exception as e:
        print(f"  ❌ 创建 Dataset 失败: {e}")
        return

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=open_vocab_collate_v2,
        num_workers=0,  # 诊断模式用单进程，避免报错信息丢失
    )

    # 统计计数器
    counters = {
        "total_batches": 0,
        "total_items": 0,
        # 模型 forward 中的 skip
        "model_no_points": 0,         # batch_indices == b 无点
        "model_no_valid_masks": 0,    # mask_valid[b].any() == False
        "model_empty_tokens": 0,      # mask_tokens.numel() == 0
        # Criterion 中的 skip
        "crit_no_outputs": 0,         # len(outputs[i]) == 0
        "crit_no_valid": 0,           # not valid.any()
        "crit_no_keep": 0,            # not keep.any() (pred threshold 过滤后无 mask)
        "crit_no_points": 0,          # x_idx.numel() == 0
        "crit_all_oob": 0,            # 缩放后所有点越界
        "crit_xy_all_zero": 0,        # x/y 全0
        "crit_empty_pred": 0,         # pred_logits.numel() == 0
        "crit_valid_loss": 0,         # 成功计算 loss 的 item
    }

    max_batches = min(NUM_BATCHES_TO_SIMULATE, len(loader))
    print(f"  模拟 {max_batches} 个 batch (batch_size={BATCH_SIZE})...")
    print(f"  (不运行模型 forward, 仅检查数据侧的 skip 条件)")

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        counters["total_batches"] += 1
        B = batch["pixel_pooled"].shape[0]
        mask_valid = batch["mask_valid"]        # (B, K_max)
        masks = batch["masks"]                  # (B, K_max, H, W)
        x_label = batch["x_label"]              # (N_total,)
        y_label = batch["y_label"]              # (N_total,)
        ori_coords = batch["ori_coords_3d"]     # (N_total, 4)
        batch_indices = ori_coords[:, 0].long()

        for b in range(B):
            counters["total_items"] += 1

            # --- 模型侧 skip 条件 ---
            point_mask = batch_indices == b
            if not point_mask.any():
                counters["model_no_points"] += 1
                counters["crit_no_outputs"] += 1  # forward 会直接 continue, outputs[b] 为空
                continue

            valid = mask_valid[b]
            if not valid.any():
                counters["model_no_valid_masks"] += 1
                counters["crit_no_outputs"] += 1
                continue

            # --- Criterion 侧 skip 条件 (假设模型正常输出了 logits) ---
            # 这里我们没有真正的 pred_mask_logits, 但可以检查 GT 侧能否构建监督
            mask_2d = masks[b][valid]       # (K_valid, H, W)
            K_valid = mask_2d.shape[0]

            x_idx = x_label[point_mask].float()
            y_idx = y_label[point_mask].float()

            if x_idx.numel() == 0:
                counters["crit_no_points"] += 1
                continue

            # 检查 x/y 全 0
            if (x_idx == 0).all() and (y_idx == 0).all():
                counters["crit_xy_all_zero"] += 1

            H, W = mask_2d.shape[1], mask_2d.shape[2]
            orig_W = max(640, x_idx.max().item() + 10)
            orig_H = max(480, y_idx.max().item() + 10)
            scale_x = W / orig_W
            scale_y = H / orig_H
            x_scaled = (x_idx * scale_x).long()
            y_scaled = (y_idx * scale_y).long()

            valid_pts = (x_scaled >= 0) & (x_scaled < W) & (y_scaled >= 0) & (y_scaled < H)
            num_valid = valid_pts.sum().item()

            if num_valid == 0:
                counters["crit_all_oob"] += 1
                continue

            # 构建 GT
            x_scaled = x_scaled[valid_pts]
            y_scaled = y_scaled[valid_pts]
            gt_3d = mask_2d[:, y_scaled, x_scaled]  # (K_valid, N_valid)
            gt_3d = (gt_3d > THRESHOLD).float()

            # 模拟 keep: 每个 mask 至少有 min_points_per_mask 个点被预测为正
            # 这里用 GT 本身代替 pred (因为没有模型),
            # 统计"GT 本身就没有足够正样本"的 mask
            keep = gt_3d.sum(dim=1) > MIN_POINTS_PER_MASK  # (K_valid,)
            if not keep.any():
                counters["crit_no_keep"] += 1
                continue

            counters["crit_valid_loss"] += 1

    # 汇总报告
    total = counters["total_items"]
    print(f"\n  === 统计结果 ({counters['total_batches']} 个 batch, {total} 个 item) ===")
    print(f"  ✅ 可正常计算 loss 的 item: {counters['crit_valid_loss']} "
          f"({counters['crit_valid_loss']/total*100:.1f}%)")
    print()
    print("  --- 模型 forward 中的 skip ---")
    print(f"  🔴 无点 (no points):        {counters['model_no_points']} ({counters['model_no_points']/total*100:.1f}%)")
    print(f"  🔴 无有效 mask (No valid):   {counters['model_no_valid_masks']} ({counters['model_no_valid_masks']/total*100:.1f}%)")
    print()
    print("  --- Criterion.loss_pt 中的 skip ---")
    print(f"  🔴 outputs 为空:             {counters['crit_no_outputs']} ({counters['crit_no_outputs']/total*100:.1f}%)")
    print(f"  🟡 x_label 无点:             {counters['crit_no_points']} ({counters['crit_no_points']/total*100:.1f}%)")
    print(f"  🟡 x/y 全 0 (无投影):        {counters['crit_xy_all_zero']} ({counters['crit_xy_all_zero']/total*100:.1f}%)")
    print(f"  🟡 缩放后全越界:             {counters['crit_all_oob']} ({counters['crit_all_oob']/total*100:.1f}%)")
    print(f"  🟡 GT 无足够正样本 (no keep): {counters['crit_no_keep']} ({counters['crit_no_keep']/total*100:.1f}%)")

    total_skipped = total - counters["crit_valid_loss"]
    print(f"\n  📊 总 skip 比例: {total_skipped}/{total} = {total_skipped/total*100:.1f}%")
    if total_skipped / total > 0.1:
        print(f"  ⚠️  超过 10% 的 batch item 被跳过, 这会显著影响训练稳定性和效率!")
    elif total_skipped / total > 0.01:
        print(f"  ⚠️  有少量 item 被跳过, 可能影响 loss 波动")
    else:
        print(f"  ✅  skip 比例很低, 问题可能主要在模型预测阈值或其他环节")


def main():
    print(SEPARATOR)
    print("   训练监督缺失诊断 —— 'No valid masks' 根因分析")
    print(SEPARATOR)
    print(f"  precomputed_dir: {PRECOMPUTED_DIR}")
    print(f"  data_root:       {DATA_ROOT}")
    print(f"  split:           {SPLIT}")
    print(f"  batch_size:      {BATCH_SIZE}")
    print(f"  max_samples_ratio: {MAX_SAMPLES_RATIO}")

    # 检查 1: NPZ mask 数量分布
    mask_counts, scene_stats = scan_npz_files()

    # 检查 2: 模拟 collate 后 mask_valid 频率
    simulate_collate_mask_valid(mask_counts)

    # 检查 3: x_label / y_label 情况
    check_3d_data_labels()

    # 检查 4: 模拟完整 criterion skip
    simulate_criterion_skips()

    print(f"\n{SEPARATOR}")
    print("  诊断完成!")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
