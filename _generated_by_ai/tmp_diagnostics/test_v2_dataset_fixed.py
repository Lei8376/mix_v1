"""
测试修复后的 OpenVocabScannetDatasetV2，验证：
1. 能够正常加载数据
2. x_label/y_label 不再是全 0
3. 投影坐标在合理范围内
4. 过滤后的点数在 400-65000 之间
"""

import sys
import os
sys.path.insert(0, '/home/featurize/work/mix')

import torch
import numpy as np
from dataset.open_vocab_dataset_v2 import (
    OpenVocabScannetDatasetV2,
    OpenVocabDatasetV2Config,
    open_vocab_collate_v2,
)

def test_dataset():
    """测试数据集加载"""
    
    print("=" * 80)
    print("测试修复后的 OpenVocabScannetDatasetV2 - OpenScene 思路")
    print("=" * 80)
    
    # 配置
    config = OpenVocabDatasetV2Config(
        data_config_path="config/data_scannet_3d.yaml",
        precomputed_dir="/home/featurize/data/pixel_pooled",
        split="train",
        max_samples=3,  # 只测试 3 个样本
    )
    
    try:
        dataset = OpenVocabScannetDatasetV2(config)
        print(f"\n✅ 数据集初始化成功，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"\n❌ 数据集初始化失败: {e}")
        return
    
    # 测试几个样本
    print("\n" + "=" * 80)
    print("测试样本加载")
    print("=" * 80)
    
    for i in range(min(3, len(dataset))):
        print(f"\n--- 样本 {i} ---")
        try:
            sample = dataset[i]
            
            # 检查关键字段
            coords_3d = sample["coords_3d"]
            x_label = sample["x_label"]
            y_label = sample["y_label"]
            masks = sample["masks"]
            
            N = coords_3d.shape[0]
            K, H, W = masks.shape
            
            print(f"3D 点数: {N}")
            print(f"2D Masks 形状: {K} masks, {H}x{W}")
            
            # 检查 x_label/y_label 是否有效
            x_nonzero = (x_label != 0).sum().item()
            y_nonzero = (y_label != 0).sum().item()
            
            print(f"x_label 非零点数: {x_nonzero}/{N} ({x_nonzero/N*100:.1f}%)")
            print(f"y_label 非零点数: {y_nonzero}/{N} ({y_nonzero/N*100:.1f}%)")
            
            # 检查坐标范围
            x_min, x_max = x_label.min().item(), x_label.max().item()
            y_min, y_max = y_label.min().item(), y_label.max().item()
            
            print(f"x_label 范围: [{x_min}, {x_max}], 图像宽度: {W}")
            print(f"y_label 范围: [{y_min}, {y_max}], 图像高度: {H}")
            
            # 验证坐标在图像范围内（排除 0）
            x_valid = x_label[x_label != 0]
            y_valid = y_label[y_label != 0]
            
            x_in_bounds = ((x_valid >= 0) & (x_valid < W)).sum().item()
            y_in_bounds = ((y_valid >= 0) & (y_valid < H)).sum().item()
            
            print(f"x_label 在范围内: {x_in_bounds}/{len(x_valid)} ({x_in_bounds/len(x_valid)*100:.1f}%)")
            print(f"y_label 在范围内: {y_in_bounds}/{len(y_valid)} ({y_in_bounds/len(y_valid)*100:.1f}%)")
            
            # 状态判断
            if x_nonzero == 0 or y_nonzero == 0:
                print("❌ 状态: 失败 - x_label/y_label 全为 0")
            elif x_in_bounds < len(x_valid) * 0.95 or y_in_bounds < len(y_valid) * 0.95:
                print("⚠️  状态: 警告 - 有超出范围的坐标")
            elif N < 400:
                print("⚠️  状态: 警告 - 可见点数过少")
            elif N > 65000:
                print("⚠️  状态: 警告 - 可见点数过多")
            else:
                print("✅ 状态: 成功 - 投影正常")
            
        except Exception as e:
            print(f"❌ 样本 {i} 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试 DataLoader
    print("\n" + "=" * 80)
    print("测试 DataLoader")
    print("=" * 80)
    
    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=open_vocab_collate_v2,
            num_workers=0,
        )
        
        batch = next(iter(dataloader))
        
        print(f"\n✅ DataLoader 测试成功")
        print(f"Batch 键: {list(batch.keys())}")
        print(f"coords_3d 形状: {batch['coords_3d'].shape}")
        print(f"x_label 形状: {batch['x_label'].shape}")
        print(f"y_label 形状: {batch['y_label'].shape}")
        print(f"masks 形状: {batch['masks'].shape}")
        
        # 检查 batch 中的投影
        x_label = batch["x_label"]
        y_label = batch["y_label"]
        
        x_nonzero = (x_label != 0).sum().item()
        y_nonzero = (y_label != 0).sum().item()
        total = x_label.shape[0]
        
        print(f"\nBatch 投影统计:")
        print(f"总点数: {total}")
        print(f"x_label 非零: {x_nonzero} ({x_nonzero/total*100:.1f}%)")
        print(f"y_label 非零: {y_nonzero} ({y_nonzero/total*100:.1f}%)")
        
        if x_nonzero > 0 and y_nonzero > 0:
            print("\n✅ 修复成功！x_label/y_label 不再全为 0")
        else:
            print("\n❌ 修复失败！x_label/y_label 仍然全为 0")
        
    except Exception as e:
        print(f"\n❌ DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset()
