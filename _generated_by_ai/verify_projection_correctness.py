"""
验证预计算投影的正确性

检查：
1. x_label/y_label 和 mask 是否真的是同一帧
2. 投影坐标是否落在正确的 mask 区域内
3. 可视化投影结果
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
# 修复中文乱码：使用系统自带的中文字体
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 修复负号显示

sys.path.insert(0, '/home/sunl/work/mix')

def verify_projection(scene_name, frame_idx, 
                     data_root_3d, data_root_2d, npz_dir, proj_dir):
    """验证一个帧的投影是否正确"""
    
    # 1. 加载 3D 数据
    pth_path = f"{data_root_3d}/train/{scene_name}.pth"
    if not os.path.exists(pth_path):
        pth_path = f"{data_root_3d}/train/{scene_name}_vh_clean_2.pth"
    
    data_3d = torch.load(pth_path)
    if isinstance(data_3d, tuple):
        locs = data_3d[0].numpy() if isinstance(data_3d[0], torch.Tensor) else data_3d[0]
    else:
        locs = data_3d['locs'].numpy() if isinstance(data_3d['locs'], torch.Tensor) else data_3d['locs']
    
    # 2. 加载预计算投影
    proj_path = f"{proj_dir}/{scene_name}/{frame_idx}_proj.npz"
    proj = np.load(proj_path)
    visible_mask = proj['visible_mask']
    y_label = proj['y_label']
    x_label = proj['x_label']
    
    # 3. 加载对应的 npz（2D 特征）
    npz_path = f"{npz_dir}/{scene_name}/{frame_idx}_odise.npz"
    npz = np.load(npz_path, allow_pickle=True)
    masks = npz['masks']  # (K, H, W)
    
    # 4. 加载 RGB 图像（用于可视化）
    img_path = f"{data_root_2d}/{scene_name}/color/{frame_idx}.jpg"
    img = np.array(Image.open(img_path))
    
    # 5. 验证投影
    print(f"场景: {scene_name}, 帧: {frame_idx}")
    print(f"3D 点数: {len(locs)}, 可见点: {visible_mask.sum()} ({visible_mask.sum()/len(locs)*100:.1f}%)")
    print(f"Masks: {masks.shape} (K={masks.shape[0]})")
    print(f"RGB: {img.shape}")
    print(f"y_label 范围: [{y_label.min()}, {y_label.max()}]")
    print(f"x_label 范围: [{x_label.min()}, {x_label.max()}]")
    
    # 6. 检查：投影点是否落在 mask 区域内
    # 对每个可见点，看它是否至少属于一个 mask
    H, W = masks.shape[1], masks.shape[2]
    print(f"\nMask 尺寸: H={H}, W={W}")
    
    # 缩小 RGB 到 mask 尺寸用于可视化
    img_resized = np.array(Image.fromarray(img).resize((W, H)))
    
    # 统计每个可见点属于多少个 mask
    num_masks_per_point = np.zeros(len(y_label), dtype=int)
    for k in range(masks.shape[0]):
        mask_k = masks[k]  # (H, W)
        in_mask = mask_k[y_label, x_label] > 0.5
        num_masks_per_point += in_mask.astype(int)
    
    print(f"\n每个可见点属于的 mask 数:")
    print(f"  无 mask: {np.sum(num_masks_per_point == 0)} ({np.sum(num_masks_per_point == 0)/len(y_label)*100:.1f}%)")
    print(f"  1 个: {np.sum(num_masks_per_point == 1)}")
    print(f"  2+ 个: {np.sum(num_masks_per_point >= 2)}")
    
    # 7. 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"原始RGB图 {img.shape}", fontsize=10)
    axes[0, 0].axis('off')
    
    # Mask 缩放后的图
    axes[0, 1].imshow(img_resized)
    axes[0, 1].scatter(x_label[::10], y_label[::10], c='red', s=1, alpha=0.5)
    axes[0, 1].set_title(f"投影点（每10个采样1个）", fontsize=10)
    axes[0, 1].set_xlim(0, W)
    axes[0, 1].set_ylim(H, 0)
    
    # 第一个 mask
    if masks.shape[0] > 0:
        axes[0, 2].imshow(masks[0], cmap='gray')
        in_mask_0 = masks[0][y_label, x_label] > 0.5
        axes[0, 2].scatter(x_label[in_mask_0][::5], y_label[in_mask_0][::5], 
                          c='red', s=2, alpha=0.8)
        axes[0, 2].set_title(f"Mask 0 + 投影点（在内部）", fontsize=10)
    
    # 所有 mask 叠加
    all_masks = masks.sum(axis=0) > 0
    axes[1, 0].imshow(all_masks, cmap='gray')
    axes[1, 0].scatter(x_label[::10], y_label[::10], c='red', s=1, alpha=0.5)
    axes[1, 0].set_title("所有Mask + 投影点", fontsize=10)
    
    # 每个点的 mask 数量分布
    point_mask_map = np.zeros((H, W), dtype=int)
    point_mask_map[y_label, x_label] = num_masks_per_point
    im = axes[1, 1].imshow(point_mask_map, cmap='viridis')
    axes[1, 1].set_title("每个点属于的Mask数量", fontsize=10)
    plt.colorbar(im, ax=axes[1, 1])
    
    # 统计图
    unique, counts = np.unique(num_masks_per_point, return_counts=True)
    axes[1, 2].bar(unique, counts)
    axes[1, 2].set_xlabel("Mask数量", fontsize=10)
    axes[1, 2].set_ylabel("点的数量", fontsize=10)
    axes[1, 2].set_title("Mask数量分布", fontsize=10)
    
    plt.tight_layout()
    output_dir = "/home/sunl/work/mix/test"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/verify_proj_{scene_name}_{frame_idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化保存到: {output_path}")
    plt.close()
    
    return {
        "total_visible": len(y_label),
        "no_mask": np.sum(num_masks_per_point == 0),
        "has_mask": np.sum(num_masks_per_point > 0),
    }


if __name__ == "__main__":
    # 测试几个帧
    data_root_3d = "/home/sunl/work/mix/data/scannet_3d"
    data_root_2d = "/home/sunl/work/mix/data/scannet_2d"
    npz_dir = "/home/sunl/work/mix/data/pixel_pooled"
    proj_dir = "/home/sunl/work/mix/data/scannet_projections"
    
    test_cases = [
        ("scene0002_01", "0"),
        ("scene0003_02", "100"),
        ("scene0004_00", "100"),
    ]
    
    print("=" * 80)
    print("验证预计算投影的正确性")
    print("=" * 80)
    
    for scene, frame in test_cases:
        print(f"\n{'=' * 80}")
        result = verify_projection(scene, frame, data_root_3d, data_root_2d, 
                                   npz_dir, proj_dir)
        print(f"结果: 可见点 {result['total_visible']}, 有mask {result['has_mask']} ({result['has_mask']/result['total_visible']*100:.1f}%)")
        print()
