#!/usr/bin/env python
"""
可视化 ODISE 分割结果
在原图上叠加 masks，显示类别标签和分数
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def generate_colors(n):
    """生成 n 个区分度高的颜色"""
    np.random.seed(42)
    colors = []
    for i in range(n):
        # 使用 HSV 色彩空间生成均匀分布的颜色
        hue = (i * 0.618033988749895) % 1.0  # 黄金分割角度
        saturation = 0.6 + (i % 3) * 0.15
        value = 0.7 + (i % 2) * 0.2
        
        # HSV to RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        colors.append(((r + m), (g + m), (b + m), 0.6))  # alpha=0.6
    return colors


def visualize_single_result(image_path, npz_path, output_path, show_score=True, min_score=0.0):
    """
    可视化单张图片的分割结果
    
    Args:
        image_path: 原始图片路径
        npz_path: ODISE 输出的 npz 文件路径
        output_path: 保存可视化结果的路径
        show_score: 是否显示分数
        min_score: 最小分数阈值（低于此分数的 mask 不显示）
    """
    # 加载图片
    img = np.array(Image.open(image_path))
    h, w = img.shape[:2]
    
    # 加载 ODISE 结果
    data = np.load(npz_path, allow_pickle=True)
    masks = data['masks']  # [N, H, W]
    info = data['info']    # [N] array of dicts
    
    # 创建画布
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=100)
    ax.imshow(img)
    ax.axis('off')
    
    # 生成颜色
    colors = generate_colors(len(masks))
    
    # 存储图例信息
    legend_patches = []
    
    # 绘制每个 mask
    for i, (mask, mask_info) in enumerate(zip(masks, info)):
        score = mask_info['score']
        
        # 过滤低分 mask
        if score < min_score:
            continue
            
        category = mask_info['category_name']
        is_thing = mask_info['is_thing']
        area = mask_info['area']
        
        # 创建彩色 mask
        color = colors[i]
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0] = color
        
        # 叠加 mask
        ax.imshow(colored_mask, alpha=0.5)
        
        # 找到 mask 的中心位置用于显示标签
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) > 0:
            center_y = int(np.mean(mask_indices[0]))
            center_x = int(np.mean(mask_indices[1]))
            
            # 构建标签文本
            if show_score:
                label_text = f"{category}\n{score:.2f}"
            else:
                label_text = category
            
            # 显示标签（带背景框）
            ax.text(center_x, center_y, label_text,
                   fontsize=8, color='white', weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=color[:3], 
                           edgecolor='white',
                           alpha=0.8))
        
        # 添加到图例
        thing_marker = "T" if is_thing else "S"
        legend_label = f"[{thing_marker}] {category} ({score:.2f}, {area}px)"
        legend_patches.append(mpatches.Patch(color=color[:3], label=legend_label))
    
    # 添加图例
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper left', 
                 bbox_to_anchor=(1.02, 1), fontsize=8,
                 title="[T]=Thing, [S]=Stuff")
    
    # 设置标题
    img_name = Path(image_path).name
    ax.set_title(f"ODISE Segmentation: {img_name}\n"
                f"Total masks: {len(masks)}, Shown: {len(legend_patches)}",
                fontsize=12, weight='bold')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()


def visualize_batch(image_dir, npz_dir, output_dir, num_samples=10, min_score=0.15, random_sample=False):
    """
    批量可视化结果
    
    Args:
        image_dir: 图片目录
        npz_dir: npz 文件目录
        output_dir: 输出目录
        num_samples: 可视化的样本数量（None 表示全部）
        min_score: 最小分数阈值
        random_sample: 是否随机采样（否则均匀采样）
    """
    image_dir = Path(image_dir)
    npz_dir = Path(npz_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 npz 文件
    npz_files = sorted(npz_dir.glob("*_odise.npz"))
    
    if num_samples is not None and len(npz_files) > num_samples:
        if random_sample:
            # 随机采样
            np.random.seed(42)
            indices = np.random.choice(len(npz_files), num_samples, replace=False)
            indices = sorted(indices)
            npz_files = [npz_files[i] for i in indices]
        else:
            # 均匀采样
            indices = np.linspace(0, len(npz_files) - 1, num_samples, dtype=int)
            npz_files = [npz_files[i] for i in indices]
    
    print(f"开始可视化 {len(npz_files)} 张图片...")
    print(f"采样方式: {'随机采样' if random_sample else '均匀采样'}")
    print(f"最小分数阈值: {min_score}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 打印第一个文件的 shape 信息
    if len(npz_files) > 0:
        print("=" * 70)
        print("数据格式信息 (第一个文件)")
        print("=" * 70)
        first_data = np.load(npz_files[0], allow_pickle=True)
        for key in first_data.keys():
            arr = first_data[key]
            if hasattr(arr, 'shape'):
                print(f"  {key:25s} shape: {str(arr.shape):20s} dtype: {arr.dtype}")
            else:
                print(f"  {key:25s} type: {type(arr).__name__}")
        print("=" * 70)
        print()
    
    for npz_path in tqdm(npz_files, desc="可视化进度"):
        # 获取对应的图片文件
        # npz 文件名格式: 0_odise.npz -> 原图: 0.jpg
        img_name = npz_path.stem.replace("_odise", "")
        
        # 尝试多种图片扩展名
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = image_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            print(f"警告: 找不到对应的图片 {img_name}")
            continue
        
        # 输出文件名
        output_path = output_dir / f"{img_name}_vis.png"
        
        try:
            visualize_single_result(img_path, npz_path, output_path, 
                                  show_score=True, min_score=min_score)
        except Exception as e:
            print(f"错误: 处理 {img_name} 时出错: {e}")
    
    print(f"\n✅ 完成！结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="可视化 ODISE 分割结果")
    parser.add_argument("--image-dir", type=str, required=True,
                       help="原始图片目录")
    parser.add_argument("--npz-dir", type=str, required=True,
                       help="ODISE 输出的 npz 文件目录")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="可视化结果输出目录")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="可视化的样本数量（默认10，None表示全部）")
    parser.add_argument("--min-score", type=float, default=0.15,
                       help="最小分数阈值（默认0.15）")
    parser.add_argument("--all", action="store_true",
                       help="可视化所有图片")
    parser.add_argument("--random", action="store_true",
                       help="随机采样（默认均匀采样）")
    
    args = parser.parse_args()
    
    num_samples = None if args.all else args.num_samples
    
    visualize_batch(
        image_dir=args.image_dir,
        npz_dir=args.npz_dir,
        output_dir=args.output_dir,
        num_samples=num_samples,
        min_score=args.min_score,
        random_sample=args.random
    )


if __name__ == "__main__":
    main()
