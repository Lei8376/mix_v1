#!/usr/bin/env python
"""
对比可视化：在同一张图上显示不同配置的分割结果
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 设置字体以支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def generate_colors(n, seed=42):
    """生成 n 个区分度高的颜色"""
    np.random.seed(seed)
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.6 + (i % 3) * 0.15
        value = 0.7 + (i % 2) * 0.2
        
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
            
        colors.append(((r + m), (g + m), (b + m), 0.6))
    return colors


def visualize_single_config(ax, image, npz_path, config_name, min_score=0.15):
    """在指定的 axes 上可视化单个配置的结果"""
    # 加载数据
    data = np.load(npz_path, allow_pickle=True)
    masks = data['masks']
    info = data['info']
    
    # 显示原图
    ax.imshow(image)
    ax.axis('off')
    
    # 生成颜色
    colors = generate_colors(len(masks))
    
    # 统计信息
    total_masks = len(masks)
    shown_masks = 0
    categories = set()
    
    # 绘制 masks
    for i, (mask, mask_info) in enumerate(zip(masks, info)):
        score = mask_info['score']
        if score < min_score:
            continue
        
        shown_masks += 1
        category = mask_info['category_name']
        categories.add(category)
        
        # 创建彩色 mask
        color = colors[i]
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0] = color
        
        # 叠加 mask
        ax.imshow(colored_mask, alpha=0.5)
        
        # 找到 mask 中心显示标签
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) > 0:
            center_y = int(np.mean(mask_indices[0]))
            center_x = int(np.mean(mask_indices[1]))
            
            label_text = f"{category}\n{score:.2f}"
            ax.text(center_x, center_y, label_text,
                   fontsize=6, color='white', weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', 
                           facecolor=color[:3], 
                           edgecolor='white',
                           alpha=0.8))
    
    # 设置标题
    title = f"{config_name}\n总masks: {total_masks} | 显示: {shown_masks} | 类别数: {len(categories)}"
    ax.set_title(title, fontsize=10, weight='bold')
    
    return total_masks, shown_masks, len(categories)


def visualize_comparison(image_path, npz_paths, config_names, output_path, min_score=0.15):
    """
    对比可视化：在同一张图上显示多个配置的结果
    
    Args:
        image_path: 原始图片路径
        npz_paths: 多个 npz 文件路径列表
        config_names: 配置名称列表
        output_path: 保存路径
        min_score: 最小分数阈值
    """
    # 加载图片
    image = np.array(Image.open(image_path))
    
    # 创建画布：1行N列
    n_configs = len(npz_paths)
    fig, axes = plt.subplots(1, n_configs, figsize=(8*n_configs, 8), dpi=100)
    
    if n_configs == 1:
        axes = [axes]
    
    # 统计信息
    stats = []
    
    # 为每个配置可视化
    for ax, npz_path, config_name in zip(axes, npz_paths, config_names):
        if not Path(npz_path).exists():
            ax.text(0.5, 0.5, f"文件不存在:\n{npz_path}", 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            stats.append((0, 0, 0))
            continue
        
        stat = visualize_single_config(ax, image, npz_path, config_name, min_score)
        stats.append(stat)
    
    # 总标题
    img_name = Path(image_path).name
    fig.suptitle(f"配置对比: {img_name}", fontsize=14, weight='bold', y=0.98)
    
    # 保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()
    
    return stats


def compare_batch(image_dir, npz_dirs, config_names, output_dir, num_samples=10, min_score=0.15):
    """
    批量对比可视化
    
    Args:
        image_dir: 图片目录
        npz_dirs: 多个 npz 目录列表
        config_names: 配置名称列表
        output_dir: 输出目录
        num_samples: 样本数量
        min_score: 最小分数阈值
    """
    image_dir = Path(image_dir)
    npz_dirs = [Path(d) for d in npz_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取第一个配置的所有 npz 文件（作为基准）
    base_npz_files = sorted(npz_dirs[0].glob("*_odise.npz"))
    
    if len(base_npz_files) == 0:
        print(f"错误: {npz_dirs[0]} 中没有找到 npz 文件")
        return
    
    # 随机采样
    if num_samples and len(base_npz_files) > num_samples:
        np.random.seed(42)
        indices = np.random.choice(len(base_npz_files), num_samples, replace=False)
        indices = sorted(indices)
        base_npz_files = [base_npz_files[i] for i in indices]
    
    print(f"对比可视化 {len(base_npz_files)} 张图片...")
    print(f"配置数量: {len(config_names)}")
    for i, name in enumerate(config_names):
        print(f"  配置{i+1}: {name}")
    print(f"最小分数阈值: {min_score}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 统计所有配置的总体信息
    all_stats = {name: {'total_masks': 0, 'shown_masks': 0, 'categories': set()} 
                 for name in config_names}
    
    # 处理每张图片
    for base_npz_path in tqdm(base_npz_files, desc="处理进度"):
        img_name = base_npz_path.stem.replace("_odise", "")
        
        # 找到对应的图片
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = image_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            print(f"警告: 找不到图片 {img_name}")
            continue
        
        # 找到所有配置对应的 npz 文件
        npz_paths = []
        for npz_dir in npz_dirs:
            npz_path = npz_dir / base_npz_path.name
            npz_paths.append(npz_path)
        
        # 输出文件名
        output_path = output_dir / f"{img_name}_compare.png"
        
        try:
            stats = visualize_comparison(img_path, npz_paths, config_names, 
                                        output_path, min_score)
            
            # 累积统计
            for i, name in enumerate(config_names):
                all_stats[name]['total_masks'] += stats[i][0]
                all_stats[name]['shown_masks'] += stats[i][1]
        except Exception as e:
            print(f"错误: 处理 {img_name} 时出错: {e}")
    
    # 打印总体统计
    print("\n" + "="*70)
    print("总体统计")
    print("="*70)
    for name in config_names:
        stat = all_stats[name]
        print(f"\n{name}:")
        print(f"  总 masks: {stat['total_masks']}")
        print(f"  显示 masks: {stat['shown_masks']}")
        avg_masks = stat['total_masks'] / len(base_npz_files) if base_npz_files else 0
        print(f"  平均每张图 masks: {avg_masks:.1f}")
    print("="*70)
    
    print(f"\n✅ 完成！结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="对比可视化不同配置的 ODISE 结果")
    parser.add_argument("--image-dir", type=str, required=True,
                       help="原始图片目录")
    parser.add_argument("--npz-dirs", type=str, nargs='+', required=True,
                       help="多个 npz 文件目录（空格分隔）")
    parser.add_argument("--config-names", type=str, nargs='+', required=True,
                       help="配置名称（空格分隔，与 npz-dirs 对应）")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="样本数量（默认10）")
    parser.add_argument("--min-score", type=float, default=0.15,
                       help="最小分数阈值（默认0.15）")
    
    args = parser.parse_args()
    
    if len(args.npz_dirs) != len(args.config_names):
        print("错误: npz-dirs 和 config-names 的数量必须相同")
        return
    
    compare_batch(
        image_dir=args.image_dir,
        npz_dirs=args.npz_dirs,
        config_names=args.config_names,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        min_score=args.min_score
    )


if __name__ == "__main__":
    main()
