#!/usr/bin/env python
"""
测试脚本：查看 ODISE 输出结果
修改下面的路径即可使用
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================
# 修改这两个路径
# ============================================
NPZ_FILE = "/home/sunl/work/odise_features/scene0149_00/180_odise.npz"
IMAGE_FILE = "/home/sunl/work/scannet_2d/scene0149_00/color/180.jpg"
OUTPUT_IMAGE = "./test_visualization.png"  # 输出可视化图片
MIN_SCORE = 0.1  # 最低分数阈值
# ============================================1

def generate_colors(n):
    """生成 n 个不同的颜色"""
    np.random.seed(42)
    colors = np.random.rand(n, 3)
    return colors

def main():
    # 加载数据
    print(f"加载: {NPZ_FILE}")
    data = np.load(NPZ_FILE, allow_pickle=True)
    
    # 打印基本信息
    print("\n" + "="*60)
    print("文件内容:")
    print("="*60)
    for key in data.files:
        val = data[key]
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")
    
    # 获取数据
    masks = data['masks']
    embeddings = data['mask_embeddings']
    info = data['info']
    num_masks = int(data['num_masks'])
    
    print("\n" + "="*60)
    print(f"检测到 {num_masks} 个物体")
    print("="*60)
    
    # 打印每个物体的信息
    if num_masks > 0:
        sorted_info = sorted(enumerate(info), key=lambda x: x[1]['score'], reverse=True)
        print(f"\n{'序号':<5} {'类别':<30} {'分数':<10} {'面积':<10} {'is_thing'}")
        print("-"*70)
        for idx, item in sorted_info:
            print(f"{idx+1:<5} {item['category_name']:<30} {item['score']:<10.3f} {item['area']:<10} {item['is_thing']}")
    
    # 加载原图
    print(f"\n加载原图: {IMAGE_FILE}")
    img = np.array(Image.open(IMAGE_FILE).convert("RGB"))
    H, W = img.shape[:2]
    print(f"图片尺寸: {W} x {H}")
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 原图
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 2. 分割蒙版
    if num_masks > 0:
        colors = generate_colors(num_masks)
        mask_overlay = np.zeros((H, W, 3), dtype=np.float32)
        
        for i in range(num_masks):
            if info[i]['score'] >= MIN_SCORE:
                mask = masks[i]
                mask_overlay[mask] = colors[i]
        
        # 混合原图和蒙版
        blended = img.astype(np.float32) / 255.0 * 0.5 + mask_overlay * 0.5
        blended = np.clip(blended, 0, 1)
        axes[1].imshow(blended)
    else:
        axes[1].imshow(img)
    axes[1].set_title(f"Segmentation Masks (score >= {MIN_SCORE})")
    axes[1].axis('off')
    
    # 3. 带标签的结果
    axes[2].imshow(img)
    if num_masks > 0:
        # 显示高分物体的标签
        legend_patches = []
        for i, item in enumerate(info):
            if item['score'] >= MIN_SCORE:
                mask = masks[i]
                # 找到 mask 的中心
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    cy, cx = int(np.mean(ys)), int(np.mean(xs))
                    label = f"{item['category_name']}: {item['score']:.2f}"
                    axes[2].text(cx, cy, label, fontsize=8, color='white',
                               bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))
                    legend_patches.append(mpatches.Patch(color=colors[i], label=f"{item['category_name']} ({item['score']:.2f})"))
        
        if legend_patches:
            axes[2].legend(handles=legend_patches[:10], loc='upper right', fontsize=7)
    axes[2].set_title("Labels & Scores")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches='tight')
    print(f"\n可视化保存到: {OUTPUT_IMAGE}")
    plt.close()
    
    # 打印 embedding 信息
    print("\n" + "="*60)
    print("Embedding 信息:")
    print("="*60)
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    if num_masks > 0:
        print(f"  每个 embedding 维度: {embeddings.shape[1]}")
        print(f"  示例 (第一个 embedding 前10维): {embeddings[0][:10]}")

if __name__ == "__main__":
    main()
