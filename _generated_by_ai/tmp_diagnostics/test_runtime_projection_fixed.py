#!/usr/bin/env python3
"""
测试修复后的运行时投影计算。

修复: compute_mapping 返回 [y, x, valid]，不是 [x, y, valid]
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from glob import glob
import time

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.mapping_util import getMapping

# 图像库导入
try:
    from PIL import Image as PIL
except ImportError:
    print("错误: 需要安装 pillow")
    sys.exit(1)


def test_projection_fixed(
    scene_name: str = "scene0000_00",
    data_3d_root: str = "/home/featurize/data/scannet_3d",
    data_2d_root: str = "/home/featurize/data/scannet_2d",
):
    """测试修复后的投影计算。"""
    
    print("=" * 80)
    print(f"测试修复后的投影计算 - 场景: {scene_name}")
    print("=" * 80)
    print()
    
    # 加载 3D 数据
    data_3d_root = Path(data_3d_root)
    data_2d_root = Path(data_2d_root)
    
    scene_file = data_3d_root / "train" / f"{scene_name}_vh_clean_2.pth"
    if not scene_file.exists():
        scene_file = data_3d_root / "train" / f"{scene_name}.pth"
    
    data = torch.load(scene_file, map_location="cpu", weights_only=False)
    
    if isinstance(data, (list, tuple)):
        locs, feats, labels = data[0], data[1], data[2]
    else:
        locs = data.get("locs", data.get("coords"))
        feats = data.get("feats", data.get("feat"))
        labels = data.get("labels")
    
    if isinstance(locs, torch.Tensor):
        locs = locs.numpy()
    
    N = len(locs)
    print(f"✅ 加载 3D 数据: {N:,} 个点")
    print()
    
    # 获取图像
    scene_2d_dir = data_2d_root / scene_name
    img_dirs = sorted(glob(str(scene_2d_dir / "color" / "*")), 
                     key=lambda x: int(os.path.basename(x)[:-4]))
    
    # 选择中间帧
    img_dir = img_dirs[len(img_dirs) // 2]
    frame_id = os.path.basename(img_dir)[:-4]
    
    print(f"测试帧: {frame_id}")
    print()
    
    # 读取数据
    img = np.array(PIL.open(img_dir))
    
    pose_path = img_dir.replace("color", "pose").replace(".jpg", ".txt")
    pose = np.loadtxt(pose_path)
    
    depth_path = img_dir.replace("color", "depth").replace("jpg", "png")
    depth = np.array(PIL.open(depth_path)) / 1000.0
    
    img_h, img_w = img.shape[:2]
    print(f"图像尺寸: {img_w} x {img_h}")
    print()
    
    # 计算投影
    mapper = getMapping()
    single_mapping = mapper.compute_mapping(pose, locs, depth)
    
    # 🔥 关键: compute_mapping 返回 [y, x, valid]
    print("投影结果分析:")
    print("-" * 80)
    
    # 方法 1: 错误的方式 (旧代码)
    print("\n❌ 错误方式 (把 y 当 x, 把 x 当 y):")
    x_wrong = single_mapping[:, 0][single_mapping[:, 0] != 0]
    y_wrong = single_mapping[:, 1][single_mapping[:, 1] != 0]
    
    x_in_bounds_wrong = (x_wrong >= 0) & (x_wrong < img_w)
    y_in_bounds_wrong = (y_wrong >= 0) & (y_wrong < img_h)
    in_bounds_wrong = x_in_bounds_wrong & y_in_bounds_wrong
    
    print(f"  x 范围: [{x_wrong.min():.0f}, {x_wrong.max():.0f}] (图像宽度: {img_w})")
    print(f"  y 范围: [{y_wrong.min():.0f}, {y_wrong.max():.0f}] (图像高度: {img_h})")
    print(f"  有效点数: {len(x_wrong):,}")
    print(f"  边界内: {np.sum(in_bounds_wrong):,} / {len(x_wrong):,} ({np.sum(in_bounds_wrong)/len(x_wrong)*100:.1f}%)")
    print(f"  覆盖率: {len(x_wrong) / N * 100:.1f}%")
    
    # 方法 2: 正确的方式
    print("\n✅ 正确方式 (第0列是 y, 第1列是 x):")
    zero_rows = np.all(single_mapping != 0, axis=1)
    valid_indices = np.where(zero_rows)[0]
    
    y_correct = single_mapping[valid_indices, 0].astype(np.int64)  # 第0列是 y
    x_correct = single_mapping[valid_indices, 1].astype(np.int64)  # 第1列是 x
    
    x_in_bounds_correct = (x_correct >= 0) & (x_correct < img_w)
    y_in_bounds_correct = (y_correct >= 0) & (y_correct < img_h)
    in_bounds_correct = x_in_bounds_correct & y_in_bounds_correct
    
    print(f"  x 范围: [{x_correct.min():.0f}, {x_correct.max():.0f}] (图像宽度: {img_w})")
    print(f"  y 范围: [{y_correct.min():.0f}, {y_correct.max():.0f}] (图像高度: {img_h})")
    print(f"  有效点数: {len(valid_indices):,}")
    print(f"  边界内: {np.sum(in_bounds_correct):,} / {len(valid_indices):,} ({np.sum(in_bounds_correct)/len(valid_indices)*100:.1f}%)")
    print(f"  覆盖率: {len(valid_indices) / N * 100:.1f}%")
    
    print()
    print("=" * 80)
    print("对比结果:")
    print("=" * 80)
    
    # 对比
    improvements = []
    
    # 边界内比例
    ratio_wrong = np.sum(in_bounds_wrong) / len(x_wrong) * 100
    ratio_correct = np.sum(in_bounds_correct) / len(valid_indices) * 100
    improvements.append(("边界内比例", ratio_wrong, ratio_correct, ratio_correct > 95))
    
    # 覆盖率 (注意: 单帧覆盖率低是正常的,不同帧看到的点不同)
    coverage_wrong = len(x_wrong) / N * 100
    coverage_correct = len(valid_indices) / N * 100
    improvements.append(("覆盖率 (单帧)", coverage_wrong, coverage_correct, True))  # 不作为失败条件
    
    # X 坐标是否在范围内
    x_valid_wrong = x_wrong.max() < img_w
    x_valid_correct = x_correct.max() < img_w
    improvements.append(("X 坐标在范围内", x_valid_wrong, x_valid_correct, x_valid_correct))
    
    # Y 坐标是否在范围内
    y_valid_wrong = y_wrong.max() < img_h
    y_valid_correct = y_correct.max() < img_h
    improvements.append(("Y 坐标在范围内", y_valid_wrong, y_valid_correct, y_valid_correct))
    
    print()
    for metric, wrong_val, correct_val, is_good in improvements:
        if isinstance(wrong_val, (bool, np.bool_)):
            status = "✅" if is_good else "❌"
            print(f"{status} {metric}:")
            print(f"     错误方式: {'是' if wrong_val else '否'}")
            print(f"     正确方式: {'是' if correct_val else '否'}")
        else:
            status = "✅" if is_good else "⚠️"
            improvement = float(correct_val) - float(wrong_val)
            print(f"{status} {metric}:")
            print(f"     错误方式: {wrong_val:.1f}%")
            print(f"     正确方式: {correct_val:.1f}%")
            print(f"     提升: {improvement:+.1f}%")
        print()
    
    # 总体判断
    all_good = all(is_good for _, _, _, is_good in improvements)
    
    print("=" * 80)
    if all_good:
        print("🎉 修复成功! 投影计算现在正确了!")
        print()
        print("下一步:")
        print("  1. 应用修复到 dataset/data_loader.py")
        print("  2. 应用修复到 dataset/open_vocab_dataset_v2.py")
        print("  3. 运行训练测试")
        return True
    else:
        print("⚠️  仍有问题，需要进一步检查")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试修复后的投影计算")
    parser.add_argument("--scene", type=str, default="scene0000_00")
    parser.add_argument("--data-3d-root", type=str, default="/home/featurize/data/scannet_3d")
    parser.add_argument("--data-2d-root", type=str, default="/home/featurize/data/scannet_2d")
    
    args = parser.parse_args()
    
    success = test_projection_fixed(
        scene_name=args.scene,
        data_3d_root=args.data_3d_root,
        data_2d_root=args.data_2d_root,
    )
    
    sys.exit(0 if success else 1)
