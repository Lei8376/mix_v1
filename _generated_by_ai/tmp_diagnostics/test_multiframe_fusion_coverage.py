#!/usr/bin/env python3
"""
测试多帧融合的覆盖率，验证 OpenScene 的做法。
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

try:
    from PIL import Image as PIL
except ImportError:
    print("错误: 需要安装 pillow")
    sys.exit(1)


def test_multiframe_fusion(
    scene_name: str = "scene0000_00",
    data_3d_root: str = "/home/featurize/data/scannet_3d",
    data_2d_root: str = "/home/featurize/data/scannet_2d",
):
    """测试多帧融合覆盖率。"""
    
    print("=" * 80)
    print(f"多帧融合覆盖率测试 - 场景: {scene_name}")
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
        locs = data[0]
    else:
        locs = data.get("locs", data.get("coords"))
    
    if isinstance(locs, torch.Tensor):
        locs = locs.numpy()
    
    N = len(locs)
    print(f"✅ 3D 点数: {N:,}")
    print()
    
    # 获取所有图像
    scene_2d_dir = data_2d_root / scene_name
    img_dirs = sorted(glob(str(scene_2d_dir / "color" / "*")), 
                     key=lambda x: int(os.path.basename(x)[:-4]))
    
    num_frames = len(img_dirs)
    print(f"✅ 总帧数: {num_frames}")
    print()
    
    # 初始化
    mapper = getMapping()
    covered = np.zeros(N, dtype=bool)  # 是否被至少一帧看到
    counter = np.zeros(N, dtype=int)   # 被多少帧看到
    x_label = np.zeros(N, dtype=np.int64)
    y_label = np.zeros(N, dtype=np.int64)
    
    print("开始多帧融合...")
    print("-" * 80)
    
    start_time = time.time()
    
    # 记录里程碑
    milestones = [10, 20, 50, 100, 150, 200, num_frames]
    milestone_results = []
    
    for i, img_dir in enumerate(img_dirs):
        try:
            pose_path = img_dir.replace("color", "pose").replace(".jpg", ".txt")
            depth_path = img_dir.replace("color", "depth").replace("jpg", "png")
            
            if not os.path.exists(pose_path) or not os.path.exists(depth_path):
                continue
            
            pose = np.loadtxt(pose_path)
            depth = np.array(PIL.open(depth_path)) / 1000.0
            
            # 计算投影
            mapping = mapper.compute_mapping(pose, locs, depth)
            mask = mapping[:, 2] == 1
            
            # 更新覆盖状态
            counter[mask] += 1
            
            # 对于新覆盖的点，保存投影坐标
            new_points = mask & ~covered
            if np.sum(new_points) > 0:
                # 正确顺序: 第0列是 y, 第1列是 x
                y_label[new_points] = mapping[new_points, 0].astype(np.int64)
                x_label[new_points] = mapping[new_points, 1].astype(np.int64)
                covered[new_points] = True
            
            # 里程碑报告
            frame_num = i + 1
            if frame_num in milestones:
                coverage = np.sum(covered) / N * 100
                elapsed = time.time() - start_time
                milestone_results.append({
                    "frames": frame_num,
                    "coverage": coverage,
                    "covered_points": np.sum(covered),
                    "time": elapsed,
                })
                print(f"帧 {frame_num:>3}: 覆盖率 {coverage:>5.1f}% ({np.sum(covered):>6,}/{N:,}), 耗时 {elapsed:.1f}s")
                
        except Exception as e:
            continue
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 80)
    print("最终结果:")
    print("=" * 80)
    
    final_coverage = np.sum(covered) / N * 100
    avg_count = np.mean(counter[covered])
    
    print(f"总覆盖率: {final_coverage:.1f}%")
    print(f"覆盖点数: {np.sum(covered):,} / {N:,}")
    print(f"未覆盖点数: {N - np.sum(covered):,}")
    print(f"平均被观察次数: {avg_count:.1f} 次/点")
    print(f"总耗时: {total_time:.1f}s")
    print()
    
    # 覆盖率分布
    print("被观察次数分布:")
    for threshold in [1, 5, 10, 20, 50]:
        count = np.sum(counter >= threshold)
        print(f"  >= {threshold:>2} 次: {count:>6,} 点 ({count/N*100:>5.1f}%)")
    print()
    
    # 投影坐标验证
    print("投影坐标验证:")
    valid_x = (x_label >= 0) & (x_label < 320)
    valid_y = (y_label >= 0) & (y_label < 240)
    valid_both = valid_x & valid_y & covered
    
    print(f"  X 范围内: {np.sum(valid_x & covered):,} / {np.sum(covered):,}")
    print(f"  Y 范围内: {np.sum(valid_y & covered):,} / {np.sum(covered):,}")
    print(f"  均有效: {np.sum(valid_both):,} / {np.sum(covered):,} ({np.sum(valid_both)/np.sum(covered)*100:.1f}%)")
    print()
    
    # 对比单帧 vs 多帧
    print("=" * 80)
    print("单帧 vs 多帧对比:")
    print("=" * 80)
    
    if len(milestone_results) > 0:
        single_frame = milestone_results[0] if milestone_results[0]["frames"] <= 10 else None
        
        print()
        print(f"{'帧数':>6} | {'覆盖率':>8} | {'覆盖点数':>10} | {'耗时':>6}")
        print("-" * 40)
        for m in milestone_results:
            print(f"{m['frames']:>6} | {m['coverage']:>7.1f}% | {m['covered_points']:>10,} | {m['time']:>5.1f}s")
        print()
    
    # 结论
    print("=" * 80)
    print("结论:")
    print("=" * 80)
    
    if final_coverage > 80:
        print(f"✅ 多帧融合效果很好!")
        print(f"   覆盖率从单帧 ~5% 提升到 {final_coverage:.1f}%")
        print()
        print("📝 建议:")
        print("   在 open_vocab_dataset_v2.py 中实现多帧投影累积")
        print("   或者使用预处理脚本生成融合特征")
    else:
        print(f"⚠️  覆盖率仍然较低: {final_coverage:.1f}%")
        print("   可能需要检查数据对齐问题")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试多帧融合覆盖率")
    parser.add_argument("--scene", type=str, default="scene0000_00")
    parser.add_argument("--data-3d-root", type=str, default="/home/featurize/data/scannet_3d")
    parser.add_argument("--data-2d-root", type=str, default="/home/featurize/data/scannet_2d")
    
    args = parser.parse_args()
    
    test_multiframe_fusion(
        scene_name=args.scene,
        data_3d_root=args.data_3d_root,
        data_2d_root=args.data_2d_root,
    )
