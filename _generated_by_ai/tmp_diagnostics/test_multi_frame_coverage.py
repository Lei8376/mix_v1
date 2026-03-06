#!/usr/bin/env python3
"""
测试多帧选择对覆盖率的影响。

验证循环选帧是否能找到覆盖率更好的帧。
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from glob import glob

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


def test_multi_frame_coverage(
    scene_name: str = "scene0000_00",
    data_3d_root: str = "/home/featurize/data/scannet_3d",
    data_2d_root: str = "/home/featurize/data/scannet_2d",
    num_frames_to_test: int = 20,
):
    """测试多帧覆盖率。"""
    
    print("=" * 80)
    print(f"测试多帧覆盖率 - 场景: {scene_name}")
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
    print(f"✅ 加载 3D 数据: {N:,} 个点")
    print()
    
    # 获取所有图像
    scene_2d_dir = data_2d_root / scene_name
    img_dirs = sorted(glob(str(scene_2d_dir / "color" / "*")), 
                     key=lambda x: int(os.path.basename(x)[:-4]))
    
    print(f"总帧数: {len(img_dirs)}")
    print(f"测试帧数: {min(num_frames_to_test, len(img_dirs))}")
    print()
    
    # 初始化映射器
    mapper = getMapping()
    
    # 测试多帧
    print("逐帧测试覆盖率:")
    print("-" * 80)
    
    results = []
    
    # 均匀采样帧
    step = max(1, len(img_dirs) // num_frames_to_test)
    test_indices = list(range(0, len(img_dirs), step))[:num_frames_to_test]
    
    for i, idx in enumerate(test_indices):
        img_dir = img_dirs[idx]
        frame_id = os.path.basename(img_dir)[:-4]
        
        try:
            # 读取位姿和深度
            pose_path = img_dir.replace("color", "pose").replace(".jpg", ".txt")
            depth_path = img_dir.replace("color", "depth").replace("jpg", "png")
            
            if not os.path.exists(pose_path) or not os.path.exists(depth_path):
                continue
            
            pose = np.loadtxt(pose_path)
            depth = np.array(PIL.open(depth_path)) / 1000.0
            
            # 计算投影
            single_mapping = mapper.compute_mapping(pose, locs, depth)
            
            # 统计可见点数
            mask = single_mapping[:, 2]
            num_visible = np.sum(mask == 1)
            coverage = num_visible / N * 100
            
            # 提取有效投影
            zero_rows = np.all(single_mapping != 0, axis=1)
            num_valid = np.sum(zero_rows)
            
            results.append({
                "frame_id": frame_id,
                "idx": idx,
                "num_visible": num_visible,
                "num_valid": num_valid,
                "coverage": coverage,
            })
            
            status = "✅" if num_visible > 400 else "⚠️"
            print(f"{status} 帧 {frame_id:>5} (索引 {idx:>3}): {num_visible:>6,} 点 ({coverage:>5.1f}%)")
            
        except Exception as e:
            print(f"❌ 帧 {frame_id}: 错误 - {e}")
            continue
    
    if len(results) == 0:
        print("❌ 没有成功测试任何帧")
        return False
    
    print()
    print("=" * 80)
    print("统计分析:")
    print("=" * 80)
    
    coverages = [r["coverage"] for r in results]
    num_visibles = [r["num_visible"] for r in results]
    
    print(f"覆盖率统计:")
    print(f"  最小: {min(coverages):.1f}%")
    print(f"  最大: {max(coverages):.1f}%")
    print(f"  平均: {np.mean(coverages):.1f}%")
    print(f"  中位数: {np.median(coverages):.1f}%")
    print()
    
    print(f"可见点数统计:")
    print(f"  最小: {min(num_visibles):,}")
    print(f"  最大: {max(num_visibles):,}")
    print(f"  平均: {int(np.mean(num_visibles)):,}")
    print()
    
    # 找到最佳帧
    best_result = max(results, key=lambda r: r["num_visible"])
    print(f"最佳帧:")
    print(f"  帧 ID: {best_result['frame_id']}")
    print(f"  索引: {best_result['idx']}")
    print(f"  可见点: {best_result['num_visible']:,} / {N:,}")
    print(f"  覆盖率: {best_result['coverage']:.1f}%")
    print()
    
    # 统计满足条件的帧
    good_frames = [r for r in results if r["num_visible"] > 400 and r["num_visible"] < 65000]
    print(f"满足条件的帧 (400 < 点数 < 65000):")
    print(f"  数量: {len(good_frames)} / {len(results)} ({len(good_frames)/len(results)*100:.1f}%)")
    if len(good_frames) > 0:
        avg_coverage = np.mean([r["coverage"] for r in good_frames])
        print(f"  平均覆盖率: {avg_coverage:.1f}%")
    print()
    
    # 对比单帧 vs 最佳帧
    print("=" * 80)
    print("单帧 vs 循环选帧对比:")
    print("=" * 80)
    
    # 假设单帧选中间帧
    middle_idx = len(img_dirs) // 2
    middle_result = None
    for r in results:
        if abs(r["idx"] - middle_idx) < step:
            middle_result = r
            break
    
    if middle_result:
        print(f"单帧策略 (中间帧):")
        print(f"  覆盖率: {middle_result['coverage']:.1f}%")
        print(f"  可见点: {middle_result['num_visible']:,}")
        print()
        
        print(f"循环选帧策略 (最佳帧):")
        print(f"  覆盖率: {best_result['coverage']:.1f}%")
        print(f"  可见点: {best_result['num_visible']:,}")
        print()
        
        improvement = best_result['coverage'] - middle_result['coverage']
        print(f"提升: {improvement:+.1f}% ({improvement/middle_result['coverage']*100:+.1f}%)")
        print()
    
    # 结论
    print("=" * 80)
    print("结论:")
    print("=" * 80)
    
    if len(good_frames) > 0:
        print(f"✅ 找到 {len(good_frames)} 个覆盖率好的帧")
        print(f"✅ 最佳覆盖率: {best_result['coverage']:.1f}%")
        print(f"✅ 循环选帧策略有效!")
        print()
        print("建议:")
        print("  在 open_vocab_dataset_v2.py 中实现循环选帧")
        print("  训练时随机尝试多帧，直到找到覆盖率 > 400 的帧")
        return True
    else:
        print(f"⚠️  没有找到覆盖率好的帧")
        print(f"⚠️  最佳覆盖率只有: {best_result['coverage']:.1f}%")
        print()
        print("可能原因:")
        print("  1. 这个场景本身覆盖率就低")
        print("  2. 3D 点云和 2D 图像不对齐")
        print("  3. 相机参数不正确")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试多帧覆盖率")
    parser.add_argument("--scene", type=str, default="scene0000_00")
    parser.add_argument("--data-3d-root", type=str, default="/home/featurize/data/scannet_3d")
    parser.add_argument("--data-2d-root", type=str, default="/home/featurize/data/scannet_2d")
    parser.add_argument("--num-frames", type=int, default=20)
    
    args = parser.parse_args()
    
    success = test_multi_frame_coverage(
        scene_name=args.scene,
        data_3d_root=args.data_3d_root,
        data_2d_root=args.data_2d_root,
        num_frames_to_test=args.num_frames,
    )
    
    sys.exit(0 if success else 1)
