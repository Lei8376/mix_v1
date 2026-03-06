#!/usr/bin/env python3
"""
测试运行时投影计算是否正确。

这个脚本会:
1. 加载一个场景的 3D 数据
2. 运行时计算投影 (模拟训练时的行为)
3. 验证投影结果的正确性
4. 可视化投影效果(可选)
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from glob import glob
import time

# 添加项目根目录到 Python 路径
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.mapping_util import getMapping

# 尝试导入图像库
imageio = None
cv2 = None
PIL = None

try:
    import imageio.v2 as imageio
except ImportError:
    try:
        import imageio
    except ImportError:
        pass

if imageio is None:
    try:
        import cv2
    except ImportError:
        pass

if imageio is None and cv2 is None:
    try:
        from PIL import Image as PIL
    except ImportError:
        pass

if imageio is None and cv2 is None and PIL is None:
    print("错误: 需要安装 imageio, opencv-python 或 pillow 中的至少一个")
    sys.exit(1)


def test_runtime_projection(
    scene_name: str = "scene0000_00",
    data_3d_root: str = "/home/featurize/data/scannet_3d",
    data_2d_root: str = "/home/featurize/data/scannet_2d",
    num_frames_to_test: int = 3,
):
    """测试运行时投影计算。"""
    
    print("=" * 80)
    print(f"测试场景: {scene_name}")
    print("=" * 80)
    print()
    
    # 路径
    data_3d_root = Path(data_3d_root)
    data_2d_root = Path(data_2d_root)
    
    # 1. 加载 3D 数据
    print("步骤 1: 加载 3D 数据")
    print("-" * 80)
    
    scene_file = data_3d_root / "train" / f"{scene_name}_vh_clean_2.pth"
    if not scene_file.exists():
        scene_file = data_3d_root / "train" / f"{scene_name}.pth"
    
    if not scene_file.exists():
        print(f"❌ 场景文件不存在: {scene_file}")
        return False
    
    start_time = time.time()
    data = torch.load(scene_file, map_location="cpu", weights_only=False)
    
    if isinstance(data, (list, tuple)):
        locs, feats, labels = data[0], data[1], data[2]
    else:
        locs = data.get("locs", data.get("coords"))
        feats = data.get("feats", data.get("feat"))
        labels = data.get("labels")
    
    if isinstance(locs, torch.Tensor):
        locs = locs.numpy()
    if isinstance(feats, torch.Tensor):
        feats = feats.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    load_time = time.time() - start_time
    
    N = len(locs)
    print(f"✅ 成功加载 3D 数据")
    print(f"   点数: {N:,}")
    print(f"   加载时间: {load_time:.3f}s")
    print(f"   坐标范围: x[{locs[:, 0].min():.2f}, {locs[:, 0].max():.2f}], "
          f"y[{locs[:, 1].min():.2f}, {locs[:, 1].max():.2f}], "
          f"z[{locs[:, 2].min():.2f}, {locs[:, 2].max():.2f}]")
    print()
    
    # 2. 检查 2D 数据
    print("步骤 2: 检查 2D 数据")
    print("-" * 80)
    
    scene_2d_dir = data_2d_root / scene_name
    if not scene_2d_dir.exists():
        print(f"❌ 2D 数据不存在: {scene_2d_dir}")
        return False
    
    img_dirs = sorted(glob(str(scene_2d_dir / "color" / "*")), 
                     key=lambda x: int(os.path.basename(x)[:-4]))
    
    if len(img_dirs) == 0:
        print(f"❌ 没有找到图像")
        return False
    
    print(f"✅ 找到 {len(img_dirs)} 帧图像")
    print()
    
    # 3. 初始化投影映射器
    print("步骤 3: 初始化投影映射器")
    print("-" * 80)
    
    start_time = time.time()
    mapper = getMapping()
    init_time = time.time() - start_time
    
    print(f"✅ 映射器初始化完成")
    print(f"   初始化时间: {init_time:.3f}s")
    print()
    
    # 4. 测试多帧投影
    print(f"步骤 4: 测试 {num_frames_to_test} 帧投影计算")
    print("-" * 80)
    
    # 选择要测试的帧(开头、中间、结尾)
    test_indices = [
        0,  # 第一帧
        len(img_dirs) // 2,  # 中间帧
        len(img_dirs) - 1,  # 最后一帧
    ][:num_frames_to_test]
    
    results = []
    
    for idx in test_indices:
        img_dir = img_dirs[idx]
        frame_id = os.path.basename(img_dir)[:-4]
        
        print(f"\n测试帧 {frame_id} (索引 {idx}/{len(img_dirs)-1}):")
        
        try:
            # 读取图像
            start_time = time.time()
            if imageio is not None:
                img = imageio.imread(img_dir)
            elif cv2 is not None:
                img = cv2.imread(img_dir)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.array(PIL.open(img_dir))
            img_time = time.time() - start_time
            
            # 读取位姿
            start_time = time.time()
            pose_path = img_dir.replace("color", "pose").replace(".jpg", ".txt")
            if not os.path.exists(pose_path):
                print(f"  ❌ 位姿文件不存在: {pose_path}")
                continue
            pose = np.loadtxt(pose_path)
            pose_time = time.time() - start_time
            
            # 读取深度图
            start_time = time.time()
            depth_path = img_dir.replace("color", "depth").replace("jpg", "png")
            if not os.path.exists(depth_path):
                print(f"  ❌ 深度图不存在: {depth_path}")
                continue
            if imageio is not None:
                depth = imageio.imread(depth_path) / 1000.0
            elif cv2 is not None:
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0
            else:
                depth = np.array(PIL.open(depth_path)) / 1000.0
            depth_time = time.time() - start_time
            
            # 计算投影
            start_time = time.time()
            single_mapping = mapper.compute_mapping(pose, locs, depth)
            mapping_time = time.time() - start_time
            
            # 分析投影结果
            mask = single_mapping[:, 2]  # 可见性标记
            valid_points = np.sum(mask == 1)
            
            # 提取有效投影坐标
            zero_rows = np.all(single_mapping != 0, axis=1)
            valid_mapping = single_mapping[zero_rows]
            
            if len(valid_mapping) > 0:
                x_coords = valid_mapping[:, 0]
                y_coords = valid_mapping[:, 1]
                
                # 检查投影坐标范围
                img_h, img_w = img.shape[:2]
                depth_h, depth_w = depth.shape
                
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                # 检查是否在图像范围内
                x_in_bounds = (x_coords >= 0) & (x_coords < img_w)
                y_in_bounds = (y_coords >= 0) & (y_coords < img_h)
                in_bounds = x_in_bounds & y_in_bounds
                in_bounds_count = np.sum(in_bounds)
                
                result = {
                    "frame_id": frame_id,
                    "valid_points": valid_points,
                    "valid_mapping": len(valid_mapping),
                    "coverage": valid_points / N * 100,
                    "x_range": (x_min, x_max),
                    "y_range": (y_min, y_max),
                    "img_size": (img_w, img_h),
                    "depth_size": (depth_w, depth_h),
                    "in_bounds": in_bounds_count,
                    "in_bounds_ratio": in_bounds_count / len(valid_mapping) * 100,
                    "times": {
                        "img": img_time,
                        "pose": pose_time,
                        "depth": depth_time,
                        "mapping": mapping_time,
                        "total": img_time + pose_time + depth_time + mapping_time,
                    }
                }
                results.append(result)
                
                print(f"  ✅ 投影计算成功")
                print(f"     可见点数: {valid_points:,} / {N:,} ({result['coverage']:.1f}%)")
                print(f"     有效映射: {len(valid_mapping):,}")
                print(f"     投影范围: x[{x_min:.0f}, {x_max:.0f}], y[{y_min:.0f}, {y_max:.0f}]")
                print(f"     图像尺寸: {img_w} x {img_h}")
                print(f"     深度尺寸: {depth_w} x {depth_h}")
                print(f"     边界内点: {in_bounds_count:,} / {len(valid_mapping):,} ({result['in_bounds_ratio']:.1f}%)")
                print(f"     耗时: 读图{img_time:.3f}s + 位姿{pose_time:.3f}s + 深度{depth_time:.3f}s + 投影{mapping_time:.3f}s = {result['times']['total']:.3f}s")
                
                # 检查潜在问题
                if result['in_bounds_ratio'] < 95:
                    print(f"  ⚠️  警告: {100-result['in_bounds_ratio']:.1f}% 的投影点超出图像边界")
                
                if result['coverage'] < 30:
                    print(f"  ⚠️  警告: 覆盖率较低 ({result['coverage']:.1f}%)")
                
            else:
                print(f"  ❌ 没有有效的投影点")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if len(results) == 0:
        print("❌ 所有帧都失败了")
        return False
    
    # 统计
    avg_coverage = np.mean([r['coverage'] for r in results])
    avg_in_bounds = np.mean([r['in_bounds_ratio'] for r in results])
    avg_time = np.mean([r['times']['total'] for r in results])
    
    print(f"✅ 成功测试 {len(results)} 帧")
    print(f"   平均覆盖率: {avg_coverage:.1f}%")
    print(f"   平均边界内比例: {avg_in_bounds:.1f}%")
    print(f"   平均耗时: {avg_time:.3f}s/帧")
    print()
    
    # 估算训练时的开销
    print("训练时开销估算:")
    print("-" * 80)
    
    # 假设每个 epoch 遍历所有场景,每个场景随机选一帧
    num_scenes = 977  # ScanNet train split
    time_per_epoch = num_scenes * avg_time
    
    print(f"假设条件:")
    print(f"  - 训练集场景数: {num_scenes}")
    print(f"  - 每个场景选 1 帧")
    print(f"  - 每帧投影耗时: {avg_time:.3f}s")
    print()
    print(f"估算结果:")
    print(f"  - 每个 epoch 投影总耗时: {time_per_epoch:.1f}s ({time_per_epoch/60:.1f} 分钟)")
    print(f"  - 10 个 epoch 总耗时: {time_per_epoch*10/60:.1f} 分钟 ({time_per_epoch*10/3600:.1f} 小时)")
    print()
    
    # 判断是否可接受
    if avg_time < 0.5:
        print("✅ 投影计算速度很快,适合运行时计算")
    elif avg_time < 1.0:
        print("⚠️  投影计算速度一般,可以接受但会增加训练时间")
    else:
        print("❌ 投影计算速度较慢,建议预先生成")
    
    print()
    
    # 验证投影正确性
    print("=" * 80)
    print("投影正确性验证")
    print("=" * 80)
    
    if len(results) > 0:
        best_result = max(results, key=lambda r: r['valid_points'])
        
        print(f"选择最佳帧 {best_result['frame_id']} 进行详细验证:")
        print(f"  - 有效点数: {best_result['valid_points']:,}")
        print(f"  - 覆盖率: {best_result['coverage']:.1f}%")
        print()
        
        # 检查关键指标
        checks = []
        
        # 1. 覆盖率应该 > 30%
        if best_result['coverage'] > 30:
            checks.append(("✅", f"覆盖率 {best_result['coverage']:.1f}% > 30%"))
        else:
            checks.append(("❌", f"覆盖率 {best_result['coverage']:.1f}% < 30% (太低)"))
        
        # 2. 边界内比例应该 > 95%
        if best_result['in_bounds_ratio'] > 95:
            checks.append(("✅", f"边界内比例 {best_result['in_bounds_ratio']:.1f}% > 95%"))
        else:
            checks.append(("⚠️", f"边界内比例 {best_result['in_bounds_ratio']:.1f}% < 95%"))
        
        # 3. 投影坐标范围应该合理
        x_min, x_max = best_result['x_range']
        y_min, y_max = best_result['y_range']
        img_w, img_h = best_result['img_size']
        
        if 0 <= x_min < img_w and 0 <= x_max < img_w:
            checks.append(("✅", f"X 坐标范围 [{x_min:.0f}, {x_max:.0f}] 在图像宽度 {img_w} 内"))
        else:
            checks.append(("⚠️", f"X 坐标范围 [{x_min:.0f}, {x_max:.0f}] 超出图像宽度 {img_w}"))
        
        if 0 <= y_min < img_h and 0 <= y_max < img_h:
            checks.append(("✅", f"Y 坐标范围 [{y_min:.0f}, {y_max:.0f}] 在图像高度 {img_h} 内"))
        else:
            checks.append(("⚠️", f"Y 坐标范围 [{y_min:.0f}, {y_max:.0f}] 超出图像高度 {img_h}"))
        
        print("验证结果:")
        for status, msg in checks:
            print(f"  {status} {msg}")
        
        print()
        
        # 总体判断
        all_pass = all(status == "✅" for status, _ in checks)
        
        if all_pass:
            print("🎉 投影计算正确!")
            print()
            print("下一步:")
            print("  1. 应用到 open_vocab_dataset_v2.py")
            print("  2. 运行训练测试")
            return True
        else:
            print("⚠️  投影计算可能有问题,请检查:")
            print("  - 相机内参是否正确")
            print("  - 坐标系是否一致")
            print("  - 深度图单位是否正确")
            return False
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试运行时投影计算")
    parser.add_argument("--scene", type=str, default="scene0000_00",
                       help="场景名称")
    parser.add_argument("--data-3d-root", type=str, 
                       default="/home/featurize/data/scannet_3d",
                       help="3D 数据根目录")
    parser.add_argument("--data-2d-root", type=str,
                       default="/home/featurize/data/scannet_2d",
                       help="2D 数据根目录")
    parser.add_argument("--num-frames", type=int, default=3,
                       help="测试帧数")
    
    args = parser.parse_args()
    
    success = test_runtime_projection(
        scene_name=args.scene,
        data_3d_root=args.data_3d_root,
        data_2d_root=args.data_2d_root,
        num_frames_to_test=args.num_frames,
    )
    
    sys.exit(0 if success else 1)
