#!/usr/bin/env python3
"""
测试投影标签生成的脚本。

这个脚本会:
1. 选择一个场景
2. 生成投影标签
3. 验证结果
4. 可视化投影效果(可选)
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

# 添加项目根目录到 Python 路径
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from tools.generate_projection_labels import (
    load_scene_data,
    compute_projection_labels,
    update_scene_data,
)
from utils.mapping_util import getMapping


def test_single_scene(
    scene_name: str = "scene0000_00",
    data_3d_root: str = "/home/featurize/data/scannet_3d",
    data_2d_root: str = "/home/featurize/data/scannet_2d",
    output_dir: str = "/tmp/test_projection",
):
    """测试单个场景的投影标签生成。"""
    
    print("=" * 80)
    print(f"测试场景: {scene_name}")
    print("=" * 80)
    print()
    
    # 路径
    data_3d_root = Path(data_3d_root)
    data_2d_root = Path(data_2d_root)
    output_dir = Path(output_dir)
    
    # 查找场景文件
    scene_file = data_3d_root / "train" / f"{scene_name}_vh_clean_2.pth"
    if not scene_file.exists():
        scene_file = data_3d_root / "train" / f"{scene_name}.pth"
    
    if not scene_file.exists():
        print(f"❌ 场景文件不存在: {scene_file}")
        return False
    
    print(f"✅ 找到场景文件: {scene_file}")
    
    # 检查 2D 数据
    scene_2d_dir = data_2d_root / scene_name
    if not scene_2d_dir.exists():
        print(f"❌ 2D 数据不存在: {scene_2d_dir}")
        return False
    
    print(f"✅ 找到 2D 数据: {scene_2d_dir}")
    
    # 加载 3D 数据
    print()
    print("步骤 1: 加载 3D 数据")
    print("-" * 80)
    
    try:
        locs, feats, labels = load_scene_data(scene_file)
        print(f"✅ 成功加载 3D 数据")
        print(f"   点数: {len(locs)}")
        print(f"   坐标范围: [{locs.min(axis=0)}, {locs.max(axis=0)}]")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False
    
    # 计算投影标签
    print()
    print("步骤 2: 计算投影标签")
    print("-" * 80)
    
    try:
        point2img_mapper = getMapping()
        projection_data = compute_projection_labels(
            locs, scene_2d_dir, point2img_mapper
        )
        
        if len(projection_data) == 0:
            print(f"⚠️  没有找到有效的投影")
            return False
        
        print(f"✅ 成功计算投影")
        print(f"   有效帧数: {len(projection_data)}")
        
        # 显示每帧的统计
        for frame_id, frame_data in sorted(projection_data.items())[:5]:
            num_valid = len(frame_data["x_label"])
            print(f"   帧 {frame_id}: {num_valid} 个有效投影点")
        
        if len(projection_data) > 5:
            print(f"   ... (共 {len(projection_data)} 帧)")
        
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 选择最佳帧
    print()
    print("步骤 3: 选择最佳投影帧")
    print("-" * 80)
    
    best_frame = max(projection_data.keys(), 
                    key=lambda k: len(projection_data[k]["x_label"]))
    best_data = projection_data[best_frame]
    
    print(f"✅ 选择帧 {best_frame}")
    print(f"   有效投影点: {len(best_data['x_label'])}")
    print(f"   覆盖率: {len(best_data['x_label']) / len(locs) * 100:.1f}%")
    print(f"   x 范围: [{best_data['x_label'].min()}, {best_data['x_label'].max()}]")
    print(f"   y 范围: [{best_data['y_label'].min()}, {best_data['y_label'].max()}]")
    
    # 保存结果
    print()
    print("步骤 4: 保存结果")
    print("-" * 80)
    
    try:
        output_path = output_dir / "train" / scene_file.name
        update_scene_data(scene_file, projection_data, output_path)
        print(f"✅ 保存成功: {output_path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False
    
    # 验证结果
    print()
    print("步骤 5: 验证结果")
    print("-" * 80)
    
    try:
        data = torch.load(output_path, map_location="cpu", weights_only=False)
        
        if not isinstance(data, dict):
            print(f"❌ 数据格式错误: {type(data)}")
            return False
        
        if "x_label" not in data or "y_label" not in data:
            print(f"❌ 缺少投影标签")
            return False
        
        x_label = data["x_label"]
        y_label = data["y_label"]
        
        if isinstance(x_label, torch.Tensor):
            x_label = x_label.numpy()
        if isinstance(y_label, torch.Tensor):
            y_label = y_label.numpy()
        
        num_nonzero_x = np.sum(x_label != 0)
        num_nonzero_y = np.sum(y_label != 0)
        
        print(f"✅ 验证通过")
        print(f"   x_label: {x_label.shape}, {num_nonzero_x} 个非零值")
        print(f"   y_label: {y_label.shape}, {num_nonzero_y} 个非零值")
        print(f"   有效率: {num_nonzero_x / len(x_label) * 100:.1f}%")
        
        if num_nonzero_x == 0 or num_nonzero_y == 0:
            print(f"⚠️  警告: 所有投影标签都是 0")
            return False
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False
    
    print()
    print("=" * 80)
    print("✅ 测试成功!")
    print("=" * 80)
    print()
    print("下一步:")
    print("  1. 运行完整的生成脚本处理所有场景:")
    print("     bash scripts/generate_projection_labels.sh")
    print()
    print("  2. 或手动处理特定场景:")
    print(f"     python tools/generate_projection_labels.py \\")
    print(f"       --data-root {data_3d_root} \\")
    print(f"       --data-2d-root {data_2d_root} \\")
    print(f"       --output-dir /home/featurize/data/scannet_3d_with_projection \\")
    print(f"       --split train")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试投影标签生成")
    parser.add_argument("--scene", type=str, default="scene0000_00",
                       help="场景名称")
    parser.add_argument("--data-3d-root", type=str, 
                       default="/home/featurize/data/scannet_3d",
                       help="3D 数据根目录")
    parser.add_argument("--data-2d-root", type=str,
                       default="/home/featurize/data/scannet_2d",
                       help="2D 数据根目录")
    parser.add_argument("--output-dir", type=str,
                       default="/tmp/test_projection",
                       help="输出目录")
    
    args = parser.parse_args()
    
    success = test_single_scene(
        scene_name=args.scene,
        data_3d_root=args.data_3d_root,
        data_2d_root=args.data_2d_root,
        output_dir=args.output_dir,
    )
    
    sys.exit(0 if success else 1)
