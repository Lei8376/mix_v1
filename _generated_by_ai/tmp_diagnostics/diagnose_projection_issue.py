#!/usr/bin/env python3
"""
诊断投影标签问题的脚本。

这个脚本会检查:
1. .pth 文件是否包含 x_label 和 y_label
2. 如果包含,检查它们的有效性
3. 如果不包含,说明需要生成
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import yaml

# 添加项目根目录到 Python 路径
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


def check_pth_file(pth_path: Path) -> dict:
    """检查单个 .pth 文件。"""
    result = {
        "path": str(pth_path),
        "format": None,
        "has_x_label": False,
        "has_y_label": False,
        "x_label_valid": False,
        "y_label_valid": False,
        "num_points": 0,
        "num_valid_projections": 0,
    }
    
    try:
        data = torch.load(pth_path, map_location="cpu", weights_only=False)
        
        if isinstance(data, (list, tuple)):
            result["format"] = "tuple"
            result["num_points"] = len(data[0]) if len(data) > 0 else 0
        elif isinstance(data, dict):
            result["format"] = "dict"
            result["num_points"] = len(data.get("locs", data.get("coords", [])))
            
            # 检查 x_label 和 y_label
            if "x_label" in data:
                result["has_x_label"] = True
                x_label = data["x_label"]
                if isinstance(x_label, torch.Tensor):
                    x_label = x_label.numpy()
                
                # 检查是否全为 0
                if not (x_label == 0).all():
                    result["x_label_valid"] = True
                    result["num_valid_projections"] = np.sum(x_label != 0)
            
            if "y_label" in data:
                result["has_y_label"] = True
                y_label = data["y_label"]
                if isinstance(y_label, torch.Tensor):
                    y_label = y_label.numpy()
                
                if not (y_label == 0).all():
                    result["y_label_valid"] = True
        else:
            result["format"] = "unknown"
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


def main():
    print("=" * 80)
    print("投影标签诊断工具")
    print("=" * 80)
    print()
    
    # 读取配置
    config_path = Path(project_root) / "config" / "data_scannet_3d.yaml"
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_root = Path(config["DATA"]["data_root"])
    data_2d_root = Path(config["DATA"].get("data_root_2d", ""))
    
    print(f"📁 3D 数据根目录: {data_root}")
    print(f"📁 2D 数据根目录: {data_2d_root}")
    print()
    
    # 检查目录是否存在
    if not data_root.exists():
        print(f"❌ 3D 数据目录不存在: {data_root}")
        return
    
    # 检查 train split
    train_dir = data_root / "train"
    if not train_dir.exists():
        print(f"❌ train 目录不存在: {train_dir}")
        return
    
    # 获取前几个 .pth 文件
    pth_files = sorted(train_dir.glob("*.pth"))[:5]
    
    if len(pth_files) == 0:
        print(f"❌ 在 {train_dir} 中没有找到 .pth 文件")
        return
    
    print(f"📊 检查前 {len(pth_files)} 个文件...")
    print()
    
    # 统计信息
    stats = {
        "total": 0,
        "tuple_format": 0,
        "dict_format": 0,
        "has_labels": 0,
        "valid_labels": 0,
    }
    
    for pth_file in pth_files:
        result = check_pth_file(pth_file)
        stats["total"] += 1
        
        print(f"文件: {pth_file.name}")
        print(f"  格式: {result['format']}")
        print(f"  点数: {result['num_points']}")
        
        if result["format"] == "tuple":
            stats["tuple_format"] += 1
            print(f"  ⚠️  Tuple 格式 - 没有 x_label/y_label")
        elif result["format"] == "dict":
            stats["dict_format"] += 1
            if result["has_x_label"] and result["has_y_label"]:
                stats["has_labels"] += 1
                print(f"  ✅ 包含 x_label 和 y_label")
                
                if result["x_label_valid"] and result["y_label_valid"]:
                    stats["valid_labels"] += 1
                    print(f"  ✅ 标签有效 ({result['num_valid_projections']} 个有效投影)")
                else:
                    print(f"  ⚠️  标签全为 0 (无效)")
            else:
                print(f"  ❌ 缺少 x_label 或 y_label")
        
        print()
    
    # 打印总结
    print("=" * 80)
    print("总结")
    print("=" * 80)
    print(f"总文件数: {stats['total']}")
    print(f"Tuple 格式: {stats['tuple_format']}")
    print(f"Dict 格式: {stats['dict_format']}")
    print(f"包含标签: {stats['has_labels']}")
    print(f"标签有效: {stats['valid_labels']}")
    print()
    
    # 给出建议
    if stats["tuple_format"] > 0:
        print("🔧 问题诊断:")
        print("  你的 .pth 文件是旧的 tuple 格式 (locs, feats, labels),")
        print("  不包含 x_label 和 y_label 投影标签。")
        print()
        print("💡 解决方案:")
        print()
        print("  方案 1 (推荐): 生成包含投影标签的新数据文件")
        print("  ----------------------------------------")
        print("  运行以下命令:")
        print()
        print("    bash scripts/generate_projection_labels.sh")
        print()
        print("  或手动运行:")
        print()
        print(f"    python tools/generate_projection_labels.py \\")
        print(f"      --data-root {data_root} \\")
        print(f"      --data-2d-root {data_2d_root} \\")
        print(f"      --output-dir {data_root.parent}/scannet_3d_with_projection \\")
        print(f"      --split train")
        print()
        print("  然后更新 config/data_scannet_3d.yaml:")
        print(f"    data_root: {data_root.parent}/scannet_3d_with_projection")
        print()
        print("  方案 2: 使用动态投影计算 (较慢)")
        print("  ----------------------------------------")
        print("  修改 dataset/open_vocab_dataset_v2.py 使用 projection_helper.py")
        print("  在运行时动态计算投影 (会降低训练速度)")
        print()
    elif stats["has_labels"] == stats["total"] and stats["valid_labels"] == stats["total"]:
        print("✅ 所有文件都包含有效的投影标签!")
        print("   可以正常训练。")
    elif stats["has_labels"] == stats["total"] and stats["valid_labels"] == 0:
        print("⚠️  文件包含 x_label/y_label,但全为 0 (无效)。")
        print("   需要重新生成投影标签。")
    
    # 检查 2D 数据
    print()
    print("=" * 80)
    print("2D 数据检查")
    print("=" * 80)
    
    if not data_2d_root.exists():
        print(f"❌ 2D 数据目录不存在: {data_2d_root}")
        print("   生成投影标签需要 2D 数据 (图像、深度图、相机位姿)")
    else:
        # 检查一个场景
        scenes = sorted([d for d in data_2d_root.iterdir() if d.is_dir() and d.name.startswith("scene")])
        if len(scenes) > 0:
            scene = scenes[0]
            print(f"✅ 2D 数据存在: {data_2d_root}")
            print(f"   示例场景: {scene.name}")
            
            # 检查子目录
            subdirs = ["color", "depth", "pose"]
            for subdir in subdirs:
                subdir_path = scene / subdir
                if subdir_path.exists():
                    if subdir_path.is_dir():
                        count = len(list(subdir_path.iterdir()))
                        print(f"   ✅ {subdir}/: {count} 个文件")
                    else:
                        count = len(list(subdir_path.parent.glob(f"{subdir}/*.txt")))
                        print(f"   ✅ {subdir}: {count} 个文件")
                else:
                    print(f"   ❌ {subdir}/: 不存在")
        else:
            print(f"❌ 在 {data_2d_root} 中没有找到场景目录")


if __name__ == "__main__":
    main()
