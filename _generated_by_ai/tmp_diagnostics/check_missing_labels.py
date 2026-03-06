#!/usr/bin/env python3
"""
临时诊断脚本：检查 3D 数据文件中 x_label/y_label 缺失情况
使用后可删除此脚本
"""

import os
from pathlib import Path
import torch
from collections import defaultdict

def check_pth_file(pth_path):
    """检查单个 .pth 文件是否包含 x_label 和 y_label"""
    try:
        data = torch.load(pth_path, map_location="cpu", weights_only=False)
        
        has_x_label = False
        has_y_label = False
        
        if isinstance(data, dict):
            has_x_label = "x_label" in data and data["x_label"] is not None
            has_y_label = "y_label" in data and data["y_label"] is not None
        
        return {
            "has_x_label": has_x_label,
            "has_y_label": has_y_label,
            "is_dict": isinstance(data, dict),
            "keys": list(data.keys()) if isinstance(data, dict) else None
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    # 从配置读取数据路径
    data_root = Path("/home/featurize/data/scannet_3d")
    train_dir = data_root / "train"
    
    if not train_dir.exists():
        print(f"❌ 训练数据目录不存在: {train_dir}")
        return
    
    print(f"📁 检查目录: {train_dir}")
    print("=" * 80)
    
    # 统计信息
    stats = {
        "total": 0,
        "missing_both": 0,
        "missing_x_only": 0,
        "missing_y_only": 0,
        "has_both": 0,
        "error": 0,
        "not_dict": 0
    }
    
    missing_files = []
    has_label_files = []
    error_files = []
    
    # 遍历所有 .pth 文件
    pth_files = sorted(train_dir.glob("*.pth"))
    print(f"🔍 找到 {len(pth_files)} 个 .pth 文件\n")
    
    for pth_path in pth_files:
        stats["total"] += 1
        result = check_pth_file(pth_path)
        
        if "error" in result:
            stats["error"] += 1
            error_files.append((pth_path.name, result["error"]))
        elif not result["is_dict"]:
            stats["not_dict"] += 1
            stats["missing_both"] += 1
            missing_files.append(pth_path.name)
        else:
            has_x = result["has_x_label"]
            has_y = result["has_y_label"]
            
            if has_x and has_y:
                stats["has_both"] += 1
                has_label_files.append(pth_path.name)
            elif not has_x and not has_y:
                stats["missing_both"] += 1
                missing_files.append(pth_path.name)
            elif not has_x:
                stats["missing_x_only"] += 1
                missing_files.append(pth_path.name)
            else:
                stats["missing_y_only"] += 1
                missing_files.append(pth_path.name)
    
    # 打印统计结果
    print("\n" + "=" * 80)
    print("📊 统计结果")
    print("=" * 80)
    print(f"总文件数:              {stats['total']}")
    print(f"✅ 有 x_label & y_label: {stats['has_both']} ({stats['has_both']/stats['total']*100:.1f}%)")
    print(f"❌ 缺少两者:            {stats['missing_both']} ({stats['missing_both']/stats['total']*100:.1f}%)")
    print(f"⚠️  只缺 x_label:        {stats['missing_x_only']}")
    print(f"⚠️  只缺 y_label:        {stats['missing_y_only']}")
    print(f"⚠️  非字典格式:          {stats['not_dict']}")
    print(f"❌ 读取错误:            {stats['error']}")
    
    # 打印示例
    if has_label_files:
        print(f"\n✅ 有标签的文件示例（前5个）:")
        for f in has_label_files[:5]:
            print(f"   - {f}")
    
    if missing_files:
        print(f"\n❌ 缺少标签的文件示例（前10个）:")
        for f in missing_files[:10]:
            print(f"   - {f}")
    
    if error_files:
        print(f"\n❌ 读取错误的文件:")
        for f, err in error_files[:5]:
            print(f"   - {f}: {err}")
    
    # 检查一个有标签的文件的详细信息
    if has_label_files:
        sample_file = train_dir / has_label_files[0]
        print(f"\n📝 示例文件详细信息: {has_label_files[0]}")
        data = torch.load(sample_file, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            print(f"   键列表: {list(data.keys())}")
            if "x_label" in data:
                print(f"   x_label shape: {data['x_label'].shape if hasattr(data['x_label'], 'shape') else type(data['x_label'])}")
            if "y_label" in data:
                print(f"   y_label shape: {data['y_label'].shape if hasattr(data['y_label'], 'shape') else type(data['y_label'])}")
    
    # 检查一个缺失标签的文件的详细信息
    if missing_files:
        sample_file = train_dir / missing_files[0]
        print(f"\n📝 缺失标签文件详细信息: {missing_files[0]}")
        data = torch.load(sample_file, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            print(f"   键列表: {list(data.keys())}")
        else:
            print(f"   数据类型: {type(data)}")
            if isinstance(data, (list, tuple)):
                print(f"   元素数量: {len(data)}")
    
    print("\n" + "=" * 80)
    print("💡 结论:")
    missing_ratio = (stats['missing_both'] + stats['missing_x_only'] + stats['missing_y_only']) / stats['total'] * 100
    if missing_ratio > 50:
        print(f"   ⚠️  超过 {missing_ratio:.1f}% 的文件缺少标签")
        print("   可能原因:")
        print("   1. 数据预处理脚本未生成 x_label/y_label")
        print("   2. 使用了旧版本的预处理流程")
        print("   3. 需要运行 2D-3D 对齐脚本生成这些字段")
    else:
        print(f"   ✅ 只有 {missing_ratio:.1f}% 的文件缺少标签，问题不大")
    print("=" * 80)

if __name__ == "__main__":
    main()
