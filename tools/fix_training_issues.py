#!/usr/bin/env python3
"""
检查和修复 open_vocab_fusion 模型训练中的常见问题。

这个脚本会：
1. 检查预计算数据的完整性
2. 验证 x_label 和 y_label 的有效性
3. 修复数据类型不一致问题
4. 提供训练前的健康检查

Usage:
    python fix_training_issues.py --precomputed-dir /path/to/precomputed --data-root /path/to/scannet_3d
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm


def check_precomputed_data(precomputed_dir: Path) -> Dict[str, List[str]]:
    """检查预计算数据的完整性。"""
    issues = {"missing_files": [], "invalid_data": [], "warnings": []}
    
    print("Checking precomputed data integrity...")
    
    scene_dirs = [d for d in precomputed_dir.iterdir() if d.is_dir() and d.name.startswith("scene")]
    
    for scene_dir in tqdm(scene_dirs, desc="Checking scenes"):
        npz_files = list(scene_dir.glob("*_odise.npz"))
        
        if len(npz_files) == 0:
            issues["missing_files"].append(f"No *_odise.npz files in {scene_dir}")
            continue
        
        for npz_file in npz_files:
            try:
                with np.load(npz_file, allow_pickle=True) as data:
                    required_keys = ["masks", "mask_embeddings"]
                    missing_keys = [k for k in required_keys if k not in data.files]
                    
                    if missing_keys:
                        issues["invalid_data"].append(f"{npz_file}: missing keys {missing_keys}")
                        continue
                    
                    # 检查数据形状和类型
                    masks = data["masks"]
                    mask_embeddings = data["mask_embeddings"]
                    
                    if masks.dtype == object:
                        masks = np.stack(masks, axis=0)
                    
                    if len(masks) != len(mask_embeddings):
                        issues["invalid_data"].append(f"{npz_file}: mask count mismatch")
                    
                    # 检查是否有 pixel_pooled
                    if "pixel_pooled" not in data.files:
                        issues["warnings"].append(f"{npz_file}: missing pixel_pooled, may need pixel-level features")
                    else:
                        pixel_pooled = data["pixel_pooled"]
                        if len(pixel_pooled) != len(masks):
                            issues["invalid_data"].append(f"{npz_file}: pixel_pooled count mismatch")
                    
            except Exception as e:
                issues["invalid_data"].append(f"{npz_file}: {str(e)}")
    
    return issues


def check_3d_data(data_root: Path, split: str = "train") -> Dict[str, List[str]]:
    """检查 3D 数据中的 x_label 和 y_label。"""
    issues = {"missing_labels": [], "invalid_labels": [], "warnings": []}
    
    print(f"Checking 3D data labels for split: {split}")
    
    split_dir = data_root / split
    if not split_dir.exists():
        issues["missing_labels"].append(f"Split directory not found: {split_dir}")
        return issues
    
    pth_files = list(split_dir.glob("*.pth"))
    
    for pth_file in tqdm(pth_files, desc="Checking 3D files"):
        try:
            data = torch.load(pth_file, map_location="cpu", weights_only=False)
            
            if isinstance(data, dict):
                x_label = data.get("x_label")
                y_label = data.get("y_label")
                
                if x_label is None or y_label is None:
                    issues["missing_labels"].append(f"{pth_file}: missing x_label or y_label")
                    continue
                
                # 检查标签的有效性
                if isinstance(x_label, (torch.Tensor, np.ndarray)):
                    if (x_label == 0).all() and (y_label == 0).all():
                        issues["warnings"].append(f"{pth_file}: all x_label/y_label are 0")
                    
                    # 检查标签范围是否合理（假设图像尺寸不超过 2000x2000）
                    if hasattr(x_label, 'max'):
                        if x_label.max() > 2000 or y_label.max() > 2000:
                            issues["invalid_labels"].append(f"{pth_file}: labels out of reasonable range")
                        if x_label.min() < 0 or y_label.min() < 0:
                            issues["invalid_labels"].append(f"{pth_file}: negative label values")
                else:
                    issues["invalid_labels"].append(f"{pth_file}: labels are not tensor/array")
            else:
                issues["missing_labels"].append(f"{pth_file}: data is not dict format")
                
        except Exception as e:
            issues["invalid_labels"].append(f"{pth_file}: {str(e)}")
    
    return issues


def fix_data_types(precomputed_dir: Path, backup: bool = True) -> int:
    """修复预计算数据中的数据类型问题。"""
    fixed_count = 0
    
    print("Fixing data type issues...")
    
    scene_dirs = [d for d in precomputed_dir.iterdir() if d.is_dir() and d.name.startswith("scene")]
    
    for scene_dir in tqdm(scene_dirs, desc="Fixing data types"):
        npz_files = list(scene_dir.glob("*_odise.npz"))
        
        for npz_file in npz_files:
            try:
                # 备份原文件
                if backup:
                    backup_file = npz_file.with_suffix(".npz.backup")
                    if not backup_file.exists():
                        import shutil
                        shutil.copy2(npz_file, backup_file)
                
                # 读取数据
                with np.load(npz_file, allow_pickle=True) as data:
                    data_dict = {k: data[k] for k in data.files}
                
                needs_fix = False
                
                # 修复 masks 数据类型
                if "masks" in data_dict:
                    masks = data_dict["masks"]
                    if masks.dtype == object:
                        masks = np.stack(masks, axis=0)
                        data_dict["masks"] = masks.astype(bool)
                        needs_fix = True
                
                # 修复 mask_embeddings 数据类型
                if "mask_embeddings" in data_dict:
                    mask_embeddings = data_dict["mask_embeddings"]
                    if mask_embeddings.dtype != np.float32:
                        data_dict["mask_embeddings"] = mask_embeddings.astype(np.float32)
                        needs_fix = True
                
                # 修复 pixel_pooled 数据类型
                if "pixel_pooled" in data_dict:
                    pixel_pooled = data_dict["pixel_pooled"]
                    if pixel_pooled.dtype != np.float32:
                        data_dict["pixel_pooled"] = pixel_pooled.astype(np.float32)
                        needs_fix = True
                
                # 保存修复后的数据
                if needs_fix:
                    np.savez_compressed(npz_file, **data_dict)
                    fixed_count += 1
                
            except Exception as e:
                print(f"Error fixing {npz_file}: {e}")
    
    return fixed_count


def generate_health_report(
    precomputed_issues: Dict[str, List[str]],
    data_3d_issues: Dict[str, List[str]]
) -> str:
    """生成健康检查报告。"""
    report = []
    report.append("=" * 60)
    report.append("OPEN VOCAB FUSION TRAINING HEALTH REPORT")
    report.append("=" * 60)
    
    # 预计算数据问题
    report.append("\n📁 PRECOMPUTED DATA ISSUES:")
    total_precomputed_issues = sum(len(issues) for issues in precomputed_issues.values())
    if total_precomputed_issues == 0:
        report.append("✅ No issues found in precomputed data")
    else:
        for issue_type, issues in precomputed_issues.items():
            if issues:
                report.append(f"\n❌ {issue_type.upper()} ({len(issues)} issues):")
                for issue in issues[:5]:  # 只显示前5个
                    report.append(f"   - {issue}")
                if len(issues) > 5:
                    report.append(f"   ... and {len(issues) - 5} more")
    
    # 3D 数据问题
    report.append("\n🎯 3D DATA LABEL ISSUES:")
    total_3d_issues = sum(len(issues) for issues in data_3d_issues.values())
    if total_3d_issues == 0:
        report.append("✅ No issues found in 3D data labels")
    else:
        for issue_type, issues in data_3d_issues.items():
            if issues:
                report.append(f"\n❌ {issue_type.upper()} ({len(issues)} issues):")
                for issue in issues[:5]:  # 只显示前5个
                    report.append(f"   - {issue}")
                if len(issues) > 5:
                    report.append(f"   ... and {len(issues) - 5} more")
    
    # 建议
    report.append("\n💡 RECOMMENDATIONS:")
    if any(data_3d_issues["missing_labels"]) or any(data_3d_issues["warnings"]):
        report.append("   - Run generate_projection_labels.py to create proper x_label/y_label")
    if any(precomputed_issues["invalid_data"]):
        report.append("   - Fix precomputed data format issues")
    if any(precomputed_issues["warnings"]):
        report.append("   - Consider regenerating precomputed features with pixel_pooled")
    
    if total_precomputed_issues == 0 and total_3d_issues == 0:
        report.append("🎉 All checks passed! Ready for training.")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Check and fix training issues")
    parser.add_argument("--precomputed-dir", type=str, required=True,
                       help="Path to precomputed features directory")
    parser.add_argument("--data-root", type=str,
                       help="Path to 3D data root (optional)")
    parser.add_argument("--split", type=str, default="train",
                       help="Data split to check")
    parser.add_argument("--fix-types", action="store_true",
                       help="Fix data type issues in precomputed data")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup files when fixing")
    parser.add_argument("--output-report", type=str,
                       help="Save report to file")
    
    args = parser.parse_args()
    
    precomputed_dir = Path(args.precomputed_dir)
    if not precomputed_dir.exists():
        print(f"Error: Precomputed directory not found: {precomputed_dir}")
        return
    
    # 检查预计算数据
    precomputed_issues = check_precomputed_data(precomputed_dir)
    
    # 检查 3D 数据（如果提供了路径）
    data_3d_issues = {"missing_labels": [], "invalid_labels": [], "warnings": []}
    if args.data_root:
        data_root = Path(args.data_root)
        if data_root.exists():
            data_3d_issues = check_3d_data(data_root, args.split)
        else:
            print(f"Warning: 3D data root not found: {data_root}")
    
    # 修复数据类型问题
    if args.fix_types:
        fixed_count = fix_data_types(precomputed_dir, backup=not args.no_backup)
        print(f"Fixed data types in {fixed_count} files")
    
    # 生成报告
    report = generate_health_report(precomputed_issues, data_3d_issues)
    print(report)
    
    # 保存报告到文件
    if args.output_report:
        with open(args.output_report, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output_report}")


if __name__ == "__main__":
    main()