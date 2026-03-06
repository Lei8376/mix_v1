"""
验证融合特征文件的正确性

检查：
1. 文件是否可以正常加载
2. 数据形状是否正确
3. 数值范围是否合理
4. 覆盖率是否符合预期
"""

import os
import sys
import torch
import numpy as np
from glob import glob
import argparse

sys.path.insert(0, '/home/featurize/work/mix')


def verify_one_file(file_path):
    """验证单个融合特征文件"""
    try:
        data = torch.load(file_path)
    except Exception as e:
        return {"success": False, "error": f"加载失败: {e}"}
    
    # 检查必需的键
    required_keys = ["locs", "feats", "labels", "mask_embeddings_3d", "pixel_pooled_3d", "mask_full", "coverage"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        return {"success": False, "error": f"缺失键: {missing_keys}"}
    
    # 检查形状
    locs = data["locs"]
    feats = data["feats"]
    labels = data["labels"]
    mask_embed = data["mask_embeddings_3d"]
    pixel_pooled = data["pixel_pooled_3d"]
    mask_full = data["mask_full"]
    coverage = data["coverage"]
    
    N = locs.shape[0]
    
    # 形状检查
    if locs.shape != (N, 3):
        return {"success": False, "error": f"locs 形状错误: {locs.shape}, 期望 ({N}, 3)"}
    
    if feats.shape != (N, 3):
        return {"success": False, "error": f"feats 形状错误: {feats.shape}, 期望 ({N}, 3)"}
    
    if labels.shape != (N,):
        return {"success": False, "error": f"labels 形状错误: {labels.shape}, 期望 ({N},)"}
    
    if mask_embed.shape != (N, 256):
        return {"success": False, "error": f"mask_embeddings_3d 形状错误: {mask_embed.shape}, 期望 ({N}, 256)"}
    
    if pixel_pooled.shape != (N, 512):
        return {"success": False, "error": f"pixel_pooled_3d 形状错误: {pixel_pooled.shape}, 期望 ({N}, 512)"}
    
    if mask_full.shape != (N,):
        return {"success": False, "error": f"mask_full 形状错误: {mask_full.shape}, 期望 ({N},)"}
    
    # 数值范围检查
    mask_embed_min = mask_embed.min().item()
    mask_embed_max = mask_embed.max().item()
    pixel_pooled_min = pixel_pooled.min().item()
    pixel_pooled_max = pixel_pooled.max().item()
    
    # 检查是否有 NaN 或 Inf
    if torch.isnan(mask_embed).any() or torch.isinf(mask_embed).any():
        return {"success": False, "error": "mask_embeddings_3d 包含 NaN 或 Inf"}
    
    if torch.isnan(pixel_pooled).any() or torch.isinf(pixel_pooled).any():
        return {"success": False, "error": "pixel_pooled_3d 包含 NaN 或 Inf"}
    
    # 覆盖率检查
    actual_coverage = mask_full.sum().item() / N
    if abs(actual_coverage - coverage) > 0.01:
        return {"success": False, "error": f"覆盖率不匹配: 实际 {actual_coverage:.2%}, 记录 {coverage:.2%}"}
    
    # 检查无效点是否为 0（mask_full=False 的点不应该有非零特征）
    invalid_mask = ~mask_full
    if invalid_mask.any():
        invalid_mask_embed_nonzero = (mask_embed[invalid_mask].abs() > 1e-6).any().item()
        invalid_pixel_pooled_nonzero = (pixel_pooled[invalid_mask].abs() > 1e-6).any().item()
        if invalid_mask_embed_nonzero or invalid_pixel_pooled_nonzero:
            return {"success": False, "error": "无效点（mask_full=False）包含非零特征值"}
    
    # 检查有效点是否确实有非零特征
    if mask_full.any():
        valid_mask_embed_all_zero = (mask_embed[mask_full].abs() < 1e-6).all().item()
        valid_pixel_pooled_all_zero = (pixel_pooled[mask_full].abs() < 1e-6).all().item()
        if valid_mask_embed_all_zero:
            return {"success": False, "error": "有效点的 mask_embeddings_3d 全为 0"}
        if valid_pixel_pooled_all_zero:
            return {"success": False, "error": "有效点的 pixel_pooled_3d 全为 0"}
    
    return {
        "success": True,
        "num_points": N,
        "coverage": coverage,
        "num_valid": mask_full.sum().item(),
        "mask_embed_range": (mask_embed_min, mask_embed_max),
        "pixel_pooled_range": (pixel_pooled_min, pixel_pooled_max),
        "num_frames": data.get("num_frames_used", "N/A"),
        "dtype_mask_embed": str(mask_embed.dtype),
        "dtype_pixel_pooled": str(pixel_pooled.dtype),
    }


def main():
    parser = argparse.ArgumentParser(description="验证融合特征文件")
    parser.add_argument("--fused-dir", type=str, default="/home/featurize/data/scannet_fused",
                        help="融合特征目录")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"],
                        help="要验证的数据集划分")
    parser.add_argument("--max-files", type=int, default=None,
                        help="最大验证文件数（用于快速测试）")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细信息")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("验证融合特征文件")
    print("=" * 80)
    print(f"融合特征目录: {args.fused_dir}")
    print(f"验证划分: {args.splits}")
    print("=" * 80)
    
    for split in args.splits:
        print(f"\n{'=' * 80}")
        print(f"验证 {split} 划分")
        print(f"{'=' * 80}")
        
        split_dir = os.path.join(args.fused_dir, split)
        if not os.path.exists(split_dir):
            print(f"⚠️  {split_dir} 不存在")
            continue
        
        pt_files = sorted(glob(os.path.join(split_dir, "*_fused.pt")))
        
        if args.max_files is not None:
            pt_files = pt_files[:args.max_files]
        
        print(f"找到 {len(pt_files)} 个文件")
        
        success_count = 0
        failed_count = 0
        failed_files = []
        
        all_coverages = []
        all_num_points = []
        all_num_valid = []
        
        for pt_file in pt_files:
            scene_name = os.path.basename(pt_file).replace("_fused.pt", "")
            result = verify_one_file(pt_file)
            
            if result["success"]:
                success_count += 1
                all_coverages.append(result["coverage"])
                all_num_points.append(result["num_points"])
                all_num_valid.append(result["num_valid"])
                
                if args.verbose:
                    print(f"✅ {scene_name}")
                    print(f"   点数: {result['num_points']}, 有效: {result['num_valid']}, 覆盖率: {result['coverage']:.2%}")
                    print(f"   mask_embed 范围: [{result['mask_embed_range'][0]:.3f}, {result['mask_embed_range'][1]:.3f}] ({result['dtype_mask_embed']})")
                    print(f"   pixel_pooled 范围: [{result['pixel_pooled_range'][0]:.3f}, {result['pixel_pooled_range'][1]:.3f}] ({result['dtype_pixel_pooled']})")
                    print(f"   使用帧数: {result['num_frames']}")
            else:
                failed_count += 1
                failed_files.append((scene_name, result["error"]))
                print(f"❌ {scene_name}: {result['error']}")
        
        # 打印统计信息
        print(f"\n{'=' * 80}")
        print(f"{split} 验证结果")
        print(f"{'=' * 80}")
        print(f"成功: {success_count} / {len(pt_files)}")
        print(f"失败: {failed_count}")
        
        if success_count > 0:
            print(f"\n统计信息:")
            print(f"  平均点数: {np.mean(all_num_points):.0f}")
            print(f"  平均有效点数: {np.mean(all_num_valid):.0f}")
            print(f"  平均覆盖率: {np.mean(all_coverages):.2%}")
            print(f"  覆盖率范围: [{np.min(all_coverages):.2%}, {np.max(all_coverages):.2%}]")
            
            # 覆盖率分布
            coverages = np.array(all_coverages)
            print(f"\n覆盖率分布:")
            print(f"  < 50%: {np.sum(coverages < 0.5)}")
            print(f"  50-70%: {np.sum((coverages >= 0.5) & (coverages < 0.7))}")
            print(f"  70-90%: {np.sum((coverages >= 0.7) & (coverages < 0.9))}")
            print(f"  >= 90%: {np.sum(coverages >= 0.9)}")
        
        if failed_count > 0:
            print(f"\n失败的文件:")
            for scene, error in failed_files[:10]:
                print(f"  - {scene}: {error}")
            if len(failed_files) > 10:
                print(f"  ... 还有 {len(failed_files) - 10} 个")
    
    print(f"\n{'=' * 80}")
    print("验证完成!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
