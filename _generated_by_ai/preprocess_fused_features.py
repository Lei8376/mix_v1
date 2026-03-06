"""
多帧融合特征预处理脚本

这个脚本遍历所有 ScanNet 场景，对每个场景：
1. 加载 3D 点云数据 (.pth)
2. 遍历该场景的所有 2D 帧
3. 对每一帧进行 3D→2D 投影
4. 从对应的 *_odise.npz 中读取 mask_embeddings 和 pixel_pooled 特征
5. 将 2D 特征累加到对应的 3D 点上
6. 最后平均并保存为 _fused.pt 文件

注意：
- 正确处理 mapping[:, 0] = y, mapping[:, 1] = x 的坐标顺序
- mask 索引使用 mask[y, x] 而不是 mask[x, y]
- 深度图单位从 mm 转换为 m
- 添加边界检查
"""

import os
import sys
import yaml
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import argparse
from PIL import Image

# 添加项目路径
sys.path.insert(0, '/home/featurize/work/mix')

from utils.mapping_util import get_point2img_mapper


def load_pose(pose_path):
    """加载相机位姿矩阵"""
    return np.loadtxt(pose_path)


def load_depth(depth_path):
    """加载深度图并转换为米"""
    depth = np.array(Image.open(depth_path))
    # ScanNet 深度图单位是毫米，需要转换为米
    depth = depth.astype(np.float32) / 1000.0
    return depth


def process_one_scene(scene_name, data_root_3d, data_root_2d, npz_dir, output_dir, split, point2img_mapper, save_fp16=True):
    """
    处理单个场景，生成融合特征
    
    Args:
        scene_name: 场景名称，如 'scene0000_00'
        data_root_3d: 3D 数据根目录
        data_root_2d: 2D 数据根目录
        npz_dir: NPZ 特征文件目录
        output_dir: 输出目录
        split: 数据集划分 ('train', 'val', 'test')
        point2img_mapper: 投影器对象
    """
    # 1. 加载 3D 点云数据（兼容 scene0000_00.pth 和 scene0000_00_vh_clean_2.pth）
    pth_path = os.path.join(data_root_3d, split, f"{scene_name}.pth")
    if not os.path.exists(pth_path):
        pth_path = os.path.join(data_root_3d, split, f"{scene_name}_vh_clean_2.pth")
    if not os.path.exists(pth_path):
        print(f"⚠️  {scene_name}.pth 和 {scene_name}_vh_clean_2.pth 都不存在，跳过")
        return None
    
    try:
        data_3d = torch.load(pth_path)
        if isinstance(data_3d, tuple):
            locs, feats, labels = data_3d
        elif isinstance(data_3d, dict):
            locs = data_3d['locs']
            feats = data_3d['feats']
            labels = data_3d['labels']
        else:
            print(f"⚠️  未知的 3D 数据格式: {type(data_3d)}")
            return None
    except Exception as e:
        print(f"⚠️  加载 {pth_path} 失败: {e}")
        return None
    
    # 转换为 numpy
    if isinstance(locs, torch.Tensor):
        locs = locs.numpy()
    if isinstance(feats, torch.Tensor):
        feats = feats.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    N = locs.shape[0]
    
    # 2. 初始化累加器
    sum_mask_embed = np.zeros((N, 256), dtype=np.float32)
    sum_pixel_pooled = np.zeros((N, 512), dtype=np.float32)
    counter = np.zeros((N,), dtype=np.float32)
    
    # 3. 获取所有帧
    scene_2d_dir = os.path.join(data_root_2d, scene_name)
    if not os.path.exists(scene_2d_dir):
        print(f"⚠️  {scene_2d_dir} 不存在，跳过")
        return None
    
    img_paths = sorted(glob(os.path.join(scene_2d_dir, "color", "*.jpg")))
    if len(img_paths) == 0:
        print(f"⚠️  {scene_name} 没有图像，跳过")
        return None
    
    num_frames_used = 0
    
    # 4. 遍历所有帧
    for img_path in img_paths:
        frame_idx = os.path.basename(img_path).replace(".jpg", "")
        
        # 检查对应的 pose 和 depth 是否存在
        pose_path = os.path.join(scene_2d_dir, "pose", f"{frame_idx}.txt")
        depth_path = os.path.join(scene_2d_dir, "depth", f"{frame_idx}.png")
        
        if not os.path.exists(pose_path) or not os.path.exists(depth_path):
            continue
        
        # 加载 pose 和 depth
        try:
            pose = load_pose(pose_path)
            depth = load_depth(depth_path)
        except Exception as e:
            print(f"⚠️  加载 {frame_idx} 的 pose/depth 失败: {e}")
            continue
        
        # 计算投影
        try:
            mapping = point2img_mapper.compute_mapping(pose, locs, depth)
        except Exception as e:
            print(f"⚠️  {scene_name} 帧 {frame_idx} 投影失败: {e}")
            continue
        
        # mapping[:, 0] = y, mapping[:, 1] = x, mapping[:, 2] = valid
        visible = mapping[:, 2] == 1
        if visible.sum() == 0:
            continue
        
        # 加载对应帧的 npz
        npz_path = os.path.join(npz_dir, scene_name, f"{frame_idx}_odise.npz")
        if not os.path.exists(npz_path):
            continue
        
        try:
            npz_data = np.load(npz_path, allow_pickle=True)
            masks = npz_data["masks"]                    # (K, H, W)
            mask_embeds = npz_data["mask_embeddings"]    # (K, 256)
            pixel_pooled = npz_data["pixel_pooled"]      # (K, 512)
        except Exception as e:
            print(f"⚠️  加载 {npz_path} 失败: {e}")
            continue
        
        K = masks.shape[0]
        if K == 0:
            continue
        
        # 提取坐标（注意：mapping[:, 0] 是 y，mapping[:, 1] 是 x）
        y_coords = mapping[visible, 0].astype(int)  # ✅ 第 0 列是 y
        x_coords = mapping[visible, 1].astype(int)  # ✅ 第 1 列是 x
        
        # 边界检查
        H, W = masks.shape[1], masks.shape[2]
        valid_bounds = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)
        
        if valid_bounds.sum() == 0:
            continue
        
        y_valid = y_coords[valid_bounds]
        x_valid = x_coords[valid_bounds]
        visible_indices = np.where(visible)[0][valid_bounds]
        
        # 对每个 mask 累加特征
        for k in range(K):
            mask_2d = masks[k]  # (H, W)
            
            # 检查哪些可见点在这个 mask 里（注意：mask[y, x] 不是 mask[x, y]）
            in_mask = mask_2d[y_valid, x_valid] > 0.5  # ✅ 先 y 后 x
            
            if in_mask.sum() == 0:
                continue
            
            # 累加这个 mask 的特征到对应的 3D 点
            point_indices = visible_indices[in_mask]
            sum_mask_embed[point_indices] += mask_embeds[k]
            sum_pixel_pooled[point_indices] += pixel_pooled[k]
            counter[point_indices] += 1
        
        num_frames_used += 1
    
    # 5. 平均特征
    valid_mask = counter > 0
    coverage = valid_mask.sum() / N
    
    if coverage < 0.5:
        print(f"⚠️  {scene_name}: 覆盖率过低 ({coverage:.2%})，可能有问题")
    
    # 避免除零
    counter_safe = np.maximum(counter, 1e-5)
    mask_embeddings_3d = sum_mask_embed / counter_safe[:, None]
    pixel_pooled_3d = sum_pixel_pooled / counter_safe[:, None]
    
    # 6. 保存
    output_split_dir = os.path.join(output_dir, split)
    os.makedirs(output_split_dir, exist_ok=True)
    output_path = os.path.join(output_split_dir, f"{scene_name}_fused.pt")
    
    # 根据精度选择保存格式
    if save_fp16:
        me_3d = torch.from_numpy(mask_embeddings_3d).half()   # float16 节省空间
        pp_3d = torch.from_numpy(pixel_pooled_3d).half()
    else:
        me_3d = torch.from_numpy(mask_embeddings_3d).float()  # float32 保留精度
        pp_3d = torch.from_numpy(pixel_pooled_3d).float()
    
    torch.save({
        "locs": torch.from_numpy(locs).float(),
        "feats": torch.from_numpy(feats).float(),
        "labels": torch.from_numpy(labels).long(),
        "mask_embeddings_3d": me_3d,     # ODISE mask-level embedding 融合到 3D
        "pixel_pooled_3d": pp_3d,        # LSeg pixel-pooled feature 融合到 3D
        "mask_full": torch.from_numpy(valid_mask),
        "coverage": coverage,
        "num_frames_used": num_frames_used,
    }, output_path)
    
    return {
        "scene": scene_name,
        "coverage": coverage,
        "num_frames": num_frames_used,
        "num_points": N,
    }


def main():
    parser = argparse.ArgumentParser(description="预处理 ScanNet 多帧融合特征")
    parser.add_argument("--data-root-3d", type=str, default="/home/featurize/data/scannet_3d",
                        help="3D 数据根目录")
    parser.add_argument("--data-root-2d", type=str, default="/home/featurize/data/scannet_2d",
                        help="2D 数据根目录")
    parser.add_argument("--npz-dir", type=str, default="/home/featurize/data/pixel_pooled",
                        help="NPZ 特征目录")
    parser.add_argument("--output-dir", type=str, default="/home/featurize/data/scannet_fused",
                        help="输出目录")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"],
                        help="要处理的数据集划分")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="最大处理场景数（用于测试）")
    parser.add_argument("--fp32", action="store_true",
                        help="用 float32 保存特征（默认 float16，磁盘空间翻倍但精度更高）")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ScanNet 多帧融合特征预处理")
    print("=" * 80)
    print(f"3D 数据根目录: {args.data_root_3d}")
    print(f"2D 数据根目录: {args.data_root_2d}")
    print(f"NPZ 特征目录: {args.npz_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"处理划分: {args.splits}")
    print(f"保存精度: {'float32' if args.fp32 else 'float16'}")
    print("=" * 80)
    
    # 初始化投影器
    print("\n初始化投影器...")
    point2img_mapper = get_point2img_mapper()
    print("✅ 投影器初始化完成")
    
    # 处理每个划分
    for split in args.splits:
        print(f"\n{'=' * 80}")
        print(f"处理 {split} 划分")
        print(f"{'=' * 80}")
        
        # 获取所有场景
        split_dir = os.path.join(args.data_root_3d, split)
        if not os.path.exists(split_dir):
            print(f"⚠️  {split_dir} 不存在，跳过")
            continue
        
        pth_files = sorted(glob(os.path.join(split_dir, "*.pth")))
        # 去掉 _vh_clean_2 后缀，使 scene_name 与 2D/NPZ 目录一致
        scene_names = [os.path.basename(f).replace(".pth", "").replace("_vh_clean_2", "") for f in pth_files]
        
        if args.max_scenes is not None:
            scene_names = scene_names[:args.max_scenes]
        
        print(f"找到 {len(scene_names)} 个场景")
        
        # 统计信息
        results = []
        failed_scenes = []
        
        # 处理每个场景
        for scene_name in tqdm(scene_names, desc=f"处理 {split}"):
            result = process_one_scene(
                scene_name=scene_name,
                data_root_3d=args.data_root_3d,
                data_root_2d=args.data_root_2d,
                npz_dir=args.npz_dir,
                output_dir=args.output_dir,
                split=split,
                point2img_mapper=point2img_mapper,
                save_fp16=not args.fp32,
            )
            
            if result is not None:
                results.append(result)
            else:
                failed_scenes.append(scene_name)
        
        # 打印统计信息
        print(f"\n{'=' * 80}")
        print(f"{split} 划分处理完成")
        print(f"{'=' * 80}")
        print(f"成功: {len(results)} / {len(scene_names)}")
        print(f"失败: {len(failed_scenes)}")
        
        if len(results) > 0:
            avg_coverage = np.mean([r["coverage"] for r in results])
            avg_frames = np.mean([r["num_frames"] for r in results])
            print(f"平均覆盖率: {avg_coverage:.2%}")
            print(f"平均使用帧数: {avg_frames:.1f}")
            
            # 覆盖率分布
            coverages = [r["coverage"] for r in results]
            print(f"覆盖率分布:")
            print(f"  < 50%: {sum(1 for c in coverages if c < 0.5)}")
            print(f"  50-70%: {sum(1 for c in coverages if 0.5 <= c < 0.7)}")
            print(f"  70-90%: {sum(1 for c in coverages if 0.7 <= c < 0.9)}")
            print(f"  >= 90%: {sum(1 for c in coverages if c >= 0.9)}")
        
        if len(failed_scenes) > 0:
            print(f"\n失败的场景:")
            for scene in failed_scenes[:10]:  # 只显示前 10 个
                print(f"  - {scene}")
            if len(failed_scenes) > 10:
                print(f"  ... 还有 {len(failed_scenes) - 10} 个")
    
    print(f"\n{'=' * 80}")
    print("✅ 全部完成!")
    print(f"{'=' * 80}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
