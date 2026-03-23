"""
方案 B：预计算投影坐标

对每个 (scene, frame) 对，预计算 3D→2D 的投影映射，存储：
  - visible_mask: (N,) bool —— 哪些 3D 点在该帧中可见
  - y_label: (N_visible,) int16 —— 可见点的 y 坐标（行索引）
  - x_label: (N_visible,) int16 —— 可见点的 x 坐标（列索引）

这样训练时 __getitem__ 不再需要：
  1. 循环搜索合适的帧（最多 50 次 I/O）
  2. 加载 pose/depth 并计算投影

关键：
  - mapping[:, 0] = y, mapping[:, 1] = x（已修复 XY 坐标 bug）
  - 深度图 /1000 转米
  - 边界检查
  - 只为有对应 npz 的帧计算（确保帧一一对应，修复帧不匹配 bug）
"""

import os
import sys
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import argparse
from PIL import Image

sys.path.insert(0, '/home/featurize/work/mix')
from utils.mapping_util import get_point2img_mapper


def load_3d_data(pth_path):
    """加载 3D 点云，返回 numpy 数组"""
    data = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(data, (list, tuple)):
        locs, feats, labels = data[0], data[1], data[2]
    elif isinstance(data, dict):
        locs = data.get("locs", data.get("coords"))
        feats = data.get("feats", data.get("feat"))
        labels = data.get("labels")
    else:
        raise ValueError(f"Unknown 3D data format: {type(data)}")

    if isinstance(locs, torch.Tensor):
        locs = locs.numpy()
    if isinstance(feats, torch.Tensor):
        feats = feats.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    return locs, feats, labels


def process_one_scene(scene_name, data_root_3d, data_root_2d, npz_dir, output_dir,
                      split, point2img_mapper, min_visible=400):
    """
    为一个场景的所有帧预计算投影坐标。
    
    只处理有对应 npz 的帧，确保投影和 npz 一一匹配。
    """
    # 1. 加载 3D 点云（每个场景只加载一次）
    pth_path = os.path.join(data_root_3d, split, f"{scene_name}.pth")
    if not os.path.exists(pth_path):
        pth_path = os.path.join(data_root_3d, split, f"{scene_name}_vh_clean_2.pth")
    if not os.path.exists(pth_path):
        return {"scene": scene_name, "total": 0, "saved": 0, "skipped": 0}

    locs, feats, labels = load_3d_data(pth_path)
    N = locs.shape[0]

    # 2. 找到该场景所有有 npz 的帧
    npz_scene_dir = os.path.join(npz_dir, scene_name)
    if not os.path.isdir(npz_scene_dir):
        return {"scene": scene_name, "total": 0, "saved": 0, "skipped": 0}

    npz_files = sorted(glob(os.path.join(npz_scene_dir, "*_odise.npz")))
    frame_indices = [os.path.basename(f).replace("_odise.npz", "") for f in npz_files]

    # 3. 2D 数据目录
    scene_2d_dir = os.path.join(data_root_2d, scene_name)
    if not os.path.isdir(scene_2d_dir):
        return {"scene": scene_name, "total": len(frame_indices), "saved": 0, "skipped": len(frame_indices)}

    # 4. 输出目录
    out_scene_dir = os.path.join(output_dir, scene_name)
    os.makedirs(out_scene_dir, exist_ok=True)

    saved = 0
    skipped = 0

    for frame_idx in frame_indices:
        pose_path = os.path.join(scene_2d_dir, "pose", f"{frame_idx}.txt")
        depth_path = os.path.join(scene_2d_dir, "depth", f"{frame_idx}.png")

        if not os.path.exists(pose_path) or not os.path.exists(depth_path):
            skipped += 1
            continue

        try:
            pose = np.loadtxt(pose_path)
            depth = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0  # mm→m
        except Exception:
            skipped += 1
            continue

        # 计算投影
        mapping = point2img_mapper.compute_mapping(pose, locs, depth)
        # mapping[:, 0] = y, mapping[:, 1] = x, mapping[:, 2] = valid
        visible_mask = mapping[:, 2] == 1
        num_visible = visible_mask.sum()

        if num_visible < min_visible:
            skipped += 1
            continue

        # 提取可见点的坐标
        y_label = mapping[visible_mask, 0].astype(np.int16)  # ✅ 第 0 列是 y
        x_label = mapping[visible_mask, 1].astype(np.int16)  # ✅ 第 1 列是 x

        # 保存（非常小：visible_mask ~200KB, x/y_label ~几十KB）
        out_path = os.path.join(out_scene_dir, f"{frame_idx}_proj.npz")
        np.savez_compressed(out_path,
                            visible_mask=visible_mask,    # (N,) bool
                            y_label=y_label,              # (N_vis,) int16
                            x_label=x_label,              # (N_vis,) int16
                            num_points=N)                  # 用于校验

        saved += 1

    return {
        "scene": scene_name,
        "total": len(frame_indices),
        "saved": saved,
        "skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="方案 B: 预计算投影坐标")
    parser.add_argument("--data-root-3d", type=str, default="/home/sunl/work/mix/data/scannet_3d")
    parser.add_argument("--data-root-2d", type=str, default="/home/sunl/work/mix/data/scannet_2d")
    parser.add_argument("--npz-dir", type=str, default="/home/sunl/work/mix/data/pixel_pooled")
    parser.add_argument("--output-dir", type=str, default="/home/sunl/work/mix/data/scannet_projections")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--min-visible", type=int, default=400,
                        help="最少可见点数，低于此值的帧跳过")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="最大场景数（测试用）")

    args = parser.parse_args()

    print("=" * 80)
    print("方案 B: 预计算投影坐标")
    print("=" * 80)
    print(f"3D 数据: {args.data_root_3d}")
    print(f"2D 数据: {args.data_root_2d}")
    print(f"NPZ 目录: {args.npz_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"最少可见点: {args.min_visible}")
    print("=" * 80)

    point2img_mapper = get_point2img_mapper()
    print("✅ 投影器初始化完成\n")

    for split in args.splits:
        print(f"{'=' * 80}")
        print(f"处理 {split}")
        print(f"{'=' * 80}")

        split_dir = os.path.join(args.data_root_3d, split)
        if not os.path.exists(split_dir):
            print(f"⚠️  {split_dir} 不存在")
            continue

        pth_files = sorted(glob(os.path.join(split_dir, "*.pth")))
        scene_names = [os.path.basename(f).replace(".pth", "").replace("_vh_clean_2", "")
                       for f in pth_files]

        if args.max_scenes is not None:
            scene_names = scene_names[:args.max_scenes]

        print(f"找到 {len(scene_names)} 个场景")

        total_saved = 0
        total_skipped = 0
        total_frames = 0

        for scene_name in tqdm(scene_names, desc=f"{split}"):
            result = process_one_scene(
                scene_name=scene_name,
                data_root_3d=args.data_root_3d,
                data_root_2d=args.data_root_2d,
                npz_dir=args.npz_dir,
                output_dir=args.output_dir,
                split=split,
                point2img_mapper=point2img_mapper,
                min_visible=args.min_visible,
            )
            total_frames += result["total"]
            total_saved += result["saved"]
            total_skipped += result["skipped"]

        print(f"\n{split} 完成:")
        print(f"  总帧数: {total_frames}")
        print(f"  已保存: {total_saved} ({total_saved/max(total_frames,1)*100:.1f}%)")
        print(f"  跳过:   {total_skipped} (可见点不足 {args.min_visible})")

    print(f"\n{'=' * 80}")
    print("✅ 完成!")
    print(f"输出目录: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
