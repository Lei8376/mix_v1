"""
层3：深度一致性检查

对每个 visible 点：
  1. 用 pose 把 3D 点变到相机坐标系，取其相机深度 z_cam
  2. 在 depth 图上查 (x,y) 处的深度 z_depth
  3. 统计 |z_cam - z_depth| 的分布

判断标准：
  - 如果投影正确，visible 点的 |z_cam - z_depth| 应小于
    visibility_threshold * z_cam（即 precompute 时的 0.25 倍）
  - 大面积超阈值说明 pose 错/depth 错帧/单位未统一/mapping 逻辑有问题

输出：
  - mean/median/p90/p99 abs depth error
  - 在阈值内点的比例（"depth-consistent ratio"）
  - 失败样本

用法：
  python test_proj/check_projection_depth_consistency.py [--max-samples 50]
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from utils.fusion_util import make_intrinsic, adjust_intrinsic

# 和 mapping_util.py 里完全相同的参数
IMG_DIM    = (320, 240)   # (W, H)
FX, FY     = 577.870605, 577.870605
MX, MY     = 319.5, 239.5
VIS_THR    = 0.25         # visibility_threshold（相对深度误差）
CUT_BOUND  = 10           # cut_num_pixel_boundary

DATA_ROOT_3D = Path("/home/sunl/work/mix/data/scannet_3d")
DATA_ROOT_2D = Path("/home/sunl/work/mix/data/scannet_2d")
NPZ_DIR      = Path("/home/sunl/work/mix/data/pixel_pooled")
PROJ_DIR     = Path("/home/sunl/work/mix/data/scannet_projections")


def build_intrinsic():
    """构建和 mapping_util 完全相同的内参矩阵"""
    intr = make_intrinsic(fx=FX, fy=FY, mx=MX, my=MY)
    intr = adjust_intrinsic(intr, intrinsic_image_dim=[640, 480], image_dim=list(IMG_DIM))
    return intr


def load_pth_locs(data_root_3d: Path, split: str, scene_name: str):
    for suffix in [f"{scene_name}_vh_clean_2.pth", f"{scene_name}.pth"]:
        pth = data_root_3d / split / suffix
        if pth.exists():
            try:
                data = torch.load(pth, map_location="cpu", weights_only=False)
                locs = data[0] if isinstance(data, (list, tuple)) else data.get("locs", data.get("coords"))
                if isinstance(locs, torch.Tensor):
                    locs = locs.numpy()
                return locs.astype(np.float64)
            except Exception:
                return None
    return None


def check_one_sample(scene_name, frame_stem, split, intrinsic,
                     data_root_3d, data_root_2d, proj_dir, locs_cache):
    res = {
        "scene": scene_name, "frame": frame_stem, "status": "ok",
        "n_vis": None,
        "mean_abs_err": None, "median_abs_err": None,
        "p90_abs_err": None,  "p99_abs_err": None,
        "depth_consistent_ratio": None,   # 比例：在 VIS_THR * z_cam 之内
    }

    def fail(reason):
        res["status"] = reason
        return res

    # 加载 proj
    proj_path = proj_dir / scene_name / f"{frame_stem}_proj.npz"
    if not proj_path.exists():
        return fail("no_proj_file")
    try:
        proj = np.load(proj_path)
        vis_mask = proj["visible_mask"].astype(bool)
        x_saved  = proj["x_label"].astype(np.int32)
        y_saved  = proj["y_label"].astype(np.int32)
        N_saved  = int(proj["num_points"])
    except Exception as e:
        return fail(f"proj_load: {e}")

    # 加载 pose / depth
    pose_path  = data_root_2d / scene_name / "pose"  / f"{frame_stem}.txt"
    depth_path = data_root_2d / scene_name / "depth" / f"{frame_stem}.png"
    if not pose_path.exists() or not depth_path.exists():
        return fail("missing_pose_or_depth")
    try:
        pose  = np.loadtxt(str(pose_path))           # camera_to_world (4x4)
        depth = np.array(Image.open(str(depth_path)), dtype=np.float32) / 1000.0  # mm→m
    except Exception as e:
        return fail(f"pose_depth_load: {e}")

    # 加载 locs
    if scene_name not in locs_cache:
        locs_cache[scene_name] = load_pth_locs(data_root_3d, split, scene_name)
    locs = locs_cache[scene_name]
    if locs is None:
        return fail("no_pth")
    if locs.shape[0] != N_saved:
        return fail(f"N_mismatch pth={locs.shape[0]} proj={N_saved}")

    # 取可见点
    vis_locs = locs[vis_mask]   # (N_vis, 3)
    n_vis = vis_locs.shape[0]
    if n_vis == 0:
        return fail("n_vis=0")
    res["n_vis"] = n_vis

    # 把可见点变换到相机坐标系
    world_to_cam = np.linalg.inv(pose)
    locs_h = np.concatenate([vis_locs, np.ones((n_vis, 1))], axis=1).T   # (4, N_vis)
    p_cam  = world_to_cam @ locs_h                                         # (4, N_vis)
    z_cam  = p_cam[2]                                                       # (N_vis,)

    # 用 intrinsic 投影到像素
    p_x = (p_cam[0] * intrinsic[0, 0]) / p_cam[2] + intrinsic[0, 2]
    p_y = (p_cam[1] * intrinsic[1, 1]) / p_cam[2] + intrinsic[1, 2]
    p_xi = np.round(p_x).astype(np.int32)
    p_yi = np.round(p_y).astype(np.int32)

    # 应和 proj 里的 x_saved/y_saved 对应
    H, W = depth.shape
    valid_depth = (
        (p_xi >= CUT_BOUND) & (p_xi < W - CUT_BOUND) &
        (p_yi >= CUT_BOUND) & (p_yi < H - CUT_BOUND) &
        (z_cam > 0)
    )
    if not valid_depth.any():
        return fail("no_valid_depth_points")

    xi_v = p_xi[valid_depth]
    yi_v = p_yi[valid_depth]
    zc_v = z_cam[valid_depth]

    z_depth = depth[yi_v, xi_v]   # 从 depth 图取

    # 只保留 depth 图里有值（>0）的点
    has_depth = z_depth > 0
    if not has_depth.any():
        return fail("depth_all_zero")

    z_depth = z_depth[has_depth]
    zc_v    = zc_v[has_depth]

    abs_err = np.abs(z_depth - zc_v)

    res["mean_abs_err"]   = float(abs_err.mean())
    res["median_abs_err"] = float(np.median(abs_err))
    res["p90_abs_err"]    = float(np.percentile(abs_err, 90))
    res["p99_abs_err"]    = float(np.percentile(abs_err, 99))

    # depth-consistent ratio：|z_cam - z_depth| <= VIS_THR * z_cam
    consistent = abs_err <= VIS_THR * zc_v
    res["depth_consistent_ratio"] = float(consistent.mean())

    return res


def main():
    parser = argparse.ArgumentParser(description="层3：深度一致性检查")
    parser.add_argument("--data-root-3d", default=str(DATA_ROOT_3D))
    parser.add_argument("--data-root-2d", default=str(DATA_ROOT_2D))
    parser.add_argument("--npz-dir",      default=str(NPZ_DIR))
    parser.add_argument("--proj-dir",     default=str(PROJ_DIR))
    parser.add_argument("--split",        default="train")
    parser.add_argument("--max-samples",  type=int, default=50)
    parser.add_argument("--scene",        default=None)
    args = parser.parse_args()

    data_root_3d = Path(args.data_root_3d)
    data_root_2d = Path(args.data_root_2d)
    npz_dir      = Path(args.npz_dir)
    proj_dir     = Path(args.proj_dir)

    print("=" * 70)
    print(f"[层3] 深度一致性检查  split={args.split}  max={args.max_samples}")
    print(f"      visibility_threshold={VIS_THR}  img_dim={IMG_DIM}")
    print("=" * 70)

    intrinsic = build_intrinsic()

    samples = []
    scene_dirs = sorted(npz_dir.glob("scene*")) if args.scene is None \
        else [npz_dir / args.scene]
    for sd in scene_dirs:
        if not sd.is_dir():
            continue
        for npz_f in sorted(sd.glob("*_odise.npz")):
            frame_stem = npz_f.stem.replace("_odise", "")
            if (proj_dir / sd.name / f"{frame_stem}_proj.npz").exists():
                samples.append((sd.name, frame_stem))

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"待检查样本数: {len(samples)}\n")

    locs_cache = {}
    results = []
    for scene_name, frame_stem in tqdm(samples, desc="深度验证中"):
        r = check_one_sample(
            scene_name, frame_stem, args.split, intrinsic,
            data_root_3d, data_root_2d, proj_dir, locs_cache,
        )
        results.append(r)

    ok_results  = [r for r in results if r["status"] == "ok"]
    err_results = [r for r in results if r["status"] != "ok"]

    print("\n" + "=" * 70)
    print(f"[汇总] 总={len(results)}  可分析={len(ok_results)}  错误={len(err_results)}")

    if ok_results:
        def stats(arr, name):
            if not arr:
                return
            a = np.array(arr)
            print(f"  {name}: mean={a.mean():.4f}  p50={np.median(a):.4f}  "
                  f"p90={np.percentile(a,90):.4f}  p99={np.percentile(a,99):.4f}  "
                  f"max={a.max():.4f}")

        print("-" * 70)
        stats([r["mean_abs_err"]            for r in ok_results], "mean |z_err| (m)       ")
        stats([r["median_abs_err"]          for r in ok_results], "median |z_err| (m)     ")
        stats([r["p90_abs_err"]             for r in ok_results], "p90 |z_err| (m)        ")
        stats([r["depth_consistent_ratio"]  for r in ok_results], "depth-consistent ratio ")

        bad = [r for r in ok_results if (r["depth_consistent_ratio"] or 1.0) < 0.5]
        print(f"\ndepth-consistent ratio < 50% 的样本: {len(bad)}/{len(ok_results)}")
        for r in bad[:10]:
            print(f"  {r['scene']}/{r['frame']}: ratio={r['depth_consistent_ratio']:.3f}  "
                  f"mean_err={r['mean_abs_err']:.3f}m  p90={r['p90_abs_err']:.3f}m")

        avg_ratio = np.mean([r["depth_consistent_ratio"] for r in ok_results])
        print(f"\n平均 depth-consistent ratio = {avg_ratio:.4f}")
        if avg_ratio > 0.7:
            print("=> PASS: 深度一致性良好")
        else:
            print("=> FAIL: 深度一致性差，请检查 pose/depth/单位")

    if err_results:
        print(f"\n[错误样本] 前10条：")
        for r in err_results[:10]:
            print(f"  {r['scene']}/{r['frame']}: {r['status']}")

    print("\n[层3检查完成]")


if __name__ == "__main__":
    main()
