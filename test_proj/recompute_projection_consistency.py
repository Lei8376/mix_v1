"""
层2：重算一致性检查（最关键）

核心思路：
  用和 precompute_projections.py 完全相同的 point2img_mapper.compute_mapping()
  对同一 (scene, frame) 重新计算投影，和 _proj.npz 里保存的逐项精确对比。

检查指标：
  - visible_mask 完全匹配率
  - x/y exact match rate（在两者 visible 点的交集上）
  - mean/max |dx|, mean/max |dy|

判断标准：
  - 理论上预计算和重算应该 100% 一致
  - 任何不一致都是直接证据，说明预计算文件或读取方式有问题

用法：
  python test_proj/recompute_projection_consistency.py [--max-samples 50]
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.mapping_util import get_point2img_mapper
import torch


DATA_ROOT_3D = Path("/home/sunl/work/mix/data/scannet_3d")
DATA_ROOT_2D = Path("/home/sunl/work/mix/data/scannet_2d")
NPZ_DIR      = Path("/home/sunl/work/mix/data/pixel_pooled")
PROJ_DIR     = Path("/home/sunl/work/mix/data/scannet_projections")


def load_pth_locs(data_root_3d: Path, split: str, scene_name: str):
    pth = data_root_3d / split / f"{scene_name}_vh_clean_2.pth"
    if not pth.exists():
        pth = data_root_3d / split / f"{scene_name}.pth"
    if not pth.exists():
        return None
    try:
        data = torch.load(pth, map_location="cpu", weights_only=False)
        if isinstance(data, (list, tuple)):
            locs = data[0]
        else:
            locs = data.get("locs", data.get("coords"))
        if isinstance(locs, torch.Tensor):
            locs = locs.numpy()
        return locs.astype(np.float64)
    except Exception as e:
        print(f"  [ERROR] 无法加载 {pth}: {e}")
        return None


def check_one_sample(scene_name, frame_stem, split, mapper,
                     data_root_3d, data_root_2d, proj_dir,
                     locs_cache):
    """
    重算并对比投影，返回统计结果 dict。
    """
    res = {
        "scene": scene_name,
        "frame": frame_stem,
        "status": "ok",
        "visible_exact_match": None,
        "x_exact_match": None,
        "y_exact_match": None,
        "mean_dx": None,
        "mean_dy": None,
        "max_dx": None,
        "max_dy": None,
        "saved_n_vis": None,
        "rt_n_vis": None,
        "n_common_vis": None,
    }

    # 1. 加载保存的投影
    proj_path = proj_dir / scene_name / f"{frame_stem}_proj.npz"
    if not proj_path.exists():
        res["status"] = "no_proj_file"
        return res

    try:
        proj = np.load(proj_path)
        vis_saved = proj["visible_mask"].astype(bool)   # (N,)
        x_saved   = proj["x_label"].astype(np.int32)    # (N_vis,)
        y_saved   = proj["y_label"].astype(np.int32)    # (N_vis,)
        N_saved   = int(proj["num_points"])
    except Exception as e:
        res["status"] = f"proj_load_error: {e}"
        return res

    # 2. 加载 pose / depth
    pose_path  = data_root_2d / scene_name / "pose"  / f"{frame_stem}.txt"
    depth_path = data_root_2d / scene_name / "depth" / f"{frame_stem}.png"
    if not pose_path.exists() or not depth_path.exists():
        res["status"] = "missing_pose_or_depth"
        return res

    try:
        pose  = np.loadtxt(str(pose_path))
        depth = np.array(Image.open(str(depth_path)), dtype=np.float32) / 1000.0
    except Exception as e:
        res["status"] = f"pose_depth_load_error: {e}"
        return res

    # 3. 加载 locs
    if scene_name not in locs_cache:
        locs_cache[scene_name] = load_pth_locs(data_root_3d, split, scene_name)
    locs = locs_cache[scene_name]
    if locs is None:
        res["status"] = "no_pth"
        return res

    if locs.shape[0] != N_saved:
        res["status"] = f"N_mismatch: pth={locs.shape[0]} proj={N_saved}"
        return res

    # 4. 重算投影（与 precompute_projections.py 完全相同逻辑）
    try:
        mapping_rt = mapper.compute_mapping(pose, locs, depth)
        # mapping_rt: (N, 3)  col0=y  col1=x  col2=valid
        vis_rt = mapping_rt[:, 2].astype(bool)   # (N,)
        y_rt   = mapping_rt[vis_rt, 0].astype(np.int32)
        x_rt   = mapping_rt[vis_rt, 1].astype(np.int32)
    except Exception as e:
        res["status"] = f"recompute_error: {e}"
        return res

    res["saved_n_vis"] = int(vis_saved.sum())
    res["rt_n_vis"]    = int(vis_rt.sum())

    # 5. visible_mask 精确匹配率
    vis_match = (vis_saved == vis_rt).sum()
    res["visible_exact_match"] = float(vis_match) / max(len(vis_saved), 1)

    # 6. 在"两者都可见"的点上对比 x/y
    common_vis = vis_saved & vis_rt                        # (N,) bool
    n_common   = int(common_vis.sum())
    res["n_common_vis"] = n_common

    if n_common == 0:
        res["status"] = "no_common_visible"
        return res

    # 需要从全集索引中提取出对应的 x/y
    # vis_saved 里第 i 个 True 对应 x_saved[i]
    # vis_rt 里第 i 个 True 对应 x_rt[i]
    # 两者在 common_vis 位置都是 True
    saved_idx = np.zeros(len(vis_saved), dtype=np.int32)
    s_cnt = 0
    for i in range(len(vis_saved)):
        if vis_saved[i]:
            saved_idx[i] = s_cnt
            s_cnt += 1

    rt_idx = np.zeros(len(vis_rt), dtype=np.int32)
    r_cnt = 0
    for i in range(len(vis_rt)):
        if vis_rt[i]:
            rt_idx[i] = r_cnt
            r_cnt += 1

    common_positions = np.where(common_vis)[0]
    x_s = x_saved[saved_idx[common_positions]]
    y_s = y_saved[saved_idx[common_positions]]
    x_r = x_rt[rt_idx[common_positions]]
    y_r = y_rt[rt_idx[common_positions]]

    dx = np.abs(x_s.astype(np.float32) - x_r.astype(np.float32))
    dy = np.abs(y_s.astype(np.float32) - y_r.astype(np.float32))

    res["x_exact_match"] = float((dx == 0).sum()) / max(n_common, 1)
    res["y_exact_match"] = float((dy == 0).sum()) / max(n_common, 1)
    res["mean_dx"]       = float(dx.mean())
    res["mean_dy"]       = float(dy.mean())
    res["max_dx"]        = float(dx.max())
    res["max_dy"]        = float(dy.max())

    return res


def main():
    parser = argparse.ArgumentParser(description="层2：重算一致性检查")
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
    print(f"[层2] 重算一致性检查  split={args.split}  max={args.max_samples}")
    print("=" * 70)

    # 构建 mapper（和预计算时完全一样）
    mapper = get_point2img_mapper()
    print("投影器初始化完成\n")

    # 收集样本
    samples = []
    scene_dirs = sorted(npz_dir.glob("scene*")) if args.scene is None \
        else [npz_dir / args.scene]
    for sd in scene_dirs:
        if not sd.is_dir():
            continue
        for npz_f in sorted(sd.glob("*_odise.npz")):
            frame_stem = npz_f.stem.replace("_odise", "")
            proj_f = proj_dir / sd.name / f"{frame_stem}_proj.npz"
            if proj_f.exists():
                samples.append((sd.name, frame_stem))

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"待检查样本数: {len(samples)}\n")

    locs_cache = {}
    results = []
    for scene_name, frame_stem in tqdm(samples, desc="重算中"):
        r = check_one_sample(
            scene_name, frame_stem, args.split, mapper,
            data_root_3d, data_root_2d, proj_dir, locs_cache,
        )
        results.append(r)

    # ---- 汇总 ----
    ok_results = [r for r in results if r["status"] == "ok"]
    err_results = [r for r in results if r["status"] != "ok"]

    print("\n" + "=" * 70)
    print(f"[汇总] 总={len(results)}  可分析={len(ok_results)}  错误={len(err_results)}")

    if ok_results:
        vis_matches = [r["visible_exact_match"] for r in ok_results]
        x_matches   = [r["x_exact_match"]       for r in ok_results if r["x_exact_match"] is not None]
        y_matches   = [r["y_exact_match"]        for r in ok_results if r["y_exact_match"] is not None]
        mean_dxs    = [r["mean_dx"]              for r in ok_results if r["mean_dx"] is not None]
        mean_dys    = [r["mean_dy"]              for r in ok_results if r["mean_dy"] is not None]
        max_dxs     = [r["max_dx"]               for r in ok_results if r["max_dx"] is not None]
        max_dys     = [r["max_dy"]               for r in ok_results if r["max_dy"] is not None]

        def stats(arr, name):
            if not arr:
                return
            a = np.array(arr)
            print(f"  {name}: mean={a.mean():.4f}  min={a.min():.4f}  "
                  f"p10={np.percentile(a,10):.4f}  p90={np.percentile(a,90):.4f}  "
                  f"max={a.max():.4f}")

        print("-" * 70)
        stats(vis_matches, "visible_mask 匹配率  ")
        stats(x_matches,   "x exact match rate  ")
        stats(y_matches,   "y exact match rate  ")
        stats(mean_dxs,    "mean |dx|           ")
        stats(mean_dys,    "mean |dy|           ")
        stats(max_dxs,     "max  |dx|           ")
        stats(max_dys,     "max  |dy|           ")

        # 不一致样本
        bad = [r for r in ok_results
               if (r["visible_exact_match"] or 1.0) < 0.99
               or (r["x_exact_match"] or 1.0) < 0.99
               or (r["y_exact_match"] or 1.0) < 0.99]
        print(f"\n不一致样本（匹配率 <99%）: {len(bad)}/{len(ok_results)}")
        for r in bad[:10]:
            print(f"  {r['scene']}/{r['frame']}: "
                  f"vis={r['visible_exact_match']:.3f}  "
                  f"x={r['x_exact_match']:.3f}  y={r['y_exact_match']:.3f}  "
                  f"mean_dx={r['mean_dx']:.2f}  mean_dy={r['mean_dy']:.2f}")

    if err_results:
        print(f"\n[错误样本] 前10条：")
        for r in err_results[:10]:
            print(f"  {r['scene']}/{r['frame']}: {r['status']}")

    print("\n[层2检查完成]")

    # 判断总体是否通过
    if ok_results:
        avg_vis = np.mean([r["visible_exact_match"] for r in ok_results])
        avg_x   = np.mean([r["x_exact_match"] for r in ok_results if r["x_exact_match"] is not None])
        avg_y   = np.mean([r["y_exact_match"] for r in ok_results if r["y_exact_match"] is not None])
        if avg_vis > 0.99 and avg_x > 0.99 and avg_y > 0.99:
            print("=> PASS: 预计算投影文件与重算结果高度一致")
            return 0
        else:
            print(f"=> FAIL: 匹配率偏低 vis={avg_vis:.3f} x={avg_x:.3f} y={avg_y:.3f}")
            return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
