"""
可视化失败案例

对层2/层3发现的有问题样本，生成可视化图片：
  - 左：RGB 图 + 投影点（保存的坐标）
  - 中：RGB 图 + 投影点（重算坐标）
  - 右：depth 图 + 可见点标注（颜色=深度误差）

同时支持"正常样本"对比，随机抽取 5 个正常和 5 个异常。

用法：
  python test_proj/render_projection_failure_cases.py \
      --output-dir /home/sunl/work/mix/test_proj/vis_output \
      [--max-samples 10]
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.mapping_util import get_point2img_mapper
from utils.fusion_util import make_intrinsic, adjust_intrinsic

DATA_ROOT_3D = Path("/home/sunl/work/mix/data/scannet_3d")
DATA_ROOT_2D = Path("/home/sunl/work/mix/data/scannet_2d")
NPZ_DIR      = Path("/home/sunl/work/mix/data/pixel_pooled")
PROJ_DIR     = Path("/home/sunl/work/mix/data/scannet_projections")

IMG_DIM  = (320, 240)
FX, FY   = 577.870605, 577.870605
MX, MY   = 319.5, 239.5
VIS_THR  = 0.25
CUT_BOUND = 10


def build_intrinsic():
    intr = make_intrinsic(fx=FX, fy=FY, mx=MX, my=MY)
    intr = adjust_intrinsic(intr, [640, 480], list(IMG_DIM))
    return intr


def load_pth_locs(data_root_3d: Path, split: str, scene_name: str):
    for suffix in [f"{scene_name}_vh_clean_2.pth", f"{scene_name}.pth"]:
        pth = data_root_3d / split / suffix
        if pth.exists():
            data = torch.load(pth, map_location="cpu", weights_only=False)
            locs = data[0] if isinstance(data, (list, tuple)) else data.get("locs", data.get("coords"))
            if isinstance(locs, torch.Tensor):
                locs = locs.numpy()
            return locs.astype(np.float64)
    return None


def render_one_sample(scene_name, frame_stem, split,
                      data_root_3d, data_root_2d, npz_dir, proj_dir,
                      mapper, intrinsic, output_dir: Path):
    """
    生成一张 3 列对比图：
      col0 = RGB + saved 投影点（蓝色）
      col1 = RGB + recomputed 投影点（橙色）
      col2 = depth 图 + 深度误差热图（颜色=|z_err|）
    """
    # 加载
    proj_path  = proj_dir  / scene_name / f"{frame_stem}_proj.npz"
    rgb_path   = data_root_2d / scene_name / "color" / f"{frame_stem}.jpg"
    depth_path = data_root_2d / scene_name / "depth" / f"{frame_stem}.png"
    pose_path  = data_root_2d / scene_name / "pose"  / f"{frame_stem}.txt"
    odise_path = npz_dir / scene_name / f"{frame_stem}_odise.npz"

    missing = [p for p in [proj_path, rgb_path, depth_path, pose_path] if not p.exists()]
    if missing:
        print(f"  跳过 {scene_name}/{frame_stem}：缺少 {[p.name for p in missing]}")
        return False

    proj    = np.load(proj_path)
    vis_s   = proj["visible_mask"].astype(bool)
    x_s     = proj["x_label"].astype(np.int32)
    y_s     = proj["y_label"].astype(np.int32)

    rgb   = np.array(Image.open(rgb_path).resize(IMG_DIM, Image.BILINEAR))
    depth = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0
    pose  = np.loadtxt(str(pose_path))

    N_saved = int(proj["num_points"])
    locs = load_pth_locs(data_root_3d, split, scene_name)
    if locs is None or locs.shape[0] != N_saved:
        print(f"  跳过 {scene_name}/{frame_stem}：locs 不可用或点数不一致")
        return False

    # 重算投影
    mapping_rt = mapper.compute_mapping(pose, locs, depth)
    vis_rt  = mapping_rt[:, 2].astype(bool)
    y_rt_all = mapping_rt[vis_rt, 0].astype(np.int32)
    x_rt_all = mapping_rt[vis_rt, 1].astype(np.int32)

    # 深度误差（在 saved 可见点上）
    world_to_cam = np.linalg.inv(pose)
    vis_locs = locs[vis_s]
    n_vis = vis_locs.shape[0]
    H_d, W_d = depth.shape

    if n_vis > 0:
        locs_h = np.concatenate([vis_locs, np.ones((n_vis, 1))], axis=1).T
        p_cam  = world_to_cam @ locs_h
        z_cam  = p_cam[2]

        p_x = (p_cam[0] * intrinsic[0, 0]) / np.maximum(p_cam[2], 1e-6) + intrinsic[0, 2]
        p_y = (p_cam[1] * intrinsic[1, 1]) / np.maximum(p_cam[2], 1e-6) + intrinsic[1, 2]
        p_xi = np.round(p_x).astype(np.int32)
        p_yi = np.round(p_y).astype(np.int32)

        in_range = (
            (p_xi >= 0) & (p_xi < W_d) &
            (p_yi >= 0) & (p_yi < H_d) &
            (z_cam > 0)
        )
        xi_v = p_xi[in_range].clip(0, W_d - 1)
        yi_v = p_yi[in_range].clip(0, H_d - 1)
        z_d  = depth[yi_v, xi_v]
        zc_v = z_cam[in_range]
        z_err = np.abs(z_d - zc_v)
    else:
        xi_v = yi_v = z_err = np.array([], dtype=np.float32)

    # ---- 画图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{scene_name} / {frame_stem}", fontsize=11)

    # col0: saved 投影点
    ax = axes[0]
    ax.imshow(rgb)
    if len(x_s) > 0:
        n_plot = min(len(x_s), 2000)
        idx = np.random.choice(len(x_s), n_plot, replace=False)
        ax.scatter(x_s[idx], y_s[idx], c="cyan", s=1, alpha=0.6, linewidths=0)
    ax.set_title(f"Saved proj ({len(x_s)} vis pts)")
    ax.axis("off")

    # col1: 重算投影点
    ax = axes[1]
    ax.imshow(rgb)
    if len(x_rt_all) > 0:
        n_plot = min(len(x_rt_all), 2000)
        idx = np.random.choice(len(x_rt_all), n_plot, replace=False)
        ax.scatter(x_rt_all[idx], y_rt_all[idx], c="orange", s=1, alpha=0.6, linewidths=0)
    ax.set_title(f"Recomputed proj ({len(x_rt_all)} vis pts)")
    ax.axis("off")

    # col2: depth 图 + 误差热图
    ax = axes[2]
    depth_disp = depth.copy()
    depth_disp[depth_disp == 0] = np.nan
    ax.imshow(depth_disp, cmap="gray")
    if len(xi_v) > 0:
        n_plot = min(len(xi_v), 2000)
        idx = np.random.choice(len(xi_v), n_plot, replace=False)
        sc = ax.scatter(xi_v[idx], yi_v[idx],
                        c=z_err[idx], cmap="hot_r", s=2,
                        vmin=0, vmax=0.5, linewidths=0, alpha=0.8)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="|z_err| (m)")
    if len(z_err) > 0:
        consistent = float((z_err <= VIS_THR * zc_v).mean())
        ax.set_title(f"Depth err  depth-consistent={consistent:.2f}")
    else:
        ax.set_title("Depth err (no valid points)")
    ax.axis("off")

    plt.tight_layout()
    out_path = output_dir / f"{scene_name}_{frame_stem}.png"
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="可视化投影失败案例")
    parser.add_argument("--data-root-3d",  default=str(DATA_ROOT_3D))
    parser.add_argument("--data-root-2d",  default=str(DATA_ROOT_2D))
    parser.add_argument("--npz-dir",       default=str(NPZ_DIR))
    parser.add_argument("--proj-dir",      default=str(PROJ_DIR))
    parser.add_argument("--split",         default="train")
    parser.add_argument("--output-dir",    default="/home/sunl/work/mix/test_proj/vis_output")
    parser.add_argument("--max-samples",   type=int, default=10,
                        help="最多可视化多少个样本（随机抽取）")
    parser.add_argument("--scene",         default=None)
    args = parser.parse_args()

    data_root_3d = Path(args.data_root_3d)
    data_root_2d = Path(args.data_root_2d)
    npz_dir      = Path(args.npz_dir)
    proj_dir     = Path(args.proj_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[可视化] 投影失败案例  output={output_dir}")
    print("=" * 70)

    mapper    = get_point2img_mapper()
    intrinsic = build_intrinsic()

    # 收集所有可用样本
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

    if not samples:
        print("未找到可用样本，请检查路径")
        return

    # 随机抽取
    n_render = min(args.max_samples, len(samples))
    chosen   = random.sample(samples, n_render)
    print(f"随机抽取 {n_render} 个样本进行可视化\n")

    saved_count = 0
    for scene_name, frame_stem in chosen:
        ok = render_one_sample(
            scene_name, frame_stem, args.split,
            data_root_3d, data_root_2d, npz_dir, proj_dir,
            mapper, intrinsic, output_dir,
        )
        if ok:
            saved_count += 1
            print(f"  保存: {output_dir}/{scene_name}_{frame_stem}.png")

    print(f"\n完成，共保存 {saved_count} 张图到 {output_dir}/")
    print("[可视化完成]")


if __name__ == "__main__":
    main()
