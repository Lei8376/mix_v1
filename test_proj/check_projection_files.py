"""
层1：文件级一致性检查

验证每个 (scene, frame) 的以下四项：
  1.1 同一帧文件是否齐全（RGB / depth / pose / odise.npz / proj.npz）
  1.2 num_points 与 .pth 点数是否一致
  1.3 visible_mask / x_label / y_label 长度是否一致
  1.4 x/y 坐标范围是否在 mask 分辨率内

输出：
  - 每项 pass/fail 统计
  - 失败样本列表（最多显示 20 条）
  - 汇总报告

用法：
  python test_proj/check_projection_files.py [--max-samples 200] [--split train]
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from glob import glob
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 数据路径（与 yaml 配置一致）
# ============================================================
DATA_ROOT_3D    = Path("/home/sunl/work/mix/data/scannet_3d")
DATA_ROOT_2D    = Path("/home/sunl/work/mix/data/scannet_2d")
NPZ_DIR         = Path("/home/sunl/work/mix/data/pixel_pooled")
PROJ_DIR        = Path("/home/sunl/work/mix/data/scannet_projections")

# ODISE mask 分辨率（在 precompute_projections 里 point2img_mapper 用 (320,240)）
MASK_W, MASK_H  = 320, 240


def load_pth_locs(pth_path: Path):
    """加载 .pth 文件，返回 locs numpy array"""
    try:
        data = torch.load(pth_path, map_location="cpu", weights_only=False)
        if isinstance(data, (list, tuple)):
            locs = data[0]
        else:
            locs = data.get("locs", data.get("coords"))
        if isinstance(locs, torch.Tensor):
            locs = locs.numpy()
        return locs
    except Exception as e:
        return None


def get_pth_path(data_root_3d: Path, split: str, scene_name: str) -> Path:
    p = data_root_3d / split / f"{scene_name}.pth"
    if not p.exists():
        p = data_root_3d / split / f"{scene_name}_vh_clean_2.pth"
    return p if p.exists() else None


def check_one_sample(scene_name: str, frame_stem: str, split: str,
                     npz_dir: Path, proj_dir: Path,
                     data_root_2d: Path, data_root_3d: Path,
                     pth_locs_cache: dict):
    """
    对一个 (scene, frame) 做层1四项检查。
    返回 dict: {pass: bool, checks: {name: bool|str}, ...}
    """
    result = {
        "scene": scene_name,
        "frame": frame_stem,
        "pass": True,
        "checks": {},
    }

    def fail(check_name, reason):
        result["checks"][check_name] = f"FAIL: {reason}"
        result["pass"] = False

    def ok(check_name, msg=""):
        result["checks"][check_name] = f"PASS" + (f" ({msg})" if msg else "")

    # ---- 1.1 文件齐全 ----
    odise_path = npz_dir / scene_name / f"{frame_stem}_odise.npz"
    proj_path  = proj_dir / scene_name / f"{frame_stem}_proj.npz"
    rgb_path   = data_root_2d / scene_name / "color" / f"{frame_stem}.jpg"
    depth_path = data_root_2d / scene_name / "depth" / f"{frame_stem}.png"
    pose_path  = data_root_2d / scene_name / "pose"  / f"{frame_stem}.txt"

    missing = []
    for name, p in [("odise.npz", odise_path), ("proj.npz", proj_path),
                    ("RGB", rgb_path), ("depth", depth_path), ("pose", pose_path)]:
        if not p.exists():
            missing.append(name)

    if missing:
        fail("1.1_files", f"缺失: {', '.join(missing)}")
    else:
        ok("1.1_files", "all 5 files exist")

    # 后续检查依赖 proj.npz，若缺则直接返回
    if not proj_path.exists():
        return result

    # ---- 加载 proj.npz ----
    try:
        proj = np.load(proj_path)
    except Exception as e:
        fail("proj_load", str(e))
        return result

    visible_mask = proj["visible_mask"]   # (N,) bool
    x_label      = proj["x_label"]        # (N_vis,)
    y_label      = proj["y_label"]        # (N_vis,)
    num_points   = int(proj["num_points"])

    # ---- 1.2 num_points 与 .pth 一致 ----
    pth_p = get_pth_path(data_root_3d, split, scene_name)
    if pth_p is None:
        fail("1.2_num_points", f"找不到 .pth 文件")
    else:
        if scene_name not in pth_locs_cache:
            locs = load_pth_locs(pth_p)
            pth_locs_cache[scene_name] = locs.shape[0] if locs is not None else -1
        pth_N = pth_locs_cache[scene_name]
        if pth_N == -1:
            fail("1.2_num_points", "无法加载 .pth")
        elif pth_N != num_points:
            fail("1.2_num_points", f"proj.num_points={num_points} vs pth.N={pth_N}")
        else:
            ok("1.2_num_points", f"N={num_points}")

    # ---- 1.3 长度一致性 ----
    n_vis = int(visible_mask.sum())
    lx, ly = len(x_label), len(y_label)
    if lx != n_vis or ly != n_vis:
        fail("1.3_length", f"visible_mask.sum()={n_vis}, len(x)={lx}, len(y)={ly}")
    else:
        ok("1.3_length", f"all={n_vis}")

    # ---- 1.4 坐标范围 ----
    if lx > 0:
        x_min, x_max = int(x_label.min()), int(x_label.max())
        y_min, y_max = int(y_label.min()), int(y_label.max())
        x_oob = int(((x_label < 0) | (x_label >= MASK_W)).sum())
        y_oob = int(((y_label < 0) | (y_label >= MASK_H)).sum())
        oob_total = x_oob + y_oob
        msg = f"x=[{x_min},{x_max}] y=[{y_min},{y_max}]"
        if oob_total > 0:
            fail("1.4_coord_range", f"越界 {oob_total}/{lx*2} 坐标。{msg}")
        else:
            ok("1.4_coord_range", msg)
    else:
        fail("1.4_coord_range", "x_label 长度为 0")

    return result


def main():
    parser = argparse.ArgumentParser(description="层1：文件级一致性检查")
    parser.add_argument("--data-root-3d", default=str(DATA_ROOT_3D))
    parser.add_argument("--data-root-2d", default=str(DATA_ROOT_2D))
    parser.add_argument("--npz-dir",      default=str(NPZ_DIR))
    parser.add_argument("--proj-dir",     default=str(PROJ_DIR))
    parser.add_argument("--split",        default="train")
    parser.add_argument("--max-samples",  type=int, default=None,
                        help="最多检查多少个样本（None=全部）")
    parser.add_argument("--scene",        default=None,
                        help="只检查特定场景（调试用）")
    args = parser.parse_args()

    data_root_3d = Path(args.data_root_3d)
    data_root_2d = Path(args.data_root_2d)
    npz_dir      = Path(args.npz_dir)
    proj_dir     = Path(args.proj_dir)

    print("=" * 70)
    print(f"[层1] 文件级一致性检查  split={args.split}")
    print("=" * 70)

    # 收集所有样本 (scene, frame)
    samples = []
    scene_dirs = sorted(npz_dir.glob("scene*")) if args.scene is None \
        else [npz_dir / args.scene]
    for sd in scene_dirs:
        if not sd.is_dir():
            continue
        scene_name = sd.name
        for npz_f in sorted(sd.glob("*_odise.npz")):
            frame_stem = npz_f.stem.replace("_odise", "")
            samples.append((scene_name, frame_stem))

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"待检查样本数: {len(samples)}\n")

    pth_locs_cache = {}
    results = []
    for scene_name, frame_stem in tqdm(samples, desc="检查中"):
        r = check_one_sample(
            scene_name, frame_stem, args.split,
            npz_dir, proj_dir, data_root_2d, data_root_3d,
            pth_locs_cache,
        )
        results.append(r)

    # ---- 汇总统计 ----
    total       = len(results)
    n_pass      = sum(1 for r in results if r["pass"])
    n_fail      = total - n_pass

    check_names = ["1.1_files", "1.2_num_points", "1.3_length", "1.4_coord_range"]
    check_fail  = defaultdict(int)
    for r in results:
        for cn in check_names:
            v = r["checks"].get(cn, "")
            if isinstance(v, str) and v.startswith("FAIL"):
                check_fail[cn] += 1

    print("\n" + "=" * 70)
    print(f"[汇总] 总样本={total}  通过={n_pass}  失败={n_fail}  "
          f"通过率={n_pass/max(total,1)*100:.1f}%")
    print("-" * 70)
    for cn in check_names:
        f = check_fail[cn]
        print(f"  {cn}: 失败={f}/{total}  ({f/max(total,1)*100:.1f}%)")
    print("=" * 70)

    # ---- 打印失败样本（最多 20 条） ----
    failed = [r for r in results if not r["pass"]]
    if failed:
        print(f"\n[失败样本] 前 {min(20, len(failed))} 条：")
        for r in failed[:20]:
            bad = {k: v for k, v in r["checks"].items() if "FAIL" in str(v)}
            print(f"  {r['scene']}/{r['frame']}: {bad}")

    print("\n[层1检查完成]")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
