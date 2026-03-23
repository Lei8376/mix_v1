"""
层4：dataset / collate 一致性检查

验证 _load_3d_with_precomputed_projection() 和 open_vocab_collate_v2()
没有打乱点与 x/y 的对应关系。

检查内容：
  4.1 dataset __getitem__ 层面：
      - len(coords_3d) == len(x_label) == len(y_label)
      - len(coords_3d) == visible_mask.sum()（原始 proj.npz 里的可见点数）
      - binary_label_3d 长度一致

  4.2 collate 层面（取一个小批次）：
      - 每个 batch item 通过 batch_indices 解析回来后，点数与原始 item 一致
      - x_label[batch_idx==b] 的长度等于该 item 的点数
      - mask_valid 前 k 个 True，padding 全 False
      - 坐标范围未越界（同层1检查）

用法：
  python test_proj/check_dataset_projection_pipeline.py [--max-samples 20] [--batch-size 4]
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

DATA_ROOT_3D  = Path("/home/sunl/work/mix/data/scannet_3d")
DATA_ROOT_2D  = Path("/home/sunl/work/mix/data/scannet_2d")
NPZ_DIR       = Path("/home/sunl/work/mix/data/pixel_pooled")
PROJ_DIR      = Path("/home/sunl/work/mix/data/scannet_projections")
DATA_CFG_PATH = Path("/home/sunl/work/mix/config/data_scannet_3d.yaml")

MASK_W, MASK_H = 320, 240


# ============================================================
# 4.1 dataset __getitem__ 检查
# ============================================================

def check_dataset_item(scene_name: str, frame_stem: str,
                       data_root_3d: Path, split: str,
                       proj_dir: Path, npz_dir: Path):
    """
    直接调用 _load_3d_with_precomputed_projection 并检查返回值。
    同时和 proj.npz 里的原始 visible_mask 做比对。
    """
    from dataset.open_vocab_dataset_v2 import _load_3d_with_precomputed_projection

    # 调用 dataset 加载函数
    try:
        out = _load_3d_with_precomputed_projection(
            data_root=data_root_3d,
            split=split,
            scene_name=scene_name,
            projection_dir=proj_dir,
            frame_stem=frame_stem,
        )
    except Exception as e:
        return {"status": f"load_error: {e}"}

    if out is None:
        return {"status": "returned_None"}

    checks = {}
    pass_all = True

    # 基本存在性
    for key in ["coords_3d", "feat_3d", "x_label", "y_label", "binary_label_3d"]:
        if key not in out:
            checks[key] = f"FAIL: missing key"
            pass_all = False
        else:
            checks[key] = "PASS"

    if not pass_all:
        return {"status": "missing_keys", "checks": checks}

    N        = out["coords_3d"].shape[0]
    lx       = len(out["x_label"])
    ly       = len(out["y_label"])
    n_label  = len(out["binary_label_3d"])

    # 长度一致性
    if not (N == lx == ly == n_label):
        checks["length_consistency"] = (
            f"FAIL: coords_3d={N} x={lx} y={ly} label={n_label}"
        )
        pass_all = False
    else:
        checks["length_consistency"] = f"PASS (N={N})"

    # 与 proj.npz 里 visible_mask 比对
    proj_path = proj_dir / scene_name / f"{frame_stem}_proj.npz"
    if proj_path.exists():
        proj_data = np.load(proj_path)
        vis_sum = int(proj_data["visible_mask"].sum())
        if N != vis_sum:
            checks["vs_proj_visible"] = (
                f"FAIL: dataset N={N} != proj visible_mask.sum()={vis_sum}"
            )
            pass_all = False
        else:
            checks["vs_proj_visible"] = f"PASS (N={N})"
    else:
        checks["vs_proj_visible"] = "SKIP (no proj file)"

    # 坐标范围
    x = out["x_label"].numpy()
    y = out["y_label"].numpy()
    x_oob = int(((x < 0) | (x >= MASK_W)).sum())
    y_oob = int(((y < 0) | (y >= MASK_H)).sum())
    if x_oob + y_oob > 0:
        checks["coord_range"] = f"FAIL: x_oob={x_oob} y_oob={y_oob}"
        pass_all = False
    else:
        checks["coord_range"] = f"PASS x=[{x.min()},{x.max()}] y=[{y.min()},{y.max()}]"

    return {
        "status": "ok",
        "pass": pass_all,
        "N": N,
        "checks": checks,
    }


# ============================================================
# 4.2 collate 层检查
# ============================================================

def check_collate_batch(batch_items: list):
    """
    对一批 dataset __getitem__ 的结果运行 open_vocab_collate_v2，
    检查 collate 后的一致性。
    """
    from dataset.open_vocab_dataset_v2 import open_vocab_collate_v2

    checks = {}
    pass_all = True

    # 记录 collate 前每个 item 的点数
    pre_Ns = [item["coords_3d"].shape[0] for item in batch_items]

    # 给每个 item 补充 npz 需要的字段（pixel_pooled / masks / mask_embeddings / mask_valid）
    # 这里用 dummy 数据（只测坐标一致性，不测 2D 特征）
    K_dummy  = 4
    H_dummy, W_dummy = MASK_H, MASK_W
    padded_items = []
    for item in batch_items:
        it = dict(item)
        it["pixel_pooled"]    = torch.zeros(K_dummy, 512)
        it["masks"]           = torch.zeros(K_dummy, H_dummy, W_dummy)
        it["mask_embeddings"] = torch.zeros(K_dummy, 256)
        it["mask_valid"]      = torch.ones(K_dummy, dtype=torch.bool)
        padded_items.append(it)

    try:
        batch = open_vocab_collate_v2(padded_items)
    except Exception as e:
        checks["collate_run"] = f"FAIL: {e}"
        return {"pass": False, "checks": checks}

    checks["collate_run"] = "PASS"

    # 解析 batch_indices 还原各 item
    # ori_coords_3d 第一列是 batch index（由 collate 填入）
    batch_indices = batch["ori_coords_3d"][:, 0]
    B = len(padded_items)

    for b in range(B):
        pt_mask = (batch_indices == b)
        post_N = int(pt_mask.sum())
        pre_N  = pre_Ns[b]

        lx = len(batch["x_label"][pt_mask])
        ly = len(batch["y_label"][pt_mask])

        if post_N != pre_N:
            checks[f"item{b}_N"] = f"FAIL: pre={pre_N} post={post_N}"
            pass_all = False
        else:
            checks[f"item{b}_N"] = f"PASS (N={post_N})"

        if lx != post_N or ly != post_N:
            checks[f"item{b}_xy"] = f"FAIL: N={post_N} lx={lx} ly={ly}"
            pass_all = False
        else:
            checks[f"item{b}_xy"] = f"PASS"

    # mask_valid padding 检查
    mv = batch["mask_valid"]   # (B, K_max)
    for b in range(B):
        k_real = K_dummy
        k_max  = mv.shape[1]
        # 前 k_real 应全 True，其余全 False
        first_k = mv[b, :k_real]
        rest    = mv[b, k_real:]
        if not first_k.all():
            checks[f"item{b}_mask_valid"] = "FAIL: first k not all True"
            pass_all = False
        elif rest.any():
            checks[f"item{b}_mask_valid"] = "FAIL: padding not all False"
            pass_all = False
        else:
            checks[f"item{b}_mask_valid"] = "PASS"

    # 检查 inds_reconstruct offset 是否只作用于 inds，不影响 x/y
    # inds_reconstruct 应当是累积偏移后的全局索引，x/y 不应有相同偏移
    if len(padded_items) > 1:
        x0 = batch["x_label"][batch_indices == 0]
        x1 = batch["x_label"][batch_indices == 1]
        # 坐标不应因 collate offset 变化（offset 只加到 inds_reconstruct）
        x_ok = (x0.max() < MASK_W) and (x1.max() < MASK_W)
        checks["xy_no_offset_contamination"] = "PASS" if x_ok else "FAIL"
        if not x_ok:
            pass_all = False

    return {"pass": pass_all, "checks": checks}


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="层4：dataset/collate 一致性检查")
    parser.add_argument("--data-root-3d", default=str(DATA_ROOT_3D))
    parser.add_argument("--npz-dir",      default=str(NPZ_DIR))
    parser.add_argument("--proj-dir",     default=str(PROJ_DIR))
    parser.add_argument("--split",        default="train")
    parser.add_argument("--max-samples",  type=int, default=20)
    parser.add_argument("--batch-size",   type=int, default=4,
                        help="collate 检查时的批大小")
    parser.add_argument("--scene",        default=None)
    args = parser.parse_args()

    data_root_3d = Path(args.data_root_3d)
    npz_dir      = Path(args.npz_dir)
    proj_dir     = Path(args.proj_dir)

    print("=" * 70)
    print(f"[层4] dataset/collate 一致性检查  split={args.split}  max={args.max_samples}")
    print("=" * 70)

    # 收集样本
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

    # ---------- 4.1 dataset __getitem__ ----------
    print("--- 4.1 dataset __getitem__ 检查 ---")
    item_results = []
    loaded_items = []

    for scene_name, frame_stem in tqdm(samples, desc="dataset 加载"):
        r = check_dataset_item(
            scene_name, frame_stem, data_root_3d, args.split,
            proj_dir, npz_dir,
        )
        r["scene"] = scene_name
        r["frame"] = frame_stem
        item_results.append(r)

        # 保留成功加载的 item 供 collate 测试
        if r.get("status") == "ok" and r.get("pass", False):
            # 重新加载一次得到真实 item（含所有字段）
            try:
                from dataset.open_vocab_dataset_v2 import _load_3d_with_precomputed_projection
                item = _load_3d_with_precomputed_projection(
                    data_root=data_root_3d, split=args.split,
                    scene_name=scene_name, projection_dir=proj_dir,
                    frame_stem=frame_stem,
                )
                if item is not None:
                    loaded_items.append(item)
            except Exception:
                pass

    ok_items   = [r for r in item_results if r.get("status") == "ok" and r.get("pass", False)]
    fail_items = [r for r in item_results if r.get("status") != "ok" or not r.get("pass", False)]

    print(f"\n4.1 结果: 通过={len(ok_items)}  失败={len(fail_items)}  "
          f"通过率={len(ok_items)/max(len(item_results),1)*100:.1f}%")
    if fail_items:
        print("失败样本（前10）：")
        for r in fail_items[:10]:
            print(f"  {r.get('scene')}/{r.get('frame')}: "
                  f"status={r.get('status')}  checks={r.get('checks', {})}")

    # ---------- 4.2 collate 检查 ----------
    print(f"\n--- 4.2 collate 检查（batch_size={args.batch_size}）---")
    if len(loaded_items) < args.batch_size:
        print(f"⚠️  可用 item 数（{len(loaded_items)}）少于 batch_size，跳过 collate 检查")
    else:
        batch_results = []
        step = args.batch_size
        for i in range(0, min(len(loaded_items), args.max_samples), step):
            batch_slice = loaded_items[i: i + step]
            if len(batch_slice) < 2:
                continue
            cr = check_collate_batch(batch_slice)
            batch_results.append(cr)

        n_pass = sum(1 for r in batch_results if r["pass"])
        n_fail = len(batch_results) - n_pass
        print(f"4.2 结果: 通过={n_pass}  失败={n_fail}  "
              f"通过率={n_pass/max(len(batch_results),1)*100:.1f}%")

        for r in batch_results:
            if not r["pass"]:
                print("  失败 batch 检查详情：")
                for k, v in r["checks"].items():
                    if "FAIL" in str(v):
                        print(f"    {k}: {v}")

    print("\n[层4检查完成]")


if __name__ == "__main__":
    main()
