"""
K维索引对齐检查 - 纯数据验证，不需要任何模型权重

检查以下问题：
1. pixel_pooled 和 masks 是否 K 维对齐
   方法：从 pixel_pooled npz 和 odise_features npz 读同一帧的 masks，
   用 odise masks 对 pixel_pooled npz 里的 masks 做重叠验证（两套 masks 是否一样）

2. pixel_pooled[k] 和 masks[k] 是否对应
   方法：pixel_pooled npz 里同时有 masks 和 pixel_pooled，
   用 masks[k] 的形心位置做一致性检查（area、bbox、overlap）

3. masks / info / num_masks 内部一致性

4. odise_features 和 pixel_pooled 目录里同一帧的 masks 是否一致（K 数量、内容）
"""

import numpy as np
import os
from pathlib import Path

PIXEL_POOLED_DIR = Path("/home/sunl/work/mix/data/pixel_pooled")
ODISE_FEAT_DIR   = Path("/home/sunl/work/mix/data/odise_features")

def cosine_sim_matrix(A, B):
    """A: (M, D), B: (N, D) -> (M, N)"""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T

def mask_iou_matrix(masks_a, masks_b):
    """masks_a: (M, H, W) bool, masks_b: (N, H, W) bool -> (M, N)"""
    M = masks_a.shape[0]
    N = masks_b.shape[0]
    mat = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            inter = (masks_a[i] & masks_b[j]).sum()
            union = (masks_a[i] | masks_b[j]).sum()
            mat[i, j] = inter / union if union > 0 else 0.0
    return mat

def check_single_frame(scene, frame_id, verbose=True):
    """
    对单帧做完整检查，返回 dict of results
    """
    pp_path   = PIXEL_POOLED_DIR / scene / f"{frame_id}_odise.npz"
    od_path   = ODISE_FEAT_DIR   / scene / f"{frame_id}_odise.npz"

    results = {"scene": scene, "frame": frame_id, "issues": []}

    if not pp_path.exists():
        results["issues"].append(f"pixel_pooled npz 不存在: {pp_path}")
        return results
    if not od_path.exists():
        results["issues"].append(f"odise_features npz 不存在: {od_path}")
        return results

    pp  = np.load(pp_path,  allow_pickle=True)
    od  = np.load(od_path,  allow_pickle=True)

    pp_masks    = pp["masks"]          # (K, H, W)
    pp_pooled   = pp["pixel_pooled"]   # (K, 512)
    pp_embed    = pp["mask_embeddings"]# (K, 256)
    pp_num      = int(pp["num_masks"])
    pp_info     = pp["info"]

    od_masks    = od["masks"]          # (K', H, W)
    od_embed    = od["mask_embeddings"]# (K', 256)
    od_num      = int(od["num_masks"])
    od_info     = od["info"]

    K_pp = pp_masks.shape[0]
    K_od = od_masks.shape[0]

    if verbose:
        print(f"\n{'='*60}")
        print(f"场景: {scene}  帧: {frame_id}")
        print(f"  pixel_pooled npz: K={K_pp}, num_masks={pp_num}")
        print(f"  odise_features npz: K={K_od}, num_masks={od_num}")

    # -------------------------------------------------------
    # 检查 1: num_masks 和实际 K 是否一致
    # -------------------------------------------------------
    if K_pp != pp_num:
        results["issues"].append(f"[pp] masks.shape[0]={K_pp} != num_masks={pp_num}")
    if K_od != od_num:
        results["issues"].append(f"[od] masks.shape[0]={K_od} != num_masks={od_num}")
    if K_pp != K_od:
        results["issues"].append(f"K 不一致: pixel_pooled={K_pp}, odise_features={K_od}")

    results["K_pp"] = K_pp
    results["K_od"] = K_od

    # -------------------------------------------------------
    # 检查 2: info[k].area 和 masks[k].sum() 是否一致（pixel_pooled npz）
    # -------------------------------------------------------
    area_ok = 0
    area_bad = []
    for k in range(K_pp):
        mask_area = int(pp_masks[k].sum())
        info_area = int(pp_info[k]["area"]) if isinstance(pp_info[k], dict) else -1
        if info_area < 0:
            continue
        # 允许 20% 误差（mask 可能经过 resize）
        ratio = abs(mask_area - info_area) / max(info_area, 1)
        if ratio < 0.25:
            area_ok += 1
        else:
            area_bad.append((k, mask_area, info_area, ratio))

    results["area_check"] = {"ok": area_ok, "bad": len(area_bad), "detail": area_bad[:3]}
    if verbose:
        print(f"\n  [检查2] area 一致性 (pixel_pooled npz):")
        print(f"    一致: {area_ok}/{K_pp}")
        if area_bad:
            print(f"    异常 (前3): {area_bad[:3]}")

    # -------------------------------------------------------
    # 检查 3: pixel_pooled npz 和 odise_features npz 里的 masks 是否相同
    #         用 mask IoU 矩阵看对角线是否最大
    # -------------------------------------------------------
    if K_pp > 0 and K_od > 0 and K_pp == K_od:
        iou_mat = mask_iou_matrix(pp_masks, od_masks)
        diag    = np.diag(iou_mat)
        diag_is_max = np.all(np.argmax(iou_mat, axis=1) == np.arange(K_pp))
        avg_diag    = float(diag.mean())
        min_diag    = float(diag.min())

        results["mask_cross_iou"] = {
            "diag_is_max": bool(diag_is_max),
            "avg_diag_iou": avg_diag,
            "min_diag_iou": min_diag,
        }
        if verbose:
            print(f"\n  [检查3] pixel_pooled vs odise_features 的 masks IoU:")
            print(f"    对角线是否最大: {diag_is_max}")
            print(f"    对角线 IoU 均值: {avg_diag:.4f}  最小值: {min_diag:.4f}")
            if not diag_is_max:
                row_max_idx = np.argmax(iou_mat, axis=1)
                bad = [(i, int(row_max_idx[i]), float(iou_mat[i, row_max_idx[i]]), float(diag[i]))
                       for i in range(K_pp) if row_max_idx[i] != i]
                print(f"    错位的 k: {bad[:5]}")
                results["issues"].append(f"masks 跨目录错位: {len(bad)}/{K_pp} 个")
    elif K_pp != K_od:
        results["mask_cross_iou"] = None
        if verbose:
            print(f"\n  [检查3] 跳过（K 数量不一致）")

    # -------------------------------------------------------
    # 检查 4: pixel_pooled[k] 和 mask_embeddings[k] 的 K 维 cos sim
    #         如果 pixel_pooled 是从 pixel 用 masks 池化来的，
    #         期望 pixel_pooled[i] 和 mask_embeddings[i] 的相关性
    #         比 pixel_pooled[i] 和 mask_embeddings[j≠i] 更强
    #         （这是弱验证，因为不同模态，但严重错位时 off-diag 会明显更高）
    # -------------------------------------------------------
    # pixel_pooled (512D) vs mask_embeddings (256D) 维度不同，不能直接做 cos sim
    # 改为：用 pixel_pooled 自身做 K×K cos sim，看 mask 面积排序和 sim 结构是否合理
    if K_pp > 1:
        sim_pp = cosine_sim_matrix(pp_pooled.astype(np.float32),
                                   pp_pooled.astype(np.float32))  # (K, K)
        np.fill_diagonal(sim_pp, 0)
        max_offdiag = float(sim_pp.max())
        mean_offdiag = float(sim_pp.mean())

        # 用 mask_embeddings 自身也做一次
        sim_me = cosine_sim_matrix(pp_embed.astype(np.float32),
                                   pp_embed.astype(np.float32))  # (K, K)
        np.fill_diagonal(sim_me, 0)

        # 关键检查：pixel_pooled 的 argmax(非对角) 和 mask_embeddings 的 argmax 是否一致
        # 如果 K 维没有错位，两者的"最相似邻居"应该大致对应
        pp_neighbors  = np.argmax(sim_pp, axis=1)   # shape (K,)
        me_neighbors  = np.argmax(sim_me, axis=1)
        neighbor_agree = int((pp_neighbors == me_neighbors).sum())

        results["pp_vs_embed_sim"] = {
            "pp_max_offdiag": max_offdiag,
            "pp_mean_offdiag": mean_offdiag,
            "neighbor_agree": neighbor_agree,
            "neighbor_agree_ratio": neighbor_agree / K_pp,
        }
        if verbose:
            print(f"\n  [检查4] pixel_pooled 内部 cos sim 结构 (弱验证):")
            print(f"    pixel_pooled 非对角最大值: {max_offdiag:.4f}")
            print(f"    pixel_pooled 非对角均值:   {mean_offdiag:.4f}")
            print(f"    最相似邻居与 mask_embed 一致: {neighbor_agree}/{K_pp} ({neighbor_agree/K_pp:.2%}")

    # -------------------------------------------------------
    # 检查 5: pixel_pooled[k] 之间的 cos sim 矩阵（同一帧内自身）
    #         如果 K 维顺序打乱，这个矩阵本身不变，所以这里只看 norm 分布
    # -------------------------------------------------------
    norms = np.linalg.norm(pp_pooled.astype(np.float32), axis=1)
    results["pp_norms"] = {"mean": float(norms.mean()), "min": float(norms.min()), "max": float(norms.max())}
    if verbose:
        print(f"\n  [检查5] pixel_pooled L2 norm 分布:")
        print(f"    mean={norms.mean():.4f}  min={norms.min():.4f}  max={norms.max():.4f}")
        if norms.min() < 0.01:
            results["issues"].append(f"pixel_pooled 存在接近零向量: min_norm={norms.min():.4f}")

    pp.close()
    od.close()

    if verbose:
        if results["issues"]:
            print(f"\n  *** 发现问题: ***")
            for iss in results["issues"]:
                print(f"    - {iss}")
        else:
            print(f"\n  [OK] 未发现明显问题")

    return results


def run_batch_check(n_scenes=20, n_frames_per_scene=3):
    """对多个场景批量检查，汇总统计"""
    scenes = sorted(PIXEL_POOLED_DIR.iterdir())
    import random
    random.seed(42)
    sampled = random.sample(scenes, min(n_scenes, len(scenes)))

    all_results = []
    total_issues = 0
    mask_iou_ok = 0
    mask_iou_bad = 0
    area_ok_total = 0
    area_bad_total = 0

    for scene_dir in sampled:
        scene = scene_dir.name
        npzs = sorted(scene_dir.glob("*_odise.npz"))
        if not npzs:
            continue
        selected = random.sample(npzs, min(n_frames_per_scene, len(npzs)))
        for npz_path in selected:
            frame_id = npz_path.stem.replace("_odise", "")
            r = check_single_frame(scene, frame_id, verbose=False)
            all_results.append(r)
            total_issues += len(r["issues"])

            cross = r.get("mask_cross_iou")
            if cross is not None:
                if cross["diag_is_max"] and cross["avg_diag_iou"] > 0.95:
                    mask_iou_ok += 1
                else:
                    mask_iou_bad += 1

            ac = r.get("area_check", {})
            area_ok_total  += ac.get("ok", 0)
            area_bad_total += ac.get("bad", 0)

    print(f"\n{'='*60}")
    print(f"批量检查汇总 ({len(all_results)} 帧，{len(sampled)} 个场景)")
    print(f"{'='*60}")
    print(f"  发现问题的帧数: {sum(1 for r in all_results if r['issues'])}/{len(all_results)}")
    print(f"  总问题条数: {total_issues}")
    print(f"\n  [检查3] masks 跨目录对齐:")
    print(f"    对齐正常 (IoU对角线最大且均值>0.95): {mask_iou_ok}")
    print(f"    对齐异常: {mask_iou_bad}")
    print(f"\n  [检查2] area 一致性:")
    print(f"    一致: {area_ok_total}")
    print(f"    异常: {area_bad_total}")

    # 打印有问题的帧
    bad_frames = [r for r in all_results if r["issues"] or
                  (r.get("mask_cross_iou") and not r["mask_cross_iou"]["diag_is_max"])]
    if bad_frames:
        print(f"\n  有问题的帧:")
        for r in bad_frames[:10]:
            print(f"    {r['scene']}/{r['frame']}: {r['issues']}")
    else:
        print(f"\n  所有采样帧均无明显问题")

    # pixel_pooled vs mask_embeddings 弱验证汇总
    sims = [r["pp_vs_embed_sim"] for r in all_results if "pp_vs_embed_sim" in r]
    if sims:
        agree_ratios = [s["neighbor_agree_ratio"] for s in sims]
        max_offdiags = [s["pp_max_offdiag"] for s in sims]
        print(f"\n  [检查4] pixel_pooled 内部结构 (弱验证):")
        print(f"    最相似邻居与 mask_embed 一致率: {np.mean(agree_ratios):.2%} ± {np.std(agree_ratios):.2%}")
        print(f"    pixel_pooled 非对角 cos sim 最大值均值: {np.mean(max_offdiags):.4f}")

    return all_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        # 单帧详细检查：python check_k_alignment.py scene0000_00 0
        scene    = sys.argv[1]
        frame_id = sys.argv[2]
        check_single_frame(scene, frame_id, verbose=True)
    else:
        # 批量检查
        n_scenes = int(sys.argv[1]) if len(sys.argv) == 2 else 30
        run_batch_check(n_scenes=n_scenes, n_frames_per_scene=5)
