"""
验证 collate 后 mask_valid / padding 是否正确（纯 numpy，不需要 torch/模型）

模拟 dataset collate 把多帧 pad 到 K_max，检查：
1. valid[:K] 全 True
2. valid[K:] 全 False（padding 槽位）
3. pixel_pooled padding 槽位全零
4. masks padding 槽位全零
"""
import numpy as np
import random
from pathlib import Path

PIXEL_POOLED_DIR = Path("/home/sunl/work/mix/data/pixel_pooled")

random.seed(42)

def load_frame(npz_path):
    f = np.load(npz_path, allow_pickle=True)
    K = int(f["num_masks"])
    pp    = f["pixel_pooled"].astype(np.float32)      # (K, 512)
    me    = f["mask_embeddings"].astype(np.float32)   # (K, 256)
    masks = f["masks"].astype(np.float32)             # (K, H, W)
    f.close()
    return {"pixel_pooled": pp, "mask_embeddings": me, "masks": masks, "K": K}

def collate_numpy(frames):
    K_max = max(f["K"] for f in frames)
    B     = len(frames)
    Cp    = frames[0]["pixel_pooled"].shape[1]
    Cm    = frames[0]["mask_embeddings"].shape[1]
    H, W  = frames[0]["masks"].shape[1], frames[0]["masks"].shape[2]

    pp_batch    = np.zeros((B, K_max, Cp),    dtype=np.float32)
    me_batch    = np.zeros((B, K_max, Cm),    dtype=np.float32)
    mask_batch  = np.zeros((B, K_max, H, W),  dtype=np.float32)
    valid_batch = np.zeros((B, K_max),        dtype=bool)

    for b, f in enumerate(frames):
        K = f["K"]
        pp_batch[b,   :K] = f["pixel_pooled"]
        me_batch[b,   :K] = f["mask_embeddings"]
        mask_batch[b, :K] = f["masks"]
        valid_batch[b,:K] = True

    return pp_batch, me_batch, mask_batch, valid_batch

# ---------- 批量检查 ----------
all_scenes = sorted(PIXEL_POOLED_DIR.iterdir())
sampled    = random.sample(all_scenes, min(40, len(all_scenes)))

stats  = {"total": 0, "valid_ok": 0, "pad_valid_ok": 0,
          "pp_pad_zero_ok": 0, "mask_pad_zero_ok": 0}
issues = []

for scene_dir in sampled:
    npzs = sorted(scene_dir.glob("*_odise.npz"))
    if len(npzs) < 2:
        continue
    selected = random.sample(npzs, min(6, len(npzs)))
    frames   = [load_frame(p) for p in selected]

    pp, me, masks, valid = collate_numpy(frames)
    B, K_max, _ = pp.shape

    for b, f in enumerate(frames):
        K = f["K"]
        stats["total"] += 1

        # 1. valid[:K] 全 True
        if valid[b, :K].all():
            stats["valid_ok"] += 1
        else:
            n_false = (~valid[b, :K]).sum()
            issues.append(f"{scene_dir.name}: valid[:K] 有 {n_false} 个 False，K={K}")

        # 2. valid[K:] 全 False
        if K < K_max:
            if not valid[b, K:].any():
                stats["pad_valid_ok"] += 1
            else:
                n_true = valid[b, K:].sum()
                issues.append(f"{scene_dir.name}: valid[K:] 有 {n_true} 个 True，K={K}, K_max={K_max}")
        else:
            stats["pad_valid_ok"] += 1

        # 3. pixel_pooled padding 槽位全零
        if K < K_max:
            max_pad = np.abs(pp[b, K:]).max()
            if max_pad < 1e-6:
                stats["pp_pad_zero_ok"] += 1
            else:
                issues.append(f"{scene_dir.name}: pixel_pooled pad 槽不为零 max={max_pad:.4f}, K={K}")
        else:
            stats["pp_pad_zero_ok"] += 1

        # 4. masks padding 槽位全零
        if K < K_max:
            max_pad_m = np.abs(masks[b, K:]).max()
            if max_pad_m < 1e-6:
                stats["mask_pad_zero_ok"] += 1
            else:
                issues.append(f"{scene_dir.name}: masks pad 槽不为零 max={max_pad_m:.4f}, K={K}")
        else:
            stats["mask_pad_zero_ok"] += 1

print("=" * 60)
print("collate padding / mask_valid 检查")
print("=" * 60)
print(f"  采样帧数: {stats['total']}")
print(f"  valid[:K] 全 True:          {stats['valid_ok']}/{stats['total']}")
print(f"  valid[K:] 全 False:         {stats['pad_valid_ok']}/{stats['total']}")
print(f"  pixel_pooled padding 全零:  {stats['pp_pad_zero_ok']}/{stats['total']}")
print(f"  masks padding 全零:         {stats['mask_pad_zero_ok']}/{stats['total']}")

if issues:
    print(f"\n  *** 发现问题 ({len(issues)} 条): ***")
    for iss in issues[:10]:
        print(f"    - {iss}")
else:
    print(f"\n  [OK] 所有 collate padding 逻辑正确")
