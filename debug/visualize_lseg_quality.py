#!/usr/bin/env python3
"""
Visualize LSeg pooled embedding quality against CLIP text features.

Three-column layout per frame:
  Col 0  ODISE  - mask overlay with category name + ODISE score
  Col 1  Norm   - LSeg pooled L2 norm (green=healthy, red=low)
  Col 2  Semantic - LSeg top-1 CLIP class and cosine similarity

Usage:
    cd /home/sunl/work/mix

    # Use ScanNet-200 (matches ODISE label set, recommended)
    python diagnostic_tools/visualize_lseg_quality.py \\
        --label-set 200 --num-scenes 3 --num-frames 4 \\
        --output-dir diagnostic_tools/lseg_vis_output

    # Use ScanNet-20
    python diagnostic_tools/visualize_lseg_quality.py \\
        --label-set 20 --scene scene0000_00 --num-frames 6

    # Force re-generate text feature cache
    python diagnostic_tools/visualize_lseg_quality.py \\
        --label-set 200 --rebuild-cache
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
PIXEL_POOLED_DIR = PROJECT_ROOT / "data" / "pixel_pooled"
ODISE_DIR        = PROJECT_ROOT / "data" / "odise_features"
IMAGE_DIR        = PROJECT_ROOT / "data" / "scannet_2d"
DIAG_DIR         = PROJECT_ROOT / "diagnostic_tools"
LABEL_MODULE     = PROJECT_ROOT / "ODISE" / "scannet_label_constant.py"


# ── Label loading ─────────────────────────────────────────────────────────
def load_label_list(label_set: int) -> list:
    """Import class names from ODISE/scannet_label_constant.py."""
    if str(LABEL_MODULE.parent) not in sys.path:
        sys.path.insert(0, str(LABEL_MODULE.parent))
    import scannet_label_constant as slc
    if label_set == 20:
        return list(slc.SCANNET_LABELS_20)
    elif label_set == 200:
        return list(slc.SCANNET_LABELS_200)
    else:
        raise ValueError(f"Unsupported label_set={label_set}. Use 20 or 200.")


# ── CLIP text feature cache ───────────────────────────────────────────────
def _cache_path(label_set: int) -> Path:
    return DIAG_DIR / f"scannet{label_set}_clip_vitb32_text_feats.npy"


def build_text_features(label_set: int, class_names: list,
                        force_rebuild: bool = False) -> np.ndarray:
    """
    Encode class_names with CLIP ViT-B/32 (same backbone as LSeg).
    Result is cached to disk; set force_rebuild=True to regenerate.
    Returns float32 array of shape (C, 512), L2-normalised.
    """
    cache = _cache_path(label_set)
    if cache.exists() and not force_rebuild:
        feats = np.load(cache).astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        print(f"Loaded cached text features: {cache}  shape={feats.shape}")
        return feats

    # Need clip — only available in 'mix' conda env
    try:
        import clip
        import torch
    except ImportError:
        raise ImportError(
            "openai-clip not found in the current Python env.\n"
            "Run with:  /home/sunl/miniconda3/envs/mix/bin/python  <script>"
        )

    print(f"Building CLIP ViT-B/32 text features for ScanNet-{label_set} "
          f"({len(class_names)} classes)…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    feats_list = []
    batch = 64
    with torch.no_grad():
        for i in range(0, len(class_names), batch):
            sub = class_names[i:i + batch]
            tokens = clip.tokenize(
                [f"a photo of a {c}" for c in sub], truncate=True
            ).to(device)
            f = model.encode_text(tokens).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats_list.append(f.cpu().numpy())

    feats = np.concatenate(feats_list, axis=0).astype(np.float32)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, feats)
    print(f"Saved text feature cache: {cache}  shape={feats.shape}")
    return feats


# ── Colour helpers ────────────────────────────────────────────────────────
def _palette(n):
    out = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        s = 0.65 + (i % 3) * 0.12
        v = 0.75 + (i % 2) * 0.15
        h6 = h * 6; c = v * s; x = c * (1 - abs(h6 % 2 - 1)); m = v - c
        if   h6 < 1: r, g, b = c, x, 0
        elif h6 < 2: r, g, b = x, c, 0
        elif h6 < 3: r, g, b = 0, c, x
        elif h6 < 4: r, g, b = 0, x, c
        elif h6 < 5: r, g, b = x, 0, c
        else:        r, g, b = c, 0, x
        out.append((r + m, g + m, b + m))
    return out


def _norm_color(norm_v):
    ratio = float(np.clip(norm_v, 0.0, 1.0))
    return (1.0 - ratio, ratio * 0.85, 0.15)


def _semantic_color(sim, match):
    if match == 2:
        g = 0.4 + 0.6 * float(np.clip(sim, 0.0, 1.0))
        return (0.1, g, 0.2)
    elif match == 1:
        return (0.95, 0.55, 0.05)
    else:
        return (0.85, 0.15, 0.1)


# ── Semantic match ────────────────────────────────────────────────────────
# Structural vs furniture groups are derived from the loaded label set,
# no hard-coded names.
_STRUCTURAL_KEYWORDS = {"wall", "floor", "ceiling", "stairs", "door way"}
_FURNITURE_KEYWORDS  = {"chair", "table", "sofa", "desk", "cabinet", "bed",
                        "bookshelf", "counter", "shelf", "stool", "ottoman",
                        "couch", "dresser", "wardrobe", "nightstand"}


def _build_groups(label_set_names: set):
    """Partition label_set_names into structural and furniture sets."""
    structural, furniture = set(), set()
    for name in label_set_names:
        n = name.lower()
        if any(kw in n for kw in _STRUCTURAL_KEYWORDS):
            structural.add(n)
        elif any(kw in n for kw in _FURNITURE_KEYWORDS):
            furniture.add(n)
    return structural, furniture


def _match_level(odise_cat: str, lseg_cat: str,
                 label_set_names: set,
                 structural: set, furniture: set) -> int:
    """
    2  exact match
    1  both in label_set, different, no structural<->furniture confusion
    0  structural<->furniture confusion, or either category outside label_set
    """
    o, l = odise_cat.lower(), lseg_cat.lower()
    if o == l:
        return 2
    if (o in structural and l in furniture) or \
       (o in furniture  and l in structural):
        return 0
    if l in label_set_names and o in label_set_names:
        return 1
    return 0


# ── Single frame ──────────────────────────────────────────────────────────
def visualize_frame(scene, frame_stem, output_path: Path,
                    text_feats, class_names, label_set_names,
                    structural, furniture,
                    norm_warn_lo=0.90, norm_warn_hi=1.05):
    img_path = None
    for ext in (".jpg", ".png", ".jpeg"):
        cand = IMAGE_DIR / scene / "color" / f"{frame_stem}{ext}"
        if cand.exists():
            img_path = cand; break
    if img_path is None:
        return False, {}

    pp_path = PIXEL_POOLED_DIR / scene / f"{frame_stem}_odise.npz"
    if not pp_path.exists():
        return False, {}

    d = np.load(pp_path, allow_pickle=True)
    if "pixel_pooled" not in d.files:
        return False, {}

    masks  = d["masks"].astype(bool)
    pp_emb = d["pixel_pooled"].astype(np.float32)        # (K, 512)
    norms  = np.linalg.norm(pp_emb, axis=1)

    pp_n      = pp_emb / (norms[:, None] + 1e-8)
    sims      = pp_n @ text_feats.T                       # (K, C)
    top1_idx  = sims.argmax(axis=1)
    top1_sim  = sims[np.arange(len(sims)), top1_idx]
    top1_name = [class_names[i] for i in top1_idx]

    if "info" in d.files:
        info = d["info"]
    else:
        od = ODISE_DIR / scene / f"{frame_stem}_odise.npz"
        info = (np.load(od, allow_pickle=True)["info"] if od.exists()
                else np.array([{"category_name": f"mask{i}", "score": 0.0}
                               for i in range(len(masks))]))

    img = np.array(Image.open(img_path).convert("RGB"))
    H_img, W_img = img.shape[:2]
    K   = len(masks)
    pal = _palette(K)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=88)
    for ax in axes:
        ax.imshow(img); ax.axis("off")

    composites  = [np.zeros((H_img, W_img, 4), dtype=np.float32) for _ in range(3)]
    legend_data = [[] for _ in range(3)]

    n_norm_warn = n_mismatch = 0
    per_mask = []

    for i, mask in enumerate(masks):
        score  = float(info[i].get("score", 0.0)) if hasattr(info[i], "get") else 0.0
        cat    = (info[i].get("category_name", f"mask{i}")
                  if hasattr(info[i], "get") else f"mask{i}")
        norm_v = float(norms[i])
        sim_v  = float(top1_sim[i])
        pred_c = top1_name[i]
        match  = _match_level(cat, pred_c, label_set_names, structural, furniture)

        norm_ok = norm_warn_lo <= norm_v <= norm_warn_hi
        if not norm_ok: n_norm_warn += 1
        if match == 0:  n_mismatch  += 1
        per_mask.append(dict(cat=cat, score=score, norm=norm_v,
                             pred=pred_c, sim=sim_v, match=match))

        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())

        # resize mask to image dims if needed
        m = mask
        if mask.shape != (H_img, W_img):
            m = np.array(Image.fromarray(mask).resize(
                (W_img, H_img), Image.NEAREST)).astype(bool)

        col_cfg = [
            (pal[i],
             f"{cat}\n{score:.2f}",
             f"[{i}] {cat}  score={score:.2f}"),
            (_norm_color(norm_v),
             f"norm={norm_v:.3f}\n{'OK' if norm_ok else 'WARN'}",
             f"[{i}] norm={norm_v:.4f}  {'OK' if norm_ok else 'WARN'}"),
            (_semantic_color(sim_v, match),
             f"{pred_c}\nsim={sim_v:.3f}",
             f"[{i}] ODISE={cat}  LSeg={pred_c}  sim={sim_v:.3f}"
             f"  {'OK' if match > 0 else 'MISMATCH'}"),
        ]

        for col, (rgb, txt, lbl) in enumerate(col_cfg):
            composites[col][m] = (*rgb, 0.50)
            axes[col].text(cx, cy, txt, fontsize=6.5, color="white",
                           weight="bold", ha="center", va="center",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=rgb,
                                     edgecolor="white", alpha=0.85))
            legend_data[col].append(mpatches.Patch(color=rgb, label=lbl))

    col_titles = [
        "ODISE  (category + score)",
        "LSeg norm  (green=healthy, red=low)",
        f"LSeg semantic  (CLIP ScanNet-{len(class_names)} top-1)",
    ]
    for col, ax in enumerate(axes):
        ax.imshow(composites[col])
        ax.set_title(col_titles[col], fontsize=10, weight="bold")
        if legend_data[col]:
            ax.legend(handles=legend_data[col], loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), fontsize=6.5,
                      title=f"K={K}", title_fontsize=8)

    fig.suptitle(
        f"{scene} / frame {frame_stem}  |  "
        f"K={K}  norm-warn={n_norm_warn}  semantic-mismatch={n_mismatch}",
        fontsize=11, weight="bold"
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=88)
    plt.close()

    return True, dict(K=K, n_norm_warn=n_norm_warn, n_mismatch=n_mismatch,
                      per_mask=per_mask)


# ── Batch runner ──────────────────────────────────────────────────────────
def run(scene_list, num_frames, output_dir: Path,
        text_feats, class_names, label_set_names,
        structural, furniture,
        norm_warn_lo=0.90, norm_warn_hi=1.05):
    output_dir.mkdir(parents=True, exist_ok=True)
    total_K = total_nw = total_mm = 0
    scene_stats = {}

    for scene in scene_list:
        scene_pp = PIXEL_POOLED_DIR / scene
        if not scene_pp.exists():
            print(f"[skip] scene not found: {scene}"); continue

        npz_files = sorted(scene_pp.glob("*_odise.npz"))
        if not npz_files:
            continue
        if num_frames and len(npz_files) > num_frames:
            idxs = np.linspace(0, len(npz_files) - 1, num_frames, dtype=int)
            npz_files = [npz_files[i] for i in idxs]

        sc_k = sc_nw = sc_mm = 0
        for npz_p in npz_files:
            frame_stem = npz_p.stem.replace("_odise", "")
            out_p = output_dir / f"{scene}_{frame_stem}.png"
            ok, stats = visualize_frame(
                scene, frame_stem, out_p,
                text_feats, class_names, label_set_names,
                structural, furniture,
                norm_warn_lo, norm_warn_hi)
            if not ok:
                print(f"  [skip] {scene}/{frame_stem}"); continue

            K, nw, mm = stats["K"], stats["n_norm_warn"], stats["n_mismatch"]
            sc_k += K; sc_nw += nw; sc_mm += mm
            print(f"  {scene}/{frame_stem}  K={K}  norm-warn={nw}  "
                  f"semantic-mismatch={mm}")
            for pm in stats["per_mask"]:
                flag = "  OK" if pm["match"] > 0 else "  !! MISMATCH"
                print(f"    [{pm['cat']:22s}] -> [{pm['pred']:22s}]  "
                      f"sim={pm['sim']:.3f}{flag}")

        scene_stats[scene] = dict(K=sc_k, norm_warn=sc_nw, mismatch=sc_mm)
        total_K += sc_k; total_nw += sc_nw; total_mm += sc_mm

    print()
    print("=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)
    print(f"  {'scene':25s}  {'masks':>6}  {'norm-warn':>10}  {'semantic-mm':>12}")
    print("  " + "-" * 60)
    for sc, st in scene_stats.items():
        K = st["K"] or 1
        print(f"  {sc:25s}  {st['K']:6d}  "
              f"{st['norm_warn']:4d} ({100*st['norm_warn']/K:4.1f}%)  "
              f"{st['mismatch']:4d} ({100*st['mismatch']/K:4.1f}%)")
    print("  " + "-" * 60)
    K_all = total_K or 1
    print(f"  {'TOTAL':25s}  {total_K:6d}  "
          f"{total_nw:4d} ({100*total_nw/K_all:4.1f}%)  "
          f"{total_mm:4d} ({100*total_mm/K_all:4.1f}%)")
    print()
    vn = "OK"       if total_nw / K_all < 0.05 else "WARN"
    vs = "OK"       if total_mm / K_all < 0.10 else "CONTAMINATED"
    print(f"  Norm health   : {vn}")
    print(f"  Semantic align: {vs}")
    print(f"\nVisualization saved to: {output_dir}")


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Visualize LSeg pooled embedding quality")
    p.add_argument("--label-set", type=int, default=200, choices=[20, 200],
                   help="ScanNet label set to use for CLIP scoring (20 or 200). "
                        "Must match the label set used when generating ODISE features "
                        "(default: 200).")
    p.add_argument("--scene", type=str, default=None,
                   help="Single scene to visualize (e.g. scene0000_00).")
    p.add_argument("--num-scenes", type=int, default=3,
                   help="Number of random scenes when --scene is not set.")
    p.add_argument("--num-frames", type=int, default=4,
                   help="Frames per scene (uniform sampling).")
    p.add_argument("--output-dir", type=str,
                   default=str(DIAG_DIR / "lseg_vis_output"),
                   help="Output directory for PNG images.")
    p.add_argument("--norm-lo",  type=float, default=0.90,
                   help="LSeg norm lower warning threshold (default 0.90).")
    p.add_argument("--norm-hi",  type=float, default=1.05,
                   help="LSeg norm upper warning threshold (default 1.05).")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force re-generate CLIP text feature cache.")
    args = p.parse_args()

    # Load labels from scannet_label_constant.py
    class_names     = load_label_list(args.label_set)
    label_set_names = set(class_names)
    print(f"Label set: ScanNet-{args.label_set}  ({len(class_names)} classes)")

    # Build / load text features
    text_feats = build_text_features(args.label_set, class_names,
                                     force_rebuild=args.rebuild_cache)

    # Derive structural/furniture groups from label set (no hard-coding)
    structural, furniture = _build_groups(label_set_names)
    print(f"  structural group ({len(structural)}): {sorted(structural)}")
    print(f"  furniture  group ({len(furniture)}): {sorted(furniture)}")

    # Scene selection
    if args.scene:
        scene_list = [args.scene]
    else:
        all_scenes = sorted([d.name for d in PIXEL_POOLED_DIR.iterdir()
                             if d.is_dir()])
        np.random.seed(42)
        scene_list = list(np.random.choice(
            all_scenes, min(args.num_scenes, len(all_scenes)), replace=False))
        print(f"Sampled {len(scene_list)} scenes: {scene_list}")

    run(scene_list=scene_list,
        num_frames=args.num_frames,
        output_dir=Path(args.output_dir),
        text_feats=text_feats,
        class_names=class_names,
        label_set_names=label_set_names,
        structural=structural,
        furniture=furniture,
        norm_warn_lo=args.norm_lo,
        norm_warn_hi=args.norm_hi)


if __name__ == "__main__":
    main()
