"""
K 维 Mask 置换不变性测试

目的: 验证 pixel_pooled / mask_embeddings / masks / mask_valid
      在 K 维度上是否严格一一对应。

核心思路:
  同时对这四个张量做相同的 K 维置换，语义内容不变，
  所以 loss / teacher mIoU 应该几乎不变。
  如果明显变化 → 代码里存在隐藏的 K 顺序依赖。

再做"故意错位"实验:
  只置换 pixel_pooled，其他不动
  → 如果结果崩塌，坐实 K 维顺序错配 bug。

输出报告会告诉你：
  1. 是否存在 K 维隐式依赖
  2. 哪个字段最可疑

运行方式:
  cd /home/sunl/work/mix
  python debug/debug_mask_permutation.py

可选: 指定样本数量（默认 10 个）
  python debug/debug_mask_permutation.py --num_samples 20
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SCANNET_LABELS_20 = [
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator",
    "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
]


# -------------------------------------------------------------------
# 工具函数
# -------------------------------------------------------------------

def load_batch(dataset, idx, device):
    from dataset.open_vocab_dataset_v2 import open_vocab_collate_v2
    item = dataset[idx]
    batch = open_vocab_collate_v2([item])
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def apply_perm_to_batch(batch, perm, fields=None):
    """
    对 batch 里指定 fields 的 K 维（dim=1）应用 perm。
    fields 默认是全部四个：pixel_pooled, mask_embeddings, masks, mask_valid
    """
    if fields is None:
        fields = ["pixel_pooled", "mask_embeddings", "masks", "mask_valid"]
    new_batch = dict(batch)
    for f in fields:
        if f in new_batch and isinstance(new_batch[f], torch.Tensor):
            t = new_batch[f]
            # perm 可能比 K 短（因为 padding 到了 K_max）
            K_actual = perm.shape[0]
            # 只置换前 K_actual 个，其余 padding 槽位不动
            t_new = t.clone()
            t_new[:, :K_actual] = t[:, perm]
            new_batch[f] = t_new
    return new_batch


def compute_loss_from_model(model, batch, device):
    """
    跑一次 forward，用 Criteria 计算 BCE+Dice loss。
    返回 float。
    """
    from MinkowskiEngine import SparseTensor
    from model.criterion import Criteria

    coords = batch["coords_3d"].int().to(device)
    feats  = batch["feat_3d"].float().to(device)
    sinput = SparseTensor(feats, coords)
    batch_dev = dict(batch)
    batch_dev["sinput"] = sinput

    with torch.no_grad():
        out = model(batch_dev)

    criteria = Criteria(
        results=out,
        batch_input=batch_dev,
        threshold=0.5,
        min_points_per_mask=5,
        bce_weight=1.0,
        dice_weight=1.0,
        use_keep_filter=False,
    )
    loss = criteria.loss_pt()
    return loss.item()


def build_point_teacher_miou(batch, pixel_pooled_override, mask_valid_override,
                              masks_override, text_features, labels, device):
    """
    用给定的 pixel_pooled / masks / mask_valid 构造 per-point teacher，
    计算 semantic mIoU（不经过 student）。
    """
    x_label    = batch["x_label"].to(device)
    y_label    = batch["y_label"].to(device)
    ori_coords = batch["ori_coords_3d"].to(device)
    gt_labels  = batch["binary_label_3d"].to(device)

    pixel_pooled = pixel_pooled_override.to(device)  # (B, K, D)
    masks        = masks_override.to(device)          # (B, K, H, W)
    mask_valid   = mask_valid_override.to(device)     # (B, K) bool

    B, K, H, W = masks.shape
    N_total = x_label.shape[0]
    D = pixel_pooled.shape[-1]

    teacher = torch.zeros(N_total, D, device=device)
    point_valid = torch.zeros(N_total, dtype=torch.bool, device=device)

    for b in range(B):
        pt_mask = ori_coords[:, 0] == b
        if not pt_mask.any():
            continue
        x_b = x_label[pt_mask].clamp(0, W - 1)
        y_b = y_label[pt_mask].clamp(0, H - 1)
        valid_k = mask_valid[b]
        if not valid_k.any():
            continue
        pp_b = pixel_pooled[b, valid_k]
        mk_b = masks[b, valid_k]
        hit = mk_b[:, y_b, x_b]
        hit = (hit > 0.5).float()
        weights = hit.sum(dim=0)
        t_b = (hit.T @ pp_b)
        valid_pt = weights > 0
        t_b[valid_pt] = t_b[valid_pt] / weights[valid_pt].unsqueeze(-1)
        idx_b = pt_mask.nonzero(as_tuple=True)[0]
        teacher[idx_b] = t_b.float()
        point_valid[idx_b] = valid_pt

    # Compute mIoU
    keep = point_valid & (gt_labels >= 0) & (gt_labels < len(labels))
    if keep.sum() == 0:
        return 0.0

    feats = F.normalize(teacher[keep].float(), dim=-1)
    sim   = feats @ text_features.T
    pred  = sim.argmax(dim=-1)
    gts   = gt_labels[keep]

    ious = []
    for c in range(len(labels)):
        gt_c = (gts == c)
        pr_c = (pred == c)
        inter = (gt_c & pr_c).sum().item()
        union = (gt_c | pr_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def build_text_features(labels, device, clip_model="ViT-L/14"):
    import clip
    model, _ = clip.load(clip_model, device=device)
    model.eval()
    prompts = [f"a {l} in a scene" for l in labels]
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        feats = model.encode_text(tokens).float()
        feats = F.normalize(feats, dim=-1)
    del model
    torch.cuda.empty_cache()
    return feats


def print_section(s):
    print(f"\n{'='*60}\n  {s}\n{'='*60}")


# -------------------------------------------------------------------
# 主测试逻辑
# -------------------------------------------------------------------

def run_permutation_test(dataset, model, text_features, args, device):
    """
    对每个样本做三轮测试：
      baseline:     原始顺序
      full_perm:    四个字段同时置换（语义不变，loss/miou 应该几乎不变）
      pixel_only:   只置换 pixel_pooled（故意错位，应该让结果变差）
    """
    from dataset.open_vocab_dataset_v2 import open_vocab_collate_v2

    results = {
        "baseline_loss":    [],
        "full_perm_loss":   [],
        "pixel_only_loss":  [],
        "baseline_miou":    [],
        "full_perm_miou":   [],
        "pixel_only_miou":  [],
    }

    n_eval = min(args.num_samples, len(dataset))
    print(f"\n评估 {n_eval} 个样本...")

    for idx in range(n_eval):
        try:
            batch = load_batch(dataset, idx, device)
        except Exception as e:
            print(f"  [SKIP] idx={idx}: {e}")
            continue

        K = batch["pixel_pooled"].shape[1]
        if K < 2:
            print(f"  [SKIP] idx={idx}: K={K} 太少，无法置换")
            continue

        # 生成随机置换（固定种子保证可复现）
        rng = np.random.default_rng(seed=42 + idx)
        perm = torch.from_numpy(rng.permutation(K)).long().to(device)

        # ---- baseline ----
        try:
            bl_loss = compute_loss_from_model(model, batch, device) if model else None
            bl_miou = build_point_teacher_miou(
                batch,
                batch["pixel_pooled"], batch["mask_valid"], batch["masks"],
                text_features, SCANNET_LABELS_20, device
            )
        except Exception as e:
            print(f"  [SKIP baseline] idx={idx}: {e}")
            continue

        # ---- full_perm (同时置换四个字段) ----
        batch_fp = apply_perm_to_batch(batch, perm,
                                        fields=["pixel_pooled", "mask_embeddings",
                                                "masks", "mask_valid"])
        try:
            fp_loss = compute_loss_from_model(model, batch_fp, device) if model else None
            fp_miou = build_point_teacher_miou(
                batch_fp,
                batch_fp["pixel_pooled"], batch_fp["mask_valid"], batch_fp["masks"],
                text_features, SCANNET_LABELS_20, device
            )
        except Exception as e:
            print(f"  [SKIP full_perm] idx={idx}: {e}")
            fp_loss, fp_miou = None, 0.0

        # ---- pixel_only perm (只置换 pixel_pooled，制造错位) ----
        batch_po = apply_perm_to_batch(batch, perm, fields=["pixel_pooled"])
        try:
            po_loss = compute_loss_from_model(model, batch_po, device) if model else None
            po_miou = build_point_teacher_miou(
                batch_po,
                batch_po["pixel_pooled"], batch_po["mask_valid"], batch_po["masks"],
                text_features, SCANNET_LABELS_20, device
            )
        except Exception as e:
            print(f"  [SKIP pixel_only] idx={idx}: {e}")
            po_loss, po_miou = None, 0.0

        if bl_loss is not None:
            results["baseline_loss"].append(bl_loss)
        if fp_loss is not None:
            results["full_perm_loss"].append(fp_loss)
        if po_loss is not None:
            results["pixel_only_loss"].append(po_loss)
        results["baseline_miou"].append(bl_miou)
        results["full_perm_miou"].append(fp_miou)
        results["pixel_only_miou"].append(po_miou)

        if (idx + 1) % 5 == 0 or idx == n_eval - 1:
            bl_m  = np.mean(results["baseline_miou"]) * 100
            fp_m  = np.mean(results["full_perm_miou"]) * 100
            po_m  = np.mean(results["pixel_only_miou"]) * 100
            print(f"  [{idx+1}/{n_eval}] mIoU: baseline={bl_m:.1f}%, "
                  f"full_perm={fp_m:.1f}%, pixel_only={po_m:.1f}%")

    return results


def print_results(results):
    print_section("置换不变性测试结果")

    def safe_mean(lst, pct=True):
        if not lst:
            return float("nan")
        v = np.mean(lst)
        return v * 100 if pct else v

    bl_loss = safe_mean(results["baseline_loss"], pct=False)
    fp_loss = safe_mean(results["full_perm_loss"], pct=False)
    po_loss = safe_mean(results["pixel_only_loss"], pct=False)
    bl_miou = safe_mean(results["baseline_miou"])
    fp_miou = safe_mean(results["full_perm_miou"])
    po_miou = safe_mean(results["pixel_only_miou"])

    print(f"  {'':30s} {'loss':>8s}  {'mIoU':>8s}")
    print(f"  {'baseline (原始顺序)':30s} {bl_loss:8.4f}  {bl_miou:7.2f}%")
    print(f"  {'full_perm (四字段同时置换)':30s} {fp_loss:8.4f}  {fp_miou:7.2f}%")
    print(f"  {'pixel_only (只置换pixel_pooled)':30s} {po_loss:8.4f}  {po_miou:7.2f}%")

    # ---- 判断 ----
    print_section("诊断结论")

    # 判断 full_perm 是否不变
    if not np.isnan(bl_miou) and not np.isnan(fp_miou):
        miou_diff_fp = abs(fp_miou - bl_miou)
        loss_diff_fp = abs(fp_loss - bl_loss) if not np.isnan(fp_loss) else float("nan")

        if miou_diff_fp > 5.0:
            print(f"  ❌ 发现 K 维隐式依赖！")
            print(f"     full_perm 后 mIoU 变化: {bl_miou:.1f}% → {fp_miou:.1f}%  (差 {miou_diff_fp:.1f}%)")
            print(f"     即使四个字段同时置换（语义内容不变），结果也变了。")
            print(f"     → 代码里存在 没有对 K 维置换的隐藏依赖")
            print(f"     → 最可疑的地方: 预处理时某字段用了不同的排序方式")
        else:
            print(f"  ✅ full_perm 不变性通过 (mIoU 变化 {miou_diff_fp:.1f}% ≤ 5%)")
            print(f"     → 四个字段在 K 维上顺序一致，没有明显的 bookkeeping 错配")

    # 判断 pixel_only 是否崩塌
    if not np.isnan(bl_miou) and not np.isnan(po_miou):
        miou_diff_po = bl_miou - po_miou

        if miou_diff_po > 5.0:
            print(f"\n  ⚠️  故意错位 (pixel_only) 后结果变差:")
            print(f"     {bl_miou:.1f}% → {po_miou:.1f}%  (下降 {miou_diff_po:.1f}%)")
            print(f"     这说明 pixel_pooled 的 K 维顺序 对结果有实质影响。")
            print(f"     如果 full_perm 也变了，说明当前代码里 pixel_pooled 和其他字段顺序不同。")
            print(f"     如果 full_perm 没变，说明 pixel_pooled 顺序本身是敏感的（但目前是对的）。")
        else:
            print(f"\n  pixel_only 错位后变化 {miou_diff_po:.1f}% ≤ 5%，不敏感")
            print(f"     → 可能 pixel_pooled 本身对最终结果影响不大（被 fuse_embed 改变了）")

    # 组合判断
    print()
    if (not np.isnan(bl_miou) and not np.isnan(fp_miou)
            and abs(fp_miou - bl_miou) > 5.0
            and not np.isnan(po_miou) and (bl_miou - po_miou) > 5.0):
        print("  🔴 综合判断：高度怀疑存在 K 维顺序错配 bug")
        print("     下一步：检查预处理 precompute_odise_features.py 里")
        print("     pixel_pooled / mask_embeddings / masks 是否用了相同的 mask 排序")
    elif not np.isnan(fp_miou) and abs(fp_miou - bl_miou) <= 5.0:
        print("  🟢 综合判断：K 维对齐看起来正常")
        print("     问题更可能在 loss 设计（mask-slot 监督 vs. teacher 蒸馏）")
        print("     建议：跑 eval_teacher_chain.py 评估 A1/A2/A3 进一步定位")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_scannet_v2_full_multi_gpu.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="模型 checkpoint（用于计算 loss；没有则只评估 mIoU）")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip_model", default="ViT-L/14")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    device = args.device
    print(f"使用设备: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- 文本特征 ----
    print_section("构造文本特征")
    text_features = build_text_features(SCANNET_LABELS_20, device=device,
                                         clip_model=args.clip_model)

    # ---- 数据集 ----
    print_section("加载数据集")
    from dataset.open_vocab_dataset_v2 import (
        OpenVocabDatasetV2Config,
        OpenVocabScannetDatasetV2,
    )
    ds_cfg = OpenVocabDatasetV2Config(
        data_config_path=cfg["dataset"]["data_config_path"],
        precomputed_dir=cfg["dataset"]["precomputed_dir"],
        projection_dir=cfg["dataset"].get("projection_dir"),
        split=args.split,
        max_samples=args.num_samples,
    )
    dataset = OpenVocabScannetDatasetV2(ds_cfg)
    print(f"  数据集: {len(dataset)} 个样本")

    # ---- 模型（可选，用于 loss 测试）----
    model = None
    if args.checkpoint:
        print_section("加载模型（用于 loss 测试）")
        from model.open_vocab_fusion_v2 import (
            OpenVocabFusionModelV2Config,
            OpenVocab3DFusionModelV2,
        )
        mcfg = OpenVocabFusionModelV2Config(device=device)
        model = OpenVocab3DFusionModelV2(mcfg).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"  加载完成: {args.checkpoint}")
    else:
        print("  [INFO] 未提供 --checkpoint，跳过 loss 测试，只做 mIoU 置换测试")

    # ---- 主测试 ----
    results = run_permutation_test(dataset, model, text_features, args, device)
    print_results(results)
    print()


if __name__ == "__main__":
    main()
