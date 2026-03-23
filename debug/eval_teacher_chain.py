"""
Teacher 三段式评估脚本

目的: 判断问题在 teacher 链路 还是 student 训练。

评估三个量：
  A1. 原始 pooled LSeg teacher-only → 文本相似度 → semantic mIoU
  A2. fused_embeddings (teacher经 fuse_embed 后) → 文本相似度 → semantic mIoU
  A3. student pred_3d → 文本相似度 → semantic mIoU (可选，需 checkpoint)

判断逻辑：
  A1 高, A2 低, A3 低  → fuse_embed 或 K维对齐坏了 teacher
  A1 高, A2 高, A3 低  → teacher 没问题，student 训练目标有问题
  A1 就很低            → pooling/projection/point-teacher对应出问题了

运行方式:
  cd /home/sunl/work/mix
  python debug/eval_teacher_chain.py

可选: 带 checkpoint 评估 A3
  python debug/eval_teacher_chain.py --checkpoint checkpoints/full.4/best_model.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# 确保项目根目录在 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ScanNet20 的 20 个类别（和 OpenScene 一致）
SCANNET_LABELS_20 = [
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator",
    "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
]

# -------------------------------------------------------------------
# 文本特征 (CLIP) 构造
# -------------------------------------------------------------------

def build_text_features(labels, device="cuda", clip_model="ViT-L/14"):
    """用 CLIP 构造固定文本特征矩阵 (C_text, D)。"""
    try:
        import clip
    except ImportError:
        print("[ERROR] 需要安装 clip: pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    print(f"  加载 CLIP 模型 {clip_model} ...")
    model, _ = clip.load(clip_model, device=device)
    model.eval()

    prompts = [f"a {label} in a scene" for label in labels]
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        text_feats = model.encode_text(tokens).float()
        text_feats = F.normalize(text_feats, dim=-1)  # (C_text, D)

    del model
    torch.cuda.empty_cache()
    print(f"  文本特征形状: {text_feats.shape}")
    return text_feats


# -------------------------------------------------------------------
# 数据加载
# -------------------------------------------------------------------

def load_data_config(yaml_path):
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_batch(dataset, idx=0):
    """从 dataset 取一个 item，包装成 mini-batch（B=1）。"""
    from dataset.open_vocab_dataset_v2 import open_vocab_collate_v2
    item = dataset[idx]
    batch = open_vocab_collate_v2([item])
    return batch


def batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# -------------------------------------------------------------------
# point-level teacher 构造（A1 用的）
# -------------------------------------------------------------------

def build_point_teacher_from_pooled(batch, device):
    """
    A1: 对每个可见点，用 pixel_pooled 向量作为其 teacher。

    做法：
      - 每个点 (x, y) 命中某些 mask（masks[b, k, y, x] == 1）
      - 取命中 mask 的 pixel_pooled 向量平均
      - 没有命中任何 mask 的点，teacher 设为 0（后续会过滤掉）

    返回:
      teacher_per_point: (N_total, D_lseg)  每点 teacher 向量
      point_valid:       (N_total,) bool     是否有有效 teacher
      batch_idx:         (N_total,) long     所属 batch
      gt_labels:         (N_total,) long     语义标签 (-1 表示未知)
    """
    pixel_pooled = batch["pixel_pooled"].to(device)   # (B, K, 512)
    masks = batch["masks"].to(device)                  # (B, K, H, W) float
    mask_valid = batch["mask_valid"].to(device)        # (B, K) bool
    x_label = batch["x_label"].to(device)              # (N_total,)
    y_label = batch["y_label"].to(device)              # (N_total,)
    ori_coords = batch["ori_coords_3d"].to(device)     # (N_total, 4)
    gt_labels = batch["binary_label_3d"].to(device)    # (N_total,) 语义标签

    B, K, H, W = masks.shape
    N_total = x_label.shape[0]
    D = pixel_pooled.shape[-1]

    teacher = torch.zeros(N_total, D, device=device)
    point_valid = torch.zeros(N_total, dtype=torch.bool, device=device)

    for b in range(B):
        pt_mask = ori_coords[:, 0] == b
        if not pt_mask.any():
            continue

        x_b = x_label[pt_mask].clamp(0, W - 1)  # (N_b,)
        y_b = y_label[pt_mask].clamp(0, H - 1)  # (N_b,)
        valid_k = mask_valid[b]                   # (K,) bool
        K_valid = valid_k.sum().item()
        if K_valid == 0:
            continue

        pp_b = pixel_pooled[b, valid_k]           # (K_valid, D)
        mk_b = masks[b, valid_k]                   # (K_valid, H, W) float

        # 每个点在哪些 mask 里？
        # hit[n, k] = mk_b[k, y_b[n], x_b[n]]
        hit = mk_b[:, y_b, x_b]                   # (K_valid, N_b)
        hit = (hit > 0.5).float()                  # (K_valid, N_b) bool→float

        # teacher = weighted mean of pixel_pooled vectors
        # weight[n] = sum of hit[:, n]
        weights = hit.sum(dim=0)                   # (N_b,)
        t_b = (hit.T @ pp_b)                       # (N_b, D)  ← sum of hit pooled vecs
        valid_pt = weights > 0
        t_b[valid_pt] = t_b[valid_pt] / weights[valid_pt].unsqueeze(-1)

        # 写回
        idx_b = pt_mask.nonzero(as_tuple=True)[0]
        teacher[idx_b] = t_b.float()
        point_valid[idx_b] = valid_pt

    return teacher, point_valid, ori_coords[:, 0].long(), gt_labels


# -------------------------------------------------------------------
# semantic mIoU 计算（OpenScene 风格）
# -------------------------------------------------------------------

def compute_semantic_miou(point_feats, point_valid, gt_labels, text_features, labels, device):
    """
    将每点特征和文本特征做 cos 相似度，取 argmax 作为预测类别，
    和 gt_labels 比较，算 per-class IoU，返回 mIoU。

    注意: gt_labels 里 -1 / 255 表示 ignore，会被跳过。
    """
    # 过滤掉无效点
    keep = point_valid & (gt_labels >= 0) & (gt_labels < len(labels))
    if keep.sum() == 0:
        print("  [WARNING] 没有有效点，无法计算 mIoU")
        return 0.0, {}

    feats = point_feats[keep]        # (M, D)
    gts   = gt_labels[keep]          # (M,)

    feats = F.normalize(feats.float(), dim=-1)
    sim   = feats @ text_features.T  # (M, C_text)
    pred  = sim.argmax(dim=-1)       # (M,)

    C = len(labels)
    ious = {}
    for c in range(C):
        gt_c   = (gts == c)
        pred_c = (pred == c)
        inter  = (gt_c & pred_c).sum().item()
        union  = (gt_c | pred_c).sum().item()
        if union == 0:
            continue
        ious[labels[c]] = inter / union

    miou = float(np.mean(list(ious.values()))) if ious else 0.0
    return miou, ious


# -------------------------------------------------------------------
# A2: fused_embeddings → per-point teacher
# -------------------------------------------------------------------

def build_point_teacher_from_fused(batch, fused_embeddings, device):
    """
    A2: 用 fused_embeddings 替换 pixel_pooled，构造 per-point teacher。
    fused_embeddings: (B, K, D_fused)  来自 model.fuse_embed 的输出
    """
    masks = batch["masks"].to(device)              # (B, K, H, W)
    mask_valid = batch["mask_valid"].to(device)    # (B, K) bool
    x_label = batch["x_label"].to(device)
    y_label = batch["y_label"].to(device)
    ori_coords = batch["ori_coords_3d"].to(device)
    gt_labels = batch["binary_label_3d"].to(device)

    B, K, H, W = masks.shape
    N_total = x_label.shape[0]
    D = fused_embeddings.shape[-1]

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
        fe_b = fused_embeddings[b, valid_k]        # (K_valid, D)
        mk_b = masks[b, valid_k]                    # (K_valid, H, W)
        hit = mk_b[:, y_b, x_b]                    # (K_valid, N_b)
        hit = (hit > 0.5).float()
        weights = hit.sum(dim=0)
        t_b = (hit.T @ fe_b)
        valid_pt = weights > 0
        t_b[valid_pt] = t_b[valid_pt] / weights[valid_pt].unsqueeze(-1)
        idx_b = pt_mask.nonzero(as_tuple=True)[0]
        teacher[idx_b] = t_b.float()
        point_valid[idx_b] = valid_pt

    return teacher, point_valid, ori_coords[:, 0].long(), gt_labels


# -------------------------------------------------------------------
# 构造 SparseTensor
# -------------------------------------------------------------------

def build_sinput(batch, device):
    from MinkowskiEngine import SparseTensor
    coords = batch["coords_3d"].int().to(device)
    feats  = batch["feat_3d"].float().to(device)
    return SparseTensor(feats, coords)


# -------------------------------------------------------------------
# 打印结果
# -------------------------------------------------------------------

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_miou(miou, ious, top_n=5):
    print(f"  mIoU = {miou * 100:.2f}%")
    if ious:
        sorted_cls = sorted(ious.items(), key=lambda x: -x[1])
        print(f"  Top-{top_n} classes:")
        for cls, iou in sorted_cls[:top_n]:
            print(f"    {cls:20s}: {iou*100:.1f}%")


# -------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_scannet_v2_full_multi_gpu.yaml")
    parser.add_argument("--checkpoint", default=None, help="用于 A3 评估的模型 checkpoint")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="评估的样本数量（越多越准，但越慢）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip_model", default="ViT-L/14")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    device = args.device
    print(f"使用设备: {device}")

    # ---- 加载配置 ----
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- 构造文本特征 ----
    print_section("构造文本特征 (CLIP)")
    text_features = build_text_features(SCANNET_LABELS_20, device=device, clip_model=args.clip_model)

    # ---- 加载数据集 ----
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
    print(f"  数据集大小: {len(dataset)} 个样本（用 {args.num_samples} 个评估）")

    # ---- 可选: 加载模型（用于 A2 / A3）----
    model = None
    if args.checkpoint:
        print_section("加载模型 (用于 A2/A3)")
        from model.open_vocab_fusion_v2 import (
            OpenVocabFusionModelV2Config,
            OpenVocab3DFusionModelV2,
        )
        model_cfg = OpenVocabFusionModelV2Config(device=device)
        model = OpenVocab3DFusionModelV2(model_cfg).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [WARNING] missing keys: {missing[:5]} ...")
        model.eval()
        print(f"  checkpoint 加载完成: {args.checkpoint}")
    else:
        # 只做 A2: fuse_embed 不需要 pc_processor，但我们仍然需要 model
        print_section("加载模型骨架（用于 A2，fuse_embed only）")
        from model.open_vocab_fusion_v2 import (
            OpenVocabFusionModelV2Config,
            OpenVocab3DFusionModelV2,
        )
        model_cfg = OpenVocabFusionModelV2Config(device=device)
        model = OpenVocab3DFusionModelV2(model_cfg).to(device)
        model.eval()
        print("  [INFO] 没有 checkpoint，fuse_embed 使用随机初始权重（A2 仅作参考）")

    # ====================================================================
    # 开始逐样本评估
    # ====================================================================
    from dataset.open_vocab_dataset_v2 import open_vocab_collate_v2

    miou_a1_list = []
    miou_a2_list = []
    miou_a3_list = []

    valid_pt_ratio_list = []

    print_section(f"开始评估（{min(args.num_samples, len(dataset))} 个样本）")

    n_eval = min(args.num_samples, len(dataset))
    for idx in range(n_eval):
        try:
            item = dataset[idx]
            batch = open_vocab_collate_v2([item])
            batch = batch_to_device(batch, device)
        except Exception as e:
            print(f"  [SKIP] idx={idx}: 加载失败 {e}")
            continue

        # ---- A1: pixel_pooled teacher ----
        with torch.no_grad():
            teacher_a1, valid_a1, b_idx, gt_lbl = build_point_teacher_from_pooled(batch, device)

        ratio = valid_a1.float().mean().item()
        valid_pt_ratio_list.append(ratio)

        # 有效 gt_labels：binary_label_3d 里存的是 ScanNet 语义标签
        miou_a1, _ = compute_semantic_miou(
            teacher_a1, valid_a1, gt_lbl, text_features, SCANNET_LABELS_20, device
        )
        miou_a1_list.append(miou_a1)

        # ---- A2: fused_embeddings teacher ----
        with torch.no_grad():
            pixel_pooled = batch["pixel_pooled"].float()
            mask_emb = batch["mask_embeddings"].float()
            masks_t = batch["masks"].float()
            mv = batch["mask_valid"]

            fused = model.fuse_embed(pixel_pooled, mask_emb, masks_t, mv)  # (B, K, D_fused)

        teacher_a2, valid_a2, _, _ = build_point_teacher_from_fused(batch, fused, device)
        miou_a2, _ = compute_semantic_miou(
            teacher_a2, valid_a2, gt_lbl, text_features, SCANNET_LABELS_20, device
        )
        miou_a2_list.append(miou_a2)

        # ---- A3: student pred_3d ----
        if args.checkpoint:
            try:
                sinput = build_sinput(batch, device)
                batch["sinput"] = sinput
                with torch.no_grad():
                    out = model(batch)
                pred_3d = out["pred_3d"]                       # (N_total, D_fused)
                valid_a3 = torch.ones(pred_3d.shape[0], dtype=torch.bool, device=device)
                miou_a3, _ = compute_semantic_miou(
                    pred_3d, valid_a3, gt_lbl, text_features, SCANNET_LABELS_20, device
                )
                miou_a3_list.append(miou_a3)
            except Exception as e:
                print(f"  [SKIP A3] idx={idx}: {e}")

        if (idx + 1) % 5 == 0 or idx == n_eval - 1:
            a1 = np.mean(miou_a1_list) * 100 if miou_a1_list else 0
            a2 = np.mean(miou_a2_list) * 100 if miou_a2_list else 0
            a3 = np.mean(miou_a3_list) * 100 if miou_a3_list else 0
            vr = np.mean(valid_pt_ratio_list) * 100 if valid_pt_ratio_list else 0
            a3_str = f", A3(student)={a3:.1f}%" if miou_a3_list else ""
            print(f"  [{idx+1}/{n_eval}] A1(pooled)={a1:.1f}%, A2(fused)={a2:.1f}%{a3_str}, "
                  f"有效点比例={vr:.1f}%")

    # ====================================================================
    # 最终结论
    # ====================================================================
    print_section("最终评估结果")
    a1_final = np.mean(miou_a1_list) * 100 if miou_a1_list else 0
    a2_final = np.mean(miou_a2_list) * 100 if miou_a2_list else 0
    a3_final = np.mean(miou_a3_list) * 100 if miou_a3_list else 0
    vr_final = np.mean(valid_pt_ratio_list) * 100 if valid_pt_ratio_list else 0

    print(f"  A1  pooled LSeg teacher mIoU  = {a1_final:.2f}%")
    print(f"  A2  fused_embed  teacher mIoU = {a2_final:.2f}%")
    if miou_a3_list:
        print(f"  A3  student pred_3d  mIoU   = {a3_final:.2f}%")
    print(f"  平均有效点比例 (命中任意mask)  = {vr_final:.1f}%")

    print_section("诊断结论")
    if a1_final < 5.0:
        print("  ❌ A1 就很低（<5%）")
        print("     → 问题出在 pooling/projection/point-teacher对应 这些早期步骤")
        print("     → 需要检查: x_label/y_label 正确性, 语义标签 binary_label_3d 是否对应20类")
    elif a1_final >= 10.0 and a2_final < a1_final * 0.5:
        print(f"  ⚠️  A1({a1_final:.1f}%) >> A2({a2_final:.1f}%)")
        print("     → fuse_embed 把 teacher 破坏了（或 K维顺序在 fuse 时错配）")
        print("     → 建议: 先做 permutation test 验证 K维对齐")
    elif a1_final >= 10.0 and a2_final >= a1_final * 0.7 and miou_a3_list and a3_final < 5.0:
        print(f"  ✅ Teacher 看起来正常 (A1={a1_final:.1f}%, A2={a2_final:.1f}%)")
        print(f"  ❌ 但 student 很差 (A3={a3_final:.1f}%)")
        print("     → 问题在 student 训练目标：BCE+Dice的mask-slot监督没有传递语义空间")
        print("     → 建议: 改用 cos-distillation loss，让 pred_3d 直接逼近 teacher 向量")
    elif a1_final >= 10.0 and a2_final >= a1_final * 0.7:
        print(f"  ✅ Teacher 看起来正常 (A1={a1_final:.1f}%, A2={a2_final:.1f}%)")
        print(f"  → 如果还没有 checkpoint，用 --checkpoint 传入模型后再看 A3")
    else:
        print(f"  A1={a1_final:.1f}%, A2={a2_final:.1f}%, 请结合具体数值判断")

    if vr_final < 30.0:
        print(f"\n  ⚠️  有效点比例 {vr_final:.1f}% 偏低（< 30%）")
        print("     → 大量点没有命中任何 mask，监督密度不足，会放大其他问题")

    print()


if __name__ == "__main__":
    main()
