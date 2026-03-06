"""
完整追踪一个 batch 的数据流，找出 IoU 低的根本原因
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/featurize/work/mix')

from dataset.open_vocab_dataset_v2 import OpenVocabDatasetV2Config, OpenVocabScannetDatasetV2, open_vocab_collate_v2
from model.open_vocab_fusion_v2 import OpenVocabFusionModelV2Config, OpenVocab3DFusionModelV2
from model.criterion import Criteria

# 创建数据集
config = OpenVocabDatasetV2Config(
    data_config_path="/home/featurize/work/mix/config/data_scannet_3d.yaml",
    precomputed_dir="/home/featurize/data/pixel_pooled",
    projection_dir="/home/featurize/data/scannet_projections",  # 使用预计算投影
    split="train",
    max_samples=10,  # 只用 10 个样本测试
)

dataset = OpenVocabScannetDatasetV2(config)
print(f"Dataset size: {len(dataset)}")

# 手动获取一个 batch
batch_items = [dataset[i] for i in range(4)]
batch = open_vocab_collate_v2(batch_items)

print("\n" + "="*80)
print("Batch 数据:")
print("="*80)
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}, dtype={v.dtype}, device={v.device}")
        if k in ["x_label", "y_label"]:
            print(f"  范围: [{v.min().item()}, {v.max().item()}]")
            print(f"  是否全零: {(v == 0).all()}")
    else:
        print(f"{k}: {type(v)}")

# 创建模型
model_config = OpenVocabFusionModelV2Config(device="cuda")
model = OpenVocab3DFusionModelV2(model_config).cuda()
model.eval()

# 准备输入
from MinkowskiEngine import SparseTensor
coords_3d = batch["coords_3d"].int().cuda()
feat_3d = batch["feat_3d"].float().cuda()
sinput = SparseTensor(feat_3d, coords_3d)

batch_input = {
    "sinput": sinput,
    "coords_3d": coords_3d,
    "ori_coords_3d": batch["ori_coords_3d"].cuda(),
    "inds_reconstruct": batch["inds_reconstruct"].cuda(),
    "pixel_pooled": batch["pixel_pooled"].cuda(),
    "masks": batch["masks"].cuda(),
    "mask_embeddings": batch["mask_embeddings"].cuda(),
    "mask_valid": batch["mask_valid"].cuda(),
    "x_label": batch["x_label"].cuda(),
    "y_label": batch["y_label"].cuda(),
}

print("\n" + "="*80)
print("模型前向:")
print("="*80)

with torch.no_grad():
    results = model(batch_input)

print(f"outputs: {len(results['outputs'])} 个 batch item")
for b, out_list in enumerate(results['outputs']):
    if len(out_list) > 0:
        logits = out_list[0]["pred_mask_logits"]
        print(f"  Batch {b}: logits shape={logits.shape}, range=[{logits.min().item():.3f}, {logits.max().item():.3f}]")
    else:
        print(f"  Batch {b}: EMPTY (no output)")

print(f"\nmask_valid_from_masks: {results['mask_valid_from_masks'].shape}")
for b in range(results['mask_valid_from_masks'].shape[0]):
    valid_count = results['mask_valid_from_masks'][b].sum().item()
    print(f"  Batch {b}: {valid_count} valid masks")

print("\n" + "="*80)
print("Loss 计算:")
print("="*80)

criterion = Criteria(results, batch_input, threshold=0.5, bce_weight=1.0, dice_weight=1.0)
loss = criterion.loss_pt()

print(f"Loss: {loss.item():.4f}")

# 详细追踪 criterion 内部
print("\n" + "="*80)
print("Criterion 内部追踪:")
print("="*80)

batch_size = len(results["outputs"])
for b in range(batch_size):
    print(f"\n--- Batch {b} ---")
    
    if len(results["outputs"][b]) == 0:
        print("  ❌ 输出为空")
        continue
    
    mask_logits = results["outputs"][b][0]["pred_mask_logits"]
    valid = results["mask_valid_from_masks"][b]
    
    print(f"  mask_logits: {mask_logits.shape}")
    print(f"  valid masks: {valid.sum().item()} / {len(valid)}")
    
    if not valid.any():
        print("  ❌ 没有有效 mask")
        continue
    
    mask_logits_valid = mask_logits[:, valid]
    print(f"  mask_logits_valid: {mask_logits_valid.shape}")
    
    # 检查 keep
    pred_probs = torch.sigmoid(mask_logits_valid)
    mask_hard = (pred_probs > 0.5).float()
    keep = torch.sum(mask_hard, dim=0) > 10
    print(f"  keep (>10 points): {keep.sum().item()} / {len(keep)}")
    
    if not keep.any():
        print("  ❌ 没有 mask 通过 min_points 过滤")
        continue
    
    # GT masks
    mask_2d = results["mask_masks"][b][valid]
    point_mask = results["batch_indices"] == b
    x_idx = batch_input["x_label"][point_mask].float()
    y_idx = batch_input["y_label"][point_mask].float()
    
    print(f"  GT masks: {mask_2d.shape}")
    print(f"  x_label: {len(x_idx)}, 全零={(x_idx==0).all()}")
    print(f"  y_label: {len(y_idx)}, 全零={(y_idx==0).all()}")
    
    if x_idx.numel() == 0:
        print("  ❌ x_label/y_label 为空")
        continue
    
    H, W = mask_2d.shape[1], mask_2d.shape[2]
    orig_W = max(640, x_idx.max().item() + 10)
    orig_H = max(480, y_idx.max().item() + 10)
    scale_x = W / orig_W
    scale_y = H / orig_H
    
    x_idx_scaled = (x_idx * scale_x).long()
    y_idx_scaled = (y_idx * scale_y).long()
    
    valid_bounds = (x_idx_scaled >= 0) & (x_idx_scaled < W) & (y_idx_scaled >= 0) & (y_idx_scaled < H)
    print(f"  缩放: {orig_W}x{orig_H} → {W}x{H} (scale {scale_x:.3f}x{scale_y:.3f})")
    print(f"  有效点(边界内): {valid_bounds.sum().item()} / {len(valid_bounds)} ({valid_bounds.sum().item()/len(valid_bounds)*100:.1f}%)")
    
    if valid_bounds.sum() == 0:
        print("  ❌ 所有点越界")
        continue
    
    print("  ✅ 这个 batch 应该能计算 loss")

print("\n" + "="*80)
print("结论:")
print("="*80)
print(f"Final loss: {loss.item():.4f}")
print(f"Loss requires_grad: {loss.requires_grad}")
