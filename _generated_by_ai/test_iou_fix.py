"""
测试 IoU=0 bug 的修复是否有效。

这个脚本模拟完整的训练和验证流程，验证：
1. 训练时即使模型输出全负 logits，loss 也不会为 0
2. 验证时 IoU 计算正确
"""

import torch
from dataset.open_vocab_dataset_v2 import (
    OpenVocabDatasetV2Config,
    OpenVocabScannetDatasetV2,
    open_vocab_collate_v2,
)
from model.criterion import Criteria
from trainer.open_vocab_trainer_v2 import MetricsTracker


def test_criterion_with_negative_logits():
    """测试 criterion 在模型输出全负 logits 时是否能正常工作。"""
    print("=" * 60)
    print("测试 1: Criterion 在全负 logits 时的表现")
    print("=" * 60)
    
    # 加载数据
    config = OpenVocabDatasetV2Config(
        data_config_path='config/data_scannet_3d.yaml',
        precomputed_dir='/home/featurize/data/pixel_pooled',
        projection_dir='/home/featurize/data/scannet_projections',
        split='train',
        max_samples_ratio=0.01,
        voxel_size=0.05,
    )
    dataset = OpenVocabScannetDatasetV2(config)
    batch = open_vocab_collate_v2([dataset[0], dataset[1]])
    
    device = 'cuda'
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    
    # 模拟模型输出（全负 logits，模拟训练崩溃的情况）
    batch_indices = batch['coords_3d'][:, 0]
    B = batch['masks'].shape[0]
    K_padded = batch['masks'].shape[1]
    
    outputs = []
    for b in range(B):
        n_b = (batch_indices == b).sum().item()
        pred_logits = torch.ones(n_b, K_padded, device=device) * (-3.0)  # 全负
        outputs.append([{'pred_mask_logits': pred_logits}])
    
    results = {
        'outputs': outputs,
        'mask_valid_from_masks': batch['mask_valid'],
        'mask_masks': batch['masks'],
        'batch_indices': batch_indices,
    }
    
    # 测试旧版本（会崩溃）
    print("\n旧版本 (use_keep_filter=True):")
    criteria_old = Criteria(results, batch, use_keep_filter=True)
    loss_old = criteria_old.loss_pt()
    print(f"  Loss: {loss_old.item():.6f}")
    if loss_old.item() == 0:
        print("  ❌ 确认 bug 存在: Loss=0，训练会卡死")
    else:
        print("  意外：旧版本 loss > 0")
    
    # 测试新版本（修复）
    print("\n新版本 (use_keep_filter=False):")
    criteria_new = Criteria(results, batch, use_keep_filter=False)
    loss_new = criteria_new.loss_pt()
    print(f"  Loss: {loss_new.item():.6f}")
    if loss_new.item() > 0:
        print("  ✅ PASSED: Loss > 0，可以正常训练")
        print(f"  修复有效：loss 从 {loss_old.item():.6f} 提升到 {loss_new.item():.6f}")
        return True
    else:
        print("  ❌ FAILED: Loss=0，修复无效")
        return False


def test_validation_iou():
    """测试验证时的 IoU 计算是否正确。"""
    print("\n" + "=" * 60)
    print("测试 2: 验证时的 IoU 计算")
    print("=" * 60)
    
    # 加载 val 数据
    config = OpenVocabDatasetV2Config(
        data_config_path='config/data_scannet_3d.yaml',
        precomputed_dir='/home/featurize/data/pixel_pooled',
        projection_dir='/home/featurize/data/scannet_projections',
        split='val',
        max_samples_ratio=0.01,
        voxel_size=0.05,
    )
    dataset = OpenVocabScannetDatasetV2(config)
    batch = open_vocab_collate_v2([dataset[0]])
    
    device = 'cuda'
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    
    # 构建 GT
    x_label = batch['x_label']
    y_label = batch['y_label']
    masks = batch['masks']
    mask_valid = batch['mask_valid'][0]
    
    mask_2d = masks[0, mask_valid]
    H, W = mask_2d.shape[1], mask_2d.shape[2]
    
    # 检查越界
    valid_mask = (x_label >= 0) & (x_label < W) & (y_label >= 0) & (y_label < H)
    if not valid_mask.all():
        print(f"  Warning: {(~valid_mask).sum()} / {len(valid_mask)} points out of bounds")
    
    x_valid = x_label[valid_mask]
    y_valid = y_label[valid_mask]
    
    gt_3d = mask_2d[:, y_valid, x_valid]
    gt_binary = (gt_3d > 0.5).float().transpose(0, 1)  # (N, K)
    
    print(f"\nGT shape: {gt_binary.shape}")
    print(f"GT positive ratio: {gt_binary.mean()*100:.2f}%")
    
    # 测试 1: 随机预测（应该得到 ~6% IoU）
    pred_random = torch.rand_like(gt_binary)
    metrics_random = MetricsTracker()
    metrics_random.update(pred_random, gt_binary)
    result_random = metrics_random.compute()
    
    print(f"\n随机预测:")
    print(f"  IoU: {result_random['iou']:.4f}")
    print(f"  Accuracy: {result_random['accuracy']:.4f}")
    if result_random['iou'] > 0.01:
        print("  ✅ PASSED: IoU > 0，计算逻辑正确")
    else:
        print("  ❌ FAILED: IoU 太低")
        return False
    
    # 测试 2: 全 0 预测（应该得到 IoU=0）
    pred_zero = torch.zeros_like(gt_binary)
    metrics_zero = MetricsTracker()
    metrics_zero.update(pred_zero, gt_binary)
    result_zero = metrics_zero.compute()
    
    print(f"\n全 0 预测 (模拟模型崩溃):")
    print(f"  IoU: {result_zero['iou']:.4f}")
    print(f"  Accuracy: {result_zero['accuracy']:.4f}")
    if result_zero['iou'] == 0:
        print("  ✅ PASSED: 全 0 预测确实会导致 IoU=0")
    else:
        print("  ❌ FAILED: IoU 应该为 0")
        return False
    
    # 测试 3: 完美预测（应该得到 IoU=1）
    metrics_perfect = MetricsTracker()
    metrics_perfect.update(gt_binary, gt_binary)
    result_perfect = metrics_perfect.compute()
    
    print(f"\n完美预测:")
    print(f"  IoU: {result_perfect['iou']:.4f}")
    print(f"  Accuracy: {result_perfect['accuracy']:.4f}")
    if result_perfect['iou'] > 0.99:
        print("  ✅ PASSED: 完美预测 IoU ≈ 1")
    else:
        print("  ❌ FAILED: IoU 应该接近 1")
        return False
    
    return True


def main():
    print("\n" + "=" * 60)
    print("IoU=0 Bug 修复验证")
    print("=" * 60)
    
    test1_passed = test_criterion_with_negative_logits()
    test2_passed = test_validation_iou()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"测试 1 (Criterion 修复): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"测试 2 (IoU 计算): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！IoU=0 bug 已修复。")
        print("\n修复说明:")
        print("1. 训练时不再使用 keep 过滤，即使模型输出全负 logits 也能正常计算 loss")
        print("2. BCE + Dice loss 会惩罚错误预测，引导模型学习正确的 mask")
        print("3. 验证时 IoU 计算逻辑正确，能准确反映模型性能")
        return 0
    else:
        print("\n❌ 部分测试失败，需要进一步调试。")
        return 1


if __name__ == "__main__":
    exit(main())
