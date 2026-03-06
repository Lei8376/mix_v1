#!/usr/bin/env python3
"""
使用Trainer验证逻辑进行评估，并输出各类别的IoU
不修改训练代码，独立计算各类别指标
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from torch.cuda.amp import autocast

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from trainer.open_vocab_trainer_v2 import OpenVocabTrainerV2Config
from dataset.open_vocab_dataset_v2 import OpenVocabDatasetV2Config, OpenVocabScannetDatasetV2, open_vocab_collate_v2
from model.open_vocab_fusion_v2 import OpenVocabFusionModelV2Config, OpenVocab3DFusionModelV2
from loss.criteria import Criteria
from torch.utils.data import DataLoader
from MinkowskiEngine import SparseTensor
import yaml


# ScanNet 20类标签（和openscene一致）
SCANNET_LABELS_20 = [
    'wall', 'floor', 'cabinet', 'bed', 'chair',
    'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator',
    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'
]


class PerClassMetricsTracker:
    """跟踪各类别的IoU指标（不修改训练代码）"""
    
    def __init__(self, num_classes: int = 20):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        # 每个类别独立统计
        self.class_intersection = [0.0] * self.num_classes
        self.class_union = [0.0] * self.num_classes
        self.class_correct = [0] * self.num_classes
        self.class_total = [0] * self.num_classes
        
        # 全局统计
        self.total_intersection = 0.0
        self.total_union = 0.0
    
    def update(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, threshold: float = 0.5):
        """
        更新各类别指标
        pred_masks: (N, K) 预测概率
        gt_masks: (N, K) GT二值mask
        """
        pred_binary = (pred_masks > threshold).float()
        gt_binary = gt_masks.float()
        
        K = pred_binary.shape[1] if pred_binary.dim() > 1 else 1
        
        for k in range(min(K, self.num_classes)):
            if pred_binary.dim() > 1:
                pred_k = pred_binary[:, k]
                gt_k = gt_binary[:, k]
            else:
                pred_k = pred_binary
                gt_k = gt_binary
            
            # 只有GT有正样本时才计算
            gt_pos = gt_k.sum().item()
            if gt_pos > 0:
                inter_k = (pred_k * gt_k).sum().item()
                union_k = ((pred_k + gt_k) > 0).float().sum().item()
                
                self.class_intersection[k] += inter_k
                self.class_union[k] += union_k
                
                correct_k = (pred_k == gt_k).sum().item()
                total_k = pred_k.numel()
                self.class_correct[k] += correct_k
                self.class_total[k] += total_k
        
        # 全局统计
        intersection = (pred_binary * gt_binary).sum().item()
        union = ((pred_binary + gt_binary) > 0).float().sum().item()
        self.total_intersection += intersection
        self.total_union += union
    
    def compute(self) -> Dict[str, float]:
        """计算各类别和全局IoU"""
        results = {}
        
        # 各类别IoU
        class_ious = []
        for k in range(self.num_classes):
            if self.class_union[k] > 0:
                iou_k = self.class_intersection[k] / self.class_union[k]
                class_ious.append(iou_k)
                
                # 使用类别名称
                class_name = SCANNET_LABELS_20[k] if k < len(SCANNET_LABELS_20) else f"class_{k}"
                results[f"{class_name}_iou"] = iou_k
            else:
                results[f"{SCANNET_LABELS_20[k]}_iou"] = 0.0
        
        # mIoU（各类别平均）
        results["miou"] = sum(class_ious) / len(class_ious) if class_ious else 0.0
        
        # 全局IoU
        results["global_iou"] = (
            self.total_intersection / self.total_union
            if self.total_union > 0 else 0.0
        )
        
        return results


def load_config_from_yaml(yaml_path: str) -> OpenVocabTrainerV2Config:
    """从YAML文件加载配置"""
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 构建数据集配置
    dataset_config = OpenVocabDatasetV2Config(
        data_root_3d=cfg['data_root_3d'],
        data_root_2d=cfg['data_root_2d'],
        precomputed_dir=cfg.get('precomputed_dir'),
        projection_dir=cfg.get('projection_dir'),
        split='val',  # 评估时使用验证集
        voxel_size=cfg.get('voxel_size', 0.05),
        max_points=cfg.get('max_points', 80000),
    )
    
    # 构建模型配置
    model_config = OpenVocabFusionModelV2Config(
        pc_in_dim=3,
        pc_arch=cfg.get('pc_arch', 'MinkUNet18A'),
        pixel_feat_dim=cfg.get('pixel_feat_dim', 768),
        mask_embed_dim=cfg.get('mask_embed_dim', 768),
        fuse_mode=cfg.get('fuse_mode', 'concat'),
        fuse_hidden_dim=cfg.get('fuse_hidden_dim', 512),
        output_dim=cfg.get('output_dim', 768),
    )
    
    # 构建Trainer配置
    trainer_config = OpenVocabTrainerV2Config(
        data_config=dataset_config,
        model_config=model_config,
        batch_size=1,  # 评估时batch_size=1避免OOM
        num_workers=0,  # 避免多进程问题
        lr=cfg.get('lr', 1e-4),
        weight_decay=cfg.get('weight_decay', 1e-4),
        epochs=1,  # 不训练
        log_interval=cfg.get('log_interval', 10),
        val_interval=1,
        checkpoint_dir=cfg.get('checkpoint_dir', 'checkpoints'),
        use_amp=cfg.get('use_amp', False),
        bce_weight=cfg.get('bce_weight', 1.0),
        dice_weight=cfg.get('dice_weight', 1.0),
        min_points_per_mask=cfg.get('min_points_per_mask', 100),
    )
    
    return trainer_config


def main():
    parser = argparse.ArgumentParser(description='使用Trainer验证代码评估模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True,
                        help='训练配置YAML文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备 (cuda/cpu)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("使用Trainer验证代码进行评估")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print("=" * 60)
    print()
    
    # 1. 加载配置
    print("📋 加载配置...")
    config = load_config_from_yaml(args.config)
    print(f"✓ 数据集: {config.data_config.data_root_3d}")
    print(f"✓ 模型架构: {config.model_config.pc_arch}")
    print()
    
    # 2. 创建Trainer（但不训练）
    print("🔧 创建Trainer...")
    trainer = OpenVocabTrainerV2(config, device=args.device)
    print("✓ Trainer创建成功")
    print()
    
    # 3. 加载checkpoint
    print(f"📦 加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # 处理DDP保存的模型（可能有module.前缀）
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    trainer.model.load_state_dict(state_dict)
    
    # 打印checkpoint信息
    if 'epoch' in checkpoint:
        print(f"✓ Epoch: {checkpoint['epoch']}")
    if 'best_iou' in checkpoint:
        print(f"✓ 训练时最佳IoU: {checkpoint['best_iou']:.4f}")
    if 'best_loss' in checkpoint:
        print(f"✓ 训练时最佳Loss: {checkpoint['best_loss']:.4f}")
    print()
    
    # 4. 运行验证（不训练）
    print("=" * 60)
    print("🚀 开始评估...")
    print("=" * 60)
    print()
    
    # 调用trainer的验证函数
    metrics = trainer._validate(epoch=0)
    
    # 5. 打印结果
    print()
    print("=" * 60)
    print("📊 评估结果")
    print("=" * 60)
    print()
    
    if 'loss' in metrics:
        print(f"验证Loss: {metrics['loss']:.4f}")
        print()
    
    # 打印各个指标
    print("指标详情:")
    print("-" * 60)
    for key, value in sorted(metrics.items()):
        if key != 'loss':
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.4f}")
            else:
                print(f"  {key:30s}: {value}")
    print()
    
    # 如果有IoU指标，特别标注
    if 'iou' in metrics or 'mean_iou' in metrics or 'miou' in metrics:
        print("=" * 60)
        print("🎯 IoU指标")
        print("=" * 60)
        for key, value in metrics.items():
            if 'iou' in key.lower():
                print(f"  {key:30s}: {value:.4f}")
    
    print()
    print("✅ 评估完成!")


if __name__ == "__main__":
    main()
