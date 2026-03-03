"""
诊断训练问题的关键指标检查脚本

检查项：
1. pred_logits 范围（验证是 logits 还是概率）
2. point→voxel 匹配率（应该 100%）
3. 预计算投影 fallback 统计（应该 0）
4. 验证集不同阈值下的 mIoU（找最佳阈值）
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.open_vocab_dataset_v2 import (
    OpenVocabDatasetV2Config,
    OpenVocabScannetDatasetV2,
    open_vocab_collate_v2,
)
from model.open_vocab_fusion_v2 import (
    OpenVocabFusionModelV2Config,
    OpenVocab3DFusionModelV2,
)
from trainer.open_vocab_trainer_v2 import MetricsTracker


class DiagnosticRunner:
    """诊断运行器"""
    
    def __init__(self, config_path: str, checkpoint_path: str = None):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 诊断结果
        self.results = {
            'pred_logits_range': [],
            'matched_ratios': [],
            'fallback_count': 0,
            'total_samples': 0,
            'threshold_miou': {},
        }
    
    def check_dataset_fallback(self):
        """检查数据集加载时的 fallback 情况"""
        print("\n" + "="*60)
        print("🔍 检查 1: 统计预计算投影的 fallback 情况")
        print("="*60)
        
        # 创建数据集
        dataset_cfg = self.config.get('dataset', {})
        repo_root = Path(__file__).parent.parent
        
        precomputed_dir = dataset_cfg.get('precomputed_dir', '')
        if precomputed_dir and not os.path.isabs(precomputed_dir):
            precomputed_dir = repo_root / precomputed_dir
        
        projection_dir = dataset_cfg.get('projection_dir', None)
        if projection_dir and not os.path.isabs(projection_dir):
            projection_dir = repo_root / projection_dir
        
        data_config_path = dataset_cfg.get('data_config_path', 'config/data_scannet_3d.yaml')
        if not os.path.isabs(data_config_path):
            data_config_path = repo_root / data_config_path
        
        config = OpenVocabDatasetV2Config(
            data_config_path=str(data_config_path),
            precomputed_dir=str(precomputed_dir) if precomputed_dir else None,
            projection_dir=str(projection_dir) if projection_dir else None,
            split='train',
            scannet200=dataset_cfg.get('scannet200', False),
            voxel_size=dataset_cfg.get('voxel_size', 0.05),
            max_samples=20,  # 只检查前20个样本
        )
        
        print(f"数据配置路径: {data_config_path}")
        print(f"预计算目录: {precomputed_dir}")
        print(f"投影目录: {projection_dir}")
        
        # 临时修改数据集加载函数，添加 fallback 统计
        from dataset import open_vocab_dataset_v2
        original_load = open_vocab_dataset_v2._load_3d_with_precomputed_projection
        
        fallback_list = []
        
        def patched_load(*args, **kwargs):
            result = original_load(*args, **kwargs)
            if result is None:
                scene_name = args[2] if len(args) > 2 else kwargs.get('scene_name', 'unknown')
                frame_stem = args[4] if len(args) > 4 else kwargs.get('frame_stem', 'unknown')
                fallback_list.append((scene_name, frame_stem))
            return result
        
        open_vocab_dataset_v2._load_3d_with_precomputed_projection = patched_load
        
        try:
            dataset = OpenVocabScannetDatasetV2(config)
            print(f"✅ 数据集创建成功，共 {len(dataset)} 个样本")
            
            # 加载前20个样本
            for i in tqdm(range(min(20, len(dataset))), desc="检查样本"):
                try:
                    sample = dataset[i]
                except Exception as e:
                    print(f"  ⚠️  样本 {i} 加载失败: {e}")
            
            # 恢复原函数
            open_vocab_dataset_v2._load_3d_with_precomputed_projection = original_load
            
            self.results['fallback_count'] = len(fallback_list)
            self.results['total_samples'] = min(20, len(dataset))
            
            print(f"\n📊 Fallback 统计:")
            print(f"  - 检查样本数: {self.results['total_samples']}")
            print(f"  - Fallback 次数: {self.results['fallback_count']}")
            
            if fallback_list:
                print(f"  ⚠️  以下样本缺少预计算投影（fallback 到运行时投影）:")
                for scene, frame in fallback_list[:10]:
                    print(f"      - {scene}/{frame}")
                if len(fallback_list) > 10:
                    print(f"      ... 还有 {len(fallback_list) - 10} 个")
            else:
                print(f"  ✅ 所有样本都使用了预计算投影（无 fallback）")
            
        except Exception as e:
            print(f"❌ 数据集检查失败: {e}")
            import traceback
            traceback.print_exc()
    
    def check_model_logits_and_matching(self):
        """检查模型输出的 logits 范围和点云匹配率"""
        print("\n" + "="*60)
        print("🔍 检查 2: pred_logits 范围 和 point→voxel 匹配率")
        print("="*60)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建数据集
        dataset_cfg = self.config.get('dataset', {})
        repo_root = Path(__file__).parent.parent
        
        precomputed_dir = dataset_cfg.get('precomputed_dir', '')
        if precomputed_dir and not os.path.isabs(precomputed_dir):
            precomputed_dir = repo_root / precomputed_dir
        
        projection_dir = dataset_cfg.get('projection_dir', None)
        if projection_dir and not os.path.isabs(projection_dir):
            projection_dir = repo_root / projection_dir
        
        data_config_path = dataset_cfg.get('data_config_path', 'config/data_scannet_3d.yaml')
        if not os.path.isabs(data_config_path):
            data_config_path = repo_root / data_config_path
        
        config = OpenVocabDatasetV2Config(
            data_config_path=str(data_config_path),
            precomputed_dir=str(precomputed_dir) if precomputed_dir else None,
            projection_dir=str(projection_dir) if projection_dir else None,
            split='val',
            scannet200=dataset_cfg.get('scannet200', False),
            voxel_size=dataset_cfg.get('voxel_size', 0.05),
            max_samples=5,  # 只检查5个样本
            aug=False,
        )
        
        dataset = OpenVocabScannetDatasetV2(config)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=open_vocab_collate_v2,
        )
        
        # 创建模型
        model_cfg = self.config.get('model', {})
        model_config = OpenVocabFusionModelV2Config(
            device=device,
            pc_arch=model_cfg.get('pc_arch', 'MinkUNet34C'),
        )
        
        model = OpenVocab3DFusionModelV2(model_config).to(device)
        
        # 加载 checkpoint（如果提供）
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"加载 checkpoint: {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        
        model.eval()
        
        # 修改模型的 forward，记录中间统计
        original_forward = model.forward
        
        def patched_forward(batch_input):
            # 调用原始 forward
            results = original_forward(batch_input)
            
            # 统计 pred_logits 范围
            for b_idx, output_list in enumerate(results['outputs']):
                if len(output_list) > 0:
                    logits = output_list[0]['pred_mask_logits']
                    min_val = logits.min().item()
                    max_val = logits.max().item()
                    self.results['pred_logits_range'].append((min_val, max_val))
            
            # 统计 point→voxel 匹配率（需要重新计算一遍）
            from MinkowskiEngine import SparseTensor
            sinput = batch_input['sinput']
            input_coords = batch_input["coords_3d"].int().to(sinput.device)
            voxel_coords = sinput.C
            
            # 使用相同的哈希逻辑
            def _encode_coords(coords_4d):
                c = coords_4d.long() + 20000
                BASE = 40001
                return c[:, 0] * (BASE ** 3) + c[:, 1] * (BASE ** 2) + c[:, 2] * BASE + c[:, 3]
            
            voxel_hash = _encode_coords(voxel_coords)
            input_hash = _encode_coords(input_coords)
            
            sort_idx = torch.argsort(voxel_hash)
            sorted_voxel_hash = voxel_hash[sort_idx]
            
            pos = torch.searchsorted(sorted_voxel_hash, input_hash)
            pos = pos.clamp(max=sorted_voxel_hash.shape[0] - 1)
            
            matched = sorted_voxel_hash[pos] == input_hash
            matched_ratio = matched.float().mean().item()
            self.results['matched_ratios'].append(matched_ratio)
            
            return results
        
        model.forward = patched_forward
        
        print(f"检查 {min(5, len(dataset))} 个验证集样本...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="处理批次"):
                # 移动到设备
                for key in ['coords_3d', 'feat_3d', 'ori_coords_3d', 'pixel_pooled', 
                           'masks', 'mask_embeddings', 'mask_valid']:
                    if key in batch and isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # 创建 SparseTensor
                from MinkowskiEngine import SparseTensor
                batch['sinput'] = SparseTensor(batch['feat_3d'], batch['coords_3d'].int())
                
                try:
                    _ = model(batch)
                except Exception as e:
                    print(f"  ⚠️  批次处理失败: {e}")
        
        # 恢复原函数
        model.forward = original_forward
        
        # 统计结果
        print(f"\n📊 pred_logits 范围统计 ({len(self.results['pred_logits_range'])} 个批次):")
        if self.results['pred_logits_range']:
            all_mins = [r[0] for r in self.results['pred_logits_range']]
            all_maxs = [r[1] for r in self.results['pred_logits_range']]
            print(f"  - Min: {min(all_mins):.4f} ~ {max(all_mins):.4f}")
            print(f"  - Max: {min(all_maxs):.4f} ~ {max(all_maxs):.4f}")
            
            avg_min = sum(all_mins) / len(all_mins)
            avg_max = sum(all_maxs) / len(all_maxs)
            print(f"  - 平均范围: [{avg_min:.4f}, {avg_max:.4f}]")
            
            if avg_min >= 0 and avg_max <= 1:
                print(f"  ⚠️  **警告**: logits 范围在 [0, 1]，可能是概率而非 logits！")
            elif avg_min < -5 and avg_max > 5:
                print(f"  ✅ logits 范围正常（像是余弦相似度 * scale）")
        
        print(f"\n📊 point→voxel 匹配率统计 ({len(self.results['matched_ratios'])} 个批次):")
        if self.results['matched_ratios']:
            avg_ratio = sum(self.results['matched_ratios']) / len(self.results['matched_ratios'])
            min_ratio = min(self.results['matched_ratios'])
            max_ratio = max(self.results['matched_ratios'])
            
            print(f"  - 平均匹配率: {avg_ratio*100:.4f}%")
            print(f"  - 最低匹配率: {min_ratio*100:.4f}%")
            print(f"  - 最高匹配率: {max_ratio*100:.4f}%")
            
            if min_ratio < 0.999:
                print(f"  ⚠️  **警告**: 存在匹配率 <99.9% 的批次，可能导致训练不稳定！")
                print(f"       未匹配的点会被错误映射到 voxel[0]")
            else:
                print(f"  ✅ 所有批次的匹配率都 ≥99.9%")
    
    def check_threshold_sweep(self):
        """在验证集上扫描不同阈值的 mIoU"""
        print("\n" + "="*60)
        print("🔍 检查 3: 不同阈值下的 mIoU（找最佳阈值）")
        print("="*60)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建数据集
        dataset_cfg = self.config.get('dataset', {})
        repo_root = Path(__file__).parent.parent
        
        precomputed_dir = dataset_cfg.get('precomputed_dir', '')
        if precomputed_dir and not os.path.isabs(precomputed_dir):
            precomputed_dir = repo_root / precomputed_dir
        
        projection_dir = dataset_cfg.get('projection_dir', None)
        if projection_dir and not os.path.isabs(projection_dir):
            projection_dir = repo_root / projection_dir
        
        data_config_path = dataset_cfg.get('data_config_path', 'config/data_scannet_3d.yaml')
        if not os.path.isabs(data_config_path):
            data_config_path = repo_root / data_config_path
        
        config = OpenVocabDatasetV2Config(
            data_config_path=str(data_config_path),
            precomputed_dir=str(precomputed_dir) if precomputed_dir else None,
            projection_dir=str(projection_dir) if projection_dir else None,
            split='val',
            scannet200=dataset_cfg.get('scannet200', False),
            voxel_size=dataset_cfg.get('voxel_size', 0.05),
            max_samples=20,  # 检查20个样本
            aug=False,
        )
        
        dataset = OpenVocabScannetDatasetV2(config)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=open_vocab_collate_v2,
        )
        
        # 创建模型
        model_cfg = self.config.get('model', {})
        model_config = OpenVocabFusionModelV2Config(
            device=device,
            pc_arch=model_cfg.get('pc_arch', 'MinkUNet34C'),
        )
        
        model = OpenVocab3DFusionModelV2(model_config).to(device)
        
        # 加载 checkpoint（如果提供）
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"加载 checkpoint: {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print("⚠️  未提供 checkpoint，使用随机初始化的模型（结果仅供参考）")
        
        model.eval()
        
        # 不同阈值
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        metrics_per_threshold = {t: MetricsTracker() for t in thresholds}
        
        print(f"在 {len(dataset)} 个验证样本上测试阈值 {thresholds}...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="处理批次"):
                # 移动到设备
                for key in ['coords_3d', 'feat_3d', 'ori_coords_3d', 'pixel_pooled', 
                           'masks', 'mask_embeddings', 'mask_valid', 'x_label', 'y_label']:
                    if key in batch and isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # 创建 SparseTensor
                from MinkowskiEngine import SparseTensor
                batch['sinput'] = SparseTensor(batch['feat_3d'], batch['coords_3d'].int())
                
                try:
                    results = model(batch)
                    
                    # 对每个 batch item 计算 metrics
                    for b in range(len(results['outputs'])):
                        if len(results['outputs'][b]) == 0:
                            continue
                        
                        pred_logits = results['outputs'][b][0]['pred_mask_logits']
                        valid = results['mask_valid_from_masks'][b]
                        
                        # Get GT masks
                        mask_2d = results['mask_masks'][b][valid]
                        point_mask = results['batch_indices'] == b
                        x_idx = batch['x_label'][point_mask].float()
                        y_idx = batch['y_label'][point_mask].float()
                        
                        if x_idx.numel() == 0:
                            continue
                        
                        H, W = mask_2d.shape[1], mask_2d.shape[2]
                        
                        # 缩放逻辑（与训练一致）
                        x_max = x_idx.max().item()
                        y_max = y_idx.max().item()
                        need_scale = (x_max > W + 20) or (y_max > H + 20)
                        
                        if need_scale:
                            orig_W = max(640, x_max + 10)
                            orig_H = max(480, y_max + 10)
                            x_idx = (x_idx * W / orig_W).long()
                            y_idx = (y_idx * H / orig_H).long()
                        else:
                            x_idx = x_idx.long()
                            y_idx = y_idx.long()
                        
                        # 过滤越界点
                        valid_mask = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
                        if valid_mask.sum() == 0:
                            continue
                        
                        x_idx = x_idx[valid_mask]
                        y_idx = y_idx[valid_mask]
                        pred_logits_filtered = pred_logits[valid_mask, :]
                        
                        gt_3d = mask_2d[:, y_idx, x_idx]
                        gt_3d = (gt_3d > 0.5).float().transpose(0, 1)
                        
                        pred_valid = pred_logits_filtered[:, valid]
                        
                        # 过滤 GT 正样本数不足的 mask
                        gt_pos = gt_3d.sum(dim=0)
                        keep_gt = gt_pos >= 10
                        
                        if keep_gt.any():
                            pred_valid = pred_valid[:, keep_gt]
                            gt_3d = gt_3d[:, keep_gt]
                            pred_probs = torch.sigmoid(pred_valid).float()
                            
                            # 对每个阈值更新 metrics
                            for threshold in thresholds:
                                metrics_per_threshold[threshold].update(pred_probs, gt_3d, threshold=threshold)
                
                except Exception as e:
                    print(f"  ⚠️  批次处理失败: {e}")
        
        # 输出结果
        print(f"\n📊 不同阈值下的 mIoU:")
        print(f"{'阈值':<8} {'IoU':<10} {'mIoU':<10} {'Acc':<10} {'mAcc':<10}")
        print("-" * 50)
        
        for threshold in thresholds:
            metrics = metrics_per_threshold[threshold].compute()
            self.results['threshold_miou'][threshold] = metrics['miou']
            print(f"{threshold:<8.1f} {metrics['iou']:<10.4f} {metrics['miou']:<10.4f} "
                  f"{metrics['accuracy']:<10.4f} {metrics.get('macc', 0):<10.4f}")
        
        # 找最佳阈值
        best_threshold = max(self.results['threshold_miou'].items(), key=lambda x: x[1])
        print(f"\n✅ 最佳阈值: {best_threshold[0]:.1f} (mIoU={best_threshold[1]:.4f})")
        
        if best_threshold[0] != 0.5:
            improvement = (best_threshold[1] - self.results['threshold_miou'][0.5]) / self.results['threshold_miou'][0.5] * 100
            print(f"   相比默认阈值 0.5，mIoU 提升: {improvement:.1f}%")
    
    def generate_report(self):
        """生成诊断报告"""
        print("\n" + "="*60)
        print("📋 诊断报告")
        print("="*60)
        
        print("\n## 问题汇总\n")
        
        issues = []
        
        # 检查 logits 范围
        if self.results['pred_logits_range']:
            all_mins = [r[0] for r in self.results['pred_logits_range']]
            all_maxs = [r[1] for r in self.results['pred_logits_range']]
            avg_min = sum(all_mins) / len(all_mins)
            avg_max = sum(all_maxs) / len(all_maxs)
            
            if avg_min >= 0 and avg_max <= 1:
                issues.append({
                    'severity': 'HIGH',
                    'title': 'pred_logits 可能是概率而非 logits',
                    'detail': f'logits 范围: [{avg_min:.4f}, {avg_max:.4f}]，应该是未限制范围的值',
                    'fix': '检查模型输出是否错误使用了 sigmoid'
                })
            else:
                print("✅ pred_logits 范围正常")
        
        # 检查匹配率
        if self.results['matched_ratios']:
            min_ratio = min(self.results['matched_ratios'])
            if min_ratio < 0.999:
                issues.append({
                    'severity': 'HIGH',
                    'title': 'point→voxel 匹配率不足 100%',
                    'detail': f'最低匹配率: {min_ratio*100:.4f}%，未匹配的点会被错误映射到 voxel[0]',
                    'fix': '添加断言: assert matched_ratio > 0.999，并检查坐标量化逻辑'
                })
            else:
                print("✅ point→voxel 匹配率正常 (≥99.9%)")
        
        # 检查 fallback
        if self.results['fallback_count'] > 0:
            ratio = self.results['fallback_count'] / self.results['total_samples'] * 100
            issues.append({
                'severity': 'MEDIUM',
                'title': '部分样本使用了 fallback 投影',
                'detail': f'{self.results["fallback_count"]}/{self.results["total_samples"]} 样本 ({ratio:.1f}%) 缺少预计算投影',
                'fix': '训练时禁止 fallback：if out_3d is None and split=="train": raise RuntimeError(...)'
            })
        else:
            print("✅ 所有样本都使用了预计算投影（无 fallback）")
        
        # 检查阈值
        if self.results['threshold_miou']:
            best_t = max(self.results['threshold_miou'].items(), key=lambda x: x[1])
            default_miou = self.results['threshold_miou'].get(0.5, 0)
            if best_t[0] != 0.5 and default_miou > 0:
                improvement = (best_t[1] - default_miou) / default_miou * 100
                if improvement > 5:
                    issues.append({
                        'severity': 'MEDIUM',
                        'title': '默认阈值 0.5 不是最佳阈值',
                        'detail': f'最佳阈值: {best_t[0]:.1f} (mIoU={best_t[1]:.4f})，比 0.5 高 {improvement:.1f}%',
                        'fix': f'在 MetricsTracker.update() 中使用 threshold={best_t[0]}'
                    })
        
        if issues:
            for i, issue in enumerate(issues, 1):
                severity_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
                print(f"\n{severity_emoji[issue['severity']]} 问题 {i}: {issue['title']}")
                print(f"   严重程度: {issue['severity']}")
                print(f"   详情: {issue['detail']}")
                print(f"   修复建议: {issue['fix']}")
        else:
            print("\n🎉 未发现严重问题！")
        
        print("\n" + "="*60)
        print("诊断完成")
        print("="*60)
    
    def run_all(self):
        """运行所有检查"""
        self.check_dataset_fallback()
        self.check_model_logits_and_matching()
        self.check_threshold_sweep()
        self.generate_report()


def main():
    parser = argparse.ArgumentParser(description='诊断训练问题')
    parser.add_argument('--config', type=str, required=True, help='训练配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型 checkpoint 路径（可选）')
    
    args = parser.parse_args()
    
    runner = DiagnosticRunner(args.config, args.checkpoint)
    runner.run_all()


if __name__ == '__main__':
    main()
