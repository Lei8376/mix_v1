"""
评估 Mix 模型的 mIoU 和各类别 IoU

用法:
    # 基本用法（需要指定checkpoint）
    python eval_model.py --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth
    
    # 指定配置文件
    python eval_model.py --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth --config config/train_scannet_v2.yaml
    
    # 多次重复评估（处理体素化随机性）
    python eval_model.py --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth --test-repeats 3
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from MinkowskiEngine import SparseTensor

# 添加当前目录和 openscene 路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'openscene'))

from dataset.open_vocab_dataset_v2 import (
    OpenVocabDatasetV2Config,
    OpenVocabScannetDatasetV2,
    open_vocab_collate_v2,
)
from model.open_vocab_fusion_v2 import (
    OpenVocabFusionModelV2Config,
    OpenVocab3DFusionModelV2,
)

# 导入 openscene 的 metric 模块和标签
try:
    from util import metric
    from dataset.label_constants import SCANNET_LABELS_20
    print("✓ 成功导入 openscene 工具模块")
except ImportError as e:
    print(f"错误: 无法导入 openscene 模块: {e}")
    print("请确保 openscene 目录存在且包含 util/metric.py 和 dataset/label_constants.py")
    sys.exit(1)

import yaml


def load_yaml_config(config_path: str) -> Dict:
    """加载 YAML 配置文件"""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int = 1342):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_checkpoint(model, checkpoint_path: str, device: str = "cuda"):
    """加载模型权重"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
    
    print(f"=> 正在加载权重 '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    
    try:
        # 尝试直接加载
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"=> 成功加载权重 (epoch {checkpoint.get('epoch', 'unknown')})")
    except Exception as ex:
        print(f"尝试直接加载失败: {ex}")
        # 尝试处理 DDP 模型的权重
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                # 移除 module. 前缀（DDP 训练的模型）
                k = k[7:]
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        print(f"=> 成功加载权重（处理 DDP 格式后，epoch {checkpoint.get('epoch', 'unknown')}）")
    
    return checkpoint.get('epoch', 0)


def extract_text_features(labelset, device='cuda'):
    """
    提取文本特征（使用 CLIP）
    这里简化处理，实际应该使用 mix 模型中的 CLIP encoder
    """
    # TODO: 这里需要使用与训练时相同的 CLIP 模型来提取文本特征
    # 暂时返回随机特征作为占位符
    import torch.nn.functional as F
    
    try:
        import clip
        model, _ = clip.load("ViT-B/16", device=device)
        
        text_tokens = clip.tokenize(labelset).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features.half()
    except ImportError:
        print("警告: 无法导入 CLIP，使用随机特征（仅用于测试结构）")
        # 返回随机归一化特征
        features = torch.randn(len(labelset), 512, device=device)
        return F.normalize(features, dim=-1).half()


@torch.no_grad()
def evaluate_model(
    model,
    val_loader,
    device: str = "cuda",
    test_repeats: int = 1,
    dataset_name: str = "scannet_3d",
):
    """
    评估模型的 mIoU 和各类别 IoU
    
    Args:
        model: 待评估的模型
        val_loader: 验证数据加载器
        device: 设备
        test_repeats: 重复测试次数（用于处理 MinkowskiNet 体素化的随机性）
        dataset_name: 数据集名称
    """
    torch.backends.cudnn.enabled = False
    model.eval()
    
    # 准备文本特征和标签集
    labelset = list(SCANNET_LABELS_20)
    labelset[-1] = 'other'  # 修改 'otherfurniture' 为 'other'
    
    print(f"\n{'='*60}")
    print(f"开始评估模型")
    print(f"数据集: {dataset_name}")
    print(f"类别数: {len(labelset)}")
    print(f"重复次数: {test_repeats}")
    print(f"{'='*60}\n")
    
    # 提取文本特征
    text_features = extract_text_features(labelset, device=device)
    
    store = 0.0
    for rep_i in range(test_repeats):
        preds, gts = [], []
        
        print(f"\n评估轮次 {rep_i+1}/{test_repeats}...\n")
        
        # 设置随机种子（用于重复评估）
        if rep_i > 0:
            seed = np.random.randint(10000)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 遍历验证集
        for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {rep_i+1}")):
            try:
                # 解包 batch
                if len(batch) == 7:
                    coords, feat, label, feat_3d, mask, inds_reverse, _ = batch
                elif len(batch) == 6:
                    coords, feat, label, feat_3d, mask, inds_reverse = batch
                else:
                    print(f"警告: batch 长度为 {len(batch)}，跳过...")
                    continue
                
                # 构建稀疏输入
                sinput = SparseTensor(
                    feat.cuda(non_blocking=True),
                    coords.cuda(non_blocking=True)
                )
                
                # 模型推理 - 获取 3D 特征
                predictions = model(sinput)
                
                # 反向索引到原始点云
                predictions = predictions[inds_reverse, :]
                
                # 与文本特征计算相似度
                pred = predictions.half() @ text_features.t()
                
                # 获取预测类别
                logits_pred = torch.max(pred, 1)[1].cpu()
                
                # 收集预测和真实标签
                if test_repeats == 1:
                    preds.append(logits_pred)
                else:
                    preds.append(pred.cpu())
                
                gts.append(label.cpu())
                
            except Exception as e:
                print(f"处理 batch {i} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 合并所有预测和真实标签
        if len(preds) == 0 or len(gts) == 0:
            print("警告: 没有有效的预测结果")
            continue
        
        gt = torch.cat(gts)
        pred = torch.cat(preds)
        
        # 计算 IoU
        if test_repeats == 1:
            pred_logit = pred
        else:
            pred_logit = pred.float().max(1)[1]
        
        print(f"\n{'='*60}")
        print(f"评估轮次 {rep_i+1} 结果:")
        print(f"{'='*60}")
        
        current_iou = metric.evaluate(
            pred_logit.numpy(),
            gt.numpy(),
            dataset=dataset_name,
            stdout=True
        )
        
        if test_repeats > 1:
            store = pred + store
            store_logit = store.float().max(1)[1]
            
            print(f"\n{'='*60}")
            print(f"累积评估结果 (轮次 1-{rep_i+1}):")
            print(f"{'='*60}")
            
            accumu_iou = metric.evaluate(
                store_logit.numpy(),
                gt.numpy(),
                stdout=True,
                dataset=dataset_name
            )
            
            print(f"\n当前轮次 mIoU: {current_iou:.4f}")
            print(f"累积 mIoU: {accumu_iou:.4f}")
    
    print(f"\n{'='*60}")
    print("评估完成!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="评估 Mix 模型的 mIoU 和各类别 IoU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 必需参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型权重路径，例如: checkpoints/full.1/checkpoint_epoch_19.pth"
    )
    
    # 可选参数
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_scannet_v2.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data-config-path",
        type=str,
        default="config/data_scannet_3d.yaml",
        help="数据配置文件路径"
    )
    parser.add_argument(
        "--precomputed-dir",
        type=str,
        default="",
        help="预计算特征目录"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="评估 batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="数据加载 worker 数量"
    )
    parser.add_argument(
        "--test-repeats",
        type=int,
        default=1,
        help="重复评估次数（处理体素化随机性）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda 或 cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1342,
        help="随机种子"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="scannet_3d",
        help="数据集名称 (用于 metric 计算)"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = "cpu"
    
    # 获取项目根目录
    repo_root = os.path.abspath(os.path.dirname(__file__))
    
    # 加载 YAML 配置
    yaml_config = load_yaml_config(args.config)
    
    # 配置数据集
    _dataset = yaml_config.get("dataset") or {}
    precomputed_dir = _dataset.get("precomputed_dir", args.precomputed_dir)
    if precomputed_dir and not os.path.isabs(precomputed_dir):
        precomputed_dir = os.path.join(repo_root, precomputed_dir)
    
    data_config_path = _dataset.get("data_config_path", args.data_config_path)
    if data_config_path and not os.path.isabs(data_config_path):
        data_config_path = os.path.join(repo_root, data_config_path)
    
    # 创建验证数据集配置
    dataset_config = OpenVocabDatasetV2Config(
        data_config_path=data_config_path,
        precomputed_dir=precomputed_dir if precomputed_dir and os.path.exists(precomputed_dir) else None,
        split="val",  # 使用验证集
        scannet200=_dataset.get("scannet200", False),
        voxel_size=_dataset.get("voxel_size", 0.05),
        aug=False,  # 评估时不使用数据增强
        eval_all=True,
        input_color=False,
    )
    
    print(f"\n{'='*60}")
    print("配置信息:")
    print(f"  权重文件: {args.checkpoint}")
    print(f"  配置文件: {args.config}")
    print(f"  数据配置: {data_config_path}")
    print(f"  预计算目录: {precomputed_dir if precomputed_dir else '无'}")
    print(f"  设备: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  重复次数: {args.test_repeats}")
    print(f"{'='*60}\n")
    
    # 创建数据加载器
    print("创建验证数据集...")
    val_dataset = OpenVocabScannetDatasetV2(dataset_config)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=open_vocab_collate_v2,
    )
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"验证集 batch 数: {len(val_loader)}")
    
    # 创建模型
    print("\n创建模型...")
    _model = yaml_config.get("model") or {}
    model_config = OpenVocabFusionModelV2Config(
        device=device,
        pc_arch=_model.get("pc_arch", "MinkUNet34C"),
    )
    
    model = OpenVocab3DFusionModelV2(model_config)
    model = model.to(device)
    
    # 加载权重
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(repo_root, checkpoint_path)
    
    epoch = load_checkpoint(model, checkpoint_path, device)
    
    # 评估模型
    evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        test_repeats=args.test_repeats,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
