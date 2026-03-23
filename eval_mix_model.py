#!/usr/bin/env python3
"""
评估 Mix 开放词汇3D模型的 mIoU (参考 OpenScene 评估方式)

用法:
    # 基本用法
    python eval_mix_model.py --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth
    
    # 指定数据集和配置
    python eval_mix_model.py \
        --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth \
        --data-root /path/to/scannet_3d \
        --config config/train_scannet_v2.yaml

环境: source /home/featurize/work/envs/mix_backup/bin/activate
"""

import os
import sys
import random
import argparse
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..', 'openscene'))

# 导入必要模块
try:
    from MinkowskiEngine import SparseTensor
    import yaml
    import clip
    print("✓ 成功导入基础模块")
except ImportError as e:
    print(f"错误: 导入基础模块失败 - {e}")
    sys.exit(1)

# 导入 openscene 模块
try:
    from util import metric
    from dataset.label_constants import SCANNET_LABELS_20
    print("✓ 成功导入 openscene 模块")
except ImportError as e:
    print(f"错误: 导入 openscene 模块失败 - {e}")
    sys.exit(1)

# 导入 Mix 模型模块
try:
    from model.open_vocab_fusion_v2 import OpenVocabFusionModelV2Config, OpenVocab3DFusionModelV2
    from dataset.open_vocab_dataset_v2 import OpenVocabDatasetV2Config, OpenVocabScannetDatasetV2, open_vocab_collate_v2
    print("✓ 成功导入 Mix 模型模块")
except ImportError as e:
    print(f"错误: 导入 Mix 模块失败 - {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python 路径: {sys.path[:3]}")
    sys.exit(1)


def get_logger():
    """创建logger"""
    logger_name = "eval-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def load_checkpoint(model, checkpoint_path, logger):
    """加载模型权重"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
    
    logger.info(f"=> 加载权重 '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info(f"=> 成功加载权重 (epoch {checkpoint.get('epoch', 'unknown')})")
    except Exception as ex:
        logger.info(f"直接加载失败，尝试处理DDP格式: {ex}")
        # 处理 DDP 模型权重
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                k = k[7:]  # 移除 'module.' 前缀
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        logger.info('成功加载并处理了DDP模型权重')
    
    return checkpoint.get('epoch', 0)


def precompute_text_features(labelset, clip_version="ViT-L/14@336px"):
    """
    提取文本特征（开放词汇）
    使用 CLIP 将类别名称编码为特征向量
    """
    logger = logging.getLogger("eval-logger")
    logger.info(f"提取 {len(labelset)} 个类别的文本特征...")
    logger.info(f"使用 CLIP 模型: {clip_version}")
    
    # 直接使用 CLIP 提取文本特征（避免导入 open3d）
    try:
        clip_model, _ = clip.load(clip_version, device='cuda', jit=False)
        logger.info("✓ CLIP 模型加载成功")
    except Exception as e:
        logger.warning(f"加载 {clip_version} 失败: {e}，尝试使用 ViT-B/16")
        clip_model, _ = clip.load("ViT-B/16", device='cuda', jit=False)
    
    # 编码文本
    text_tokens = clip.tokenize(labelset).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    logger.info(f"✓ 文本特征维度: {text_features.shape}")
    return text_features


@torch.no_grad()
def evaluate(model, val_loader, labelset, text_features, args, logger):
    """
    评估模型 - 参考 OpenScene 的评估流程
    
    返回:
        mean_iou: 总体 mIoU
    """
    torch.backends.cudnn.enabled = False
    model.eval()
    
    logger.info(f"\n{'='*60}")
    logger.info("开始评估...")
    logger.info(f"类别数: {len(labelset)}")
    logger.info(f"测试重复次数: {args.test_repeats}")
    logger.info(f"{'='*60}\n")
    
    # 显示类别列表
    logger.info("评估类别:")
    for i, label in enumerate(labelset[:-1]):  # 不包括最后的 'unlabeled'
        logger.info(f"  {i:2d}: {label}")
    logger.info("")
    
    store = 0.0
    for rep_i in range(args.test_repeats):
        preds, gts = [], []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"评估轮次 {rep_i+1}/{args.test_repeats}")
        logger.info(f"{'='*60}\n")
        
        # 设置随机种子（用于处理体素化的随机性）
        if rep_i > 0:
            seed = np.random.randint(10000)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 遍历数据集
        for i, batch in enumerate(tqdm(val_loader, desc=f"轮次 {rep_i+1}")):
            try:
                # 解包 batch - Mix 模型的格式
                if len(batch) == 7:
                    coords, feat, label, feat_3d, mask, inds_reverse, coords_3d = batch
                elif len(batch) == 6:
                    coords, feat, label, feat_3d, mask, inds_reverse = batch
                    coords_3d = coords  # 如果没有单独的 coords_3d，使用 coords
                else:
                    logger.warning(f"批次 {i} 格式不对，跳过 (长度={len(batch)})")
                    continue
                
                # 将数据移到GPU
                coords = coords.cuda(non_blocking=True)
                feat = feat.cuda(non_blocking=True)
                label = label.cpu()  # GT标签保持在CPU
                
                # 构建 SparseTensor 输入
                sinput = SparseTensor(feat, coords)
                
                # 构建 Mix 模型的输入字典
                # Mix 模型需要 3D 特征直接作为输出（已经预计算好）
                # 我们使用 fusion 模式：直接用预计算的 3D 特征
                if feat_3d is not None and feat_3d.numel() > 0:
                    # 使用预计算的融合特征
                    feat_3d = feat_3d.cuda(non_blocking=True)
                    predictions = feat_3d[inds_reverse, :]
                else:
                    # 使用模型推理（如果没有预计算特征）
                    # 这里需要根据实际情况调整
                    logger.warning(f"批次 {i} 没有预计算特征，跳过")
                    continue
                
                # 计算与文本特征的相似度 (开放词汇匹配)
                pred = predictions.half() @ text_features.t()
                
                # 获取预测类别（最高相似度）
                logits_pred = torch.max(pred, 1)[1].cpu()
                
                # 收集预测和真实标签
                if args.test_repeats == 1:
                    preds.append(logits_pred)
                else:
                    preds.append(pred.cpu())
                
                gts.append(label)
                
            except Exception as e:
                logger.error(f"处理批次 {i} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 合并所有预测和标签
        if len(preds) == 0 or len(gts) == 0:
            logger.warning("没有有效的预测结果!")
            continue
        
        gt = torch.cat(gts)
        pred = torch.cat(preds)
        
        logger.info(f"\n收集了 {gt.shape[0]} 个点的预测结果")
        
        # 计算预测类别
        if args.test_repeats == 1:
            pred_logit = pred
        else:
            pred_logit = pred.float().max(1)[1]
        
        # 使用 OpenScene 的 metric 计算 mIoU
        logger.info(f"\n{'='*60}")
        logger.info(f"评估轮次 {rep_i+1} 结果:")
        logger.info(f"{'='*60}\n")
        
        current_iou = metric.evaluate(
            pred_logit.numpy(),
            gt.numpy(),
            dataset=args.dataset_name,
            stdout=True  # 打印详细的每个类别的 IoU
        )
        
        # 如果有多次重复，累积结果
        if args.test_repeats > 1:
            store = pred + store
            store_logit = store.float().max(1)[1]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"累积评估结果 (轮次 1-{rep_i+1}):")
            logger.info(f"{'='*60}\n")
            
            accumu_iou = metric.evaluate(
                store_logit.numpy(),
                gt.numpy(),
                stdout=True,
                dataset=args.dataset_name
            )
            
            logger.info(f"\n当前轮次 mIoU: {current_iou:.4f}")
            logger.info(f"累积 mIoU: {accumu_iou:.4f}")
            
            return accumu_iou
        else:
            return current_iou
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='评估 Mix 开放词汇3D模型',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重路径')
    
    # 数据集参数
    parser.add_argument('--data-root', type=str,
                        default='/home/sunl/work/mix/data/scannet_3d',
                        help='数据集根目录')
    parser.add_argument('--precomputed-dir', type=str,
                        default='/home/sunl/work/mix/data/pixel_pooled',
                        help='预计算特征目录')
    parser.add_argument('--data-config-path', type=str,
                        default='config/data_scannet_3d.yaml',
                        help='数据配置文件')
    
    # 评估参数
    parser.add_argument('--split', type=str, default='val',
                        help='数据集划分 (train/val)')
    parser.add_argument('--dataset-name', type=str, default='scannet_3d',
                        help='数据集名称 (用于metric计算)')
    parser.add_argument('--labelset', type=str, default='scannet',
                        help='标签集 (scannet, matterport, nuscenes)')
    parser.add_argument('--test-repeats', type=int, default=1,
                        help='重复评估次数（处理体素化随机性）')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        help='评估batch size')
    parser.add_argument('--test-workers', type=int, default=2,
                        help='数据加载worker数')
    
    # 模型参数
    parser.add_argument('--pc-arch', type=str, default='MinkUNet18A',
                        help='3D backbone架构')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='体素大小')
    
    # 其他参数
    parser.add_argument('--config', type=str, default='',
                        help='YAML配置文件（可选）')
    parser.add_argument('--manual-seed', type=int, default=1342,
                        help='随机种子')
    parser.add_argument('--test-gpu', type=int, nargs='+', default=[0],
                        help='使用的GPU ID')
    
    # CLIP 参数（用于文本特征提取）
    parser.add_argument('--clipversion', type=str, default='ViT-L/14@336px',
                        help='CLIP模型版本')
    
    args = parser.parse_args()
    
    # 创建 logger
    logger = get_logger()
    
    # 设置随机种子
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    
    # 设置GPU
    if len(args.test_gpu) == 1:
        torch.cuda.set_device(args.test_gpu[0])
    
    logger.info(f"\n{'='*60}")
    logger.info("Mix 开放词汇3D模型评估")
    logger.info(f"{'='*60}")
    logger.info(f"权重: {args.checkpoint}")
    logger.info(f"数据集: {args.dataset_name}")
    logger.info(f"划分: {args.split}")
    logger.info(f"GPU: {args.test_gpu}")
    logger.info(f"{'='*60}\n")
    
    # 加载配置文件（如果提供）
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"✓ 加载配置文件: {args.config}")
    
    # 准备文本特征和标签集（开放词汇的关键）
    if 'scannet' in args.labelset:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other'  # 修改 'otherfurniture' 为 'other'
    else:
        raise NotImplementedError(f"暂不支持标签集: {args.labelset}")
    
    text_features = precompute_text_features(labelset, args.clipversion)
    labelset.append('unlabeled')  # 添加 unlabeled 类别
    
    # 创建模型
    logger.info("创建模型...")
    model_config = OpenVocabFusionModelV2Config(
        device='cuda',
        pc_arch=args.pc_arch,
    )
    model = OpenVocab3DFusionModelV2(model_config)
    model = model.cuda()
    
    # 加载权重
    epoch = load_checkpoint(model, args.checkpoint, logger)
    
    # 创建数据集
    logger.info("创建数据集...")
    
    # 处理路径
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_config_path = args.data_config_path
    if not os.path.isabs(data_config_path):
        data_config_path = os.path.join(repo_root, data_config_path)
    
    precomputed_dir = args.precomputed_dir
    if precomputed_dir and os.path.exists(precomputed_dir):
        logger.info(f"使用预计算特征: {precomputed_dir}")
    else:
        precomputed_dir = None
        logger.warning("未找到预计算特征目录，将使用在线特征提取（可能很慢）")
    
    dataset_config = OpenVocabDatasetV2Config(
        data_config_path=data_config_path,
        precomputed_dir=precomputed_dir,
        split=args.split,
        voxel_size=args.voxel_size,
        aug=False,
        eval_all=True,
        input_color=False,
    )
    
    val_dataset = OpenVocabScannetDatasetV2(dataset_config)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=open_vocab_collate_v2,
    )
    
    logger.info(f"✓ 数据集创建完成: {len(val_dataset)} 个样本\n")
    
    # 评估
    mean_iou = evaluate(model, val_loader, labelset, text_features, args, logger)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"最终结果:")
    logger.info(f"  权重: {os.path.basename(args.checkpoint)}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Mean IoU: {mean_iou:.4f}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
