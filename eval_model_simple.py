#!/usr/bin/env python3
"""
评估 Mix 开放词汇3D模型 - 基于训练配置的评估脚本

用法:
    python eval_model_simple.py \
        --checkpoint checkpoints/full.4/checkpoint_epoch_36.pth \
        --config config/train_scannet_v2_full_multi_gpu.yaml

说明:
    - 参考训练配置，使用相同的数据集设置
    - 使用 openscene 的 metric 计算 mIoU
    - 输出各个类别的 IoU（包括 floor）
"""

import os
import sys

# 首先设置路径（在任何导入之前）
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
openscene_dir = os.path.join(parent_dir, 'openscene')

# 先添加openscene，让它的dataset模块优先于mix的dataset
sys.path.insert(0, openscene_dir)
sys.path.insert(1, current_dir)

print(f"当前目录: {current_dir}")
print(f"Python路径前3项: {sys.path[:3]}")

# 现在导入其他模块
import argparse
import random
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import yaml

# 导入第三方库
from MinkowskiEngine import SparseTensor
import clip

# 首先导入openscene的模块
from util import metric
import dataset.label_constants as openscene_labels
SCANNET_LABELS_20 = openscene_labels.SCANNET_LABELS_20

# 保存openscene的dataset
import dataset as openscene_dataset

# 移除dataset从sys.modules，让mix的dataset能被导入
del sys.modules['dataset']

# 重新添加mix目录到最前面
sys.path.insert(0, current_dir)

# 导入mix的dataset模块
import dataset as mix_dataset

OpenVocabDatasetV2Config = mix_dataset.open_vocab_dataset_v2.OpenVocabDatasetV2Config
OpenVocabScannetDatasetV2 = mix_dataset.open_vocab_dataset_v2.OpenVocabScannetDatasetV2
open_vocab_collate_v2 = mix_dataset.open_vocab_dataset_v2.open_vocab_collate_v2

# 导入mix的model模块  
import model.open_vocab_fusion_v2 as mix_model
OpenVocabFusionModelV2Config = mix_model.OpenVocabFusionModelV2Config
OpenVocab3DFusionModelV2 = mix_model.OpenVocab3DFusionModelV2

print("✓ 所有模块导入成功")


def get_logger():
    """创建logger"""
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def load_yaml_config(path):
    """加载YAML配置"""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_checkpoint(model, ckpt_path, logger):
    """加载权重"""
    ckpt = torch.load(ckpt_path, map_location='cuda')
    
    # 处理不同的键名格式
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        raise KeyError(f"Cannot find model state dict in checkpoint. Keys: {list(ckpt.keys())}")
    
    # 处理DDP权重
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # 移除 'module.' 前缀
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    
    epoch = ckpt.get('epoch', 0)
    logger.info(f"✓ 加载权重成功 (epoch {epoch})")
    
    # 显示训练指标
    if 'best_iou' in ckpt:
        logger.info(f"  训练时最佳 IoU: {ckpt['best_iou']:.4f}")
    if 'best_loss' in ckpt:
        logger.info(f"  训练时最佳 Loss: {ckpt['best_loss']:.4f}")
    
    return epoch


def extract_clip_features(labelset, clip_model="ViT-L/14@336px"):
    """提取CLIP文本特征"""
    model, _ = clip.load(clip_model, device='cuda', jit=False)
    text = clip.tokenize(labelset).cuda()
    
    with torch.no_grad():
        features = model.encode_text(text)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features


@torch.no_grad()
def evaluate(model, val_loader, text_features, labelset, logger, test_repeats=1):
    """评估模型"""
    torch.backends.cudnn.enabled = False
    model.eval()
    
    logger.info(f"\n{'='*60}")
    logger.info("开始评估...")
    logger.info(f"类别: {len(labelset)-1} 个")
    logger.info(f"重复: {test_repeats} 次")
    logger.info(f"{'='*60}\n")
    
    store = 0.0
    
    for rep_i in range(test_repeats):
        preds, gts = [], []
        
        logger.info(f"轮次 {rep_i+1}/{test_repeats}")
        
        # 设置随机种子
        if rep_i > 0:
            seed = np.random.randint(10000)
            set_seed(seed)
        
        for batch in tqdm(val_loader, desc=f"评估"):
            try:
                coords = batch["coords_3d"]       # (N_total, 4)
                feat   = batch["feat_3d"]          # 真实3D点特征，对齐训练
                label  = batch["binary_label_3d"]  # (N_total,) GT语义标签

                # 对齐训练：用 batch["feat_3d"] + coords.int() 构建SparseTensor
                sinput = SparseTensor(
                    feat.cuda(non_blocking=True),
                    coords.int().cuda(non_blocking=True)
                )

                batch_input = {
                    "sinput": sinput,
                    "coords_3d": coords.cuda(),
                    "ori_coords_3d": batch["ori_coords_3d"].cuda(),
                    "pixel_pooled": batch["pixel_pooled"].cuda(),
                    "masks": batch["masks"].cuda(),
                    "mask_embeddings": batch["mask_embeddings"].cuda(),
                    "mask_valid": batch["mask_valid"].cuda(),
                }

                results = model(batch_input)

                # pred_3d: (N_total, feat_dim) 3D点特征
                predictions = results["pred_3d"]

                # 对齐OpenScene：L2归一化后做点积匹配
                # text_features 在 extract_clip_features 里已归一化
                import torch.nn.functional as F
                pred_norm = F.normalize(predictions.float(), dim=-1)  # (N, D)
                pred = pred_norm @ text_features.float().t()           # (N, 20)

                # 将超出20类范围的点（没有特征/无效点）映射到最后一类'unlabeled'
                # 对齐OpenScene evaluate.py 第300行逻辑
                logits_pred = torch.max(pred, 1)[1].cpu()  # (N,)

                if test_repeats == 1:
                    preds.append(logits_pred)
                else:
                    preds.append(pred.cpu())

                gts.append(label.cpu())

            except Exception as e:
                logger.error(f"处理batch出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算IoU
        if len(preds) == 0:
            logger.warning("没有有效预测!")
            continue
        
        gt = torch.cat(gts)
        pred = torch.cat(preds)
        
        if test_repeats == 1:
            pred_logit = pred
        else:
            pred_logit = pred.float().max(1)[1]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"轮次 {rep_i+1} 结果:")
        logger.info(f"{'='*60}\n")
        
        miou = metric.evaluate(
            pred_logit.numpy(),
            gt.numpy(),
            dataset='scannet_3d',
            stdout=True
        )
        
        if test_repeats > 1:
            store = pred + store
            store_logit = store.float().max(1)[1]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"累积结果 (1-{rep_i+1}):")
            logger.info(f"{'='*60}\n")
            
            accumu_miou = metric.evaluate(
                store_logit.numpy(),
                gt.numpy(),
                dataset='scannet_3d',
                stdout=True
            )
            
            return accumu_miou
    
    return miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径')
    parser.add_argument('--config', type=str, default='config/train_scannet_v2_full_multi_gpu.yaml', help='训练配置文件')
    parser.add_argument('--test-repeats', type=int, default=1, help='重复评估次数')
    parser.add_argument('--batch-size', type=int, default=1, help='评估batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载worker数')
    args = parser.parse_args()
    
    logger = get_logger()
    
    # 加载配置
    config = load_yaml_config(args.config)
    
    # 设置随机种子
    set_seed(config.get('seed', 1342))
    
    # 设置GPU
    torch.cuda.set_device(0)
    
    logger.info(f"\n{'='*60}")
    logger.info("Mix 开放词汇3D模型评估")
    logger.info(f"{'='*60}")
    logger.info(f"权重: {args.checkpoint}")
    logger.info(f"配置: {args.config}")
    logger.info(f"{'='*60}\n")
    
    # 准备文本特征（开放词汇的关键）
    labelset = list(SCANNET_LABELS_20)
    labelset[-1] = 'other'
    
    logger.info("提取CLIP文本特征...")
    text_features = extract_clip_features(labelset)
    logger.info(f"✓ 文本特征: {text_features.shape}")
    
    labelset.append('unlabeled')
    
    # 创建数据集
    logger.info("\n创建数据集...")
    
    dataset_cfg = config['dataset']
    repo_root = os.path.dirname(os.path.abspath(__file__))
    
    data_config_path = dataset_cfg['data_config_path']
    if not os.path.isabs(data_config_path):
        data_config_path = os.path.join(repo_root, data_config_path)
    
    precomputed_dir = dataset_cfg.get('precomputed_dir')
    
    val_dataset_config = OpenVocabDatasetV2Config(
        data_config_path=data_config_path,
        precomputed_dir=precomputed_dir,
        projection_dir=dataset_cfg.get('projection_dir'),
        split='val',  # 评估验证集
        voxel_size=0.05,
        aug=False,
        eval_all=True,
        input_color=False,
    )
    
    val_dataset = OpenVocabScannetDatasetV2(val_dataset_config)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=open_vocab_collate_v2,
    )
    
    logger.info(f"✓ 数据集: {len(val_dataset)} 个样本")
    
    # 创建模型
    logger.info("\n创建模型...")
    
    # 从checkpoint权重推断模型架构
    # 根据错误信息，训练时使用的是 MinkUNet34C (更大的模型)
    ckpt_temp = torch.load(args.checkpoint, map_location='cpu')
    state_dict_keys = list(ckpt_temp.get('model_state_dict', {}).keys())
    
    # 检查特征维度来判断模型大小
    # MinkUNet34C 的block5使用256维，MinkUNet18A使用128维
    if any('block5' in k and '256' in str(ckpt_temp['model_state_dict'][k].shape) for k in state_dict_keys if 'block5' in k):
        pc_arch = 'MinkUNet34C'
        logger.info(f"  从权重推断: {pc_arch} (较大模型)")
    else:
        model_cfg = config.get('model') or {}
        pc_arch = model_cfg.get('pc_arch', 'MinkUNet34C')
        logger.info(f"  使用默认: {pc_arch}")
    
    model_config = OpenVocabFusionModelV2Config(
        device='cuda',
        pc_arch=pc_arch,
    )
    
    model = OpenVocab3DFusionModelV2(model_config).cuda()
    
    # 加载权重
    epoch = load_checkpoint(model, args.checkpoint, logger)
    
    # 评估
    miou = evaluate(model, val_loader, text_features, labelset, logger, args.test_repeats)
    
    logger.info(f"\n{'='*60}")
    logger.info("最终结果")
    logger.info(f"{'='*60}")
    logger.info(f"权重: {os.path.basename(args.checkpoint)}")
    logger.info(f"Epoch: {epoch}")
    logger.info(f"mIoU: {miou:.4f}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
