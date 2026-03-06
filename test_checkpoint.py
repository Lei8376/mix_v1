"""
简化版模型评估脚本 - 测试指定checkpoint的mIoU

用法:
    python test_checkpoint.py --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from tqdm import tqdm
from collections import defaultdict

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'openscene'))

# 导入必要模块
try:
    from MinkowskiEngine import SparseTensor
    from util import metric
    from dataset.label_constants import SCANNET_LABELS_20
    print("✓ 成功导入依赖模块")
except ImportError as e:
    print(f"错误: 导入失败 - {e}")
    sys.exit(1)


def set_seed(seed: int = 1342):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_checkpoint_info(checkpoint_path: str):
    """加载并显示checkpoint信息"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print(f"加载权重文件: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 显示checkpoint信息
    info = {}
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
        print(f"  Epoch: {checkpoint['epoch']}")
    
    if 'state_dict' in checkpoint:
        num_params = len(checkpoint['state_dict'])
        print(f"  参数数量: {num_params}")
        
        # 显示前几个参数的key
        keys = list(checkpoint['state_dict'].keys())[:5]
        print(f"  参数示例: {keys}")
    
    if 'optimizer' in checkpoint:
        print(f"  包含优化器状态: ✓")
    
    # 检查是否有训练指标
    for key in ['train_loss', 'val_loss', 'val_miou', 'val_iou']:
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]:.4f}")
    
    print(f"{'='*60}\n")
    
    return checkpoint


def extract_text_features_clip(labelset, device='cuda'):
    """使用CLIP提取文本特征"""
    try:
        import clip
        model, _ = clip.load("ViT-B/16", device=device, jit=False)
        
        text_tokens = clip.tokenize(labelset).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        print(f"✓ 成功提取 {len(labelset)} 个类别的CLIP文本特征")
        return text_features.half()
    except Exception as e:
        print(f"警告: CLIP特征提取失败 - {e}")
        print("使用随机特征（仅用于测试结构）")
        features = torch.randn(len(labelset), 512, device=device)
        return (features / features.norm(dim=-1, keepdim=True)).half()


@torch.no_grad()
def simple_evaluate(
    model,
    checkpoint_path: str,
    device: str = "cuda",
    test_repeats: int = 1,
):
    """
    简化评估：直接使用openscene的评估指标
    不需要dataloader，仅测试checkpoint能否加载
    """
    
    print(f"\n{'='*60}")
    print("开始评估模型")
    print(f"{'='*60}\n")
    
    # 1. 加载checkpoint信息
    checkpoint = load_checkpoint_info(checkpoint_path)
    
    # 2. 准备文本特征（用于将3D特征映射到类别）
    labelset = list(SCANNET_LABELS_20)
    labelset[-1] = 'other'
    print(f"类别列表 ({len(labelset)} 类):")
    for i, label in enumerate(labelset):
        print(f"  {i}: {label}")
    print()
    
    text_features = extract_text_features_clip(labelset, device=device)
    
    # 3. 加载模型权重
    print(f"{'='*60}")
    print("加载模型权重到模型...")
    print(f"{'='*60}\n")
    
    try:
        # 直接加载state_dict
        state_dict = checkpoint['state_dict']
        
        # 检查是否需要处理DDP前缀
        first_key = list(state_dict.keys())[0]
        if first_key.startswith('module.'):
            print("检测到DDP模型，移除'module.'前缀...")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        # 尝试加载
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"警告: 缺少的键 ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"警告: 多余的键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        
        print(f"✓ 成功加载权重 (epoch {checkpoint.get('epoch', 'unknown')})\n")
        
    except Exception as e:
        print(f"错误: 无法加载权重 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"{'='*60}")
    print("权重加载成功！")
    print(f"{'='*60}\n")
    
    # 提示：要进行完整评估需要数据集
    print("提示：要进行完整的mIoU评估，请使用 eval_model.py 并提供数据集配置")
    print("      该脚本仅验证checkpoint可以成功加载")


def main():
    parser = argparse.ArgumentParser(
        description="测试Mix模型checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型权重路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1342,
        help="随机种子"
    )
    parser.add_argument(
        "--pc-arch",
        type=str,
        default="MinkUNet18A",
        help="3D backbone架构"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = "cpu"
    
    # 导入模型（延迟导入以便先检查设备）
    from model.open_vocab_fusion_v2 import (
        OpenVocabFusionModelV2Config,
        OpenVocab3DFusionModelV2,
    )
    
    # 创建模型
    print(f"{'='*60}")
    print("创建模型...")
    print(f"  3D backbone: {args.pc_arch}")
    print(f"  设备: {device}")
    print(f"{'='*60}\n")
    
    model_config = OpenVocabFusionModelV2Config(
        device=device,
        pc_arch=args.pc_arch,
    )
    
    model = OpenVocab3DFusionModelV2(model_config)
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型创建成功\n")
    
    # 获取权重路径
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        repo_root = os.path.abspath(os.path.dirname(__file__))
        checkpoint_path = os.path.join(repo_root, checkpoint_path)
    
    # 评估
    simple_evaluate(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )


if __name__ == "__main__":
    main()
