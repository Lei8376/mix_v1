#!/usr/bin/env python
"""简单的批量处理脚本"""
import os
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MASK2FORMER_PATH = os.path.join(_SCRIPT_DIR, "third_party", "Mask2Former")
if _MASK2FORMER_PATH not in sys.path:
    sys.path.insert(0, _MASK2FORMER_PATH)

import numpy as np
from pathlib import Path
from tqdm import tqdm
from odise_feature import ODISEMaskEmbeddingExtractor
import random

# 设置随机种子
random.seed(42)

# 配置
scene_dir = "/home/sunl/work/scene0000_00/color"
output_dir = "./test_output"
label_sets = ["COCO", "ADE", "LVIS", "SCANNET_20"]
max_images = 10

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 获取所有图片
img_paths = sorted(Path(scene_dir).glob("*.jpg"))
print(f"找到 {len(img_paths)} 张图片")

# 随机采样
if len(img_paths) > max_images:
    img_paths = sorted(random.sample(img_paths, max_images))
print(f"处理 {len(img_paths)} 张图片")

# 初始化模型
print("加载模型...")
extractor = ODISEMaskEmbeddingExtractor(
    model_config="Panoptic/odise_caption_coco_50e.py",
    label_sets=label_sets,
    device="cuda"
)
print("模型加载完成!")

# 处理图片
for img_path in tqdm(img_paths):
    img_idx = img_path.stem
    out_path = Path(output_dir) / f"{img_idx}_odise.npz"
    
    if out_path.exists():
        continue
    
    try:
        # 直接传入图片路径字符串
        results = extractor.extract(str(img_path))
        
        # 保存结果
        np.savez_compressed(
            out_path,
            masks=results["masks"],
            mask_embeddings=results["mask_embeddings"].cpu().numpy(),
            panoptic_seg=results["panoptic_seg"].cpu().numpy(),
            info=results["results"],
            num_masks=results["num_masks"]
        )
    except Exception as e:
        print(f"\n错误: {img_path}: {e}")

print(f"\n完成！结果保存在 {output_dir}")
