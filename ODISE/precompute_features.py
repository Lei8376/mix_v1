#!/usr/bin/env python
"""
Precompute ODISE features from YAML config.
"""

# 设置路径（必须在所有import之前）
import os
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MASK2FORMER_PATH = os.path.join(_SCRIPT_DIR, "third_party", "Mask2Former")
if _MASK2FORMER_PATH not in sys.path:
    sys.path.insert(0, _MASK2FORMER_PATH)

import argparse
import random
from glob import glob
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# 设置随机种子
random.seed(42)

# 导入 ODISE 特征提取器
from odise_feature import ODISEMaskEmbeddingExtractor


class ImagePreloader:
    """后台预加载图片，减少IO等待时间"""
    def __init__(self, img_paths: List[str], prefetch_count: int = 2):
        self.img_paths = img_paths
        self.prefetch_count = prefetch_count
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.thread = None
        
    def _load_worker(self):
        for img_path in self.img_paths:
            if self.stop_event.is_set():
                break
            try:
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)
                self.queue.put((img_path, img_np, None))
            except Exception as e:
                self.queue.put((img_path, None, e))
        # 放入结束标记
        self.queue.put((None, None, None))
    
    def start(self):
        self.thread = threading.Thread(target=self._load_worker, daemon=True)
        self.thread.start()
        
    def get_next(self):
        return self.queue.get()
    
    def stop(self):
        self.stop_event.set()


def load_config(config_path: str) -> dict:
    """加载 YAML 配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_scene_image_paths(data_root: str, scene_pattern: str = "scene*") -> Dict[str, list]:
    """获取所有场景和图片路径"""
    scenes = {}
    scene_dirs = sorted(glob(os.path.join(data_root, scene_pattern)))
    
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        img_paths = []
        
        # 尝试多种可能的图片位置
        for pattern in [
            os.path.join(scene_dir, "color", "*.jpg"),
            os.path.join(scene_dir, "color", "*.png"),
            os.path.join(scene_dir, "*.jpg"),
            os.path.join(scene_dir, "*.png"),
        ]:
            found = glob(pattern)
            if found:
                img_paths.extend(found)
        
        if img_paths:
            # 按数字排序
            try:
                img_paths = sorted(img_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            except ValueError:
                img_paths = sorted(img_paths)
            scenes[scene_name] = img_paths
    
    return scenes


def extract_features_from_numpy(extractor, img_np: np.ndarray) -> Dict[str, np.ndarray]:
    """从 numpy 数组提取特征（用于预加载模式）"""
    img_tensor = torch.from_numpy(img_np)
    
    with torch.no_grad():
        results = extractor.extract(img_tensor)
    
    num_masks = results["num_masks"]
    
    if num_masks == 0:
        H, W = img_np.shape[:2]
        return {
            "masks": np.zeros((0, H, W), dtype=bool),
            "mask_embeddings": np.zeros((0, 256), dtype=np.float16),
            "num_masks": np.array(0, dtype=np.int64),
            "info": np.array([], dtype=object),
        }
    
    masks = torch.stack(results["masks"]).cpu().numpy().astype(bool)
    mask_embeddings = results["mask_embeddings"].cpu().numpy().astype(np.float16)
    
    info = []
    for r in results["results"]:
        info.append({
            "category_name": r.category_name,
            "category_id": r.category_id,
            "is_thing": r.is_thing,
            "score": r.score,
            "area": r.area,
        })
    
    return {
        "masks": masks,
        "mask_embeddings": mask_embeddings,
        "num_masks": np.array(num_masks, dtype=np.int64),
        "info": np.array(info, dtype=object),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute ODISE features")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    data_root = config['data']['data_root']
    output_dir = config['data']['output_dir']
    scene_pattern = config['data']['scene_pattern']
    
    max_scenes = config['processing']['max_scenes']
    max_images = config['processing']['max_images_per_scene']
    skip_existing = config['processing']['skip_existing']
    device = config['processing']['device']
    
    print("="*70)
    print("ODISE 特征预计算")
    print("="*70)
    print(f"数据路径: {data_root}")
    print(f"输出路径: {output_dir}")
    print(f"设备: {device}")
    print(f"模型: {config['odise']['model_config']}")
    print("="*70)
    print()
    
    # 获取场景
    scenes = get_scene_image_paths(data_root, scene_pattern)
    print(f"找到 {len(scenes)} 个场景")
    
    if len(scenes) == 0:
        print(f"错误: 在 {data_root} 中找不到场景")
        return
    
    # 限制场景数
    if max_scenes > 0:
        scenes = dict(list(scenes.items())[:max_scenes])
        print(f"限制为 {len(scenes)} 个场景")
    
    # 显示示例
    for i, (name, paths) in enumerate(list(scenes.items())[:3]):
        print(f"  {name}: {len(paths)} 张图片")
    print()
    
    # 初始化提取器
    print("加载 ODISE 模型...")
    extractor = ODISEMaskEmbeddingExtractor(
        model_config=config['odise']['model_config'],
        label_sets=config['odise']['label_sets'],
        vocab=config['odise']['vocab'],
        overlap_threshold=config['odise']['overlap_threshold'],
        object_mask_threshold=config['odise']['object_mask_threshold'],
        device=device,
    )
    print("✅ 模型加载成功!")
    print()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理图片
    processed = 0
    skipped = 0
    errors = 0
    
    # 随机采样图片
    sampled_scenes = {}
    for scene_name, img_paths in scenes.items():
        if max_images > 0 and len(img_paths) > max_images:
            # 随机选取 max_images 张图片
            sampled_scenes[scene_name] = sorted(random.sample(img_paths, max_images))
        else:
            sampled_scenes[scene_name] = img_paths
    
    # 计算采样后的总图片数
    total_images = sum(len(paths) for paths in sampled_scenes.values())
    print(f"采样后总图片数: {total_images}")
    print()
    
    with tqdm(total=total_images, desc="处理进度") as pbar:
        for scene_name, img_paths in sampled_scenes.items():
            scene_out = os.path.join(output_dir, scene_name)
            os.makedirs(scene_out, exist_ok=True)
            
            # 过滤需要处理的图片（跳过已存在的）
            paths_to_process = []
            for img_path in img_paths:
                img_idx = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(scene_out, f"{img_idx}_odise.npz")
                if skip_existing and os.path.exists(out_path):
                    skipped += 1
                    pbar.update(1)
                else:
                    paths_to_process.append((img_path, out_path))
            
            if not paths_to_process:
                continue
            
            # 使用预加载器
            preloader = ImagePreloader([p[0] for p in paths_to_process], prefetch_count=2)
            preloader.start()
            
            path_dict = {p[0]: p[1] for p in paths_to_process}
            
            while True:
                img_path, img_np, error = preloader.get_next()
                if img_path is None:  # 结束标记
                    break
                    
                out_path = path_dict[img_path]
                
                if error:
                    print(f"\n加载错误: {img_path}: {error}")
                    errors += 1
                    pbar.update(1)
                    continue
                
                try:
                    features = extract_features_from_numpy(extractor, img_np)
                    np.savez_compressed(out_path, **features)
                    processed += 1
                except Exception as e:
                    print(f"\n处理错误: {img_path}: {e}")
                    errors += 1
                
                pbar.update(1)
                pbar.set_postfix({"处理": processed, "跳过": skipped, "错误": errors})
            
            preloader.stop()
    
    print()
    print("="*70)
    print("完成!")
    print("="*70)
    print(f"总图片数: {total_images}")
    print(f"已处理: {processed}")
    print(f"已跳过: {skipped}")
    print(f"错误: {errors}")
    print(f"输出目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
