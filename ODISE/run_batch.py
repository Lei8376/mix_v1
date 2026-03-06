#!/usr/bin/env python
"""
ODISE Batch Extraction Script
从 config.yaml 读取输入输出配置，批量提取 masks 和 embeddings
"""

# 设置 PYTHONPATH（必须在所有import之前）
import os
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MASK2FORMER_PATH = os.path.join(_SCRIPT_DIR, "third_party", "Mask2Former")
if _MASK2FORMER_PATH not in sys.path:
    sys.path.insert(0, _MASK2FORMER_PATH)

# 禁用xFormers (4090不需要memory_efficient_attention)
# 必须设置在导入任何使用xFormers的模块之前
class _XformersDisabler:
    """阻止 xformers 模块导入，强制使用标准 attention"""
    def find_module(self, name, path=None):
        if name.startswith('xformers'):
            return self
        return None
    
    def load_module(self, name):
        raise ImportError(f"xformers is disabled (no CUDA support)")

sys.meta_path.insert(0, _XformersDisabler())
os.environ["XFORMERS_DISABLED"] = "1"

import time
import yaml
import shutil
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

# 导入ODISE特征提取器（使用原始 GitHub 版本，不是 F/ODISE 修改过的版本）
from odise_feature import ODISEMaskEmbeddingExtractor


def load_yaml(p: Path) -> dict:
    """加载yaml配置"""
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(input_dir: Path, pattern: str, recursive: bool):
    """列出图片"""
    if pattern == "auto":
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        paths = []
        for ext in exts:
            if recursive:
                paths.extend(Path(input_dir).rglob(ext))
            else:
                paths.extend(Path(input_dir).glob(ext))
        return sorted(set(paths))
    else:
        if recursive:
            return sorted(Path(input_dir).rglob(pattern))
        else:
            return sorted(Path(input_dir).glob(pattern))


def get_output_path(img_path: Path, out_dir: Path, postfix: str, suffix: str) -> Path:
    """生成输出文件路径"""
    return out_dir / f"{img_path.stem}{postfix}{suffix}"


def preload_images(img_paths, queue, num_workers=2):
    """预加载图片到队列"""
    def load_worker():
        for img_path in img_paths:
            try:
                img = np.array(Image.open(img_path))
                queue.put((img_path, img, None))
            except Exception as e:
                queue.put((img_path, None, e))
    
    thread = Thread(target=load_worker, daemon=True)
    thread.start()
    return thread


def process_one_scene(
    scene_name: str,
    input_dir: Path,
    output_dir: Path,
    extractor: ODISEMaskEmbeddingExtractor,
    postfix: str,
    suffix: str,
    fp16: bool,
    batch_size: int = 1,
    use_preload: bool = True,
):
    """处理一个scene"""
    # 查找图片
    img_paths = list_images(input_dir, "auto", False)
    n_total = len(img_paths)
    
    if n_total == 0:
        print(f"⚠️  [跳过] {scene_name}: 没有找到图片")
        return 0, 0, 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找缺失的
    missing_paths = []
    for img_path in img_paths:
        out_path = get_output_path(img_path, output_dir, postfix, suffix)
        if not out_path.exists():
            missing_paths.append(img_path)
    
    n_missing = len(missing_paths)
    if n_missing == 0:
        print(f"✅ [完成] {scene_name}: 全部 {n_total} 张已处理，跳过")
        return n_total, 0, 0
    
    print(f"--- ▶ {scene_name}: 总数 {n_total} | 缺失 {n_missing} | 已完成 {n_total - n_missing} ---")
    
    # 提取特征
    success = 0
    fail = 0
    
    # 使用预加载优化I/O
    if use_preload and len(missing_paths) > 1:
        img_queue = Queue(maxsize=3)
        preload_thread = preload_images(missing_paths, img_queue)
        
        for _ in tqdm(range(len(missing_paths)), desc=scene_name):
            img_path, img, error = img_queue.get()
            
            if error:
                print(f"❌ 加载失败 {img_path.name}: {error}")
                fail += 1
                continue
            
            try:
                # 提取masks和embeddings
                result = extractor.extract(img)
                
                # 保存为npz
                out_path = get_output_path(img_path, output_dir, postfix, suffix)
                
                save_data = {
                    "masks": torch.stack([m.cpu() for m in result["masks"]]) if result["masks"] else torch.empty(0),
                    "mask_embeddings": result["mask_embeddings"].cpu(),
                    "num_masks": result["num_masks"],
                    "info": [
                        {
                            "category_name": r.category_name,
                            "category_id": r.category_id,
                            "is_thing": r.is_thing,
                            "score": r.score,
                            "area": r.area,
                        }
                        for r in result["results"]
                    ]
                }
                
                save_data["masks"] = save_data["masks"].numpy()
                
                if fp16:
                    save_data["mask_embeddings"] = save_data["mask_embeddings"].half()
                
                np.savez_compressed(out_path, **save_data)
                success += 1
                
            except Exception as e:
                print(f"❌ 处理失败 {img_path.name}: {e}")
                fail += 1
        
        preload_thread.join(timeout=1)
    else:
        # 不使用预加载的原始逻辑
        for img_path in tqdm(missing_paths, desc=scene_name):
            try:
                result = extractor.extract(str(img_path))
                
                out_path = get_output_path(img_path, output_dir, postfix, suffix)
                
                save_data = {
                    "masks": torch.stack([m.cpu() for m in result["masks"]]) if result["masks"] else torch.empty(0),
                    "mask_embeddings": result["mask_embeddings"].cpu(),
                    "num_masks": result["num_masks"],
                    "info": [
                        {
                            "category_name": r.category_name,
                            "category_id": r.category_id,
                            "is_thing": r.is_thing,
                            "score": r.score,
                            "area": r.area,
                        }
                        for r in result["results"]
                    ]
                }
                
                save_data["masks"] = save_data["masks"].numpy()
                
                if fp16:
                    save_data["mask_embeddings"] = save_data["mask_embeddings"].half()
                
                np.savez_compressed(out_path, **save_data)
                success += 1
                
            except Exception as e:
                print(f"❌ 处理失败 {img_path.name}: {e}")
                fail += 1
    
    print(f"🎊 [完成] {scene_name}: 成功 {success} | 失败 {fail}")
    return n_total, success, fail


def main():
    parser = argparse.ArgumentParser(description="ODISE批量特征提取")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_yaml(Path(args.config))
    
    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})
    model_cfg = cfg.get("model", {})
    runtime_cfg = cfg.get("runtime", {})
    
    # 解析配置
    input_path = Path(input_cfg.get("path", "inputs"))
    pattern = input_cfg.get("pattern", "auto")
    recursive = input_cfg.get("recursive", False)
    
    output_dir = Path(output_cfg.get("dir", "out"))
    keep_dir = output_cfg.get("keep_dir_structure", True)
    postfix = output_cfg.get("postfix", "_odise")
    suffix = output_cfg.get("suffix", ".npz")
    
    model_config = model_cfg.get("config", "configs/Panoptic/odise_caption_coco_50e.py")
    # 注意：model_zoo.get_config() 期望相对路径，不需要转换
    # 相对路径格式如 "Panoptic/odise_label_coco_50e.py"
    label_sets = model_cfg.get("label_sets", ["COCO", "ADE", "LVIS"])
    extra_vocab = model_cfg.get("extra_vocab", "")  # 自定义类别
    caption = model_cfg.get("caption", "a photo of")  # caption提示词
    
    gpus = runtime_cfg.get("gpus", [0])
    workers = runtime_cfg.get("workers", 4)
    batch_size = runtime_cfg.get("batch_size", 1)
    fp16 = runtime_cfg.get("fp16", True)
    use_xformers = runtime_cfg.get("xformers", False)
    use_preload = runtime_cfg.get("preload", True)

    # 禁用xFormers (4090不需要memory_efficient_attention)
    if not use_xformers:
        os.environ["XFORMERS_DISABLED"] = "1"
        print("[配置] xFormers: 已禁用 (使用标准attention)")

    device = f"cuda:{gpus[0]}" if gpus else "cpu"
    
    print(f"[配置] 输入: {input_path}")
    print(f"[配置] 输出: {output_dir}")
    print(f"[配置] 模型: {model_config}")
    print(f"[配置] 标签: {label_sets}")
    print(f"[配置] Caption: {caption}")
    print(f"[配置] 自定义: {extra_vocab if extra_vocab else '无'}")
    print(f"[配置] 批大小: {batch_size}")
    print(f"[配置] 预加载: {'启用' if use_preload else '禁用'}")
    print(f"[配置] 设备: {device}")
    print()
    
    # 初始化提取器
    print("🚀 初始化ODISE模型...")
    extractor = ODISEMaskEmbeddingExtractor(
        model_config=model_config,
        label_sets=label_sets,
        vocab=extra_vocab,
        device=device
    )
    
    # 查找所有scene
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")
    
    # 检查是scene模式还是直接图片模式
    scenes = sorted([p for p in input_path.iterdir() if p.is_dir() and p.name.startswith("scene")])
    
    # 如果没有scene目录，检查是否有直接图片
    if len(scenes) == 0:
        direct_images = list_images(input_path, "auto", False)
        if len(direct_images) > 0:
            print(f"🚀 发现直接图片模式，处理 {len(direct_images)} 张图片...\n")
            # 直接处理 inputs 目录下的图片
            scene_out_dir = output_dir
            done, success, fail = process_one_scene(
                scene_name="direct_images",
                input_dir=input_path,
                output_dir=scene_out_dir,
                extractor=extractor,
                postfix=postfix,
                suffix=suffix,
                fp16=fp16,
                batch_size=batch_size,
                use_preload=use_preload,
            )
            total_done = done
            total_success = success
            total_fail = fail
        else:
            print(f"⚠️  没有找到scene目录也没有找到图片")
            total_done = total_success = total_fail = 0
    else:
        print(f"🚀 发现 {len(scenes)} 个场景，开始批量ODISE导出...\n")
        
        # 处理每个scene
        total_success = 0
        total_fail = 0
        total_done = 0
        
        for idx, scene in enumerate(scenes, 1):
            scene_name = scene.name
            color_dir = scene / "color"
            
            if not color_dir.exists():
                print(f"⚠️  [跳过] {scene_name}: color目录不存在")
                continue
            
            scene_out_dir = output_dir / scene_name if keep_dir else output_dir
            
            done, success, fail = process_one_scene(
                scene_name=scene_name,
                input_dir=color_dir,
                output_dir=scene_out_dir,
                extractor=extractor,
                postfix=postfix,
                suffix=suffix,
                fp16=fp16,
                batch_size=batch_size,
                use_preload=use_preload,
            )
            
            total_done += done
            total_success += success
            total_fail += fail
    
    print("\n" + "=" * 50)
    print(f"✅ 总成功: {total_success}")
    print(f"❌ 总失败: {total_fail}")
    print(f"📊 总处理: {total_done}")
    print("=" * 50)


if __name__ == "__main__":
    main()
