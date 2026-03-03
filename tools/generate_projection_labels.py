#!/usr/bin/env python3
"""
生成 x_label 和 y_label 投影标签的工具脚本。

这个脚本从 ScanNet 数据中读取 3D 点云和相机参数，
计算 3D 点到 2D 图像的投影，生成正确的 x_label 和 y_label。

Usage:
    python generate_projection_labels.py --data-root /path/to/scannet_3d --output-dir /path/to/output
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import imageio.v2 as imageio
from glob import glob
from tqdm import tqdm

# 添加项目根目录到 Python 路径
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.mapping_util import getMapping


def load_scene_data(scene_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载场景的 3D 点云数据。"""
    if scene_path.suffix == '.pth':
        data = torch.load(scene_path, map_location="cpu", weights_only=False)
        if isinstance(data, (list, tuple)):
            locs, feats, labels = data[0], data[1], data[2]
        else:
            locs = data.get("locs", data.get("coords"))
            feats = data.get("feats", data.get("feat"))
            labels = data.get("labels")
    else:
        raise ValueError(f"Unsupported file format: {scene_path.suffix}")
    
    # 转换为 numpy
    if isinstance(locs, torch.Tensor):
        locs = locs.numpy()
    if isinstance(feats, torch.Tensor):
        feats = feats.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    return locs, feats, labels


def compute_projection_labels(
    locs_3d: np.ndarray,
    scene_2d_dir: Path,
    point2img_mapper,
    min_points: int = 400,
    max_points: int = 65000
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    计算 3D 点到 2D 图像的投影标签。
    
    Returns:
        Dict[frame_id, Dict[label_type, array]]
    """
    results = {}
    
    # 获取所有图像文件
    img_dirs = sorted(
        glob(str(scene_2d_dir / "color" / "*")), 
        key=lambda x: int(os.path.basename(x)[:-4])
    )
    
    if len(img_dirs) == 0:
        print(f"Warning: No images found in {scene_2d_dir / 'color'}")
        return results
    
    for img_dir in tqdm(img_dirs, desc="Processing frames"):
        frame_id = os.path.basename(img_dir)[:-4]
        
        try:
            # 读取相机位姿
            pose_path = img_dir.replace("color", "pose").replace(".jpg", ".txt")
            if not os.path.exists(pose_path):
                continue
            pose = np.loadtxt(pose_path)
            
            # 读取深度图
            depth_path = img_dir.replace("color", "depth").replace("jpg", "png")
            if not os.path.exists(depth_path):
                continue
            depth = imageio.imread(depth_path) / 1000.0
            
            # 计算 3D 到 2D 的映射
            single_mapping = point2img_mapper.compute_mapping(pose, locs_3d, depth)
            
            # 过滤有效点
            mask = single_mapping[:, 2]  # 第三列是有效性标记
            valid_points = np.sum(mask == 1)
            
            # 检查点数是否在合理范围内
            if valid_points < min_points or valid_points > max_points:
                continue
            
            # 提取有效的投影坐标
            zero_rows = np.all(single_mapping != 0, axis=1)
            valid_mapping = single_mapping[zero_rows]
            
            if len(valid_mapping) == 0:
                continue
            
            # 提取 x_label 和 y_label
            x_label = valid_mapping[:, 0][valid_mapping[:, 0] != 0]
            y_label = valid_mapping[:, 1][valid_mapping[:, 1] != 0]
            
            if len(x_label) == 0 or len(y_label) == 0:
                continue
            
            # 获取对应的 3D 点索引
            valid_indices = np.where(zero_rows)[0]
            
            results[frame_id] = {
                "x_label": x_label.astype(np.int64),
                "y_label": y_label.astype(np.int64),
                "valid_indices": valid_indices.astype(np.int64),
                "valid_mask": mask.astype(bool),
                "mapping": single_mapping
            }
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")
            continue
    
    return results


def update_scene_data(
    scene_path: Path,
    projection_data: Dict[str, Dict[str, np.ndarray]],
    output_path: Path
) -> None:
    """更新场景数据，添加投影标签。"""
    # 加载原始数据
    locs, feats, labels = load_scene_data(scene_path)
    
    # 选择一个代表性的帧（通常选择中间的帧）
    if len(projection_data) == 0:
        print(f"Warning: No projection data for {scene_path.name}")
        # 创建零填充的标签
        N = len(locs)
        x_label = np.zeros(N, dtype=np.int64)
        y_label = np.zeros(N, dtype=np.int64)
    else:
        # 选择点数最多的帧
        best_frame = max(projection_data.keys(), 
                        key=lambda k: len(projection_data[k]["x_label"]))
        frame_data = projection_data[best_frame]
        
        N = len(locs)
        x_label = np.zeros(N, dtype=np.int64)
        y_label = np.zeros(N, dtype=np.int64)
        
        # 填充有效点的投影标签
        valid_indices = frame_data["valid_indices"]
        if len(valid_indices) == len(frame_data["x_label"]):
            x_label[valid_indices] = frame_data["x_label"]
            y_label[valid_indices] = frame_data["y_label"]
        
        print(f"Updated {len(valid_indices)} points with projection labels for {scene_path.name}")
    
    # 保存更新后的数据
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建包含投影标签的数据字典
    updated_data = {
        "locs": locs,
        "feats": feats,
        "labels": labels,
        "x_label": x_label,
        "y_label": y_label,
    }
    
    torch.save(updated_data, output_path)
    print(f"Saved updated data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate projection labels for ScanNet data")
    parser.add_argument("--data-root", type=str, required=True,
                       help="Path to ScanNet 3D data root")
    parser.add_argument("--data-2d-root", type=str, required=True,
                       help="Path to ScanNet 2D data root")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for updated data")
    parser.add_argument("--split", type=str, default="train",
                       help="Data split to process")
    parser.add_argument("--scene-pattern", type=str, default="scene*",
                       help="Pattern to match scene names")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    data_2d_root = Path(args.data_2d_root)
    output_dir = Path(args.output_dir)
    
    # 初始化点到图像映射器
    point2img_mapper = getMapping()
    
    # 获取所有场景文件
    split_dir = data_root / args.split
    scene_files = list(split_dir.glob(f"{args.scene_pattern}.pth"))
    if len(scene_files) == 0:
        scene_files = list(split_dir.glob(f"{args.scene_pattern}_vh_clean_2.pth"))
    
    print(f"Found {len(scene_files)} scene files")
    
    for scene_file in tqdm(scene_files, desc="Processing scenes"):
        scene_name = scene_file.stem.replace("_vh_clean_2", "")
        scene_2d_dir = data_2d_root / scene_name
        
        if not scene_2d_dir.exists():
            print(f"Warning: 2D data not found for {scene_name}")
            continue
        
        try:
            # 加载 3D 数据
            locs, feats, labels = load_scene_data(scene_file)
            
            # 计算投影标签
            projection_data = compute_projection_labels(
                locs, scene_2d_dir, point2img_mapper
            )
            
            # 更新并保存数据
            output_path = output_dir / args.split / scene_file.name
            update_scene_data(scene_file, projection_data, output_path)
            
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
            continue
    
    print("Projection label generation completed!")


if __name__ == "__main__":
    main()