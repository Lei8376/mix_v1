"""
Precompute 2D features (LSeg pixel embeddings + ODISE masks/embeddings) for training.

This script processes ScanNet images and saves:
- LSeg pixel embeddings: (H, W, 512) float16
- ODISE masks: (K, H, W) bool
- ODISE mask embeddings: (K, 256) float16
- Metadata: num_masks, category info

Usage:
    python scripts/precompute_2d_features.py \
        --data-config-path /path/to/config.yaml \
        --output-dir /path/to/output \
        --label-path lang_seg/label_files/ade20k_objectInfo150.txt \
        --lseg-ckpt-path lang_seg/checkpoints/demo_e200.ckpt \
        --odise-model-config-path Panoptic/odise_caption_coco_50e.py
"""

import argparse
import os
import sys
from glob import glob
from os.path import join
from typing import Dict, Any

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lang_seg import lseg_feature as lf
from ODISE import odise_feature as of


def read_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_scene_image_paths(data_root_2d: str) -> Dict[str, list]:
    """Get all scene names and their image paths."""
    scenes = {}
    scene_dirs = sorted(glob(join(data_root_2d, "scene*")))
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        img_paths = sorted(
            glob(join(scene_dir, "color", "*.jpg")),
            key=lambda x: int(os.path.basename(x).replace(".jpg", ""))
        )
        if img_paths:
            scenes[scene_name] = img_paths
    return scenes


def precompute_lseg_features(
    lseg_extractor: lf.LSegExtractor,
    img_path: str,
) -> np.ndarray:
    """Extract LSeg pixel embeddings from an image."""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    # LSegExtractor expects numpy array or PIL Image
    feat = lseg_extractor(img_np)  # Returns (H, W, C) float16
    return feat


def precompute_odise_features(
    odise_extractor: of.ODISEMaskEmbeddingExtractor,
    img_path: str,
) -> Dict[str, np.ndarray]:
    """Extract ODISE masks and embeddings from an image."""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np)
    
    with torch.no_grad():
        results = odise_extractor.extract(img_tensor)
    
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
    
    # Store category info
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
    parser = argparse.ArgumentParser(description="Precompute 2D features for training")
    parser.add_argument(
        "--data-config-path",
        type=str,
        default="/home/sunl/work/mix/config/data_scannet_3d.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/sunl/work/mix/data/precomputed_2d",
        help="Output directory for precomputed features",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        default="lang_seg/label_files/ade20k_objectInfo150.txt",
        help="Path to LSeg label file",
    )
    parser.add_argument(
        "--lseg-ckpt-path",
        type=str,
        default="lang_seg/checkpoints/demo_e200.ckpt",
        help="Path to LSeg checkpoint",
    )
    parser.add_argument(
        "--odise-model-config-path",
        type=str,
        default="Panoptic/odise_caption_coco_50e.py",
        help="ODISE model config name",
    )
    parser.add_argument(
        "--skip-lseg",
        action="store_true",
        help="Skip LSeg feature extraction",
    )
    parser.add_argument(
        "--skip-odise",
        action="store_true",
        help="Skip ODISE feature extraction",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=-1,
        help="Max number of scenes to process (-1 for all)",
    )
    parser.add_argument(
        "--max-images-per-scene",
        type=int,
        default=-1,
        help="Max images per scene (-1 for all)",
    )
    args = parser.parse_args()

    # Load dataset config
    config = read_yaml(args.data_config_path)
    data_root_2d = config["DATA"]["data_root_2d"]

    # Resolve paths
    if not os.path.isabs(args.label_path):
        args.label_path = os.path.join(project_root, args.label_path)
    if not os.path.isabs(args.lseg_ckpt_path):
        args.lseg_ckpt_path = os.path.join(project_root, args.lseg_ckpt_path)

    # Get all scenes and images
    scenes = get_scene_image_paths(data_root_2d)
    print(f"Found {len(scenes)} scenes")

    if args.max_scenes > 0:
        scene_names = list(scenes.keys())[:args.max_scenes]
        scenes = {k: scenes[k] for k in scene_names}
        print(f"Processing {len(scenes)} scenes (limited)")

    # Initialize extractors
    lseg_extractor = None
    odise_extractor = None

    if not args.skip_lseg:
        print("Loading LSeg extractor...")
        lseg_extractor = lf.LSegExtractor(args.label_path, args.lseg_ckpt_path)
        print("LSeg extractor loaded")

    if not args.skip_odise:
        print("Loading ODISE extractor...")
        odise_extractor = of.ODISEMaskEmbeddingExtractor(args.odise_model_config_path)
        print("ODISE extractor loaded")

    # Process scenes
    os.makedirs(args.output_dir, exist_ok=True)

    for scene_name, img_paths in tqdm(scenes.items(), desc="Scenes"):
        scene_output_dir = join(args.output_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        if args.max_images_per_scene > 0:
            img_paths = img_paths[:args.max_images_per_scene]

        for img_path in tqdm(img_paths, desc=f"{scene_name}", leave=False):
            img_idx = os.path.basename(img_path).replace(".jpg", "")

            # LSeg features
            if lseg_extractor is not None:
                lseg_output_path = join(scene_output_dir, f"{img_idx}_lseg.npy")
                if not os.path.exists(lseg_output_path):
                    try:
                        lseg_feat = precompute_lseg_features(lseg_extractor, img_path)
                        np.save(lseg_output_path, lseg_feat)
                    except Exception as e:
                        print(f"Error processing LSeg for {img_path}: {e}")

            # ODISE features
            if odise_extractor is not None:
                odise_output_path = join(scene_output_dir, f"{img_idx}_odise.npz")
                if not os.path.exists(odise_output_path):
                    try:
                        odise_feat = precompute_odise_features(odise_extractor, img_path)
                        np.savez_compressed(odise_output_path, **odise_feat)
                    except Exception as e:
                        print(f"Error processing ODISE for {img_path}: {e}")

    print("Precomputation complete!")


if __name__ == "__main__":
    main()

