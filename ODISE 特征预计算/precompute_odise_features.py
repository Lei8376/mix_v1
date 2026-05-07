"""
Precompute ODISE features (masks + embeddings) for training.

This script processes images and saves:
- ODISE masks: (K, H, W) bool
- ODISE mask embeddings: (K, 256) float16
- Metadata: num_masks, category info

Usage:
    # Activate environment first
    conda activate f_bak
    
    # Run script
    python scripts/precompute_odise_features.py \
        --data-root /path/to/scannet/scans \
        --output-dir /path/to/output \
        --odise-model-config Panoptic/odise_caption_coco_50e.py \
        --max-scenes 10 \
        --max-images-per-scene 50
"""

# IMPORTANT: Setup paths BEFORE any other imports
import os
import sys

# Add ODISE third_party paths (must be done before importing ODISE)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
odise_root = os.path.join(project_root, "ODISE")
mask2former_path = os.path.join(odise_root, "third_party", "Mask2Former")

# Insert at the beginning of sys.path
if mask2former_path not in sys.path:
    sys.path.insert(0, mask2former_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import everything else (but NOT ODISE yet - will import when needed)
import argparse
from glob import glob
from os.path import join, basename, dirname
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def get_scene_image_paths(data_root: str, scene_pattern: str = "scene*") -> Dict[str, list]:
    """
    Get all scene names and their image paths.
    
    Args:
        data_root: Root directory containing scene folders
        scene_pattern: Pattern to match scene directories
        
    Returns:
        Dict mapping scene_name -> list of image paths
    """
    scenes = {}
    scene_dirs = sorted(glob(join(data_root, scene_pattern)))
    
    for scene_dir in scene_dirs:
        scene_name = basename(scene_dir)
        
        # Try multiple possible image directory structures
        possible_paths = [
            join(scene_dir, "color", "*.jpg"),
            join(scene_dir, "color", "*.png"),
            join(scene_dir, "*.jpg"),
            join(scene_dir, "*.png"),
        ]
        
        img_paths = []
        for pattern in possible_paths:
            found = glob(pattern)
            if found:
                img_paths = found
                break
        
        if img_paths:
            # Sort by image number if possible
            try:
                img_paths = sorted(
                    img_paths,
                    key=lambda x: int(basename(x).split('.')[0])
                )
            except ValueError:
                # If sorting by number fails, just sort alphabetically
                img_paths = sorted(img_paths)
            
            scenes[scene_name] = img_paths
    
    return scenes


def precompute_odise_features(
    odise_extractor: of.ODISEMaskEmbeddingExtractor,
    img_path: str,
) -> Dict[str, np.ndarray]:
    """
    Extract ODISE masks and embeddings from an image.
    
    Args:
        odise_extractor: ODISE feature extractor
        img_path: Path to input image
        
    Returns:
        Dict containing:
            - masks: (K, H, W) bool array
            - mask_embeddings: (K, 256) float16 array
            - num_masks: int64 scalar
            - info: object array with category info
    """
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np)
    
    with torch.no_grad():
        results = odise_extractor.extract(img_tensor)
    
    num_masks = results["num_masks"]
    
    # Handle case with no masks
    if num_masks == 0:
        H, W = img_np.shape[:2]
        return {
            "masks": np.zeros((0, H, W), dtype=bool),
            "mask_embeddings": np.zeros((0, 256), dtype=np.float16),
            "num_masks": np.array(0, dtype=np.int64),
            "info": np.array([], dtype=object),
        }
    
    # Convert masks to numpy
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
    parser = argparse.ArgumentParser(
        description="Precompute ODISE features for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all scenes
    python scripts/precompute_odise_features.py \\
        --data-root /data/scannet/scans \\
        --output-dir /data/precomputed_odise
    
    # Process limited number of scenes for testing
    python scripts/precompute_odise_features.py \\
        --data-root /data/scannet/scans \\
        --output-dir /data/test_output \\
        --max-scenes 2 \\
        --max-images-per-scene 10
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing scene folders (e.g., /data/scannet/scans)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for precomputed features",
    )
    
    # ODISE model arguments
    parser.add_argument(
        "--odise-model-config",
        type=str,
        default="Panoptic/odise_caption_coco_50e.py",
        help="ODISE model config name (default: Panoptic/odise_caption_coco_50e.py)",
    )
    parser.add_argument(
        "--label-sets",
        type=str,
        nargs="+",
        default=["COCO", "ADE", "LVIS", "SCANNET_20"],
        help="Label sets to use (default: COCO ADE LVIS SCANNET_20)",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="",
        help="Additional vocabulary (comma-separated words, semicolon-separated groups)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.0,
        help="Overlap threshold for mask merging (default: 0.0)",
    )
    parser.add_argument(
        "--object-mask-threshold",
        type=float,
        default=0.0,
        help="Score threshold for keeping masks (default: 0.0)",
    )
    
    # Processing options
    parser.add_argument(
        "--scene-pattern",
        type=str,
        default="scene*",
        help="Pattern to match scene directories (default: scene*)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=-1,
        help="Max number of scenes to process (-1 for all, default: -1)",
    )
    parser.add_argument(
        "--max-images-per-scene",
        type=int,
        default=-1,
        help="Max images per scene (-1 for all, default: -1)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (default: cuda if available)",
    )
    
    args = parser.parse_args()
    
    # Check if data root exists
    if not os.path.exists(args.data_root):
        print(f"ERROR: Data root does not exist: {args.data_root}")
        sys.exit(1)
    
    # Get all scenes and images
    print(f"Scanning for scenes in: {args.data_root}")
    scenes = get_scene_image_paths(args.data_root, args.scene_pattern)
    print(f"Found {len(scenes)} scenes")
    
    if len(scenes) == 0:
        print(f"ERROR: No scenes found in {args.data_root}")
        print(f"Expected directory structure: {args.data_root}/{args.scene_pattern}/color/*.jpg")
        sys.exit(1)
    
    # Limit scenes if requested
    if args.max_scenes > 0:
        scene_names = list(scenes.keys())[:args.max_scenes]
        scenes = {k: scenes[k] for k in scene_names}
        print(f"Limited to {len(scenes)} scenes")
    
    # Print some examples
    for i, (scene_name, img_paths) in enumerate(list(scenes.items())[:3]):
        print(f"  {scene_name}: {len(img_paths)} images")
        if i == 2 and len(scenes) > 3:
            print(f"  ... and {len(scenes) - 3} more scenes")
    
    # Initialize ODISE extractor
    print(f"\nLoading ODISE extractor...")
    print(f"  Model config: {args.odise_model_config}")
    print(f"  Label sets: {args.label_sets}")
    print(f"  Device: {args.device}")
    
    try:
        odise_extractor = of.ODISEMaskEmbeddingExtractor(
            model_config=args.odise_model_config,
            label_sets=args.label_sets,
            vocab=args.vocab,
            overlap_threshold=args.overlap_threshold,
            object_mask_threshold=args.object_mask_threshold,
            device=args.device,
        )
        print("ODISE extractor loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load ODISE extractor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")
    
    # Process scenes
    total_images = sum(len(paths) for paths in scenes.values())
    print(f"\nProcessing {total_images} images across {len(scenes)} scenes...")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    with tqdm(total=total_images, desc="Overall progress") as pbar:
        for scene_name, img_paths in scenes.items():
            scene_output_dir = join(args.output_dir, scene_name)
            os.makedirs(scene_output_dir, exist_ok=True)
            
            # Limit images per scene if requested
            if args.max_images_per_scene > 0:
                img_paths = img_paths[:args.max_images_per_scene]
            
            for img_path in img_paths:
                # Get image identifier (without extension)
                img_name = basename(img_path)
                img_idx = os.path.splitext(img_name)[0]
                
                # Output path
                output_path = join(scene_output_dir, f"{img_idx}_odise.npz")
                
                # Skip if exists and requested
                if args.skip_existing and os.path.exists(output_path):
                    skipped_count += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "processed": processed_count,
                        "skipped": skipped_count,
                        "errors": error_count
                    })
                    continue
                
                # Process image
                try:
                    odise_feat = precompute_odise_features(odise_extractor, img_path)
                    np.savez_compressed(output_path, **odise_feat)
                    processed_count += 1
                except Exception as e:
                    print(f"\nError processing {img_path}: {e}")
                    error_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    "processed": processed_count,
                    "skipped": skipped_count,
                    "errors": error_count
                })
    
    # Summary
    print("\n" + "="*60)
    print("Precomputation complete!")
    print("="*60)
    print(f"Total scenes:      {len(scenes)}")
    print(f"Total images:      {total_images}")
    print(f"Processed:         {processed_count}")
    print(f"Skipped (exists):  {skipped_count}")
    print(f"Errors:            {error_count}")
    print(f"Output directory:  {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
