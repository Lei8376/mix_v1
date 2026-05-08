"""
Precompute ODISE features (masks + embeddings) for training - Version 2 with YAML config.

This script processes images and saves:
- ODISE masks: (K, H, W) bool
- ODISE mask embeddings: (K, 256) float16
- Metadata: num_masks, category info

Usage:
    # Method 1: Using config file (recommended)
    conda activate mix
    python scripts/precompute_odise_features_v2.py --config config/odise_config.yaml
    
    # Method 2: Override config with command line arguments
    python scripts/precompute_odise_features_v2.py \
        --config config/odise_config.yaml \
        --data-root /custom/path \
        --batch-size 4
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
from os.path import join, basename, dirname, exists
from typing import Dict, Any, List

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(yaml_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge YAML config with command line arguments.
    Command line arguments have higher priority.
    """
    config = yaml_config.copy()
    
    # Override with command line arguments if provided
    if args.data_root is not None:
        config['data']['data_root'] = args.data_root
    if args.output_dir is not None:
        config['data']['output_dir'] = args.output_dir
    if args.batch_size is not None:
        config['processing']['batch_size'] = args.batch_size
    if args.max_scenes is not None:
        config['processing']['max_scenes'] = args.max_scenes
    if args.max_images_per_scene is not None:
        config['processing']['max_images_per_scene'] = args.max_images_per_scene
    if args.device is not None:
        config['processing']['device'] = args.device
    if args.skip_existing is not None:
        config['processing']['skip_existing'] = args.skip_existing
    
    return config


def get_scene_image_paths(
    data_root: str,
    scene_pattern: str = "scene*",
    image_extensions: List[str] = None
) -> Dict[str, list]:
    """
    Get all scene names and their image paths.
    
    Args:
        data_root: Root directory containing scene folders
        scene_pattern: Pattern to match scene directories
        image_extensions: List of image extensions to look for
        
    Returns:
        Dict mapping scene_name -> list of image paths
    """
    if image_extensions is None:
        image_extensions = ["jpg", "png"]
    
    scenes = {}
    scene_dirs = sorted(glob(join(data_root, scene_pattern)))
    
    for scene_dir in scene_dirs:
        scene_name = basename(scene_dir)
        
        # Try multiple possible image directory structures
        img_paths = []
        for ext in image_extensions:
            possible_paths = [
                join(scene_dir, "color", f"*.{ext}"),
                join(scene_dir, f"*.{ext}"),
            ]
            
            for pattern in possible_paths:
                found = glob(pattern)
                if found:
                    img_paths.extend(found)
        
        if img_paths:
            # Sort by image number if possible
            try:
                img_paths = sorted(
                    img_paths,
                    key=lambda x: int(os.path.splitext(basename(x))[0])
                )
            except ValueError:
                # If sorting by number fails, just sort alphabetically
                img_paths = sorted(img_paths)
            
            scenes[scene_name] = img_paths
    
    return scenes


def precompute_odise_features_batch(
    odise_extractor,  # Type: ODISEMaskEmbeddingExtractor
    img_paths: List[str],
) -> List[Dict[str, np.ndarray]]:
    """
    Extract ODISE masks and embeddings from a batch of images.
    
    Args:
        odise_extractor: ODISE feature extractor
        img_paths: List of image paths
        
    Returns:
        List of dicts, each containing masks, embeddings, and metadata
    """
    results = []
    
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np)
        
        with torch.no_grad():
            result = odise_extractor.extract(img_tensor)
        
        num_masks = result["num_masks"]
        
        # Handle case with no masks
        if num_masks == 0:
            H, W = img_np.shape[:2]
            results.append({
                "masks": np.zeros((0, H, W), dtype=bool),
                "mask_embeddings": np.zeros((0, 256), dtype=np.float16),
                "num_masks": np.array(0, dtype=np.int64),
                "info": np.array([], dtype=object),
            })
            continue
        
        # Convert masks to numpy
        masks = torch.stack(result["masks"]).cpu().numpy().astype(bool)
        mask_embeddings = result["mask_embeddings"].cpu().numpy().astype(np.float16)
        
        # Store category info
        info = []
        for r in result["results"]:
            info.append({
                "category_name": r.category_name,
                "category_id": r.category_id,
                "is_thing": r.is_thing,
                "score": r.score,
                "area": r.area,
            })
        
        results.append({
            "masks": masks,
            "mask_embeddings": mask_embeddings,
            "num_masks": np.array(num_masks, dtype=np.int64),
            "info": np.array(info, dtype=object),
        })
    
    return results


def save_features(
    output_path: str,
    features: Dict[str, np.ndarray],
    compressed: bool = True
):
    """Save features to disk."""
    if compressed:
        np.savez_compressed(output_path, **features)
    else:
        np.savez(output_path, **features)


def print_config_summary(config: Dict[str, Any]):
    """Print configuration summary."""
    print("="*70)
    print("Configuration Summary")
    print("="*70)
    print(f"Data root:              {config['data']['data_root']}")
    print(f"Output directory:       {config['data']['output_dir']}")
    print(f"Scene pattern:          {config['data']['scene_pattern']}")
    print(f"Batch size:             {config['processing']['batch_size']}")
    print(f"Max scenes:             {config['processing']['max_scenes']}")
    print(f"Max images per scene:   {config['processing']['max_images_per_scene']}")
    print(f"Skip existing:          {config['processing']['skip_existing']}")
    print(f"Device:                 {config['processing']['device']}")
    print(f"ODISE model:            {config['odise']['model_config']}")
    print(f"Label sets:             {', '.join(config['odise']['label_sets'])}")
    print("="*70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Precompute ODISE features using YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file (required)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    
    # Optional overrides for config file
    parser.add_argument("--data-root", type=str, help="Override data root path")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--max-scenes", type=int, help="Override max scenes")
    parser.add_argument("--max-images-per-scene", type=int, help="Override max images per scene")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument("--skip-existing", type=bool, help="Override skip existing flag")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    yaml_config = load_config(args.config)
    config = merge_configs(yaml_config, args)
    
    # Print config summary
    print_config_summary(config)
    
    # Extract config values
    data_root = config['data']['data_root']
    output_dir = config['data']['output_dir']
    scene_pattern = config['data']['scene_pattern']
    image_extensions = config['data']['image_extensions']
    
    batch_size = config['processing']['batch_size']
    max_scenes = config['processing']['max_scenes']
    max_images_per_scene = config['processing']['max_images_per_scene']
    skip_existing = config['processing']['skip_existing']
    device = config['processing']['device']
    verbose = config['processing']['verbose']
    
    odise_config = config['odise']
    output_config = config['output']
    
    # Validate paths
    if not exists(data_root):
        print(f"ERROR: Data root does not exist: {data_root}")
        sys.exit(1)
    
    # Get all scenes and images
    print(f"Scanning for scenes in: {data_root}")
    scenes = get_scene_image_paths(data_root, scene_pattern, image_extensions)
    print(f"Found {len(scenes)} scenes")
    
    if len(scenes) == 0:
        print(f"ERROR: No scenes found in {data_root}")
        print(f"Expected structure: {data_root}/{scene_pattern}/color/*.jpg")
        sys.exit(1)
    
    # Limit scenes if requested
    if max_scenes > 0:
        scene_names = list(scenes.keys())[:max_scenes]
        scenes = {k: scenes[k] for k in scene_names}
        print(f"Limited to {len(scenes)} scenes")
    
    # Print sample scenes
    if verbose:
        for i, (scene_name, img_paths) in enumerate(list(scenes.items())[:3]):
            print(f"  {scene_name}: {len(img_paths)} images")
            if i == 2 and len(scenes) > 3:
                print(f"  ... and {len(scenes) - 3} more scenes")
        print()
    
    # Initialize ODISE extractor (import here to avoid circular import issues)
    print("Loading ODISE extractor...")
    print(f"  Model: {odise_config['model_config']}")
    print(f"  Device: {device}")
    
    try:
        # Import ODISE only when needed (after paths are set)
        from ODISE import odise_feature as of
        
        odise_extractor = of.ODISEMaskEmbeddingExtractor(
            model_config=odise_config['model_config'],
            label_sets=odise_config['label_sets'],
            vocab=odise_config['vocab'],
            overlap_threshold=odise_config['overlap_threshold'],
            object_mask_threshold=odise_config['object_mask_threshold'],
            device=device,
        )
        print("✅ ODISE extractor loaded successfully!")
        print()
    except Exception as e:
        print(f"❌ ERROR: Failed to load ODISE extractor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process scenes
    total_images = sum(len(paths) for paths in scenes.values())
    print(f"Processing {total_images} images across {len(scenes)} scenes...")
    print(f"Batch size: {batch_size}")
    print()
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    with tqdm(total=total_images, desc="Overall progress", disable=not verbose) as pbar:
        for scene_name, img_paths in scenes.items():
            scene_output_dir = join(output_dir, scene_name)
            os.makedirs(scene_output_dir, exist_ok=True)
            
            # Limit images per scene if requested
            if max_images_per_scene > 0:
                img_paths = img_paths[:max_images_per_scene]
            
            # Process in batches
            for i in range(0, len(img_paths), batch_size):
                batch_paths = img_paths[i:i + batch_size]
                
                for img_path in batch_paths:
                    # Get image identifier
                    img_name = basename(img_path)
                    img_idx = os.path.splitext(img_name)[0]
                    
                    # Output path
                    output_filename = output_config['output_name_template'].format(
                        img_idx=img_idx,
                        scene_name=scene_name
                    )
                    output_path = join(scene_output_dir, output_filename)
                    
                    # Skip if exists and requested
                    if skip_existing and exists(output_path):
                        skipped_count += 1
                        pbar.update(1)
                        if verbose:
                            pbar.set_postfix({
                                "processed": processed_count,
                                "skipped": skipped_count,
                                "errors": error_count
                            })
                        continue
                    
                    # Process image
                    try:
                        features = precompute_odise_features_batch(
                            odise_extractor, [img_path]
                        )[0]
                        
                        save_features(
                            output_path,
                            features,
                            compressed=output_config['compressed']
                        )
                        processed_count += 1
                    except Exception as e:
                        if verbose:
                            print(f"\n❌ Error processing {img_path}: {e}")
                        error_count += 1
                    
                    pbar.update(1)
                    if verbose:
                        pbar.set_postfix({
                            "processed": processed_count,
                            "skipped": skipped_count,
                            "errors": error_count
                        })
    
    # Summary
    print("\n" + "="*70)
    print("Precomputation Complete!")
    print("="*70)
    print(f"Total scenes:          {len(scenes)}")
    print(f"Total images:          {total_images}")
    print(f"✅ Processed:          {processed_count}")
    print(f"⏭️  Skipped (exists):   {skipped_count}")
    print(f"❌ Errors:             {error_count}")
    print(f"📁 Output directory:   {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
