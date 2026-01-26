import torch
import numpy as np
from PIL import Image
#import extract_mask_feat as emt

import torch
import time
import argparse
import glob
import itertools
import numpy as np
import os
import tempfile
import warnings
import requests
from PIL import Image
from contextlib import ExitStack
import cv2
import nltk
import torch
import tqdm
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.engine import create_ddp_model
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
#load scannet labels
from scannet_label_constant import SCANNET_LABELS_20, SCANNET_COLOR_MAP_20, SCANNET_LABELS_200, SCANNET_COLOR_MAP_200

from torch import nn

from odise import model_zoo
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.engine.defaults import get_model_from_module
from odise.modeling.wrapper import OpenPanopticInference


# Global constants
COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
LVIS_COLORS = list(
    itertools.islice(itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES))
)

SCANNET_20_THINGS_CLASSES = list(SCANNET_LABELS_20)
SCANNET_20_THINGS_COLOR = [list(x) for x in SCANNET_COLOR_MAP_20.values()]


def setup_class_labels(label_list, vocab=""):
    """
    Set up class labels and metadata for the demo.
    
    Args:
        label_list (List[str]): List of dataset names to include (e.g., ["COCO", "ADE", "LVIS"])
        vocab (str): Additional vocabulary in format "class1,alias1;class2,alias2"
    
    Returns:
        Tuple[MetadataCatalog, List]: demo_metadata and demo_classes
    """
    extra_classes = []
    if vocab:
        for words in vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]

    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if "COCO" in label_list:
        demo_thing_classes += COCO_THING_CLASSES
        demo_stuff_classes += COCO_STUFF_CLASSES
        demo_thing_colors += COCO_THING_COLORS
        demo_stuff_colors += COCO_STUFF_COLORS
    if "ADE" in label_list:
        demo_thing_classes += ADE_THING_CLASSES
        demo_stuff_classes += ADE_STUFF_CLASSES
        demo_thing_colors += ADE_THING_COLORS
        demo_stuff_colors += ADE_STUFF_COLORS
    if "LVIS" in label_list:
        demo_thing_classes += LVIS_CLASSES
        demo_thing_colors += LVIS_COLORS
    if "SCANNET_20" in label_list:
        demo_thing_classes += SCANNET_20_THINGS_CLASSES
        demo_thing_colors += SCANNET_20_THINGS_COLOR
    
    MetadataCatalog.pop("odise_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors

    demo_metadata.thing_dataset_id_to_contiguous_id = {idx: idx for idx in range(len(demo_metadata.thing_classes))}
    demo_metadata.stuff_dataset_id_to_contiguous_id = {idx: idx for idx in range(len(demo_metadata.stuff_classes))}

    demo_classes = demo_thing_classes + demo_stuff_classes

    return demo_metadata, demo_classes


# Load the metadata used during inference to map category IDs to class names
label_list = ["COCO", "ADE", "LVIS"]  # Use the same labels as during inference
demo_metadata, demo_classes = setup_class_labels(label_list)

img_file = "/home/featurize/work/data/Replica_RGBD/Replica/room0/results/frame000001.jpg"
input_image = Image.open(img_file)
height,width = input_image.size

mask_binary = torch.zeros(width,height)

predictions = torch.load("predictions.pt")

mask_img = predictions['panoptic_seg'][0]
class_cat = predictions['panoptic_seg'][1]

print(mask_binary.shape)
print(mask_img)
print(class_cat)

# Count unique classes in mask_img
unique_classes = torch.unique(mask_img)
print(unique_classes)
num_classes = len(unique_classes)
print(f"Number of unique classes in mask: {num_classes}")
print(f"Unique class IDs: {unique_classes.tolist()}")

unique_classes = unique_classes.tolist()

cnt = 0
for i in range(mask_img.shape[0]):
    for j in range(mask_img.shape[1]):
        if mask_img[i,j]==15:
            cnt+=1

print(cnt)

# Helper function to get class name from category ID
def get_class_name(category_id, metadata):
    """
    Get the actual class name from category ID using metadata.
    
    Args:
        category_id (int): The category ID from predictions
        metadata: The demo metadata containing class names
    
    Returns:
        str: The actual class/object name
    """
    if category_id < len(metadata.stuff_classes):
        return metadata.stuff_classes[category_id]
    else:
        return f"unknown_class_{category_id}"

# Split mask_image into individual masks with labels
individual_masks = {}
mask_labels = {}

# Get all unique segment IDs (excluding background)
segment_ids = [cat["id"] for cat in class_cat]

print(f"\n{'='*60}")
print("EXTRACTING INDIVIDUAL MASKS WITH OBJECT NAMES")
print(f"{'='*60}")

for cat_info in class_cat:
    segment_id = cat_info["id"]
    category_id = cat_info["category_id"]
    isthing = cat_info["isthing"]
    
    # Get the actual class/object name
    class_name = get_class_name(category_id, demo_metadata)
    
    # Create binary mask for this specific class
    class_mask = (mask_img == segment_id).float()
    
    # Store the mask and its label information
    individual_masks[segment_id] = class_mask
    mask_labels[segment_id] = {
        "category_id": category_id,
        "class_name": class_name,  # Add the actual object name
        "isthing": isthing,
        "segment_id": segment_id,
        "mask_area": torch.sum(class_mask).item()
    }

print(f"Created {len(individual_masks)} individual masks")
print(f"\n{'Segment ID':<12} {'Category ID':<12} {'Object Name':<20} {'Type':<8} {'Area (pixels)':<15}")
print("-" * 75)

for segment_id, label_info in mask_labels.items():
    object_type = "Thing" if label_info['isthing'] else "Stuff"
    print(f"{segment_id:<12} {label_info['category_id']:<12} {label_info['class_name']:<20} "
          f"{object_type:<8} {label_info['mask_area']:<15.0f}")

# Print summary of detected objects
print(f"\n{'='*40}")
print("DETECTED OBJECTS SUMMARY")
print(f"{'='*40}")
object_counts = {}
for label_info in mask_labels.values():
    class_name = label_info['class_name']
    if class_name in object_counts:
        object_counts[class_name] += 1
    else:
        object_counts[class_name] = 1

for class_name, count in sorted(object_counts.items()):
    print(f"{class_name}: {count} instance(s)")

# Example: Access individual masks by object name
print(f"\n{'='*40}")
print("EXAMPLE: ACCESS MASKS BY OBJECT NAME")
print(f"{'='*40}")

# Create a reverse mapping from class name to segment info
class_name_to_segments = {}
for segment_id, label_info in mask_labels.items():
    class_name = label_info['class_name']
    if class_name not in class_name_to_segments:
        class_name_to_segments[class_name] = []
    class_name_to_segments[class_name].append({
        'segment_id': segment_id,
        'mask': individual_masks[segment_id],
        'area': label_info['mask_area']
    })

# Example usage
for class_name, segments in class_name_to_segments.items():
    print(f"\n{class_name}: {len(segments)} instance(s)")
    for i, segment in enumerate(segments):
        print(f"  Instance {i+1}: Segment ID {segment['segment_id']}, Area: {segment['area']:.0f} pixels")

# Example: Get all masks for a specific object type (e.g., "chair")
def get_masks_for_object(object_name, class_name_to_segments_dict):
    """
    Get all masks for a specific object type.
    
    Args:
        object_name (str): Name of the object (e.g., "chair", "table")
        class_name_to_segments_dict (dict): Mapping from class names to segments
    
    Returns:
        List[dict]: List of segment information for the specified object
    """
    return class_name_to_segments_dict.get(object_name, [])

# Example usage:
if "chair" in class_name_to_segments:
    chair_masks = get_masks_for_object("chair", class_name_to_segments)
    print(f"\nFound {len(chair_masks)} chair(s) in the image")
    for i, chair in enumerate(chair_masks):
        print(f"Chair {i+1}: Segment ID {chair['segment_id']}, Area: {chair['area']:.0f} pixels")
















