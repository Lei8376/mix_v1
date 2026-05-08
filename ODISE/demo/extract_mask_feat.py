# here we realize the function:
# 1.  extract masks from odise model
# 2. clip the image according to the mask
# 3. encoding mask features and caption txt features

import torch
import time

import argparse
import glob
import itertools
import numpy as np
import os
import tempfile
import time
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
# use beautiful coco colors
LVIS_COLORS = list(
    itertools.islice(itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES))
)

SCANNET_20_THINGS_CLASSES = list(SCANNET_LABELS_20)
SCANNET_20_THINGS_COLOR = [list(x) for x in SCANNET_COLOR_MAP_20.values()]


def class_label(label_list, vocab):
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
        demo_thing_classes+=COCO_THING_CLASSES
        demo_stuff_classes+=COCO_STUFF_CLASSES
        demo_thing_colors += COCO_THING_COLORS
        demo_stuff_colors += COCO_STUFF_COLORS
    if "ADE" in label_list:
        demo_thing_classes+=ADE_THING_CLASSES
        demo_stuff_classes+=ADE_STUFF_CLASSES
        demo_thing_colors += ADE_THING_COLORS
        demo_stuff_colors += ADE_STUFF_COLORS
    if "LVIS" in label_list:
        demo_thing_classes+=LVIS_CLASSES
        demo_thing_colors += LVIS_COLORS
    if "SCANNET_20" in label_list:
        demo_thing_classes+=SCANNET_20_THINGS_CLASSES
        demo_thing_colors += SCANNET_20_THINGS_COLOR
    
    MetadataCatalog.pop("odise_demo_metadata",None)
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


#img_file = "/home/sunl/work/data/Replica_RGBD/Replica/room0/results/frame000001.jpg"
img_file = "/home/sunl/work/mix/data/scannet_2d/scene0000_00/color/1120.jpg"
input_image = Image.open(img_file)
label_list = ["SCANNET_20", "COCO", "ADE", "LVIS"]
#label_list = ["COCO", "ADE", "LVIS", "SCANNET_20"]
vocab=""

demo_metadata, demo_classes = class_label(label_list, vocab)

#print(demo_metadata.stuff_classes)
#print(demo_metadata.stuff_colors)
#print(demo_metadata.stuff_dataset_id_to_contiguous_id)
#print(demo_metadata.thing_classes)
#print(demo_metadata.thing_colors)
#print(demo_metadata.thing_dataset_id_to_contiguous_id)

pred = torch.load("/home/sunl/work/data/scannet/mask/scene0000_00/predictions_0.pt")

#print(pred[0]['panoptic_seg'][1])

mask_info = pred[0]['panoptic_seg'][1]
print(len(mask_info))
mask_img = pred[0]['panoptic_seg'][0]
#print(mask_img)

mask_list = []

# Extract individual masks based on mask_info
for segment_info in mask_info:
    segment_id = segment_info['id']
    is_thing = segment_info['isthing']
    category_id = segment_info['category_id']
    area = segment_info['area']
    
    # Get the corresponding category name
    if is_thing:
        contiguous_id = demo_metadata.thing_dataset_id_to_contiguous_id.get(category_id, None)
        if contiguous_id is not None:
            category_name = demo_metadata.thing_classes[contiguous_id]
        else:
            category_name = f"unknown_thing_{category_id}"
    else:
        contiguous_id = demo_metadata.stuff_dataset_id_to_contiguous_id.get(category_id, None)
        if contiguous_id is not None:
            category_name = demo_metadata.stuff_classes[contiguous_id]
        else:
            category_name = f"unknown_stuff_{category_id}"
    
    # Extract the mask for this segment
    individual_mask = (mask_img == segment_id)
    #print(individual_mask)
    
    #print(f"Segment ID: {segment_id}, Category: {category_name}, Is Thing: {is_thing}, Area: {area}")
    print(f"Mask shape: {individual_mask.shape}, True pixels: {individual_mask.sum().item()}")
    mask_list.append({"mask": individual_mask, "object": category_name})

    
    # The individual_mask is a boolean tensor where True indicates pixels belonging to this segment
    # You can use this mask for further processing like cropping the original image

mask_0 = mask_list[0]
print(mask_0["mask"])
print(mask_0["object"])
#for each image, we are able to extract masks and their corresponding object names. The mask is used to crop the image to extract image features. Moreover, the label is used to extract caption features.



embed_pred = torch.load("/home/featurize/work/data/scannet/mask/scene0000_00/mask_embed_1540.pt")
print(embed_pred.shape)


