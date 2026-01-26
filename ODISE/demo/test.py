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


class VisualizationDemo(object):
    def __init__(self, model, metadata, aug, instance_mode=ColorMode.IMAGE):
        self.model = model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

    def predict(self, original_image):
        """
        Input: original_image(np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns: prediction (dict): the output of the model for one image only.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        #print(image.shape)
        image = image.to("cuda")
        inputs = {
            "image": image,
            "height": height,
            "width": width,
        }
        #output = self.model([inputs])#[0]
        predictions, mask_embed = self.model([inputs])#[0]
        #print(predictions)
        #print(mask_embed.shape)
        return predictions, mask_embed
    
    def run_on_image(self, image):
        vis_output = None
        print("start predicting...")
        time_s = time.time()
        predictions, mask_embed = self.predict(image)
        #print(predictions)
        print(mask_embed.shape)
        time_e = time.time()
        print("end predicting...")
        print("*"*100)
        print("time cost: ", time_e - time_s)
        print("*"*100)
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions[0]:
            panoptic_seg, segments_info = predictions[0]["panoptic_seg"]
            #print(panoptic_seg.shape)
            #print(segments_info)
            vis_output = visualizer.draw_panoptic_seg(
                panoptic_seg.to(self.cpu_device), segments_info
            )
            #print(vis_output)
        else: 
            if "sem_seg" in predictions[0]:
                vis_output = visualizer.draw_sem_seg(
                    predictions[0]["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions[0]:
                instances = predictions[0]["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
        return predictions, mask_embed, vis_output


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


def load_model_and_config(model_config="Panoptic/odise_caption_coco_50e.py", overlap_threshold=0, seed=42):
    """
    Load the ODISE model and configuration.
    
    Args:
        model_config (str): Model configuration name
        overlap_threshold (float): Overlap threshold for predictions
        seed (int): Random seed for reproducibility
    
    Returns:
        Tuple: (model, aug) - the loaded model and augmentations
    """
    cfg = model_zoo.get_config(model_config, trained=True)
    cfg.model.overlap_threshold = overlap_threshold
    seed_all_rng(seed)

    dataset_cfg = cfg.dataloader.test
    aug = instantiate(dataset_cfg.mapper).augmentations
    
    model = instantiate_odise(cfg.model)
    ODISECheckpointer(model).load(cfg.train.init_checkpoint)
    print("Finished loading model")
    
    return model, aug


#def run_inference(input_image, model, aug, demo_metadata, demo_classes, 
#                 output_image_path="output_image.jpg", save_predictions=True, predictions_path="predictions.pt", mask_embed_path="mask_embed.pt"):
#def run_inference(input_image_file, model_config, label_list, vocab):
def run_inference(input_image_file, model, aug, label_list, vocab):
    """
    Run inference on an input image and save results.
    
    Args:
        input_image (PIL.Image or np.ndarray): Input image
        model: Loaded ODISE model
        aug: Augmentations
        demo_metadata: Metadata for visualization
        demo_classes: Class labels
    
    Returns:
        dict: Predictions from the model
    """
    # Convert PIL Image to numpy array if needed
    print("Loading image...")
    input_image = Image.open(input_image_file)

    if isinstance(input_image, Image.Image):
        input_image_array = np.array(input_image)
    else:
        input_image_array = input_image


    #print("Loading model and configuration...")
    #model, aug = load_model_and_config(model_config)

    demo_metadata, demo_classes = setup_class_labels(label_list, vocab)

    print("Running inference...")
    start_time = time.time()
    with ExitStack() as stack:
        inference_model = OpenPanopticInference(
            model=model, 
            labels=demo_classes, 
            metadata=demo_metadata, 
            semantic_on=False, 
            instance_on=False, 
            panoptic_on=True
        )
        inference_model = inference_model.to("cuda")
        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())
        
        demo = VisualizationDemo(inference_model, demo_metadata, aug)
        predictions, mask_embedding, visualized_output = demo.run_on_image(input_image_array)
        #print(visualized_output)
        #print(predictions)
    end_time = time.time()
    print("+++++++++++++++++++++")
    print("Time cost: ", end_time - start_time)
    print("+++++++++++++++++++++")
    return predictions, mask_embedding, visualized_output
