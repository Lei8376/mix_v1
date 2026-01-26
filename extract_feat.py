import os
import sys
import torch
import yaml
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from dataset import data_loader as dl
from lang_seg import lseg_feature as lf

from ODISE import odise_feature as of

import time

#for LSeg module
os.environ.setdefault("TORCH_HOME", "/home/featurize/.cache/torch")
os.makedirs("/home/featurize/.cache/torch/checkpoints", exist_ok=True)


if __name__ == "__main__":

    file = "/home/featurize/work/XMask3D/config/scannet/xmask3d_scannet_B10N9.yaml"
    
    label_path = "lang_seg/label_files/ade20k_objectInfo150.txt"
    ckpt_path = "lang_seg/checkpoints/demo_e200.ckpt"

    res = dl.read_yaml(file)
    data_loader = dl.ScannetLoader(
        datapath_prefix=res['DATA']['data_root'],
        datapath_prefix_2d=res['DATA']['data_root_2d'],
        category_split=res['DATA']['category_split'],
        label_2d=res['DATA']['label_2d'],
    )
    img = data_loader[5][7]
    
    #for lseg model
    feat_extractor = lf.LSegExtractor(label_path, ckpt_path)
    start = time.time()
    feat = feat_extractor(img)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print(feat.shape)

    #here for odise model
    model_config_path = "Panoptic/odise_caption_coco_50e.py"
    extractor = of.ODISEMaskEmbeddingExtractor(
        model_config = model_config_path,
        label_sets = ["COCO", "ADE","LVIS","SCANNET_20"],
        device = "cuda"
    )
    results = extractor.extract(img)
    print(len(results["masks"]))
    for i in range(len(results["masks"])):
        mask = results["masks"][i]
        embedding = results["mask_embeddings"][i]
        print(f"Mask {i+1}: {mask.shape}, Embedding: {embedding.shape}")



