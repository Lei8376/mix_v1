import os
import sys

#del sys.path[5]
#print(sys.path)
import torch
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange

from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule
from encoding.models.sseg import BaseNet
import torchvision.transforms as transforms

import torch
import timm

import os
os.environ["TORCH_HOME"] = "/home/sunl/.cache/torch"
os.environ["XDG_CACHE_HOME"] = "/home/sunl/.cache"
os.makedirs("/home/sunl/.cache/torch/checkpoints", exist_ok=True)

import fusion_util

# This will download and cache the model
#model = timm.create_model("vit_large_patch16_384", pretrained=True)
#print("Model downloaded successfully")

#load Lseg model
module = LSegModule.load_from_checkpoint(
    checkpoint_path='checkpoints/demo_e200.ckpt',
    data_path='dataset/',
    dataset='ade20k',
    backbone='clip_vitl16_384',
    aux=False,
    num_features=256,
    aux_weight=0,
    se_loss=False,
    se_weight=0,
    base_lr=0,
    batch_size=1,
    max_epochs=0,
    ignore_index=255,
    dropout=0.0,
    scale_inv=False,
    augment=False,
    no_batchnorm=False,
    widehead=True,
    widehead_hr=False,
    map_locatin="cpu",
    arch_option=0,
    block_depth=0,
    activation='lrelu',
)

def load_ade_labels():
    path = "/home/sunl/work/mix/lang_seg/label_files/ade20k_objectInfo150.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run in repo root.")
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            label = line.strip().split(",")[-1].split(";")[0]
            labels.append(label)
    return labels[1:]  # drop background

labels = load_ade_labels()

if isinstance(module.net, BaseNet):
    model = module.net
else:
    model = module

model = model.eval()
model = model.cpu()

model.mean = [0.5,0.5,0.5]
model.std = [0.5,0.5,0.5]

scales = ([1])
model.crop_size = 640
model.base_size = 640

evaluator = LSeg_MultiEvalModule(model, scales=scales, flip=True).cuda()
evaluator.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ]
)


img_dir = "/home/sunl/work/mix/data/scannet_2d/scene0000_00/color/0.jpg"
feat_2d = fusion_util.extract_lseg_img_feature(img_dir, transform, evaluator, labels)

print(feat_2d.shape)