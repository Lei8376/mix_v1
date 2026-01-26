import torch
import os
import yaml
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.util import AverageMeter
#from model.modeling import MaskStablePixle
from dataset import data_loader as dl
from MinkowskiEngine import SparseTensor
import random
from model.criterion import Criteria

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Reproducibility
SEED = 132
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.cuda.empty_cache()

# Create directories
os.makedirs("runs/mask_sd_pix", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

writer = SummaryWriter(log_dir="runs/mask_sd_pix")

# Config
file = "/home/featurize/work/XMask3D/config/scannet/xmask3d_scannet_B10N9.yaml"
label_path = "lang_seg/label_files/ade20k_objectInfo150.txt"
ckpt_path = "lang_seg/checkpoints/demo_e200.ckpt"

res = dl.read_yaml(file)

data_loader = dl.ScannetLoader(
    datapath_prefix=res['DATA']['data_root'],
    datapath_prefix_2d=res['DATA']['data_root_2d'],
    datapath_lseg_feat=res['DATA']['data_root_lseg_feat'],
    datapath_odise_feat=res['DATA']['data_root_odise_feat'],
    category_split=res['DATA']['category_split'],
    label_2d=res['DATA']['label_2d'],
    scannet200=res['DATA']['scannet200'],
)

train_loader = torch.utils.data.DataLoader(
    data_loader,
    batch_size=2,
    shuffle=True,  # Enable shuffle for better training
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    collate_fn=dl.collation_fn,
)

for i, batch_data in enumerate(train_loader):
    (ori_coords_3d, coords_3d, feat_3d, labels_3d, binary_label_3d, 
     binary_label_2d, label_2d, img, x_label, y_label, 
     inds_reconstruct, captions, lseg_feat, masks_odise, masks_feat_odise, masks_info) = batch_data
    print(lseg_feat.shape)
    print(masks_odise.shape)
    print(masks_feat_odise.shape)
    print(masks_info.shape)

"""
# Model setup
odise_model_config_path = "Panoptic/odise_caption_coco_50e.py"
model = MaskStablePixle(label_path, ckpt_path, odise_model_config_path)
model.cuda()
model.train()

# Freeze pretrained feature extractors, train only fusion and 3D parts
for param in model.pix_extractor.parameters():
    param.requires_grad = False
for param in model.mask_extractor.model.parameters():
    param.requires_grad = False

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

# Training hyperparameters
num_epochs = 10
base_lr = 1e-4  # Lower learning rate
weight_decay = 1e-4
grad_clip_norm = 1.0

# Optimizer with weight decay
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=base_lr,
    weight_decay=weight_decay
)

# Learning rate scheduler - cosine annealing with warm restarts
steps_per_epoch = len(train_loader)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=steps_per_epoch,  # Restart every epoch
    T_mult=2,  # Double the period after each restart
    eta_min=1e-6
)

global_step = 0
best_loss = float('inf')

print(f"Starting training for {num_epochs} epochs...")
print(f"Steps per epoch: {steps_per_epoch}")

for epoch in range(num_epochs):
    epoch_loss = AverageMeter()
    model.train()
    
    for i, batch_data in enumerate(train_loader):
        (ori_coords_3d, coords_3d, feat_3d, labels_3d, binary_label_3d, 
         binary_label_2d, label_2d, img, x_label, y_label, 
         inds_reconstruct, captions) = batch_data
        
        sinput = SparseTensor(
            feat_3d.cuda(non_blocking=True), 
            coords_3d.cuda(non_blocking=True)
        )
        batch_input = {
            "sinput": sinput,
            "img": img.cuda(),
            "x_label": x_label.cuda(),
            "y_label": y_label.cuda(),
            "inds_reconstruct": inds_reconstruct.cuda(),
            "captions": captions,
            "binary_label_3d": binary_label_3d.cuda(),
            "binary_label_2d": binary_label_2d.cuda(),
            "label_2d": label_2d.cuda(),
            "ori_coords_3d": ori_coords_3d.cuda(),
        }

        # Forward pass
        result = model(batch_input)
        criteria = Criteria(result, batch_input, bce_weight=1.0, dice_weight=1.0)
        loss = criteria.loss_pt()
        
        # Skip if loss is zero (no valid masks)
        if loss.item() == 0:
            continue
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            grad_clip_norm
        )
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        epoch_loss.update(loss.item())
        writer.add_scalar("Loss/Train_Step", loss.item(), global_step)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
        global_step += 1
        
        if i % 50 == 0:
            lr_current = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{steps_per_epoch}] "
                  f"Loss: {loss.item():.4f} LR: {lr_current:.2e}")
    
    # Epoch summary
    writer.add_scalar("Loss/Train_Epoch", epoch_loss.avg, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss.avg:.4f}")
    
    # Save best model
    if epoch_loss.avg < best_loss:
        best_loss = epoch_loss.avg
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_loss,
        }, "checkpoints/best_model.pth")
        print(f"  -> Saved best model (loss: {best_loss:.4f})")
"""
""" 
    # Save checkpoint every 2 epochs
    if (epoch + 1) % 2 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss.avg,
        }, f"checkpoints/model_epoch_{epoch+1}.pth")
"""

#print(f"Training complete! Best loss: {best_loss:.4f}")
#writer.close()
