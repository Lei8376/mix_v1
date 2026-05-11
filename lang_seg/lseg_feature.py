import os
import sys
import torch
import yaml
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# Insert project_root first so top-level packages (utils, dataset) take precedence
# over any same-named modules in lang_seg/
if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# data_loader 仅下方 __main__ 使用，延迟导入避免 train_open_vocab_v2 时拉取 SharedArray
from lang_seg.modules.models.lseg_net import LSegNet

# for LSeg module：使用项目内可写路径，避免写入无权限目录
_default_cache_root = os.path.join(project_root, "checkpoints", "pretrained")
_torch_home = os.environ.get("TORCH_HOME", os.path.join(_default_cache_root, "torch"))
os.environ.setdefault("TORCH_HOME", _torch_home)
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_default_cache_root, "xdg"))
os.makedirs(os.path.join(_torch_home, "checkpoints"), exist_ok=True)

class LSegExtractor(nn.Module):
    def __init__(self, label_path, ckpt_path, backbone = "clip_vitl16_384", crop_size=512, feat_dim = 256):
        super(LSegExtractor, self).__init__()
        self.label_path = label_path
        self.ckpt_path = ckpt_path
        self.backbone = backbone
        self.crop_size = crop_size
        self.feat_dim = feat_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.labels = self.load_labels()
        #self.net = self.build_net() 
        self.net = self.load_ckpt_into_net()
        
    def load_labels(self):
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Missing {self.label_path}. Run in repo root.")
        labels = []
        with open(self.label_path, "r", encoding="utf-8") as f:
            for line in f:
                label = line.strip().split(",")[-1].split(";")[0]
                labels.append(label)
        return labels[1:]  # drop background
    
    def build_net(self):
        net = LSegNet(
            labels=self.labels,
            backbone=self.backbone,
            features=self.feat_dim,
            crop_size=self.crop_size,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )
        net.pretrained.model.patch_embed.img_size = (self.crop_size, self.crop_size)

        #net.pretrained.model.patch_embed.img_size = self.crop_size
        return net
        
    def load_ckpt_into_net(self):
        net = self.build_net()#.eval().to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        new_sd = {}
        for k, v in sd.items():
            if k.startswith("net."):
                new_sd[k[len("net."):]] = v
            else:
                new_sd[k] = v
        missing, unexpected = net.load_state_dict(new_sd, strict=False)
        print(f"[ckpt] loaded. missing={len(missing)} unexpected={len(unexpected)}")
        return net.eval().to(self.device)
        
    def forward(self,img, use_autocast=True, fp16=True):

        net = self.net

        img_transform = T.Compose([
            T.Resize((self.crop_size, self.crop_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu()
            if img_np.dtype != torch.uint8:
                img_np = img_np.clamp(0, 255).byte()
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = img_np.permute(1, 2, 0)
            img_np = img_np.numpy()
            img = Image.fromarray(img_np)
        elif isinstance(img, np.ndarray):
            # Handle numpy array input
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
        img_x = img_transform(img).unsqueeze(0).to(self.device)

        cache = {}
        def hook_fn(m, inp, out):
            # out: [B, C, H', W']
            cache["feat"] = out

        hhook = net.scratch.head1.register_forward_hook(hook_fn)
        autocast_ctx = torch.cuda.amp.autocast if self.device == "cuda" else torch.cpu.amp.autocast
        with autocast_ctx(enabled=True):
            _ = net(img_x, labelset=["object"])
        hhook.remove()

        feat = cache["feat"]
        feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-6)  # L2 norm
        #print(img.size)
        feat = F.interpolate(feat, size=(img.size[0], img.size[1]), mode="bilinear", align_corners=True)
        feat_np = feat[0].permute(1, 2, 0).detach().cpu().numpy()  # [H', W', C]
        if fp16:
            feat_np = feat_np.astype(np.float16)
        return feat_np


"""
if __name__ == "__main__":
    from dataset import data_loader as dl

    label_path = "label_files/ade20k_objectInfo150.txt"
    ckpt_path = "checkpoints/demo_e200.ckpt"

    file = os.path.join(project_root, "config", "data_scannet_3d.yaml")
    res = dl.read_yaml(file)
    data_loader = dl.ScannetLoader(
        datapath_prefix=res['DATA']['data_root'],
        datapath_prefix_2d=res['DATA']['data_root_2d'],
        category_split=res['DATA']['category_split'],
        label_2d=res['DATA']['label_2d'],
    )
    img = data_loader[1][7]
    feat_extractor = LSegExtractor(label_path, ckpt_path)
    feat = feat_extractor(img)
    print(feat.shape)
"""
