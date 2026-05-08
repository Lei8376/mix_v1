#python feat_img.py --img /home/sunl/work/mix/data/scannet_2d/scene0000_00/color/0.jpg --out out/feat.npy
## choose a writable cache root
#export TORCH_HOME=/home/sunl/.cache/torch
#mkdir -p "$TORCH_HOME/hub/checkpoints"

# copy your existing file into the cache
#cp /home/sunl/work/mix_v1/lang_seg/checkpoints/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz \
#   "$TORCH_HOME/hub/checkpoints/"

import os

import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from modules.models.lseg_net import LSegNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

os.environ.setdefault("TORCH_HOME", "/home/sunl/.cache/torch")
os.makedirs("/home/sunl/.cache/torch/checkpoints", exist_ok=True)

def load_ade_labels():
    path = "label_files/ade20k_objectInfo150.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run in repo root.")
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            label = line.strip().split(",")[-1].split(";")[0]
            labels.append(label)
    return labels[1:]  # drop background

def build_net(backbone="clip_vitl16_384", crop_size=480, features=256):
    labels = load_ade_labels()
    net = LSegNet(
        labels=labels,
        backbone=backbone,
        features=features,
        crop_size=crop_size,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # match repo's training wrapper behavior
    net.pretrained.model.patch_embed.img_size = (crop_size, crop_size)
    return net

def load_ckpt_into_net(net, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cuda:0", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # lightning ckpt often prefixes weights with "net."
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("net."):
            new_sd[k[len("net."):]] = v
        else:
            new_sd[k] = v

    missing, unexpected = net.load_state_dict(new_sd, strict=False)
    print(f"[ckpt] loaded. missing={len(missing)} unexpected={len(unexpected)}")

@torch.no_grad()
def export_feat(net, img_path, out_path, crop_size=480, fp16=True, use_autocast=True):
    img = Image.open(img_path).convert("RGB")
    print(img.size)
    x = T.Compose([
        T.Resize((crop_size, crop_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])(img).unsqueeze(0).to(device)

    cache = {}
    def hook_fn(m, inp, out):
        # out: [B, C, H', W']
        cache["feat"] = out

    hhook = net.scratch.head1.register_forward_hook(hook_fn)
    autocast_ctx = torch.cuda.amp.autocast if (use_autocast and device == "cuda") else torch.cpu.amp.autocast
    with autocast_ctx(enabled=True):
        _ = net(x, labelset=["object"])

    hhook.remove()

    feat = cache["feat"]  # [1, C, H', W']
    feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-6)  # L2 norm

    feat = F.interpolate(feat, size=(img.size[0], img.size[1]), mode="bilinear", align_corners=True)
    #if upsample:
        #feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=True)
        #print(img.size)

    feat_np = feat[0].permute(1, 2, 0).cpu().numpy()  # [H', W', C]
    if fp16:
        feat_np = feat_np.astype(np.float16)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, feat_np)
    print(f"[save] {out_path} shape={feat_np.shape} dtype={feat_np.dtype}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/demo_e200.ckpt")
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="out/feat.npy")
    ap.add_argument("--backbone", default="clip_vitl16_384")
    ap.add_argument("--crop", type=int, default=480)
    ap.add_argument("--features", type=int, default=256)
    #ap.add_argument("--upsample", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--use_autocast", action="store_true")
    args = ap.parse_args()

    net = build_net(backbone=args.backbone, crop_size=args.crop, features=args.features)
    load_ckpt_into_net(net, args.ckpt)
    net.eval().to(device)

    export_feat(
        net,
        img_path=args.img,
        out_path=args.out,
        crop_size=args.crop,
        fp16=(not args.fp32),
        use_autocast=args.use_autocast,
    )

if __name__ == "__main__":
    main()
