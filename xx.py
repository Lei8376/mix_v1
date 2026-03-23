import cv2
import os
import numpy as np
from PIL import Image
import numpy as np

path = "/home/sunl/work/mix/data/precomputed_2d"

scene = "scene0000_00"

odise_res = np.load(os.path.join(path, scene, "0_odise.npz"), allow_pickle=True)

with np.load(os.path.join(path, scene, "0_odise.npz"), allow_pickle=True) as data:
    print(data.files)
    print(data["masks"].shape)
    print(data["mask_embeddings"].shape)
    print(data["info"].shape)
    print(data["info"][0])


