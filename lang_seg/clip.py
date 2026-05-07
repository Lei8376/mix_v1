"""
Shim module: redirects `import clip` to open_clip equivalents.
open-clip-torch is already installed in the mix conda env.

open_clip uses dash-separated names (e.g. 'ViT-B-32') while openai/CLIP
uses slash-separated names (e.g. 'ViT-B/32').  We normalise here.
"""
from open_clip import tokenize, load_openai_model as _load_openai_model
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

# Map openai/CLIP names → open_clip names
_NAME_MAP = {
    "ViT-B/32":    "ViT-B-32",
    "ViT-B/16":    "ViT-B-16",
    "ViT-L/14":    "ViT-L-14",
    "ViT-L/14@336px": "ViT-L-14-336",
    "RN50":        "RN50",
    "RN50x4":      "RN50x4",
    "RN50x16":     "RN50x16",
    "RN50x64":     "RN50x64",
    "RN101":       "RN101",
}


def load(name, device="cpu", jit=True, cache_dir=None):
    """Drop-in replacement for openai/CLIP clip.load().
    Always loads on CPU first to avoid OOM during model construction.
    The caller (LSegNet) can move the model to GPU later via model.to(device).
    """
    mapped = _NAME_MAP.get(name, name)
    model = _load_openai_model(mapped, device="cpu", jit=False)
    return model, None
