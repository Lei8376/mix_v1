
import argparse
import glob
import itertools
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

from odise import model_zoo
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.modeling.wrapper import OpenPanopticInference

#load scannet labels
from scannet_label_constant import SCANNET_LABELS_20, SCANNET_COLOR_MAP_20, SCANNET_LABELS_200, SCANNET_COLOR_MAP_200


# ============================================================================
# Label Constants
# ============================================================================
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


@dataclass
class MaskResult:
    """Result for a single detected mask."""
    mask: torch.Tensor           # Binary mask [H, W]
    mask_embedding: torch.Tensor # CLIP-aligned embedding [C]
    category_name: str           # Predicted category name
    category_id: int             # Category ID
    is_thing: bool               # Whether it's a thing or stuff
    score: float                 # Confidence score
    area: int                    # Mask area in pixels


class ODISEMaskEmbeddingExtractor:
    """
    Extract masks and their corresponding embeddings from ODISE model.
    
    Guarantees: If N masks are detected, exactly N mask embeddings are returned,
    with one-to-one correspondence.
    """
    
    def __init__(
        self,
        model_config: str = "Panoptic/odise_caption_coco_50e.py",
        label_sets: List[str] = None,
        vocab: str = "",
        overlap_threshold: float = 0,
        object_mask_threshold: float = 0.0,
        seed: int = 42,
        device: str = "cuda"
    ):
        """
        Initialize the extractor.
        
        Args:
            model_config: Model config name
            label_sets: Label sets to use (default: ["COCO", "ADE", "LVIS"])
            vocab: Additional vocabulary
            overlap_threshold: Overlap threshold for mask merging
            object_mask_threshold: Score threshold for keeping masks
            seed: Random seed
            device: Computation device
        """
        self.device = device
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        
        if label_sets is None:
            label_sets = ["COCO", "ADE", "LVIS", "SCANNET_20"]
        
        # Load model
        self.model, self.aug, self.cfg = self._load_model(model_config, seed)
        
        # Setup labels
        self.metadata, self.demo_classes = self._setup_class_labels(label_sets, vocab)
        self.inference_model = self._build_inference_model()

    def _build_inference_model(self):
        """Build and cache panoptic inference wrapper."""
        inference_model = OpenPanopticInference(
            model=self.model,
            labels=self.demo_classes,
            metadata=self.metadata,
            semantic_on=False,
            instance_on=False,
            panoptic_on=True,
        )
        inference_model = inference_model.to(self.device)
        inference_model.eval()
        return inference_model
        
    def _load_model(self, model_config: str, seed: int):
        """Load ODISE model."""
        cfg = model_zoo.get_config(model_config, trained=True)
        cfg.model.overlap_threshold = self.overlap_threshold
        seed_all_rng(seed)
        
        dataset_cfg = cfg.dataloader.test
        aug = instantiate(dataset_cfg.mapper).augmentations
        
        model = instantiate_odise(cfg.model)
        ODISECheckpointer(model).load(cfg.train.init_checkpoint)
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded: {model_config}")
        return model, aug, cfg
    
    def _setup_class_labels(self, label_sets: List[str], vocab: str = ""):
        """Setup class labels and metadata."""
        extra_classes = []
        if vocab:
            for words in vocab.split(";"):
                extra_classes.append([word.strip() for word in words.split(",")])
        extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]
        
        demo_thing_classes = extra_classes
        demo_stuff_classes = []
        demo_thing_colors = extra_colors
        demo_stuff_colors = []
        
        if "COCO" in label_sets:
            demo_thing_classes += COCO_THING_CLASSES
            demo_stuff_classes += COCO_STUFF_CLASSES
            demo_thing_colors += COCO_THING_COLORS
            demo_stuff_colors += COCO_STUFF_COLORS
        if "ADE" in label_sets:
            demo_thing_classes += ADE_THING_CLASSES
            demo_stuff_classes += ADE_STUFF_CLASSES
            demo_thing_colors += ADE_THING_COLORS
            demo_stuff_colors += ADE_STUFF_COLORS
        if "LVIS" in label_sets:
            demo_thing_classes += LVIS_CLASSES
            demo_thing_colors += LVIS_COLORS
        if "SCANNET_20" in label_sets:
            demo_thing_classes += SCANNET_20_THINGS_CLASSES
            demo_thing_colors += SCANNET_20_THINGS_COLOR
        MetadataCatalog.pop("odise_extractor_metadata", None)
        metadata = MetadataCatalog.get("odise_extractor_metadata")
        metadata.thing_classes = [c[0] for c in demo_thing_classes]
        metadata.stuff_classes = [
            *metadata.thing_classes,
            *[c[0] for c in demo_stuff_classes],
        ]
        metadata.thing_colors = demo_thing_colors
        metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
        metadata.thing_dataset_id_to_contiguous_id = {
            idx: idx for idx in range(len(metadata.thing_classes))
        }
        metadata.stuff_dataset_id_to_contiguous_id = {
            idx: idx for idx in range(len(metadata.stuff_classes))
        }
        
        demo_classes = demo_thing_classes + demo_stuff_classes
        return metadata, demo_classes
    
    def _get_category_name(self, category_id: int, is_thing: bool) -> str:
        """Get category name from ID."""
        if is_thing:
            contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id.get(category_id)
            if contiguous_id is not None:
                return self.metadata.thing_classes[contiguous_id]
            return f"unknown_thing_{category_id}"
        else:
            contiguous_id = self.metadata.stuff_dataset_id_to_contiguous_id.get(category_id)
            if contiguous_id is not None:
                return self.metadata.stuff_classes[contiguous_id]
            return f"unknown_stuff_{category_id}"
    
    #def extract(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
    def extract(self, image: torch.Tensor) -> Dict:
        """
        Extract masks and mask embeddings from an image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Dict containing:
                - masks: List[torch.Tensor] - N binary masks, each [H, W]
                - mask_embeddings: torch.Tensor - [N, C] embeddings
                - results: List[MaskResult] - Detailed results per mask
                - panoptic_seg: torch.Tensor - Full panoptic segmentation [H, W]
                - num_masks: int - Number of detected masks
        """
        # Load image
        #if isinstance(image, str):
        #    image = np.array(Image.open(image))
        #elif isinstance(image, Image.Image):
        #    image = np.array(image)
        #print(image.shape)
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        else:
            image = np.array(image)
        height, width = image.shape[:2]
        
        # Preprocess
        aug_input = T.AugInput(image, sem_seg=None)
        self.aug(aug_input)
        processed_image = aug_input.image
        processed_image = torch.as_tensor(
            processed_image.astype("float32").transpose(2, 0, 1)
        ).to(self.device)
        
        inputs = {"image": processed_image, "height": height, "width": width}
        
        # Run model forward pass with custom panoptic inference
        with torch.no_grad():
            results = self._forward_with_embeddings([inputs])
        
        return results
    
    def _forward_with_embeddings(self, batched_inputs: List[Dict]) -> Dict:
        """
        Custom forward pass that returns mask embeddings aligned with detected segments.
        """
        from detectron2.structures import ImageList
        from detectron2.modeling.postprocessing import sem_seg_postprocess
        from detectron2.utils.memory import retry_if_cuda_oom
        
        inference_model = self.inference_model
        
        # Get base model
        model = inference_model.model
        
        # Prepare images
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - model.pixel_mean) / model.pixel_std for x in images]
        images = ImageList.from_tensors(images, model.size_divisibility)
        
        denormalized_images = ImageList.from_tensors(
            [x["image"].to(self.device) / 255.0 for x in batched_inputs]
        )
        
        # Forward through backbone and head
        features = model.backbone(images.tensor)
        outputs = model.sem_seg_head(features)
        outputs["images"] = denormalized_images.tensor
        
        # Get mask embeddings from word_head
        if model.word_head is not None:
            # Set test labels
            model.word_head.test_labels = self.demo_classes
            outputs.update(model.word_head(outputs))
        
        # Calculate open logits and get mask embeddings
        # mask_embed shape: [B, Q, C] where Q=100 queries, C=embedding dim
        pred_open_logits, mask_embed = model.cal_pred_open_logits(outputs)
        outputs["pred_open_logits"] = pred_open_logits
        
        # Get CLIP head refinement
        if model.clip_head is not None:
            model.clip_head.test_labels = self.demo_classes
            outputs.update(model.clip_head(outputs))
        
        # Process classification results
        mask_cls_results = outputs["pred_logits"]  # [B, Q, 2] binary classification
        mask_pred_results = outputs["pred_masks"]  # [B, Q, H, W]
        
        open_logits = outputs["pred_open_logits"]
        binary_probs = F.softmax(mask_cls_results, dim=-1)
        masks_class_probs = F.softmax(open_logits, dim=-1)
        
        # Combine probabilities
        mask_cls_results = torch.cat(
            [masks_class_probs * binary_probs[..., 0:1], binary_probs[..., 1:2]], dim=-1
        )
        mask_cls_results = torch.log(mask_cls_results + 1e-8)
        
        # Upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        
        # Process each image (typically batch size = 1)
        all_results = []
        for batch_idx, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(
            zip(mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes)
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            
            # Post-process masks to original size
            if model.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)
            
            # Custom panoptic inference with embedding tracking
            result = self._panoptic_inference_with_embeddings(
                mask_cls_result,
                mask_pred_result,
                mask_embed[batch_idx],  # [Q, C]
                open_logits[batch_idx],  # [Q, K] raw open vocab logits
            )
            all_results.append(result)
        
        # Return first result (single image)
        return all_results[0]
    
    def _panoptic_inference_with_embeddings(
        self,
        mask_cls: torch.Tensor,      # [Q, K+1]
        mask_pred: torch.Tensor,     # [Q, H, W]
        mask_embed: torch.Tensor,    # [Q, C]
        open_logits: torch.Tensor,   # [Q, K] raw open vocab logits
    ) -> Dict:
        """
        Panoptic inference that tracks which query indices are used for each segment.
        
        Returns mask embeddings that correspond exactly to detected masks.
        """
        num_classes = len(self.demo_classes)
        
        # Get scores and labels
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)  # [Q], [Q]
        mask_pred = mask_pred.sigmoid()  # [Q, H, W]
        
        # Filter by score and non-background
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        keep_indices = torch.where(keep)[0]  # Original query indices that are kept
        
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_embeddings = mask_embed[keep]  # [num_kept, C]
        cur_logits = open_logits[keep]     # [num_kept, K]
        
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        
        masks_list = []
        embeddings_list = []
        logits_list = []
        results_list = []
        
        if cur_masks.shape[0] == 0:
            # No masks detected
            return {
                "masks": [],
                "mask_embeddings": torch.empty(0, mask_embed.shape[-1], device=mask_embed.device),
                "mask_logits": torch.empty(0, open_logits.shape[-1], device=open_logits.device),
                "results": [],
                "panoptic_seg": panoptic_seg,
                "num_masks": 0,
            }
        
        # Compute weighted masks for argmax
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        cur_mask_ids = cur_prob_masks.argmax(0)  # [H, W]
        
        stuff_memory_list = {}
        current_segment_id = 0
        
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            is_thing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
            
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
            
            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < self.overlap_threshold:
                    continue
                
                # Handle stuff merging
                if not is_thing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1
                
                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id
                
                # Get category name
                category_name = self._get_category_name(pred_class, is_thing)
                
                # Store mask, embedding, and logits
                masks_list.append(mask)
                embeddings_list.append(cur_embeddings[k])  # Embedding for this query
                logits_list.append(cur_logits[k])          # Logits for this query
                
                results_list.append(MaskResult(
                    mask=mask,
                    mask_embedding=cur_embeddings[k],
                    category_name=category_name,
                    category_id=pred_class,
                    is_thing=is_thing,
                    score=cur_scores[k].item(),
                    area=mask.sum().item(),
                ))
        
        # Stack embeddings and logits
        if embeddings_list:
            mask_embeddings = torch.stack(embeddings_list, dim=0)  # [N, C]
            mask_logits = torch.stack(logits_list, dim=0)          # [N, K]
        else:
            mask_embeddings = torch.empty(0, mask_embed.shape[-1], device=mask_embed.device)
            mask_logits = torch.empty(0, open_logits.shape[-1], device=open_logits.device)
        
        return {
            "masks": masks_list,#mask_num, H,W
            "mask_embeddings": mask_embeddings,#mask_num, C-256dim
            "mask_logits": mask_logits,
            "results": results_list,
            "panoptic_seg": panoptic_seg,
            "num_masks": len(masks_list),
        }
    
    def visualize(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Visualize results on image."""
        visualizer = Visualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)
        
        # Build segments_info for visualization
        segments_info = []
        for i, res in enumerate(results["results"]):
            segments_info.append({
                "id": i + 1,
                "isthing": res.is_thing,
                "category_id": res.category_id,
            })
        
        vis_output = visualizer.draw_panoptic_seg(
            results["panoptic_seg"].cpu(), segments_info
        )
        return vis_output.get_image()

"""
extractor = ODISEMaskEmbeddingExtractor(
    model_config = "Panoptic/odise_caption_coco_50e.py",
    label_sets = ["COCO", "ADE","LVIS","SCANNET_20"],
    device = "cuda"
)

results = extractor.extract("/home/featurize/data/scannet_2d/scene0000_00/color/100.jpg")
print(results["num_masks"])
print(len(results["masks"]))
print(results["mask_embeddings"].shape)

image = np.array(Image.open("/home/featurize/data/scannet_2d/scene0000_00/color/100.jpg"))
image = extractor.visualize(image, results)
Image.fromarray(image).save("visualization.jpg")

for i in range(results["num_masks"]):
    mask = results["masks"][i]
    embedding = results["mask_embeddings"][i]
    print(f"Mask {i+1}: {mask.shape}, Embedding: {embedding.shape}")

"""