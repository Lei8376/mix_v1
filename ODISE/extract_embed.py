#!/usr/bin/env python
"""
ODISE Inference Script for Extracting Masks and Mask Embeddings from RGB Images

This script outputs exactly N mask embeddings for N detected masks.
Each mask embedding corresponds one-to-one with its detected mask.
"""

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

# Import ScanNet labels
from scannet_label_constant import SCANNET_LABELS_20, SCANNET_COLOR_MAP_20


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

# ScanNet-20 标签集
SCANNET_20_THING_CLASSES = list(SCANNET_LABELS_20)
SCANNET_20_THING_COLORS = [list(x) for x in SCANNET_COLOR_MAP_20.values()]


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
        caption: str = "a photo of",
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
            vocab: Additional vocabulary (comma-separated categories)
            caption: Caption prompt for CaptionODISE (will extract nouns from this)
            overlap_threshold: Overlap threshold for mask merging
            object_mask_threshold: Score threshold for keeping masks
            seed: Random seed
            device: Computation device
        """
        self.device = device
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.caption = caption
        
        if label_sets is None:
            label_sets = ["COCO", "ADE", "LVIS"]
        
        # Load model
        self.model, self.aug, self.cfg = self._load_model(model_config, seed)
        
        # Setup labels
        self.metadata, self.demo_classes = self._setup_class_labels(label_sets, vocab)
        
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
        
        # 如果指定了 caption，使用 NLTK 提取名词
        if self.caption and self.caption != "a photo of":
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                try:
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                except LookupError:
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                try:
                    nltk.data.find('tokenizers/punkt_tab')
                except LookupError:
                    nltk.download('punkt_tab', quiet=True)
                
                # 提取名词
                from nltk import word_tokenize, pos_tag
                tokens = word_tokenize(self.caption)
                tagged = pos_tag(tokens)
                # 提取名词 (NN, NNS, JJ 等)
                nouns = [word for word, tag in tagged if tag.startswith('NN') or tag.startswith('JJ')]
                
                if nouns:
                    print(f"[Caption] 从 '{self.caption}' 提取的名词: {nouns}")
                    extra_classes.append(nouns)
                else:
                    print(f"[Caption] 警告: 无法从 '{self.caption}' 提取名词，跳过")
            except Exception as e:
                print(f"[Caption] 警告: NLTK 提取失败 ({e})，跳过")
        else:
            if self.caption == "a photo of":
                print(f"[Caption] 使用默认 caption，跳过额外类别提取")
        
        if vocab:
            for words in vocab.split(";"):
                extra_classes.append([word.strip() for word in words.split(",")])
        
        extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]
        
        demo_thing_classes = extra_classes
        demo_stuff_classes = []
        demo_thing_colors = extra_colors
        demo_stuff_colors = []
        
        # Add ScanNet-20 support (placed first for priority)
        if "SCANNET_20" in label_sets:
            demo_thing_classes += SCANNET_20_THING_CLASSES
            demo_thing_colors += SCANNET_20_THING_COLORS
            print(f"[INFO] 加载了 {len(SCANNET_20_THING_CLASSES)} 个ScanNet-20类别")
        
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
        
        MetadataCatalog.pop("odise_extractor_metadata", None)
        metadata = MetadataCatalog.get("odise_extractor_metadata")
        
        # Handle mixed format: some are strings (ScanNet), some are lists (COCO/ADE/LVIS)
        thing_class_names = []
        for c in demo_thing_classes:
            if isinstance(c, list):
                thing_class_names.append(c[0])
            else:
                thing_class_names.append(c)
        
        stuff_class_names = []
        for c in demo_stuff_classes:
            if isinstance(c, list):
                stuff_class_names.append(c[0])
            else:
                stuff_class_names.append(c)
        
        metadata.thing_classes = thing_class_names
        # Stuff classes should be only the stuff names (not include things).
        metadata.thing_classes = thing_class_names
        metadata.stuff_classes = stuff_class_names

        metadata.thing_colors = demo_thing_colors
        metadata.stuff_colors = demo_stuff_colors

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
    
    def extract(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
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
        if isinstance(image, str):
            image = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
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
        Uses the full ODISE model forward pass for correct panoptic segmentation.
        """
        from detectron2.structures import ImageList
        from detectron2.modeling.postprocessing import sem_seg_postprocess
        from detectron2.utils.memory import retry_if_cuda_oom

        # Set model to use our labels
        inference_model = OpenPanopticInference(
            model=self.model,
            labels=self.demo_classes,
            metadata=self.metadata,
            semantic_on=False,
            instance_on=False,
            panoptic_on=True
        )
        inference_model = inference_model.to(self.device)
        inference_model.eval()

        # Get base model
        model = inference_model.model

        # Prepare images
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - model.pixel_mean) / model.pixel_std for x in images]

        # Pad images to be divisible by size_divisibility (default 64)
        size_divisibility = model.size_divisibility
        h, w = images[0].shape[-2:]
        pad_h = (size_divisibility - h % size_divisibility) % size_divisibility
        pad_w = (size_divisibility - w % size_divisibility) % size_divisibility
        if pad_h > 0 or pad_w > 0:
            images = [F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0) for img in images]

        images = ImageList.from_tensors(images, size_divisibility)

        # Run standard ODISE forward pass
        features = model.backbone(images.tensor)
        outputs = model.sem_seg_head(features)
        outputs["images"] = images.tensor

        # Process based on model type
        if hasattr(model, 'category_head') and model.category_head is not None:
            # CategoryODISE path
            model.category_head.test_labels = self.demo_classes
            outputs.update(model.category_head(outputs))
            outputs["pred_logits"] = model.cal_pred_logits(outputs)
            
            # CLIP head refinement
            if model.clip_head is not None:
                model.clip_head.test_labels = self.demo_classes
                if model.clip_head.with_bg:
                    outputs["pred_open_logits"] = outputs["pred_logits"]
                    outputs.update(model.clip_head(outputs))
                else:
                    outputs["pred_open_logits"] = outputs["pred_logits"][..., :-1]
                    outputs.update(model.clip_head(outputs))
            else:
                outputs["pred_open_logits"] = outputs["pred_logits"][..., :-1]
            
            mask_embed = outputs["mask_embed"]
            mask_pred_results = outputs["pred_masks"]
            mask_cls_results = outputs.get("pred_open_logits", outputs["pred_logits"][..., :-1])
            
        elif hasattr(model, 'word_head') and model.word_head is not None:
            # CaptionODISE path
            model.word_head.test_labels = self.demo_classes
            outputs.update(model.word_head(outputs))
            
            # CaptionODISE: use cal_pred_open_logits which returns (pred, mask_embed)
            pred_open_logits, mask_embed = model.cal_pred_open_logits(outputs)
            outputs["pred_open_logits"] = pred_open_logits
            
            # Get pred_logits from sem_seg_head (binary classification)
            mask_pred_results = outputs["pred_masks"]
            # # Create binary pred_logits [B, Q, 2] from pred_open_logits [B, Q, K]
            # pred_open = outputs["pred_open_logits"]
            # binary_scores = torch.ones(*pred_open.shape[:-1], 2, device=pred_open.device)
            # binary_scores[..., 0] = pred_open.mean(-1)  # foreground
            # binary_scores[..., 1] = 1 - binary_scores[..., 0]  # background
            # mask_cls_results = binary_scores
            # pred_open_logits: [B, Q, K]
            mask_cls_results = outputs["pred_open_logits"]

            # 追加 background，让它变成 [B, Q, K+1]
            bg = torch.zeros(
                (*mask_cls_results.shape[:-1], 1),
                device=mask_cls_results.device,
                dtype=mask_cls_results.dtype,
            )
            mask_cls_results = torch.cat([mask_cls_results, bg], dim=-1)

        else:
            raise RuntimeError("Unknown model type: neither category_head nor word_head found")

        # Post-process
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        # Get original image size
        input_per_image = batched_inputs[0]
        height = input_per_image.get("height", images.tensor.shape[-2])
        width = input_per_image.get("width", images.tensor.shape[-1])

        # Post-process to original size
        if model.sem_seg_postprocess_before_inference:
            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_results[0], images.image_sizes[0], height, width
            )
            mask_cls_result = mask_cls_results[0].to(mask_pred_result)
        else:
            mask_pred_result = mask_pred_results[0]
            mask_cls_result = mask_cls_results[0]

        # Panoptic inference with embedding tracking
        result = self._panoptic_inference_with_embeddings(
            mask_cls_result,
            mask_pred_result,
            mask_embed[0],  # [Q, C]
        )

        return result
    
    def _panoptic_inference_with_embeddings(
        self,
        mask_cls: torch.Tensor,      # [Q, K+1]
        mask_pred: torch.Tensor,     # [Q, H, W]
        mask_embed: torch.Tensor,    # [Q, C]
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
        
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        
        masks_list = []
        embeddings_list = []
        results_list = []
        
        if cur_masks.shape[0] == 0:
            # No masks detected
            return {
                "masks": [],
                "mask_embeddings": torch.empty(0, mask_embed.shape[-1], device=mask_embed.device),
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
                
                # Store mask and corresponding embedding
                masks_list.append(mask)
                embeddings_list.append(cur_embeddings[k])  # Embedding for this query
                
                results_list.append(MaskResult(
                    mask=mask,
                    mask_embedding=cur_embeddings[k],
                    category_name=category_name,
                    category_id=pred_class,
                    is_thing=is_thing,
                    score=cur_scores[k].item(),
                    area=mask.sum().item(),
                ))
        
        # Stack embeddings
        if embeddings_list:
            mask_embeddings = torch.stack(embeddings_list, dim=0)  # [N, C]
        else:
            mask_embeddings = torch.empty(0, mask_embed.shape[-1], device=mask_embed.device)
        
        return {
            "masks": masks_list,
            "mask_embeddings": mask_embeddings,
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


# 延迟初始化 - 不会在import时执行
extractor = None

def main():
    parser = argparse.ArgumentParser(
        description="Extract masks and mask embeddings from RGB images"
    )
    parser.add_argument("--input", nargs="+", required=True, help="Input image(s)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--model-config", type=str, 
                        default="Panoptic/odise_caption_coco_50e.py")
    parser.add_argument("--labels", nargs="+", default=["COCO", "ADE", "LVIS"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization")
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize extractor
    extractor = ODISEMaskEmbeddingExtractor(
        model_config=args.model_config,
        label_sets=args.labels,
        device=args.device
    )
    
    # Process images
    image_paths = []
    for pattern in args.input:
        expanded = glob.glob(os.path.expanduser(pattern))
        image_paths.extend(expanded if expanded else [pattern])
    
    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")
        
        # Extract
        image = np.array(Image.open(img_path))
        results = extractor.extract(image)
        
        print(f"  Detected {results['num_masks']} masks")
        print(f"  Mask embeddings shape: {results['mask_embeddings'].shape}")
        
        # Verify one-to-one correspondence
        assert len(results['masks']) == results['mask_embeddings'].shape[0], \
            "Mismatch between masks and embeddings!"
        
        # Print per-mask info
        for i, res in enumerate(results['results']):
            print(f"    Mask {i+1}: {res.category_name} (score={res.score:.3f}, area={res.area})")
        
        # Save results
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Save as .pt file with all information
        save_data = {
            "masks": [m.cpu() for m in results['masks']],
            "mask_embeddings": results['mask_embeddings'].cpu(),
            "panoptic_seg": results['panoptic_seg'].cpu(),
            "num_masks": results['num_masks'],
            "mask_info": [
                {
                    "category_name": r.category_name,
                    "category_id": r.category_id,
                    "is_thing": r.is_thing,
                    "score": r.score,
                    "area": r.area,
                }
                for r in results['results']
            ]
        }
        torch.save(save_data, os.path.join(args.output, f"{base_name}_masks_embeddings.pt"))
        
        # Save visualization
        if args.save_vis:
            vis_image = extractor.visualize(image, results)
            Image.fromarray(vis_image).save(
                os.path.join(args.output, f"{base_name}_vis.jpg")
            )
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()