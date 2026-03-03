"""
Open Vocabulary Dataset V2: 支持「仅 npz（含 pixel_pooled）」的预计算格式。

预计算目录下每张图只有一个 *_odise.npz，内含:
  - masks (K, H, W) bool
  - mask_embeddings (K, 256)
  - pixel_pooled (K, 512)   # LSeg 按 mask 池化后的向量
  - num_masks, info

无需单独的 *_lseg.npy。npz 用 np.load(npz_path, allow_pickle=True) 即可读取。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from glob import glob

import numpy as np
import torch
import yaml
from PIL import Image


@dataclass
class OpenVocabDatasetV2Config:
    data_config_path: str
    precomputed_dir: Optional[str] = None
    projection_dir: Optional[str] = None  # 方案 B: 预计算投影目录
    split: str = "train"
    scannet200: bool = False
    voxel_size: float = 0.05
    aug: bool = False
    memcache_init: bool = False
    identifier: int = 7791
    loop: int = 1
    eval_all: bool = False
    input_color: bool = False
    max_samples: Optional[int] = None  # 若 >0 则只用前 max_samples 个样本；0/None 不按数量截断
    max_samples_ratio: Optional[float] = None  # 若 (0,1) 则只用前 ratio 比例，如 0.25=1/4；单卡测试可设 0.25


def _load_yaml(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _cached_load_pth(pth_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """加载 .pth 文件并返回 (locs, feats, labels)。"""
    data = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(data, (list, tuple)):
        locs, feats, labels = data[0], data[1], data[2]
    else:
        locs = data.get("locs", data.get("coords"))
        feats = data.get("feats", data.get("feat"))
        labels = data.get("labels")
    
    if isinstance(locs, np.ndarray):
        locs = torch.from_numpy(locs)
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    return locs, feats, labels


def _list_samples_from_precomputed(precomputed_dir: Path, data_root: Optional[Path] = None, split: str = "train") -> List[Tuple[str, str]]:
    """
    列出 (scene_name, frame_stem)，例如 ('scene0000_00', '0')。
    
    如果提供 data_root，则只列出同时有 3D 数据（.pth）的场景。
    """
    samples = []
    for scene_dir in sorted(precomputed_dir.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene"):
            continue
        
        scene_name = scene_dir.name
        
        # 如果提供了 data_root，检查是否有对应的 3D 数据
        if data_root is not None and data_root.exists():
            pth_path = data_root / split / f"{scene_name}.pth"
            if not pth_path.exists():
                pth_path = data_root / split / f"{scene_name}_vh_clean_2.pth"
            if not pth_path.exists():
                # 跳过没有 3D 数据的场景，避免使用 placeholder
                continue
        
        for npz_path in sorted(scene_dir.glob("*_odise.npz")):
            stem = npz_path.stem.replace("_odise", "")
            samples.append((scene_dir.name, stem))
    return samples


def _load_npz_pooled(npz_path: Path) -> Dict[str, torch.Tensor]:
    """从 npz 读取 pixel_pooled、masks、mask_embeddings、mask_valid。"""
    with np.load(npz_path, allow_pickle=True) as f:
        keys = list(f.files)
        if "pixel_pooled" not in keys:
            raise KeyError(f"npz 缺少 pixel_pooled: {npz_path}")

        masks = f["masks"]
        if masks.dtype == object:
            masks = np.stack(masks, axis=0)
        mask_embeddings = np.asarray(f["mask_embeddings"], dtype=np.float32)
        pixel_pooled = np.asarray(f["pixel_pooled"], dtype=np.float32)

        K = masks.shape[0]
        mask_valid = np.ones(K, dtype=bool)

    return {
        "pixel_pooled": torch.from_numpy(pixel_pooled),
        "masks": torch.from_numpy(masks),
        "mask_embeddings": torch.from_numpy(mask_embeddings),
        "mask_valid": torch.from_numpy(mask_valid),
    }


def _load_3d_with_precomputed_projection(
    data_root: Path, 
    split: str, 
    scene_name: str,
    projection_dir: Path,
    frame_stem: str,
    pth_cache: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    voxel_size: float = 0.05,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    方案 B: 从预计算的投影文件加载 3D 数据和投影坐标。
    
    投影文件由 precompute_projections.py 生成，保证和 npz 是同一帧，
    修复了之前帧不匹配的 bug。
    """
    # 1. 检查投影文件是否存在
    proj_path = projection_dir / scene_name / f"{frame_stem}_proj.npz"
    if not proj_path.exists():
        return None  # fallback 到运行时投影

    # 2. 加载 3D 数据（优先从缓存读取）
    if pth_cache is not None and scene_name in pth_cache:
        locs, feats, labels = pth_cache[scene_name]
    else:
        pth_path = data_root / split / f"{scene_name}.pth"
        if not pth_path.exists():
            pth_path = data_root / split / f"{scene_name}_vh_clean_2.pth"
        if not pth_path.exists():
            return None
        locs, feats, labels = _cached_load_pth(pth_path)

    # 3. 加载预计算投影
    proj_data = np.load(proj_path)
    visible_mask = proj_data["visible_mask"]  # (N,) bool
    y_label = torch.from_numpy(proj_data["y_label"].astype(np.int64))  # (N_vis,) 
    x_label = torch.from_numpy(proj_data["x_label"].astype(np.int64))  # (N_vis,)

    # 校验点数一致
    if proj_data["num_points"] != locs.shape[0]:
        print(f"Warning: point count mismatch for {scene_name}/{frame_stem}: "
              f"proj={proj_data['num_points']}, pth={locs.shape[0]}")
        return None

    # 4. 过滤到可见点
    locs_filtered = locs[visible_mask]
    feats_filtered = feats[visible_mask]
    labels_filtered = labels[visible_mask]

    N = locs_filtered.shape[0]
    batch_idx = torch.zeros(N, 1, dtype=torch.long)
    # 关键：浮点坐标必须先除以 voxel_size 再取整，才能正确量化为体素坐标
    # 之前直接用 locs_filtered.long() 导致坐标坍缩（如 [1.46, 1.93, 0.15] → [1,1,0]）
    # 修复后：[1.46, 1.93, 0.15] / 0.05 → [29, 38, 3]，体素数从 ~12 恢复到正常 ~数千
    locs_quantized = torch.floor(locs_filtered / voxel_size).long()
    coords_3d = torch.cat([batch_idx, locs_quantized], dim=1)
    feat_3d = feats_filtered.float() if feats_filtered.dim() > 1 else feats_filtered.unsqueeze(1).expand(N, 3)

    return {
        "coords_3d": coords_3d,
        "feat_3d": feat_3d,
        "ori_coords_3d": coords_3d.clone(),
        "inds_reconstruct": torch.arange(N, dtype=torch.long),
        "x_label": x_label,
        "y_label": y_label,
        "binary_label_3d": labels_filtered.long(),
        "binary_label_2d": torch.zeros(N, dtype=torch.long),
        "label_2d": torch.zeros(N, dtype=torch.long),
    }


def _load_3d_with_projection(
    data_root: Path, 
    split: str, 
    scene_name: str,
    data_root_2d: Optional[Path] = None,
    point2img_mapper = None,
    min_visible: int = 400,
    max_visible: int = 65000,
    voxel_size: float = 0.05,  # 🔥 新增参数
) -> Dict[str, torch.Tensor]:
    """
    加载 3D 数据并计算运行时投影，参考 OpenScene 的 data_loader.py 逻辑：
    1. 加载 3D 点云 (locs, feats, labels)
    2. 循环遍历 2D 帧，找到可见点数量在 [min_visible, max_visible] 的帧
    3. 计算投影，只保留可见点 (mask == 1)
    4. 返回过滤后的数据
    
    参考: mix/dataset/data_loader.py 第 185-238 行
    """
    # 1. 加载 3D 数据
    pth_path = data_root / split / f"{scene_name}.pth"
    if not pth_path.exists():
        pth_path = data_root / split / f"{scene_name}_vh_clean_2.pth"
    if not pth_path.exists():
        raise FileNotFoundError(f"3D data not found: {pth_path}")

    data = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(data, (list, tuple)):
        locs, feats, labels = data[0], data[1], data[2]
    else:
        locs = data.get("locs", data.get("coords"))
        feats = data.get("feats", data.get("feat"))
        labels = data.get("labels")

    if isinstance(locs, np.ndarray):
        locs_np = locs
        locs = torch.from_numpy(locs)
    else:
        locs_np = locs.numpy()
        
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # 2. 如果没有 2D 数据或 mapper，返回所有点（但没有有效投影）
    if data_root_2d is None or point2img_mapper is None:
        N = locs.shape[0]
        batch_idx = torch.zeros(N, 1, dtype=torch.long)
        # 🔥 修复: 使用正确的量化方式
        locs_quantized = torch.floor(locs / voxel_size).long()
        coords_3d = torch.cat([batch_idx, locs_quantized], dim=1)
        feat_3d = feats.float() if feats.dim() > 1 else feats.unsqueeze(1).expand(N, 3)
        
        return {
            "coords_3d": coords_3d,
            "feat_3d": feat_3d,
            "ori_coords_3d": coords_3d.clone(),
            "inds_reconstruct": torch.arange(N, dtype=torch.long),
            "x_label": torch.zeros(N, dtype=torch.long),
            "y_label": torch.zeros(N, dtype=torch.long),
            "binary_label_3d": labels.long(),
            "binary_label_2d": torch.zeros(N, dtype=torch.long),
            "label_2d": torch.zeros(N, dtype=torch.long),
        }

    # 3. 找到 2D 图像目录
    scene_2d_dir = data_root_2d / scene_name
    if not scene_2d_dir.exists():
        raise FileNotFoundError(f"2D data not found: {scene_2d_dir}")
    
    img_dirs = sorted(glob(str(scene_2d_dir / "color" / "*.jpg")))
    if len(img_dirs) == 0:
        raise FileNotFoundError(f"No color images found in {scene_2d_dir / 'color'}")

    # 4. 循环找到合适的帧（参考 data_loader.py 第 185-238 行）
    img_idx = 0
    max_tries = min(len(img_dirs), 50)  # 最多尝试 50 帧
    
    while img_idx < max_tries:
        img_dir = img_dirs[img_idx]
        
        # 加载 pose 和 depth
        pose_path = img_dir.replace("color", "pose").replace(".jpg", ".txt")
        depth_path = img_dir.replace("color", "depth").replace(".jpg", ".png")
        
        if not os.path.exists(pose_path) or not os.path.exists(depth_path):
            img_idx += 1
            continue
        
        pose = np.loadtxt(pose_path)
        depth = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0  # mm to meter
        
        # 计算投影 (返回 [y, x, valid])
        single_mapping = point2img_mapper.compute_mapping(pose, locs_np, depth)
        
        mask = single_mapping[:, 2]  # valid mask
        num_visible = np.sum(mask == 1)
        
        # 检查可见点数量是否合适
        if min_visible <= num_visible <= max_visible:
            # 找到合适的帧！
            # 只保留可见点（使用 valid mask，而不是"所有列非零"）
            # 🔥 修复 Bug D: 之前用 np.all(single_mapping != 0, axis=1) 会错误过滤 x=0 或 y=0 的有效点
            mask_bool = mask == 1
            
            locs_filtered = locs[mask_bool]
            feats_filtered = feats[mask_bool]
            labels_filtered = labels[mask_bool]
            
            # 提取投影坐标（注意：mapping 是 [y, x, valid]）
            # 🔥 修复：使用与 locs_filtered 相同的 mask，确保长度一致
            single_mapping_filtered = single_mapping[mask_bool]
            
            y_label = torch.from_numpy(single_mapping_filtered[:, 0].copy()).long()  # [:, 0] 是 y
            x_label = torch.from_numpy(single_mapping_filtered[:, 1].copy()).long()  # [:, 1] 是 x
            
            # 构建返回数据
            N = locs_filtered.shape[0]
            batch_idx = torch.zeros(N, 1, dtype=torch.long)
            # 🔥 修复：使用正确的量化方式（和预计算投影一致）
            locs_quantized = torch.floor(locs_filtered / voxel_size).long()
            coords_3d = torch.cat([batch_idx, locs_quantized], dim=1)
            feat_3d = feats_filtered.float() if feats_filtered.dim() > 1 else feats_filtered.unsqueeze(1).expand(N, 3)
            
            return {
                "coords_3d": coords_3d,
                "feat_3d": feat_3d,
                "ori_coords_3d": coords_3d.clone(),
                "inds_reconstruct": torch.arange(N, dtype=torch.long),
                "x_label": x_label,
                "y_label": y_label,
                "binary_label_3d": labels_filtered.long(),
                "binary_label_2d": torch.zeros(N, dtype=torch.long),
                "label_2d": torch.zeros(N, dtype=torch.long),
                "img_idx": img_idx,  # 记录使用的帧索引
            }
        
        img_idx += 1
    
    # 如果没找到合适的帧，使用第一帧并警告
    print(f"Warning: No suitable frame found for {scene_name}, using first frame with {num_visible} visible points")
    img_dir = img_dirs[0]
    pose = np.loadtxt(img_dir.replace("color", "pose").replace(".jpg", ".txt"))
    depth = np.array(Image.open(img_dir.replace("color", "depth").replace(".jpg", ".png")), dtype=np.float32) / 1000.0
    
    single_mapping = point2img_mapper.compute_mapping(pose, locs_np, depth)
    mask_bool = single_mapping[:, 2] == 1
    
    locs_filtered = locs[mask_bool]
    feats_filtered = feats[mask_bool]
    labels_filtered = labels[mask_bool]
    
    # 🔥 修复 Bug D: 使用与 locs_filtered 相同的 mask，确保长度一致
    single_mapping_filtered = single_mapping[mask_bool]
    
    y_label = torch.from_numpy(single_mapping_filtered[:, 0].copy()).long()
    x_label = torch.from_numpy(single_mapping_filtered[:, 1].copy()).long()
    
    N = locs_filtered.shape[0]
    batch_idx = torch.zeros(N, 1, dtype=torch.long)
    # 🔥 修复: 使用正确的量化方式
    locs_quantized = torch.floor(locs_filtered / voxel_size).long()
    coords_3d = torch.cat([batch_idx, locs_quantized], dim=1)
    feat_3d = feats_filtered.float() if feats_filtered.dim() > 1 else feats_filtered.unsqueeze(1).expand(N, 3)
    
    return {
        "coords_3d": coords_3d,
        "feat_3d": feat_3d,
        "ori_coords_3d": coords_3d.clone(),
        "inds_reconstruct": torch.arange(N, dtype=torch.long),
        "x_label": x_label,
        "y_label": y_label,
        "binary_label_3d": labels_filtered.long(),
        "binary_label_2d": torch.zeros(N, dtype=torch.long),
        "label_2d": torch.zeros(N, dtype=torch.long),
        "img_idx": 0,
    }


class OpenVocabScannetDatasetV2(torch.utils.data.Dataset):
    """从「仅 npz（含 pixel_pooled）」的 precomputed_dir 读取；3D 从 data_config 的 data_root/split 读取。"""

    def __init__(self, config: OpenVocabDatasetV2Config):
        self.config = config
        self.precomputed_dir = Path(config.precomputed_dir) if config.precomputed_dir else None
        if not self.precomputed_dir or not self.precomputed_dir.exists():
            raise FileNotFoundError(f"precomputed_dir 不存在: {config.precomputed_dir}")

        # 先获取 data_root，以便过滤只有 3D 数据的场景
        data_cfg = _load_yaml(config.data_config_path)
        data_root = data_cfg.get("DATA", {}).get("data_root", "")
        self.data_root = Path(data_root) if data_root else None
        
        # 获取 2D 数据路径（用于运行时投影）
        data_root_2d = data_cfg.get("DATA", {}).get("data_root_2d", "")
        self.data_root_2d = Path(data_root_2d) if data_root_2d else None
        
        self.split = config.split

        # 方案 B: 预计算投影目录
        self.projection_dir = None
        if getattr(config, "projection_dir", None):
            self.projection_dir = Path(config.projection_dir)
            if self.projection_dir.exists():
                print(f"✅ 使用预计算投影: {self.projection_dir}")
            else:
                print(f"⚠️  projection_dir 不存在: {self.projection_dir}，将使用运行时投影")
                self.projection_dir = None

        # 初始化投影器（仅在没有预计算投影时需要）
        self.point2img_mapper = None
        if self.projection_dir is None:
            if self.data_root_2d and self.data_root_2d.exists():
                try:
                    from utils.mapping_util import get_point2img_mapper
                except ImportError:
                    from mix.utils.mapping_util import get_point2img_mapper
                self.point2img_mapper = get_point2img_mapper()
                print("Initialized point2img_mapper for runtime projection (fallback)")

        # 列出样本，只包含有 3D 数据的场景
        self.samples = _list_samples_from_precomputed(
            self.precomputed_dir, 
            data_root=self.data_root, 
            split=self.split
        )
        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"在 {self.precomputed_dir} 下未找到有对应 3D 数据的 *_odise.npz。"
                f"请检查 {self.data_root}/{self.split} 下是否有匹配的 .pth 文件。"
            )
        
        # 如果使用预计算投影，进一步过滤只有投影文件的样本
        if self.projection_dir is not None:
            original_count = len(self.samples)
            self.samples = [
                (scene, frame) for scene, frame in self.samples
                if (self.projection_dir / scene / f"{frame}_proj.npz").exists()
            ]
            print(f"Found {len(self.samples)}/{original_count} samples with precomputed projections")
        else:
            print(f"Found {len(self.samples)} samples with both NPZ and PTH data")
        
        n = len(self.samples)
        if getattr(config, "max_samples_ratio", None) and 0 < config.max_samples_ratio < 1:
            n = max(1, int(n * config.max_samples_ratio))
        if getattr(config, "max_samples", None) and config.max_samples > 0:
            n = min(n, config.max_samples) if n != len(self.samples) else config.max_samples
        if n < len(self.samples):
            self.samples = self.samples[:n]
            print(f"Using {n} samples after filtering")

        # 预加载 3D 数据到内存，避免反复从磁盘读取 .pth 文件
        # DDP 多进程情况下只在主进程预加载，避免重复占用内存
        self._pth_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._enable_cache = getattr(config, 'memcache_init', True)  # 默认启用预加载
        
        # 检查是否在 DDP 环境中
        try:
            import torch.distributed as dist
            is_distributed = dist.is_available() and dist.is_initialized()
            is_main_process = not is_distributed or dist.get_rank() == 0
        except:
            is_distributed = False
            is_main_process = True
        
        if self._enable_cache and self.data_root and self.data_root.exists():
            # DDP 情况下只在主进程预加载，子进程从磁盘读取（避免重复占用内存）
            if is_main_process:
                unique_scenes = sorted(set(scene for scene, _ in self.samples))
                print(f"Pre-loading {len(unique_scenes)} scenes into memory (main process only)...")
                for scene_name in unique_scenes:
                    pth_path = self.data_root / self.split / f"{scene_name}.pth"
                    if not pth_path.exists():
                        pth_path = self.data_root / self.split / f"{scene_name}_vh_clean_2.pth"
                    if pth_path.exists():
                        self._pth_cache[scene_name] = _cached_load_pth(pth_path)
                print(f"Pre-loaded {len(self._pth_cache)} scenes ✅")
            else:
                # 子进程不预加载，从磁盘读取（会慢一些，但避免 OOM）
                print(f"Worker process: will load scenes from disk on-demand")

    def __len__(self) -> int:
        return len(self.samples) * max(1, self.config.loop)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % len(self.samples)
        scene_name, frame_stem = self.samples[idx]

        npz_path = self.precomputed_dir / scene_name / f"{frame_stem}_odise.npz"
        out = _load_npz_pooled(npz_path)

        # 优先使用预计算投影（方案 B），保证投影和 npz 是同一帧
        out_3d = None
        if self.projection_dir is not None and self.data_root:
            out_3d = _load_3d_with_precomputed_projection(
                data_root=self.data_root,
                split=self.split,
                scene_name=scene_name,
                projection_dir=self.projection_dir,
                frame_stem=frame_stem,
                pth_cache=self._pth_cache,
                voxel_size=self.config.voxel_size,
            )

        # 如果没有预计算投影或加载失败，回退到运行时投影
        if out_3d is None:
            # 🔥 关键修复：训练时禁止 fallback 到运行时投影，确保坐标体系一致性
            if self.split == 'train':
                raise RuntimeError(
                    f"Missing precomputed projection for training sample: "
                    f"{scene_name}/{frame_stem}. "
                    f"Training requires all samples to use precomputed projections "
                    f"to ensure coordinate consistency. "
                    f"Please check if projection file exists: "
                    f"{self.projection_dir / scene_name / f'{frame_stem}_proj.npz'}"
                )
            
            # val/test 可以 fallback 到运行时投影（但会打印警告）
            print(f"⚠️  Warning: Fallback to runtime projection for {self.split} sample {scene_name}/{frame_stem}")
            
            if self.data_root and self.data_root.exists():
                out_3d = _load_3d_with_projection(
                    data_root=self.data_root,
                    split=self.split,
                    scene_name=scene_name,
                    data_root_2d=self.data_root_2d,
                    point2img_mapper=self.point2img_mapper,
                    min_visible=400,
                    max_visible=65000,
                    voxel_size=self.config.voxel_size,  # 🔥 传递 voxel_size
                )
            else:
                raise FileNotFoundError(f"data_root not found: {self.data_root}")

        out["coords_3d"] = out_3d["coords_3d"]
        out["feat_3d"] = out_3d["feat_3d"]
        out["ori_coords_3d"] = out_3d["ori_coords_3d"]
        out["inds_reconstruct"] = out_3d["inds_reconstruct"]
        out["x_label"] = out_3d["x_label"]
        out["y_label"] = out_3d["y_label"]
        out["binary_label_3d"] = out_3d["binary_label_3d"]
        out["binary_label_2d"] = out_3d["binary_label_2d"]
        out["label_2d"] = out_3d["label_2d"]
        return out


def open_vocab_collate_v2(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate：pad pixel_pooled / masks / mask_embeddings 到 K_max，stack 3D。"""
    B = len(batch)
    max_k = max(b["pixel_pooled"].shape[0] for b in batch)
    _, H, W = batch[0]["masks"].shape
    Cp = batch[0]["pixel_pooled"].shape[1]
    Cm = batch[0]["mask_embeddings"].shape[1]

    pixel_pooled = torch.zeros(B, max_k, Cp, dtype=torch.float32)
    masks = torch.zeros(B, max_k, H, W, dtype=torch.float32)
    mask_embeddings = torch.zeros(B, max_k, Cm, dtype=torch.float32)
    mask_valid = torch.zeros(B, max_k, dtype=torch.bool)

    coords_3d_list = []
    feat_3d_list = []
    ori_coords_3d_list = []
    inds_reconstruct_list = []
    x_label_list = []
    y_label_list = []
    binary_label_3d_list = []
    binary_label_2d_list = []
    label_2d_list = []

    offset = 0
    for b, item in enumerate(batch):
        k = item["pixel_pooled"].shape[0]
        pixel_pooled[b, :k] = item["pixel_pooled"]
        masks[b, :k] = item["masks"]
        mask_embeddings[b, :k] = item["mask_embeddings"]
        mask_valid[b, :k] = item["mask_valid"]

        coords_3d_list.append(item["coords_3d"])
        feat_3d_list.append(item["feat_3d"])
        ori_coords_3d_list.append(item["ori_coords_3d"])
        inds_reconstruct_list.append(item["inds_reconstruct"] + offset)
        x_label_list.append(item["x_label"])
        y_label_list.append(item["y_label"])
        binary_label_3d_list.append(item["binary_label_3d"])
        binary_label_2d_list.append(item["binary_label_2d"])
        label_2d_list.append(item["label_2d"])
        offset += item["coords_3d"].shape[0]

    # 给 coords_3d / ori_coords_3d 第一列填 batch 索引
    coords_3d_list2 = []
    ori_list2 = []
    for b, (c, o) in enumerate(zip(coords_3d_list, ori_coords_3d_list)):
        c = c.clone()
        o = o.clone()
        c[:, 0] = b
        o[:, 0] = b
        coords_3d_list2.append(c)
        ori_list2.append(o)

    return {
        "pixel_pooled": pixel_pooled,
        "masks": masks,
        "mask_embeddings": mask_embeddings,
        "mask_valid": mask_valid,
        "coords_3d": torch.cat(coords_3d_list2, dim=0),
        "feat_3d": torch.cat(feat_3d_list, dim=0),
        "ori_coords_3d": torch.cat(ori_list2, dim=0),
        "inds_reconstruct": torch.cat(inds_reconstruct_list, dim=0),
        "x_label": torch.cat(x_label_list, dim=0),
        "y_label": torch.cat(y_label_list, dim=0),
        "binary_label_3d": torch.cat(binary_label_3d_list, dim=0),
        "binary_label_2d": torch.cat(binary_label_2d_list, dim=0),
        "label_2d": torch.cat(label_2d_list, dim=0),
    }
