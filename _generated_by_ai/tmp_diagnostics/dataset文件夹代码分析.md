# Dataset 文件夹代码分析

## 📁 文件列表和作用

### 1. **`data_loader.py`** - 旧版完整训练数据加载器
**作用**: OpenScene 原始训练流程，运行时提取 2D 特征
- ✅ **有运行时投影计算** (第 209 行)
- ✅ **动态选择最佳帧** (循环直到找到覆盖率 > 400 点的帧)
- ✅ **读取 LSeg 和 ODISE 预计算特征**
- ✅ **体素化和数据增强**
- ⚠️ **坐标顺序有 bug** (第 326-327 行把 y 当 x)

**关键特性**:
```python
# 循环选帧，直到找到覆盖率好的帧
while True:
    # 计算投影
    single_mapping = self.point2img_mapper.compute_mapping(pose, locs_in, depth)
    mask = single_mapping[:, 2]
    
    # 检查可见点数
    if np.sum(mask) > 400 and valid_point_num > 10 and np.sum(mask) < 65000:
        break  # 找到好的帧
    # 否则继续尝试下一帧
```

**数据流**:
```
3D .pth → 选择场景 → 循环尝试不同帧 → 找到覆盖率好的帧 → 
读取该帧的 LSeg/ODISE 特征 → 体素化 → 返回训练数据
```

---

### 2. **`open_vocab_dataset_v2.py`** - 新版快速训练数据加载器
**作用**: 使用预计算 2D 特征的快速版本
- ❌ **没有运行时投影计算** (期望从 .pth 读取)
- ✅ **使用预计算的 npz 文件** (pixel_pooled)
- ✅ **更快的训练速度** (不需要读图像)
- ⚠️ **缺少 x_label/y_label 导致训练失败**

**设计理念**: 预计算一切，加速训练
```python
# 直接读取预计算的特征
npz_path = precomputed_dir / scene_name / f"{frame_stem}_odise.npz"
out = _load_npz_pooled(npz_path)  # 包含 pixel_pooled, masks, mask_embeddings

# 期望 .pth 包含投影标签
out_3d = _load_3d_simple(data_root, split, scene_name)
# 如果没有 x_label/y_label，用零填充 ❌
```

**数据流**:
```
预计算 npz (2D 特征) + 3D .pth (点云) → 直接合并 → 返回训练数据
```

---

### 3. **`data_loader_infer.py`** - 推理数据加载器
**作用**: 用于测试/推理，类似 `data_loader.py` 但简化
- ✅ 有运行时投影计算
- ✅ 用于评估和可视化

---

### 4. **`open_vocab_dataset.py`** - 旧版 open vocab 数据集
**作用**: 可能是早期版本，功能类似 `open_vocab_dataset_v2.py`

---

### 5. **`feature_loader.py`** - 特征加载工具
**作用**: 加载预计算的 2D 特征 (LSeg/ODISE)

---

### 6. **`point_loader.py`** - 3D 点云加载工具
**作用**: 加载和预处理 3D 点云数据

---

### 7. **`voxelizer.py`** + **`voxelization_utils.py`** - 体素化工具
**作用**: 将 3D 点云体素化，用于 MinkowskiEngine

---

### 8. **`augmentation.py`** - 数据增强
**作用**: 3D 点云的数据增强 (旋转、缩放、抖动等)

---

## 🔍 关键问题分析

### 问题 1: 为什么覆盖率这么低 (6.9%)?

**原因**: 你的测试**只用了单帧**，但 `data_loader.py` 会**循环尝试多帧**直到找到覆盖率好的！

#### `data_loader.py` 的做法 (正确):

```python
# 第 184-240 行
while True:
    # 随机或顺序选择一帧
    if self.split in ["val", "test"]:
        img_idx = img_idx % len(img_dirs)
        img_dir = img_dirs[img_idx]
    else:
        img_dir = np.random.choice(img_dirs, 1, replace=False)[0]
    
    # 计算投影
    single_mapping = self.point2img_mapper.compute_mapping(pose, locs_in, depth)
    mask = single_mapping[:, 2]
    
    # 检查覆盖率
    if np.sum(mask) > 400 and valid_point_num > 10 and np.sum(mask) < 65000:
        break  # ✅ 找到覆盖率好的帧 (> 400 点)
    
    # 否则继续尝试下一帧
    if self.split in ["val", "test"]:
        img_idx += 2
```

**关键**: 
- 训练时**随机选帧**，直到找到覆盖率 > 400 点的帧
- 验证时**顺序选帧**，跳过覆盖率低的
- **不是所有帧都能看到所有点**，需要多帧融合

#### 你的测试只用了单帧 (不完整):

```python
# 只选了中间帧
img_dir = img_dirs[len(img_dirs) // 2]
# 没有循环尝试其他帧
```

**解决方案**: 在 `open_vocab_dataset_v2.py` 中也要**循环选帧**！

---

### 问题 2: 2D 特征尺寸对齐问题

#### 当前的尺寸:

根据代码注释：
```python
# data_loader.py 第 248 行
lseg_feat = np.load(...)  # 320, 240, 512

# data_loader.py 第 252 行
masks_odise = data["masks"]  # num_mask, 240, 320
```

**注意**: LSeg 和 ODISE 的尺寸**不一样**！
- LSeg: `(320, 240, 512)` = (W, H, C)
- ODISE masks: `(K, 240, 320)` = (K, H, W)

#### 投影映射器的配置:

```python
# utils/mapping_util.py 第 34 行
img_dim = (320, 240)  # (W, H)

# 第 46-48 行
intrinsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
intrinsic = adjust_intrinsic(
    intrinsic, intrinsic_image_dim=[640, 480], image_dim=img_dim
)
```

**关键**: 
- 原始图像: 640 x 480
- 下采样到: 320 x 240
- 内参已经调整: `adjust_intrinsic` 会缩放焦距和主点

#### 投影坐标的范围:

```python
# compute_mapping 返回的坐标是针对 320x240 的
# x 范围: [0, 320)
# y 范围: [0, 240)
```

**结论**: ✅ **代码已经正确处理了下采样！**

---

## 🔧 修复 `open_vocab_dataset_v2.py` 的完整方案

### 需要添加的功能:

1. **循环选帧** (像 `data_loader.py` 一样)
2. **运行时投影计算**
3. **正确的坐标顺序** (y, x)

### 完整的修复代码:

```python
def _load_3d_simple_with_projection(
    data_root: Path, 
    split: str, 
    scene_name: str,
    data_2d_root: Optional[Path] = None,
    min_visible_points: int = 400,
    max_visible_points: int = 65000,
    max_attempts: int = 20,
) -> Dict[str, torch.Tensor]:
    """
    加载 3D 数据并计算投影标签。
    
    关键改进:
    1. 循环尝试多帧，找到覆盖率好的帧
    2. 运行时计算投影
    3. 正确的坐标顺序 (y, x)
    """
    pth_path = data_root / split / f"{scene_name}.pth"
    if not pth_path.exists():
        pth_path = data_root / split / f"{scene_name}_vh_clean_2.pth"
    if not pth_path.exists():
        # 返回占位数据...
        return {...}

    # 加载 3D 数据
    data = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(data, (list, tuple)):
        locs, feats, labels = data[0], data[1], data[2]
    else:
        locs = data.get("locs", data.get("coords", torch.zeros(100, 3)))
        feats = data.get("feats", data.get("feat", torch.ones(locs.shape[0], 3)))
        labels = data.get("labels", torch.zeros(locs.shape[0], dtype=torch.long))

    if isinstance(locs, np.ndarray):
        locs = torch.from_numpy(locs)
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    N = locs.shape[0]
    batch_idx = torch.zeros(N, 1, dtype=torch.long)
    coords_3d = torch.cat([batch_idx, locs.long()], dim=1)
    feat_3d = feats.float() if feats.dim() > 1 else feats.unsqueeze(1).expand(N, 3)

    # 尝试从数据中获取 x_label 和 y_label
    x_label = data.get("x_label") if isinstance(data, dict) else None
    y_label = data.get("y_label") if isinstance(data, dict) else None
    
    if x_label is None or y_label is None:
        # 🔥 运行时计算投影 (循环选帧)
        if data_2d_root is not None and data_2d_root.exists():
            try:
                from utils.mapping_util import getMapping
                from glob import glob
                import random
                try:
                    from PIL import Image as PIL
                except ImportError:
                    PIL = None
                
                if PIL is None:
                    raise ImportError("需要 pillow 库")
                
                scene_2d_dir = data_2d_root / scene_name
                img_dirs = sorted(glob(str(scene_2d_dir / "color" / "*")),
                                 key=lambda x: int(os.path.basename(x)[:-4]))
                
                if len(img_dirs) == 0:
                    raise FileNotFoundError(f"No images for {scene_name}")
                
                mapper = getMapping()
                best_mapping = None
                best_num_visible = 0
                
                # 🔥 关键: 循环尝试多帧，找到覆盖率好的
                if split == "train":
                    # 训练时随机尝试
                    attempt_indices = random.sample(range(len(img_dirs)), 
                                                   min(max_attempts, len(img_dirs)))
                else:
                    # 验证时顺序尝试
                    attempt_indices = range(0, min(len(img_dirs), max_attempts * 2), 2)
                
                for idx in attempt_indices:
                    img_dir = img_dirs[idx]
                    
                    try:
                        # 读取位姿和深度
                        pose_path = img_dir.replace("color", "pose").replace(".jpg", ".txt")
                        depth_path = img_dir.replace("color", "depth").replace("jpg", "png")
                        
                        if not os.path.exists(pose_path) or not os.path.exists(depth_path):
                            continue
                        
                        pose = np.loadtxt(pose_path)
                        depth = np.array(PIL.open(depth_path)) / 1000.0
                        
                        # 计算投影
                        single_mapping = mapper.compute_mapping(pose, locs.numpy(), depth)
                        
                        # 统计可见点数
                        mask = single_mapping[:, 2]
                        num_visible = np.sum(mask == 1)
                        
                        # 🔥 检查覆盖率 (像 data_loader.py 一样)
                        if num_visible > min_visible_points and num_visible < max_visible_points:
                            # 找到好的帧！
                            best_mapping = single_mapping
                            best_num_visible = num_visible
                            print(f"Found good frame for {scene_name}: {num_visible}/{N} points ({num_visible/N*100:.1f}%)")
                            break
                        
                        # 记录最好的
                        if num_visible > best_num_visible:
                            best_mapping = single_mapping
                            best_num_visible = num_visible
                    
                    except Exception as e:
                        continue
                
                # 使用找到的最佳映射
                if best_mapping is not None:
                    # 🔥 关键: compute_mapping 返回 [y, x, valid]
                    zero_rows = np.all(best_mapping != 0, axis=1)
                    valid_indices = np.where(zero_rows)[0]
                    
                    x_label = np.zeros(N, dtype=np.int64)
                    y_label = np.zeros(N, dtype=np.int64)
                    
                    # ✅ 正确: 第0列是 y，第1列是 x
                    y_label[valid_indices] = best_mapping[valid_indices, 0].astype(np.int64)
                    x_label[valid_indices] = best_mapping[valid_indices, 1].astype(np.int64)
                    
                    x_label = torch.from_numpy(x_label)
                    y_label = torch.from_numpy(y_label)
                    
                    if best_num_visible < min_visible_points:
                        print(f"Warning: Low coverage for {scene_name}: {best_num_visible}/{N} points")
                else:
                    raise RuntimeError(f"No valid projection found for {scene_name}")
                    
            except Exception as e:
                print(f"Warning: Failed to compute projection for {scene_name}: {e}")
                x_label = torch.zeros(N, dtype=torch.long)
                y_label = torch.zeros(N, dtype=torch.long)
        else:
            print(f"Warning: No 2D data for {scene_name}")
            x_label = torch.zeros(N, dtype=torch.long)
            y_label = torch.zeros(N, dtype=torch.long)
    else:
        # 已有投影标签
        if isinstance(x_label, np.ndarray):
            x_label = torch.from_numpy(x_label)
        if isinstance(y_label, np.ndarray):
            y_label = torch.from_numpy(y_label)
        x_label = x_label.long()
        y_label = y_label.long()

    return {
        "coords_3d": coords_3d,
        "feat_3d": feat_3d,
        "ori_coords_3d": coords_3d.clone(),
        "inds_reconstruct": torch.arange(N, dtype=torch.long),
        "x_label": x_label,
        "y_label": y_label,
        "binary_label_3d": labels.long(),
        "binary_label_2d": torch.zeros(N, dtype=torch.long),
        "label_2d": torch.zeros(N, dtype=torch.long),
    }
```

---

## 📊 总结对比

| 特性 | `data_loader.py` | `open_vocab_dataset_v2.py` (当前) | `open_vocab_dataset_v2.py` (修复后) |
|------|------------------|-----------------------------------|-------------------------------------|
| 运行时投影 | ✅ 有 | ❌ 没有 | ✅ 有 |
| 循环选帧 | ✅ 有 (找覆盖率好的) | ❌ 没有 (固定帧) | ✅ 有 |
| 坐标顺序 | ❌ 错误 (x,y 颠倒) | N/A | ✅ 正确 (y,x) |
| 覆盖率 | ✅ 高 (>400 点) | ❌ 低 (随机) | ✅ 高 (>400 点) |
| 训练速度 | 慢 (读图像) | 快 (预计算) | 中等 (运行时投影) |
| 2D 特征 | 运行时读取 | 预计算 npz | 预计算 npz |

---

## 🎯 下一步行动

1. ✅ **修复坐标顺序** - 已验证
2. ✅ **添加循环选帧** - 提高覆盖率
3. ✅ **运行时投影计算** - 解决 x_label/y_label 缺失
4. ✅ **尺寸对齐** - 已正确 (320x240)

**预期结果**:
- 覆盖率: 6.9% → 30-60% (通过循环选帧)
- 边界内比例: 100% (已修复)
- 训练 Loss: 不再为 0
