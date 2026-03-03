# 多帧融合 vs 单帧训练 - 关键区别

## 📊 覆盖率的含义

### 分母是什么？

```
覆盖率 = 有 2D 投影的 3D 点数 / 总 3D 点数

例如 scene0000_00:
- 总 3D 点数: 81,369 个
- 单帧可见: 5,555 个 → 覆盖率 = 5,555 / 81,369 = 6.8%
- 多帧融合: 77,681 个 → 覆盖率 = 77,681 / 81,369 = 95.5%
```

**分母**: 场景中的**所有 3D 点** (整个点云)

## 🎯 两种训练思路对比

### 思路 1: OpenScene 的多帧融合 (预处理)

**流程**:
```
预处理阶段 (一次性):
  遍历所有 279 帧 → 每个 3D 点累积多帧的 2D 特征 → 
  平均融合 → 保存融合特征 (95% 的点有特征)

训练阶段:
  加载融合特征 → 所有点都有有效特征 → 直接训练
```

**优点**:
- ✅ 95% 的点有 2D 特征
- ✅ 特征质量高 (多帧平均，更鲁棒)
- ✅ 训练速度快 (预计算)

**缺点**:
- ❌ 需要预处理时间 (~3秒/场景)
- ❌ 需要存储空间 (融合特征)

---

### 思路 2: data_loader.py 的单帧训练 (运行时)

**流程**:
```
训练阶段 (每次):
  随机选一帧 → 计算投影 → 只保留可见点 (400-65000个) → 
  只用这些点训练 → 下一个 batch 换一帧
```

**关键**: **过滤掉不可见的点！**

```python
# data_loader.py 第 211-217 行
single_mapping = self.point2img_mapper.compute_mapping(pose, locs_in, depth)
mask = single_mapping[:, 2]

# 🔥 关键: 只保留可见点
label_3d = labels_in[mask == 1]       # 只保留 5% 的点
feature_3d = feats_in[mask == 1]
locals_3d = locs_in[mask == 1]
```

**优点**:
- ✅ 不需要预处理
- ✅ 每个 batch 看到不同的点 (多样性)
- ✅ 多个 epoch 后累积覆盖所有点

**缺点**:
- ❌ 每次只用 5% 的点
- ❌ 训练速度慢 (运行时投影)

---

### 思路 3: 你的 V2 模型 (当前 - 错误)

**流程**:
```
训练阶段:
  加载预计算 npz (单帧 2D 特征) → 
  加载 3D 点云 (所有点) → 
  用单帧投影 (只有 5% 的点有投影) → 
  训练时 95% 的点没有对应的 2D 特征 ❌
```

**问题**: 
- ❌ 想用所有 3D 点训练
- ❌ 但只有 5% 的点有 2D 对应
- ❌ 95% 的点的 x_label/y_label 是 0
- ❌ Loss 计算时这些点被跳过

---

## 🔧 你的模型应该怎么做？

### 选项 A: 采用多帧融合 (推荐，和 OpenScene 一样)

**实现**: 在数据加载时累积多帧投影

```python
def _load_3d_with_multiframe_projection(data_root, split, scene_name, data_2d_root):
    # 加载 3D 数据
    locs, feats, labels = load_3d_data(...)
    N = len(locs)
    
    # 初始化
    covered = np.zeros(N, dtype=bool)
    x_label = np.zeros(N, dtype=np.int64)
    y_label = np.zeros(N, dtype=np.int64)
    
    # 🔥 关键: 遍历多帧累积
    img_dirs = sorted(glob(str(scene_2d_dir / "color" / "*")))
    
    # 可以只用部分帧加速 (如 50 帧)
    step = max(1, len(img_dirs) // 50)
    for i in range(0, len(img_dirs), step):
        img_dir = img_dirs[i]
        
        # 计算投影
        pose = np.loadtxt(pose_path)
        depth = np.array(PIL.open(depth_path)) / 1000.0
        mapping = mapper.compute_mapping(pose, locs, depth)
        
        # 更新未覆盖的点
        mask = mapping[:, 2] == 1
        new_points = mask & ~covered
        
        if np.sum(new_points) > 0:
            # 正确顺序: [y, x, valid]
            y_label[new_points] = mapping[new_points, 0]
            x_label[new_points] = mapping[new_points, 1]
            covered[new_points] = True
        
        # 如果覆盖率够高，提前退出
        if np.sum(covered) / N > 0.9:
            break
    
    print(f"Coverage: {np.sum(covered)}/{N} ({np.sum(covered)/N*100:.1f}%)")
    
    return {
        "x_label": torch.from_numpy(x_label),
        "y_label": torch.from_numpy(y_label),
        # ...
    }
```

**效果**:
- 50 帧: 覆盖率 ~33% (0.6秒)
- 100 帧: 覆盖率 ~70% (1.1秒)
- 150 帧: 覆盖率 ~90% (1.6秒)

**训练时**:
- 90% 的点有有效的 x_label/y_label
- Loss 计算时不会被大量跳过
- 训练正常进行

---

### 选项 B: 过滤不可见点 (和 data_loader.py 一样)

**实现**: 只保留可见点训练

```python
def __getitem__(self, idx):
    # 加载数据
    npz_data = load_npz(...)
    out_3d = load_3d(...)
    
    # 计算投影 (单帧)
    mapping = compute_projection_single_frame(...)
    mask = mapping[:, 2] == 1
    
    # 🔥 关键: 只保留可见点
    coords_3d = out_3d["coords_3d"][mask]
    feat_3d = out_3d["feat_3d"][mask]
    labels = out_3d["labels"][mask]
    x_label = out_3d["x_label"][mask]
    y_label = out_3d["y_label"][mask]
    
    # 返回过滤后的数据
    return {
        "coords_3d": coords_3d,  # 只有 5% 的点
        "feat_3d": feat_3d,
        "x_label": x_label,
        "y_label": y_label,
        # ...
    }
```

**效果**:
- 每个 batch 只用 400-5000 个点
- 但每个点都有有效的 2D 对应
- 多个 batch/epoch 累积覆盖所有点

---

### 选项 C: 预处理多帧融合特征 (最佳，但需要时间)

**实现**: 运行 OpenScene 的预处理脚本

```bash
# 生成融合特征
cd /home/featurize/work/openscene
python scripts/feature_fusion/scannet_openseg.py \
    --data_dir /home/featurize/data \
    --split train \
    --output_dir /home/featurize/data/scannet_fused_features
```

**效果**:
- 一次性处理，生成融合特征
- 训练时直接加载，95% 覆盖率
- 训练速度最快

---

## 📊 性能对比

| 方法 | 覆盖率 | 加载耗时 | 训练速度 | 预处理 |
|------|--------|---------|---------|--------|
| 单帧 (当前) | 5% | 0.1s | 快 | 无 | ❌ 不可用 |
| 单帧+过滤 | 100%* | 0.1s | 快 | 无 | ✅ 可用 |
| 多帧累积 (50帧) | 33% | 0.6s | 中 | 无 | ✅ 可用 |
| 多帧累积 (150帧) | 90% | 1.6s | 慢 | 无 | ✅ 推荐 |
| 预处理融合 | 95% | 0.1s | 快 | 3s/场景 | ✅ 最佳 |

*注: 单帧+过滤的 100% 是指"使用的点中 100% 有对应"，但只用了 5% 的点

---

## 🎯 推荐方案

### 短期 (快速开始训练)

**使用选项 B: 单帧+过滤**

优点:
- ✅ 实现简单
- ✅ 立即可用
- ✅ 不需要预处理

缺点:
- ⚠️ 每次只用少量点
- ⚠️ 需要更多 epoch

### 长期 (最佳性能)

**使用选项 A: 多帧累积 (150帧)**

优点:
- ✅ 90% 覆盖率
- ✅ 不需要预处理
- ✅ 训练效果好

缺点:
- ⚠️ 加载稍慢 (~1.6s/场景)

或

**使用选项 C: 预处理融合**

优点:
- ✅ 95% 覆盖率
- ✅ 训练最快
- ✅ 特征质量最高

缺点:
- ⚠️ 需要预处理 (~3s/场景 × 977 = 50分钟)

---

## 💡 关键理解

### 1. 覆盖率的分母

```
覆盖率 = 有 2D 对应的 3D 点数 / 场景中所有 3D 点数
```

### 2. 两种训练策略

**策略 A: 用所有点，需要高覆盖率**
- 需要多帧融合
- 覆盖率 > 90%
- 适合你的 V2 模型

**策略 B: 只用可见点，覆盖率无所谓**
- 单帧即可
- 过滤掉不可见点
- 适合 data_loader.py

### 3. 你的模型应该选哪个？

**如果你想用所有 3D 点训练** → 必须用多帧融合 (选项 A 或 C)

**如果你可以只用部分点训练** → 可以用单帧+过滤 (选项 B)

---

## 🔧 立即可用的修复

最简单的方法是**选项 B: 过滤不可见点**

```python
# 在 open_vocab_dataset_v2.py 的 __getitem__ 中添加
def __getitem__(self, idx):
    # ... 加载数据 ...
    
    # 计算投影 (单帧)
    out_3d = _load_3d_simple(...)
    
    # 🔥 过滤: 只保留有有效投影的点
    valid_mask = (out_3d["x_label"] != 0) | (out_3d["y_label"] != 0)
    
    if valid_mask.sum() < 400:
        # 如果可见点太少，重新选一帧
        # ... 循环逻辑 ...
        pass
    
    # 只返回可见点
    for key in ["coords_3d", "feat_3d", "x_label", "y_label", "labels"]:
        out_3d[key] = out_3d[key][valid_mask]
    
    return out_3d
```

这样:
- ✅ 立即可用
- ✅ 每个点都有有效的 2D 对应
- ✅ Loss 不会为 0
- ✅ 可以正常训练
