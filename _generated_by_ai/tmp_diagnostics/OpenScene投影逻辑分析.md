# OpenScene 投影逻辑分析 - 多视角融合

## 🔍 关键发现

**OpenScene 的特征融合是遍历所有帧的多视角融合！不是单帧！**

### OpenScene 原始代码 (`scannet_openseg.py` 第 80-108 行)

```python
# 初始化累积器
counter = torch.zeros((n_points_cur, 1), device=device)       # 每个点被看到的次数
sum_features = torch.zeros((n_points_cur, feat_dim), device=device)  # 特征累加

################ Feature Fusion ###################
# 🔥 关键: 遍历所有图像帧!!!
for img_id, img_dir in enumerate(tqdm(img_dirs)):
    # 读取位姿和深度
    pose = np.loadtxt(posepath)
    depth = imageio.v2.imread(depth_path) / depth_scale
    
    # 计算 3D-2D 映射
    mapping = np.ones([n_points, 4], dtype=int)
    mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
    
    # 如果这帧没有可见点，跳过
    if mapping[:, 3].sum() == 0:
        continue
    
    # 提取 2D 特征
    feat_2d = extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=[240, 320])
    
    # 将 2D 特征映射到 3D 点
    feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)
    
    # 🔥 关键: 累加特征和计数器
    mask = mapping[:, 3]
    counter[mask != 0] += 1                    # 计数器 +1
    sum_features[mask != 0] += feat_2d_3d[mask != 0]  # 累加特征

# 🔥 关键: 平均融合
counter[counter == 0] = 1e-5  # 避免除零
feat_bank = sum_features / counter  # 平均特征

# 统计有多少点被至少一个视角看到
point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])
```

## 📊 这意味着什么？

### 1. 每个点被多帧观察

```
帧 1: 看到 5000 个点 (6%)
帧 2: 看到 4500 个点 (5.5%)，其中 2000 个与帧 1 重叠
帧 3: 看到 5200 个点 (6.4%)，其中 1500 个与前面重叠
...
帧 279: 看到 4800 个点 (5.9%)

累积后: 70000+ 个点被至少一帧看到 (86%+)
```

### 2. 特征是多帧平均

```python
# 如果点 P 被 5 帧看到:
feat_P = (feat_frame1 + feat_frame2 + feat_frame3 + feat_frame4 + feat_frame5) / 5
```

### 3. 最终覆盖率接近 100%

OpenScene 的目标是让**每个 3D 点都有对应的 2D 特征**:
- 遍历所有帧
- 每帧贡献一部分点的特征
- 多帧融合后覆盖率接近 100%

## ❌ 你的代码的问题

### 当前做法 (错误)

```python
# open_vocab_dataset_v2.py
# 只用一帧!
img_dir = img_dirs[len(img_dirs) // 2]  # 选中间帧
mapping = mapper.compute_mapping(pose, locs, depth)
# 只有 5% 的点有投影 ❌
```

### OpenScene 的做法 (正确)

```python
# 遍历所有帧
for img_dir in img_dirs:
    mapping = mapper.compute_mapping(pose, locs, depth)
    # 累加特征
    feat_bank[visible_points] += feat_2d[mapping]
    counter[visible_points] += 1

# 平均
feat_bank = feat_bank / counter
# 几乎所有点都有特征 ✅
```

## 🔍 为什么 data_loader.py 使用单帧？

`data_loader.py` 的设计目标不同:

```python
# data_loader.py 第 211-238 行
while True:
    # 计算投影
    single_mapping = self.point2img_mapper.compute_mapping(pose, locs_in, depth)
    mask = single_mapping[:, 2]
    
    # 🔥 关键: 只保留可见点!
    label_3d = labels_in[mask == 1]       # 只保留可见点的标签
    feature_3d = feats_in[mask == 1]      # 只保留可见点的特征
    locals_3d = locs_in[mask == 1]        # 只保留可见点的坐标
    
    # 检查可见点数
    if np.sum(mask) > 400 and np.sum(mask) < 65000:
        break
```

**关键区别**:
- `data_loader.py`: **过滤掉不可见的点**，只用可见的 400+ 点训练
- `scannet_openseg.py`: **保留所有点**，遍历所有帧融合特征

## 🎯 对于训练的影响

### OpenScene 训练流程

1. **预处理阶段** (一次性):
   - 遍历所有帧，融合特征
   - 保存融合后的特征 (覆盖率接近 100%)

2. **训练阶段**:
   - 加载预融合的特征
   - 每个点都有有效特征

### 你的 V2 训练流程

1. **运行时**:
   - 使用预计算的 npz (只是单帧特征，不是融合特征)
   - 需要投影将 3D 点映射到 2D 特征

**问题**: 单帧只有 5% 的点可见！

## ✅ 解决方案

### 方案 A: 像 data_loader.py 一样过滤点

只用可见的点训练:

```python
# 计算投影
mapping = mapper.compute_mapping(pose, locs, depth)
mask = mapping[:, 2] == 1

# 只保留可见点
locs_visible = locs[mask]           # 只有 5% 的点
feat_visible = feat[mask]
labels_visible = labels[mask]

# 用这些点训练
# 优点: 逻辑简单
# 缺点: 每次只用少量点
```

### 方案 B: 多帧融合 (推荐)

像 OpenScene 一样遍历多帧:

```python
counter = np.zeros(N)
x_label_sum = np.zeros(N)
y_label_sum = np.zeros(N)

# 遍历多帧
for img_dir in selected_frames:
    mapping = mapper.compute_mapping(pose, locs, depth)
    mask = mapping[:, 2] == 1
    
    # 累加
    counter[mask] += 1
    x_label_sum[mask] += mapping[mask, 1]  # x
    y_label_sum[mask] += mapping[mask, 0]  # y

# 平均 (取最后一次的投影坐标，因为坐标不能平均)
valid = counter > 0
x_label[valid] = last_x[valid]
y_label[valid] = last_y[valid]

# 覆盖率大幅提升
```

### 方案 C: 使用预融合特征 (最佳)

像 OpenScene 一样:

1. **预处理**: 运行 `scannet_openseg.py` 生成融合特征
2. **训练**: 使用融合后的特征 (每个点都有特征)

```bash
# 生成融合特征
python scripts/feature_fusion/scannet_openseg.py \
    --data_dir /path/to/data \
    --split train \
    --output_dir /path/to/output
```

## 📊 覆盖率对比

| 方法 | 覆盖率 | 说明 |
|------|--------|------|
| 单帧 | 5-7% | 一个视角只能看到一部分 |
| 多帧 (20帧) | 40-60% | 多视角累积 |
| 全帧融合 (279帧) | 85-95% | OpenScene 的做法 |

## 🔧 实际修复建议

### 对于 open_vocab_dataset_v2.py:

**选项 1**: 使用多帧累积投影 (推荐)

```python
def _compute_projection_multi_frame(locs, scene_2d_dir, mapper, max_frames=50):
    """多帧累积计算投影，覆盖更多点"""
    N = len(locs)
    x_label = np.zeros(N, dtype=np.int64)
    y_label = np.zeros(N, dtype=np.int64)
    covered = np.zeros(N, dtype=bool)
    
    img_dirs = sorted(glob(str(scene_2d_dir / "color" / "*")))
    
    # 均匀采样帧
    step = max(1, len(img_dirs) // max_frames)
    selected = list(range(0, len(img_dirs), step))[:max_frames]
    
    for idx in selected:
        img_dir = img_dirs[idx]
        try:
            pose = np.loadtxt(pose_path)
            depth = np.array(PIL.open(depth_path)) / 1000.0
            
            mapping = mapper.compute_mapping(pose, locs.numpy(), depth)
            mask = mapping[:, 2] == 1
            
            # 只更新还没被覆盖的点
            new_points = mask & ~covered
            if np.sum(new_points) > 0:
                # 正确顺序: 第0列是 y, 第1列是 x
                y_label[new_points] = mapping[new_points, 0].astype(np.int64)
                x_label[new_points] = mapping[new_points, 1].astype(np.int64)
                covered[new_points] = True
            
            # 如果覆盖率足够高，提前退出
            if np.sum(covered) / N > 0.8:
                break
                
        except:
            continue
    
    print(f"Multi-frame coverage: {np.sum(covered)}/{N} ({np.sum(covered)/N*100:.1f}%)")
    return x_label, y_label
```

**选项 2**: 像 data_loader.py 一样过滤点

只保留可见点训练，忽略不可见点。

## 🎯 总结

1. **OpenScene 使用多视角融合**，遍历所有帧
2. **单帧覆盖率 5-7% 是正常的**，但不能只用单帧
3. **需要多帧累积**才能获得足够的覆盖率
4. **或者过滤掉不可见点**，只用可见点训练
