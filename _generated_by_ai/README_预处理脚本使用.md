# 预处理融合特征脚本使用说明

## 📁 生成的文件

1. **`preprocess_fused_features.py`** - 主要的预处理脚本，生成多帧融合特征
2. **`verify_fused_features.py`** - 验证脚本，检查生成的文件是否正确

---

## 🚀 使用步骤

### 步骤 1: 测试运行（处理少量场景）

先用几个场景测试，确保脚本能正常工作：

```bash
cd /home/featurize/work/mix

# 只处理前 5 个场景进行测试
python preprocess_fused_features.py \
    --data-root-3d /home/featurize/data/scannet_3d \
    --data-root-2d /home/featurize/data/scannet_2d \
    --npz-dir /home/featurize/data/pixel_pooled \
    --output-dir /home/featurize/data/scannet_fused \
    --splits train val \
    --max-scenes 5
```

**预期输出**：
- 每个场景的处理进度
- 覆盖率统计（应该 > 90%）
- 生成的文件路径

### 步骤 2: 验证测试文件

验证刚才生成的文件是否正确：

```bash
python verify_fused_features.py \
    --fused-dir /home/featurize/data/scannet_fused \
    --splits train val \
    --verbose
```

**检查项**：
- ✅ 所有文件都能正常加载
- ✅ 数据形状正确 (mask_embeddings_3d: N×256, pixel_pooled_3d: N×512)
- ✅ 覆盖率 > 90%
- ✅ 没有 NaN 或 Inf
- ✅ 数值范围合理

### 步骤 3: 完整处理（所有场景）

如果测试通过，运行完整处理：

```bash
# 处理所有场景（可能需要 1-2 小时）
python preprocess_fused_features.py \
    --data-root-3d /home/featurize/data/scannet_3d \
    --data-root-2d /home/featurize/data/scannet_2d \
    --npz-dir /home/featurize/data/pixel_pooled \
    --output-dir /home/featurize/data/scannet_fused \
    --splits train val
```

**注意**：
- 总共约 977 个场景
- 预计耗时：1-2 小时
- 磁盘空间需求：约 60 GB

### 步骤 4: 最终验证

处理完成后，验证所有文件：

```bash
python verify_fused_features.py \
    --fused-dir /home/featurize/data/scannet_fused \
    --splits train val
```

---

## 📊 输出文件格式

生成的文件位置：
```
/home/featurize/data/scannet_fused/
├── train/
│   ├── scene0000_00_fused.pt
│   ├── scene0000_01_fused.pt
│   └── ...
└── val/
    ├── scene0011_00_fused.pt
    └── ...
```

每个 `*_fused.pt` 文件包含：

```python
{
    "locs": torch.Tensor,              # (N, 3) 3D 坐标
    "feats": torch.Tensor,             # (N, 3) 颜色特征
    "labels": torch.Tensor,            # (N,) 语义标签
    "mask_embeddings_3d": torch.Tensor,  # (N, 256) ODISE 融合特征 (float16)
    "pixel_pooled_3d": torch.Tensor,     # (N, 512) LSeg 融合特征 (float16)
    "mask_full": torch.Tensor,          # (N,) bool，哪些点有有效特征
    "coverage": float,                  # 覆盖率 (0-1)
    "num_frames_used": int,             # 使用的帧数
}
```

---

## 🔧 参数说明

### `preprocess_fused_features.py` 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-root-3d` | `/home/featurize/data/scannet_3d` | 3D 点云数据根目录 |
| `--data-root-2d` | `/home/featurize/data/scannet_2d` | 2D RGB-D 数据根目录 |
| `--npz-dir` | `/home/featurize/data/pixel_pooled` | NPZ 特征文件目录 |
| `--output-dir` | `/home/featurize/data/scannet_fused` | 输出目录 |
| `--splits` | `train val` | 要处理的数据集划分 |
| `--max-scenes` | `None` | 最大处理场景数（测试用） |

### `verify_fused_features.py` 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fused-dir` | `/home/featurize/data/scannet_fused` | 融合特征目录 |
| `--splits` | `train val` | 要验证的数据集划分 |
| `--max-files` | `None` | 最大验证文件数（测试用） |
| `--verbose` | `False` | 显示详细信息 |

---

## ⚠️ 常见问题

### 1. 覆盖率过低（< 50%）

**原因**：
- 该场景的 2D 帧太少
- NPZ 文件缺失或损坏
- 投影失败

**解决**：检查日志中的警告信息

### 2. 内存不足

**解决**：
- 一次处理一个 split
- 减少 `--max-scenes` 分批处理

### 3. 文件加载失败

**原因**：
- 3D 数据格式不对
- 文件损坏

**解决**：查看具体错误信息

---

## 🎯 预期结果

- **覆盖率**：大部分场景 > 90%
- **处理速度**：每个场景 3-5 秒
- **文件大小**：每个场景约 60 MB
- **总耗时**：977 场景 × 4 秒 ≈ 65 分钟

---

## 📝 后续步骤

生成完成后，需要修改数据集加载器来使用这些预计算的特征：

1. 创建新的数据集类 `OpenVocabScannetDatasetV3`
2. 直接加载 `*_fused.pt` 文件
3. 不再需要运行时投影计算
4. 训练速度提升 10-20 倍

---

## 🐛 已修复的 Bug

脚本中已经修复了以下问题：

1. ✅ **X/Y 坐标交换** - `mapping[:, 0]` 是 y，`mapping[:, 1]` 是 x
2. ✅ **Mask 索引顺序** - 使用 `mask[y, x]` 而不是 `mask[x, y]`
3. ✅ **深度单位转换** - 从 mm 转换为 m
4. ✅ **边界检查** - 确保投影坐标在有效范围内
5. ✅ **模块导入** - 添加路径兼容性处理

---

## 💡 提示

- 建议先用 `--max-scenes 5` 测试
- 使用 `--verbose` 查看详细进度
- 处理完一个 split 再处理下一个，避免占用太多磁盘空间
- 可以用 `tmux` 或 `screen` 在后台运行长时间任务
