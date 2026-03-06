# ODISE 特征预计算 - 快速开始指南

## 📁 我创建的文件

1. **`scripts/precompute_odise_features.py`** - 主脚本（只包含 ODISE，不含 LSeg）
2. **`scripts/run_odise.sh`** - 运行脚本模板
3. **`scripts/README_odise.md`** - 详细使用文档
4. **`test_odise_quick.sh`** - 快速测试脚本

## 🚀 立即开始

### 第一步：激活环境

```bash
conda activate f_bak
```

### 第二步：快速测试（推荐！）

修改测试脚本中的数据路径，然后运行：

```bash
cd /home/sunl/work/mix

# 编辑 test_odise_quick.sh，修改 DATA_ROOT 变量
vim test_odise_quick.sh

# 运行测试（只处理1个场景的2张图片）
bash test_odise_quick.sh
```

### 第三步：完整运行

如果测试成功，运行完整处理：

```bash
python scripts/precompute_odise_features.py \
    --data-root /your/scannet/path \
    --output-dir /your/output/path
```

## 📋 主要命令示例

### 1. 最简单的运行（处理所有数据）

```bash
conda activate f_bak
cd /home/sunl/work/mix

python scripts/precompute_odise_features.py \
    --data-root /data/scannet/scans \
    --output-dir /data/precomputed_odise
```

### 2. 测试运行（少量数据）

```bash
python scripts/precompute_odise_features.py \
    --data-root /data/scannet/scans \
    --output-dir ./test_output \
    --max-scenes 2 \
    --max-images-per-scene 10
```

### 3. 断点续传（跳过已处理的文件）

```bash
python scripts/precompute_odise_features.py \
    --data-root /data/scannet/scans \
    --output-dir /data/precomputed_odise \
    --skip-existing
```

### 4. 使用 CPU（没有 GPU）

```bash
python scripts/precompute_odise_features.py \
    --data-root /data/scannet/scans \
    --output-dir /data/precomputed_odise \
    --device cpu
```

## 📊 输入数据格式

脚本支持两种目录结构：

**结构 1（标准 ScanNet）：**
```
/data/scannet/scans/
  scene0000_00/
    color/
      0.jpg
      1.jpg
      ...
```

**结构 2（简化）：**
```
/data/scannet/scans/
  scene0000_00/
    0.jpg
    1.jpg
    ...
```

## 📦 输出数据格式

```
output_dir/
  scene0000_00/
    0_odise.npz
    1_odise.npz
    ...
```

每个 `.npz` 文件包含：
- `masks`: (K, H, W) bool - K 个掩码
- `mask_embeddings`: (K, 256) float16 - K 个嵌入向量
- `num_masks`: int64 - 掩码数量
- `info`: 每个掩码的详细信息（类别、分数等）

## 🔍 如何读取输出

```python
import numpy as np

# 加载
data = np.load("scene0000_00/0_odise.npz", allow_pickle=True)

# 访问
masks = data["masks"]  # (K, H, W)
embeddings = data["mask_embeddings"]  # (K, 256)
num_masks = data["num_masks"]
info = data["info"]

# 查看信息
for i, mask_info in enumerate(info):
    print(f"Mask {i}: {mask_info['category_name']}, "
          f"score={mask_info['score']:.3f}")
```

## ❓ 常见问题

### Q1: 找不到场景？
```
Found 0 scenes
```
**解决：** 检查 `--data-root` 路径，确保包含 `scene*` 文件夹

### Q2: 环境激活失败？
**解决：** 
```bash
conda env list  # 查看所有环境
conda activate f_bak
```

### Q3: CUDA 内存不足？
**解决：** 使用 `--device cpu` 或减少 `--max-images-per-scene`

### Q4: 中断后想继续？
**解决：** 添加 `--skip-existing` 参数，会跳过已处理的文件

## 🎯 与原脚本的区别

| 特性 | 原脚本 | 新脚本 |
|-----|-------|-------|
| LSeg 特征 | ✅ | ❌ 移除 |
| ODISE 特征 | ✅ | ✅ 保留 |
| 依赖 | LSeg + ODISE | 只需 ODISE |
| 配置文件 | 需要 YAML | 不需要 |
| 路径指定 | 通过配置文件 | 命令行参数 |
| 错误处理 | 基本 | 增强 |
| 进度显示 | tqdm | tqdm + 统计 |

## 📝 完整参数列表

```bash
python scripts/precompute_odise_features.py --help
```

**必需参数：**
- `--data-root`: 数据根目录
- `--output-dir`: 输出目录

**可选参数：**
- `--odise-model-config`: 模型配置（默认：Panoptic/odise_caption_coco_50e.py）
- `--label-sets`: 标签集（默认：COCO ADE LVIS SCANNET_20）
- `--max-scenes`: 最大场景数（默认：-1，全部）
- `--max-images-per-scene`: 每场景最大图片数（默认：-1，全部）
- `--skip-existing`: 跳过已存在的文件
- `--device`: 设备（默认：cuda）
- `--scene-pattern`: 场景匹配模式（默认：scene*）

## 💡 性能建议

1. **GPU 推理比 CPU 快 10-50 倍**，强烈推荐使用 GPU
2. **使用 `--skip-existing`** 可以安全地中断和继续
3. **批量处理**：可以同时运行多个实例处理不同场景
4. **存储空间**：每张图约 1-5 MB，1000 张图约 1-5 GB

## 📚 更多信息

查看详细文档：
```bash
cat scripts/README_odise.md
```

---

**需要帮助？**
- 查看错误信息并根据提示修复
- 先用 `--max-scenes 1` 测试
- 确认环境 `conda activate f_bak`
- 检查数据路径是否正确
