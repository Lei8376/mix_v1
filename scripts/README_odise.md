# ODISE 特征预计算使用指南

这是一个简化版的特征预计算脚本，只包含 ODISE 部分，不需要 LSeg。

## 快速开始

### 1. 激活环境

```bash
conda activate f_bak
```

### 2. 测试运行（处理少量数据）

```bash
cd /home/sunl/work/mix

python scripts/precompute_odise_features.py \
    --data-root /path/to/scannet/scans \
    --output-dir ./test_output \
    --max-scenes 1 \
    --max-images-per-scene 5
```

### 3. 完整运行

```bash
python scripts/precompute_odise_features.py \
    --data-root /path/to/scannet/scans \
    --output-dir /path/to/output/precomputed_odise
```

## 命令行参数说明

### 必需参数

- `--data-root`: ScanNet 数据根目录
  - 示例：`/data/scannet/scans`
  - 应包含 `scene0000_00`, `scene0000_01` 等文件夹

- `--output-dir`: 输出目录
  - 示例：`/data/precomputed_odise`

### 可选参数

#### ODISE 模型设置

- `--odise-model-config`: ODISE 模型配置
  - 默认：`Panoptic/odise_caption_coco_50e.py`
  
- `--label-sets`: 使用的标签集
  - 默认：`COCO ADE LVIS SCANNET_20`
  - 示例：`--label-sets COCO ADE`

- `--vocab`: 额外的词汇
  - 格式：用逗号分隔单词，用分号分隔组
  - 示例：`--vocab "table,desk;chair,seat"`

- `--overlap-threshold`: 掩码合并的重叠阈值
  - 默认：0.0

- `--object-mask-threshold`: 保留掩码的分数阈值
  - 默认：0.0

#### 处理选项

- `--scene-pattern`: 场景目录匹配模式
  - 默认：`scene*`

- `--max-scenes`: 最多处理的场景数
  - 默认：-1（处理所有）
  - 测试时建议：1-5

- `--max-images-per-scene`: 每个场景最多处理的图片数
  - 默认：-1（处理所有）
  - 测试时建议：5-10

- `--skip-existing`: 跳过已存在的输出文件
  - 默认：不跳过
  - 添加此选项可以断点续传

- `--device`: 推理设备
  - 默认：`cuda`（如果可用）
  - 可选：`cpu`

## 数据目录结构

### 输入数据结构

脚本支持以下两种目录结构：

**结构 1：标准 ScanNet 格式**
```
data_root/
  scene0000_00/
    color/
      0.jpg
      1.jpg
      ...
  scene0000_01/
    color/
      0.jpg
      ...
```

**结构 2：简化格式**
```
data_root/
  scene0000_00/
    0.jpg
    1.jpg
    ...
  scene0000_01/
    0.jpg
    ...
```

### 输出数据结构

```
output_dir/
  scene0000_00/
    0_odise.npz
    1_odise.npz
    ...
  scene0000_01/
    0_odise.npz
    ...
```

每个 `.npz` 文件包含：
- `masks`: (K, H, W) bool - K 个二值掩码
- `mask_embeddings`: (K, 256) float16 - K 个掩码嵌入
- `num_masks`: int64 - 检测到的掩码数量
- `info`: object array - 每个掩码的详细信息
  - `category_name`: 类别名称
  - `category_id`: 类别 ID
  - `is_thing`: 是否为物体（thing）
  - `score`: 置信度分数
  - `area`: 掩码面积（像素数）

## 使用示例

### 示例 1: 快速测试（1个场景，5张图片）

```bash
python scripts/precompute_odise_features.py \
    --data-root /data/scannet/scans \
    --output-dir ./test_output \
    --max-scenes 1 \
    --max-images-per-scene 5
```

### 示例 2: 处理所有数据

```bash
python scripts/precompute_odise_features.py \
    --data-root /data/scannet/scans \
    --output-dir /data/precomputed_odise
```

### 示例 3: 断点续传（跳过已处理的图片）

```bash
python scripts/precompute_odise_features.py \
    --data-root /data/scannet/scans \
    --output-dir /data/precomputed_odise \
    --skip-existing
```

### 示例 4: 只使用 COCO 标签集

```bash
python scripts/precompute_odise_features.py \
    --data-root /data/scannet/scans \
    --output-dir /data/precomputed_odise \
    --label-sets COCO
```

### 示例 5: CPU 推理（没有 GPU 时）

```bash
python scripts/precompute_odise_features.py \
    --data-root /data/scannet/scans \
    --output-dir /data/precomputed_odise \
    --device cpu \
    --max-scenes 5
```

### 示例 6: 使用 Shell 脚本运行

```bash
# 编辑 scripts/run_odise.sh，修改路径
vim scripts/run_odise.sh

# 添加执行权限
chmod +x scripts/run_odise.sh

# 运行
./scripts/run_odise.sh
```

## 读取预计算的特征

```python
import numpy as np

# 加载特征
data = np.load("output_dir/scene0000_00/0_odise.npz", allow_pickle=True)

# 获取数据
masks = data["masks"]  # (K, H, W) bool
mask_embeddings = data["mask_embeddings"]  # (K, 256) float16
num_masks = data["num_masks"]  # int64
info = data["info"]  # object array

print(f"Number of masks: {num_masks}")
print(f"Masks shape: {masks.shape}")
print(f"Embeddings shape: {mask_embeddings.shape}")

# 查看每个掩码的信息
for i, mask_info in enumerate(info):
    print(f"Mask {i}:")
    print(f"  Category: {mask_info['category_name']}")
    print(f"  Score: {mask_info['score']:.3f}")
    print(f"  Area: {mask_info['area']} pixels")
```

## 性能优化建议

1. **GPU 内存不足时：**
   - 减少 `--max-images-per-scene`
   - 使用 `--device cpu`（会慢很多）

2. **加速处理：**
   - 确保使用 GPU（`--device cuda`）
   - 使用 `--skip-existing` 避免重复处理

3. **断点续传：**
   - 脚本支持中断后继续运行
   - 使用 `--skip-existing` 跳过已处理的文件

4. **批量处理：**
   - 可以同时运行多个脚本实例处理不同的场景
   - 修改 `--scene-pattern` 参数，例如：
     - 实例1: `--scene-pattern "scene0*"`
     - 实例2: `--scene-pattern "scene1*"`

## 故障排除

### 问题 1: 找不到场景

```
Found 0 scenes
```

**解决方法：**
- 检查 `--data-root` 路径是否正确
- 确认目录下有 `scene*` 格式的文件夹
- 尝试使用 `--scene-pattern` 修改匹配模式

### 问题 2: ODISE 加载失败

```
ERROR: Failed to load ODISE extractor
```

**解决方法：**
- 确认已激活正确的 conda 环境：`conda activate f_bak`
- 检查 ODISE 依赖是否安装完整
- 确认 `ODISE/` 目录下的文件完整

### 问题 3: CUDA 错误

```
RuntimeError: CUDA out of memory
```

**解决方法：**
- 使用 `--device cpu`
- 减少并行处理的图片数量
- 关闭其他占用 GPU 的程序

### 问题 4: 模块导入错误

```
ModuleNotFoundError: No module named 'xxx'
```

**解决方法：**
- 确认已激活 `f_bak` 环境
- 安装缺失的包：`pip install xxx`
- 检查 ODISE 目录下的依赖

## 环境要求

- Python 3.7+
- PyTorch with CUDA
- detectron2
- ODISE 相关依赖
- 其他依赖见 `requirements.txt`

## 输出文件大小估算

每张图片的输出文件大小取决于检测到的掩码数量：
- 假设平均每张图片检测到 50 个掩码
- 图像大小 640x480
- 每个文件约 **1-5 MB**（压缩后）

处理 1000 张图片约需要 **1-5 GB** 存储空间。
