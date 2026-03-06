# 运行 precompute_2d_features.py 指南

## 前置要求

### 1. 下载 LSeg 检查点

LSeg 模型检查点需要手动下载：

```bash
# 创建检查点目录
mkdir -p lang_seg/checkpoints

# 下载检查点（示例链接，需要替换为实际下载地址）
# LSeg 官方仓库：https://github.com/isl-org/lang-seg
# 下载 demo_e200.ckpt 并放到 lang_seg/checkpoints/ 目录
```

可能的下载来源：
- LSeg 官方 GitHub releases
- Google Drive 或其他分享链接
- 项目团队提供的模型文件

### 2. 准备数据配置文件

创建一个 YAML 配置文件，指向你的 ScanNet 数据：

```yaml
DATA:
  data_root_2d: "/path/to/your/scannet/scans"
```

数据目录结构应该是：
```
/path/to/your/scannet/scans/
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

### 3. 检查依赖

确保安装了所有必需的 Python 包：

```bash
cd /home/sunl/work/mix
pip install -r lang_seg/requirements.txt
# 可能还需要安装 ODISE 的依赖
```

## 运行步骤

### 快速测试运行

首先用较小的数据集测试是否能正常运行：

```bash
cd /home/sunl/work/mix

python scripts/precompute_2d_features.py \
    --data-config-path /path/to/your/config.yaml \
    --output-dir ./output_test \
    --label-path lang_seg/label_files/ade20k_objectInfo150.txt \
    --lseg-ckpt-path lang_seg/checkpoints/demo_e200.ckpt \
    --odise-model-config-path Panoptic/odise_caption_coco_50e.py \
    --max-scenes 1 \
    --max-images-per-scene 5
```

### 只运行某一个模型

如果某个模型有问题，可以跳过它：

```bash
# 只运行 LSeg，跳过 ODISE
python scripts/precompute_2d_features.py \
    --data-config-path /path/to/your/config.yaml \
    --output-dir ./output_test \
    --label-path lang_seg/label_files/ade20k_objectInfo150.txt \
    --lseg-ckpt-path lang_seg/checkpoints/demo_e200.ckpt \
    --skip-odise \
    --max-scenes 1

# 只运行 ODISE，跳过 LSeg
python scripts/precompute_2d_features.py \
    --data-config-path /path/to/your/config.yaml \
    --output-dir ./output_test \
    --odise-model-config-path Panoptic/odise_caption_coco_50e.py \
    --skip-lseg \
    --max-scenes 1
```

### 完整运行

测试成功后，可以处理所有数据：

```bash
python scripts/precompute_2d_features.py \
    --data-config-path /path/to/your/config.yaml \
    --output-dir /path/to/output/precomputed_2d \
    --label-path lang_seg/label_files/ade20k_objectInfo150.txt \
    --lseg-ckpt-path lang_seg/checkpoints/demo_e200.ckpt \
    --odise-model-config-path Panoptic/odise_caption_coco_50e.py
```

## 输出结果

脚本会在 `--output-dir` 中创建以下结构：

```
output_dir/
  scene0000_00/
    0_lseg.npy          # LSeg 像素嵌入 (H, W, 512) float16
    0_odise.npz         # ODISE 结果：
                        #   - masks: (K, H, W) bool
                        #   - mask_embeddings: (K, 256) float16
                        #   - num_masks: int64
                        #   - info: object array
    1_lseg.npy
    1_odise.npz
    ...
```

## 常见问题

### 1. 找不到 LSeg 检查点
```
FileNotFoundError: lang_seg/checkpoints/demo_e200.ckpt
```
解决：下载 LSeg 检查点文件并放到正确位置

### 2. 找不到数据配置文件
```
FileNotFoundError: /path/to/config.yaml
```
解决：创建数据配置 YAML 文件，指向你的 ScanNet 数据

### 3. 没有找到场景
```
Found 0 scenes
```
解决：检查 `data_root_2d` 路径是否正确，确保包含 `scene*` 目录

### 4. CUDA 内存不足
```
RuntimeError: CUDA out of memory
```
解决：
- 使用 `--max-images-per-scene` 限制每次处理的图片数量
- 分批处理不同的场景
- 使用更小的输入图像尺寸

### 5. 模块导入错误
```
ModuleNotFoundError: No module named 'xxx'
```
解决：安装缺失的依赖包

## 性能提示

1. **GPU 使用**：脚本会自动使用 CUDA（如果可用）
2. **批量处理**：脚本会跳过已经存在的输出文件
3. **增量运行**：可以多次运行脚本来处理更多数据
