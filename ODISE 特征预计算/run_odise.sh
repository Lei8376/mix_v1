#!/bin/bash

# ====================================================================
# ODISE 特征预计算脚本
# ====================================================================
# 使用方法：
#   1. 修改下面的路径配置
#   2. chmod +x scripts/run_odise.sh
#   3. ./scripts/run_odise.sh
# ====================================================================

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh  # 或 ~/anaconda3/etc/profile.d/conda.sh
conda activate f_bak

# 检查环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "f_bak" ]; then
    echo "ERROR: Failed to activate conda environment 'f_bak'"
    exit 1
fi

echo "Environment activated: $CONDA_DEFAULT_ENV"

# ==================== 路径配置 ====================

# ScanNet 数据根目录（包含 scene0000_00, scene0000_01 等文件夹）
DATA_ROOT="/path/to/your/scannet/scans"

# 输出目录
OUTPUT_DIR="/path/to/output/precomputed_odise"

# ODISE 模型配置
ODISE_MODEL_CONFIG="Panoptic/odise_caption_coco_50e.py"

# ==================== 可选参数 ====================

# 限制处理的场景数量（-1 表示处理所有场景）
MAX_SCENES=-1

# 每个场景最多处理多少张图片（-1 表示处理所有图片）
MAX_IMAGES_PER_SCENE=-1

# 是否跳过已存在的输出文件
SKIP_EXISTING="--skip-existing"

# 设备（cuda 或 cpu）
DEVICE="cuda"

# ==================== 运行脚本 ====================

cd /home/featurize/work/mix2_v1

python scripts/precompute_odise_features.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --odise-model-config "$ODISE_MODEL_CONFIG" \
    --max-scenes "$MAX_SCENES" \
    --max-images-per-scene "$MAX_IMAGES_PER_SCENE" \
    --device "$DEVICE" \
    $SKIP_EXISTING

echo ""
echo "完成！输出保存在: $OUTPUT_DIR"
