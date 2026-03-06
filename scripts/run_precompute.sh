#!/bin/bash

# 运行 precompute_2d_features.py 的示例脚本
# 根据你的实际路径修改以下变量

# ==================== 需要配置的路径 ====================

# 1. 数据配置文件路径（需要创建，见下方说明）
DATA_CONFIG_PATH="config/scannet/data_config.yaml"

# 2. 输出目录
OUTPUT_DIR="/path/to/output/precomputed_2d"

# 3. LSeg 检查点路径（需要下载，见下方说明）
LSEG_CKPT_PATH="lang_seg/checkpoints/demo_e200.ckpt"

# 4. LSeg 标签文件（已存在）
LABEL_PATH="lang_seg/label_files/ade20k_objectInfo150.txt"

# 5. ODISE 模型配置
ODISE_MODEL_CONFIG="Panoptic/odise_caption_coco_50e.py"

# ==================== 可选参数 ====================

# 限制处理的场景数量（测试时使用）
MAX_SCENES=1

# 每个场景最多处理多少张图片（测试时使用）
MAX_IMAGES_PER_SCENE=5

# 跳过某个模型的特征提取
# SKIP_LSEG="--skip-lseg"
# SKIP_ODISE="--skip-odise"

# ==================== 运行命令 ====================

cd /home/sunl/work/mix

python scripts/precompute_2d_features.py \
    --data-config-path "$DATA_CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --label-path "$LABEL_PATH" \
    --lseg-ckpt-path "$LSEG_CKPT_PATH" \
    --odise-model-config-path "$ODISE_MODEL_CONFIG" \
    --max-scenes "$MAX_SCENES" \
    --max-images-per-scene "$MAX_IMAGES_PER_SCENE"

echo "完成！"
