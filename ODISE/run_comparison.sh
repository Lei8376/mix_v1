#!/bin/bash
# 完整对比测试：3种配置 + 对比可视化

set -e

echo "=========================================================================="
echo "ODISE 配置对比测试"
echo "=========================================================================="
echo ""
echo "配置1: COCO + ADE + LVIS + SCANNET_20"
echo "配置2: SCANNET_200 (仅)"
echo "配置3: SCANNET_200 + SCANNET_20"
echo ""

# 配置变量
IMAGE_DIR="/home/sunl/work/scene0000_00/color"
NUM_SAMPLES=10

# 步骤1: 运行配置1
echo "=========================================="
echo "步骤 1/4: 配置1 - COCO+ADE+LVIS+SCANNET_20"
echo "=========================================="
python precompute_features.py --config ../config/compare_config1.yaml
echo ""

# 步骤2: 运行配置2
echo "=========================================="
echo "步骤 2/4: 配置2 - SCANNET_200"
echo "=========================================="
python precompute_features.py --config ../config/compare_config2.yaml
echo ""

# 步骤3: 运行配置3
echo "=========================================="
echo "步骤 3/4: 配置3 - SCANNET_200+SCANNET_20"
echo "=========================================="
python precompute_features.py --config ../config/compare_config3.yaml
echo ""

# 步骤4: 对比可视化
echo "=========================================="
echo "步骤 4/4: 对比可视化"
echo "=========================================="
python visualize_compare.py \
    --image-dir ${IMAGE_DIR} \
    --npz-dirs \
        ./compare_output/config1_coco_ade_lvis_s20/scene0000_00 \
        ./compare_output/config2_s200_only/scene0000_00 \
        ./compare_output/config3_s200_s20/scene0000_00 \
    --config-names \
        "COCO+ADE+LVIS+S20" \
        "SCANNET_200" \
        "S200+S20" \
    --output-dir ./compare_output/visualizations \
    --num-samples ${NUM_SAMPLES} \
    --min-score 0.15

echo ""
echo "=========================================================================="
echo "✅ 完成！"
echo "=========================================================================="
echo "对比可视化结果: ./compare_output/visualizations/"
echo ""
echo "每张图片包含3个子图，从左到右分别是："
echo "  1. COCO+ADE+LVIS+SCANNET_20"
echo "  2. SCANNET_200 (仅)"
echo "  3. SCANNET_200+SCANNET_20"
echo "=========================================================================="
