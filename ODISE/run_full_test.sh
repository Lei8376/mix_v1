#!/bin/bash
# 完整测试流程：处理所有图片 + 可视化结果

set -e  # 遇到错误立即退出

echo "=========================================================================="
echo "ODISE 完整测试流程"
echo "=========================================================================="
echo ""

# 配置
IMAGE_DIR="/home/sunl/work/scene0000_00/color"
OUTPUT_DIR="./test_output"
VIS_OUTPUT_DIR="./test_output_vis"
CONFIG="../config/odise_config_test.yaml"

# 步骤1: 处理所有图片
echo "步骤 1/2: 处理所有图片..."
echo "----------------------------------------"
python precompute_features.py --config ${CONFIG}
echo ""
echo "✅ 特征提取完成！"
echo ""

# 步骤2: 可视化结果（默认取10张样本）
echo "步骤 2/2: 可视化结果..."
echo "----------------------------------------"
python visualize_results.py \
    --image-dir ${IMAGE_DIR} \
    --npz-dir ${OUTPUT_DIR} \
    --output-dir ${VIS_OUTPUT_DIR} \
    --num-samples 10 \
    --min-score 0.15

echo ""
echo "=========================================================================="
echo "完成！"
echo "=========================================================================="
echo "特征文件目录: ${OUTPUT_DIR}"
echo "可视化结果目录: ${VIS_OUTPUT_DIR}"
echo ""
echo "提示: 如需可视化所有图片，运行："
echo "  python visualize_results.py --image-dir ${IMAGE_DIR} --npz-dir ${OUTPUT_DIR} --output-dir ${VIS_OUTPUT_DIR} --all"
echo "=========================================================================="
