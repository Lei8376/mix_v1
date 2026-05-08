#!/bin/bash
# 批量处理 ScanNet 2D 数据集
# 使用 SCANNET_200 标签集

echo "=========================================="
echo "开始批量处理 ScanNet 2D 数据集"
echo "=========================================="
echo ""

# 设置环境变量
export ODISE_MODEL_ZOO="/home/sunl/work/mix_v1/ODISE/checkpoints"
export OMP_NUM_THREADS=1

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mix

# 进入工作目录
cd /home/sunl/work/mix_v1/ODISE

# 运行批处理
python precompute_features.py --config ../config/odise_config_production.yaml

echo ""
echo "=========================================="
echo "处理完成！"
echo "=========================================="
echo "结果保存在: /home/sunl/work/mix/data/odise_features"
