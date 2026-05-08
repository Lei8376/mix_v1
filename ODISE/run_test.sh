#!/bin/bash
# 运行 ODISE 特征提取测试脚本

# 设置环境变量
export ODISE_MODEL_ZOO="/home/sunl/work/mix_v1/ODISE/checkpoints"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mix

# 进入目录
cd /home/sunl/work/mix_v1/ODISE

# 运行配置1: COCO+ADE+LVIS+SCANNET_20
echo "运行配置1: COCO+ADE+LVIS+SCANNET_20"
python precompute_features.py --config ../config/compare_config1.yaml

# 运行配置2: SCANNET_20
echo "运行配置2: SCANNET_20"
python precompute_features.py --config ../config/compare_config2.yaml

# 运行配置3: SCANNET_200+SCANNET_20
echo "运行配置3: SCANNET_200+SCANNET_20"
python precompute_features.py --config ../config/compare_config3.yaml

echo "完成！"
