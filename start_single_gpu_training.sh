#!/bin/bash
# 单卡训练启动脚本

set -e

cd /home/sunl/work/mix

echo "=========================================="
echo "Mix 项目 - 单卡训练"
echo "=========================================="
echo ""

# 检查 GPU
echo "检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
    echo ""
else
    echo "⚠️  警告: 未找到 nvidia-smi，无法检查 GPU"
    echo ""
fi

# 显示配置
echo "训练配置:"
echo "  - 配置文件: config/train_scannet_v2_full_multi_gpu.yaml"
echo "  - Batch Size: 8 (单卡优化)"
echo "  - 数据量: 100% (1201个场景)"
echo "  - Checkpoint: checkpoints/full.4/"
echo "  - 日志: runs/full.4/"
echo ""

# 询问是否继续
read -p "开始训练? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

# 开始训练
python train_open_vocab_v2_ddp.py \
    --config config/train_scannet_v2_full_multi_gpu.yaml

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  - Checkpoints: ls -lh checkpoints/full.4/"
echo "  - TensorBoard: tensorboard --logdir runs/full.4"
echo ""
