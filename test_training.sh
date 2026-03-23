#!/bin/bash
# 快速测试训练脚本（使用 20% 数据）

set -e

cd /home/sunl/work/mix

echo "=========================================="
echo "Mix 项目 - 快速测试（20% 数据）"
echo "=========================================="
echo ""

echo "测试配置:"
echo "  - 配置文件: config/train_scannet_v2_minimal.yaml"
echo "  - 数据量: 20% (~240个场景)"
echo "  - Epochs: 10"
echo "  - 用途: 快速验证配置是否正常"
echo ""

read -p "开始测试训练? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "开始测试训练..."
echo ""

python train_open_vocab_v2_ddp.py \
    --config config/train_scannet_v2_minimal.yaml

echo ""
echo "测试完成！如果没有报错，说明配置正常，可以开始完整训练。"
echo ""
