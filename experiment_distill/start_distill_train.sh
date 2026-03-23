#!/bin/bash
# 新版蒸馏训练启动脚本
# 使用方式: bash experiment_distill/start_distill_train.sh

set -e
cd "$(dirname "$0")/.."   # 切到项目根目录

echo "=========================================="
echo "  新版蒸馏训练 (experiment_distill)"
echo "=========================================="
echo "  主 loss : cosine feature distillation"
echo "  辅助 loss: mask BCE+Dice (weight=0.1)"
echo "  评估指标 : 语义 mIoU (20类)"
echo "  数据量  : 10% (快速验证)"
echo "  checkpoint: checkpoints/distill.1/"
echo "  日志     : runs/distill.1/"
echo "=========================================="

# 等 2 秒让你看清楚再跑
sleep 2

# ---- 单卡启动 ----
python train_open_vocab_v2_ddp.py \
    --config experiment_distill/train_distill.yaml \
    --use-distill   # 见下方说明

# ---- 多卡启动（取消注释使用）----
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
#     train_open_vocab_v2_ddp.py \
#     --config experiment_distill/train_distill.yaml \
#     --use-distill
