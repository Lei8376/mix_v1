#!/bin/bash
# Mask Distillation 训练启动脚本（Diff2Scene 方案）
# 使用方式: bash experiment_mask_distill/start_mask_distill_train.sh

set -e
cd "$(dirname "$0")/.."   # 切到项目根目录

echo "=========================================="
echo "  Mask Distillation 训练 (experiment_mask_distill)"
echo "=========================================="
echo "  主 loss : mask-level cosine distillation"
echo "            L = (1/K) * sum_k [1 - cos(B_k', B_k)]"
echo "  辅助 loss: 无（纯 mask distillation）"
echo "  评估指标 : 语义 mIoU (20类) + Mask-level mIoU"
echo "  数据量   : 30% (快速验证)"
echo "  checkpoint: checkpoints/mask_distill.1/"
echo "  日志      : runs/mask_distill.1/"
echo "=========================================="

sleep 2

# ---- 单卡启动 ----
python train_open_vocab_v2_ddp.py \
    --config experiment_mask_distill/train_mask_distill.yaml \
    --use-mask-distill

# ---- 多卡启动（取消注释使用）----
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
#     train_open_vocab_v2_ddp.py \
#     --config experiment_mask_distill/train_mask_distill.yaml \
#     --use-mask-distill
