#!/bin/bash
# Mask Distillation 训练启动脚本（Diff2Scene 方案）
# 使用方式: bash experiment_mask_distill/start_mask_distill_train.sh

set -e
cd "$(dirname "$0")/.."   # 切到项目根目录


# ---- 单卡启动 ----
python train_open_vocab_v2_ddp.py \
    --config experiment_mask_distill/train_mask_distill.yaml \
    --use-mask-distill

# ---- 多卡启动（取消注释使用）----
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
#     train_open_vocab_v2_ddp.py \
#     --config experiment_mask_distill/train_mask_distill.yaml \
#     --use-mask-distill
