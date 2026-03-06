#!/bin/bash
# 🚀 重新训练脚本 - 应用所有 Bug 修复后
# 生成时间: 2026-02-21

set -e  # 遇到错误立即退出

# ============================================================
# 环境变量设置
# ============================================================
echo "⚙️  设置环境变量..."
export OMP_NUM_THREADS=20  # OpenMP 线程数（推荐 20，避免 CPU 性能浪费）
export CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 1

echo "  - OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# ============================================================
# 切换到项目目录
# ============================================================
cd /home/featurize/work/mix

# ============================================================
# 创建目录
# ============================================================
echo "📁 创建 checkpoint 和日志目录..."
mkdir -p checkpoints/full.2
mkdir -p runs/full.2
echo ""

# ============================================================
# 显示配置信息
# ============================================================
echo "📋 训练配置:"
echo "  - 配置文件: config/train_scannet_v2_full_multi_gpu.yaml"
echo "  - GPU 数量: 2"
echo "  - Batch size: 32 (per GPU, 总计 64)"
echo "  - Base LR: 2e-5 (降低 60% 防止过拟合)"
echo "  - Weight Decay: 0.001 (增加 10 倍防止过拟合)"
echo "  - Early Stopping: 5 epochs"
echo "  - 保存间隔: 每 5 epochs"
echo "  - 验证间隔: 每 1 epoch"
echo ""

# ============================================================
# 训练命令（多卡 DDP）
# ============================================================
echo "🚀 开始训练..."
echo "═══════════════════════════════════════════════════════"
echo ""

torchrun --nproc_per_node=2 \
    train_open_vocab_v2_ddp.py \
    --config config/train_scannet_v2_full_multi_gpu.yaml

# ============================================================
# 训练完成
# ============================================================
echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ 训练完成！"
echo ""
echo "📊 查看结果:"
echo "  - Checkpoints: checkpoints/full.2/"
echo "  - TensorBoard: tensorboard --logdir runs/full.2 --port 6006"
echo "  - 最佳模型: checkpoints/full.2/best_model.pth"
echo ""
echo "🔍 检查修复效果:"
echo "  1. mIoU 是否从 0.20 提升？"
echo "  2. 验证损失是否稳定（不上升）？"
echo "  3. Checkpoint 是否只在 epoch 5, 10, 15... 保存？"
echo "  4. 是否看到 '🎯 New best mIoU' 输出？"
echo ""
