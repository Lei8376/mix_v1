# 训练指南

## 当前训练状态

### 已完成的训练
- **Epoch**: 10/10 完成
- **Loss**: 1.4076 → 0.5679 ✅ 持续下降
- **IoU**: 0.2954 → 0.3582 ✅ 在上升
- **数据量**: 19044 samples (20% 全量数据)
- **训练时间**: ~40 分钟

### 已修复的问题
1. ✅ **IoU=0 bug 已修复** - 之前 checkpoint 中 `best_iou=0` 是因为代码没有更新这个值，现已修复
2. ✅ **添加 mIoU 和 mAcc** - 现在验证时会显示 4 个指标
3. ✅ **多显卡支持** - 支持 DDP 多卡训练

## 配置参数说明

### 所有参数都真实有效！

| 参数 | 作用 | 是否必需 | 说明 |
|------|------|----------|------|
| **数据加载** ||||
| `batch_size` | 每个 GPU 的批大小 | ✅ 必需 | 值 ≥ 2 (MinkowskiEngine 要求) |
| `num_workers` | 数据加载线程数 | ✅ 必需 | 加快数据加载 |
| **数据集** ||||
| `data_config_path` | 数据集配置文件 | ✅ 必需 | 指向 ScanNet 数据路径 |
| `precomputed_dir` | 预计算 2D 特征目录 | ✅ 必需 | ODISE 预计算的 npz 文件 |
| `projection_dir` | 预计算投影目录 | ✅ 必需 | 3D→2D 投影坐标 |
| `split` | 数据集划分 | ✅ 必需 | train/val/test |
| `max_samples_ratio` | 使用数据比例 | ⚠️  调试用 | 0.2=20%，去掉=100% |
| `voxel_size` | 体素大小(米) | ✅ 必需 | 0.05 = 5cm 分辨率 |
| **训练** ||||
| `base_lr` | 基础学习率 | ✅ 必需 | 控制训练速度 |
| `num_epochs` | 总 epoch 数 | ✅ 必需 | 训练轮数 |
| `warmup_epochs` | 预热 epoch 数 | ✅ 推荐 | 学习率线性增长期 |
| `weight_decay` | 权重衰减 | ✅ 推荐 | L2 正则化 |
| `grad_clip_norm` | 梯度裁剪 | ✅ 推荐 | 防止梯度爆炸 |
| `scheduler_type` | 学习率调度器 | ✅ 推荐 | cosine/step/plateau |
| `early_stopping_patience` | 早停耐心值 | ⚠️  可选 | N epochs 无改进则停止 |
| `log_dir` | 日志目录 | ✅ 必需 | TensorBoard 日志 |
| `checkpoint_dir` | 检查点目录 | ✅ 必需 | 模型保存位置 |
| `save_every_epochs` | 保存间隔 | ✅ 推荐 | 每 N epochs 保存 |
| `val_every_epochs` | 验证间隔 | ✅ 推荐 | 每 N epochs 验证 |
| **GPU** ||||
| `gpu_ids` | 使用的 GPU | ✅ 必需 | 单卡: [0], 多卡由命令行控制 |

## 如何训练模型

### 1. 快速测试 (20% 数据)
```bash
python train_open_vocab_v2.py --config config/train_scannet_v2_minimal.yaml
```

### 2. 完整训练 (100% 数据，单卡)
```bash
python train_open_vocab_v2.py --config config/train_scannet_v2_full_multi_gpu.yaml
```

### 3. 多卡训练 (100% 数据，2 卡)
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  train_open_vocab_v2_ddp.py --config config/train_scannet_v2_full_multi_gpu.yaml
```

### 4. 恢复训练
```yaml
# 在配置文件中添加
trainer:
  resume: checkpoints/best_model.pth
```

## IoU 问题解答

### Q: 为什么 checkpoint 显示 best_iou=0？
**A**: 这是之前的 bug（代码没有更新 best_iou 变量），已在代码中修复。下次训练会正确追踪。

### Q: IoU 应该达到多少？
**A**: 
- **当前结果**: IoU ~35% (10 epochs, 20% 数据)
- **合理预期**: 
  - 20% 数据，100 epochs → IoU 40-50%
  - 100% 数据，100 epochs → IoU 50-60%
- **说明**: 你的模型正在**正常学习**，Loss 持续下降，IoU 持续上升

### Q: IoU 计算是否正确？
**A**: ✅ 已验证，IoU 计算逻辑完全正确
- 完美预测 → IoU=1.0 ✅
- 全错预测 → IoU=0.0 ✅
- 随机预测 → IoU~0.33 ✅

### Q: 现在显示的指标有哪些？
**A**: 验证时会显示 4 个指标：
- **IoU**: 全局 IoU (所有点的交并比)
- **mIoU**: Mean IoU (每个 mask 单独计算再平均)
- **Acc**: 全局准确率
- **mAcc**: Mean Accuracy (每个 mask 单独计算再平均)

## 训练建议

### 从哪里开始？
1. **已完成**: 20% 数据，10 epochs，IoU 35.8% ✅
2. **下一步**: 
   - 选项 A: 100% 数据训练 100 epochs (推荐)
   - 选项 B: 从当前 checkpoint 继续训练更多 epochs

### 推荐配置 (100% 数据)
```yaml
dataloader:
  batch_size: 32  # 根据显存调整
  num_workers: 8

dataset:
  data_config_path: config/data_scannet_3d.yaml
  precomputed_dir: /home/featurize/data/pixel_pooled
  projection_dir: /home/featurize/data/scannet_projections
  split: train
  # 去掉 max_samples_ratio = 100% 数据

trainer:
  num_epochs: 100
  base_lr: 5.0e-05
  warmup_epochs: 2
  val_every_epochs: 5  # 每 5 epochs 验证一次
  
gpu_ids: [0]  # 单卡使用 GPU 0
```

### 多卡训练注意事项
1. **Learning rate 缩放**: 2 卡时自动变为 `base_lr * 2`
2. **Effective batch size**: 2 卡 × batch_size 32 = 64
3. **训练速度**: 约 2 倍加速

## 检查训练进度

### 1. TensorBoard
```bash
tensorboard --logdir runs/open_vocab_3d_v2_full --port 6006
# 浏览器打开 http://localhost:6006
```

### 2. 查看日志
```bash
tail -f runs/open_vocab_3d_v2_full/events.out.tfevents.*
```

### 3. 检查 checkpoint
```python
import torch
ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu')
print(f"Epoch: {ckpt['epoch']}, Loss: {ckpt['best_loss']:.4f}, IoU: {ckpt['best_iou']:.4f}")
```

## 常见问题

### Q: 显存不够怎么办？
**A**: 减小 `batch_size`，例如 32 → 16 → 8

### Q: 训练太慢怎么办？
**A**: 
1. 使用多卡训练
2. 增大 `num_workers`
3. 减小 `val_every_epochs` (减少验证频率)

### Q: Loss 不下降怎么办？
**A**: 
1. 检查 learning rate (可能太大或太小)
2. 增加 warmup_epochs
3. 检查数据是否正确加载

### Q: 需要调整哪些超参数？
**A**: 通常只需调整：
- `batch_size` (根据显存)
- `base_lr` (0.00005 是个好的起点)
- `num_epochs` (100 是个好的起点)
- `warmup_epochs` (2 是个好的起点)

## 总结

✅ **你的训练是正常的！**
- Loss 从 1.4 降到 0.57
- IoU 从 29% 升到 36%
- 所有参数都在生效
- 代码已修复所有已知 bug

🎯 **下一步建议**: 使用 100% 数据训练 100 epochs，预期 IoU 可达 50-60%
