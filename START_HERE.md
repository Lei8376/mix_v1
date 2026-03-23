# 单卡训练 - 快速开始 🚀

## ✅ 配置已优化为单卡环境

配置文件 `config/train_scannet_v2_full_multi_gpu.yaml` 已针对单卡训练优化：
- Batch Size: 16 → **8**（降低显存占用）
- Num Workers: 12 → **8**（适配单卡负载）
- Resume: 已注释（从头训练）

## 三种启动方式

### 方式 1: 使用便捷脚本（推荐）

```bash
cd /home/sunl/work/mix

# 开始完整训练
bash start_single_gpu_training.sh

# 或者快速测试（20% 数据）
bash test_training.sh
```

### 方式 2: 直接运行 Python

```bash
cd /home/sunl/work/mix

# 使用 DDP 脚本（会自动适配单卡）
python train_open_vocab_v2_ddp.py \
    --config config/train_scannet_v2_full_multi_gpu.yaml

# 或使用单卡专用脚本
python train_open_vocab_v2.py \
    --config config/train_scannet_v2_full_multi_gpu.yaml
```

### 方式 3: 指定 GPU

如果有多张 GPU，可以指定使用哪一张：

```bash
# 使用第 0 张 GPU
CUDA_VISIBLE_DEVICES=0 python train_open_vocab_v2_ddp.py \
    --config config/train_scannet_v2_full_multi_gpu.yaml

# 使用第 1 张 GPU
CUDA_VISIBLE_DEVICES=1 python train_open_vocab_v2_ddp.py \
    --config config/train_scannet_v2_full_multi_gpu.yaml
```

## 显存要求

| Batch Size | 预估显存 | 推荐显卡 |
|-----------|---------|---------|
| 2 | ~8 GB | RTX 3060 12GB |
| 4 | ~12 GB | RTX 3070/3060Ti |
| **8** | ~16 GB | **RTX 3080/4070Ti** |
| 16 | ~24 GB | RTX 3090/4090 |

当前配置使用 **batch_size=8**，推荐 **16GB+ 显存**的 GPU。

### 如果显存不足

编辑 `config/train_scannet_v2_full_multi_gpu.yaml`：

```yaml
dataloader:
  batch_size: 4  # 改为 4 或 2
```

## 训练配置说明

当前配置：
- **数据**: 100% 训练集（1201 个场景）
- **Batch Size**: 8
- **梯度累积**: 2 步（有效 batch = 16）
- **学习率**: 5e-5
- **总轮数**: 100 epochs
- **验证间隔**: 每 2 epochs
- **保存间隔**: 每 2 epochs

## 监控训练

### 实时日志

```bash
# 查看训练输出
tail -f runs/full.4/train.log

# 或直接在终端查看（训练时会自动输出）
```

### TensorBoard

```bash
tensorboard --logdir runs/full.4 --port 6006
```

然后在浏览器打开：`http://localhost:6006`

## 输出位置

**Checkpoints**:
- `checkpoints/full.4/checkpoint_epoch_*.pth` - 定期保存
- `checkpoints/full.4/best_model.pth` - 最佳模型

**日志**:
- `runs/full.4/` - TensorBoard 日志

## 常见问题

### Q: 训练可以中断后恢复吗？

可以！编辑配置文件，取消注释 resume 行：

```yaml
trainer:
  resume: checkpoints/full.4/checkpoint_epoch_XX.pth
```

### Q: 显存 OOM 怎么办？

1. 降低 batch_size（8 → 4 → 2）
2. 或增加梯度累积步数

### Q: 想先快速测试？

使用测试脚本（仅 20% 数据）：

```bash
bash test_training.sh
```

### Q: DDP 脚本和普通脚本有什么区别？

DDP 脚本的优势：
- ✅ 自动适配单卡/多卡环境
- ✅ 代码更新，包含最新优化
- ✅ 将来升级多卡无需修改

推荐使用 `train_open_vocab_v2_ddp.py`！

## 详细文档

查看完整说明：

```bash
cat docs/路径更新/单卡训练说明.md
```

---

**一切就绪！现在可以开始训练了。** 🎉

推荐命令：
```bash
bash start_single_gpu_training.sh
```
