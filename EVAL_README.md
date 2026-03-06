# Mix 模型评估脚本使用说明

## 概述

`eval_mix_model.py` 是一个完整的评估脚本，用于测试 Mix 项目训练的开放词汇（Open Vocabulary）3D分割模型。

该脚本参考了 OpenScene 的评估方式，可以：
- 加载训练好的模型权重
- 使用 CLIP 提取文本特征（开放词汇的关键）
- 在验证集上评估模型性能
- 输出总体 mIoU 和各个类别的 IoU（如 floor, wall, chair 等）

## 环境准备

```bash
# 激活环境
source /home/featurize/work/envs/mix_backup/bin/activate

# 确保在 mix 目录下
cd /home/featurize/work/mix
```

## 基本用法

### 1. 最简单的用法（使用默认配置）

```bash
python eval_mix_model.py --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth
```

### 2. 指定数据集路径

```bash
python eval_mix_model.py \
    --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth \
    --data-root /home/featurize/data/scannet_3d \
    --precomputed-dir /home/featurize/data/pixel_pooled
```

### 3. 多次重复评估（减少体素化随机性影响）

```bash
python eval_mix_model.py \
    --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth \
    --test-repeats 3
```

### 4. 评估多个checkpoint

```bash
# Epoch 10
python eval_mix_model.py --checkpoint checkpoints/full.1/checkpoint_epoch_10.pth

# Epoch 19
python eval_mix_model.py --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth

# Epoch 22
python eval_mix_model.py --checkpoint checkpoints/full.4/checkpoint_epoch_22.pth
```

## 输出说明

脚本会输出：

1. **每个类别的 IoU**（参考 OpenScene 格式）：
   ```
   classes          IoU
   ----------------------------
   wall          : 0.756   (123456/234567)
   floor         : 0.823   (234567/345678)
   cabinet       : 0.654   (  3456/  5678)
   bed           : 0.712   ( 12345/ 17890)
   chair         : 0.689   ( 23456/ 34567)
   ...
   ```

2. **总体 mIoU**：
   ```
   Mean IoU: 0.6834
   Mean Acc: 0.7512
   ```

## 参数说明

### 必需参数
- `--checkpoint`: 模型权重文件路径

### 数据集参数
- `--data-root`: 数据集根目录（默认：/home/featurize/data/scannet_3d）
- `--precomputed-dir`: 预计算特征目录（默认：/home/featurize/data/pixel_pooled）
- `--split`: 数据集划分，train/val（默认：val）

### 评估参数
- `--test-repeats`: 重复评估次数，用于减少体素化随机性（默认：1）
- `--test-batch-size`: 评估batch size（默认：1）
- `--dataset-name`: 数据集名称，用于metric计算（默认：scannet_3d）

### 模型参数
- `--pc-arch`: 3D backbone架构（默认：MinkUNet18A）
- `--voxel-size`: 体素大小（默认：0.05）

## 工作原理

### 开放词汇（Open Vocabulary）评估流程

1. **加载模型权重**
   - 从checkpoint加载训练好的3D分割模型

2. **提取文本特征**（关键！）
   - 使用 CLIP 将类别名称（如 "wall", "floor", "chair"）编码为特征向量
   - 这是开放词汇的核心：模型不需要预先训练固定的类别，而是通过文本-视觉对齐进行匹配

3. **3D特征提取**
   - 对每个3D点云场景，提取点级别的特征
   - 使用 MinkowskiNet 等3D backbone

4. **开放词汇匹配**
   - 计算3D特征与文本特征的相似度（点积）
   - 为每个3D点分配最相似的类别

5. **计算 mIoU**
   - 使用 OpenScene 的 metric 模块
   - 计算每个类别的 IoU
   - 计算平均 mIoU

## 常见问题

### Q1: 找不到数据集路径
```bash
# 检查数据集是否存在
ls /home/featurize/data/scannet_3d/val/

# 如果路径不对，使用 --data-root 指定正确路径
python eval_mix_model.py --checkpoint xxx.pth --data-root /your/path/to/scannet_3d
```

### Q2: 缺少预计算特征
如果没有预计算特征，脚本会警告但仍会尝试运行（会很慢）。

建议先运行特征预计算脚本生成特征。

### Q3: CUDA out of memory
```bash
# 减小 batch size
python eval_mix_model.py --checkpoint xxx.pth --test-batch-size 1
```

### Q4: 想看哪个epoch的结果最好
```bash
# 批量评估所有epoch
for epoch in 1 10 19; do
    echo "=== Evaluating epoch $epoch ==="
    python eval_mix_model.py --checkpoint checkpoints/full.1/checkpoint_epoch_${epoch}.pth
done
```

## 输出示例

```
============================================================
Mix 开放词汇3D模型评估
============================================================
权重: checkpoints/full.1/checkpoint_epoch_19.pth
数据集: scannet_3d
划分: val
GPU: [0]
============================================================

=> 加载权重 'checkpoints/full.1/checkpoint_epoch_19.pth'
=> 成功加载权重 (epoch 19)

提取 20 个类别的文本特征...
✓ 文本特征维度: torch.Size([20, 768])

创建数据集...
✓ 数据集创建完成: 312 个样本

============================================================
开始评估...
类别数: 21
测试重复次数: 1
============================================================

评估类别:
   0: wall
   1: floor
   2: cabinet
   3: bed
   ...

轮次 1: 100%|████████████████| 312/312 [05:23<00:00,  1.04s/it]

收集了 15234567 个点的预测结果

============================================================
评估轮次 1 结果:
============================================================

classes          IoU
----------------------------
wall          : 0.756   (123456/234567)
floor         : 0.823   (234567/345678)
cabinet       : 0.654   (  3456/  5678)
...
Mean IoU: 0.6834
Mean Acc: 0.7512

============================================================
最终结果:
  权重: checkpoint_epoch_19.pth
  Epoch: 19
  Mean IoU: 0.6834
============================================================
```

## 注意事项

1. **确保环境正确**：必须激活包含 MinkowskiEngine、CLIP 等依赖的环境
2. **数据路径**：确保数据集和预计算特征路径正确
3. **GPU内存**：评估较大模型可能需要较多显存，可以调整 batch size
4. **评估时间**：完整评估可能需要几分钟到十几分钟，取决于数据集大小

## 与训练日志对比

训练日志在 `runs/` 目录下，可以用 TensorBoard 查看：
```bash
tensorboard --logdir runs/
```

评估脚本的 mIoU 可以与训练时的验证 mIoU 对比，确认模型性能。
