# Mix 模型评估脚本修复记录

## 问题描述

原始评估脚本 `eval_model_simple.py` 在运行时遇到数据格式问题：

1. **初始错误**: `unhashable type: 'slice'` - batch 数据解包方式不正确
2. **模块导入问题**: `dataset` 模块名冲突（mix和openscene都有dataset目录）
3. **模型参数不匹配**: checkpoint是MinkUNet34C训练的，但脚本默认是MinkUNet18A

## 修复过程

### 1. 理解训练代码

参考用户提供的训练命令：
```bash
train_open_vocab_v2_ddp.py --config config/train_scannet_v2_full_multi_gpu.yaml
```

关键发现：
- 训练使用 `open_vocab_collate_v2` 返回字典格式batch
- 模型 `forward` 需要特定字段: `sinput`, `ori_coords_3d`, `pixel_pooled`, `masks`, `mask_embeddings`, `mask_valid`
- 模型输出中包含 `pred_3d` 字段（3D点的特征）

### 2. 修复数据处理

**之前（错误）**:
```python
coords, feat, label, feat_3d, mask, inds_reverse = batch[:6]
```

**之后（正确）**:
```python
# batch 是字典
coords = batch["coords_3d"]  
feat = torch.zeros(coords.shape[0], 3)  # 占位特征
label = batch["binary_label_3d"]  
inds_reverse = batch["inds_reconstruct"]
```

### 3. 修复模型推理

**之前**:
- 尝试直接使用预计算的feat_3d
- 导致slice indexing错误

**之后**:
```python
# 构建完整batch输入
batch_input = {
    "sinput": sinput,
    "coords_3d": coords.cuda(),
    "ori_coords_3d": batch["ori_coords_3d"].cuda(),  # 🔥 必需
    "pixel_pooled": batch["pixel_pooled"].cuda(),
    "masks": batch["masks"].cuda(),
    "mask_embeddings": batch["mask_embeddings"].cuda(),
    "mask_valid": batch["mask_valid"].cuda(),
}

# 模型推理
results = model(batch_input)

# 使用pred_3d字段
predictions = results["pred_3d"]  # (N_total, feat_dim)
```

### 4. 修复开放词汇匹配

```python
# 计算与CLIP文本特征的相似度
pred = predictions.half() @ text_features.t()  # (N_total, 20)
logits_pred = torch.max(pred, 1)[1].cpu()  # 预测类别
```

## 最终脚本特点

1. **完全对齐训练流程**: 使用相同的数据加载和处理方式
2. **正确的batch格式**: 从字典中提取所有必需字段
3. **完整的模型输入**: 包含所有模型forward需要的参数
4. **开放词汇评估**: 使用CLIP文本特征进行语义匹配
5. **动态架构推断**: 从checkpoint自动识别MinkUNet34C/18A

## 运行方式

```bash
cd /home/featurize/work/mix
source /home/featurize/work/envs/mix_backup/bin/activate

# 评估单个checkpoint
python eval_model_simple.py \
  --checkpoint checkpoints/full.4/checkpoint_epoch_36.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --batch-size 1 \
  --num-workers 0
```

## 关键配置参数

- `checkpoint`: 模型权重文件路径
- `config`: 训练时使用的YAML配置（用于数据集设置）
- `batch-size`: 评估batch大小（推荐=1避免OOM）
- `num-workers`: DataLoader worker数（=0避免多进程问题）

## 输出结果

脚本会输出：
1. **整体mIoU**: 所有类别的平均IoU
2. **各类别IoU**: 包括floor, wall, cabinet等20个类别
3. **训练时最佳指标**: 从checkpoint中提取的best_iou和best_loss

## 注意事项

1. **环境激活**: 必须先激活 `mix_backup` 环境
2. **GPU内存**: batch_size=1 适合单GPU评估
3. **预计算特征**: 脚本会自动使用预计算的2D投影特征
4. **CLIP模型**: 使用 ViT-L/14@336px 提取文本特征

## 修复时间

- 开始时间: 2026-03-06 06:23
- 完成时间: 2026-03-06 06:28 (约5分钟)
- 主要问题: batch数据格式理解和模型输入构建
