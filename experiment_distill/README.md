# experiment_distill — 新版蒸馏损失实验



---

## 文件说明

| 文件 | 作用 |
|---|---|
| `criterion_distill.py` | 新版损失函数 `DistillCriteria` |
| `trainer_distill.py` | 新版 Trainer `DistillTrainer`，内部使用 `DistillCriteria` |
| `__init__.py` | 包标识 |

---

## 旧方案 vs 新方案对比

| | 旧方案 | 新方案 |
|---|---|---|
| 主监督 | `BCE + Dice` on `pred_mask_logits` | `1 - cos(pred_3d, teacher)` |
| teacher 形式 | 无直接 teacher（用 mask-slot GT） | `fused_embeddings` 加权平均到点级 |
| mask BCE+Dice | 主损失（权重 1.0） | 辅助项（权重 0.1，可调） |
| `fuse_embed` | 可训练 | **仍然可训练**（不冻结） |
| text loss | 无 | **暂不加入**（后续补充） |

---

## 切换步骤

在 `train_open_vocab_v2_ddp.py` 里修改三处：

### 1. import 替换

```python
# 旧
from trainer.open_vocab_trainer_v2 import OpenVocabTrainerV2, OpenVocabTrainerV2Config

# 新
from experiment_distill.trainer_distill import DistillTrainer, DistillTrainerConfig
```

### 2. Config 类替换

```python
# 旧
trainer_config = OpenVocabTrainerV2Config(...)

# 新
trainer_config = DistillTrainerConfig(
    ...
    feat_loss_weight=1.0,   # L_feat 主损失权重
    mask_loss_weight=0.1,   # L_mask 辅助损失权重（原 BCE+Dice 降到 0.1）
)
```

### 3. Trainer 类替换

```python
# 旧
trainer = OpenVocabTrainerV2(model, train_loader, val_loader, config=trainer_config, ...)

# 新
trainer = DistillTrainer(model, train_loader, val_loader, config=trainer_config, ...)
```

---

## 新增的 YAML 配置项

在 `config/train_scannet_v2_full_multi_gpu.yaml` 的 `trainer:` 块里加：

```yaml
trainer:
  feat_loss_weight: 1.0   # 主损失：point-level feature distillation
  mask_loss_weight: 0.1   # 辅助损失：mask BCE+Dice（旧损失降权）
  bce_weight: 1.0
  dice_weight: 1.0
```

其余所有配置项（`base_lr`, `num_epochs`, `checkpoint_dir` 等）和旧版完全相同。

---

## 新版 Loss 原理

### 构造 point-level teacher

对每个 3D 点 `i`（属于 batch `b`）：

1. 找出所有它落入的有效 mask `k`（通过 x_label/y_label 投影到 mask 上）
2. 把对应的 `fused_embeddings[b, k]` 做等权平均
3. 得到 `teacher[i]`

`fused_embeddings` 本身由 `fuse_embed(pixel_pooled, mask_embeddings, masks, mask_valid)` 产生，
`fuse_embed` **保持可训练**，所以 teacher 也会随训练优化。

### 主损失

```
L_feat = mean(1 - cos(pred_3d[i], teacher[i]))   对所有 teacher_valid=True 的点
```

### 辅助损失（旧 BCE+Dice 降权）

```
L_mask = BCE + Dice   （和旧版逻辑完全相同，只是权重从 1.0 降到 0.1）
```

### 总损失

```
L = feat_loss_weight * L_feat + mask_loss_weight * L_mask
```

---

## TensorBoard 新增指标

训练时每步记录：
- `Loss/Train_Feat_Step`：当前 step 的 `L_feat`
- `Loss/Train_Mask_Step`：当前 step 的 `L_mask`
- `Loss/Train_Feat_Epoch`：epoch 平均 `L_feat`
- `Loss/Train_Mask_Epoch`：epoch 平均 `L_mask`

验证时记录：
- `Loss/Val_Feat`：验证集 `L_feat`
- `Loss/Val_Mask`：验证集 `L_mask`

---

## 删除方式

```bash
rm -rf /home/sunl/work/mix/experiment_distill/
```

主项目代码完全不受影响。
