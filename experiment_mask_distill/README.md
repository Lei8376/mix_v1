# experiment_mask_distill — Diff2Scene Mask Distillation

## 方案说明

本实验目录实现论文 **Diff2Scene**（Zhu et al. 2025）中的 **3D Mask Distillation** 方案，用于替代原先的 point-level feature distillation（experiment_distill）和 BCE+Dice（原版）。

### 核心公式（论文 Eq.1 & Eq.2）

```
S_k = <F^{3d}, f_k^{2d}>               # 3D 特征与 2D mask token 的点积
B_k^{3d'} = sigmoid(S_k)               # 预测 3D 概率 mask
B_k^{3d}  = lifted 2D soft mask        # 伪 GT 3D mask（直接取 2D mask 概率值）

L = (1/K) * sum_k [1 - cos(B_k^{3d'}, B_k^{3d})]
```

### 与旧版的区别

| 版本 | 主损失监督对象 | 损失类型 |
|------|--------------|---------|
| 原版 | 每个点 × 每个 mask 的二值 pair | BCE + Dice |
| experiment_distill | 每个点的 feature 向量 | cosine 特征蒸馏 |
| **experiment_mask_distill** | **整张 3D mask 向量** | **mask-level cosine** |

### 变量对应关系

- `f_k^{2d}` → `fused_embeddings[k]`（hybrid 2D token）
- `F^{3d}`   → `pred_3d`（3D backbone 输出特征）
- `B_k^{3d}` → 由 `x_label/y_label + mask_masks` lifted 出来的 soft 3D mask
- `B_k^{3d'}` → `sigmoid(pred_3d @ fused_embeddings[k].T)`

---

## 文件结构

```
experiment_mask_distill/
├── __init__.py
├── criterion_mask_distill.py   # 主损失：MaskDistillCriteria
├── trainer_mask_distill.py     # 训练器：MaskDistillTrainer
├── semantic_miou.py            # 评估：SemanticMIoUTracker + MaskMIoUTracker
├── train_mask_distill.yaml     # 训练配置
├── start_mask_distill_train.sh # 启动脚本
└── README.md                   # 本文件
```

---

## 快速启动

```bash
# 单卡训练
bash experiment_mask_distill/start_mask_distill_train.sh

# 或者手动调用
python train_open_vocab_v2_ddp.py \
    --config experiment_mask_distill/train_mask_distill.yaml \
    --use-mask-distill
```

---

## 评估指标

验证时同时计算：

1. **语义 mIoU**（主指标）：用 CLIP 文本特征对 pred_3d 做 argmax 分类，与 GT nyu40 label 对比
2. **Mask-level mIoU**：对每个预测 3D mask 与 lifted GT 3D mask 计算 IoU 后取均值

---

## 配置说明

`train_mask_distill.yaml` 中关键字段：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `mask_distill_weight` | 1.0 | L_mask_distill 主损失权重 |
| `bce_weight` | 0.0 | 辅助 BCE（0 = 不使用） |
| `dice_weight` | 0.0 | 辅助 Dice（0 = 不使用） |
| `min_points_per_mask` | 10 | GT mask 最少正样本点数 |

如需加辅助稳定项，可将 `bce_weight` 改为 `0.1`：

```yaml
bce_weight: 0.1
dice_weight: 0.05
```

---

## 日志和 checkpoint

- TensorBoard 日志：`runs/mask_distill.1/`
- checkpoint：`checkpoints/mask_distill.1/`
- 最优模型：`checkpoints/mask_distill.1/best_model.pth`
