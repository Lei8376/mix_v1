# 配置文件修改说明

## 修改的文件

`/home/featurize/work/mix/config/train_scannet_v2_full_multi_gpu.yaml`

## 修改内容（抗过拟合优化）

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| **base_lr** | 5e-5 | **2e-5** | 降低 60%，防止过快拟合 |
| **weight_decay** | 0.0001 | **0.001** | 增加 10 倍，增强 L2 正则化 |
| **early_stopping_patience** | 15 | **5** | 更激进早停，及时停止过拟合 |
| **warmup_epochs** | 0 | **1** | 从头训练，需要 warmup |
| **checkpoint_dir** | full.1 | **full.2** | 新目录，避免覆盖 |
| **log_dir** | runs/full.1 | **runs/full.2** | 新日志目录 |
| **resume** | checkpoint_epoch_16 | **null** | 从头训练，不 resume |

## 如何使用

```bash
cd /home/featurize/work/mix

# 直接运行，使用修改后的配置
python train_open_vocab_v2.py --config config/train_scannet_v2_full_multi_gpu.yaml
```

## 预期效果

- ✅ Train Loss 和 Val Loss 差距缩小
- ✅ Val Loss 不再持续上升
- ✅ mIoU 可能提升 2-5%
- ✅ 最佳 epoch 出现在更合理的位置（如 epoch 15-25）

## 如果还是过拟合

可以进一步调整：
- `base_lr: 1.0e-05`（更激进）
- `weight_decay: 0.01`（更强正则化）

## 文件中的标记

- 🔴 表示旧值（已注释）
- 🔥 表示新值（当前使用）
