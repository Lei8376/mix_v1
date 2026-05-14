# 2026-05-14 Dual-Space Semantic Probability Fusion

## 目标

验证一种 eval-only 语义读出方案：不再把 LSeg raw512 强行投影到 ODISE-256。3D student 仍然使用当前 `fused_embeddings` 产生 point-mask geometry logits；语义类别概率在 mask/query 层分别从两个原生空间读取，然后融合：

```text
ODISE raw256 @ ODISE text256 -> P_odise
LSeg raw512  @ CLIP text512  -> P_lseg
P_sem = w_odise * P_odise + w_lseg * P_lseg
point_class_prob = sigmoid(pred_mask_logits) @ P_sem
```

该方案不使用：

- `model.fuse_embed.pixel_proj` 作为 LSeg 语义分支
- `model.pixel_sem_proj`
- ridge probe
- `LSeg512 -> ODISE256` 投影

## 改动

新增 eval-only 脚本：

```text
evaluate/eval_dual_space_semantic_fusion.py
```

脚本评估：

- `odise_only_text256`
- `lseg_only_text512`
- `current_fused_text256`
- `dual_space_fixed`
- `dual_space_confidence`

接入 trainer validation：

```text
experiment_mask_distill/trainer_mask_distill.py
```

新增验证指标：

```text
semantic_miou_dual_space_fixed
semantic_macc_dual_space_fixed
semantic_miou_dual_space_confidence
semantic_macc_dual_space_confidence
semantic_miou_odise_only_text256
semantic_miou_lseg_only_text512
semantic_miou_current_fused_text256
```

TensorBoard 新增：

```text
Metrics/Semantic_mIoU_DualSpaceFixed
Metrics/Semantic_mIoU_DualSpaceConfidence
Metrics/Semantic_mIoU_ODISEOnlyText256
Metrics/Semantic_mIoU_LSegOnlyText512
Metrics/Semantic_mIoU_CurrentFusedText256
```

训练配置新增：

```yaml
dual_space_eval: true
dual_space_odise_weight: 0.5
dual_space_lseg_weight: 0.5
dual_space_tau_odise: 0.07
dual_space_tau_lseg: 0.07
dual_space_use_confidence: false
dual_space_conf_min: 0.2
dual_space_conf_max: 0.7
best_monitor: "semantic_miou_dual_space_fixed"
```

## 分支

```text
dual-space-semantic-prob-fusion
```

## 验证命令

语法检查：

```bash
python -m py_compile evaluate/eval_dual_space_semantic_fusion.py
python -m py_compile experiment_mask_distill/trainer_mask_distill.py
```

val20：

```bash
python evaluate/eval_dual_space_semantic_fusion.py \
  --checkpoint checkpoints/diff2scene_hybrid_lseg_odise256_semantic_query/checkpoint_epoch_23.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --batch-size 4 \
  --num-workers 4 \
  --max-samples 20 \
  --tau-odise 0.07 \
  --tau-lseg 0.07 \
  --odise-weight 0.5 \
  --lseg-weight 0.5 \
  --output-json runs/eval_only/epoch23_dual_space_val20.json
```

完整 val：

```bash
python evaluate/eval_dual_space_semantic_fusion.py \
  --checkpoint checkpoints/diff2scene_hybrid_lseg_odise256_semantic_query/checkpoint_epoch_23.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --batch-size 12 \
  --num-workers 8 \
  --tau-odise 0.07 \
  --tau-lseg 0.07 \
  --odise-weight 0.5 \
  --lseg-weight 0.5 \
  --output-json runs/eval_only/epoch23_dual_space_full_val.json
```

可训练命令：

```bash
cd /home/sunl/work/mix_v1
conda activate mix
python train_open_vocab_v2_ddp.py \
  --config config/train_scannet_v2_full_multi_gpu.yaml
```

训练仍然是 mask distillation；dual-space 只接入 validation / best monitor。

## 当前验证结果

语法检查通过：

```bash
/home/sunl/miniconda3/envs/mix/bin/python -m py_compile evaluate/eval_dual_space_semantic_fusion.py
/home/sunl/miniconda3/envs/mix/bin/python -m py_compile experiment_mask_distill/trainer_mask_distill.py
/home/sunl/miniconda3/envs/mix/bin/python -m py_compile train_open_vocab_v2.py train_open_vocab_v2_ddp.py evaluate/eval_mask_distill_checkpoint.py
```

当前工具环境 CUDA 不可用，脚本自动退到 CPU。`num_workers=4` 在该交互环境中长时间无输出，因此 val20 实际验证使用 `num_workers=0`；命令其余参数相同。

val20 结果文件：

```text
runs/eval_only/epoch23_dual_space_val20.json
```

核心结果：

| Method | mIoU | mAcc | n valid |
|---|---:|---:|---:|
| `odise_only_text256` | `0.232077` | `0.422039` | `13` |
| `lseg_only_text512` | `0.312811` | `0.477216` | `11` |
| `current_fused_text256` | `0.238289` | `0.424422` | `14` |
| `dual_space_fixed` (`0.5/0.5`) | `0.285242` | `0.493568` | `12` |
| `dual_space_confidence` | `0.278225` | `0.485776` | `12` |

结论：在 val20 上，`dual_space_fixed` 明显高于 `current_fused_text256`，并且高于较弱的单路 `odise_only_text256`。这支持“不统一维度，改在类别概率层做双空间融合”的第一阶段方案。

## val20 权重扫描

固定 `tau_odise=0.07`、`tau_lseg=0.07`：

| ODISE weight | LSeg weight | dual fixed mIoU | confidence mIoU |
|---:|---:|---:|---:|
| `0.7` | `0.3` | `0.276596` | `0.278225` |
| `0.5` | `0.5` | `0.285242` | `0.278225` |
| `0.3` | `0.7` | `0.329880` | `0.278225` |

观察：当前 val20 子集上加大 LSeg 权重到 `0.7` 最强，超过 LSeg-only 的 `0.312811`。这说明 ODISE 对 LSeg 有补充，但简单固定权重比 entropy confidence 更有效。

## val20 tau_lseg 扫描

固定 `odise_weight=0.5`、`lseg_weight=0.5`、`tau_odise=0.07`：

| tau_lseg | dual fixed mIoU | confidence mIoU |
|---:|---:|---:|
| `0.05` | `0.323553` | `0.317850` |
| `0.07` | `0.285242` | `0.278225` |
| `0.10` | `0.279158` | `0.243533` |
| `0.20` | `0.270536` | `0.241149` |

观察：当前 val20 上 `tau_lseg=0.05` 优于默认 `0.07`，说明 LSeg 分支概率需要更尖锐的类别分布。完整 val 应优先验证：

```text
odise_weight=0.3, lseg_weight=0.7, tau_lseg=0.07
odise_weight=0.5, lseg_weight=0.5, tau_lseg=0.05
```

## Trainer validation smoke

通过 `evaluate/eval_mask_distill_checkpoint.py --max-samples 1` 确认 trainer `_validate()` 接入后可输出 dual-space metrics：

```text
semantic_miou_dual_space_fixed      = 0.211886
semantic_miou_dual_space_confidence = 0.214391
semantic_miou_odise_only_text256    = 0.318083
semantic_miou_lseg_only_text512     = 0.254360
semantic_miou_current_fused_text256 = 0.192994
```

## 后续判断

当前结果满足第一阶段成立条件：

```text
dual_space_fixed > current_fused_text256
dual_space_fixed >= min(odise_only_text256, lseg_only_text512)
```

下一步不是训练 LSeg->ODISE projector，而是在完整 val 上确认权重和温度。如果完整 val 稳定，第二阶段可以考虑 source-aware gate，但需要额外 semantic loss 或 pseudo label，不在本次实现范围内。
