# 2026-05-13 mix_v1 项目理解记录

## 项目主线

`mix_v1` 是开放词汇 3D 语义/实例 mask 蒸馏项目。核心目标是把 2D teacher 的 mask token 语义和 mask 几何迁移到 3D 点云 student：

```text
ScanNet 3D points
  + 预计算 3D->2D projection
  + 预计算 ODISE masks / ODISE mask embeddings / LSeg pixel-pooled features
  -> 3D student pred_3d
  -> 通过 2D mask query 生成 3D mask logits
  -> 用 mask/text 概率得到 open-vocabulary semantic mIoU
```

当前主线不是闭集 ScanNet20 分类训练，而是 mask-query 式开放词汇评估：先预测每个 3D 点属于哪些 2D mask，再由 mask token 与文本特征决定类别。

## 数据路径

当前 README 记录的本机路径仍是主路径：

```text
3D ScanNet:          /home/sunl/work/mix/data/scannet_3d
2D ScanNet:          /home/sunl/work/mix/data/scannet_2d
预计算 2D 特征:      /home/sunl/work/mix/data/pixel_pooled
预计算 3D->2D 投影:  /home/sunl/work/mix/data/scannet_projections
```

数据读取入口：

- `dataset/open_vocab_dataset_v2.py`
- `OpenVocabScannetDatasetV2`
- `open_vocab_collate_v2`

训练时必须使用预计算 projection，防止 2D 特征帧和 3D 投影帧错位。`train` split 缺 projection 会直接报错；`val/test` 才允许 fallback 到运行时投影。

## 当前模型

主模型入口：

- `model/open_vocab_fusion_v2.py`
- `OpenVocab3DFusionModelV2`

3D backbone：

- `model/pc_net.py`
- `PC_Processor`
- 默认 `MinkUNet34C`

当前主配置维度：

```yaml
pixel_embedding_dim: 512   # LSeg / pixel_pooled
mask_embedding_dim: 256    # ODISE mask embedding
fused_embedding_dim: 256   # 当前统一回 ODISE 原生 256D 空间
pc_last_dim: 256
```

2D 融合网络：

- `model/modeling.py`
- `ODISEPixelMaskFusionNet`

当前融合公式：

```text
mask_tokens  = mask_proj(mask_embed)
             = raw ODISE 256D                 # 当前 mask_dim == out_dim，所以 Identity

clip_tokens  = pixel_proj(pixel_pooled)
             = LSeg 512D -> ODISE 256D

gate         = sigmoid(gate([mask_tokens, clip_tokens]))

base         = mask_tokens + gate * clip_tokens

delta        = refine(base)

fused        = mask_tokens + alpha * delta
```

其中：

- `mask_tokens` 是 ODISE 分支。
- `clip_tokens` 是模型学习出来的 LSeg/CLIP 分支到 ODISE-256 空间的投影。
- `base` 是 refine 前的混合输入，可用于判断 gate 融合本身是否合理。
- `fused` 是 refine 后最终 hybrid token，也是训练和主评估默认使用的 token。
- `alpha` 由配置控制，支持 `learnable` 和 `fixed`。

当前 `config/train_scannet_v2_full_multi_gpu.yaml` 里：

```yaml
alpha_mode: fixed
alpha_init: 1.25
alpha_max: 2.0
```

注意：`alpha_mode: fixed` 时，checkpoint 里的旧 alpha 不会覆盖 YAML 当前值。

## 当前 logits 实现

模型用 `pred_3d` 和每个 mask token 生成 point-mask logits：

```text
pred_mask_logits = pred_3d @ fused_embeddings.T
```

当前代码实际使用的是未归一化点积：

```python
point_features = pred_3d[point_mask]
logits = point_features @ mask_tokens_unnorm.t()
```

早期 README 和部分注释里仍有 `normalize + logit_scale` 的描述，但当前 `open_vocab_fusion_v2.py` 不是这条路径。这个差异会影响训练行为：未归一化点积可能让模型通过 feature norm 改变 mask logits，而不只是学习语义方向。

## Loss 路线

项目里有三套 loss。

### 1. 当前主线：Diff2Scene mask distillation

目录：

- `experiment_mask_distill/`
- `experiment_mask_distill/criterion_mask_distill.py`

配置：

```yaml
trainer:
  use_mask_distill: true
  mask_distill_weight: 1.0
  bce_weight: 0.0
  dice_weight: 0.0
```

公式：

```text
S_k          = pred_3d @ fused_token_k
B'_k^3d      = sigmoid(S_k)
B_k^3d       = lifted 2D mask
L_mask       = mean_k [1 - cos(B'_k^3d, B_k^3d)]
L_total      = mask_distill_weight * L_mask + L_aux
```

当前默认 `bce_weight = 0`、`dice_weight = 0`，所以只使用 mask-level cosine distillation。

这个 loss 主要监督 3D mask 几何对齐，不直接保证 `fused_embeddings @ text_features` 的开放词汇语义可读性。

### 2. 旧版原始 loss：BCE + Dice

文件：

- `model/criterion.py`

目标：

```text
每个 3D 点 × 每个 ODISE mask slot 的 membership
```

形式：

```text
L = BCEWithLogits(pred_mask_logits, lifted_2d_mask)
  + per-mask Dice(sigmoid(pred_mask_logits), lifted_2d_mask)
```

已有修正：

- 训练时默认不硬阈值过滤预测 mask，避免早期 loss 变成 0。
- 使用 `pos_weight` 缓解正负点不均衡。
- Dice 按 per-mask 计算后平均。
- 对 `x_label/y_label` 做 mask 尺寸缩放判断。

### 3. 历史实验：point-level feature distillation

目录：

- `experiment_distill/`
- `experiment_distill/criterion_distill.py`

形式：

```text
teacher_i = 当前点命中的 fused mask tokens 的平均
L_feat    = mean_i [1 - cos(pred_3d_i, teacher_i)]
L_mask    = BCE + Dice 低权重辅助
L_total   = feat_loss_weight * L_feat + mask_loss_weight * L_mask
```

这条路线直接监督 3D feature，但 teacher 仍依赖当前 fused token 的语义质量。

## 评估主命令

常用 checkpoint eval：

```bash
cd /home/sunl/work/mix_v1
conda activate mix
python evaluate/eval_mask_distill_checkpoint.py \
  --checkpoint checkpoints/diff2scene_hybrid_lseg_odise256_fusion/checkpoint_epoch_15.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --batch-size 12 \
  --num-workers 8
```

该脚本会构造 `MaskDistillTrainer` 并调用 `_validate()`。主评估逻辑在：

- `experiment_mask_distill/trainer_mask_distill.py`
- `evaluate/semantic_iou.py`

## 语义评估逻辑

ScanNet20 文本类：

```text
wall, floor, cabinet, bed, chair, sofa, table, door, window, bookshelf,
picture, counter, desk, curtain, refrigerator, shower curtain, toilet,
sink, bathtub, otherfurniture
```

当前 `semantic_clip_model: "ODISE-256"` 的含义：

```text
open_clip ViT-L/14 text 768D
  -> ODISE checkpoint word_head.text_proj
  -> ODISE text 256D
```

它不是一个新模型名，而是项目内部对 ODISE text-readable 256D 空间的标识。

默认验证指标：

```text
Hybrid/Text: fused hybrid 256D token @ ODISE text256
CLIP/Text:   raw LSeg/CLIP 512D token @ CLIP-B text512
Final-PC:    Hybrid/Text 和 CLIP/Text 的几何平均
MaskIoU:     mask-level 预测和 lifted mask 的 IoU
```

`best_model.pth` 当前按 `Final-PC semantic mIoU` 保存，不按 mask loss 或 mask IoU 保存。

## 2026-05-13 新增评估补充

为了看清混合模型内部两路分支和 refine 是否破坏语义，`eval_mask_distill_checkpoint.py` 现在通过 trainer 验证额外输出 ODISE-256 空间 probe：

```text
hybrid_odise256 = fused token @ ODISE text256
clip_odise256   = pixel_proj(LSeg 512D) @ ODISE text256
odise_odise256  = mask_proj(ODISE 256D) @ ODISE text256
base_odise256   = (mask_tokens + gate * clip_tokens) @ ODISE text256
refine_odise256 = final fused token @ ODISE text256
```

解释：

- `odise_odise256`：ODISE 分支本身在 ODISE 256D 文本空间是否可读。
- `clip_odise256`：模型学习出来的 LSeg/CLIP 分支投到 ODISE 256D 后是否可读。
- `base_odise256`：refine 前的混合状态；用于看 gate 混合是否已经破坏或改善语义。
- `refine_odise256`：refine 后最终 token；与 `base_odise256` 对比可判断 refine 是否破坏语义。
- `hybrid_odise256`：最终混合模型 token；数值上与 `refine_odise256` 是同一 token，只是保留“混合模型”命名方便读日志。

重点比较：

```text
odise_odise256 vs clip_odise256
```

看模型两方哪个更适合作为 3D point query 的语义来源。

```text
base_odise256 vs refine_odise256
```

看 `refine()` 是否提高或破坏 open-vocabulary 语义可读性。

```text
refine_odise256 vs Final-PC
```

看最终 hybrid token 本身是否足够，还是仍主要依赖 LSeg/CLIP-B 分支拉高最终语义 mIoU。

## 诊断脚本

单独看融合前后语义空间，可以用：

```bash
cd /home/sunl/work/mix_v1
conda activate mix
python evaluate/projection_space_diagnostic.py \
  --checkpoint checkpoints/diff2scene_hybrid_lseg_odise256_fusion/checkpoint_epoch_15.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --device cuda \
  --batch-size 2 \
  --num-workers 0 \
  --max-samples 20
```

旧诊断已有：

```text
odise_raw_label
odise_proj_256_text
lseg_raw_512_text
lseg_proj_256_text
fused_256_text
```

常用 eval 现在已覆盖更直接的：

```text
hybrid/clip/odise/base/refine @ ODISE-256
```

## 当前判断

当前代码已经回到 ODISE 256D 主空间，避免旧 512/768 投影导致的明显语义不可读问题。但当前主 loss 仍然主要是 mask 几何蒸馏，不直接约束文本语义。因此：

- mask IoU 变好不必然代表 open-vocabulary mIoU 变好。
- `clip_odise256` 和 `odise_odise256` 的对比很重要，可以看两路分支谁更保语义。
- `base_odise256` 和 `refine_odise256` 的对比很重要，可以判断 refine 是否对语义有破坏。
- 如果 `refine_odise256` 低于 `base_odise256`，说明 refine/residual 可能在为 mask loss 服务时损伤文本可读性。
- 如果 `Final-PC` 远高于 `hybrid_odise256/refine_odise256`，说明最终结果仍主要靠 LSeg/CLIP-B 分支补语义，hybrid token 本身还没有足够可读。

## 2026-05-13 smoke eval

已用最小样本确认新增评估链路能跑通：

```bash
/home/sunl/miniconda3/envs/mix/bin/python evaluate/eval_mask_distill_checkpoint.py \
  --checkpoint checkpoints/diff2scene_hybrid_lseg_odise256_fusion/checkpoint_epoch_15.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --batch-size 1 \
  --num-workers 0 \
  --max-samples 1 \
  --device cuda
```

运行环境里 CUDA 不可用，脚本自动退到 CPU。`max_samples=1` 只用于验证指标链路，不代表完整 val 结果。

该 smoke run 输出的 ODISE-256 probe：

```text
hybrid_odise256 = 0.2140406234420066
clip_odise256   = 0.08381340927621357
odise_odise256  = 0.30288591660624975
base_odise256   = 0.4258434742257523
refine_odise256 = 0.2140406234420066
```

这个单样本现象显示：`base` 明显高于 `refine/fused`，提示 refine 在该样本上可能破坏了 ODISE-256 文本可读性。需要用完整 val 再确认，不应只根据单样本下结论。

## 2026-05-13 LSeg->ODISE256 val20 结果

按需求测试：

```text
LSeg / pixel_pooled 512D
  -> 模型学习到的 pixel_proj 压到 256D
  -> 与 ODISE text256 计算 mask class probability
  -> 用 3D point-mask logits 投到点云
  -> 算 ScanNet20 semantic IoU
```

运行命令：

```bash
/home/sunl/miniconda3/envs/mix/bin/python evaluate/eval_mask_distill_checkpoint.py \
  --checkpoint checkpoints/diff2scene_hybrid_lseg_odise256_fusion/checkpoint_epoch_15.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --batch-size 4 \
  --num-workers 0 \
  --max-samples 20 \
  --device cuda \
  --metrics-json runs/eval_only/epoch15_lseg_proj_odise256_val20.json
```

当前工具环境 CUDA 不可用，脚本自动退到 CPU。结果保存到：

```text
runs/eval_only/epoch15_lseg_proj_odise256_val20.json
```

核心结果：

```text
semantic_miou_clip_odise256 = 0.09108888606912259
semantic_macc_clip_odise256 = 0.22329006967182274
n_valid_classes_clip_odise256 = 12
```

同一 val20 子集对比：

```text
hybrid_odise256 = 0.23404305472516132
clip_odise256   = 0.09108888606912259
odise_odise256  = 0.22680432160039687
base_odise256   = 0.2070336527052057
refine_odise256 = 0.23404305472516132

raw CLIP/Text 512D = 0.2766443333814461
Final-PC           = 0.2918763559404166
mask_miou          = 0.26597658676259656
```

结论：在这个 checkpoint 的 val20 子集上，`LSeg 512D -> pixel_proj -> ODISE256` 后再用 ODISE text256 读语义效果很弱，`mIoU=0.0911`，明显低于 raw LSeg/CLIP-B 512D 文本分支的 `0.2766`，也低于 ODISE 分支本身的 `0.2268`。这说明当前学到的 `pixel_proj` 并没有把 LSeg 语义稳定映射到 ODISE-256 文本空间。

## 2026-05-13 Raw LSeg vs Raw ODISE 2D 语义相似度

按需求进一步测试：不使用模型学习到的 `pixel_proj`，直接比较两个 2D teacher 本身的语义分布。

测试方式：

```text
同一个 mask slot 上：

LSeg raw512
  -> CLIP-B text512
  -> ScanNet20 class probability

ODISE raw256
  -> ODISE text256
  -> ScanNet20 class probability

然后比较两组 class probability 的相似度/top1 一致率。
同时用当前 checkpoint 的 3D point-mask logits 把两组概率分别投到 3D 点云，计算 semantic IoU。
```

脚本：

```text
evaluate/lseg_odise_semantic_similarity.py
```

运行命令：

```bash
/home/sunl/miniconda3/envs/mix/bin/python evaluate/lseg_odise_semantic_similarity.py \
  --checkpoint checkpoints/diff2scene_hybrid_lseg_odise256_fusion/checkpoint_epoch_15.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --batch-size 4 \
  --num-workers 0 \
  --max-samples 20 \
  --device cuda \
  --output-json runs/eval_only/epoch15_lseg_raw_vs_odise_raw_val20.json
```

当前工具环境 CUDA 不可用，脚本自动退到 CPU。结果保存到：

```text
runs/eval_only/epoch15_lseg_raw_vs_odise_raw_val20.json
```

val20 结果：

```text
total_masks = 191
mask_prob_cosine_mean = 0.566978
mask_top1_agreement = 0.523560
mask_js_divergence_mean = 0.295710

lseg_raw512_clip_text_3d:
  mIoU = 0.276644
  mAcc = 0.462063
  n_valid = 12

odise_raw256_odise_text_3d:
  mIoU = 0.226804
  mAcc = 0.410808
  n_valid = 13
```

结论：raw LSeg 和 raw ODISE 在同一批 mask 上语义分布有中等一致性，平均概率 cosine 约 `0.567`，top1 一致率约 `52.4%`。投到 3D 后，raw LSeg/CLIP-B 文本分支在 val20 上比 raw ODISE/ODISE-text256 更高：`0.2766` vs `0.2268`。

## 2026-05-13 LSeg raw512 -> ODISE256 probe -> ODISE text256

重要修正：本节最早一次记录中，`evaluate/odise_256_space_probe.py` 的 `_metric()` 因 prefix 不含 `miou`，导致 mAcc 覆盖了 mIoU 输出。旧记录里的 `0.4108/0.4159/0.4295/0.3701` 等数值实际是 mAcc，不是 mIoU。以下为修复脚本后的 mIoU 结果，旧数值不再作为 mIoU 结论使用。

进一步澄清：上一节 `Raw LSeg vs Raw ODISE` 中的 `lseg_raw512_clip_text_3d` 是 raw LSeg 在自己的 512D CLIP-B 文本空间里读语义，即：

```text
LSeg raw512 @ CLIP-B text512
```

这不是“投影到 ODISE256 后再和 ODISE text256 比”。

本节测试的是：

```text
原始 2D LSeg raw512
  -> 用 ridge probe 拟合到 ODISE raw256 空间
  -> 和 ODISE text256 做相似度
  -> 再通过 3D point-mask logits 投到点云算 semantic IoU
```

注意：这里不使用模型训练得到的 `pixel_proj` 权重，而是用诊断 probe 从 raw LSeg512 到 raw ODISE256 拟合一个线性映射。

运行命令：

```bash
/home/sunl/miniconda3/envs/mix/bin/python evaluate/odise_256_space_probe.py \
  --checkpoint checkpoints/diff2scene_hybrid_lseg_odise256_fusion/checkpoint_epoch_15.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --batch-size 4 \
  --num-workers 0 \
  --max-samples 20 \
  --probe-train-records 10 \
  --ridge 0.001 \
  --output-json runs/eval_only/epoch15_lseg_raw512_to_odise256_probe_val20.json \
  --device cuda
```

当前工具环境 CUDA 不可用，脚本自动退到 CPU。结果保存到：

```text
runs/eval_only/epoch15_lseg_raw512_to_odise256_probe_val20.json
```

修复后重新运行：

```text
runs/eval_only/epoch15_dual_anchor_probe_val20_fixed_metric.json
```

修复后的 mIoU：

```text
all20:
  ODISE raw256 @ ODISE text256              0.226804
  LSeg raw512 -> ODISE256 probe @ text256   0.243416
  current fused direct @ text256            0.234043
  fused -> ODISE256 post-hoc probe          0.228306

train10 probe-fit:
  ODISE raw256 @ ODISE text256              0.248589
  LSeg raw512 -> ODISE256 probe @ text256   0.266203
  current fused direct @ text256            0.252870
  fused -> ODISE256 post-hoc probe          0.249068

test10 probe-eval:
  ODISE raw256 @ ODISE text256              0.207411
  LSeg raw512 -> ODISE256 probe @ text256   0.188831
  current fused direct @ text256            0.196939
  fused -> ODISE256 post-hoc probe          0.195891
```

结论：`LSeg raw512 -> ODISE256 probe` 在 all20/train10 上高于 raw ODISE，但在 held-out test10 上低于 raw ODISE，说明该映射有训练划分内收益，但泛化仍不稳定。当前原代码 `current fused direct` 在 all20 介于 ODISE 和 LSeg probe 之间，但在 test10 低于 raw ODISE。

## 2026-05-13 双锚点 semantic query 离线验证

重要修正：本节最早一次记录同样受到 `_metric()` mAcc 覆盖 mIoU bug 影响。以下为修复后的 mIoU。

根据方案建议，先不改训练，只做验证侧 semantic query 替换：

```text
odise_q = normalize(ODISE raw256)
lseg_q  = normalize(LSeg raw512 -> ODISE256 ridge probe)

semantic_q = normalize((1 - w) * odise_q + w * lseg_q)
```

3D point-mask logits 仍使用当前 checkpoint 生成的 logits；只替换 mask-text semantic query。

运行命令：

```bash
/home/sunl/miniconda3/envs/mix/bin/python evaluate/odise_256_space_probe.py \
  --checkpoint checkpoints/diff2scene_hybrid_lseg_odise256_fusion/checkpoint_epoch_15.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --batch-size 4 \
  --num-workers 0 \
  --max-samples 20 \
  --probe-train-records 10 \
  --ridge 0.001 \
  --mix-lseg-weights 0.3,0.5,0.7 \
  --output-json runs/eval_only/epoch15_dual_anchor_probe_val20.json \
  --device cuda
```

结果保存到：

```text
runs/eval_only/epoch15_dual_anchor_probe_val20_fixed_metric.json
```

修复后的 mIoU：

```text
all20:
  ODISE raw256                         0.226804
  LSeg->ODISE256                       0.243416
  current fused direct                 0.234043
  current fused post-hoc probe         0.228306
  mix 0.7 ODISE + 0.3 LSeg             0.230104
  mix 0.5 ODISE + 0.5 LSeg             0.246078
  mix 0.3 ODISE + 0.7 LSeg             0.250293

train10 probe-fit:
  ODISE raw256                         0.248589
  LSeg->ODISE256                       0.266203
  current fused direct                 0.252870
  current fused post-hoc probe         0.249068
  mix 0.7 ODISE + 0.3 LSeg             0.256440
  mix 0.5 ODISE + 0.5 LSeg             0.258782
  mix 0.3 ODISE + 0.7 LSeg             0.261842

test10 probe-eval:
  ODISE raw256                         0.207411
  LSeg->ODISE256                       0.188831
  current fused direct                 0.196939
  current fused post-hoc probe         0.195891
  mix 0.7 ODISE + 0.3 LSeg             0.207641
  mix 0.5 ODISE + 0.5 LSeg             0.231794
  mix 0.3 ODISE + 0.7 LSeg             0.214809
```

结论：修复后仍支持继续验证双锚点 semantic query，但收益幅度比旧误报小。`0.5 ODISE + 0.5 LSeg` 在 held-out test10 上为 `0.231794`，高于 raw ODISE `0.207411`、LSeg 单路 `0.188831` 和当前原代码 direct fused `0.196939`。这支持“保持 geometry logits 不变，只替换 semantic query 可以恢复一部分语义”的判断，但还需要完整 val 验证。

## 2026-05-13 LSeg 低的诊断

### 对比项

同一套 geometry logits 下比较：

```text
ODISE raw256 @ ODISE text256
LSeg raw512 -> ODISE256 probe @ ODISE text256
current fused direct @ ODISE text256
mix 0.5 ODISE + 0.5 LSeg
mix entropy confidence clamp[0.2,0.6]
mix agreement gate
```

运行结果文件：

```text
runs/eval_only/epoch15_lseg_diagnostic_val20.json
```

### mIoU 结果

```text
all20:
  ODISE raw256                    0.226804
  LSeg->ODISE256                  0.243416
  current fused direct            0.234043
  mix 0.5/0.5                     0.246078
  entropy confidence mix          0.244844
  agreement gate mix              0.245279

test10:
  ODISE raw256                    0.207411
  LSeg->ODISE256                  0.188831
  current fused direct            0.196939
  mix 0.5/0.5                     0.231794
  entropy confidence mix          0.229921
  agreement gate mix              0.230115
```

结论：

- 固定 `0.5/0.5` 是当前 val20/test10 上最强的简单 semantic query。
- confidence/entropy 和 agreement gate 没有超过固定 `0.5/0.5`，但接近。
- `current fused direct` 低于 `mix 0.5/0.5`，说明当前训练得到的 fused query 没有充分保留 LSeg 互补语义。

### per-class 现象

test10 per-class IoU 摘要：

```text
class    ODISE    LSeg256   current fused   mix 0.5
cabinet  0.1970   0.2949    0.2845          0.3253
chair    0.6563   0.5511    0.6547          0.6416
counter  0.1254   0.1371    0.1472          0.1985
floor    0.4683   0.4382    0.4613          0.4855
table    0.2629   0.2834    0.2300          0.3092
wall     0.7145   0.7500    0.6698          0.7164
window   0.0644   0.0000    0.1127          0.1050
```

观察：

- LSeg256 对 `cabinet/table/wall/counter` 有明显互补。
- LSeg256 对 `chair/floor/window` 不如 ODISE。
- `mix 0.5/0.5` 的提升主要来自 `cabinet/counter/table/floor`，但会略降 `chair/window`。

这说明不能简单“加大 LSeg 权重”。LSeg 有互补类，但也有弱类；更合理的是 semantic query 解耦后，继续研究 class/confidence/mask-aware 的 LSeg 权重。

### probe/ridge 稳定性

小网格：

```text
probe_train_records = 5 / 10 / 15
ridge = 1e-4 / 1e-3 / 1e-2 / 1e-1
max_samples = 20
```

注意：不同 `probe_train_records` 会改变 held-out test 子集，所以横向比较不同 train_records 时不能当成严格同一测试集。更应该在同一个 train_records 内看 ridge 稳定性。

test 部分 mIoU：

```text
train_records,ridge,odise_test,lseg256_test,mix05_test
5,0.0001,0.232812,0.114121,0.205941
5,0.001, 0.232812,0.117180,0.233383
5,0.01,  0.232812,0.196290,0.277310
5,0.1,   0.232812,0.205442,0.267301

10,0.0001,0.207411,0.199188,0.240198
10,0.001, 0.207411,0.188831,0.231794
10,0.01,  0.207411,0.200719,0.238243
10,0.1,   0.207411,0.261892,0.237442

15,0.0001,0.143525,0.097755,0.136200
15,0.001, 0.143525,0.156729,0.140248
15,0.01,  0.143525,0.160384,0.178196
15,0.1,   0.143525,0.184017,0.181918
```

观察：

- LSeg256 对 ridge 很敏感，说明 `LSeg raw512 -> ODISE256` 投影本身不稳定。
- 在 `train_records=10` 时，`ridge=0.1` 的 LSeg256 test mIoU 最高，但 `mix05` 反而不如 `ridge=1e-4/1e-2`，说明“让 LSeg 单路更强”不等于“融合更强”。
- 这支持当前问题之一是 projection/text-space mismatch，而不只是 LSeg 原始语义差。

### 阶段结论

当前最可信结论：

```text
1. LSeg 原始语义有用，且在 cabinet/table/wall/counter 等类上提供互补。
2. LSeg->ODISE256 投影不稳定，对 ridge 和 split 敏感。
3. 当前 fused query 没有稳定保留 LSeg 的互补语义。
4. 直接加大 LSeg 权重不可取；固定 0.5/0.5 是当前简单强 baseline，但最终更可能需要 confidence/class/mask-aware 的 semantic fusion。
```

下一步应该在完整 val 上验证：

```text
current fused direct
mix 0.5/0.5
entropy/confidence mix
agreement gate mix
```

如果完整 val 上 `mix 0.5/0.5` 或 confidence-aware 稳定高于 current fused direct，则再正式把 geometry query 与 semantic query 解耦。

## 2026-05-13 原模型 pixel_proj vs ridge semantic projector

问题定位目标：

```text
同一个 checkpoint
同一套 3D geometry logits
同一批 val20 数据
只替换 LSeg->256 的语义投影方式
```

比较：

```text
1. ODISE raw256 @ ODISE text256
2. LSeg model_pixel_proj256 @ ODISE text256
3. LSeg ridge_sem_proj256 @ ODISE text256
4. current fused direct @ ODISE text256
5. mix 0.5 ODISE + 0.5 model_pixel_proj256
6. mix 0.5 ODISE + 0.5 ridge_sem_proj256
```

运行结果：

```text
runs/eval_only/epoch15_lseg_projection_compare_val20.json
```

整体 mIoU：

```text
all20:
  ODISE raw256                            0.226804
  LSeg model_pixel_proj256                0.091089
  LSeg ridge_sem_proj256                  0.243416
  current fused direct                    0.234043
  mix ODISE + model_pixel_proj            0.195630
  mix ODISE + ridge_sem_proj              0.246078

train10 probe-fit:
  ODISE raw256                            0.248589
  LSeg model_pixel_proj256                0.114643
  LSeg ridge_sem_proj256                  0.266203
  current fused direct                    0.252870
  mix ODISE + model_pixel_proj            0.223665
  mix ODISE + ridge_sem_proj              0.258782

test10 probe-eval:
  ODISE raw256                            0.207411
  LSeg model_pixel_proj256                0.047860
  LSeg ridge_sem_proj256                  0.188831
  current fused direct                    0.196939
  mix ODISE + model_pixel_proj            0.174806
  mix ODISE + ridge_sem_proj              0.231794
```

结论：

```text
条件 1 成立：
  原模型训练出来的 LSeg model_pixel_proj256 非常低：
  all20 0.0911，test10 0.0479。

条件 2 成立：
  ridge semantic projector 明显高于 model_pixel_proj：
  all20 0.2434 vs 0.0911，
  test10 0.1888 vs 0.0479。

条件 3 成立：
  mix ODISE + ridge_sem_proj 明显高于 current fused direct 和 mix ODISE + model_pixel_proj：
  all20 0.2461 > 0.2340 / 0.1956，
  test10 0.2318 > 0.1969 / 0.1748。
```

per-class test10 摘要：

```text
class    ODISE   model_proj  ridge_proj  fused   mix_model  mix_ridge
cabinet  0.1970  0.0000      0.2949      0.2845  0.1592     0.3253
chair    0.6563  0.0000      0.5511      0.6547  0.7355     0.6416
counter  0.1254  0.1329      0.1371      0.1472  0.1586     0.1985
floor    0.4683  0.4367      0.4382      0.4613  0.5041     0.4855
table    0.2629  0.0000      0.2834      0.2300  0.0853     0.3092
wall     0.7145  0.0000      0.7500      0.6698  0.4200     0.7164
window   0.0644  0.0000      0.0000      0.1127  0.0181     0.1050
```

这说明之前 `LSeg` 低到 `0.0x/0.09` 的来源就是原模型里的共享 `pixel_proj`。它被 mask distillation / geometry fusion 训练成了非 text-readable 的 geometry-oriented projection，而不是 LSeg semantic projector。

因此后续方案应避免复用当前 `pixel_proj` 做 semantic readout。更合理的最小改法是：

```text
geometry query:
  继续用当前 fused query，负责 3D point-mask logits。

semantic query:
  新增独立 pixel_sem_proj。
  用 ridge semantic projector 初始化。
  前几轮冻结或低学习率。
  semantic_q = normalize(0.5 * ODISE raw256 + 0.5 * pixel_sem_proj(LSeg512))
```
