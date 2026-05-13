# mix_v1

## 本机数据路径

当前配置已经从旧远端数据路径改到本机可用的 `mix` 数据目录：

- 3D ScanNet: `/home/sunl/work/mix/data/scannet_3d`
- 2D ScanNet: `/home/sunl/work/mix/data/scannet_2d`
- 预计算 2D 特征: `/home/sunl/work/mix/data/pixel_pooled`
- 预计算 3D->2D 投影: `/home/sunl/work/mix/data/scannet_projections`

对应配置文件：

- `config/data_scannet_3d.yaml`
- `config/train_scannet_v2_full_multi_gpu.yaml`
- `experiment_mask_distill/train_mask_distill.yaml`
- `experiment_distill/train_distill.yaml`

如果后续要改用 `mix2` 的数据，把上面路径中的 `/home/sunl/work/mix/data` 替换为 `/home/sunl/work/mix2/data` 即可；当前本机只检查到 `mix/data` 存在完整数据目录。

## 环境

本机存在 conda 环境 `mix`，运行前使用：

```bash
conda activate mix
```

非交互式可使用：

```bash
/home/sunl/miniconda3/envs/mix/bin/python
```

## 当前维度状态

当前代码已经从上一版 `512` 维融合/文本空间切到 ODISE 原生 `256` 维融合空间。上一版 512D 相关代码已备份到：

- `backups/512_projection_2026-05-08/`

现在混合模型最终输出 `256` 维 token，并与 ODISE checkpoint 中 `word_head.text_proj` 生成的 `256` 维文本特征做相似度计算：

- ODISE mask embedding: `256`
- LSeg / `pixel_pooled`: `512`
- `mask_proj`: identity，保留 ODISE raw `256`
- `pixel_proj`: `512 -> 256`
- `fused_embeddings`: `256`
- 3D `pred_3d`: `256`
- text features: `256` (`ODISE word_head.text_proj`)

关键代码位置：

- `model/open_vocab_fusion_v2.py`: `mask_embedding_dim=256`, `pixel_embedding_dim=512`, `fused_embedding_dim=256`
- `model/modeling.py`: `ODISEPixelMaskFusionNet(pixel_dim=512, mask_dim=256, out_dim=256)`
- `model/pc_net.py`: `decoder_proj_out_dim=256`
- `evaluate/semantic_iou.py`: `semantic_clip_model: "ODISE-256"` 时加载 ODISE `word_head.text_proj`
- `config/train_scannet_v2_full_multi_gpu.yaml`: `semantic_clip_model: "ODISE-256"`

融合公式：

```text
mask_tokens  = mask_embed                   # raw ODISE 256
pixel_tokens = pixel_proj(pixel_pooled)     # LSeg 512 -> 256
gate         = sigmoid(gate([mask_tokens, pixel_tokens]))
delta        = refine(mask_tokens + gate * pixel_tokens)
fused        = mask_tokens + alpha * delta  # 256
```

当前 `alpha` 是可学习参数，初始值 `1.0`。

`alpha` 现在可在配置文件的 `model` 段控制：

```yaml
model:
  alpha_mode: learnable  # learnable 或 fixed
  alpha_init: 1.0
  alpha_max: 2.0      # 只在 learnable 模式使用
```

- `alpha_mode: learnable`：`alpha` 从 `alpha_init` 初始化，并通过 sigmoid 参数化限制在 `[0, alpha_max]` 内自学习。
- `alpha_mode: fixed`：`alpha` 固定为 `alpha_init`，不参与训练，没有 `alpha_max`/`alpha_min` 的概念；例如固定为 `1.0` 就设 `alpha_init: 1.0`。
- 旧 checkpoint 中直接保存的 `fuse_embed.alpha` 会在加载时转换到新的 bounded alpha 参数；如果切换到 fixed 模式，配置文件里的固定值优先。

注意：旧的 `768` 或 `512` 维 checkpoint 不能直接恢复到当前 `256` 维模型，相关投影层和 3D decoder 的参数形状会不匹配。重新训练时保持 `resume: ""`。当前主配置已经把 `resume` 清空，并把输出目录改成：

- `checkpoints/diff2scene_hybrid_lseg_odise256_fusion`
- `runs/diff2scene_hybrid_lseg_odise256_fusion`

## Loss

根配置 `config/train_scannet_v2_full_multi_gpu.yaml` 当前启用 `use_mask_distill: true`，实际走 `experiment_mask_distill` 的 Diff2Scene-style mask distillation：

```text
S_k = <normalize(F_3d), normalize(f_k_2d)> / tau
B'_k_3d = sigmoid(S_k)
B_k_3d = lifted 2D mask
L_mask_distill = mean_k [1 - cos(B'_k_3d, B_k_3d)]
L_total = mask_distill_weight * L_mask_distill + L_aux
```

当前权重：

- `mask_distill_weight: 1.0`
- `bce_weight: 0.0`
- `dice_weight: 0.0`

因此默认训练只用 mask-level cosine distillation，不使用 BCE/Dice 辅助项。

另有两个历史/实验 loss：

- `model/criterion.py`: 原始 point-mask `BCEWithLogits + per-mask Dice`
- `experiment_distill/criterion_distill.py`: point-level feature cosine distillation + 低权重 mask BCE/Dice

## 评估

语义评估使用 `evaluate/semantic_iou.py` 中的 `SCANNET_LABELS_20` 作为文本类别：

```text
wall, floor, cabinet, bed, chair, sofa, table, door, window, bookshelf,
picture, counter, desk, curtain, refrigerator, shower curtain, toilet,
sink, bathtub, otherfurniture
```

### `ODISE-256` 是什么

`ODISE-256` 不是 ODISE 论文里的正式名词，而是本项目为了配置清晰起的内部标识。它表示：

```text
open_clip ViT-L/14 text embedding 768D
  -> ODISE checkpoint 的 word_head.text_proj
  -> 256D text embedding
```

来源在 ODISE 实现里：

- `ODISE/odise/modeling/meta_arch/odise.py`
- 类：`WordEmbed`
- 关键层：`self.text_proj = nn.Linear(self.clip.dim_latent, projection_dim)`

对当前使用的 `odise_caption_coco_50e` checkpoint，文本投影头参数形状是：

```text
word_head.text_proj.weight: (256, 768)
word_head.text_proj.bias:   (256,)
```

这个下载的 ODISE 文本头方向是 `768 -> 256`：先得到 CLIP ViT-L/14 的 `768D` 文本特征，再经过 `word_head.text_proj` 投到 ODISE mask embedding 的 `256D` 空间。

注意不要把这个头和本项目的 3D/fusion 投影头混淆：

- ODISE 下载文本头：`word_head.text_proj`，`768D CLIP text -> 256D ODISE text`
- 当前 `mix_v1` mask 分支：raw ODISE mask embedding 保持 `256D`
- 当前 `mix_v1` LSeg 分支：`pixel_proj`，`512D -> 256D`
- 当前 `mix_v1` 3D decoder：`256D backbone feature -> 256D pred_3d`
- 旧 `mix2_v1` 768D 实验里的 project decoder/fusion head 才会出现 `256 -> 768` 或 `512 -> 768`

如果使用 ODISE label checkpoint，则对应头是：

```text
category_head.text_proj.weight: (256, 768)
category_head.text_proj.bias:   (256,)
```

所以这里的 `256D` 来自 ODISE checkpoint 的 `word_head` 投影维度。它的用途是把 CLIP 文本特征投到 ODISE mask embedding 所在的 256D 空间，使得可以计算：

```text
similarity = normalize(mask_embedding_256) @ normalize(text_embedding_256).T
```

因此，当前配置里的 `semantic_clip_model: "ODISE-256"` 含义是：用 ODISE 自己训练得到的 `word_head.text_proj` 作为文本器，生成 256D 文本特征，与 `fused_embeddings(256)` 做相似度。它不是新的模型，也不是论文术语，而是 ODISE checkpoint 中 text-readable mask embedding space 的一个实现名称。

验证时会调用 `build_text_features(..., clip_model="ODISE-256")` 生成 `(20, 256)` 的 ScanNet 文本特征。该路径会：

1. 从本地 ODISE checkpoint 读取 `word_head.text_proj.weight/bias`
2. 用 `open_clip ViT-L-14` 编码 prompt
3. 经过 ODISE `word_head.text_proj` 投影到 `256` 维

默认 checkpoint 路径：

```text
~/.torch/iopath_cache/NVlabs/ODISE/releases/download/v1.0.0/odise_caption_coco_50e-853cc971.pth
```

然后用：

```text
mask_class_probs = softmax(normalize(fused_embeddings_256) @ normalize(text_features_256).T * 100)
point_scores = sigmoid(pred_mask_logits) @ mask_class_probs
```

因此当前主语义指标是 `fused_embeddings(256)` 和 ODISE `ScanNet text_features(256)` 的相似度评估。`semantic_pixel_clip_model: "ViT-B/32"` 仍用于可选的 LSeg/CLIP 512D 分支和 PC 几何平均评估。

验证日志只保留三项语义 IoU：

- `Hybrid/Text`: 混合 token `fused_embeddings(256)` vs ODISE `text256`
- `CLIP/Text`: LSeg/CLIP `pixel_pooled(512)` vs CLIP-B `text512`
- `Final-PC`: `Hybrid/Text` 和 `CLIP/Text` 的几何平均最终结果

`best_model.pth` 现在按 `Final-PC` 语义 mIoU 保存，不再按 mask-level `mask_mIoU` 保存。

## 运行命令

单卡训练：

```bash
cd /home/sunl/work/mix_v1
conda activate mix
python train_open_vocab_v2.py --config config/train_scannet_v2_full_multi_gpu.yaml
```

多卡训练：

```bash
cd /home/sunl/work/mix_v1
conda activate mix
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_open_vocab_v2_ddp.py --config config/train_scannet_v2_full_multi_gpu.yaml
```

单独运行 mask distillation 实验配置：

```bash
cd /home/sunl/work/mix_v1
conda activate mix
python train_open_vocab_v2.py --config experiment_mask_distill/train_mask_distill.yaml
```

快速验证配置和前向是否能跑通（CPU/GPU 均可，建议先用小样本 checkpoint eval；新结构需要使用 256D checkpoint）：

```bash
cd /home/sunl/work/mix_v1
conda activate mix
python -m py_compile model/modeling.py model/open_vocab_fusion_v2.py evaluate/semantic_iou.py train_open_vocab_v2.py
```

如果要单独验证某个 256D checkpoint，使用从远端 `run_in_f_main` 分支同步后的评估入口：

```bash
cd /home/sunl/work/mix_v1
conda activate mix
python evaluate/eval_mask_distill_checkpoint.py \
  --checkpoint checkpoints/diff2scene_hybrid_lseg_odise256_fusion/checkpoint_epoch_1.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --batch-size 2 \
  --num-workers 0
```

需要保存完整指标时可追加：

```bash
  --metrics-json runs/eval_only/metrics_epoch_1.json
```

如果要看融合前后语义结果，也就是 base/projection 和 refine/fused 后的对比，用诊断脚本：

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

输出里重点看这几项：

- `odise_proj_256_text`: ODISE mask embedding/base 256D 和 ODISE text256 的结果
- `lseg_raw_512_text`: LSeg 原始 512D 和 CLIP-B text512 的结果
- `lseg_proj_256_text`: LSeg 512D 投影到 256D 后和 ODISE text256 的结果
- `fused_256_text`: 经过 gate/refine/alpha 后的最终 fused 256D 和 ODISE text256 的结果

其中 `fused_256_text` 就是 refine 后结果；和 `odise_proj_256_text`、`lseg_proj_256_text` 对比，可以判断 refine/fusion 是否提升或破坏语义。

TensorBoard：

```bash
cd /home/sunl/work/mix_v1
conda activate mix
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```


2. 模型有没有变化？
2.1 融合网络主体没有明显变化

两个分支的配置都还是：

pixel_embedding_dim: 512
mask_embedding_dim: 256
fused_embedding_dim: 256
pc_last_dim: 256

也就是说，LSeg 512D + ODISE 256D → fused 256D 这个模型设计没有变。main 配置如此，run_in_test03_main 配置也是如此。

2.2 但是 point-mask logits 的计算变了

这是最重要的差异。

当前 main 里，3D 点特征和 mask token 计算 logits 时使用了归一化和 logit_scale：

point_features = F.normalize(pred_3d[point_mask], dim=-1)
mask_tokens_norm = F.normalize(mask_tokens, dim=-1)
logits = logit_scale * (point_features @ mask_tokens_norm.t())

也就是优化角度相似度，和文本空间评估更一致。

但是 run_in_test03_main 里这一段被改成了不归一化、不乘 logit_scale：

point_features = pred_3d[point_mask]
mask_tokens_unnorm = mask_tokens
logits = point_features @ mask_tokens_unnorm.t()

而且代码注释里写了“怀疑梯度消失”。

所以这里必须明确：
模型主体没变，但 forward 行为变了。这个变化会直接影响训练 loss、mask prediction、semantic eval，不是单纯评估变化。

这个区别很大。因为归一化版本学的是：

点特征方向 ↔ mask token 方向

非归一化版本学的是：

点特征范数 × mask token 范数 × 方向相似度

后者可能让模型通过放大/缩小 feature norm 来降低 loss，而不一定提升语义方向。
