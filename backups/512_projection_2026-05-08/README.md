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

当前代码已经改成统一的 `512` 维融合/文本空间。混合模型最终输出 `512` 维 token，并与 `ViT-B/32` 的 `512` 维文本特征做相似度计算：

- ODISE mask embedding: `256`
- LSeg / `pixel_pooled`: `512`
- `mask_proj`: `256 -> 512`
- `pixel_proj`: `512 -> 512`
- `fused_embeddings`: `512`
- 3D `pred_3d`: `512`
- text features: `512` (`ViT-B/32`)

关键代码位置：

- `model/open_vocab_fusion_v2.py`: `mask_embedding_dim=256`, `pixel_embedding_dim=512`, `fused_embedding_dim=512`
- `model/modeling.py`: `ODISEPixelMaskFusionNet(pixel_dim=512, mask_dim=256, out_dim=512)`
- `model/pc_net.py`: `decoder_proj_out_dim=512`
- `config/train_scannet_v2_full_multi_gpu.yaml`: `semantic_clip_model: "ViT-B/32"`

融合公式：

```text
mask_tokens  = mask_proj(mask_embed)        # 256 -> 512
pixel_tokens = pixel_proj(pixel_pooled)     # 512 -> 512
gate         = sigmoid(gate([mask_tokens, pixel_tokens]))
delta        = refine(mask_tokens + gate * pixel_tokens)
fused        = mask_tokens + alpha * delta  # 512
```

当前 `alpha` 是可学习参数，初始值 `1.0`。

注意：旧的 `768` 维 checkpoint 不能直接恢复到当前 `512` 维模型，相关投影层和 3D decoder 的参数形状会不匹配。重新训练时保持 `resume: ""`。

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

验证时会调用 `build_text_features(..., clip_model="ViT-B/32")` 生成 `(20, 512)` 的 ScanNet 文本特征，然后用：

```text
mask_class_probs = softmax(normalize(fused_embeddings_512) @ normalize(text_features_512).T * 100)
point_scores = sigmoid(pred_mask_logits) @ mask_class_probs
```

因此当前主语义指标是 `fused_embeddings(512)` 和 `ScanNet text_features(512)` 的相似度评估。当前环境已验证 `ViT-B/32` 文本特征可以生成，shape 为 `(20, 512)`。

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

TensorBoard：

```bash
cd /home/sunl/work/mix_v1
conda activate mix
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```
