# 2026-05-12 方法 A：Mask Loss 梯度分流 + NCE + VICReg

## 背景

- 本记录基于当前 `mix_v1` 的 256D ODISE-native 融合代码整理，目标是为下一轮实现和实验留存清晰方案。
- 当前 `README.md` 已记录主线状态：ODISE mask embedding 为 `256D`，LSeg / `pixel_pooled` 为 `512D`，`pixel_proj` 将 LSeg 投到 `256D`，最终 `fused_embeddings(256)` 与 ODISE `text256` 做语义评估。
- 当前主训练启用 Diff2Scene-style mask distillation：用 `pred_3d @ fused_embeddings.T` 预测 3D mask，再与 lifted 2D mask 做 cosine 对齐。

## 当前代码状态

- 本分支已实现方法 A 的训练代码，分支名：`method-a-hybrid-fusion-reg`。
- **修订**：`vicreg_loss_batch` 对 fused/base **不做** L2 normalize（与 `gamma=1` 的 variance 项匹配）；第一轮配置将 `vicreg_weight` 降为 `0.01`。
- 新增/修改的主要文件：
  - `model/modeling.py`
  - `model/open_vocab_fusion_v2.py`
  - `experiment_mask_distill/criterion_mask_distill.py`
  - `experiment_mask_distill/fusion_regularization.py`
  - `experiment_mask_distill/trainer_mask_distill.py`
  - `config/train_scannet_v2_full_multi_gpu.yaml`
  - `experiment_mask_distill/train_mask_distill.yaml`
- 融合模块在 `model/modeling.py` 的 `ODISEPixelMaskFusionNet`：

```text
mask_tokens  = mask_proj(mask_embed)        # ODISE 256 -> 256, 当前为 Identity
pixel_tokens = pixel_proj(pixel_pooled)     # LSeg 512 -> 256
gate         = sigmoid(gate([mask_tokens, pixel_tokens]))
delta        = refine(mask_tokens + gate * pixel_tokens)
fused        = mask_tokens + alpha * delta
```

- 本分支已将 `alpha` 改为 `alpha_max * sigmoid(alpha_raw)`，默认 `alpha_max=0.2`，避免 refine 一开始无约束地主导 ODISE token 空间。
- 顶层模型在 `model/open_vocab_fusion_v2.py` 的 `OpenVocab3DFusionModelV2.forward()`：
  - 已调用 `self.fuse_embed(..., return_aux=True)` 得到 `fusion_aux`。
  - 每个 batch item 已输出 `pred_mask_logits` 和 `pred_mask_logits_detached_teacher` 两套 logits。
  - 返回结果已包含 `fusion_aux`，供 trainer/criterion 计算 NCE、VICReg 和 gate/alpha 诊断。
- mask distillation loss 在 `experiment_mask_distill/criterion_mask_distill.py`：
  - `_mask_distill_loss(logits_key=...)` 可读取不同 logits。
  - `compute_loss()` 已组合 `loss_mask_student` 和 `loss_mask_joint`。

## Alpha 结论（bounded adaptive）

**不要**把 refine 残差系数固定为 `1`，也**不建议**固定为 `0.5`。当前实现里的 **bounded adaptive alpha** 更合理：

```python
self.alpha_raw = nn.Parameter(torch.tensor(-1.5))
self.alpha_max = 0.2
alpha = alpha_max * torch.sigmoid(alpha_raw)
```

初始近似：`alpha ≈ 0.2 * sigmoid(-1.5) ≈ 0.036`。第一轮实验偏保守是故意的：refine 是自由 MLP，若 `alpha` 固定为 `1` 或较大的常数，很容易把 ODISE-256 语义空间拉飞；在已有 NCE/VICReg 的前提下，仍不建议第一轮就把 residual 权重放得太大。

建议：

- **第一轮**：保留 bounded adaptive，`alpha_max=0.2`，`alpha_raw=-1.5`。暂时不要用 `alpha=0.5` 或 `1`。
- **第二轮（仅在第一轮诊断满足时再改）**：若 `loss_nce` 在降、`loss_vicreg` 正常、Hybrid mIoU 不再掉，但 fused 相对 base 提升仍弱，可把 `alpha_raw` 初始改为 `0.0`，使初始 `alpha = 0.1 * sigmoid(0) = 0.1`（在 `alpha_max=0.2` 下）。**本轮不要提前改 alpha。**

## 问题判断

当前 mask distillation 会同时更新 3D student 和 `fused_embeddings`。如果该 loss 对 fused 的梯度过强，fused 容易变成只服务 `pred_3d @ fused -> mask` 的 latent classifier，而不是保持开放词汇语义可读的 Hybrid token。

方法 A 的核心判断是：mask loss 应主要训练 3D student 的几何 mask 预测能力；fused token 仍参与动态训练，但只接受少量 mask loss 梯度；fused 的主要语义结构由跨模态 NCE 负责；VICReg 用来稳定 refine 前后的表示空间。

## 方法 A 总损失

正式推荐版本：

```text
loss_total =
    1.0  * loss_mask_student
  + 0.05 * loss_mask_joint
  + 0.5  * loss_nce
  + 0.01 * loss_vicreg
```

其中：

- `loss_mask_student` 使用 `fused.detach()` 计算 mask logits，只更新 `pred_3d` / 3D student。
- `loss_mask_joint` 使用正常 `fused` 计算 mask logits，保留少量 mask loss 梯度给 fusion 模块。
- `loss_nce` 让同一 mask 区域的 `fused_i` 接近 `ODISE_i` 和 `LSeg-projected_i`，同时与其他 mask token 区分。
- `loss_vicreg` 约束 `base = mask_tokens + gate * pixel_tokens` 与 `fused = mask_tokens + alpha * refine(base)`，防止 refine 后空间漂移、塌缩和维度冗余。

## 模型输出改造

`ODISEPixelMaskFusionNet.forward()` 建议增加 `return_aux=False` 参数。默认保持旧行为，只返回 `fused`；训练方法 A 时传 `return_aux=True` 并返回：

```text
fused:        final fused token
mask_tokens:  ODISE 256D token
pixel_tokens: LSeg 512D 投影后的 256D token
base:         pre-refine hybrid token = mask_tokens + gate * pixel_tokens
delta:        refine(base)
gate:         gate 值
alpha:        有上界的残差系数
```

建议把当前：

```text
alpha = nn.Parameter(torch.tensor(1.0))
fused = mask_tokens + alpha * delta
```

改为：

```text
alpha_raw = nn.Parameter(torch.tensor(-1.5))
alpha = alpha_max * sigmoid(alpha_raw)
fused = mask_tokens + alpha * delta
```

第一版建议 `alpha_max=0.2`。

## 顶层 forward 改造

`OpenVocab3DFusionModelV2.forward()` 中，将融合调用改为：

```text
fusion_out = self.fuse_embed(
    pixel_embeddings,
    mask_embeddings,
    mask_tensors,
    mask_valid,
    return_aux=True,
)
fused_embeddings = fusion_out["fused"]
```

返回结果中新增：

```text
fusion_aux: fusion_out
```

同时在计算每个 batch item 的 mask logits 时多算一套 detached-teacher logits：

```text
pred_mask_logits:                  point_features @ normalize(fused)
pred_mask_logits_detached_teacher: point_features @ normalize(fused.detach())
```

注意这里只 detach mask token，不 detach `point_features`，所以 `loss_mask_student` 仍然能训练 3D student。

## Criterion 改造

`MaskDistillCriteria._mask_distill_loss()` 增加 `logits_key` 参数：

```text
_mask_distill_loss(logits_key="pred_mask_logits")
```

读取 logits 时改为：

```text
pred_logits_full = self.outputs[b][0][logits_key]
```

`compute_loss()` 中分别计算：

```text
loss_mask_student = _mask_distill_loss("pred_mask_logits_detached_teacher")
loss_mask_joint   = _mask_distill_loss("pred_mask_logits")
mask_loss = mask_student_weight * loss_mask_student + mask_joint_weight * loss_mask_joint
```

建议保留旧配置兼容路径：如果没有配置 `mask_student_weight/mask_joint_weight`，仍可退回原来的 `mask_distill_weight` 单损失，方便对比旧 run。

## NCE 设计

第一阶段使用 hard NCE：

```text
z = normalize(fused)
m = normalize(mask_tokens.detach())
l = normalize(pixel_tokens.detach())

loss_nce = 0.5 * CE(z @ m.T / tau, arange(K))
         + 0.5 * CE(z @ l.T / tau, arange(K))
```

默认：

```text
nce_type = hard
nce_tau = 0.1
nce_weight = 0.5
```

如果 hard NCE 把同类不同实例推得太远，例如两个 chair 被当成强负样本，可以第二阶段切换到 soft NCE，用 ODISE relation 与 LSeg relation 的平均相似度作为 teacher distribution。

## VICReg 设计

VICReg 的两个 view 使用：

```text
z = fused
y = base.detach()
```

不要用 raw ODISE token 作为 reference，否则会把 fused 重新拉回 ODISE-only 空间，削弱 hybrid 的意义。

建议实现 `vicreg_loss_batch(fused, base, valid_mask)`，每个 batch item 只在有效 mask 数不少于 4 时计算：

```text
loss_vicreg = sim_w * invariance
            + var_w * variance
            + cov_w * covariance
```

### NCE 与 VICReg 是否做 L2 normalize（重要）

- **NCE**：继续对 `fused` / teacher tokens 做 `F.normalize`，与对比学习惯例一致。
- **VICReg**：**不要**在 `vicreg_loss_batch` 里对 `fused`/`base` 再做 `F.normalize`。默认 `gamma=1.0` 针对的是各维标准差；若先 L2 归一化成近似单位向量，256 维下每维标准差量级约 `1/sqrt(256)≈0.0625`，variance 惩罚项会异常偏大、训练不稳定。

第一轮 **VICReg 外部权重** 建议 `vicreg_weight = 0.01`，避免一开始就过强；若曲线仍被 VICReg 主导，可再降到 `0.005`。

## 新增 Fusion Regularization Loss

建议新增独立模块，例如：

```text
experiment_mask_distill/fusion_regularization.py
```

内部包含：

- `hard_cross_modal_nce_loss`
- `soft_cross_modal_nce_loss`
- `vicreg_loss_batch`
- `HybridFusionRegularizationLoss`

trainer 中组合方式：

```text
mask_loss, mask_loss_dict = criterion.compute_loss()
fusion_reg_loss, fusion_reg_dict = fusion_reg_criterion(results)
loss_total = mask_loss + fusion_reg_loss
```

这样 mask loss 和 fusion semantic regularization 的职责分开，便于 ablation。

## 配置建议

`model` 下新增：

```yaml
model:
  alpha_max: 0.2
```

`trainer` 下新增：

```yaml
trainer:
  mask_student_weight: 1.0
  mask_joint_weight: 0.05

  nce_weight: 0.5
  nce_type: "hard"
  nce_tau: 0.1

  vicreg_weight: 0.01
```

如果先做最小实验、不拆 mask loss，可临时使用：

```yaml
trainer:
  mask_distill_weight: 0.5
  nce_weight: 0.5
  vicreg_weight: 0.01
```

## 训练日志必须补充

训练期每个 epoch / step 建议记录：

```text
loss_mask_student
loss_mask_joint
loss_nce
loss_vicreg
loss_fusion_reg
alpha
gate_mean
gate_std
```

验证期建议补诊断读数：

```text
ODISE-only: mask_tokens @ text256
Pixel256:   pixel_tokens @ text256
Base:       base @ text256
Fused:      fused @ text256
```

这些诊断用于判断：

- `pixel_tokens` 是否在 ODISE text256 空间里有语义。
- `base` 是否已经比 ODISE-only 更好。
- `refine` 是否把 `base` 拉坏。
- `fused` 是否真正提升，而不是只提升 mask 对齐。

## 推荐实验顺序

1. 最小版：`0.5 * loss_mask_distill + 0.5 * loss_nce`，先确认 NCE 是否能阻止 Hybrid/Text mIoU 下降。
2. 加 VICReg：`0.5 * loss_mask_distill + 0.5 * loss_nce + 0.01 * loss_vicreg`，观察曲线是否更平稳。
3. 正式方法 A（第一轮推荐）：`1.0 * loss_mask_student + 0.05 * loss_mask_joint + 0.5 * loss_nce + 0.01 * loss_vicreg`，并保持 bounded adaptive alpha（见上文）。
4. soft NCE：如果 hard NCE 不稳定或同类实例互相排斥明显，再把 `nce_type` 改为 `soft`。

## 第一轮推荐 YAML 快照

```yaml
mask_student_weight: 1.0
mask_joint_weight: 0.05
nce_weight: 0.5
nce_type: "hard"
nce_tau: 0.1
vicreg_weight: 0.01
alpha_max: 0.2
```

## TensorBoard 与判断标准

重点看：

```text
Loss/Train_NCE
Loss/Train_VICReg
Loss/Train_MaskStudent
Loss/Train_MaskJoint
Fusion/alpha
Fusion/gate_mean
Metrics/Semantic_mIoU_HybridText
semantic_miou_base
semantic_miou_odise_only
semantic_miou_pixel256
```

经验规则：

- 若 **fused < base**：refine 仍在拉坏语义，保持小 alpha；必要时把 `vicreg_weight` 提到 `0.02` 加强约束。
- 若 **fused ≈ base 但不提升**：第二轮再把 `alpha_raw` 初始改为 `0.0`（初始 alpha=0.1），不要第一轮就做。
- 若 **VICReg 数值仍然很大**：继续降低 `vicreg_weight` 到 `0.005`。
- 若 **NCE 不降**：尝试 `nce_type: "soft"`。

## 一句话总结

方法 A 保留 Diff2Scene 的 mask distillation 来训练 3D student 的几何 mask 预测能力，但不再让该 loss 单独主导可学习的 Hybrid fused token。为避免 fused token 在动态联合训练中退化为仅服务 mask 对齐的 latent classifier，引入跨模态 NCE，使 fused token 与同一 ODISE mask 区域下的 ODISE embedding 和 LSeg-projected embedding 保持一致，并与其他区域区分；同时引入 VICReg 对 pre-refine hybrid token 和 post-refine fused token 进行表示稳定约束，防止 refine 导致语义空间漂移、塌缩和维度冗余。
