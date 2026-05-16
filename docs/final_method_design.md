# Final Method Design

## Final Method

The final method is **Fused Query Alignment + No-text Learned Region Reliability Gate**.

It separates 3D-2D alignment from open-vocabulary semantic readout:

1. Training uses a fused semantic-region query to supervise 3D point to 2D mask/query alignment.
2. Semantic readout never classifies directly with the fused training embedding.
3. A learned **region-level** gate predicts whether each region should trust LSeg more or ODISE more.
4. The gate is trained from **no-text**, **no-label** teacher signals:
   - multi-view same-source consistency
   - same-source neighborhood sharpness
5. Final semantics come from dual-space score fusion:
   - `P_odise = sim(ODISE_256_feature, ODISE_text_256)`
   - `P_lseg = sim(LSeg_512_feature, CLIP_text_512)`
   - `P_final = g_pred * P_lseg + (1 - g_pred) * P_odise`

## Training Objective

The default training objective is:

- `L_total = L_align + lambda_region_gate * L_region_gate`
- `L_align = MaskDistillLoss(pred_mask_logits, lifted_2d_masks)`
- `L_region_gate = MSE(g_pred, g_target)`

where:

- `pred_mask_logits = pred_3d @ fused_query_tokens.T`
- `g_pred = sigmoid(region_gate_mlp(region_gate_input))`
- `g_target = clamp(sigmoid(5.0 * R_diff), 0.35, 0.85)`

The region-gate target uses:

- `R_diff = 1.0 * (C_lseg - C_odise) + 0.5 * (sharp_lseg - sharp_odise)`

The region-gate input uses:

- `fused_region_query`
- `C_lseg`, `C_odise`, `C_diff`
- `sharp_lseg`, `sharp_odise`, `sharp_diff`
- `response_margin`, `response_conf`
- `mask_area_ratio`
- `lifted_point_count`
- `overlap_iou_mean`

The fused query is used only for **query-conditioned mask distillation**. It is not the final semantic classifier.

## Open-Vocabulary Constraints

Training does **not** use:

- semantic labels
- ScanNet20 semantic CE
- `binary_label_3d`
- `gt_b`
- `F.nll_loss(point_probs, gt_b)`
- text-query supervision
- semantic-query supervision
- text-response entropy
- ODISE/LSeg text distributions as gate targets

Training uses only:

- ODISE masks
- lifted 2D masks
- ODISE mask embeddings
- LSeg pooled features
- fused semantic-region queries
- 3D point features
- no-text region reliability signals

Text is used only at eval/inference time for open-vocabulary score readout.

## Why Fused Query for Alignment

Alignment ablation showed:

- `fused` gives the best semantic result under projected-gate readout
- `odise_only` is worse overall
- `lseg_only` is a negative ablation and not the default

Therefore the default alignment query mode is:

- `alignment_query_mode: fused`

## Why Learned Region Gate

The final gate is not point-level and is not the old SourceGate.

It learns at the **region level** because:

- region-level signals are cheaper than online point-level target building
- multi-view consistency is strongest at the region level
- sharpness complements consistency and often carries ODISE-friendly structure
- quality signals help learnability even when they are not part of the target

The rule-based `projected_gate` is still kept as a baseline/readout teacher, but the final default readout is:

- `semantic_readout_mode: learned_region_gate`

## Main Method vs Ablation

Main method:

- fused query alignment
- dual-space semantic score readout
- learned no-text region reliability gate

Ablations:

- `alignment_query_mode: odise_only`
- `alignment_query_mode: lseg_only`
- rule-based `projected_gate`
- fixed fusion weights
- size-aware fusion
- projected-size fusion

Legacy experiments:

- old SourceGate MLP
- TextFreeMV
- open-reliability gate targets
- GT upper-bound gate targets
- dual-branch probes
- projected-sem probe logging
- learned point-level gate

These legacy paths are disabled by default and kept only for ablation or historical comparison.
