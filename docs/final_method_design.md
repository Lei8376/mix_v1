# Final Method Design

## Final Method

The final method is **Fused Query Alignment + Learned Point-level Projected Semantic Gate**.

It separates training-time alignment from inference-time semantic readout:

1. Training uses a fused semantic-region query to supervise 3D point to 2D mask/query alignment.
2. Semantic readout does not classify with the fused training embedding directly.
3. A point-level semantic gate is predicted from 3D point features and trained from projected multi-view consistency targets.
4. Final open-vocabulary semantics come from dual-space score fusion:
   - `P_odise = sim(ODISE_256_feature, ODISE_text_256)`
   - `P_lseg = sim(LSeg_512_feature, CLIP_text_512)`
   - `P_final = g_pred * P_lseg + (1 - g_pred) * P_odise`
5. `g_pred` is a learned point-level gate. Its target is produced from projected multi-view semantic consistency, without semantic labels or text supervision in training.

## Training Path

Training uses:

- ODISE masks
- lifted 2D masks
- ODISE mask embeddings
- LSeg pooled features
- fused semantic-region queries
- 3D point features
- mask distillation loss

The default training loss is:

- `L_total = L_align + lambda_point_gate * L_point_gate`
- `L_align = MaskDistillLoss(pred_mask_logits, lifted_2d_masks)`
- `L_point_gate = MSE(g_pred, g_target)`

where:

- `pred_mask_logits = pred_3d @ fused_query_tokens.T`

The fused query is used only for query-conditioned mask distillation. It is **not** the final semantic classifier.

## Open-Vocabulary Constraints

Training does **not** use:

- semantic labels
- ScanNet20 semantic CE
- `binary_label_3d`
- `gt_b`
- `F.nll_loss(point_probs, gt_b)`
- text-query supervision
- semantic-query supervision

Text is used only at eval/inference time to produce open-vocabulary readout scores.

The default final configuration keeps the old training-time routing path disabled:

- `use_source_reliability_gate: false`
- `source_gate_train: false`
- `source_gate_training_target: none`
- `use_semantic_query: false`

## Why Fused Query for Alignment

Alignment ablation showed:

- `fused` gives the best semantic result under projected-gate readout
- `odise_only` is worse overall
- `lseg_only` is a useful negative ablation and is not the default

Therefore the default alignment query mode is:

- `alignment_query_mode: fused`

## Why Learned Point Gate for Semantic Readout

The rule-based projected gate remains the teacher/baseline. The learned gate upgrades it by:

- predicting `g_pred` directly from 3D point features
- supervising `g_pred` with projected multi-view ODISE/LSeg consistency
- keeping the final readout in the dual-score space instead of embedding fusion

Therefore the default semantic readout mode is:

- `semantic_readout_mode: learned_point_gate`

## Projected Gate vs Old SourceGate

The learned point gate is **not** the old SourceGate.

Differences:

- no semantic labels
- no text supervision during training
- no ScanNet20 CE
- no GT semantic supervision
- no open-reliability target in the default path
- no GT upper-bound target in the default path

The rule-based `projected_gate` is still kept as a baseline/readout teacher. The final gate is the learned point-level version trained from projected consistency targets.

## What Is Main vs Ablation

Main method:

- fused query alignment
- dual-space semantic score readout
- learned point-level projected semantic gate

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

These legacy paths are disabled by default, archived under legacy configs/modules, and should only be used for ablation or historical comparison. The final semantic classifier does not use `semantic_embeddings` or `fused_embeddings` directly; it uses ODISE/LSeg dual-space score fusion with the projected semantic gate.
