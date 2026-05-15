# Final Method Design

## Final Method

The final method is **Fused Query Alignment + Projected Semantic Gate Readout**.

It separates training-time alignment from inference-time semantic readout:

1. Training uses a fused semantic-region query to supervise 3D point to 2D mask/query alignment.
2. Semantic readout does not classify with the fused training embedding directly.
3. Final open-vocabulary semantics come from dual-space score fusion:
   - `P_odise = sim(ODISE_256_feature, ODISE_text_256)`
   - `P_lseg = sim(LSeg_512_feature, CLIP_text_512)`
   - `P_final = g_sem * P_lseg + (1 - g_sem) * P_odise`
4. `g_sem` is a **projected semantic gate** computed from point/region-level projected consistency. It is a rule gate used at eval/inference time, not a trained MLP gate.

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

- `L_total = L_align`
- `L_align = MaskDistillLoss(pred_mask_logits, lifted_2d_masks)`

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

## Why Fused Query for Alignment

Alignment ablation showed:

- `fused` gives the best semantic result under projected-gate readout
- `odise_only` is worse overall
- `lseg_only` is a useful negative ablation and is not the default

Therefore the default alignment query mode is:

- `alignment_query_mode: fused`

## Why Projected Gate for Semantic Readout

Readout ablation showed:

- `projected_gate` is more stable than `odise_only`
- `projected_gate` is stronger than `lseg_only`
- `projected_gate` is slightly stronger and more stable than simple fixed fusion
- size-aware variants are kept as ablations, not the default

Therefore the default semantic readout mode is:

- `semantic_readout_mode: projected_gate`

## Projected Gate vs Old SourceGate

The final `projected_gate` is **not** the old SourceGate.

Differences:

- no MLP gate training
- no GT semantic supervision
- no text-free MV loss in the default path
- no open-reliability target in the default path
- no GT upper-bound target in the default path

The projected gate is a rule-based readout gate computed from projected semantic consistency across views.

## What Is Main vs Ablation

Main method:

- fused query alignment
- dual-space semantic score readout
- projected semantic gate

Ablations:

- `alignment_query_mode: odise_only`
- `alignment_query_mode: lseg_only`
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

These legacy paths are disabled by default and should only be used for ablation or historical comparison.
