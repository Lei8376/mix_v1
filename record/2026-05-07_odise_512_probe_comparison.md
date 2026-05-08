# 2026-05-07 ODISE 512 Probe Comparison

## Test Setup

| Item | Value |
|---|---|
| Repo | `/home/sunl/work/mix_v1` |
| Checkpoint | `checkpoints/diff2scene_hybrid_lseg_pc_adaptive_alpha/checkpoint_epoch_20.pth` |
| Checkpoint epoch | `19` |
| Config | `config/train_scannet_v2_full_multi_gpu.yaml` |
| Split | `val` |
| Max samples | `20` |
| Records used | `20` |
| Probe train records | `10` |
| Probe test records | `10` |
| Probe fit | `ridge least squares on normalized ODISE raw256 -> normalized LSeg raw512, bias included` |
| Ridge | `0.001` |
| TextB | `open_clip ViT-B-32, prompt: a {label} in a scene` |
| Text256 photo | `open_clip ViT-L-14 text768 + ODISE word_head.text_proj, prompt: a photo of a {label}` |
| Text256 scene | `open_clip ViT-L-14 text768 + ODISE word_head.text_proj, prompt: a {label} in a scene` |
| Raw JSON | `record/odise_512_probe_comparison_2026-05-07.json` |

## Summary

| Method | All20 mIoU | All20 valid | Train10 mIoU | Train10 valid | Test10 mIoU | Test10 valid |
|---|---:|---:|---:|---:|---:|---:|
| `ODISE raw256 @ ODISE text256(photo)` | 0.256471 | 12 | 0.314247 | 11 | 0.211182 | 12 |
| `ODISE raw256 @ ODISE text256(scene)` | 0.178057 | 14 | 0.209325 | 14 | 0.178572 | 12 |
| `LSeg raw512 @ CLIP-B text512` | 0.282567 | 13 | 0.345964 | 12 | 0.204916 | 13 |
| `Current mask_proj 256->512 @ CLIP-B` | 0.047320 | 13 | 0.046935 | 12 | 0.059554 | 12 |
| `Current pixel_proj 512->512 @ CLIP-B` | 0.006065 | 16 | 0.000812 | 15 | 0.014346 | 15 |
| `Current fused512 @ CLIP-B` | 0.044698 | 15 | 0.048427 | 14 | 0.043597 | 15 |
| `Trained ODISE256->512 probe @ CLIP-B` | 0.293798 | 11 | 0.377496 | 11 | 0.205819 | 11 |

## Interpretation

- `current_odise_proj512_textB` is the projector already inside the current fusion model. It is trained by the current model objective, not by semantic probe supervision.
- `trained_probe_odise256_to512_textB` is newly fit on the first 10 records to map raw ODISE 256D tokens to raw LSeg 512D tokens, then evaluated on the last 10 records.
- Therefore these two 512D ODISE paths answer different questions: current projector tests the trained fusion model; probe tests whether ODISE 256D information is transferable to a CLIP/LSeg 512D semantic space.
- On the same current `mix_v1` setup, the probe test10 result is much higher than the current projector/fused result, which supports adding an explicit semantic-alignment head/loss instead of relying on mask distillation to learn the 512D space.

## All20 Per-Class IoU

| Class | `ODISE raw256 @ ODISE text256(photo)` | `ODISE raw256 @ ODISE text256(scene)` | `LSeg raw512 @ CLIP-B text512` | `Current mask_proj 256->512 @ CLIP-B` | `Current pixel_proj 512->512 @ CLIP-B` | `Current fused512 @ CLIP-B` | `Trained ODISE256->512 probe @ CLIP-B` |
|---|---:|---:|---:|---:|---:|---:|---:|
| wall | 0.715950 | 0.690030 | 0.730758 | 0.436258 | 0.000000 | 0.459009 | 0.709456 |
| floor | 0.755031 | 0.753786 | 0.742721 | 0.000000 | 0.000000 | 0.000000 | 0.779100 |
| cabinet | 0.371063 | 0.262279 | 0.467691 | 0.169708 | 0.000000 | 0.207840 | 0.379326 |
| chair | 0.614133 | 0.364716 | 0.614880 | 0.000000 | 0.029020 | 0.000000 | 0.575056 |
| sofa | 0.000000 | 0.000000 | nan | nan | 0.000000 | nan | nan |
| table | 0.149308 | 0.111255 | 0.174423 | 0.000000 | 0.000000 | 0.000000 | 0.163527 |
| door | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.000000 | 0.000000 |
| window | 0.295447 | 0.245827 | 0.485956 | 0.000000 | 0.005342 | 0.000000 | 0.170405 |
| counter | 0.165301 | 0.039765 | 0.421159 | 0.000000 | 0.000000 | 0.003297 | 0.410928 |
| refrigerator | 0.000000 | 0.025003 | 0.035782 | 0.009192 | 0.062670 | 0.000321 | 0.043980 |
| sink | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| otherfurniture | 0.011419 | 0.000139 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Train10 probe-fit Per-Class IoU

| Class | `ODISE raw256 @ ODISE text256(photo)` | `ODISE raw256 @ ODISE text256(scene)` | `LSeg raw512 @ CLIP-B text512` | `Current mask_proj 256->512 @ CLIP-B` | `Current pixel_proj 512->512 @ CLIP-B` | `Current fused512 @ CLIP-B` | `Trained ODISE256->512 probe @ CLIP-B` |
|---|---:|---:|---:|---:|---:|---:|---:|
| wall | 0.700301 | 0.705304 | 0.700719 | 0.327419 | 0.000000 | 0.425567 | 0.701563 |
| floor | 0.855333 | 0.849831 | 0.871205 | 0.000000 | 0.000000 | 0.000000 | 0.871302 |
| cabinet | 0.490497 | 0.493537 | 0.531258 | 0.228089 | 0.000000 | 0.244914 | 0.547075 |
| chair | 0.626907 | 0.272674 | 0.690651 | 0.000000 | 0.001373 | 0.000000 | 0.681543 |
| table | 0.099451 | 0.101997 | 0.200545 | 0.000000 | 0.000000 | 0.000000 | 0.193792 |
| door | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.000000 | 0.000000 |
| window | 0.526398 | 0.470763 | 0.596127 | 0.000000 | 0.000000 | 0.000000 | 0.594366 |
| counter | 0.157826 | 0.000000 | 0.509003 | 0.000000 | 0.000000 | 0.006792 | 0.510757 |
| refrigerator | 0.000000 | 0.036442 | 0.052060 | 0.007713 | 0.010814 | 0.000701 | 0.052060 |
| sink | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| otherfurniture | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Test10 probe-eval Per-Class IoU

| Class | `ODISE raw256 @ ODISE text256(photo)` | `ODISE raw256 @ ODISE text256(scene)` | `LSeg raw512 @ CLIP-B text512` | `Current mask_proj 256->512 @ CLIP-B` | `Current pixel_proj 512->512 @ CLIP-B` | `Current fused512 @ CLIP-B` | `Trained ODISE256->512 probe @ CLIP-B` |
|---|---:|---:|---:|---:|---:|---:|---:|
| wall | 0.727953 | 0.678176 | 0.753256 | 0.574816 | 0.000000 | 0.495851 | 0.715440 |
| floor | 0.488142 | 0.507922 | 0.431249 | 0.000000 | 0.000000 | 0.000000 | 0.535540 |
| cabinet | 0.277886 | 0.159643 | 0.394215 | 0.129393 | 0.000000 | 0.158099 | 0.255281 |
| chair | 0.587710 | 0.540801 | 0.481780 | 0.000000 | 0.092154 | 0.000000 | 0.397671 |
| sofa | 0.000000 | 0.000000 | nan | nan | 0.000000 | nan | nan |
| table | 0.193690 | 0.119135 | 0.152934 | 0.000000 | 0.000000 | 0.000000 | 0.140299 |
| door | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.000000 | 0.000000 |
| window | 0.063930 | 0.063370 | 0.128764 | 0.000000 | 0.011574 | 0.000000 | 0.016260 |
| counter | 0.169631 | 0.073181 | 0.321711 | 0.000000 | 0.000000 | 0.000000 | 0.168543 |
| refrigerator | 0.000000 | 0.000000 | 0.000000 | 0.010444 | 0.111465 | 0.000000 | 0.034973 |
| sink | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| otherfurniture | 0.025247 | 0.000631 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Data Source

- Full machine-readable data: [odise_512_probe_comparison_2026-05-07.json](odise_512_probe_comparison_2026-05-07.json)
