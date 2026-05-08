# 2026-05-07 ODISE 256 Space Probe

## Test Setup

| Item | Value |
|---|---|
| Repo | `/home/sunl/work/mix_v1` |
| Script | `evaluate/odise_256_space_probe.py` |
| Checkpoint | `checkpoints/diff2scene_hybrid_lseg_pc_adaptive_alpha/checkpoint_epoch_20.pth` |
| Checkpoint epoch | `19` |
| Config | `config/train_scannet_v2_full_multi_gpu.yaml` |
| Split | `val` |
| Max samples | `20` |
| Probe train records | `10` |
| Probe test records | `10` |
| Probe fit | `ridge least squares on normalized source512 -> normalized ODISE raw256, bias included` |
| Ridge | `0.001` |
| ODISE text256 | `Panoptic/odise_caption_coco_50e.py word_head.text_proj, prompt: a photo of a {label}` |
| Raw JSON | `record/odise_256_space_probe_2026-05-07.json` |
| Runtime note | `mix` env did not expose CUDA, so this run used CPU |

## Summary

| Method | All20 mIoU | All20 valid | Train10 mIoU | Train10 valid | Test10 mIoU | Test10 valid |
|---|---:|---:|---:|---:|---:|---:|
| `ODISE raw256 @ ODISE text256(photo)` | 0.242701 | 13 | 0.273933 | 13 | 0.213309 | 12 |
| `LSeg512->ODISE256 probe @ ODISE text256(photo)` | 0.267758 | 13 | 0.320896 | 12 | 0.187682 | 13 |
| `Current fused512->ODISE256 probe @ ODISE text256(photo)` | 0.234825 | 14 | 0.274058 | 13 | 0.206615 | 13 |

## Test10 Per-Class IoU

| Class | `ODISE raw256` | `LSeg512->ODISE256` | `fused512->ODISE256` |
|---|---:|---:|---:|
| wall | 0.727830 | 0.760830 | 0.730978 |
| floor | 0.489247 | 0.424392 | 0.485821 |
| cabinet | 0.233022 | 0.373069 | 0.205891 |
| chair | 0.609557 | 0.524222 | 0.612614 |
| sofa | 0.000000 | 0.000000 | 0.000000 |
| table | 0.266660 | 0.280557 | 0.270090 |
| door | 0.000000 | 0.000000 | 0.000000 |
| window | 0.063586 | 0.000000 | 0.199007 |
| picture | nan | 0.000000 | 0.000000 |
| counter | 0.169807 | 0.076796 | 0.181591 |
| refrigerator | 0.000000 | 0.000000 | 0.000000 |
| sink | 0.000000 | 0.000000 | 0.000000 |
| otherfurniture | 0.000000 | 0.000000 | 0.000000 |

## Interpretation

- `LSeg512->ODISE256` is technically feasible: a linear ridge probe can map LSeg's 512D mask features into ODISE's 256D text-readable space, and the projected features still produce meaningful ScanNet20 mIoU.
- It is not stronger than ODISE raw256 on held-out records. Test10 drops from `0.213309` for raw ODISE to `0.187682` for projected LSeg, despite train10 being higher (`0.320896`). This indicates overfitting/space mismatch rather than a clean semantic-space conversion.
- `fused512->ODISE256` is close to raw ODISE on this small test split (`0.206615` vs `0.213309`), but this is still a fitted probe, not evidence that the current fused512 space is inherently ODISE-text-readable.
- For open-vocabulary use, `LSeg512 -> ODISE256 -> ODISE text256` is usable as a diagnostic or auxiliary alignment target, but it is risky as the only semantic route: it compresses CLIP-B/LSeg semantics through ODISE's learned 256D caption-head projection and was validated here only with ScanNet20 prompts.
- A more open-vocabulary-safe variant would preserve both heads: keep LSeg/CLIP-B 512D readout for broad CLIP text compatibility, and add an ODISE256 auxiliary head/loss for compatibility with ODISE's mask/text space.

