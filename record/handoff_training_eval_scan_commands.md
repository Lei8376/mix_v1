# 交接：ScanNet 训练 / 验证命令（给其他对话）

仓库根：`/home/featurize/work/mix_v1`。数据默认见 `config/data_scannet_3d.yaml`（本机常为 `/home/featurize/data/...`）。

---

## 1. 先确认有没有 test 数据

```bash
cd /home/featurize/work/mix_v1

echo "3D train:"
ls /home/featurize/data/scannet_3d/train | head

echo "3D val:"
ls /home/featurize/data/scannet_3d/val | head

echo "3D test:"
ls /home/featurize/data/scannet_3d/test | head
```

若最后一条报 `No such file or directory`，说明没有 `test` 目录，**不要**跑 `--split test` 的 mIoU。

---

## 2. 正常训练（train 训练，val 在 trainer 里验证）

训练脚本**不会**用 test 当训练集；配置里保持：

```yaml
dataset:
  split: train
```

训练过程中的周期性验证数据 loader 使用 **`trainer.eval_split`**（默认 `val`，脚本强制必须为 `val`，不可用 `test` 做训练期验证或选 best）。

```yaml
trainer:
  eval_split: val
```

完整示例（含 PYTHONPATH 与缓存目录）：

```bash
cd /home/featurize/work/mix_v1

PYTHONPATH="/home/featurize/work/mix_v1/ODISE:/home/featurize/work/mix_v1/ODISE/third_party/Mask2Former:$PYTHONPATH" \
CLIP_CACHE_DIR="/home/featurize/work/mix_v1/checkpoints/pretrained/clip" \
TORCH_HOME="/home/featurize/work/mix_v1/checkpoints/pretrained/torch" \
CUDA_VISIBLE_DEVICES=0 \
/home/featurize/work/envs/mix_backup/bin/python train_open_vocab_v2.py \
  --config config/train_scannet_v2_full_multi_gpu.yaml
```

---

## 3. 快速验证：val 抽样（不调参时不要全量）

全量 val 约 26687 个 frame，很慢。调参阶段建议 `--max-samples 500` 或 `1000`：

```bash
cd /home/featurize/work/mix_v1

PYTHONPATH="/home/featurize/work/mix_v1/ODISE:/home/featurize/work/mix_v1/ODISE/third_party/Mask2Former:$PYTHONPATH" \
CLIP_CACHE_DIR="/home/featurize/work/mix_v1/checkpoints/pretrained/clip" \
TORCH_HOME="/home/featurize/work/mix_v1/checkpoints/pretrained/torch" \
CUDA_VISIBLE_DEVICES=0 \
/home/featurize/work/envs/mix_backup/bin/python evaluate/eval_mask_distill_checkpoint.py \
  --checkpoint checkpoints/hfg/checkpoint_epoch_2.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --device cuda \
  --max-samples 500
```

这是 **val** 上的小规模评估，不是 test。

---

## 4. 正式验证：全量 val

需要报告可复现的 semantic mIoU，且 test 无标签时，应以 **全量 val** 为准：

```bash
cd /home/featurize/work/mix_v1

PYTHONPATH="/home/featurize/work/mix_v1/ODISE:/home/featurize/work/mix_v1/ODISE/third_party/Mask2Former:$PYTHONPATH" \
CLIP_CACHE_DIR="/home/featurize/work/mix_v1/checkpoints/pretrained/clip" \
TORCH_HOME="/home/featurize/work/mix_v1/checkpoints/pretrained/torch" \
CUDA_VISIBLE_DEVICES=0 \
/home/featurize/work/envs/mix_backup/bin/python evaluate/eval_mask_distill_checkpoint.py \
  --checkpoint checkpoints/hfg/checkpoint_epoch_2.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split val \
  --device cuda
```

---

## 5. 仅当你确实有 test 数据时才用 `--split test`

`evaluate/eval_mask_distill_checkpoint.py` 默认只允许 **`--split val`**；若要评估 test，必须额外传入 **`--allow-test`**，避免误跑 held-out。

需同时存在类似：

- `/home/featurize/data/scannet_3d/test/`（或与 `data_scannet_3d.yaml` 一致的路径）
- `pixel_pooled` 下对应 test scene 的预计算
- `scannet_projections` 下对应 test scene 的投影

示例：

```bash
cd /home/featurize/work/mix_v1

PYTHONPATH="/home/featurize/work/mix_v1/ODISE:/home/featurize/work/mix_v1/ODISE/third_party/Mask2Former:$PYTHONPATH" \
CLIP_CACHE_DIR="/home/featurize/work/mix_v1/checkpoints/pretrained/clip" \
TORCH_HOME="/home/featurize/work/mix_v1/checkpoints/pretrained/torch" \
CUDA_VISIBLE_DEVICES=0 \
/home/featurize/work/envs/mix_backup/bin/python evaluate/eval_mask_distill_checkpoint.py \
  --checkpoint checkpoints/hfg/checkpoint_epoch_2.pth \
  --config config/train_scannet_v2_full_multi_gpu.yaml \
  --split test \
  --allow-test \
  --device cuda
```

若 test **没有语义标签**，即使能跑也无法得到可靠 mIoU。

---

## 建议流程小结

| 阶段       | 做法 |
|------------|------|
| 训练       | `dataset.split: train`，`trainer.eval_split: val`，trainer 内验证 |
| 调参评估   | `--split val --max-samples 500`（或 1000） |
| 最终对比   | `--split val` 全量 |
| 不要做     | 用 train 做语义评估；无 test 数据/标签时强行 `--split test` |

说明：你看到的「26687」是 **val split 的全量 frame 数**，不是「验证了全部 train」。调参请加 `--max-samples`。
