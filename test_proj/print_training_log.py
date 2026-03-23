"""
从 TensorBoard runs 目录读取训练记录，按 epoch 打印关键指标。

用法：
  python test_proj/print_training_log.py
  python test_proj/print_training_log.py --runs-dir runs/distill.1
  python test_proj/print_training_log.py --show-steps
  python test_proj/print_training_log.py --top-k 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_scalars(ea, tag):
    """安全读取 scalar，返回 {step: value} dict"""
    try:
        events = ea.Scalars(tag)
        return {e.step: e.value for e in events}
    except KeyError:
        return {}


def fmt(v, width=9):
    return f"{v:{width}.4f}" if v == v else " " * (width - 3) + "nan"


def main():
    parser = argparse.ArgumentParser(description="打印 TensorBoard 训练记录")
    parser.add_argument("--runs-dir", default="runs/mask_distill.1",
                        help="TensorBoard runs 目录")
    parser.add_argument("--show-steps", action="store_true",
                        help="同时显示 step 级别的 loss（首/中/末三条）")
    parser.add_argument("--top-k", type=int, default=10,
                        help="显示 Top-K 语义类别 IoU（默认10）")
    parser.add_argument("--epoch", type=int, default=None,
                        help="只显示指定 epoch 的 top-k（不填则显示最优 epoch）")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = Path(__file__).resolve().parent.parent / runs_dir

    if not runs_dir.exists():
        print(f"[ERROR] 目录不存在: {runs_dir}")
        sys.exit(1)

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[ERROR] 请安装 tensorboard: pip install tensorboard")
        sys.exit(1)

    print(f"读取: {runs_dir}")
    ea = EventAccumulator(str(runs_dir))
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    print(f"找到 {len(available)} 个 scalar tag\n")

    # ---- epoch 级别指标 ----
    train_loss_ep  = load_scalars(ea, "Loss/Train_Epoch")
    train_distill  = load_scalars(ea, "Loss/Train_MaskDistill_Epoch")
    train_feat     = load_scalars(ea, "Loss/Train_Feat_Epoch")   # experiment_distill 兼容
    val_loss       = load_scalars(ea, "Loss/Val")
    val_distill    = load_scalars(ea, "Loss/Val_MaskDistill")
    val_feat       = load_scalars(ea, "Loss/Val_Feat")           # experiment_distill 兼容
    sem_miou       = load_scalars(ea, "Metrics/Semantic_mIoU")
    n_classes      = load_scalars(ea, "Metrics/N_Valid_Classes")
    mask_miou      = load_scalars(ea, "Metrics/Mask_mIoU")
    n_masks        = load_scalars(ea, "Metrics/N_Masks")
    bin_miou       = load_scalars(ea, "Metrics/Binary_mIoU")     # experiment_distill 兼容
    lr_all         = load_scalars(ea, "LR")

    # ---- 每类 IoU（PerClass_IoU/类名 格式）----
    per_class_tags = sorted(t for t in available if t.startswith("PerClass_IoU/"))
    per_class_data = {}   # {cls_name: {epoch: iou}}
    for tag in per_class_tags:
        cls_name = tag.replace("PerClass_IoU/", "")
        per_class_data[cls_name] = load_scalars(ea, tag)

    # ---- 合并 epoch 列表 ----
    all_epochs = sorted(set(
        list(train_loss_ep.keys()) +
        list(val_loss.keys()) +
        list(sem_miou.keys())
    ))

    if not all_epochs:
        print("[INFO] 暂无 epoch 级别记录（训练可能还未完成第一个 epoch）")
        step_loss = load_scalars(ea, "Loss/Train_Step")
        if step_loss:
            steps = sorted(step_loss.keys())
            print(f"已有 {len(steps)} 个 training step，"
                  f"最新 step={steps[-1]}  loss={step_loss[steps[-1]]:.4f}")
        return

    # ---- 主表格 ----
    W = 100
    print("=" * W)
    print(f"{'Epoch':>5} | {'TrainLoss':>9} | {'TrainDistill':>12} | "
          f"{'ValLoss':>8} | {'ValDistill':>10} | "
          f"{'SemMIoU':>8} | {'Cls':>4} | {'MaskIoU':>8} | {'Masks':>7} | {'LR':>10}")
    print("-" * W)

    best_sem       = 0.0
    best_epoch_sem = -1
    best_mask      = 0.0
    best_epoch_mask = -1

    lr_steps = sorted(lr_all.keys())

    for ep in all_epochs:
        tl = train_loss_ep.get(ep, float("nan"))
        td = train_distill.get(ep, train_feat.get(ep, float("nan")))
        vl = val_loss.get(ep, float("nan"))
        vd = val_distill.get(ep, val_feat.get(ep, float("nan")))
        sm = sem_miou.get(ep, float("nan"))
        nc = n_classes.get(ep, float("nan"))
        mm = mask_miou.get(ep, bin_miou.get(ep, float("nan")))
        nm = n_masks.get(ep, float("nan"))

        # LR：取最新 step 的值（粗略近似，step 级别是连续记录的）
        lr_val = lr_all[lr_steps[-1]] if lr_steps else float("nan")

        best_mark = ""
        if sm == sm:          # not nan
            if sm > best_sem:
                best_sem = sm
                best_epoch_sem = ep
                best_mark += " ★sem"
        if mm == mm:
            if mm > best_mask:
                best_mask = mm
                best_epoch_mask = ep
                best_mark += " ★mask"

        nc_str = f"{int(nc):>4}" if nc == nc else "   -"
        nm_str = f"{int(nm):>7}" if nm == nm else "      -"

        print(
            f"{ep+1:>5} | {fmt(tl)} | {fmt(td):>12} | "
            f"{fmt(vl):>8} | {fmt(vd):>10} | "
            f"{fmt(sm):>8} | {nc_str} | "
            f"{fmt(mm):>8} | {nm_str} | "
            f"{lr_val:>10.2e}"
            + best_mark
        )

    print("=" * W)
    print(f"\n最优语义mIoU : {best_sem:.4f}  @ Epoch {best_epoch_sem+1}")
    print(f"最优MaskIoU  : {best_mask:.4f}  @ Epoch {best_epoch_mask+1}")

    # ---- Top-K 每类 IoU ----
    if per_class_data:
        # 选目标 epoch：命令行指定 > 最优语义 epoch > 最新 epoch
        if args.epoch is not None:
            target_ep = args.epoch - 1  # 用户输入从1开始
        elif best_epoch_sem >= 0:
            target_ep = best_epoch_sem
        else:
            target_ep = all_epochs[-1]

        # 从 per_class_data 里取该 epoch 的每类 IoU
        cls_iou_at_ep = {}
        for cls_name, ep_dict in per_class_data.items():
            # 如果精确匹配不到，取最近的 epoch
            if target_ep in ep_dict:
                cls_iou_at_ep[cls_name] = ep_dict[target_ep]
            elif ep_dict:
                nearest = min(ep_dict.keys(), key=lambda e: abs(e - target_ep))
                cls_iou_at_ep[cls_name] = ep_dict[nearest]

        if cls_iou_at_ep:
            k = min(args.top_k, len(cls_iou_at_ep))
            sorted_cls = sorted(cls_iou_at_ep.items(), key=lambda x: -x[1])

            print(f"\n{'─'*60}")
            print(f"Top-{k} 语义类别 IoU  (Epoch {target_ep+1}，最优语义 epoch)")
            print(f"{'─'*60}")
            for rank, (cls_name, iou) in enumerate(sorted_cls[:k], 1):
                bar = "█" * int(iou * 30)
                print(f"  {rank:>2}. {cls_name:<22} {iou:.4f}  {bar}")
            print(f"{'─'*60}")

            if len(sorted_cls) > k:
                print(f"\nBottom-{min(k, len(sorted_cls)-k)} 最差类别 IoU:")
                for rank, (cls_name, iou) in enumerate(sorted_cls[-k:][::-1], 1):
                    bar = "░" * int(iou * 30)
                    print(f"  {rank:>2}. {cls_name:<22} {iou:.4f}  {bar}")
        else:
            print(f"\n[INFO] Epoch {target_ep+1} 暂无每类 IoU 数据")
            print("      （每类 IoU 从下一次验证后才会写入 TensorBoard）")
    else:
        print(f"\n[INFO] 未找到 PerClass_IoU/* 数据（首次验证前或旧版 trainer）")

    # ---- 可选：step 级 loss 摘要 ----
    if args.show_steps:
        step_loss = load_scalars(ea, "Loss/Train_Step")
        if step_loss:
            steps = sorted(step_loss.keys())
            print(f"\n[Step 级 Train Loss] 共 {len(steps)} 步")
            print(f"  首步  Step {steps[0]:>6}: {step_loss[steps[0]]:.4f}")
            mid = len(steps) // 2
            print(f"  中间  Step {steps[mid]:>6}: {step_loss[steps[mid]]:.4f}")
            print(f"  末步  Step {steps[-1]:>6}: {step_loss[steps[-1]]:.4f}")


if __name__ == "__main__":
    main()
