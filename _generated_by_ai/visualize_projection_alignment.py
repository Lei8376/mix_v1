import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# 避免中文乱码
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "AR PL UMing CN",
    "WenQuanYi Micro Hei",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def visualize_single_sample(proj_path: Path, odise_path: Path, output_path: Optional[Path] = None):
    """
    可视化单个样本的投影对齐情况

    Args:
        proj_path: 投影文件路径 (*_proj.npz)
        odise_path: ODISE 文件路径 (*_odise.npz)
        output_path: 输出图像路径（可选）
    """
    proj_path = Path(proj_path)
    odise_path = Path(odise_path)

    # 加载投影数据
    proj_data = np.load(proj_path)
    x_label = proj_data["x_label"]
    y_label = proj_data["y_label"]

    # 加载 ODISE masks
    odise_data = np.load(odise_path, allow_pickle=True)
    masks = odise_data["masks"]
    if masks.dtype == object:
        masks = np.stack(masks, axis=0)

    K, H, W = masks.shape

    print(f"投影坐标: x 范围 [{x_label.min()}, {x_label.max()}], y 范围 [{y_label.min()}, {y_label.max()}]")
    print(f"Mask 尺寸: K={K}, H={H}, W={W}")
    print(f"投影点数: {len(x_label)}")

    # 创建可视化（最多显示前 4 个 mask）
    num_masks = min(K, 4)
    fig, axes = plt.subplots(2, num_masks, figsize=(5 * num_masks, 10))
    if num_masks == 1:
        axes = axes.reshape(2, 1)

    # 预先计算越界 mask，便于统计
    in_bounds_all = (x_label >= 0) & (x_label < W) & (y_label >= 0) & (y_label < H)

    for i in range(num_masks):
        mask_2d = masks[i]

        # 第一行：显示 mask + 投影点
        ax1 = axes[0, i]
        ax1.imshow(mask_2d, cmap="gray", alpha=0.5)
        ax1.scatter(x_label[in_bounds_all], y_label[in_bounds_all], c="red", s=1, alpha=0.3)
        ax1.set_title(f"Mask {i}: 投影点叠加")
        ax1.set_xlim(0, W)
        ax1.set_ylim(H, 0)
        ax1.grid(True, alpha=0.3)

        oob_ratio = (~in_bounds_all).sum() / len(x_label) * 100.0
        ax1.text(
            5,
            H - 10,
            f"越界: {oob_ratio:.1f}%",
            color="white",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.8),
        )

        # 第二行：显示采样结果（哪些点采样到了这个 mask）
        ax2 = axes[1, i]

        # 采样 mask 值（只在合法坐标上采样）
        x_valid = x_label[in_bounds_all].astype(int)
        y_valid = y_label[in_bounds_all].astype(int)
        sampled_values = mask_2d[y_valid, x_valid]

        # RGB 图：红色通道为 mask，绿色/蓝色为采样结果
        sample_img = np.zeros((H, W, 3), dtype=np.float32)
        sample_img[:, :, 0] = mask_2d  # 红色：mask 区域

        # 在采样点位置标记
        for x, y, val in zip(x_valid, y_valid, sampled_values):
            if val > 0.5:
                sample_img[y, x, 1] = 1.0  # 绿色：点采样到 mask
            else:
                sample_img[y, x, 2] = 1.0  # 蓝色：点采样到背景

        ax2.imshow(sample_img)
        ax2.set_title(f"Mask {i}: 采样结果")

        # 统计
        num_positive = int((sampled_values > 0.5).sum())
        num_negative = int((sampled_values <= 0.5).sum())
        total = num_positive + num_negative
        accuracy = num_positive / total * 100.0 if total > 0 else 0.0

        legend_text = (
            "红色: Mask 区域\n"
            f"绿色: 点采样到 Mask ({num_positive})\n"
            f"蓝色: 点采样到背景 ({num_negative})\n"
            f"采样准确率: {accuracy:.1f}%"
        )
        ax2.text(
            5,
            H - 30,
            legend_text,
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.8),
            verticalalignment="top",
        )
        ax2.set_xlim(0, W)
        ax2.set_ylim(H, 0)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ 保存可视化图像: {output_path}")
    else:
        plt.show()

    plt.close(fig)

    # 返回整体诊断信息
    return {
        "oob_ratio": float((~in_bounds_all).sum() / len(x_label)),
        "num_points": int(len(x_label)),
        "mask_shape": (int(H), int(W)),
        "coord_range": (float(x_label.max()), float(y_label.max())),
    }


def visualize_alignment_grid(proj_dir: Path, odise_dir: Path, output_dir: Path, num_samples: int = 5):
    """
    批量可视化多个样本，生成网格对比

    Args:
        proj_dir: 投影目录（包含 *_proj.npz 的 scene 子目录）
        odise_dir: ODISE 目录（包含 *_odise.npz 的 scene 子目录）
        output_dir: 输出目录
        num_samples: 可视化样本数
    """
    proj_dir = Path(proj_dir)
    odise_dir = Path(odise_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("扫描目录...")
    print(f"  投影: {proj_dir}")
    print(f"  ODISE: {odise_dir}")

    # 找到匹配的样本
    samples: List[dict] = []
    for scene_dir in sorted(proj_dir.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name
        odise_scene_dir = odise_dir / scene_name
        if not odise_scene_dir.exists():
            continue

        for proj_path in sorted(scene_dir.glob("*_proj.npz")):
            frame_stem = proj_path.stem.replace("_proj", "")
            odise_path = odise_scene_dir / f"{frame_stem}_odise.npz"
            if odise_path.exists():
                samples.append(
                    {
                        "scene": scene_name,
                        "frame": frame_stem,
                        "proj_path": proj_path,
                        "odise_path": odise_path,
                    }
                )
                if len(samples) >= num_samples:
                    break

        if len(samples) >= num_samples:
            break

    if not samples:
        print("❌ 未找到匹配的样本")
        return

    print(f"✅ 找到 {len(samples)} 个样本\n")

    diagnostics: List[dict] = []
    for i, sample in enumerate(samples):
        print(f"[{i + 1}/{len(samples)}] 处理: {sample['scene']}/{sample['frame']}")
        out_path = output_dir / f"{sample['scene']}_{sample['frame']}_alignment.png"
        diag = visualize_single_sample(sample["proj_path"], sample["odise_path"], out_path)
        diag["scene"] = sample["scene"]
        diag["frame"] = sample["frame"]
        diagnostics.append(diag)
        print()

    # 生成诊断报告（UTF-8，避免中文乱码）
    report_path = output_dir / "alignment_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("投影和 Mask 对齐情况诊断报告\n\n")

        for diag in diagnostics:
            f.write(f"样本: {diag['scene']}/{diag['frame']}\n")
            f.write(f"  Mask 尺寸: {diag['mask_shape']}\n")
            f.write(
                f"  投影坐标范围: x_max={diag['coord_range'][0]}, "
                f"y_max={diag['coord_range'][1]}\n"
            )
            f.write(f"  投影点数: {diag['num_points']}\n")
            f.write(f"  越界比例: {diag['oob_ratio'] * 100:.2f}%\n")

            H, W = diag["mask_shape"]
            x_max, y_max = diag["coord_range"]

            if x_max < W - 20 and y_max < H - 20:
                f.write("  ⚠️  投影坐标明显小于 mask 尺寸，可能存在分辨率不匹配\n")
            elif x_max > W + 20 or y_max > H + 20:
                f.write("  ⚠️  投影坐标超出 mask 尺寸，需要缩放\n")
            else:
                f.write("  ✅ 投影坐标和 mask 尺寸匹配\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("总结\n")
        f.write("=" * 80 + "\n")
        avg_oob = float(np.mean([d["oob_ratio"] for d in diagnostics]) * 100.0)
        f.write(f"平均越界率: {avg_oob:.2f}%\n")
        if avg_oob > 5:
            f.write("❌ 越界率偏高，可能存在对齐问题\n")
        else:
            f.write("✅ 越界率正常\n")

    print(f"✅ 生成诊断报告: {report_path}")
    print(f"✅ 所有可视化图像保存在: {output_dir}")
    print()
    print("请检查生成的图像，确认：")
    print("  1. 红色投影点是否落在 mask 区域内")
    print("  2. 绿色点（采样到 mask）是否集中在 mask 区域")
    print("  3. 蓝色点（采样到背景）是否在 mask 外")
    print("  4. 是否存在明显的偏移或错位")


def main():
    # 数据在 mix/data 下，基于脚本位置定位到 mix 根目录
    mix_root = Path(__file__).resolve().parent.parent  
    proj_dir = mix_root / "data" / "scannet_projections"
    odise_dir = mix_root / "data" / "pixel_pooled"
    output_dir = mix_root / "test"  # 输出到同目录便于查看

    print("=" * 80)
    print("投影和 Mask 对齐情况可视化工具")
    print("=" * 80)
    print()

    if not proj_dir.exists():
        print(f"❌ 投影目录不存在: {proj_dir}")
        sys.exit(1)
    if not odise_dir.exists():
        print(f"❌ ODISE 目录不存在: {odise_dir}")
        sys.exit(1)

    visualize_alignment_grid(
        proj_dir=proj_dir,
        odise_dir=odise_dir,
        output_dir=output_dir,
        num_samples=4,
    )

    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"查看结果: {output_dir}")


if __name__ == "__main__":
    main()

