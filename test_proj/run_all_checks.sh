#!/bin/bash
# 一键运行四层投影验证
# 用法: bash test_proj/run_all_checks.sh [--max-samples N]
#
# 建议执行顺序：层1 → 层2 → 层3 → 层4 → 可视化

set -e
cd "$(dirname "$0")/.."   # 切到项目根目录

MAX=${1:-100}
SPLIT="train"

echo "=========================================="
echo "  投影链路分层验证工具"
echo "  split=${SPLIT}  max_samples=${MAX}"
echo "=========================================="

echo ""
echo "--- 层1：文件级一致性检查 ---"
python test_proj/check_projection_files.py \
    --split ${SPLIT} \
    --max-samples ${MAX}

echo ""
echo "--- 层2：重算一致性检查（最关键）---"
python test_proj/recompute_projection_consistency.py \
    --split ${SPLIT} \
    --max-samples 50

echo ""
echo "--- 层3：深度一致性检查 ---"
python test_proj/check_projection_depth_consistency.py \
    --split ${SPLIT} \
    --max-samples 50

echo ""
echo "--- 层4：dataset/collate 一致性检查 ---"
python test_proj/check_dataset_projection_pipeline.py \
    --split ${SPLIT} \
    --max-samples 20 \
    --batch-size 4

echo ""
echo "--- 可视化：随机抽样渲染（10个样本）---"
python test_proj/render_projection_failure_cases.py \
    --split ${SPLIT} \
    --max-samples 10 \
    --output-dir test_proj/vis_output

echo ""
echo "=========================================="
echo "  全部检查完成"
echo "  可视化结果: test_proj/vis_output/"
echo "=========================================="
