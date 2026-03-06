#!/bin/bash
# 快速测试评估脚本是否可以运行

echo "============================================================"
echo "Mix 模型评估环境测试"
echo "============================================================"
echo ""

# 1. 检查环境
echo "1. 检查 Python 环境..."
if [ -f "/home/featurize/work/envs/mix_backup/bin/activate" ]; then
    source /home/featurize/work/envs/mix_backup/bin/activate
    echo "   ✓ 环境已激活"
else
    echo "   ✗ 找不到环境: /home/featurize/work/envs/mix_backup/bin/activate"
    exit 1
fi

echo "   Python: $(which python)"
echo "   版本: $(python --version)"
echo ""

# 2. 检查依赖
echo "2. 检查依赖模块..."
python -c "import torch; print(f'   ✓ PyTorch: {torch.__version__}')" 2>/dev/null || echo "   ✗ PyTorch 未安装"
python -c "import MinkowskiEngine; print('   ✓ MinkowskiEngine: OK')" 2>/dev/null || echo "   ✗ MinkowskiEngine 未安装"
python -c "import clip; print('   ✓ CLIP: OK')" 2>/dev/null || echo "   ✗ CLIP 未安装"
python -c "import yaml; print('   ✓ PyYAML: OK')" 2>/dev/null || echo "   ✗ PyYAML 未安装"
echo ""

# 3. 检查文件
echo "3. 检查必要文件..."
if [ -f "eval_mix_model.py" ]; then
    echo "   ✓ eval_mix_model.py 存在"
else
    echo "   ✗ eval_mix_model.py 不存在"
fi

if [ -d "../openscene" ]; then
    echo "   ✓ openscene 目录存在"
else
    echo "   ✗ openscene 目录不存在"
fi

if [ -f "../openscene/util/metric.py" ]; then
    echo "   ✓ openscene/util/metric.py 存在"
else
    echo "   ✗ openscene/util/metric.py 不存在"
fi
echo ""

# 4. 检查权重文件
echo "4. 检查可用的权重文件..."
if [ -d "checkpoints/full.1" ]; then
    echo "   ✓ checkpoints/full.1/ 存在"
    echo "   可用的权重文件:"
    ls -lh checkpoints/full.1/*.pth 2>/dev/null | awk '{print "     - " $9 " (" $5 ")"}'
else
    echo "   ✗ checkpoints/full.1/ 不存在"
fi

if [ -d "checkpoints/full.4" ]; then
    echo "   ✓ checkpoints/full.4/ 存在"
    echo "   可用的权重文件:"
    ls -lh checkpoints/full.4/*.pth 2>/dev/null | awk '{print "     - " $9 " (" $5 ")"}'
fi
echo ""

# 5. 测试导入
echo "5. 测试模块导入..."
python << 'EOF'
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'openscene'))

try:
    from MinkowskiEngine import SparseTensor
    print("   ✓ MinkowskiEngine.SparseTensor")
except Exception as e:
    print(f"   ✗ MinkowskiEngine: {e}")

try:
    from util import metric
    print("   ✓ openscene.util.metric")
except Exception as e:
    print(f"   ✗ openscene.util.metric: {e}")

try:
    from dataset.label_constants import SCANNET_LABELS_20
    print(f"   ✓ SCANNET_LABELS_20 ({len(SCANNET_LABELS_20)} 类)")
except Exception as e:
    print(f"   ✗ label_constants: {e}")

try:
    from model.open_vocab_fusion_v2 import OpenVocab3DFusionModelV2
    print("   ✓ OpenVocab3DFusionModelV2")
except Exception as e:
    print(f"   ✗ Mix model: {e}")

try:
    import clip
    print("   ✓ CLIP")
except Exception as e:
    print(f"   ✗ CLIP: {e}")
EOF

echo ""
echo "============================================================"
echo "环境测试完成"
echo "============================================================"
echo ""
echo "如果所有检查都通过 (✓)，可以运行:"
echo "  python eval_mix_model.py --checkpoint checkpoints/full.1/checkpoint_epoch_19.pth"
echo ""
echo "详细使用说明请查看: EVAL_README.md"
echo ""
