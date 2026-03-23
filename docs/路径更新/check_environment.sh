#!/bin/bash
# Mix 项目环境检查脚本

echo "=========================================="
echo "Mix 项目环境检查"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        if [ "$2" != "" ]; then
            count=$(find "$1" -maxdepth 1 -type f 2>/dev/null | wc -l)
            echo "  └─ 包含 $count 个文件"
        fi
    else
        echo -e "${RED}✗${NC} $1 (不存在)"
        return 1
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1 (不存在)"
        return 1
    fi
}

# 检查工作目录
echo "1. 检查工作目录"
check_dir "/home/sunl/work/mix"
echo ""

# 检查数据目录
echo "2. 检查数据目录"
check_dir "/home/sunl/work/mix/data"
check_dir "/home/sunl/work/mix/data/scannet_3d"
check_dir "/home/sunl/work/mix/data/scannet_2d"
check_dir "/home/sunl/work/mix/data/pixel_pooled"
check_dir "/home/sunl/work/mix/data/odise_features"
check_dir "/home/sunl/work/mix/data/scannet_projections"
echo ""

# 检查配置文件
echo "3. 检查配置文件"
check_file "/home/sunl/work/mix/config/data_scannet_3d.yaml"
check_file "/home/sunl/work/mix/config/train_scannet_v2_full_multi_gpu.yaml"
check_file "/home/sunl/work/mix/config/train_scannet_v2.yaml"
echo ""

# 检查训练脚本
echo "4. 检查训练脚本"
check_file "/home/sunl/work/mix/train_open_vocab_v2_ddp.py"
check_file "/home/sunl/work/mix/train_open_vocab_v2.py"
echo ""

# 统计数据量
echo "5. 数据统计"
if [ -d "/home/sunl/work/mix/data/scannet_3d/train" ]; then
    train_count=$(ls /home/sunl/work/mix/data/scannet_3d/train/*.pth 2>/dev/null | wc -l)
    echo -e "${GREEN}训练集:${NC} $train_count 个场景"
fi
if [ -d "/home/sunl/work/mix/data/scannet_3d/val" ]; then
    val_count=$(ls /home/sunl/work/mix/data/scannet_3d/val/*.pth 2>/dev/null | wc -l)
    echo -e "${GREEN}验证集:${NC} $val_count 个场景"
fi
if [ -d "/home/sunl/work/mix/data/pixel_pooled" ]; then
    feature_dirs=$(ls -d /home/sunl/work/mix/data/pixel_pooled/scene* 2>/dev/null | wc -l)
    echo -e "${GREEN}预计算特征:${NC} $feature_dirs 个场景"
fi
echo ""

# 检查 YAML 配置是否正确
echo "6. 验证配置文件内容"
cd /home/sunl/work/mix

# 检查数据配置
python3 -c "
import yaml
try:
    config = yaml.safe_load(open('config/data_scannet_3d.yaml'))
    data_root = config['DATA']['data_root']
    if '/home/sunl/work/mix' in data_root:
        print('✓ data_scannet_3d.yaml 路径正确')
    else:
        print('✗ data_scannet_3d.yaml 路径仍指向旧服务器')
except Exception as e:
    print(f'✗ 配置文件读取失败: {e}')
" 2>&1

# 检查训练配置
python3 -c "
import yaml
try:
    config = yaml.safe_load(open('config/train_scannet_v2_full_multi_gpu.yaml'))
    precomputed = config['dataset']['precomputed_dir']
    if '/home/sunl/work/mix' in precomputed:
        print('✓ train_scannet_v2_full_multi_gpu.yaml 路径正确')
    else:
        print('✗ train_scannet_v2_full_multi_gpu.yaml 路径仍指向旧服务器')
except Exception as e:
    print(f'✗ 配置文件读取失败: {e}')
" 2>&1

echo ""

# 检查 GPU
echo "7. GPU 检查"
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}可用 GPU:${NC} $gpu_count 个"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=, read -r idx name mem; do
        echo "  GPU $idx: $name ($mem)"
    done
else
    echo -e "${YELLOW}⚠${NC} 未找到 nvidia-smi"
fi
echo ""

# 总结
echo "=========================================="
echo "检查完成！"
echo "=========================================="
echo ""
echo "如果所有检查都通过，可以运行训练："
echo ""
echo -e "${YELLOW}多卡训练:${NC}"
echo "  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\"
echo "      train_open_vocab_v2_ddp.py \\"
echo "      --config config/train_scannet_v2_full_multi_gpu.yaml"
echo ""
echo -e "${YELLOW}单卡训练:${NC}"
echo "  python train_open_vocab_v2.py \\"
echo "      --config config/train_scannet_v2_full_multi_gpu.yaml"
echo ""
