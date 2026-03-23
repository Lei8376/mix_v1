# Mix 项目路径更新完成

## 📁 更新文档位置

所有路径更新相关的文档已整理到：`docs/路径更新/`

### 文档列表

1. **快速启动.md** - 快速开始训练指南
2. **路径更新记录.md** - 详细的更新记录
3. **check_environment.sh** - 环境检查脚本

### 查看文档

```bash
cd /home/sunl/work/mix/docs/路径更新

# 查看快速启动指南
cat 快速启动.md

# 查看更新记录
cat 路径更新记录.md

# 运行环境检查
bash check_environment.sh
```

## ✅ 更新总结

**已更新 17 个核心文件**，包括：
- 5 个配置文件（config/*.yaml）
- 9 个 Python 脚本
- 3 个测试脚本

**路径更新**：所有 `/home/featurize` 已改为 `/home/sunl/work/mix`

**数据确认**：
- 训练集: 1201 个场景 ✓
- 验证集: 312 个场景 ✓
- 预计算特征: 1513 个场景 ✓

## 🚀 开始训练

```bash
cd /home/sunl/work/mix

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    train_open_vocab_v2_ddp.py \
    --config config/train_scannet_v2_full_multi_gpu.yaml
```

---
更新时间: 2026-03-06
