# Mix 项目 - 路径更新完成 ✅

## 更新说明

所有数据路径已从旧服务器（`/home/featurize`）更新到新环境（`/home/sunl/work/mix`）。

## 📁 相关文档

所有更新文档已整理到：**`docs/路径更新/`**

```
docs/路径更新/
├── README.md              # 概览和快速链接
├── 快速启动.md            # 训练启动指南
├── 路径更新记录.md        # 详细的更新记录（17个文件）
└── check_environment.sh   # 环境检查脚本
```

## 🚀 快速开始

```bash
# 1. 检查环境（推荐）
bash docs/路径更新/check_environment.sh

# 2. 开始训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    train_open_vocab_v2_ddp.py \
    --config config/train_scannet_v2_full_multi_gpu.yaml
```

## 📊 数据概况

- ✅ 训练集: 1201 个场景
- ✅ 验证集: 312 个场景  
- ✅ 预计算特征: 1513 个场景
- ✅ 所有配置文件已更新并验证

## 📖 查看详细文档

```bash
# 查看完整文档
cat docs/路径更新/README.md

# 查看快速启动指南
cat docs/路径更新/快速启动.md

# 查看详细更新记录
cat docs/路径更新/路径更新记录.md
```

---
**准备就绪！可以直接运行训练了。** 🎉

更新时间: 2026-03-06
