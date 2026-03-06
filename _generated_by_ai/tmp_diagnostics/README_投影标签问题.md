# 投影标签 (x_label/y_label) 问题 - 快速指南

## 🔍 问题确认

运行诊断脚本:
```bash
cd /home/featurize/work/mix
python scripts/diagnose_projection_issue.py
```

如果看到:
```
⚠️  Tuple 格式 - 没有 x_label/y_label
```

说明你的数据需要生成投影标签。

## ✅ 解决方案

### 选项 1: 快速测试(推荐先做)

测试单个场景,确保流程正常:
```bash
python scripts/test_projection_generation.py --scene scene0000_00
```

如果成功,会看到:
```
✅ 测试成功!
   有效投影点: 45000+
   覆盖率: 55%+
```

### 选项 2: 生成所有数据

**方法 A - 使用脚本(推荐)**:
```bash
bash scripts/generate_projection_labels.sh
```

**方法 B - 手动运行**:
```bash
# 处理训练集
python tools/generate_projection_labels.py \
    --data-root /home/featurize/data/scannet_3d \
    --data-2d-root /home/featurize/data/scannet_2d \
    --output-dir /home/featurize/data/scannet_3d_with_projection \
    --split train

# 处理验证集(如果需要)
python tools/generate_projection_labels.py \
    --data-root /home/featurize/data/scannet_3d \
    --data-2d-root /home/featurize/data/scannet_2d \
    --output-dir /home/featurize/data/scannet_3d_with_projection \
    --split val
```

**预计时间**: 
- 单个场景: ~10-30 秒
- 977 个训练场景: 约 3-8 小时

### 选项 3: 更新配置

生成完成后,修改 `config/data_scannet_3d.yaml`:
```yaml
DATA:
  data_root: /home/featurize/data/scannet_3d_with_projection  # 改这里
  data_root_2d: /home/featurize/data/scannet_2d
  # ... 其他配置不变
```

### 选项 4: 开始训练

```bash
python train_open_vocab_v2.py \
    --config config/train_scannet_v2_minimal.yaml \
    --num-epochs 5
```

## 📊 验证训练正常

### 正常情况:
```
Epoch [1/5] Step [0/100] Loss: 0.8234 (0.8234) LR: 5.00e-05
```
- Loss 不为 0
- 逐渐下降

### 异常情况:
```
Warning: All x_label/y_label are 0 for batch 0, skipping
Epoch [1/5] Step [0/100] Loss: 0.0000 (0.0000) LR: 5.00e-05
```
- Loss 一直为 0
- 大量警告

## 🛠️ 故障排除

### 问题 1: 没有 2D 数据

**错误**:
```
❌ 2D 数据目录不存在: /home/featurize/data/scannet_2d
```

**解决**:
- 下载完整的 ScanNet 数据集
- 或使用纯 3D 训练方法(不需要 2D)

### 问题 2: 投影点太少

**警告**:
```
⚠️  没有找到有效的投影
```

**原因**:
- 相机参数不正确
- 深度图缺失或损坏
- 3D 点云和 2D 图像不对应

**检查**:
```bash
# 检查场景结构
ls /home/featurize/data/scannet_2d/scene0000_00/
# 应该有: color/ depth/ pose/
```

### 问题 3: 磁盘空间不足

**症状**: 生成过程中断

**解决**:
- 检查磁盘空间: `df -h`
- 只处理部分场景: `--scene-pattern "scene000*"`
- 使用动态投影(方案 2,见完整文档)

## 📚 相关文档

- **详细说明**: `docs/投影标签问题解决方案.md`
- **技术原理**: 查看 `tools/generate_projection_labels.py` 注释
- **代码位置**: 
  - 数据加载: `dataset/open_vocab_dataset_v2.py`
  - Loss 计算: `model/criterion.py`
  - 投影工具: `utils/mapping_util.py`

## 🎯 核心要点

1. **问题**: `.pth` 文件是旧格式 `(locs, feats, labels)`,缺少投影标签
2. **原因**: 投影标签需要用相机参数计算,不是原始数据的一部分
3. **影响**: 没有投影标签,训练时 loss 为 0,无法学习
4. **解决**: 运行生成脚本,创建包含投影标签的新数据文件
5. **时间**: 约 3-8 小时处理全部训练数据

## ❓ 常见问题

**Q: 为什么之前没发现这个问题?**
A: 代码会用零填充缺失的标签,但零标签会导致训练失败。

**Q: 能否跳过投影标签生成?**
A: 不能,这是 3D-2D 融合训练的必需数据。

**Q: 生成一次后还需要再生成吗?**
A: 不需要,生成的数据可以重复使用。

**Q: 可以并行处理加速吗?**
A: 可以,修改脚本使用多进程(需要自己实现)。

## 🚀 快速开始(TL;DR)

```bash
# 1. 确认问题
python scripts/diagnose_projection_issue.py

# 2. 测试单个场景
python scripts/test_projection_generation.py

# 3. 生成所有数据(需要几小时)
bash scripts/generate_projection_labels.sh

# 4. 更新配置
# 编辑 config/data_scannet_3d.yaml
# data_root: /home/featurize/data/scannet_3d_with_projection

# 5. 开始训练
python train_open_vocab_v2.py --config config/train_scannet_v2_minimal.yaml --num-epochs 5
```

---

**需要帮助?** 查看 `docs/投影标签问题解决方案.md` 获取完整文档。
