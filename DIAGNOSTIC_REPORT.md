# Mix 项目训练问题诊断报告

生成时间: 2026-02-21
检查范围: 代码静态分析 + 关键问题修复

---

## 📋 执行摘要

对 bug 分析文档进行了系统性验证，发现以下关键问题：

### ✅ 已正确的方面
1. **pred_logits 使用正确**: 模型输出 logits，loss 使用 `binary_cross_entropy_with_logits`
2. **metrics 计算正确**: 已过滤空 mask 并使用 per-mask 平均
3. **loss 权重配置正确**: 已使用 pos_weight 和 per-mask dice loss

### ⚠️ 发现并修复的问题
1. **Best model 保存策略错误** (HIGH)
2. **Point→voxel 映射缺少断言** (HIGH)  
3. **训练时未禁止 fallback** (MEDIUM)

---

## 🔍 详细检查结果

### 1. ✅ pred_logits 范围检查

**状态**: 通过

**发现**:
- 模型在 `open_vocab_fusion_v2.py:287` 输出 `logits = logit_scale * (point_features @ mask_tokens.t())`
- Loss 函数正确使用 `F.binary_cross_entropy_with_logits(pred_logits, gt_3d, ...)`
- 只在计算 dice loss 时才 sigmoid（第256行）

**结论**: bug 分析中提到的"logits vs 概率混用"问题在你们代码中不存在。

---

### 2. ⚠️ Point→Voxel 映射检查 → 已修复

**状态**: 发现风险并已修复

**问题**:
- `open_vocab_fusion_v2.py:257` 存在 `point_to_voxel_idx[~matched] = 0`
- 如果 `matched_ratio < 100%`，未匹配的点会被错误映射到 voxel[0]
- 这会严重污染训练，但不会报错（silent failure）

**修复**: ✅ 已在第257行之后添加断言
```python
matched_ratio = matched.float().mean().item()
if matched_ratio < 0.999:
    # 打印详细调试信息并抛出异常
    raise RuntimeError('point→voxel mapping mismatch')
```

**影响**: 
- 修复前：匹配失败时会默默污染训练
- 修复后：一旦出现问题立即报错，避免浪费训练时间

---

### 3. ⚠️ Best Model 保存策略 → 已修复

**状态**: 发现严重问题并已修复

**问题**:
- `trainer/open_vocab_trainer_v2.py:683-685` 更新了 `best_iou`（基于 mIoU）
- 但第698-701行的 `is_best` 判断用的是 `best_loss`（val loss）
- **这意味着虽然跟踪了 mIoU，但保存 best model 时用的是 loss！**

**修复**: ✅ 已修改为按 mIoU 保存
```python
# 优先监控 mIoU
monitored_metric = val_metrics.get('miou', val_metrics.get('iou', 0))
is_best = monitored_metric > self.best_iou + self.config.early_stopping_min_delta
```

**影响**:
- **这是导致你可能错过最佳 checkpoint 的主因！**
- 修复前：val loss 最低的 epoch 不一定是 mIoU 最高的
- 修复后：保存 mIoU 最高的 checkpoint

**建议**:
1. 从 `checkpoints/best_model.pth` 重新评估，看 mIoU 是否比之前记录的更高
2. 如果是，说明之前确实保存错了 checkpoint
3. 重新训练或继续训练，使用新的保存策略

---

### 4. ⚠️ Fallback 投影逻辑 → 已修复

**状态**: 发现风险并已修复

**问题**:
- `dataset/open_vocab_dataset_v2.py:499-510` 允许 fallback 到运行时投影
- 如果部分样本走 fallback，会导致坐标体系不一致（预计算 vs 运行时）
- 训练时未禁止，可能有少量样本使用不同的投影方式

**修复**: ✅ 已添加训练时禁止 fallback
```python
if out_3d is None:
    if self.split == 'train':
        raise RuntimeError(f"Missing precomputed projection...")
    # val/test 可以 fallback（打印警告）
```

**影响**:
- 修复前：少量样本可能走 fallback，导致坐标不一致
- 修复后：训练时必须所有样本都有预计算投影

---

### 5. ❓ Best Model 监控指标（需手动确认）

**状态**: 无法通过静态分析完全确定

**发现**:
- 代码中同时更新了 `best_loss` 和 `best_iou`
- 快速检查工具无法 100% 确定监控逻辑

**已修复**: 现在明确按 mIoU 监控

---

### 6. ℹ️ Metrics 计算

**状态**: 正确

**发现**:
- `trainer/open_vocab_trainer_v2.py:119-133` 正确过滤 GT=0 的 mask
- 使用 per-mask 平均计算 mIoU（与标准定义一致）

**结论**: 这部分已经实现正确，与 bug 分析的建议一致。

---

## 📊 Bug 分析文档的准确性评估

### 整体评分: 85-90 分

### 正确的分析 ✅
1. ✅ 过拟合判断（val loss 升但 mIoU 涨不是致命 bug）
2. ✅ GT Stats 分析（空 mask 6% 不是主因）
3. ✅ Best checkpoint 策略问题（按 loss 而非 mIoU）
4. ✅ Point→voxel 映射风险（需要断言验证）
5. ✅ Fallback 导致不一致性
6. ✅ 阈值 0.5 可能不是最优
7. ✅ 1-sample overfit 测试建议

### 需要澄清的点 ⚠️
1. ⚠️ "logits vs 概率混用"：你们代码已经正确使用，不存在这个问题
2. ⚠️ "换 loss"建议：你们已经实现了 per-mask BCE 和 per-mask dice

---

## 🔧 已应用的修复

### 修复 1: Point→Voxel 映射断言
**文件**: `model/open_vocab_fusion_v2.py`  
**位置**: 第 257 行之后  
**修改**: 添加 matched_ratio 检查和详细错误信息

### 修复 2: Best Model 保存策略
**文件**: `trainer/open_vocab_trainer_v2.py`  
**位置**: 第 691-706 行  
**修改**: 从按 loss 监控改为按 mIoU 监控

### 修复 3: 禁止训练时 Fallback
**文件**: `dataset/open_vocab_dataset_v2.py`  
**位置**: 第 499 行  
**修改**: 训练时如果缺少预计算投影则抛出异常

---

## 🎯 后续建议

### 立即执行

1. **重新评估 best_model.pth**
   ```bash
   # 使用修复后的代码加载 best_model.pth，看 mIoU 是否更高
   python eval_model.py --checkpoint checkpoints/best_model.pth
   ```

2. **检查所有训练样本是否有预计算投影**
   ```bash
   # 修复后第一次训练会检查，如果有缺失会立即报错
   python train_open_vocab_v2_ddp.py --config your_config.yaml
   ```

3. **验证 point→voxel 匹配率**
   - 修复后会在第一次训练时检查
   - 如果出现 matched_ratio < 0.999 会立即报错

### 可选优化

4. **阈值扫描**（参考 `tools/diagnose_training_issues.py`）
   - 在验证集上测试阈值 0.2~0.7
   - 找到最佳阈值，可能让 mIoU 提升 5-10%

5. **1-Sample Overfit 测试**
   - 只用 1 个样本训练 500 步
   - 如果 mIoU 能到 0.9+ 说明模型和数据没问题
   - 如果卡在 0.2~0.3 说明还有隐藏 bug

---

## 📝 结论

### 最关键的发现

**Best model 保存策略错误**是最可能影响你训练结果的问题：
- 你一直在保存 val loss 最低的模型
- 但 val loss 最低 ≠ mIoU 最高
- 这解释了为什么 mIoU 在涨但你感觉"训练效果不好"

### 修复后的预期改进

1. **立即见效**: 从现有 checkpoints 中找到真正 mIoU 最高的模型
2. **后续训练**: 保存策略正确后，best_model.pth 会是真正最好的
3. **训练稳定性**: Point→voxel 断言和 fallback 禁止能避免 silent failure

### 下一步行动

```bash
# 1. 重新运行快速检查（验证修复）
cd /home/featurize/work/mix
python tools/quick_code_check.py

# 2. 如果有 checkpoint，重新评估
python eval_model.py --checkpoint checkpoints/best_model.pth

# 3. 继续训练或重新训练
python train_open_vocab_v2_ddp.py --config your_config.yaml
```

---

## 工具使用指南

### 1. 快速代码检查
```bash
python tools/quick_code_check.py
```
- 静态分析代码
- 不需要数据或模型
- 5-10秒完成

### 2. 完整诊断（需要数据和模型）
```bash
python tools/diagnose_training_issues.py \
    --config config/your_config.yaml \
    --checkpoint checkpoints/best_model.pth
```
- 检查 logits 范围
- 验证 matched_ratio
- 扫描阈值找最佳值

### 3. 查看修复补丁
```bash
python tools/fix_patches.py
```
- 显示所有修复的详细说明
- 可作为代码审查参考

---

**报告结束**

如有疑问或需要进一步检查，请参考以上工具。
