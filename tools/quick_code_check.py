"""
快速代码检查：静态分析关键问题

不需要运行训练，直接检查代码中的潜在问题
"""

import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_best_model_strategy():
    """检查 best model 保存策略"""
    print("="*60)
    print("🔍 检查 1: Best Model 保存策略")
    print("="*60)
    
    trainer_path = Path(__file__).parent.parent / "trainer" / "open_vocab_trainer_v2.py"
    
    with open(trainer_path, 'r') as f:
        lines = f.readlines()
    
    # 查找 best model 保存逻辑
    save_logic_found = False
    monitor_metric = None
    
    for i, line in enumerate(lines):
        if 'monitored_loss' in line or 'monitored' in line:
            save_logic_found = True
            print(f"\n第 {i+1} 行: {line.rstrip()}")
            
            # 检查监控的是什么指标
            if 'val_metrics["loss"]' in line or 'val_loss' in line:
                monitor_metric = 'loss'
            elif 'val_metrics["miou"]' in line or 'miou' in line:
                monitor_metric = 'miou'
            elif 'val_metrics["iou"]' in line or '"iou"' in line:
                monitor_metric = 'iou'
        
        if 'is_best' in line and 'best_loss' in line:
            print(f"第 {i+1} 行: {line.rstrip()}")
        
        if 'self.best_iou' in line and 'is_best' not in line:
            print(f"第 {i+1} 行: {line.rstrip()}")
    
    print("\n📊 分析结果:")
    if monitor_metric == 'loss':
        print("  ⚠️  **问题**: 当前按 val_loss 保存 best model")
        print("       但你真正关心的是 mIoU，可能错过了 mIoU 最好的 checkpoint")
        print("\n  🔧 修复建议:")
        print("       在 trainer/open_vocab_trainer_v2.py 中修改:")
        print("       ```python")
        print("       # 第 692-704 行左右，改为:")
        print("       monitored_metric = val_metrics.get('miou', val_metrics['iou'])")
        print("       is_best = monitored_metric > self.best_iou + self.config.early_stopping_min_delta")
        print("       if is_best:")
        print("           self.best_iou = monitored_metric")
        print("       ```")
        return False
    elif monitor_metric in ['miou', 'iou']:
        print("  ✅ 当前按 mIoU/IoU 保存 best model，策略正确")
        return True
    else:
        print("  ⚠️  无法确定监控指标，请手动检查")
        return None


def check_point_voxel_mapping():
    """检查 point→voxel 映射逻辑"""
    print("\n" + "="*60)
    print("🔍 检查 2: Point→Voxel 映射逻辑")
    print("="*60)
    
    model_path = Path(__file__).parent.parent / "model" / "open_vocab_fusion_v2.py"
    
    with open(model_path, 'r') as f:
        lines = f.readlines()
    
    # 查找映射逻辑
    in_mapping_section = False
    has_assertion = False
    uses_fallback_zero = False
    
    for i, line in enumerate(lines):
        if 'point_to_voxel_idx' in line or 'matched' in line:
            in_mapping_section = True
            if i > 220 and i < 270:  # 大致范围
                print(f"第 {i+1} 行: {line.rstrip()}")
        
        if in_mapping_section and 'assert' in line and 'matched' in line:
            has_assertion = True
        
        if 'point_to_voxel_idx[~matched] = 0' in line:
            uses_fallback_zero = True
            print(f"⚠️  第 {i+1} 行: {line.rstrip()}")
    
    print("\n📊 分析结果:")
    if uses_fallback_zero:
        print("  ⚠️  **潜在风险**: 未匹配的点被映射到 voxel[0]")
        print("       如果 matched_ratio < 100%，会严重污染训练")
        
        if not has_assertion:
            print("\n  🔧 修复建议:")
            print("       在 model/open_vocab_fusion_v2.py 的映射代码后添加:")
            print("       ```python")
            print("       matched_ratio = matched.float().mean().item()")
            print("       if matched_ratio < 0.999:")
            print("           print(f'[FATAL] point→voxel matched_ratio={matched_ratio:.4f}')")
            print("           # 打印调试信息")
            print("           print(f'  input_coords shape: {input_coords.shape}')")
            print("           print(f'  voxel_coords shape: {voxel_coords.shape}')")
            print("           print(f'  input range: {input_coords.min(0)[0]} ~ {input_coords.max(0)[0]}')")
            print("           raise RuntimeError('point→voxel mapping mismatch')")
            print("       ```")
            return False
        else:
            print("  ✅ 已有断言检查，风险可控")
            return True
    else:
        print("  ✅ 未发现明显的 fallback 问题")
        return True


def check_logits_vs_probs():
    """检查 logits vs 概率使用"""
    print("\n" + "="*60)
    print("🔍 检查 3: Logits vs 概率使用")
    print("="*60)
    
    model_path = Path(__file__).parent.parent / "model" / "open_vocab_fusion_v2.py"
    criterion_path = Path(__file__).parent.parent / "model" / "criterion.py"
    
    # 检查模型输出
    print("\n📁 检查模型输出 (open_vocab_fusion_v2.py):")
    with open(model_path, 'r') as f:
        lines = f.readlines()
    
    uses_sigmoid_in_output = False
    outputs_logits = False
    
    for i, line in enumerate(lines):
        if 'pred_mask_logits' in line and 'append' in line:
            print(f"第 {i+1} 行: {line.rstrip()}")
            outputs_logits = True
        if 'torch.sigmoid' in line or 'F.sigmoid' in line:
            if i > 280 and i < 310:  # 输出附近
                uses_sigmoid_in_output = True
                print(f"⚠️  第 {i+1} 行: {line.rstrip()}")
    
    # 检查 criterion 使用
    print("\n📁 检查 loss 函数 (criterion.py):")
    with open(criterion_path, 'r') as f:
        lines = f.readlines()
    
    uses_bce_with_logits = False
    uses_bce = False
    
    for i, line in enumerate(lines):
        if 'binary_cross_entropy_with_logits' in line:
            uses_bce_with_logits = True
            print(f"第 {i+1} 行: {line.rstrip()}")
        elif 'binary_cross_entropy' in line and 'with_logits' not in line:
            uses_bce = True
            print(f"⚠️  第 {i+1} 行: {line.rstrip()}")
    
    print("\n📊 分析结果:")
    if outputs_logits and uses_bce_with_logits and not uses_sigmoid_in_output:
        print("  ✅ 正确使用 logits + BCEWithLogitsLoss")
        return True
    elif uses_sigmoid_in_output and uses_bce:
        print("  ✅ 正确使用 sigmoid + BCE")
        return True
    else:
        if uses_sigmoid_in_output and uses_bce_with_logits:
            print("  ⚠️  **问题**: 模型输出 sigmoid 概率，但 loss 用 BCEWithLogitsLoss")
            print("       会导致二次 sigmoid（logits → sigmoid → sigmoid），训练不稳定")
        elif outputs_logits and uses_bce:
            print("  ⚠️  **问题**: 模型输出 logits，但 loss 用 BCE")
            print("       BCE 需要 [0,1] 概率，logits 未限制范围")
        
        print("\n  🔧 修复建议:")
        print("       统一使用: logits + BCEWithLogitsLoss（推荐）")
        print("       或: sigmoid + BCE")
        return False


def check_fallback_logic():
    """检查 fallback 逻辑"""
    print("\n" + "="*60)
    print("🔍 检查 4: Fallback 投影逻辑")
    print("="*60)
    
    dataset_path = Path(__file__).parent.parent / "dataset" / "open_vocab_dataset_v2.py"
    
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
    
    has_fallback = False
    fallback_lines = []
    has_train_check = False
    
    for i, line in enumerate(lines):
        if 'out_3d is None' in line or 'return None' in line:
            if i > 120 and i < 200:  # _load_3d_with_precomputed_projection 范围
                has_fallback = True
                fallback_lines.append((i+1, line.rstrip()))
        
        if 'split == "train"' in line and ('raise' in line or 'RuntimeError' in line):
            has_train_check = True
            print(f"✅ 第 {i+1} 行: {line.rstrip()}")
    
    if has_fallback:
        print("\n发现 fallback 逻辑:")
        for line_num, line in fallback_lines[:3]:
            print(f"  第 {line_num} 行: {line}")
    
    print("\n📊 分析结果:")
    if has_fallback and not has_train_check:
        print("  ⚠️  **问题**: 存在 fallback 到运行时投影，但训练时未禁止")
        print("       少量样本走 fallback 会导致坐标体系不一致")
        print("\n  🔧 修复建议:")
        print("       在 dataset/open_vocab_dataset_v2.py 的 __getitem__ 中添加:")
        print("       ```python")
        print("       out_3d = _load_3d_with_precomputed_projection(...)")
        print("       if out_3d is None:")
        print("           if self.split == 'train':")
        print("               raise RuntimeError(")
        print("                   f'Missing precomputed projection: {scene_name}/{frame_stem}'")
        print("               )")
        print("           # val/test 可以 fallback")
        print("       ```")
        return False
    elif has_fallback and has_train_check:
        print("  ✅ 训练时已禁止 fallback，策略正确")
        return True
    else:
        print("  ℹ️  未发现明显的 fallback 逻辑")
        return None


def check_metrics_computation():
    """检查 metrics 计算逻辑"""
    print("\n" + "="*60)
    print("🔍 检查 5: Metrics 计算（mIoU）")
    print("="*60)
    
    trainer_path = Path(__file__).parent.parent / "trainer" / "open_vocab_trainer_v2.py"
    
    with open(trainer_path, 'r') as f:
        lines = f.readlines()
    
    filters_empty_masks = False
    uses_per_mask = False
    
    for i, line in enumerate(lines):
        if 'gt_pos > 0' in line or 'gt_pos >= ' in line:
            if i > 100 and i < 140:  # MetricsTracker 范围
                filters_empty_masks = True
                print(f"第 {i+1} 行: {line.rstrip()}")
        
        if 'per_mask_iou' in line or 'per_mask_acc' in line:
            uses_per_mask = True
    
    print("\n📊 分析结果:")
    if filters_empty_masks and uses_per_mask:
        print("  ✅ 正确过滤空 mask 并使用 per-mask 平均计算 mIoU")
        return True
    elif not filters_empty_masks:
        print("  ⚠️  **问题**: 未过滤 GT=0 的 mask")
        print("       会导致 mIoU 被严重拉低")
        print("\n  🔧 修复建议: 已在代码中实现，检查是否生效")
        return False
    else:
        print("  ℹ️  metrics 计算逻辑正常")
        return True


def main():
    """运行所有检查"""
    print("\n")
    print("🔍 " + "="*56)
    print("🔍  Mix 项目训练问题快速代码检查")
    print("🔍 " + "="*56)
    print()
    
    results = {}
    
    # 运行所有检查
    results['best_model'] = check_best_model_strategy()
    results['point_voxel'] = check_point_voxel_mapping()
    results['logits_probs'] = check_logits_vs_probs()
    results['fallback'] = check_fallback_logic()
    results['metrics'] = check_metrics_computation()
    
    # 汇总报告
    print("\n" + "="*60)
    print("📋 检查汇总")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    unknown = sum(1 for v in results.values() if v is None)
    
    print(f"\n✅ 通过: {passed}")
    print(f"⚠️  需要修复: {failed}")
    print(f"❓ 需要手动检查: {unknown}")
    
    if failed > 0:
        print("\n🔧 优先级修复建议:")
        priority = []
        
        if results['best_model'] is False:
            priority.append("1. 修改 best model 保存策略（按 mIoU 而非 loss）")
        
        if results['point_voxel'] is False:
            priority.append("2. 添加 point→voxel 匹配率断言")
        
        if results['fallback'] is False:
            priority.append("3. 训练时禁止 fallback 投影")
        
        if results['logits_probs'] is False:
            priority.append("4. 修复 logits/概率混用问题")
        
        for item in priority:
            print(f"   {item}")
    else:
        print("\n🎉 代码检查全部通过！")
    
    print("\n" + "="*60)
    print("💡 建议:")
    print("   1. 运行此检查后，根据建议修改代码")
    print("   2. 使用 tools/diagnose_training_issues.py 进行运行时验证")
    print("   3. 在验证集上扫描阈值（0.2~0.6）找最佳阈值")
    print("="*60)
    print()


if __name__ == '__main__':
    main()
