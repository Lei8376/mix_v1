#!/usr/bin/env python3
"""
最终验证：确认所有修复已应用
"""

import sys
from pathlib import Path

def check_fix(file_path, search_text, fix_name):
    """检查文件中是否包含修复代码"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    if search_text in content:
        print(f"✅ {fix_name}: 已应用")
        return True
    else:
        print(f"❌ {fix_name}: 未找到")
        return False

def main():
    root = Path(__file__).parent.parent
    
    print("="*60)
    print("🔍 最终验证：检查所有修复是否已应用")
    print("="*60)
    print()
    
    fixes = []
    
    # 修复 1: Point→Voxel 匹配率断言
    fixes.append(check_fix(
        root / "model" / "open_vocab_fusion_v2.py",
        "matched_ratio = matched.float().mean().item()",
        "修复1: Point→Voxel 匹配率断言"
    ))
    
    # 修复 2: Best model 按 mIoU 保存
    fixes.append(check_fix(
        root / "trainer" / "open_vocab_trainer_v2.py",
        "monitored_metric = val_metrics.get('miou', val_metrics.get('iou', 0))",
        "修复2: Best model 按 mIoU 保存"
    ))
    
    # 修复 3: 训练时禁止 fallback
    fixes.append(check_fix(
        root / "dataset" / "open_vocab_dataset_v2.py",
        "if self.split == 'train':\n                raise RuntimeError(\n                    f\"Missing precomputed projection",
        "修复3: 训练时禁止 fallback"
    ))
    
    print()
    print("="*60)
    
    passed = sum(fixes)
    total = len(fixes)
    
    if passed == total:
        print(f"🎉 所有修复已应用！({passed}/{total})")
        print()
        print("✅ 代码已准备好重新训练/评估")
        print()
        print("建议后续步骤:")
        print("  1. 重新评估 best_model.pth")
        print("  2. 继续训练（会自动验证修复）")
        print("  3. 扫描阈值找最佳值")
        return 0
    else:
        print(f"⚠️  部分修复未应用 ({passed}/{total})")
        print()
        print("请检查上述失败的修复项")
        return 1

if __name__ == '__main__':
    sys.exit(main())
