#!/usr/bin/env python
"""测试读取 npz 文件并打印结果（模拟图片中的输出格式）"""
import numpy as np
from pathlib import Path

# 读取一个测试输出文件
npz_file = Path("./test_output/0_odise.npz")

if not npz_file.exists():
    print(f"文件不存在: {npz_file}")
    exit(1)

# 加载数据
data = np.load(npz_file, allow_pickle=True)

print(f"\n加载文件: {npz_file}")
print(f"文件包含的键: {list(data.keys())}")
print()

# 打印数组信息
for key in data.keys():
    arr = data[key]
    print(f"key: {key:30s} {type(arr).__name__:15s} shape/len: {arr.shape if hasattr(arr, 'shape') else len(arr)}")
    
    # 如果是 mask_embeddings，打印部分内容
    if 'mask_embeddings' in key and hasattr(arr, 'shape'):
        print(arr)
    
    # 如果是 info_array，打印详细信息
    if 'info' in key:
        print(arr)
        if hasattr(arr, '__iter__'):
            for item in arr:
                print(item)

print()
