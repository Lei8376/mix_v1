"""
诊断 criterion 问题的脚本
"""
import torch
import sys
sys.path.insert(0, '.')
from dataset.open_vocab_dataset_v2 import OpenVocabDatasetV2Config, OpenVocabScannetDatasetV2, open_vocab_collate_v2
from model.open_vocab_fusion_v2 import OpenVocab3DFusionModelV2, OpenVocabFusionModelV2Config
from model.criterion import Criteria
import MinkowskiEngine as ME

# 创建配置
config = OpenVocabDatasetV2Config(
    data_config_path='config/data_scannet_3d.yaml',
    precomputed_dir='/home/featurize/data/pixel_pooled',
    projection_dir='/home/featurize/data/scannet_projections',
    split='train',
    max_samples_ratio=0.01,
)

# 创建 dataset
dataset = OpenVocabScannetDatasetV2(config)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=open_vocab_collate_v2)
batch = next(iter(loader))

# 创建模型
model_config = OpenVocabFusionModelV2Config()
model = OpenVocab3DFusionModelV2(model_config)
model.eval()

# 将 batch 移动到 GPU
device = 'cuda'
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)
model = model.to(device)

# 创建 sinput
batch["sinput"] = ME.SparseTensor(batch["feat_3d"], batch["coords_3d"])

# 前向传播
with torch.no_grad():
    results = model(batch)

print("=" * 60)
print("诊断 criterion 问题")
print("=" * 60)

# 检查关键数据
print(f"\n1. batch_input 信息：")
print(f"   x_label shape: {batch['x_label'].shape}")
print(f"   y_label shape: {batch['y_label'].shape}")
print(f"   ori_coords_3d shape: {batch['ori_coords_3d'].shape}")
print(f"   x_label 范围: [{batch['x_label'].min()}, {batch['x_label'].max()}]")
print(f"   y_label 范围: [{batch['y_label'].min()}, {batch['y_label'].max()}]")

print(f"\n2. model results 信息：")
print(f"   batch_indices shape: {results['batch_indices'].shape}")
print(f"   batch_indices unique: {results['batch_indices'].unique()}")
print(f"   outputs 长度: {len(results['outputs'])}")

for i in range(len(results['outputs'])):
    if len(results['outputs'][i]) > 0:
        logits = results['outputs'][i][0]['pred_mask_logits']
        print(f"   outputs[{i}] pred_mask_logits shape: {logits.shape}")
    else:
        print(f"   outputs[{i}] 为空")

print(f"\n3. 关键长度对比：")
print(f"   batch_indices 长度: {len(results['batch_indices'])}")
print(f"   x_label 长度: {len(batch['x_label'])}")
print(f"   两者是否相等: {len(results['batch_indices']) == len(batch['x_label'])}")

print(f"\n4. 每个 batch 的点数分布：")
for i in range(4):
    point_mask = results['batch_indices'] == i
    n_model = point_mask.sum().item()
    
    # x_label 通过 batch_indices 过滤
    x_for_batch = batch['x_label'][point_mask]
    y_for_batch = batch['y_label'][point_mask]
    
    print(f"   batch {i}: model outputs 有 {n_model} 个点")
    
    if len(results['outputs'][i]) > 0:
        logits_shape = results['outputs'][i][0]['pred_mask_logits'].shape
        print(f"            pred_mask_logits shape: {logits_shape}")
        
        # 这是关键检查：logits 的第一维应该等于 x_for_batch 的长度
        if logits_shape[0] != n_model:
            print(f"            ⚠️ 不匹配！logits 有 {logits_shape[0]} 个点，但 point_mask 选出 {n_model} 个点")
        else:
            print(f"            ✅ 匹配正确")
        
    if x_for_batch.numel() > 0:
        print(f"            x_label 范围: [{x_for_batch.min()}, {x_for_batch.max()}]")
        print(f"            y_label 范围: [{y_for_batch.min()}, {y_for_batch.max()}]")
        
        # 检查越界
        H, W = 240, 320
        oob = ((x_for_batch < 0) | (x_for_batch >= W) | (y_for_batch < 0) | (y_for_batch >= H)).sum().item()
        print(f"            越界点数: {oob}/{x_for_batch.numel()} ({oob/x_for_batch.numel()*100:.1f}%)")

print(f"\n5. 测试 Criteria：")
try:
    criteria = Criteria(results, batch, bce_weight=1.0, dice_weight=1.0)
    loss = criteria.loss_pt()
    print(f"   Loss: {loss.item()}")
    if loss.item() == 0:
        print("   ⚠️ Loss 为 0，说明所有 batch 都被跳过了！")
except Exception as e:
    print(f"   ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
