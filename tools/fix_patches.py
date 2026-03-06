"""
修复补丁：关键训练问题

根据检查结果，需要修复以下问题：
1. ✅ pred_logits 正确使用（已正确）
2. ⚠️ point→voxel 映射需要添加断言
3. ⚠️ best model 保存策略（按 loss 而非 mIoU）
4. ⚠️ fallback 逻辑需要在训练时禁止
"""

# ===================================================================
# 修复 1: 在 model/open_vocab_fusion_v2.py 添加 point→voxel 匹配率检查
# ===================================================================

PATCH_1_MODEL = """
# 在 model/open_vocab_fusion_v2.py 第 257 行之后添加：

        # 🔥 关键检查：验证 point→voxel 映射是否 100% 匹配
        matched_ratio = matched.float().mean().item()
        if matched_ratio < 0.999:
            print(f'[FATAL] point→voxel matched_ratio={matched_ratio:.6f}')
            print(f'  input_coords shape: {input_coords.shape}')
            print(f'  voxel_coords shape: {voxel_coords.shape}')
            print(f'  input_coords range: batch={input_coords[:, 0].unique()}, '
                  f'x=[{input_coords[:, 1].min()},{input_coords[:, 1].max()}], '
                  f'y=[{input_coords[:, 2].min()},{input_coords[:, 2].max()}], '
                  f'z=[{input_coords[:, 3].min()},{input_coords[:, 3].max()}]')
            print(f'  voxel_coords range: batch={voxel_coords[:, 0].unique()}, '
                  f'x=[{voxel_coords[:, 1].min()},{voxel_coords[:, 1].max()}], '
                  f'y=[{voxel_coords[:, 2].min()},{voxel_coords[:, 2].max()}], '
                  f'z=[{voxel_coords[:, 3].min()},{voxel_coords[:, 3].max()}]')
            num_unmatched = (~matched).sum().item()
            print(f'  未匹配点数: {num_unmatched}/{input_coords.shape[0]} ({100*(1-matched_ratio):.3f}%)')
            raise RuntimeError('point→voxel mapping mismatch: 未匹配的点会被错误映射到 voxel[0]')
"""

# ===================================================================
# 修复 2: 修改 trainer/open_vocab_trainer_v2.py 的 best model 保存策略
# ===================================================================

PATCH_2_TRAINER_OLD = """
            # Save best model (only on main process)
            # Monitor validation loss when available, otherwise use training loss
            monitored_loss = (
                val_metrics["loss"]
                if val_metrics is not None and isinstance(val_metrics, dict) and "loss" in val_metrics
                else train_loss
            )
            is_best = (
                monitored_loss > 0
                and monitored_loss < self.best_loss - self.config.early_stopping_min_delta
            )
            if is_best:
                self.best_loss = monitored_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
"""

PATCH_2_TRAINER_NEW = """
            # Save best model (only on main process)
            # 🔥 关键修复：按 mIoU 而非 loss 保存 best model
            # Monitor mIoU when available (validation), otherwise fall back to loss
            if val_metrics is not None and isinstance(val_metrics, dict):
                # 优先监控 mIoU
                monitored_metric = val_metrics.get('miou', val_metrics.get('iou', 0))
                is_best = monitored_metric > self.best_iou + self.config.early_stopping_min_delta
                
                if is_best:
                    self.best_iou = monitored_metric
                    self.epochs_without_improvement = 0
                    if self.is_main_process:
                        print(f"  🎯 New best mIoU: {monitored_metric:.4f} (prev: {self.best_iou:.4f})")
                else:
                    self.epochs_without_improvement += 1
                
                # 同时更新 best_loss（用于日志）
                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
            else:
                # 训练初期没有 val_metrics 时，用 train_loss 作为备用
                monitored_metric = train_loss
                is_best = (
                    train_loss > 0
                    and train_loss < self.best_loss - self.config.early_stopping_min_delta
                )
                if is_best:
                    self.best_loss = train_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
"""

# ===================================================================
# 修复 3: 在 dataset/open_vocab_dataset_v2.py 禁止训练时 fallback
# ===================================================================

PATCH_3_DATASET = """
# 在 dataset/open_vocab_dataset_v2.py 的 __getitem__ 方法中
# 找到调用 _load_3d_with_precomputed_projection 的地方，添加：

        # 尝试加载预计算投影
        out_3d = _load_3d_with_precomputed_projection(
            self.data_root, self.split, scene_name,
            self.projection_dir, frame_stem,
            pth_cache=self.pth_cache,
            voxel_size=self.voxel_size,
        )
        
        # 🔥 关键修复：训练时禁止 fallback 到运行时投影
        if out_3d is None:
            if self.split == 'train':
                raise RuntimeError(
                    f"Missing precomputed projection for training sample: "
                    f"{scene_name}/{frame_stem}. "
                    f"Training requires all samples to use precomputed projections "
                    f"to ensure coordinate consistency."
                )
            # val/test 可以 fallback 到运行时投影
            print(f"Warning: Fallback to runtime projection for {scene_name}/{frame_stem}")
            # ... 继续原有的 fallback 逻辑
"""

# ===================================================================
# 可选：添加阈值扫描来找最佳阈值
# ===================================================================

PATCH_4_THRESHOLD_SWEEP = """
# 在 trainer/open_vocab_trainer_v2.py 的 _validate 方法最后，可以添加阈值扫描：

    @torch.no_grad()
    def _validate_with_threshold_sweep(self, epoch: int) -> Dict[str, float]:
        \"\"\"在验证时扫描多个阈值找最佳值\"\"\"
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss = AverageMeter()
        
        # 不同阈值的 metrics
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        metrics_per_threshold = {t: MetricsTracker() for t in thresholds}

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)
            batch["sinput"] = self._build_sparse_tensor(batch)

            with autocast(enabled=self.config.use_amp):
                results = self.model(batch)
                criteria = Criteria(
                    results, batch,
                    bce_weight=self.config.bce_weight,
                    dice_weight=self.config.dice_weight,
                    min_points_per_mask=self.config.min_points_per_mask,
                    use_pos_weight=True,
                    use_per_mask_dice=True,
                )
                loss = criteria.loss_pt()

            val_loss.update(loss.item())

            # 对每个阈值计算 metrics（复用现有代码，但用不同阈值）
            for b in range(len(results["outputs"])):
                if len(results["outputs"][b]) == 0:
                    continue
                # ... [与现有 _validate 相同的 GT 提取逻辑] ...
                pred_probs = torch.sigmoid(pred_valid).float()
                
                # 对每个阈值更新 metrics
                for threshold in thresholds:
                    metrics_per_threshold[threshold].update(pred_probs, gt_3d, threshold=threshold)

        # 输出不同阈值的结果
        if self.is_main_process:
            print(f"\\n  Threshold sweep:")
            for t in thresholds:
                m = metrics_per_threshold[t].compute()
                print(f"    t={t:.1f}: mIoU={m['miou']:.4f}, IoU={m['iou']:.4f}")
        
        # 返回默认阈值 0.5 的结果（保持向后兼容）
        val_metrics = metrics_per_threshold[0.5].compute()
        val_metrics["loss"] = val_loss.avg
        return val_metrics
"""

def print_patches():
    print("="*70)
    print("🔧 修复补丁汇总")
    print("="*70)
    
    print("\n【修复 1】添加 point→voxel 匹配率检查")
    print("-"*70)
    print("文件: model/open_vocab_fusion_v2.py")
    print("位置: 第 257 行之后")
    print(PATCH_1_MODEL)
    
    print("\n【修复 2】修改 best model 保存策略（按 mIoU）")
    print("-"*70)
    print("文件: trainer/open_vocab_trainer_v2.py")
    print("位置: 第 691-706 行")
    print("替换以下代码：")
    print(PATCH_2_TRAINER_OLD)
    print("\n为：")
    print(PATCH_2_TRAINER_NEW)
    
    print("\n【修复 3】训练时禁止 fallback 投影")
    print("-"*70)
    print("文件: dataset/open_vocab_dataset_v2.py")
    print("位置: __getitem__ 方法中加载 3D 数据的部分")
    print(PATCH_3_DATASET)
    
    print("\n【可选修复 4】添加阈值扫描")
    print("-"*70)
    print("文件: trainer/open_vocab_trainer_v2.py")
    print("说明: 可以新增一个方法或修改 _validate")
    print(PATCH_4_THRESHOLD_SWEEP)
    
    print("\n" + "="*70)
    print("💡 应用建议")
    print("="*70)
    print("""
1. 优先级 1（必须修）：修复 2 - best model 保存策略
   - 这是导致你可能错过最佳 checkpoint 的主因
   
2. 优先级 2（强烈推荐）：修复 1 - point→voxel 匹配率检查
   - 添加断言，一旦出问题立即报错，避免 silent failure
   
3. 优先级 3（推荐）：修复 3 - 禁止训练时 fallback
   - 确保训练样本的坐标体系一致性
   
4. 优先级 4（可选）：修复 4 - 阈值扫描
   - 帮助找到最佳评估阈值，可能让 mIoU 提升 5-10%

修复后建议：
- 从 best_model.pth 重新评估，看 mIoU 是否比你之前记录的更高
- 如果是，说明之前确实保存错了 checkpoint
- 重新训练或继续训练，使用新的保存策略
""")

if __name__ == '__main__':
    print_patches()
