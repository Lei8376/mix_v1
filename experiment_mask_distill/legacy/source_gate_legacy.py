from typing import Dict, Optional, Tuple

import torch


def compute_source_gate_loss(
    trainer,
    results: Dict,
    batch: Dict,
    text_feats: Optional[torch.Tensor],
    pixel_text_feats: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    target = str(trainer.config.source_gate_training_target).lower()
    if target == "none":
        return None, trainer._empty_source_gate_logs()
    if target == "text_free_mv_stability":
        return trainer._compute_source_gate_text_free_mv_loss(results, batch)
    if target == "open_reliability":
        return trainer._compute_source_gate_open_reliability_loss(
            results,
            batch,
            text_feats,
            pixel_text_feats,
        )
    if target == "gt_ce_upper_bound":
        if not trainer.config.allow_source_gate_gt_ce_upper_bound:
            raise RuntimeError(
                "gt_ce_upper_bound uses semantic GT and is not open-vocabulary training. "
                "Set allow_source_gate_gt_ce_upper_bound=true only for upper-bound ablation."
            )
        if trainer.is_main and not trainer._warned_source_gate_gt_ce:
            print(
                "WARNING: SourceGate GT CE uses semantic ground-truth labels. "
                "This is a closed-set supervised upper-bound ablation, not open-vocabulary training."
            )
            trainer._warned_source_gate_gt_ce = True
        return trainer._compute_source_gate_gt_ce_upper_bound_loss(
            results,
            batch,
            text_feats,
            pixel_text_feats,
        )
    raise ValueError(
        "source_gate_training_target must be one of "
        "{'none', 'text_free_mv_stability', 'open_reliability', 'gt_ce_upper_bound'}, "
        f"got {trainer.config.source_gate_training_target!r}"
    )


def compute_dual_branch_mask_probe(trainer, results: Dict, batch: Dict) -> Dict[str, float]:
    return trainer._compute_dual_branch_mask_probe_impl(results, batch)


def compute_projected_semantic_consistency_probe(
    trainer,
    results: Dict,
    batch: Dict,
) -> Dict[str, float]:
    return trainer._compute_projected_semantic_consistency_probe_impl(results, batch)
