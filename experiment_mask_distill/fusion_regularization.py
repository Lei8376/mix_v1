"""Hybrid fusion semantic regularization losses.

This module keeps semantic constraints separate from Diff2Scene-style mask
distillation so ablations can turn NCE/VICReg on or off independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_cross_modal_nce_loss(
    fused: torch.Tensor,
    mask_tokens: torch.Tensor,
    pixel_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Contrast each fused token against same-region ODISE and LSeg tokens."""
    losses = []
    B = fused.shape[0]

    for b in range(B):
        valid = valid_mask[b].bool()
        if valid.sum() < 2:
            continue

        z = F.normalize(fused[b, valid], dim=-1)
        m = F.normalize(mask_tokens[b, valid].detach(), dim=-1)
        l = F.normalize(pixel_tokens[b, valid].detach(), dim=-1)
        target = torch.arange(z.shape[0], device=z.device)

        loss_zm = F.cross_entropy((z @ m.t()) / tau, target)
        loss_zl = F.cross_entropy((z @ l.t()) / tau, target)
        losses.append(0.5 * loss_zm + 0.5 * loss_zl)

    if not losses:
        return fused.new_tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


def soft_cross_modal_nce_loss(
    fused: torch.Tensor,
    mask_tokens: torch.Tensor,
    pixel_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    tau_student: float = 0.1,
    tau_teacher: float = 0.2,
) -> torch.Tensor:
    """Use teacher token relations as a soft target distribution."""
    losses = []
    B = fused.shape[0]

    for b in range(B):
        valid = valid_mask[b].bool()
        if valid.sum() < 2:
            continue

        z = F.normalize(fused[b, valid], dim=-1)
        m = F.normalize(mask_tokens[b, valid].detach(), dim=-1)
        l = F.normalize(pixel_tokens[b, valid].detach(), dim=-1)

        teacher_sim = 0.5 * (m @ m.t()) + 0.5 * (l @ l.t())
        target_prob = F.softmax(teacher_sim / tau_teacher, dim=-1)

        log_prob_zm = F.log_softmax((z @ m.t()) / tau_student, dim=-1)
        log_prob_zl = F.log_softmax((z @ l.t()) / tau_student, dim=-1)

        loss_zm = F.kl_div(log_prob_zm, target_prob, reduction="batchmean")
        loss_zl = F.kl_div(log_prob_zl, target_prob, reduction="batchmean")
        losses.append(0.5 * loss_zm + 0.5 * loss_zl)

    if not losses:
        return fused.new_tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise AssertionError("off_diagonal expects a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss_single(
    z: torch.Tensor,
    y: torch.Tensor,
    sim_w: float = 25.0,
    var_w: float = 25.0,
    cov_w: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg for one scene's valid mask tokens."""
    z = z.float()
    y = y.detach().float()

    if z.shape[0] < 2:
        return z.new_tensor(0.0, requires_grad=True)

    loss_inv = F.mse_loss(z, y)

    std_z = torch.sqrt(z.var(dim=0) + eps)
    std_y = torch.sqrt(y.var(dim=0) + eps)
    loss_var = torch.mean(F.relu(gamma - std_z))
    loss_var = loss_var + torch.mean(F.relu(gamma - std_y))

    zc = z - z.mean(dim=0)
    yc = y - y.mean(dim=0)
    cov_z = (zc.t() @ zc) / (z.shape[0] - 1)
    cov_y = (yc.t() @ yc) / (y.shape[0] - 1)

    loss_cov = off_diagonal(cov_z).pow(2).sum() / z.shape[1]
    loss_cov = loss_cov + off_diagonal(cov_y).pow(2).sum() / y.shape[1]

    return sim_w * loss_inv + var_w * loss_var + cov_w * loss_cov


def vicreg_loss_batch(
    fused: torch.Tensor,
    base: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply VICReg between pre-refine base and final fused tokens per scene."""
    losses = []
    B = fused.shape[0]

    for b in range(B):
        valid = valid_mask[b].bool()
        if valid.sum() < 4:
            continue

        z = F.normalize(fused[b, valid], dim=-1)
        y = F.normalize(base[b, valid], dim=-1)
        losses.append(vicreg_loss_single(z, y))

    if not losses:
        return fused.new_tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


class HybridFusionRegularizationLoss(nn.Module):
    """NCE + VICReg loss for Method A hybrid token training."""

    def __init__(
        self,
        nce_weight: float = 0.5,
        vicreg_weight: float = 0.03,
        nce_type: str = "hard",
        tau: float = 0.1,
        tau_teacher: float = 0.2,
    ):
        super().__init__()
        self.nce_weight = nce_weight
        self.vicreg_weight = vicreg_weight
        self.nce_type = nce_type
        self.tau = tau
        self.tau_teacher = tau_teacher

    def forward(self, results):
        aux = results["fusion_aux"]
        valid_mask = results["mask_valid_from_masks"]

        fused = aux["fused"]
        mask_tokens = aux["mask_tokens"]
        pixel_tokens = aux["pixel_tokens"]
        base = aux["base"]

        if self.nce_weight > 0:
            if self.nce_type == "soft":
                loss_nce = soft_cross_modal_nce_loss(
                    fused=fused,
                    mask_tokens=mask_tokens,
                    pixel_tokens=pixel_tokens,
                    valid_mask=valid_mask,
                    tau_student=self.tau,
                    tau_teacher=self.tau_teacher,
                )
            else:
                loss_nce = hard_cross_modal_nce_loss(
                    fused=fused,
                    mask_tokens=mask_tokens,
                    pixel_tokens=pixel_tokens,
                    valid_mask=valid_mask,
                    tau=self.tau,
                )
        else:
            loss_nce = fused.new_tensor(0.0)

        if self.vicreg_weight > 0:
            loss_vicreg = vicreg_loss_batch(
                fused=fused,
                base=base,
                valid_mask=valid_mask,
            )
        else:
            loss_vicreg = fused.new_tensor(0.0)

        total = self.nce_weight * loss_nce + self.vicreg_weight * loss_vicreg
        stats = {
            "loss_nce": float(loss_nce.detach().cpu()),
            "loss_vicreg": float(loss_vicreg.detach().cpu()),
            "loss_fusion_reg": float(total.detach().cpu()),
        }
        return total, stats
