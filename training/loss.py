"""Multi-task loss: classification + physics NLL + calibration regularization.

Total loss = w_cls * (outcome_focal + location_focal)
           + w_phy * (EV_NLL + LA_NLL)   [contact pitches only]
           + w_cal * ECE                  [outcome head only]

Classification uses focal loss: it down-weights easy, high-confidence pitches
(ball/strike/field-out) so the gradient is not dominated by frequent classes,
which lifts recall on rare contact classes (double, triple, HR). The class
weights act as the focal alpha term; gamma=0 recovers plain weighted CE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.full_model import N_MOG_COMPONENTS


def focal_loss(
    logits:       torch.Tensor,
    targets:      torch.Tensor,
    gamma:        float = 2.0,
    weight:       torch.Tensor = None,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Multi-class focal loss with optional per-class alpha weights.

    FL = alpha_t * (1 - p_t)^gamma * (-log p_t), averaged over valid samples.
    With weights, normalizes by the sum of alpha (matching weighted-CE scale).
    gamma=0 reduces to (weighted) cross-entropy.
    """
    valid = targets != ignore_index
    if not valid.any():
        return logits.new_zeros(())

    logits  = logits[valid]
    targets = targets[valid]

    log_p  = F.log_softmax(logits, dim=-1)
    log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt     = log_pt.exp()
    loss   = (1.0 - pt).pow(gamma) * (-log_pt)

    if weight is not None:
        alpha = weight[targets]
        return (alpha * loss).sum() / alpha.sum().clamp_min(1e-8)
    return loss.mean()


def mog_nll(params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of a K-component mixture of Gaussians.

    params:  (N, 3*K)  layout: [logit_w × K | mu × K | log_sigma × K]
    targets: (N,)
    """
    K       = N_MOG_COMPONENTS
    log_w   = F.log_softmax(params[:, :K], dim=-1)
    mu      = params[:, K:2 * K]
    log_sig = params[:, 2 * K:].clamp(min=-3, max=3)
    sig     = log_sig.exp()

    t        = targets.unsqueeze(1)
    log_prob = -0.5 * ((t - mu) / sig) ** 2 - log_sig - 0.9189
    return -torch.logsumexp(log_w + log_prob, dim=1).mean()


def soft_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    """Vectorized soft ECE calibration regularization."""
    probs               = torch.softmax(logits, dim=-1)
    confidences, preds  = probs.max(dim=-1)
    correct             = (preds == labels).float()

    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    bin_ids    = torch.bucketize(confidences, boundaries[1:-1])  # (N,) in [0, n_bins-1]

    n_samples   = len(labels)
    bin_counts  = torch.zeros(n_bins, device=logits.device)
    bin_acc_sum = torch.zeros(n_bins, device=logits.device)
    bin_conf_sum = torch.zeros(n_bins, device=logits.device)

    bin_counts.scatter_add_(0, bin_ids, torch.ones_like(confidences))
    bin_acc_sum.scatter_add_(0, bin_ids, correct)
    bin_conf_sum.scatter_add_(0, bin_ids, confidences)

    nonempty = bin_counts > 0
    return (
        (bin_counts[nonempty] / n_samples) *
        (bin_conf_sum[nonempty] / bin_counts[nonempty] -
         bin_acc_sum[nonempty]  / bin_counts[nonempty]).abs()
    ).sum()


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        w_classification: float = 0.7,
        w_physics:        float = 0.2,
        w_calibration:    float = 0.1,
        gamma:            float = 2.0,
        outcome_weights:  torch.Tensor = None,
        location_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.w_cls = w_classification
        self.w_phy = w_physics
        self.w_cal = w_calibration
        self.gamma = gamma
        self.register_buffer('outcome_weights',  outcome_weights)
        self.register_buffer('location_weights', location_weights)

    def forward(self, preds: dict, batch: dict) -> dict:
        cls_loss = (
            focal_loss(preds['pitch_outcome_logits'], batch['pitch_outcome_label'],
                       gamma=self.gamma, weight=self.outcome_weights,  ignore_index=-1) +
            focal_loss(preds['hit_location_logits'],  batch['hit_location_label'],
                       gamma=self.gamma, weight=self.location_weights, ignore_index=-1)
        )

        contact  = batch['is_contact']
        phy_loss = cls_loss.new_zeros(())
        if contact.any():
            phy_loss = (
                mog_nll(preds['ev_params'][contact], batch['ev_target'][contact]) +
                mog_nll(preds['la_params'][contact], batch['la_target'][contact])
            )

        valid_mask = batch['pitch_outcome_label'] >= 0
        cal_loss   = cls_loss.new_zeros(())
        if valid_mask.any():
            cal_loss = soft_ece(
                preds['pitch_outcome_logits'][valid_mask],
                batch['pitch_outcome_label'][valid_mask],
            )

        total = self.w_cls * cls_loss + self.w_phy * phy_loss + self.w_cal * cal_loss
        return {
            'loss':     total,
            'cls_loss': cls_loss.detach(),
            'phy_loss': phy_loss.detach(),
            'cal_loss': cal_loss.detach(),
        }
