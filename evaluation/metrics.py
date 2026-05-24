"""Evaluation metrics for pitch outcome and physics predictions."""

import numpy as np
import torch
import torch.nn.functional as F

from training.loss import mog_nll as _mog_nll_tensor
from models.full_model import N_MOG_COMPONENTS


def top_k_precision(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 4,
    ignore_index: int = -1,
) -> float:
    valid  = labels != ignore_index
    if not valid.any():
        return 0.0
    logits, labels = logits[valid], labels[valid]
    topk    = logits.topk(k, dim=-1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def log_loss_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
) -> float:
    valid = labels != ignore_index
    if not valid.any():
        return float('nan')
    return F.cross_entropy(logits[valid], labels[valid]).item()


def brier_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    ignore_index: int = -1,
) -> float:
    valid = labels != ignore_index
    if not valid.any():
        return float('nan')
    probs   = torch.softmax(logits[valid], dim=-1).cpu().numpy()
    lbl_np  = labels[valid].cpu().numpy()
    one_hot = np.eye(n_classes)[lbl_np]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
    ignore_index: int = -1,
) -> float:
    valid = labels != ignore_index
    if not valid.any():
        return float('nan')
    probs               = torch.softmax(logits[valid], dim=-1)
    confidences, preds  = probs.max(dim=-1)
    correct             = (preds == labels[valid]).float()
    n_valid             = valid.sum().item()

    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        in_bin = (confidences > lo) & (confidences <= hi)
        n      = in_bin.float().sum().item()
        if n > 0:
            ece += (n / n_valid) * abs(
                confidences[in_bin].mean().item() - correct[in_bin].mean().item()
            )
    return ece


def mog_nll(params: torch.Tensor, targets: torch.Tensor) -> float:
    return _mog_nll_tensor(params, targets).item()


def per_class_top4(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    k: int = 4,
    ignore_index: int = -1,
) -> list:
    """Top-k recall per class: for each class c, fraction of true-c samples
    where c appears in the top-k predicted classes. Returns list of length n_classes
    (None if class has no samples)."""
    valid = labels != ignore_index
    logits, labels = logits[valid], labels[valid]
    topk = logits.topk(k, dim=-1).indices  # (N, k)
    results = []
    for c in range(n_classes):
        mask = labels == c
        if not mask.any():
            results.append(None)
        else:
            correct = (topk[mask] == c).any(dim=1).float().mean().item()
            results.append(correct)
    return results


def mog_mean(params: torch.Tensor) -> torch.Tensor:
    K       = N_MOG_COMPONENTS
    weights = torch.softmax(params[:, :K], dim=-1)
    mu      = params[:, K:2 * K]
    return (weights * mu).sum(dim=-1)


def physics_mae(
    params:  torch.Tensor,
    targets: torch.Tensor,
    scale:   float = 1.0,
) -> float:
    """MAE between predicted MoG mean and target.

    scale: outcome_scaler.scale_[i] converts from standardized to original units.
    """
    pred = mog_mean(params) * scale
    tgt  = targets * scale
    return (pred - tgt).abs().mean().item()
