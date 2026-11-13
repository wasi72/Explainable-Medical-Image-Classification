"""Loss functions tailored for multi-label medical image classification with extreme class imbalance."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy with per-class positive weights to penalize false negatives on rare diseases."""

    def __init__(self, pos_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weights = pos_weights
        if pos_weights is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weights is not None and self.pos_weights.device != logits.device:
            self.pos_weights = self.pos_weights.to(logits.device)
            self.loss_fn.pos_weight = self.pos_weights
        return self.loss_fn(logits, targets)


class MultiLabelFocalLoss(nn.Module):
    """Focal Loss for Multi-Label Classification (Lin et al.).

    Dynamically scales loss down for well-classified examples, focusing learning on hard cases.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for Multi-Label Classification (Ridnik et al., ICCV 2021).

    Applies asymmetric focusing on positive vs negative samples to combat extreme negative dominance.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic CE calculation
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


def get_loss_function(
    loss_type: str = "weighted_bce",
    pos_weights: Optional[torch.Tensor] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> nn.Module:
    """Loss function factory."""
    loss_type = loss_type.lower()
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type in ("weighted_bce", "pos_weighted_bce"):
        return WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
    elif loss_type == "focal":
        return MultiLabelFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == "asymmetric":
        return AsymmetricLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
