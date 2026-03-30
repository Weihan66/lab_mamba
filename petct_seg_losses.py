from __future__ import annotations

import torch
import torch.nn as nn

from petct_seg_metrics import _flatten_binary_scores, binary_dice_score


def _prepare_probs_and_targets(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    probs, targets = _flatten_binary_scores(probs, targets.float())
    return probs, targets


def _tversky_scores(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float,
) -> torch.Tensor:
    true_positive = (probs * targets).sum(dim=1)
    false_positive = (probs * (1.0 - targets)).sum(dim=1)
    false_negative = ((1.0 - probs) * targets).sum(dim=1)
    return (true_positive + eps) / (true_positive + alpha * false_positive + beta * false_negative + eps)


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, targets = _prepare_probs_and_targets(logits, targets)

        intersection = (probs * targets).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, bce_pos_weight: float = 1.0) -> None:
        super().__init__()
        if bce_pos_weight <= 0.0:
            raise ValueError(f"Expected bce_pos_weight > 0, got {bce_pos_weight}.")
        self.dice = DiceLoss()
        self.register_buffer("bce_pos_weight", torch.tensor(float(bce_pos_weight), dtype=torch.float32))
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.bce_pos_weight)
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * bce


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> None:
        super().__init__()
        if alpha < 0.0 or beta < 0.0 or alpha + beta <= 0.0:
            raise ValueError(f"Expected alpha >= 0, beta >= 0 and alpha + beta > 0, got alpha={alpha}, beta={beta}.")
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, targets = _prepare_probs_and_targets(logits, targets)
        scores = _tversky_scores(probs, targets, alpha=self.alpha, beta=self.beta, eps=self.eps)
        return 1.0 - scores.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if gamma <= 0.0:
            raise ValueError(f"Expected gamma > 0, got {gamma}.")
        if alpha < 0.0 or beta < 0.0 or alpha + beta <= 0.0:
            raise ValueError(f"Expected alpha >= 0, beta >= 0 and alpha + beta > 0, got alpha={alpha}, beta={beta}.")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, targets = _prepare_probs_and_targets(logits, targets)
        scores = _tversky_scores(probs, targets, alpha=self.alpha, beta=self.beta, eps=self.eps)
        return torch.pow(1.0 - scores, self.gamma).mean()


def build_loss(
    name: str,
    *,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    bce_pos_weight: float = 1.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_tversky_gamma: float = 0.75,
) -> nn.Module:
    normalized_name = name.strip().lower()
    if normalized_name == "dice_bce":
        return DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight, bce_pos_weight=bce_pos_weight)
    if normalized_name == "tversky":
        return TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
    if normalized_name == "focal_tversky":
        return FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=focal_tversky_gamma)
    raise ValueError(f"Unsupported loss: {name}")
