"""Representation-level complementarity mechanisms for ConFu++.

Implements the S1/S2 mechanisms from
``AGENT.md — ConFu++ Representation-Level Complementarity.md``:

S1  Adversarial de-shortcutting: an interaction representation must be
    unpredictable from each SINGLE modality (nonlinear adversaries),
    while remaining a deterministic function of the modality pair.
S2  Factor health: per-factor variance floors prevent the
    constant-factor collapse escape route of the low-rank product.

Training uses alternating minimax updates instead of gradient reversal
so the adversary fit quality is a directly loggable "shortcut meter".
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AdversarialPredictor(nn.Module):
    """MLP adversary predicting a standardized interaction from one modality.

    Capacity intentionally matches the evaluation predictability probe
    (one hidden layer of width ``hidden_dim``, GELU) so that the
    training-time mechanism and the offline measurement see the same
    function class.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int | None = None) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class EMAStandardizer(nn.Module):
    """Running mean/variance of a representation (stop-grad statistics).

    Buffers are updated under ``torch.no_grad``; ``standardize`` keeps
    gradients flowing through its input while treating the statistics
    as constants.
    """

    def __init__(self, dim: int, momentum: float = 0.99) -> None:
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, value: Tensor) -> None:
        batch_mean = value.mean(0)
        batch_var = value.var(dim=0, unbiased=False)
        if self.updates.item() == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var.clamp_min(1e-4))
        else:
            self.mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
            self.var.mul_(self.momentum).add_(
                batch_var.clamp_min(1e-4), alpha=1 - self.momentum
            )
        self.updates.add_(1)

    def standardize(self, value: Tensor) -> Tensor:
        """Standardize ``value`` with frozen stats; gradient flows through ``value``."""
        std = (self.var + 1e-4).sqrt()
        return (value - self.mean) / std


def adversary_loss(
    predictor_left: AdversarialPredictor,
    predictor_right: AdversarialPredictor,
    r_left: Tensor,
    r_right: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Adversary step loss: predict a standardized interaction from each modality.

    All inputs are treated as constants (callers must pass detached
    tensors). Returns the total loss and the per-modality MSE values.
    """
    prediction_left = predictor_left(r_left)
    prediction_right = predictor_right(r_right)
    mse_left = F.mse_loss(prediction_left, target)
    mse_right = F.mse_loss(prediction_right, target)
    return mse_left + mse_right, mse_left.detach(), mse_right.detach()


def shortcut_hinge_loss(
    prediction_left: Tensor,
    prediction_right: Tensor,
    target: Tensor,
    tau: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generator-side hinged shortcut penalty.

    ``prediction_left``/``prediction_right`` must be constants (computed
    under ``torch.no_grad``); ``target`` carries the gradient back into
    the interaction representation. Because the target is standardized,
    per-element MSE equals ``1 - R^2`` of the adversary; the hinge taxes
    single-modality R^2 above ``1 - tau`` without demanding
    unpredictability beyond ``tau`` (collapse guard).
    """
    mse_left = F.mse_loss(prediction_left, target)
    mse_right = F.mse_loss(prediction_right, target)
    penalty = F.relu(tau - mse_left) + F.relu(tau - mse_right)
    return penalty, mse_left.detach(), mse_right.detach()


def factor_variance_loss(factors: list[Tensor], gamma: float = 1.0) -> Tensor:
    """Per-factor VICReg variance floor (S2).

    Prevents the constant-factor escape route of the low-rank product:
    if one factor collapses to a constant, the product degenerates to
    ``u_i * c`` and the interaction becomes a unimodal shortcut.
    """
    losses = []
    for factor in factors:
        std = torch.sqrt(factor.var(dim=0, unbiased=False) + 1e-4)
        losses.append(F.relu(gamma - std).square().mean())
    return torch.stack(losses).mean()


def interaction_predictability_r2(model: nn.Module, dataloader: object, device: torch.device,
                                  extract: object, max_samples: int = 4096) -> dict[str, float]:
    """Quick online meter: R^2 of predicting r12 from r1 alone / r2 alone.

    ``extract(model, images, audios)`` must return ``(r1, r2, r12)``.
    Uses closed-form ridge regression on up to ``max_samples`` samples.
    """
    model.eval()
    lefts, rights, targets = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            images, audios = batch[0].to(device), batch[1].to(device)
            r1, r2, r12 = extract(model, images, audios)
            lefts.append(r1.cpu())
            rights.append(r2.cpu())
            targets.append(r12.cpu())
            if sum(t.shape[0] for t in targets) >= max_samples:
                break
    left = torch.cat(lefts)[:max_samples]
    right = torch.cat(rights)[:max_samples]
    target = torch.cat(targets)[:max_samples]

    def ridge_r2(source: Tensor) -> float:
        ones = torch.ones(source.shape[0], 1)
        design = torch.cat([source, ones], dim=1)
        solution = torch.linalg.lstsq(design, target).solution
        predicted = design @ solution
        residual = (target - predicted).var(dim=0, unbiased=False)
        total = target.var(dim=0, unbiased=False).clamp_min(1e-12)
        return (1 - residual / total).mean().item()

    return {"r2_from_r1": ridge_r2(left), "r2_from_r2": ridge_r2(right)}
