"""Nested Product Joint plus a small residual joint class for v5.6.1."""

from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, left_dim: int, right_dim: int, target_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(left_dim + right_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((left, right), dim=1))


class NestedJointPredictor(nn.Module):
    """Frozen Product Joint plus a zero-initialized concatenation residual."""

    def __init__(self, product: nn.Module, left_dim: int, right_dim: int, target_dim: int, hidden_dim: int):
        super().__init__()
        self.product = product
        for parameter in self.product.parameters():
            parameter.requires_grad_(False)
        self.product.eval()
        self.residual = ResidualMLP(left_dim, right_dim, target_dim, hidden_dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            product = self.product(left, right)
        return product + self.residual(left, right)


def residual_parameter_count(left_dim: int, right_dim: int, target_dim: int, hidden_dim: int) -> int:
    return (left_dim + right_dim + 1) * hidden_dim + (hidden_dim + 1) * target_dim


def choose_residual_width(left_dim: int, right_dim: int, target_dim: int, product_parameters: int, budget_fraction: float = 0.25) -> dict:
    budget = int(product_parameters * budget_fraction)
    valid = [
        (hidden_dim, residual_parameter_count(left_dim, right_dim, target_dim, hidden_dim))
        for hidden_dim in range(1, 1025)
        if residual_parameter_count(left_dim, right_dim, target_dim, hidden_dim) <= budget
    ]
    if not valid:
        raise RuntimeError("Residual budget cannot fit one hidden unit")
    hidden_dim, parameters = valid[-1]
    return {
        "hidden_dim": hidden_dim,
        "residual_parameters": parameters,
        "product_parameters": product_parameters,
        "budget_fraction": budget_fraction,
        "budget_parameters": budget,
        "residual_ratio": parameters / product_parameters,
    }
