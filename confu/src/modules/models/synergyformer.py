"""Minimal order-specific multimodal representation model."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """Encode global ``[B, F]`` or token-level ``[B, T, F]`` inputs."""

    def __init__(self, input_dim: int, dim: int) -> None:
        super().__init__()
        self.output_dim = dim
        self.net = nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        tokens = self.net(x)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
        return tokens, pooled, mask


class LowRankInteraction(nn.Module):
    """Low-rank multiplicative interaction for two or three ``[B, D]`` inputs."""

    def __init__(
        self,
        dim: int,
        rank: int,
        order: int,
        *,
        normalize_factors: bool = False,
        factor_bias: bool = True,
        output_bias: bool = True,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Linear(dim, rank, bias=factor_bias) for _ in range(order))
        self.factor_norms = nn.ModuleList(
            nn.LayerNorm(rank) if normalize_factors else nn.Identity() for _ in range(order)
        )
        self.normalize_factors = normalize_factors
        self.scale = math.sqrt(rank) if normalize_factors else 1.0
        self.output = nn.Linear(rank, dim, bias=output_bias)

    def project_factors(self, *inputs: Tensor) -> list[Tensor]:
        """Return the per-input factors before the elementwise product."""
        factors = [projection(value) for projection, value in zip(self.projections, inputs)]
        if self.normalize_factors:
            factors = [norm(value) for norm, value in zip(self.factor_norms, factors)]
        return factors

    def forward(self, *inputs: Tensor) -> Tensor:
        factors = self.project_factors(*inputs)
        product = torch.stack(factors).prod(0) / self.scale
        return self.output(product)


class GatedComposition(nn.Module):
    """Scale-aware composition of two unimodal terms and one interaction term."""

    def __init__(self, dim: int, initial_gate_logit: float = 0.0) -> None:
        super().__init__()
        self.input_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.output_norm = nn.LayerNorm(dim)
        self.gate_logit = nn.Parameter(torch.tensor(initial_gate_logit))

    @property
    def gate(self) -> Tensor:
        return self.gate_logit.sigmoid()

    def forward(
        self, r1: Tensor, r2: Tensor, r12: Tensor, gate: Tensor | float | None = None
    ) -> Tensor:
        terms = [norm(value) for norm, value in zip(self.input_norms, (r1, r2, r12))]
        interaction_gate = self.gate if gate is None else torch.as_tensor(gate, device=r1.device, dtype=r1.dtype)
        return self.output_norm(terms[0] + terms[1] + interaction_gate * terms[2])

    def from_base(self, base: Tensor, interaction: Tensor, gate: Tensor | float | None = None) -> Tensor:
        interaction_gate = self.gate if gate is None else torch.as_tensor(gate, device=base.device, dtype=base.dtype)
        return self.output_norm(base + interaction_gate * self.input_norms[2](interaction))


class SynergyFormer(nn.Module):
    """Decompose three modalities into order-1, order-2, and order-3 terms.

    Inputs may be global tensors ``[B, F_i]`` or token tensors ``[B, T_i, F_i]``.
    Missing modalities are passed as ``None`` and invalidate dependent interactions.
    """

    def __init__(
        self,
        input_dims: Sequence[int] | None = None,
        *,
        encoders: Sequence[nn.Module] | None = None,
        dim: int = 32,
        rank: int = 16,
        aggregator: str = "transformer",
        num_layers: int = 1,
        num_heads: int = 4,
        include_third_order: bool = True,
    ) -> None:
        super().__init__()
        if (input_dims is None) == (encoders is None):
            raise ValueError("provide exactly one of input_dims or encoders")
        if encoders is None:
            encoders = [MLPEncoder(input_dim, dim) for input_dim in input_dims or ()]
        if len(encoders) != 3:
            raise ValueError("SynergyFormer requires exactly three encoders")
        if aggregator not in {"sum", "transformer"}:
            raise ValueError("aggregator must be 'sum' or 'transformer'")

        self.dim = dim
        self.aggregator = aggregator
        self.include_third_order = include_third_order
        self.encoders = nn.ModuleList(encoders)
        self.projections = nn.ModuleList(
            nn.Identity() if encoder.output_dim == dim else nn.Linear(encoder.output_dim, dim)
            for encoder in self.encoders
        )
        self.pair_interactions = nn.ModuleList(LowRankInteraction(dim, rank, 2) for _ in range(3))
        self.third_interaction = LowRankInteraction(dim, rank, 3)
        self.order_embeddings = nn.Parameter(torch.zeros(3, dim))
        self.identity_embeddings = nn.Parameter(torch.zeros(7, dim))

        if aggregator == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.structured_aggregator = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.global_projection = nn.Linear(dim, dim)

    def _encode(self, index: int, value: object, mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        output = self.encoders[index](value, mask) if isinstance(self.encoders[index], MLPEncoder) else self.encoders[index](value)
        if len(output) == 3:
            tokens, pooled, output_mask = output
        else:
            pooled, tokens = output
            tokens = pooled.unsqueeze(1) if tokens is None else tokens
            output_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=pooled.device)
        projection = self.projections[index]
        return projection(tokens), projection(pooled), output_mask

    def forward(
        self,
        modality1: object,
        modality2: object | None = None,
        modality3: object | None = None,
        masks: Sequence[Tensor | None] | None = None,
    ) -> dict[str, object]:
        modalities = (modality1, modality2, modality3)
        masks = masks or (None, None, None)
        encoded: list[tuple[Tensor, Tensor, Tensor] | None] = [
            None if value is None else self._encode(i, value, masks[i])
            for i, value in enumerate(modalities)
        ]
        present = next(item for item in encoded if item is not None)
        batch_size, device, dtype = present[1].shape[0], present[1].device, present[1].dtype
        zero = torch.zeros(batch_size, self.dim, device=device, dtype=dtype)

        pooled = [zero if item is None else item[1] for item in encoded]
        availability = torch.stack([
            torch.zeros(batch_size, dtype=torch.bool, device=device)
            if item is None else item[2].any(1)
            for item in encoded
        ], dim=1)
        pooled = [value * availability[:, i : i + 1] for i, value in enumerate(pooled)]

        pair_indices = ((0, 1), (0, 2), (1, 2))
        pair_values: list[Tensor] = []
        pair_available: list[Tensor] = []
        for interaction, (left, right) in zip(self.pair_interactions, pair_indices):
            valid = availability[:, left] & availability[:, right]
            pair_values.append(interaction(pooled[left], pooled[right]) * valid.unsqueeze(1))
            pair_available.append(valid)

        third_available = availability.all(1) & self.include_third_order
        third = self.third_interaction(*pooled) * third_available.unsqueeze(1)
        representations = pooled + pair_values + [third]
        structured_available = torch.stack(
            [*availability.unbind(1), *pair_available, third_available], dim=1
        )
        order_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2], device=device)
        structured = torch.stack(representations, dim=1)
        structured = structured + self.order_embeddings[order_ids] + self.identity_embeddings

        if self.aggregator == "sum":
            weights = structured_available.unsqueeze(-1).to(dtype)
            global_representation = (structured * weights).sum(1) / weights.sum(1).clamp_min(1)
        else:
            cls = self.cls_token.expand(batch_size, -1, -1)
            transformer_input = torch.cat([cls, structured], dim=1)
            padding_mask = torch.cat(
                [torch.zeros(batch_size, 1, dtype=torch.bool, device=device), ~structured_available], dim=1
            )
            global_representation = self.structured_aggregator(
                transformer_input, src_key_padding_mask=padding_mask
            )[:, 0]

        names = ("r1", "r2", "r3", "r12", "r13", "r23", "r123")
        result: dict[str, object] = dict(zip(names, representations))
        result.update(
            z_global=self.global_projection(global_representation),
            availability=availability,
            structured_availability=structured_available,
            tokens={f"h{i + 1}": None if item is None else item[0] for i, item in enumerate(encoded)},
        )
        return result


def symmetric_info_nce(left: Tensor, right: Tensor, temperature: float = 0.07) -> Tensor:
    left, right = F.normalize(left, dim=-1), F.normalize(right, dim=-1)
    logits = left @ right.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def cross_covariance_loss(left: Tensor, right: Tensor) -> Tensor:
    if left.shape[0] < 2:
        return left.sum() * 0
    left, right = left - left.mean(0), right - right.mean(0)
    covariance = left.T @ right / (left.shape[0] - 1)
    return covariance.square().sum() / left.shape[1]


def variance_loss(representation: Tensor, gamma: float = 1.0) -> Tensor:
    """VICReg variance floor; collapsed dimensions receive a positive gradient."""
    std = torch.sqrt(representation.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(gamma - std).square().mean()


def pair_cross_covariance_loss(interaction: Tensor, left: Tensor, right: Tensor) -> Tensor:
    return cross_covariance_loss(interaction, left) + cross_covariance_loss(interaction, right)


def covariance_loss(representation: Tensor) -> Tensor:
    """VICReg off-diagonal covariance penalty for representation rank."""
    if representation.shape[0] < 2:
        return representation.sum() * 0
    centered = representation - representation.mean(0)
    covariance = centered.T @ centered / (representation.shape[0] - 1)
    squared = covariance.square()
    return (squared.sum() - squared.diagonal().sum()) / representation.shape[1]


def conditional_utility_loss(full_losses: Tensor, base_losses: Tensor, margin: float = 0.0) -> Tensor:
    """Require each full prediction to outperform its detached lower-order baseline."""
    return F.relu(full_losses - base_losses.detach() + margin).mean()


def dependence_ranking_loss(
    positive_logits: Tensor,
    corrupted_logits: Sequence[Tensor],
    labels: Tensor,
    margin: float = 0.1,
) -> Tensor:
    """Rank the true paired interaction above each single-modality corruption."""
    positive = positive_logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
    losses = []
    for logits in corrupted_logits:
        negative = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
        losses.append(F.relu(margin - positive + negative))
    return torch.stack(losses).sum(0).mean()


def redundancy_loss(outputs: dict[str, object]) -> Tensor:
    tensors = {key: value for key, value in outputs.items() if isinstance(value, Tensor)}
    pairs = (("r12", "r1"), ("r12", "r2"), ("r13", "r1"), ("r13", "r3"),
             ("r23", "r2"), ("r23", "r3"), ("r123", "r12"),
             ("r123", "r13"), ("r123", "r23"))
    return torch.stack([cross_covariance_loss(tensors[a], tensors[b]) for a, b in pairs]).mean()
