"""Controlled order-decomposed synthetic multimodal benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


PAIR_KEYS = ("12_to_3", "13_to_2", "23_to_1")


@dataclass(frozen=True)
class SyntheticOrderConfig:
    latent_dim: int = 32
    observation_dim: int = 64
    train_size: int = 20_000
    validation_size: int = 5_000
    test_size: int = 5_000
    noise_std: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    active_pairs: tuple[str, ...] = ()
    active_triple: bool = False


REGIMES = {
    "s0_first_order": SyntheticOrderConfig(alpha=1.0, beta=0.0, gamma=0.0),
    "s1_pair12": SyntheticOrderConfig(alpha=0.5, beta=1.0, gamma=0.0, active_pairs=("12_to_3",)),
    "s2_all_pairs": SyntheticOrderConfig(alpha=0.5, beta=1.0, gamma=0.0, active_pairs=PAIR_KEYS),
    "s3_triple": SyntheticOrderConfig(alpha=0.0, beta=0.0, gamma=1.0, active_triple=True),
    "s4_mixed": SyntheticOrderConfig(alpha=0.5, beta=1.0, gamma=1.0, active_pairs=PAIR_KEYS, active_triple=True),
}


def _orthogonal_projection(observation_dim: int, latent_dim: int, generator: torch.Generator) -> Tensor:
    matrix = torch.randn(observation_dim, latent_dim, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q


def make_world(config: SyntheticOrderConfig, seed: int = 0) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    projections = torch.stack(
        tuple(_orthogonal_projection(config.observation_dim, config.latent_dim, generator) for _ in range(3))
    )
    weights = torch.randn(7, config.latent_dim, generator=generator)
    weights = weights / weights.norm(dim=1, keepdim=True)
    return {"projections": projections, "weights": weights}


def _pair_component(z_left: Tensor, z_right: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    product = z_left * z_right
    return product @ weight, product


def sample_split(
    world: dict[str, Tensor],
    config: SyntheticOrderConfig,
    size: int,
    seed: int,
) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    latent_dim = config.latent_dim
    z = torch.randint(0, 2, (size, 3, latent_dim), generator=generator).float().mul(2).sub(1)
    projections = world["projections"]
    observations = torch.einsum("bmd,mod->bmo", z, projections)
    if config.noise_std:
        observations = observations + config.noise_std * torch.randn(observations.shape, generator=generator)

    weights = world["weights"]
    s1 = z[:, 0] @ weights[0]
    s2 = z[:, 1] @ weights[1]
    s3 = z[:, 2] @ weights[2]
    s12, g12 = _pair_component(z[:, 0], z[:, 1], weights[3])
    s13, g13 = _pair_component(z[:, 0], z[:, 2], weights[4])
    s23, g23 = _pair_component(z[:, 1], z[:, 2], weights[5])
    s123 = (z[:, 0] * z[:, 1] * z[:, 2]) @ weights[6]
    g123 = z[:, 0] * z[:, 1] * z[:, 2]

    score = config.alpha * (s1 + s2 + s3) / 3**0.5
    active_pair_count = len(config.active_pairs)
    if active_pair_count:
        pair_sum = sum({"12_to_3": s12, "13_to_2": s13, "23_to_1": s23}[key] for key in config.active_pairs)
        score = score + config.beta * pair_sum / active_pair_count**0.5
    if config.active_triple:
        score = score + config.gamma * s123
    labels = (score > 0).long()
    return {
        "x1": observations[:, 0].float(),
        "x2": observations[:, 1].float(),
        "x3": observations[:, 2].float(),
        "z1": z[:, 0].float(),
        "z2": z[:, 1].float(),
        "z3": z[:, 2].float(),
        "g12": g12.float(),
        "g13": g13.float(),
        "g23": g23.float(),
        "g123": g123.float(),
        "s1": s1.float(),
        "s2": s2.float(),
        "s3": s3.float(),
        "s12": s12.float(),
        "s13": s13.float(),
        "s23": s23.float(),
        "s123": s123.float(),
        "labels": labels,
        "score": score,
    }


def make_dataset(config: SyntheticOrderConfig, seed: int, world_seed: int = 0) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    world = make_world(config, world_seed)
    return (
        sample_split(world, config, config.train_size, seed * 10 + 1),
        sample_split(world, config, config.validation_size, seed * 10 + 2),
        sample_split(world, config, config.test_size, seed * 10 + 3),
    )
