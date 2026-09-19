"""Identifiable Predictive-Interaction Benchmark (IPIB), 12 -> 3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class IPIBConfig:
    latent_dim: int = 32
    source_dim: int = 64
    target_dim: int = 64
    train_size: int = 20_000
    validation_size: int = 5_000
    test_size: int = 5_000
    beta: float = 0.5
    private_strength: float = 1.0
    noise_std: float = 0.0


REGIMES = {
    "i0_no_joint": IPIBConfig(beta=0.0),
    "i1_weak": IPIBConfig(beta=0.25),
    "i2_medium": IPIBConfig(beta=0.5),
    "i3_strong": IPIBConfig(beta=1.0),
    "i4_private_noise": IPIBConfig(beta=1.0, private_strength=2.0),
}


def _orthogonal(rows: int, columns: int, generator: torch.Generator) -> Tensor:
    return torch.linalg.qr(torch.randn(rows, columns, generator=generator), mode="reduced").Q


def _unit_variance_matrix(matrix: Tensor) -> Tensor:
    return matrix / (matrix.square().sum() / matrix.shape[0]).sqrt()


def make_world(config: IPIBConfig, seed: int = 0) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    p1 = _orthogonal(config.source_dim, config.latent_dim, generator)
    p2 = _orthogonal(config.source_dim, config.latent_dim, generator)
    a31 = torch.randn(config.target_dim, config.latent_dim, generator=generator)
    a32 = torch.randn(config.target_dim, config.latent_dim, generator=generator)
    additive_scale = (a31.square().sum() + a32.square().sum()).div(config.target_dim).sqrt()
    a31, a32 = a31 / additive_scale, a32 / additive_scale
    b3 = _unit_variance_matrix(torch.randn(config.target_dim, config.latent_dim, generator=generator))
    # Same target subspace for predictable joint and unpredictable private content.
    # Otherwise a linear recovery probe can separate the two by projection geometry.
    p3 = b3
    label_weight = torch.randn(config.latent_dim, generator=generator)
    label_weight = label_weight / label_weight.norm()
    return {"p1": p1, "p2": p2, "p3": p3, "a31": a31, "a32": a32, "b3": b3, "label_weight": label_weight}


def sample_split(world: dict[str, Tensor], config: IPIBConfig, size: int, seed: int) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randint(0, 2, (size, 3, config.latent_dim), generator=generator).float().mul(2).sub(1)
    z1, z2, u3 = latent[:, 0], latent[:, 1], latent[:, 2]
    g12 = z1 * z2
    x1 = z1 @ world["p1"].T
    x2 = z2 @ world["p2"].T
    additive = z1 @ world["a31"].T + z2 @ world["a32"].T
    joint = g12 @ world["b3"].T
    private = u3 @ world["p3"].T
    x3 = additive + config.beta * joint + config.private_strength * private
    if config.noise_std:
        x1 = x1 + config.noise_std * torch.randn(x1.shape, generator=generator)
        x2 = x2 + config.noise_std * torch.randn(x2.shape, generator=generator)
        x3 = x3 + config.noise_std * torch.randn(x3.shape, generator=generator)
    task_score = g12 @ world["label_weight"]
    labels = (task_score > 0).long()
    return {
        "x1": x1.float(), "x2": x2.float(), "x3": x3.float(),
        "z1": z1.float(), "z2": z2.float(), "u3": u3.float(),
        "g12": g12.float(), "additive": additive.float(), "joint": joint.float(), "private": private.float(),
        "joint_target": (config.beta * joint).float(), "task_score": task_score.float(), "labels": labels,
    }


def make_dataset(config: IPIBConfig, seed: int, world_seed: int = 0):
    world = make_world(config, world_seed)
    return (
        sample_split(world, config, config.train_size, seed * 10 + 1),
        sample_split(world, config, config.validation_size, seed * 10 + 2),
        sample_split(world, config, config.test_size, seed * 10 + 3),
    )
