"""Three-way parity benchmark for order-specific interaction models."""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.modules.models.synergyformer import (
    MLPEncoder,
    SynergyFormer,
    redundancy_loss,
    symmetric_info_nce,
)


def make_parity(size: int, noise: float, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (size, 3), generator=generator)
    inputs = bits.float().mul(2).sub(1)
    inputs += torch.randn(inputs.shape, generator=generator) * noise
    labels = bits.sum(1).remainder(2).long()
    return TensorDataset(inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], labels)


class ConcatMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        return self.net(torch.cat([x1, x2, x3], dim=-1)), {}


class PairwiseContrastive(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(MLPEncoder(1, dim) for _ in range(3))
        self.head = nn.Linear(dim * 3, 2)

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        reps = [encoder(value)[1] for encoder, value in zip(self.encoders, (x1, x2, x3))]
        return self.head(torch.cat(reps, dim=-1)), dict(zip(("r1", "r2", "r3"), reps))


class ConFuPairFusion(nn.Module):
    """Pairwise concat MLPs followed by an additive linear classifier."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pairs = nn.ModuleList(
            nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim)) for _ in range(3)
        )
        self.head = nn.Linear(dim * 3, 2)

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        values = (x1, x2, x3)
        reps = [module(torch.cat([values[i], values[j]], -1))
                for module, (i, j) in zip(self.pairs, ((0, 1), (0, 2), (1, 2)))]
        return self.head(torch.cat(reps, -1)), dict(zip(("r12", "r13", "r23"), reps))


class SynergyClassifier(nn.Module):
    def __init__(self, dim: int, rank: int, include_third_order: bool) -> None:
        super().__init__()
        self.model = SynergyFormer(
            (1, 1, 1), dim=dim, rank=rank, aggregator="sum",
            include_third_order=include_third_order,
        )
        self.head = nn.Linear(dim, 2)

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> tuple[Tensor, dict[str, object]]:
        outputs = self.model(x1, x2, x3)
        return self.head(outputs["z_global"]), outputs


@dataclass(frozen=True)
class ModelCase:
    name: str
    kind: str


CASES = (
    ModelCase("concat_mlp", "concat"),
    ModelCase("pairwise_contrastive", "contrastive"),
    ModelCase("confu_pair_fusion", "confu"),
    ModelCase("synergy_without_r123", "synergy_without"),
    ModelCase("synergy_with_r123", "synergy_with"),
)


def build_model(case: ModelCase, cfg: DictConfig) -> nn.Module:
    if case.kind == "concat":
        return ConcatMLP(cfg.dim)
    if case.kind == "contrastive":
        return PairwiseContrastive(cfg.dim)
    if case.kind == "confu":
        return ConFuPairFusion(cfg.dim)
    return SynergyClassifier(cfg.dim, cfg.rank, include_third_order=case.kind == "synergy_with")


def auxiliary_loss(case: ModelCase, outputs: dict[str, object], cfg: DictConfig, reference: Tensor) -> Tensor:
    loss = reference.sum() * 0
    def contrastive(left: Tensor, right: Tensor) -> Tensor:
        size = min(cfg.contrastive_batch_size, left.shape[0])
        return symmetric_info_nce(left[:size], right[:size], cfg.temperature)

    if case.kind == "contrastive":
        loss = cfg.lambda_contrastive * sum(
            contrastive(outputs[a], outputs[b])
            for a, b in (("r1", "r2"), ("r1", "r3"), ("r2", "r3"))
        ) / 3
    elif case.kind.startswith("synergy"):
        loss = cfg.lambda_contrastive * sum(
            contrastive(outputs[a], outputs[b])
            for a, b in (("r1", "r2"), ("r1", "r3"), ("r2", "r3"))
        ) / 3 + cfg.lambda_redundancy * redundancy_loss(outputs)
    return loss


def macro_f1(predictions: Tensor, labels: Tensor) -> float:
    scores = []
    for label in (0, 1):
        true_positive = ((predictions == label) & (labels == label)).sum().item()
        false_positive = ((predictions == label) & (labels != label)).sum().item()
        false_negative = ((predictions != label) & (labels == label)).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def train_one(case: ModelCase, cfg: DictConfig, seed: int, device: torch.device) -> dict[str, float]:
    torch.manual_seed(seed)
    model = build_model(case, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_data = make_parity(cfg.train_size, cfg.noise, seed)
    loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    for _ in range(cfg.epochs):
        for batch in loader:
            x1, x2, x3, labels = (value.to(device) for value in batch)
            logits, outputs = model(x1, x2, x3)
            loss = F.cross_entropy(logits, labels) + auxiliary_loss(case, outputs, cfg, logits)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss for {case.name}, seed {seed}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    test_data = make_parity(cfg.test_size, cfg.noise, seed + 10_000)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size)
    predictions, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x1, x2, x3, labels = (value.to(device) for value in batch)
            logits, _ = model(x1, x2, x3)
            predictions.append(logits.argmax(1).cpu())
            labels_all.append(labels.cpu())
    predictions, labels_all = torch.cat(predictions), torch.cat(labels_all)
    return {
        "accuracy": (predictions == labels_all).float().mean().item(),
        "macro_f1": macro_f1(predictions, labels_all),
    }


@hydra.main(config_path="../../configs", config_name="xor", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device)
    raw = {case.name: [train_one(case, cfg, seed, device) for seed in cfg.seeds] for case in CASES}
    summary = {
        name: {
            metric: {
                "mean": statistics.mean(run[metric] for run in runs),
                "std": statistics.stdev(run[metric] for run in runs),
            }
            for metric in ("accuracy", "macro_f1")
        }
        for name, runs in raw.items()
    }
    summary["synergy_gain_r123"] = {
        "mean": statistics.mean(
            full["accuracy"] - ablated["accuracy"]
            for full, ablated in zip(raw["synergy_with_r123"], raw["synergy_without_r123"])
        ),
    }
    result = {"config": OmegaConf.to_container(cfg, resolve=True), "runs": raw, "summary": summary}
    os.makedirs(os.path.dirname(cfg.results_path), exist_ok=True)
    with open(cfg.results_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    for case in CASES:
        metrics = summary[case.name]
        print(f"{case.name:24} accuracy={metrics['accuracy']['mean']:.4f}±{metrics['accuracy']['std']:.4f} "
              f"macro_f1={metrics['macro_f1']['mean']:.4f}±{metrics['macro_f1']['std']:.4f}")
    print(f"synergy_gain_r123       {summary['synergy_gain_r123']['mean']:+.4f}")


if __name__ == "__main__":
    main()
