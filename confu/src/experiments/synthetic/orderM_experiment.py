"""M-modality order ladder: at which order does capacity stop working?

Task: M modalities observe latent bits x_1..x_M via fixed prototypes + noise.
Label: y = x_1 XOR x_2 XOR ... XOR x_M  (pure order-M synergy).

Information facts:
- any proper subset of modalities is independent of y  -> order-(M-1)
  interactions are at chance;
- concat-MLP is a universal approximator but parity hardness grows with M
  (SGD struggles; known staircase/grokking behavior);
- an explicit order-M multiplicative term should keep solving it.

Variants per M: additive (order-1), pairs_only (order-1+2), full (order-M term),
concat_mlp. Metrics: accuracy, per-modality shuffle drops, variance of the
top interaction, train seconds, parameter count. 5 seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.synthetic.synergy_experiment import make_prototypes
from src.modules.models.synergyformer import LowRankInteraction


def generate_M(n_samples: int, M: int, obs_dim: int, noise: float,
               protos: list[Tensor], gen: torch.Generator) -> tuple[list[Tensor], Tensor]:
    bits = torch.randint(0, 2, (n_samples, M), generator=gen)
    xs = [protos[i][bits[:, i]] + noise * torch.randn(n_samples, obs_dim, generator=gen)
          for i in range(M)]
    y = bits.sum(1) % 2
    return [x.float() for x in xs], y


class OrderMModel(nn.Module):
    def __init__(self, M: int, obs_dim: int, dim: int, rank: int,
                 pairs: bool, top: bool) -> None:
        super().__init__()
        self.M = M
        self.encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(obs_dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(M)
        )
        self.pair_indices = list(itertools.combinations(range(M), 2))
        self.pair_interactions = nn.ModuleList(
            LowRankInteraction(dim, rank, 2, normalize_factors=True,
                               factor_bias=False, output_bias=False)
            for _ in self.pair_indices
        ) if pairs else None
        self.top_interaction = (
            LowRankInteraction(dim, rank, M, normalize_factors=True,
                               factor_bias=False, output_bias=False)
            if top else None
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(M))
        self.pair_norm = nn.LayerNorm(dim)
        self.top_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)
        self.top_gate = nn.Parameter(torch.zeros(1))

    def forward(self, xs: list[Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        rs = [enc(x) for enc, x in zip(self.encoders, xs)]
        pooled = sum(norm(r) for norm, r in zip(self.norms, rs))
        pair_sum = torch.zeros_like(pooled)
        if self.pair_interactions is not None:
            for inter, (i, j) in zip(self.pair_interactions, self.pair_indices):
                pair_sum = pair_sum + self.pair_norm(inter(rs[i], rs[j]))
            pair_sum = pair_sum / max(1, len(self.pair_indices))
        top = (self.top_interaction(*rs) if self.top_interaction is not None
               else torch.zeros_like(pooled))
        # NOTE: no LayerNorm after aggregation — a final LN divides by a
        # cross-modal statistic and silently leaks higher-order information
        # (measured: additive+LN reaches 0.75 on 3-bit parity, additive without
        # LN is exactly 0.5). Keep the readout purely linear for a valid ladder.
        z = pooled + pair_sum + torch.sigmoid(self.top_gate) * self.top_norm(top)
        return self.head(z), {"top": top, **{f"r{i+1}": r for i, r in enumerate(rs)}}


class ConcatMLP(nn.Module):
    def __init__(self, M: int, obs_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M * obs_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, xs: list[Tensor]) -> tuple[Tensor, dict]:
        return self.net(torch.cat(xs, -1)), {}


def train_eval(variant: str, M: int, seed: int, args: argparse.Namespace) -> dict:
    torch.manual_seed(seed)
    proto_gen = torch.Generator().manual_seed(999)
    protos = [make_prototypes(args.obs_dim, proto_gen) for _ in range(M)]
    noise_gen = torch.Generator().manual_seed(seed)
    xs, y = generate_M(args.n_train, M, args.obs_dim, args.noise, protos, noise_gen)
    xs_t, y_t = generate_M(args.n_test, M, args.obs_dim, args.noise, protos, noise_gen)
    loader = DataLoader(TensorDataset(*xs, y), batch_size=512, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    if variant == "concat_mlp":
        model = ConcatMLP(M, args.obs_dim)
    else:
        model = OrderMModel(M, args.obs_dim, args.dim, args.rank,
                            pairs=variant != "additive", top=variant == "full")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    started = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for batch in loader:
            loss = F.cross_entropy(model(list(batch[:-1]))[0], batch[-1])
            opt.zero_grad(); loss.backward(); opt.step()
    duration = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        logits, extra = model(xs_t)
        acc = (logits.argmax(1) == y_t).float().mean().item()
        drops = []
        gen = torch.Generator().manual_seed(0)
        for i in range(M):
            xs_sh = list(xs_t); xs_sh[i] = xs_sh[i][torch.randperm(len(y_t), generator=gen)]
            drops.append(acc - (model(xs_sh)[0].argmax(1) == y_t).float().mean().item())
    return {
        "variant": variant, "M": M, "seed": seed, "accuracy": acc,
        "min_shuffle_drop": min(drops), "mean_shuffle_drop": sum(drops) / len(drops),
        "var_top": extra["top"].var(dim=0, unbiased=False).mean().item() if "top" in extra else None,
        "train_seconds": round(duration, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--Ms", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--variants", nargs="+",
                        default=["additive", "pairs_only", "full", "concat_mlp"])
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_test", type=int, default=4000)
    parser.add_argument("--obs_dim", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--out", default="results/synthetic/orderM")
    args = parser.parse_args()

    all_runs = []
    for M in args.Ms:
        for variant in args.variants:
            runs = [train_eval(variant, M, seed, args) for seed in range(1, args.seeds + 1)]
            accs = [r["accuracy"] for r in runs]
            import statistics
            mean, std = statistics.mean(accs), statistics.stdev(accs) if len(accs) > 1 else 0
            md = statistics.mean(r["min_shuffle_drop"] for r in runs)
            print(f"M={M} {variant:<11} acc={mean:.4f}±{std:.4f} min_shuffle_drop={md:+.4f} "
                  f"params={runs[0]['parameters']} time={sum(r['train_seconds'] for r in runs):.0f}s",
                  flush=True)
            all_runs.extend(runs)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "orderM_ladder.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "runs": all_runs}, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
