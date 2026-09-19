"""Controlled synthetic order benchmark: third-order synergy with ground truth.

Three modalities observe independent latent bits a, b, c through fixed random
prototypes + Gaussian noise:

    Xi = proto_i(bit_i) + sigma * noise,   i in {1, 2, 3}

Labels:
    T3 (order-3):  y = a XOR b XOR c
        Pure third-order synergy. y is INDEPENDENT of any single modality and
        of any PAIR (a XOR b is statistically independent of a XOR b XOR c).
        Additive, concat, and pair-interaction models are information-
        theoretically capped at 50%. Only a genuine third-order interaction
        can exceed chance. This is the decisive "the more, the merrier" test.

    T2 (order-2 control):  y = a XOR b   (c is a distractor modality)
        Pair interaction must solve it; the third-order term must NOT be
        needed; honesty control: nothing may fabricate c dependence.

Variants:
    concat_mlp     raw concat MLP on [x1;x2;x3]
    additive       order-1 sum composition (SynergyFormer, pairs+third off)
    pairs_only     SynergyFormer with pair interactions, no r123
    full           SynergyFormer with pair + third-order interactions
    full + S1      full + adversarial de-shortcutting on r123
                   (r123 must be unpredictable from each single modality
                   and from each PAIR of modalities)

5 seeds per cell; metrics: test accuracy, per-representation probes,
predictability of r123 from singles/pairs, modality shuffle drops.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.synthetic.synergy_experiment import make_prototypes, probe_accuracy, predictability_r2
from src.modules.models.complementarity import (
    AdversarialPredictor,
    EMAStandardizer,
    shortcut_hinge_loss,
)
from src.modules.models.synergyformer import (
    MLPEncoder,
    SynergyFormer,
    covariance_loss,
    variance_loss,
)


def generate_triplet(
    n_samples: int,
    obs_dim: int,
    noise: float,
    protos: tuple[Tensor, Tensor, Tensor],
    gen: torch.Generator,
    order: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    a = torch.randint(0, 2, (n_samples,), generator=gen)
    b = torch.randint(0, 2, (n_samples,), generator=gen)
    c = torch.randint(0, 2, (n_samples,), generator=gen)
    p1, p2, p3 = protos
    x1 = p1[a] + noise * torch.randn(n_samples, obs_dim, generator=gen)
    x2 = p2[b] + noise * torch.randn(n_samples, obs_dim, generator=gen)
    x3 = p3[c] + noise * torch.randn(n_samples, obs_dim, generator=gen)
    y = (a ^ b ^ c) if order == 3 else (a ^ b)
    return x1.float(), x2.float(), x3.float(), y


class ConcatMLPBaseline(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * obs_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> Tensor:
        return self.net(torch.cat([x1, x2, x3], dim=-1))


class OrderModel(nn.Module):
    """SynergyFormer (order-1/2/3) with a linear task head."""

    def __init__(self, obs_dim: int, dim: int, rank: int, pairs: bool, third: bool) -> None:
        super().__init__()
        encoders = [MLPEncoder(obs_dim, dim) for _ in range(3)]
        self.core = SynergyFormer(
            encoders=encoders, dim=dim, rank=rank, aggregator="sum",
            include_third_order=third,
        )
        if not pairs:
            # keep only order-1 terms: interactions return zero regardless of input
            class _ZeroInteraction(nn.Module):
                def forward(self, *inputs: Tensor) -> Tensor:
                    return torch.zeros_like(inputs[0])
            self.core.pair_interactions = nn.ModuleList(_ZeroInteraction() for _ in range(3))
        self.head = nn.Linear(dim, 2)

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> tuple[Tensor, dict]:
        out = self.core(x1, x2, x3)
        return self.head(out["z_global"]), out


def shuffle_drops(model: nn.Module, x: list[Tensor], y: Tensor, kind: str) -> dict[str, float]:
    model.eval()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        logits, out = model(*x) if kind != "concat" else (model(*x), None)
        base_acc = (logits.argmax(1) == y).float().mean().item()
        result = {"accuracy": base_acc}
        if out is not None:
            for i, name in enumerate(["r1", "r2", "r3"]):
                xs = list(x)
                perm = torch.randperm(len(y), generator=gen)
                xs[i] = xs[i][perm]
                shuffled_logits, _ = model(*xs)
                result[f"shuffle_{name}_drop"] = base_acc - (
                    shuffled_logits.argmax(1) == y
                ).float().mean().item()
    return result


def train_one(variant: str, s1: float, seed: int, args: argparse.Namespace) -> dict:
    torch.manual_seed(seed)
    proto_gen = torch.Generator().manual_seed(54321)
    protos = tuple(make_prototypes(args.obs_dim, proto_gen) for _ in range(3))
    noise_gen = torch.Generator().manual_seed(seed)
    x1, x2, x3, y = generate_triplet(args.n_train, args.obs_dim, args.noise, protos, noise_gen, args.order)
    x1t, x2t, x3t, yt = generate_triplet(args.n_test, args.obs_dim, args.noise, protos, noise_gen, args.order)
    train_loader = DataLoader(TensorDataset(x1, x2, x3, y), batch_size=512, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))

    if variant == "concat_mlp":
        model = ConcatMLPBaseline(args.obs_dim)
    else:
        model = OrderModel(args.obs_dim, args.dim, args.rank,
                           pairs=variant != "additive", third=variant == "full")
    use_s1 = s1 > 0 and variant == "full"
    # S1 for order-3: adversaries predict r123 from each single AND each pair
    adv_singles = nn.ModuleList(AdversarialPredictor(args.dim, args.adv_hidden) for _ in range(3))
    adv_pairs = nn.ModuleList(AdversarialPredictor(2 * args.dim, args.adv_hidden, args.dim) for _ in range(3))
    standardizer = EMAStandardizer(args.dim)

    gen_params = list(model.parameters())
    opt_gen = torch.optim.AdamW(gen_params, lr=args.lr, weight_decay=args.weight_decay)
    adv_params = list(adv_singles.parameters()) + list(adv_pairs.parameters())
    opt_adv = torch.optim.AdamW(adv_params, lr=args.lr * 0.5, weight_decay=1e-4)

    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for bx1, bx2, bx3, by in train_loader:
            if variant == "concat_mlp":
                logits = model(bx1, bx2, bx3)
                loss = F.cross_entropy(logits, by)
                opt_gen.zero_grad(); loss.backward(); opt_gen.step()
                continue
            logits, out = model(bx1, bx2, bx3)
            loss = F.cross_entropy(logits, by)
            r123 = out["r123"]
            loss = loss + args.lambda_variance * variance_loss(r123, 1.0)
            loss = loss + args.lambda_covariance * covariance_loss(r123)
            if use_s1:
                standardizer.update(r123)
                t_const = standardizer.standardize(r123.detach())
                singles = [out["r1"], out["r2"], out["r3"]]
                pairs = [torch.cat([out["r1"], out["r2"]], -1),
                         torch.cat([out["r1"], out["r3"]], -1),
                         torch.cat([out["r2"], out["r3"]], -1)]
                # adversary step
                for _ in range(args.k_adv):
                    loss_adv = sum(
                        F.mse_loss(q(s.detach()), t_const) for q, s in zip(adv_singles, singles)
                    ) + sum(F.mse_loss(q(p.detach()), t_const) for q, p in zip(adv_pairs, pairs))
                    opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()
                # generator step (hinged)
                with torch.no_grad():
                    preds = [q(s) for q, s in zip(adv_singles, singles)] + \
                            [q(p) for q, p in zip(adv_pairs, pairs)]
                t_grad = standardizer.standardize(r123)
                pen = torch.stack([F.relu(args.tau - F.mse_loss(p, t_grad)) for p in preds]).sum()
                warmup = min(1.0, (epoch + 1) / max(1.0, args.adv_warmup_fraction * args.epochs))
                loss = loss + (s1 * warmup) * pen
            opt_gen.zero_grad(); loss.backward(); opt_gen.step()
    duration = time.perf_counter() - started

    # ---------------- evaluation (probes: train on train-split, score on test-split) ----------------
    model.eval()
    result = {"variant": variant, "s1": s1, "order": args.order, "seed": seed,
              "train_seconds": round(duration, 1)}
    n_probe = min(10000, len(y))
    with torch.no_grad():
        if variant == "concat_mlp":
            logits = model(x1t, x2t, x3t)
            result["accuracy"] = (logits.argmax(1) == yt).float().mean().item()
            result.update(shuffle_drops(model, [x1t, x2t, x3t], yt, "concat"))
        else:
            logits, out = model(x1t, x2t, x3t)
            result["accuracy"] = (logits.argmax(1) == yt).float().mean().item()
            result.update(shuffle_drops(model, [x1t, x2t, x3t], yt, "order"))
            _, out_tr = model(x1[:n_probe], x2[:n_probe], x3[:n_probe])
            y_tr = y[:n_probe]
            for name in ("r1", "r2", "r3", "r12", "r13", "r23", "r123"):
                result[f"probe_{name}"] = probe_accuracy(out_tr[name], y_tr, out[name], yt)
            if variant == "full":
                pair_cat_tr = torch.cat([out_tr["r12"], out_tr["r13"], out_tr["r23"]], -1)
                pair_cat = torch.cat([out["r12"], out["r13"], out["r23"]], -1)
                result["probe_pairs_cat"] = probe_accuracy(pair_cat_tr, y_tr, pair_cat, yt)
                result["pred_r123_from_pairs"] = predictability_r2(pair_cat_tr, out_tr["r123"])
                result["pred_r123_from_r1"] = predictability_r2(out_tr["r1"], out_tr["r123"])
                result["var_r123"] = out["r123"].var(dim=0, unbiased=False).mean().item()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=3, choices=[2, 3])
    parser.add_argument("--variant", default="full",
                        choices=["concat_mlp", "additive", "pairs_only", "full"])
    parser.add_argument("--s1", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--k_adv", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_test", type=int, default=4000)
    parser.add_argument("--obs_dim", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--adv_hidden", type=int, default=64)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_variance", type=float, default=1.0)
    parser.add_argument("--lambda_covariance", type=float, default=0.5)
    parser.add_argument("--adv_warmup_fraction", type=float, default=0.2)
    parser.add_argument("--out", default="results/synthetic/order3")
    args = parser.parse_args()

    runs = []
    for seed in range(1, args.seeds + 1):
        metrics = train_one(args.variant, args.s1, seed, args)
        runs.append(metrics)
        line = (f"[{args.variant} s1={args.s1} order={args.order} seed={seed}] "
                f"acc={metrics['accuracy']:.4f}")
        if "probe_r123" in metrics:
            line += (f" probe_r123={metrics['probe_r123']:.3f} probe_r12={metrics['probe_r12']:.3f} "
                     f"probe_pairs={metrics.get('probe_pairs_cat', 0):.3f}")
        print(line, flush=True)

    import statistics
    summary = {}
    for k in runs[0]:
        if isinstance(runs[0][k], float) and k != "train_seconds":
            vals = [r[k] for r in runs]
            summary[k] = f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals) if len(vals) > 1 else 0:.4f}"
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"order{args.order}_{args.variant}_s1_{args.s1}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "runs": runs, "summary": summary}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
