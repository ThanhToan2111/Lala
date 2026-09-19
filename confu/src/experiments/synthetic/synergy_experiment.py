"""Stage 0: synthetic ground-truth validation of representation-level complementarity.

Generates two-modal data with KNOWN latent structure (see AGENT.md
ConFu++ Representation-Level Complementarity, section 5):

    a, b ~ Bernoulli(0.5)
    X1 = prototype1(a) + sigma * noise      (sees only a)
    X2 = prototype2(b) + sigma * noise      (sees only b)

Configs:
    A  y = a XOR b            (pure synergy: neither modality alone predicts y)
    B  y = a                  (no synergy: modality 1 sufficient)
    C  y = a with prob rho, else a XOR b   (tunable redundancy)

The experiment trains the same low-rank interaction + gated composition
architecture used on AV-MNIST, with and without the S1 adversarial
de-shortcutting objective, and measures against ground truth:

    - task accuracy
    - selectivity: probe(r12 -> y) - max(probe(r1 -> y), probe(r2 -> y))
    - predictability R2 of r12 from r1 alone / r2 alone (nonlinear probes)
    - single-modality shuffle drops
    - probe(r1 -> a), probe(r2 -> b)  (encoders keep their factor)

Usage:
    python -m src.experiments.synthetic.synergy_experiment --config A --s1 0 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.modules.models.complementarity import (
    AdversarialPredictor,
    EMAStandardizer,
    adversary_loss,
    factor_variance_loss,
    shortcut_hinge_loss,
)
from src.modules.models.synergyformer import (
    GatedComposition,
    LowRankInteraction,
    conditional_utility_loss,
    covariance_loss,
    pair_cross_covariance_loss,
    variance_loss,
)


def make_prototypes(obs_dim: int, generator: torch.Generator) -> Tensor:
    """Two well-separated random prototypes per modality."""
    prototypes = torch.randn(2, obs_dim, generator=generator)
    prototypes = F.normalize(prototypes, dim=-1) * math.sqrt(obs_dim)
    return prototypes


def generate_split(
    config: str,
    n_samples: int,
    obs_dim: int,
    noise: float,
    rho: float,
    protos: tuple[Tensor, Tensor, Tensor],
    noise_gen: torch.Generator,
    flag_dim: int = 8,
    noise2: float | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Train/test MUST share the same prototypes (one latent world).

    Config C is shortcut-structured but SOLVABLE: an observable routing
    flag is appended to modality 2 (flag -> y = a; else y = a XOR b).
    Bayes-optimal accuracy is 1.0; ignoring the XOR rule caps at
    rho + (1 - rho) / 2 (e.g. 0.75 at rho=0.5) - the correct shortcut floor.
    """
    p1, p2, uf = protos
    a = torch.randint(0, 2, (n_samples,), generator=noise_gen)
    b = torch.randint(0, 2, (n_samples,), generator=noise_gen)
    x1 = p1[a] + noise * torch.randn(n_samples, obs_dim, generator=noise_gen)
    xor = (a ^ b)
    flag = torch.zeros(n_samples, dtype=torch.long)
    if config == "A":
        y = xor
        use_a = torch.ones(n_samples, dtype=torch.bool)
    elif config == "B":
        y = a
        use_a = torch.ones(n_samples, dtype=torch.bool)
    elif config == "C":
        flag = (torch.rand(n_samples, generator=noise_gen) < rho).long()
        y = torch.where(flag.bool(), a, xor)
        use_a = flag.bool()
    else:
        raise ValueError(f"unknown config {config}")
    n2 = noise if noise2 is None else noise2
    x2_core = p2[b] + n2 * torch.randn(n_samples, obs_dim, generator=noise_gen)
    x2_flag = uf[flag] + noise * torch.randn(n_samples, flag_dim, generator=noise_gen)
    x2 = torch.cat([x2_core, x2_flag], dim=1)
    mask_a = use_a if config == "C" else torch.ones(n_samples, dtype=torch.bool)
    return x1.float(), x2.float(), y, a, b, mask_a


class SyntheticSynergyModel(nn.Module):
    """Same architectural family as the AV-MNIST synergy model."""

    def __init__(self, input_dims: tuple[int, int], dim: int, rank: int, classifier: str = "prototype") -> None:
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Linear(input_dims[0], dim), nn.GELU(), nn.Linear(dim, dim))
        self.encoder2 = nn.Sequential(nn.Linear(input_dims[1], dim), nn.GELU(), nn.Linear(dim, dim))
        self.interaction = LowRankInteraction(
            dim, rank, 2, normalize_factors=True, factor_bias=False, output_bias=False
        )
        self.composition = GatedComposition(dim)
        self.unimodal_norms = nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim)])
        self.additive_norm = nn.LayerNorm(dim)
        self.classifier_type = classifier
        self.prototypes = nn.Parameter(torch.randn(2, dim) / dim**0.5)
        self.head = nn.Linear(dim, 2)

    def additive(self, r1: Tensor, r2: Tensor) -> Tensor:
        return self.additive_norm(self.unimodal_norms[0](r1) + self.unimodal_norms[1](r2))

    def classify(self, representation: Tensor) -> Tensor:
        if self.classifier_type == "linear":
            return self.head(representation)
        return F.normalize(representation, dim=-1) @ F.normalize(self.prototypes, dim=-1).T / 0.07

    def forward(self, x1: Tensor, x2: Tensor) -> dict[str, Tensor]:
        r1 = self.encoder1(x1)
        r2 = self.encoder2(x2)
        r12 = self.interaction(r1, r2)
        return {"r1": r1, "r2": r2, "r12": r12, "fused": self.composition(r1, r2, r12)}


@torch.enable_grad()
def probe_accuracy(train_x: Tensor, train_y: Tensor, test_x: Tensor, test_y: Tensor,
                   hidden: int = 64, epochs: int = 30, seed: int = 0) -> float:
    """Small MLP probe (parameter-matched across comparisons by caller)."""
    dim = train_x.shape[1]
    mean, std = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    probe = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for idx in torch.randperm(len(train_x), generator=generator).split(512):
            loss = F.cross_entropy(probe(train_x[idx]), train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    probe.eval()
    with torch.no_grad():
        return (probe(test_x).argmax(1) == test_y).float().mean().item()


@torch.enable_grad()
def predictability_r2(source: Tensor, target: Tensor, hidden: int = 64, epochs: int = 40,
                      seed: int = 0) -> float:
    """Nonlinear R2 of predicting target from source (matches eval probe class)."""
    s_mean, s_std = source.mean(0), source.std(0, unbiased=False).clamp_min(1e-6)
    t_mean, t_std = target.mean(0), target.std(0, unbiased=False).clamp_min(1e-6)
    src = (source - s_mean) / s_std
    tgt = (target - t_mean) / t_std
    torch.manual_seed(seed)
    predictor = nn.Sequential(nn.Linear(src.shape[1], hidden), nn.GELU(), nn.Linear(hidden, tgt.shape[1]))
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    n = len(src)
    for _ in range(epochs):
        for idx in torch.randperm(n, generator=generator).split(512):
            loss = F.mse_loss(predictor(src[idx]), tgt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    predictor.eval()
    with torch.no_grad():
        mse = F.mse_loss(predictor(src), tgt).item()
    return 1.0 - mse


@torch.no_grad()
def evaluate(model: SyntheticSynergyModel, loader: DataLoader, config: str,
             a_test: Tensor, b_test: Tensor, mask_test: Tensor | None = None) -> dict[str, float]:
    model.eval()
    stored = {"r1": [], "r2": [], "r12": []}
    logits = {"normal": [], "additive": [], "zero": [], "shuffle1": [], "shuffle2": []}
    labels_all = []
    generator = torch.Generator().manual_seed(0)
    for x1, x2, y in loader:
        out = model(x1, x2)
        for name in stored:
            stored[name].append(out[name])
        logits["normal"].append(model.classify(out["fused"]))
        logits["additive"].append(model.classify(model.additive(out["r1"], out["r2"])))
        logits["zero"].append(model.classify(model.composition(out["r1"], out["r2"], torch.zeros_like(out["r12"]))))
        perm1 = torch.randperm(len(y), generator=generator)
        perm2 = torch.randperm(len(y), generator=generator)
        s1 = model.interaction(out["r1"][perm1], out["r2"])
        s2 = model.interaction(out["r1"], out["r2"][perm2])
        logits["shuffle1"].append(model.classify(model.composition(out["r1"], out["r2"], s1)))
        logits["shuffle2"].append(model.classify(model.composition(out["r1"], out["r2"], s2)))
        labels_all.append(y)
    labels = torch.cat(labels_all)
    reps = {k: torch.cat(v) for k, v in stored.items()}
    result = {}
    accs = {}
    for name, chunks in logits.items():
        accs[name] = (torch.cat(chunks).argmax(1) == labels).float().mean().item()
    result["accuracy"] = accs["normal"]
    result["gain12"] = accs["normal"] - accs["additive"]
    result["zero_drop"] = accs["normal"] - accs["zero"]
    result["shuffle1_drop"] = accs["normal"] - accs["shuffle1"]
    result["shuffle2_drop"] = accs["normal"] - accs["shuffle2"]
    result["var_r12"] = reps["r12"].var(dim=0, unbiased=False).mean().item()
    if config == "C" and mask_test is not None:
        pred = torch.cat(logits["normal"]).argmax(1)
        a_frac = mask_test.bool()
        result["acc_a_fraction"] = (pred[a_frac] == labels[a_frac]).float().mean().item()
        result["acc_xor_fraction"] = (pred[~a_frac] == labels[~a_frac]).float().mean().item()
    # selectivity against GROUND TRUTH label
    probe_r12 = probe_accuracy(reps["r12"], labels, reps["r12"], labels)
    probe_r1 = probe_accuracy(reps["r1"], labels, reps["r1"], labels)
    probe_r2 = probe_accuracy(reps["r2"], labels, reps["r2"], labels)
    result["probe_r12"] = probe_r12
    result["probe_r1"] = probe_r1
    result["probe_r2"] = probe_r2
    result["selectivity"] = probe_r12 - max(probe_r1, probe_r2)
    # factor retention: r1 should know a, r2 should know b
    result["probe_r1_to_a"] = probe_accuracy(reps["r1"], a_test, reps["r1"], a_test)
    result["probe_r2_to_b"] = probe_accuracy(reps["r2"], b_test, reps["r2"], b_test)
    # the headline meter: nonlinear predictability of r12 from ONE modality
    result["pred_r2_from_r1"] = predictability_r2(reps["r1"], reps["r12"])
    result["pred_r2_from_r2"] = predictability_r2(reps["r2"], reps["r12"])
    return result


def train_one(config: str, s1: float, seed: int, args: argparse.Namespace) -> dict[str, float]:
    torch.manual_seed(seed)
    proto_gen = torch.Generator().manual_seed(12345)  # same world across seeds
    protos = (make_prototypes(args.obs_dim, proto_gen), make_prototypes(args.obs_dim, proto_gen),
              make_prototypes(args.flag_dim, proto_gen))
    noise_gen = torch.Generator().manual_seed(seed)
    x1_tr, x2_tr, y_tr, _, _, _ = generate_split(config, args.n_train, args.obs_dim, args.noise,
                                              args.rho, protos, noise_gen, noise2=args.noise2)
    x1_te, x2_te, y_te, a_te, b_te, mask_te = generate_split(config, args.n_test, args.obs_dim, args.noise,
                                                    args.rho, protos, noise_gen, noise2=args.noise2)
    train_loader = DataLoader(TensorDataset(x1_tr, x2_tr, y_tr), batch_size=256, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(TensorDataset(x1_te, x2_te, y_te), batch_size=512)

    model = SyntheticSynergyModel((args.obs_dim, args.obs_dim + args.flag_dim), args.dim,
                                  args.rank, args.classifier)
    q1 = AdversarialPredictor(args.dim, args.adv_hidden)
    q2 = AdversarialPredictor(args.dim, args.adv_hidden)
    standardizer = EMAStandardizer(args.dim)
    gen_params = list(model.parameters())
    opt_gen = torch.optim.AdamW(gen_params, lr=args.lr, weight_decay=args.weight_decay)
    opt_adv = torch.optim.AdamW(list(q1.parameters()) + list(q2.parameters()), lr=args.lr * 0.5,
                                weight_decay=1e-4)

    use_s1 = s1 > 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        meters = {"task": 0.0, "shortcut": 0.0, "adv_mse1": 0.0, "adv_mse2": 0.0, "n": 0}
        for x1, x2, y in train_loader:
            out = model(x1, x2)
            # ---------------- adversary step ----------------
            if use_s1:
                standardizer.update(out["r12"])
            t_const = standardizer.standardize(out["r12"].detach())
            if use_s1:
                for _ in range(args.k_adv):
                    loss_adv, mse1, mse2 = adversary_loss(
                        q1, q2, out["r1"].detach(), out["r2"].detach(), t_const)
                    opt_adv.zero_grad()
                    loss_adv.backward()
                    opt_adv.step()
            # ---------------- generator step ----------------
            full_logits = model.classify(out["fused"])
            full_losses = F.cross_entropy(full_logits, y, reduction="none")
            base_losses = F.cross_entropy(model.classify(model.additive(out["r1"], out["r2"])), y,
                                          reduction="none")
            if args.lambda_hard > 0:
                w = base_losses.detach()
                w = w + args.hard_floor * w.mean()  # keep a floor for easy samples
                hard_term = (w * full_losses).sum() / w.sum().clamp_min(1e-6)
                loss = (1 - args.lambda_hard) * full_losses.mean() + args.lambda_hard * hard_term
            else:
                loss = full_losses.mean()
            loss = loss + args.lambda_unimodal * sum(
                F.cross_entropy(model.classify(out[n]), y) for n in ("r1", "r2"))
            loss = loss + args.lambda_variance * variance_loss(out["r12"], 1.0)
            loss = loss + args.lambda_cross_covariance * pair_cross_covariance_loss(
                out["r12"], out["r1"], out["r2"])
            loss = loss + args.lambda_covariance * covariance_loss(out["r12"])
            loss = loss + args.lambda_utility * conditional_utility_loss(
                full_losses, base_losses, args.utility_margin)
            if use_s1:
                with torch.no_grad():
                    pred1 = q1(out["r1"])
                    pred2 = q2(out["r2"])
                t_grad = standardizer.standardize(out["r12"])
                penalty, gmse1, gmse2 = shortcut_hinge_loss(pred1, pred2, t_grad, args.tau)
                warmup = min(1.0, (epoch + 1) / max(1.0, args.adv_warmup_fraction * args.epochs))
                loss = loss + (s1 * warmup) * penalty
                u1, u2 = model.interaction.project_factors(out["r1"], out["r2"])
                loss = loss + args.lambda_factorvar * factor_variance_loss([u1, u2])
                meters["shortcut"] += penalty.item() * len(y)
                meters["adv_mse1"] += gmse1.item() * len(y)
                meters["adv_mse2"] += gmse2.item() * len(y)
            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()
            meters["task"] += full_losses.mean().item() * len(y)
            meters["n"] += len(y)
    duration = time.perf_counter() - started

    metrics = evaluate(model, test_loader, config, a_te, b_te, mask_te)
    metrics.update({
        "config": config, "s1": s1, "seed": seed,
        "train_seconds": round(duration, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
    })
    return metrics


def mean_std(values: list[float]) -> str:
    t = torch.tensor(values, dtype=torch.float64)
    return f"{t.mean().item():.4f} ± {t.std().item():.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="A", choices=["A", "B", "C"])
    parser.add_argument("--s1", type=float, default=0.0, help="lambda_shortcut (0 = baseline)")
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
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lambda_unimodal", type=float, default=0.2)
    parser.add_argument("--lambda_variance", type=float, default=1.0)
    parser.add_argument("--lambda_cross_covariance", type=float, default=0.01)
    parser.add_argument("--lambda_covariance", type=float, default=0.5)
    parser.add_argument("--lambda_utility", type=float, default=0.2)
    parser.add_argument("--lambda_factorvar", type=float, default=0.5)
    parser.add_argument("--adv_warmup_fraction", type=float, default=0.2)
    parser.add_argument("--classifier", default="prototype", choices=["prototype", "linear"])
    parser.add_argument("--lambda_hard", type=float, default=0.0)
    parser.add_argument("--utility_margin", type=float, default=0.0)
    parser.add_argument("--hard_floor", type=float, default=0.3)
    parser.add_argument("--flag_dim", type=int, default=8)
    parser.add_argument("--noise2", type=float, default=None)
    parser.add_argument("--out", default="results/synthetic")
    args = parser.parse_args()

    runs = []
    for seed in range(1, args.seeds + 1):
        metrics = train_one(args.config, args.s1, seed, args)
        runs.append(metrics)
        print(f"[config={args.config} s1={args.s1} seed={seed}] "
              f"acc={metrics['accuracy']:.4f} gain12={metrics['gain12']:+.4f} "
              f"sel={metrics['selectivity']:+.4f} "
              f"predR2(r1)={metrics['pred_r2_from_r1']:.3f} predR2(r2)={metrics['pred_r2_from_r2']:.3f} "
              f"sh1={metrics['shuffle1_drop']:+.4f} sh2={metrics['shuffle2_drop']:+.4f}", flush=True)

    summary = {k: mean_std([r[k] for r in runs]) for k in (
        "accuracy", "gain12", "zero_drop", "shuffle1_drop", "shuffle2_drop",
        "selectivity", "pred_r2_from_r1", "pred_r2_from_r2",
        "probe_r1_to_a", "probe_r2_to_b", "var_r12")}
    payload = {"args": vars(args), "runs": runs, "summary_mean_std": summary}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"synergy_config{args.config}_s1_{args.s1}_hard_{args.lambda_hard}_n2_{args.noise2}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
