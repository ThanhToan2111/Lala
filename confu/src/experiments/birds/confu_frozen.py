"""ConFu on frozen Bird-MML embeddings, with optional ConFu++ S1 de-shortcutting.

Faithful reproduction of the original ConFu objective (pairwise InfoNCE +
higher-order fused-vs-remaining InfoNCE) on frozen pretrained unimodal
features, plus the ConFu++ S1 extension: per-pair adversarial hinge penalties
so that z_ij stays unpredictable from z_i alone and from z_j alone.

Protocol (matches the project's confirmatory rules):
    - fixed stratified split from data/bird_mml/split.npz
    - 5 seeds; paired comparison confu vs confu+s1
    - evaluation: frozen linear probes on lower-order vs full representation
      (conditional utility), predictability probes R2(zi -> zij), and
      probe-level per-modality shuffle drops (dependence)
    - inference parameter count reported (adversaries are training-only)

Usage:
    python -m src.experiments.birds.confu_frozen --s1 0.0 --seeds 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.modules.models.complementarity import (
    AdversarialPredictor,
    EMAStandardizer,
    shortcut_hinge_loss,
)
from src.modules.models.synergyformer import symmetric_info_nce

PAIRS = [(0, 1), (0, 2), (1, 2)]  # image, audio, text
MODALITY = ["image", "audio", "text"]


class ConFuFrozen(nn.Module):
    def __init__(self, in_dims: list[int], embed_dim: int, hidden: int, n_classes: int = 0) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Linear(d, embed_dim) for d in in_dims)
        self.fusions = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * embed_dim, hidden), nn.GELU(),
                          nn.Linear(hidden, embed_dim))
            for _ in PAIRS
        )
        self.n_classes = n_classes
        if n_classes:
            # task heads for the conditional-utility objective (training-time;
            # heads stay in the model but add < 3% params)
            self.head_base = nn.Linear(3 * embed_dim, n_classes)
            self.head_full = nn.Linear(6 * embed_dim, n_classes)
            # per-pair utility heads: force each z_ij to add over (z_i, z_j)
            self.pair_heads = nn.ModuleDict({
                f"{i+1}{j+1}": nn.ModuleDict({
                    "lo": nn.Linear(2 * embed_dim, n_classes),
                    "fu": nn.Linear(3 * embed_dim, n_classes),
                }) for (i, j) in PAIRS
            })

    def task_logits(self, out: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        base = torch.cat([out["z1"], out["z2"], out["z3"]], -1)
        full = torch.cat([base, out["z12"], out["z13"], out["z23"]], -1)
        return self.head_base(base), self.head_full(full)

    def forward(self, feats: list[Tensor]) -> dict[str, Tensor]:
        z = [F.normalize(p(f), dim=-1) for p, f in zip(self.projections, feats)]
        out = {"z1": z[0], "z2": z[1], "z3": z[2]}
        for (i, j), fusion in zip(PAIRS, self.fusions):
            out[f"z{i+1}{j+1}"] = F.normalize(fusion(torch.cat([z[i], z[j]], -1)), dim=-1)
        return out


def confu_loss(out: dict[str, Tensor], higher_order: bool = True) -> Tensor:
    z1, z2, z3 = out["z1"], out["z2"], out["z3"]
    pairwise = symmetric_info_nce(z1, z2) + symmetric_info_nce(z1, z3) + symmetric_info_nce(z2, z3)
    if not higher_order:
        # CLIP-lineage control: pairwise alignment only, fusion heads exist
        # but receive no alignment signal (tests whether the gap is ConFu-specific)
        return pairwise / 3.0
    higher = (symmetric_info_nce(out["z12"], z3) + symmetric_info_nce(out["z13"], z2)
              + symmetric_info_nce(out["z23"], z1))
    return (pairwise + higher) / 6.0


def standardize(x_tr: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True) + 1e-6
    return (x_tr - mean) / std, (x - mean) / std


def linear_probe_acc(x_tr, y_tr, x_te, y_te, max_iter=400) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, tol=1e-3))
    probe.fit(x_tr, y_tr)
    return float(probe.score(x_te, y_te))


def predictability_r2(source_tr, target_tr, source_te, target_te, hidden=256, epochs=25,
                      seed=0, device="cuda") -> float:
    torch.manual_seed(seed)
    s_tr, s_te = standardize(source_tr, source_te)
    t_tr, t_te = standardize(target_tr, target_te)
    predictor = nn.Sequential(nn.Linear(s_tr.shape[1], hidden), nn.GELU(),
                              nn.Linear(hidden, t_tr.shape[1])).to(device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    xtr = torch.tensor(s_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(t_tr, dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(xtr), generator=gen).split(1024):
            loss = F.mse_loss(predictor(xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    predictor.eval()
    with torch.no_grad():
        mse = F.mse_loss(predictor(torch.tensor(s_te, dtype=torch.float32, device=device)),
                         torch.tensor(t_te, dtype=torch.float32, device=device)).item()
    return 1.0 - mse


def train_one(s1: float, seed: int, args, feats: dict[str, np.ndarray],
              labels: np.ndarray, tr_idx: np.ndarray, device: str) -> dict:
    torch.manual_seed(seed)
    train_tensors = [torch.tensor(feats[m][tr_idx], dtype=torch.float32, device=device)
                     for m in MODALITY]
    y_tr = labels[tr_idx]
    n_classes = int(labels.max() + 1)
    model = ConFuFrozen([feats[m].shape[1] for m in MODALITY], args.embed_dim, args.hidden,
                        n_classes if (args.lambda_utility > 0 or args.lambda_pair > 0) else 0).to(device)
    n_pairs = len(PAIRS)
    adversaries = nn.ModuleList()  # per pair: (from_i, from_j)
    standardizers = []
    for _ in PAIRS:
        adversaries.append(AdversarialPredictor(args.embed_dim, args.adv_hidden))
        adversaries.append(AdversarialPredictor(args.embed_dim, args.adv_hidden))
        standardizers.append(EMAStandardizer(args.embed_dim).to(device))
    adversaries = adversaries.to(device)
    opt_gen = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_adv = torch.optim.AdamW(adversaries.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)

    n = len(tr_idx)
    gen = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for idx in torch.randperm(n, generator=gen).split(args.batch_size):
            batch = [t[idx] for t in train_tensors]
            out = model(batch)
            loss = confu_loss(out, higher_order=not args.pairwise_only)
            if args.lambda_utility > 0 or args.lambda_pair > 0:
                yb = torch.tensor(np.asarray(y_tr)[idx.cpu().numpy()], dtype=torch.long, device=device) \
                    if not torch.is_tensor(y_tr) else y_tr[idx].to(device)
            if args.lambda_utility > 0:
                logits_base, logits_full = model.task_logits(out)
                ce_full = F.cross_entropy(logits_full, yb, reduction="none")
                ce_base = F.cross_entropy(logits_base, yb, reduction="none")
                task = ce_full.mean() + 0.5 * ce_base.mean()
                utility = F.relu(ce_full - ce_base.detach() + args.utility_margin).mean()
                loss = loss + args.lambda_task * task + args.lambda_utility * utility
            if args.lambda_pair > 0:
                pair_util = []
                pair_task = []
                for (i, j) in PAIRS:
                    key = f"{i+1}{j+1}"
                    lo_in = torch.cat([out[f"z{i+1}"], out[f"z{j+1}"]], -1)
                    fu_in = torch.cat([lo_in, out[f"z{i+1}{j+1}"]], -1)
                    ce_lo = F.cross_entropy(model.pair_heads[key]["lo"](lo_in), yb, reduction="none")
                    ce_fu = F.cross_entropy(model.pair_heads[key]["fu"](fu_in), yb, reduction="none")
                    pair_task.append(ce_fu.mean() + 0.5 * ce_lo.mean())
                    pair_util.append(F.relu(ce_fu - ce_lo.detach() + args.utility_margin).mean())
                loss = loss + args.lambda_task * torch.stack(pair_task).mean() \
                    + args.lambda_pair * torch.stack(pair_util).mean()
            if s1 > 0:
                targs = [standardizers[k].standardize(out[f"z{i+1}{j+1}"].detach())
                         for k, (i, j) in enumerate(PAIRS)]
                for k, std in enumerate(standardizers):
                    i, j = PAIRS[k]
                    std.update(out[f"z{i+1}{j+1}"])
                targs = [standardizers[k].standardize(out[f"z{PAIRS[k][0]+1}{PAIRS[k][1]+1}"].detach())
                         for k in range(n_pairs)]
                # adversary step
                for _ in range(args.k_adv):
                    loss_adv = torch.stack([
                        F.mse_loss(adversaries[2 * k](out[f"z{i+1}"].detach()), targs[k])
                        + F.mse_loss(adversaries[2 * k + 1](out[f"z{j+1}"].detach()), targs[k])
                        for k, (i, j) in enumerate(PAIRS)
                    ]).sum()
                    opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()
                # generator step
                with torch.no_grad():
                    preds = [(adversaries[2 * k](out[f"z{i+1}"]),
                              adversaries[2 * k + 1](out[f"z{j+1}"]))
                             for k, (i, j) in enumerate(PAIRS)]
                penalty = torch.stack([
                    shortcut_hinge_loss(preds[k][0], preds[k][1],
                                        standardizers[k].standardize(
                                            out[f"z{PAIRS[k][0]+1}{PAIRS[k][1]+1}"]),
                                        args.tau)[0]
                    for k in range(n_pairs)
                ]).sum()
                warmup = min(1.0, (epoch + 1) / max(1.0, args.adv_warmup_fraction * args.epochs))
                loss = loss + (s1 * warmup) * penalty
            opt_gen.zero_grad(); loss.backward(); opt_gen.step()
    duration = time.perf_counter() - started

    if args.save_model:
        os.makedirs(args.out, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.out, f"model_s1_{s1}_lu_{args.lambda_utility}_lp_{args.lambda_pair}_seed_{seed}.pt"))
    return evaluate(model, s1, seed, duration, args, feats, labels, tr_idx, device)


@torch.no_grad()
def embed_all(model: ConFuFrozen, feats: dict[str, np.ndarray], idx: np.ndarray,
              device: str, shuffle_modality: int | None = None) -> dict[str, np.ndarray]:
    tensors = [torch.tensor(feats[m][idx], dtype=torch.float32, device=device) for m in MODALITY]
    if shuffle_modality is not None:
        gen = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(len(idx), generator=gen).to(device)
        tensors[shuffle_modality] = tensors[shuffle_modality][perm]
    outs = []
    for start in range(0, len(idx), 2048):
        outs.append({k: v.cpu().numpy() for k, v in model([t[start:start + 2048] for t in tensors]).items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def evaluate(model, s1, seed, duration, args, feats, labels, tr_idx, device) -> dict:
    split = dict(np.load("data/bird_mml/split.npz"))
    te_idx = split["test"]
    rng = np.random.default_rng(seed)
    probe_tr = np.sort(rng.choice(tr_idx, size=min(args.n_probe, len(tr_idx)), replace=False))

    out_tr = embed_all(model, feats, probe_tr, device)
    out_te = embed_all(model, feats, te_idx, device)
    y_tr, y_te = labels[probe_tr], labels[te_idx]

    def lower(o): return np.concatenate([o["z1"], o["z2"], o["z3"]], 1)
    def full(o): return np.concatenate([o["z1"], o["z2"], o["z3"], o["z12"], o["z13"], o["z23"]], 1)

    result = {"s1": s1, "seed": seed, "train_seconds": round(duration, 1),
              "parameters_inference": sum(p.numel() for p in model.parameters())}
    # conditional utility (linear probes, frozen representations)
    result["probe_lower"] = linear_probe_acc(lower(out_tr), y_tr, lower(out_te), y_te)
    result["probe_full"] = linear_probe_acc(full(out_tr), y_tr, full(out_te), y_te)
    result["gain_full_over_lower"] = result["probe_full"] - result["probe_lower"]
    # pair-conditional utility: does z_ij add over its constituents?
    for (i, j) in PAIRS:
        lo_tr = np.concatenate([out_tr[f"z{i+1}"], out_tr[f"z{j+1}"]], 1)
        lo_te = np.concatenate([out_te[f"z{i+1}"], out_te[f"z{j+1}"]], 1)
        fu_tr = np.concatenate([lo_tr, out_tr[f"z{i+1}{j+1}"]], 1)
        fu_te = np.concatenate([lo_te, out_te[f"z{i+1}{j+1}"]], 1)
        acc_lo = linear_probe_acc(lo_tr, y_tr, lo_te, y_te)
        acc_fu = linear_probe_acc(fu_tr, y_tr, fu_te, y_te)
        result[f"pair_{i+1}{j+1}_probe_low"] = acc_lo
        result[f"pair_{i+1}{j+1}_probe_full"] = acc_fu
        result[f"pair_{i+1}{j+1}_gain"] = acc_fu - acc_lo
    # unimodal probes (modality dominance after training)
    for k, name in [("z1", "image"), ("z2", "audio"), ("z3", "text")]:
        result[f"probe_{name}"] = linear_probe_acc(out_tr[k], y_tr, out_te[k], y_te)
    # dependence: probe accuracy of full representation when a modality is corrupted
    for i, name in enumerate(MODALITY):
        out_te_sh = embed_all(model, feats, te_idx, device, shuffle_modality=i)
        result[f"shuffle_{name}_drop"] = result["probe_full"] - linear_probe_acc(
            full(out_tr), y_tr, full(out_te_sh), y_te)
    # shortcut meters: predictability of each fused pair from each constituent
    for (i, j) in PAIRS:
        key = f"z{i+1}{j+1}"
        result[f"R2_z{i+1}_to_{key}"] = predictability_r2(
            out_tr[f"z{i+1}"], out_tr[key], out_te[f"z{i+1}"], out_te[key])
        result[f"R2_z{j+1}_to_{key}"] = predictability_r2(
            out_tr[f"z{j+1}"], out_tr[key], out_te[f"z{j+1}"], out_te[key])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="data/bird_mml/embeddings.npz")
    parser.add_argument("--out", default="results/birds")
    parser.add_argument("--s1", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--k_adv", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--adv_hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adv_warmup_fraction", type=float, default=0.2)
    parser.add_argument("--n_probe", type=int, default=60000)
    parser.add_argument("--lambda_task", type=float, default=1.0)
    parser.add_argument("--lambda_utility", type=float, default=0.0)
    parser.add_argument("--utility_margin", type=float, default=0.0)
    parser.add_argument("--pairwise_only", action="store_true")
    parser.add_argument("--lambda_pair", type=float, default=0.0)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.emb, allow_pickle=True)
    feats = {"image": data["img"], "audio": data["aud"], "text": data["txt"]}
    labels = data["labels"]
    split = dict(np.load("data/bird_mml/split.npz"))
    tr_idx = split["train"]

    runs = []
    for seed in range(1, args.seeds + 1):
        metrics = train_one(args.s1, seed, args, feats, labels, tr_idx, device)
        runs.append(metrics)
        print(f"[s1={args.s1} lu={args.lambda_utility} seed={seed}] full={metrics['probe_full']*100:.2f}% "
              f"g13={metrics['pair_13_gain']*100:+.2f} "
              f"gain={metrics['gain_full_over_lower']*100:+.2f} "
              f"R2img={metrics['R2_z1_to_z12']:.3f} R2aud={metrics['R2_z2_to_z12']:.3f} "
              f"sh_img={metrics['shuffle_image_drop']*100:+.2f} sh_aud={metrics['shuffle_audio_drop']*100:+.2f} "
              f"sh_txt={metrics['shuffle_text_drop']*100:+.2f}", flush=True)
    import statistics
    summary = {}
    for k in runs[0]:
        if isinstance(runs[0][k], float):
            vals = [r[k] for r in runs]
            summary[k] = f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals) if len(vals) > 1 else 0:.4f}"
    tag = "pairwise" if args.pairwise_only else "confu"
    path = os.path.join(args.out, f"{tag}_frozen_s1_{args.s1}_lu_{args.lambda_utility}_lp_{args.lambda_pair}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "runs": runs, "summary": summary}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
