"""Generate paper figures from the repository's real result artifacts.

Outputs PDFs to results/figures/:
    fig1_avmnist_paradox      - healthy surface metrics vs image-derived content
    fig3_predictability       - single-modality predictability of fused interactions
                                across benchmarks (AV-MNIST / MOSI / UR-FUNNY)
    fig4_order3               - order-3 benchmark: accuracy by model family +
                                probe decomposition + dependence drops
    fig5_ladder_adversary_gap - M-modality ladder (a) and S1 adversary gap (b)

All numbers come from committed result JSONs / EXPERIMENTS.md entries.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 200, "savefig.bbox": "tight",
})


def load_runs(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def mean_std(vals):
    vals = [float(v) for v in vals]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


# ---------------- Fig 1: the AV-MNIST paradox ----------------
def fig1():
    ab = load_runs("results/av_mnist/avmnist_s1_w000_official/synergy_s1_seed_*.json")
    if not ab:
        print("fig1 skipped: missing ablation runs"); return
    # surface metrics (from the 5-seed ablation runs)
    r12_var = mean_std([r["metrics"]["representations"]["r12_variance"] for r in ab])
    r12_rank = mean_std([r["metrics"]["representations"]["r12_effective_rank"] for r in ab])
    shuffle = mean_std([r["shuffle_drop12"] * 100 for r in ab])
    pred1 = mean_std([r["predictability"]["r1"]["r2"] for r in ab])
    pred2 = mean_std([r["predictability"]["r2"]["r2"] for r in ab])

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ax = axes[0]
    names = ["variance\n(scaled)", "effective\nrank /10", "shuffle\ndrop (pp)"]
    vals = [r12_var[0] * 100, r12_rank[0] / 10, shuffle[0]]
    errs = [r12_var[1] * 100, r12_rank[1] / 10, shuffle[1]]
    ax.bar(names, vals, yerr=errs, color="#4c9254", capsize=3)
    ax.set_title("Surface health: looks fine")
    ax.set_ylabel("value")
    ax = axes[1]
    names = ["R²(image→r12)", "R²(audio→r12)"]
    vals = [pred1[0], pred2[0]]
    errs = [pred1[1], pred2[1]]
    bars = ax.bar(names, vals, yerr=errs, color=["#b23a48", "#4c9254"], capsize=3)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_title("True content: a unimodal function")
    ax.set_ylabel("predictability R²")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center")
    fig.savefig(f"{OUT}/fig1_avmnist_paradox.pdf")
    plt.close(fig)
    print("fig1 saved")


# ---------------- Fig 3: predictability across natural benchmarks ----------------
def fig3():
    mosi = {"r_va": 0.915, "r_vt": 0.953}   # EXPERIMENTS.md: CMU-MOSI audit, R² from lower order
    urfunny = {"r_va": 0.963, "r_vt": 0.970, "r_at": 0.970}
    ab = load_runs("results/av_mnist/avmnist_s1_w000_official/synergy_s1_seed_*.json")
    s1 = load_runs("results/av_mnist/avmnist_s1_w010_official/synergy_s1_seed_*.json")
    if ab:
        av0 = mean_std([r["predictability"]["r1"]["r2"] for r in ab])
        av1 = mean_std([r["predictability"]["r1"]["r2"] for r in s1]) if s1 else (np.nan, 0)
    else:
        av0 = (0.989, 0.002); av1 = (0.682, 0.155)

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    labels = ["AV-MNIST\nr1→r12 (base)", "AV-MNIST\nr1→r12 (+S1)", "MOSI\nlow→r_vt", "MOSI\nlow→r_va",
              "UR-FUNNY\nlow→r_vt", "UR-FUNNY\nlow→r_va", "UR-FUNNY\nlow→r_at"]
    vals = [av0[0], av1[0], mosi["r_vt"], mosi["r_va"], urfunny["r_vt"], urfunny["r_va"], urfunny["r_at"]]
    errs = [av0[1], av1[1], 0, 0, 0, 0, 0]
    colors = ["#b23a48", "#e4852e"] + ["#7a6aa0"] * 2 + ["#5a87b0"] * 3
    bars = ax.bar(labels, vals, yerr=errs, color=colors, capsize=3)
    ax.set_ylabel("R² of fused from constituent(s)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="red", ls="--", lw=0.8)
    ax.text(4.6, 0.91, "shortcut zone", color="red", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.savefig(f"{OUT}/fig3_predictability.pdf")
    plt.close(fig)
    print("fig3 saved")


# ---------------- Fig 4: order-3 benchmark ----------------
def fig4():
    def acc_of(path):
        d = json.load(open(path))
        vals = [r["accuracy"] for r in d["runs"]]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)
    try:
        additive = acc_of("results/synthetic/order3/order3_additive_s1_0.0.json")
        pairs = acc_of("results/synthetic/order3/order3_pairs_only_s1_0.0.json")
        concat = acc_of("results/synthetic/order3/order3_concat_mlp_s1_0.0.json")
        full = json.load(open("results/synthetic/order3/order3_full_s1_0.0.json"))
    except FileNotFoundError as exc:
        print("fig4 skipped:", exc); return
    full_acc = acc_of("results/synthetic/order3/order3_full_s1_0.0.json")
    runs = full["runs"]
    probes1 = statistics.mean(r["probe_r1"] for r in runs)
    probesp = statistics.mean(r["probe_r12"] for r in runs)
    probe123 = statistics.mean(r["probe_r123"] for r in runs)
    drops = [statistics.mean(r[f"shuffle_r{i}_drop"] for r in runs) for i in (1, 2, 3)]

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4))
    ax = axes[0]
    names = ["additive\n(order-1)", "pairs\n(order-1+2)", "concat\nMLP", "full\n(order 1+2+3)"]
    vals = [additive[0], pairs[0], concat[0], full_acc[0]]
    errs = [additive[1], pairs[1], concat[1], full_acc[1]]
    ax.bar(names, vals, yerr=errs, color=["#999999", "#999999", "#5a87b0", "#4c9254"], capsize=3)
    ax.axhline(0.5, color="red", ls="--", lw=0.8)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("y = a⊕b⊕c")
    ax = axes[1]
    ax.bar(["single\nr1", "pair\nr12", "triple\nr123"], [probes1, probesp, probe123],
           color=["#999999", "#999999", "#4c9254"])
    ax.axhline(0.5, color="red", ls="--", lw=0.8)
    ax.set_ylim(0.4, 1.05)
    ax.set_title("held-out probes (full model)")
    ax.set_ylabel("probe accuracy → y")
    ax = axes[2]
    ax.bar(["shuffle\nmod 1", "shuffle\nmod 2", "shuffle\nmod 3"], drops, color="#4c9254")
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("accuracy drop")
    ax.set_title("dependence balance")
    fig.savefig(f"{OUT}/fig4_order3.pdf")
    plt.close(fig)
    print("fig4 saved")


# ---------------- Fig 5: ladder + adversary gap ----------------
def fig5():
    try:
        ladder = json.load(open("results/synthetic/orderM/orderM_ladder.json"))
    except FileNotFoundError as exc:
        print("fig5a skipped:", exc)
        ladder = None
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    if ladder:
        ax = axes[0]
        runs = ladder["runs"]
        for variant, color, marker in [("additive", "#999999", "o"), ("pairs_only", "#777777", "s"),
                                       ("full", "#4c9254", "D"), ("concat_mlp", "#5a87b0", "^")]:
            Ms, accs, errs = [], [], []
            for M in (3, 4, 5, 6):
                vals = [r["accuracy"] for r in runs if r["M"] == M and r["variant"] == variant]
                Ms.append(M)
                accs.append(statistics.mean(vals))
                errs.append(statistics.stdev(vals) if len(vals) > 1 else 0)
            ax.errorbar(Ms, accs, yerr=errs, label=variant, color=color, marker=marker, ms=3, lw=1.2, capsize=2)
        ax.axhline(0.5, color="red", ls="--", lw=0.8)
        ax.set_xlabel("M modalities (y = XOR of M bits)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.45, 1.03)
        ax.legend(frameon=False, ncol=2)
        ax.set_title("(a) order ladder")
    # adversary gap dynamics from the long run CSV
    path = "outputs/bimodal_avmnist/avmnist_s1_long/synergy_s1/seed_1/metrics.csv"
    ax = axes[1]
    if os.path.exists(path):
        rows = list(csv.DictReader(open(path)))
        data = {}
        for r in rows:
            d = data.setdefault(r["epoch"], {})
            for k, v in r.items():
                if v not in (None, "") and k not in ("epoch", "step"):
                    d[k] = v
        eps = sorted(data, key=int)
        adv1 = [float(data[e]["adv/r2_1"]) for e in eps if data[e].get("adv/r2_1")]
        xs = [int(e) for e in eps if data[e].get("adv/r2_1")]
        ax.plot(xs, adv1, color="#b23a48", lw=1.2, label="online adversary R²(r1→r12)")
        ax.axhline(0.84, color="#7a6aa0", ls="--", lw=1.0, label="offline converged probe (0.84)")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="hinge band edge (τ=0.5)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False)
        ax.set_title("(b) S1 adversary gap")
    fig.savefig(f"{OUT}/fig5_ladder_adversary_gap.pdf")
    plt.close(fig)
    print("fig5 saved")


if __name__ == "__main__":
    fig1()
    fig3()
    fig4()
    fig5()
    print("all figures in", OUT)
