"""Fig. 2 — ConFu++ method schematic (matplotlib, PDF)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs("results/figures", exist_ok=True)
plt.rcParams.update({"font.size": 8.5})

fig, ax = plt.subplots(figsize=(7.0, 3.1))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis("off")

def box(x, y, w, h, text, fc="#eef3ee", ec="#333333", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2, style="-", color="#333", lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, color=color, lw=lw, linestyle=style))

# encoders
for i, (name, c) in enumerate([("image", "#dcebf7"), ("audio", "#f7e8dc"), ("text", "#e8f0dc")]):
    box(0.2, 4.4 - i * 1.5, 1.9, 1.0, f"$X_{i+1}$\n{name}", fc=c)
    box(2.6, 4.4 - i * 1.5, 1.9, 1.0, f"$E_{i+1}$\n$r_{i+1}$", fc=c)
    arrow(2.1, 4.9 - i * 1.5, 2.6, 4.9 - i * 1.5)

# fusion
box(5.3, 2.2, 2.6, 1.8, "Fusion\n$r_{ij}=F_{ij}(r_i,r_j)$\n(low-rank multiplicative)",
    fc="#efe6f5", fs=8)
for i in range(3):
    arrow(4.5, 4.9 - i * 1.5, 5.3, 3.6 - i * 0.4)

# composition + task
box(8.6, 3.4, 2.3, 1.2, "Composition\n$c_{base}, c_{full}$\n(gated)", fc="#f5f0e6")
box(8.6, 1.2, 2.3, 1.2, "Task heads\n$CE_{base}, CE_{full}$", fc="#f5f0e6")
arrow(7.9, 3.4, 8.6, 4.0)
arrow(7.9, 2.8, 8.6, 1.8)

# utility loss
box(11.4, 3.4, 2.4, 1.2, "Utility loss\n$\\mathrm{ReLU}(CE_{full}-\\mathrm{sg}(CE_{base}))$",
    fc="#fdeaea", fs=8)
arrow(10.9, 4.0, 11.4, 4.0)
arrow(10.9, 1.8, 12.6, 3.4, style="--", color="#888")

# S1 adversaries
box(11.4, 0.5, 2.4, 1.6, "S1 adversaries\n$q_i(r_i)\\!\\to\\!\\mathrm{std}(r_{ij})$\nhinge $\\mathrm{ReLU}(\\tau-\\mathrm{MSE})$",
    fc="#fdeaea", fs=8)
arrow(6.6, 2.2, 11.6, 1.7, style="--", color="#b23a48")
arrow(12.6, 2.1, 12.6, 3.4, color="#b23a48")   # penalty back up to utility/loss
ax.text(12.85, 2.7, "penalty", fontsize=7.5, color="#b23a48", rotation=90, va="center")

# diagnostics
box(5.3, 0.3, 2.6, 1.4, "Diagnostics\nshuffle / zero / probes\npredictability $R^2$", fc="#e8e8e8", fs=8)
arrow(6.6, 2.2, 6.6, 1.7, style="--", color="#555")

fig.savefig("results/figures/fig2_pipeline.pdf", bbox_inches="tight")
print("saved results/figures/fig2_pipeline.pdf")
