"""Generate the conceptual ConFu++ diagnostic pipeline figure."""
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    out = Path("results/figures/v60")
    out.mkdir(parents=True, exist_ok=True)
    labels = ["Raw modalities", "Frozen representations", "Additive vs joint\nprediction", "Cross-modal\nidentifiability", "Task joint\nheadroom", "JAD", "Fidelity", "Accessibility"]
    fig, ax = plt.subplots(figsize=(13, 2.4))
    ax.axis("off")
    for i, label in enumerate(labels):
        ax.text(i, 0.5, label, ha="center", va="center", fontsize=10, bbox={"boxstyle": "round,pad=0.6", "facecolor": "#e8f1fb" if i < 3 else "#f7efe0", "edgecolor": "#345"})
        if i + 1 < len(labels):
            ax.annotate("", xy=(i + 0.68, 0.5), xytext=(i + 0.32, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "fig_framework.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_framework.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
