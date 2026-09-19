"""Paper figure: direct and JAD recovery of the known Naturalized IPIB interaction."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    root = Path("results/naturalized_ipib/v60")
    rows = json.loads((root / "jad_recovery.json").read_text())["records"]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["beta"], {"direct": [], "jad": []})
        if row["d_to_ground_truth_r2"] is not None:
            grouped[row["beta"]]["direct"].append(row["d_to_ground_truth_r2"])
        if row["h_to_ground_truth_r2"] is not None:
            grouped[row["beta"]]["jad"].append(row["h_to_ground_truth_r2"])
    betas = sorted(grouped)
    out = Path("results/figures/v60")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for key, label, marker in (("direct", "direct d → βj", "o"), ("jad", "JAD h → βj", "s")):
        means = [sum(grouped[b][key]) / len(grouped[b][key]) if grouped[b][key] else float("nan") for b in betas]
        stds = [((sum((x - means[i]) ** 2 for x in grouped[b][key]) / max(len(grouped[b][key]) - 1, 1)) ** 0.5) if len(grouped[b][key]) > 1 else 0.0 for i, b in enumerate(betas)]
        ax.errorbar(betas, means, yerr=stds, marker=marker, capsize=3, label=label)
    ax.set_xlabel("Injected interaction strength β")
    ax.set_ylabel("Ground-truth recovery R²")
    ax.set_title("Naturalized IPIB recovery")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_naturalized_recovery.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_naturalized_recovery.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
