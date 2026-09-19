"""Plot the v6.0 Naturalized IPIB beta curve from its JSON artifact."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    root = Path("results/naturalized_ipib/v60")
    payload = json.loads((root / "predictor_screen.json").read_text())
    grouped = {}
    for record in payload["records"]:
        grouped.setdefault(record["beta"], []).append(record["predictor"]["valid"]["J"])
    betas = sorted(grouped)
    means = [sum(grouped[b]) / len(grouped[b]) for b in betas]
    stds = [(sum((x - means[i]) ** 2 for x in grouped[b]) / max(len(grouped[b]) - 1, 1)) ** 0.5 for i, b in enumerate(betas)]
    out = Path("results/figures/v60")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.errorbar(betas, means, yerr=stds, marker="o", capsize=3)
    ax.axhline(0.01, color="black", ls="--", lw=1, label="historical gate")
    ax.set_xlabel("Injected interaction strength β")
    ax.set_ylabel("Validation joint advantage J")
    ax.set_title("Naturalized IPIB")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_naturalized_ipib.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_naturalized_ipib.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
