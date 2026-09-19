"""Plot fidelity/accessibility using Naturalized IPIB or existing IPIB records."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    root = Path("results/naturalized_ipib/v60")
    payload = json.loads((root / "accessibility.json").read_text())
    by_beta = {}
    for row in payload["records"]:
        by_beta.setdefault(row["beta"], []).append(row["delta_access"])
    betas = sorted(by_beta)
    values = [sum(by_beta[b]) / len(by_beta[b]) for b in betas]
    out = Path("results/figures/v60")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(betas, values, marker="o")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Injected interaction strength β")
    ax.set_ylabel("Accessibility Δ")
    ax.set_title("Fidelity–accessibility diagnostic")
    fig.tight_layout()
    fig.savefig(out / "fig_fidelity_accessibility.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_fidelity_accessibility.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
