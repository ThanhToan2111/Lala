"""Plot natural cross-modal J against task-headroom summaries when available."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _g1_mean(path, mapping):
    payload = json.loads(Path(path).read_text())
    return payload["g1"][mapping]["summary"]["valid"]["joint_advantage"]["mean"]


def _mosei_mean(path):
    payload = json.loads(Path(path).read_text())
    values = [record["metrics"]["valid"]["joint_advantage"] for record in payload["records"] if record["mapping"] == "VA_to_T"]
    return sum(values) / len(values)


def _mosei_task_mean(path):
    payload = json.loads(Path(path).read_text())
    values = [record["models"]["joint_product"]["valid"]["accuracy"] - record["models"]["additive"]["valid"]["accuracy"] for record in payload["records"]]
    return sum(values) / len(values)


def main():
    out = Path("results/figures/v60")
    out.mkdir(parents=True, exist_ok=True)
    root = Path("results")
    points = [
        ("MOSEI", _mosei_mean(root / "mosei/identifiability_exact/mosei_identifiability.json"), _mosei_task_mean(root / "mosei/v53/t0_task_headroom.json")),
        ("MUStARD", _g1_mean(root / "mustard/v54/mustard_two_gate.json", "VA_to_T"), None),
        ("MELD", _g1_mean(root / "meld/v55/meld_two_gate.json", "VA_to_T"), None),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, j, h in points:
        ax.scatter(j, 0 if h is None else h, s=70, label=name)
        ax.annotate(name, (j, 0 if h is None else h), xytext=(5, 5), textcoords="offset points")
    ax.axvline(0.01, color="black", ls="--", lw=1, label="historical J gate")
    ax.set_xlabel("Cross-modal predictive advantage J")
    ax.set_ylabel("Task joint headroom H (reported when available)")
    ax.set_title("Natural-regime screening; missing H is not imputed")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_two_gate.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_two_gate.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
