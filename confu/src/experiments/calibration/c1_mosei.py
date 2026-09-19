"""v6.1-C1: preregistered MOSEI alignment-breaking null confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.experiments.calibration.interaction_null import (
    _fit_pair,
    _natural_splits,
    _shuffle_right,
)
from src.experiments.hypothesis_class.h0_audit import _load_data, _standardize
from src.experiments.multibench.mosei_identifiability import MAPPINGS


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "per_replicate": values.tolist(),
    }


def _observed(root: Path) -> dict:
    path = root / "results/mosei/identifiability_exact/mosei_identifiability.json"
    payload = json.loads(path.read_text())
    values = [record["metrics"]["valid"]["joint_advantage"] for record in payload["records"] if record["mapping"] == "VA_to_T"]
    return {"source": str(path), "per_seed": values, "mean": float(np.mean(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/calibration/v61"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--mosei-data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--mosei-cache", type=Path, default=Path("results/mosei/identifiability_exact/mosei_pooled.npz"))
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if args.replicates < 50:
        raise ValueError("C1 requires at least B=50")
    args.out.mkdir(parents=True, exist_ok=True)
    raw, metadata = _load_data(args.mosei_data_dir, args.mosei_cache)
    splits = _natural_splits(_standardize(raw), MAPPINGS["VA_to_T"])
    observed = _observed(args.project_root)
    values = []
    capacity = None
    for replicate in range(args.replicates):
        result = _fit_pair(_shuffle_right(splits, replicate), 1000 + replicate, args)
        capacity = result["capacity"]
        values.append(result["scores"]["valid"]["J"])
        if (replicate + 1) % 5 == 0:
            print(json.dumps({"replicate": replicate + 1, "B": args.replicates, "last_J": values[-1]}))
    summary = _summary(values)
    threshold = 0.01
    observed_j = observed["mean"]
    payload = {
        "experiment_id": "v61-C1",
        "status": "complete",
        "protocol": {
            "dataset": "MOSEI",
            "mapping": "VA_to_T",
            "null": "independently shuffle right/source modality within each split",
            "replicates": args.replicates,
            "selection_split": "validation",
            "test_used_for_selection": False,
            "historical_threshold": threshold,
            "training": {"batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience, "lr": args.lr, "weight_decay": args.weight_decay},
        },
        "source_metadata": metadata,
        "capacity": capacity,
        "observed": observed,
        "null": summary,
        "empirical_exceedance": {
            "threshold_0.01": (1 + sum(value >= threshold for value in values)) / (args.replicates + 1),
            "observed_J": (1 + sum(value >= observed_j for value in values)) / (args.replicates + 1),
        },
    }
    (args.out / "C1_MOSEI_NULL.json").write_text(json.dumps(payload, indent=2) + "\n")
    exceed_tau = payload["empirical_exceedance"]["threshold_0.01"]
    exceed_obs = payload["empirical_exceedance"]["observed_J"]
    interpretation = "operational positive and separated from this alignment-breaking null" if exceed_obs <= 0.05 else "operational positive under the frozen predictor comparison but not separated from this alignment-breaking null"
    report = [
        "# v6.1-C1 MOSEI calibration confirmation",
        "",
        "This is a calibration-only extension. No model, representation, historical threshold, or prior label was changed.",
        "",
        f"Observed frozen validation J mean: `{observed_j:.8f}` from `{observed['per_seed']}`.",
        f"Null: independently shuffled source modality within each split; B={args.replicates}.",
        "",
        "## Null summary",
        "",
        "| Mean | Std | Median | q90 | q95 | q99 | Max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['mean']:.5f} | {summary['std']:.5f} | {summary['median']:.5f} | {summary['q90']:.5f} | {summary['q95']:.5f} | {summary['q99']:.5f} | {summary['max']:.5f} |",
        "",
        "## Empirical exceedance probabilities",
        "",
        f"- `p_hat(J >= 0.01) = {exceed_tau:.5f}` with +1 correction.",
        f"- `p_hat(J >= observed_J) = {exceed_obs:.5f}` with +1 correction.",
        "",
        "## Decision-independent interpretation",
        "",
        f"MOSEI is **{interpretation}**.",
        "The historical `tau=0.01` remains an operational historical threshold, not a universal discovery threshold. This null preserves the v6.0 alignment-breaking procedure and does not establish PID, causality, or universal statistical significance.",
    ]
    (args.out / "C1_MOSEI_NULL_REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
