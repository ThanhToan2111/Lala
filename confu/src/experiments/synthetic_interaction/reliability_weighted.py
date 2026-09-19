"""C1 reliability-weighted joint-advantage distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.identifiable_interaction import REGIMES, make_dataset
from src.experiments.synthetic_interaction.ipib import (
    MIN_JOINT_ADVANTAGE,
    _evaluate,
    _fit_predictors,
    _train_interaction,
)
from src.experiments.synthetic_interaction.reliability_audit import reliability_score


def _validation_accuracy(features, splits):
    labels = [split["labels"].numpy() for split in splits]
    base = [np.concatenate((split["x1"].numpy(), split["x2"].numpy()), axis=1) for split in splits]
    augmented = [np.concatenate((base[i], features[i]), axis=1) for i in range(3)]

    def score(values):
        best = -float("inf")
        for c in (1e-4, 1e-2, 1.0, 1e2, 1e4):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=0))
            model.fit(values[0], labels[0])
            best = max(best, float(model.score(values[1], labels[1])))
        return best

    return {
        "base_accuracy": score(base),
        "augmented_accuracy": score(augmented),
        "delta12_linear": score(augmented) - score(base),
    }


def run_seed(seed: int, output_dir: Path):
    records = []
    for regime, config in REGIMES.items():
        splits = make_dataset(config, seed)
        predictions, predictor_metrics = _fit_predictors(splits, seed)
        _, _, reliability = reliability_score(
            predictions["additive"][1],
            predictions["joint"][1],
            splits[1]["x3"].numpy(),
        )
        identifiable = predictor_metrics["validation_joint_advantage"] > MIN_JOINT_ADVANTAGE
        raw_target = [
            predictions["joint"][i] - predictions["additive"][i] for i in range(3)
        ]
        if identifiable:
            target = [value * (reliability / (reliability.mean() + 1e-8)) for value in raw_target]
        else:
            target = [np.zeros_like(value) for value in raw_target]
        ground_truth = [split["joint_target"].numpy() for split in splits]
        c1_outputs, c1_training = _train_interaction(target, splits, seed + 22, "d2")
        d2_outputs, d2_training = _train_interaction(
            raw_target if identifiable else [np.zeros_like(value) for value in raw_target],
            splits,
            seed + 22,
            "d2",
        )
        records.append(
            {
                "regime": regime,
                "seed": seed,
                "config": config.__dict__,
                "alpha": 1.0,
                "d2_target_identifiable_on_validation": identifiable,
                "predictors": predictor_metrics,
                "reliability_distribution": {
                    "mean": float(reliability.mean()),
                    "min": float(reliability.min()),
                    "max": float(reliability.max()),
                    "fraction_zero": float(np.mean(reliability <= 1e-12)),
                },
                "validation": {
                    "d2": _validation_accuracy(d2_outputs, splits),
                    "c1": _validation_accuracy(c1_outputs, splits),
                },
                "test": {
                    "d2": _evaluate("d2", d2_outputs, splits, raw_target, ground_truth),
                    "c1": _evaluate("c1", c1_outputs, splits, target, ground_truth),
                },
                "training": {"d2": d2_training, "c1": c1_training},
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"seed_{seed}.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_interaction/v44/c1_seed1"))
    args = parser.parse_args()
    all_records = []
    for seed in range(1, args.seeds + 1):
        all_records.extend(run_seed(seed, args.out))
        print(json.dumps({"seed": seed, "regimes": len(REGIMES)}))
    (args.out / "summary.json").write_text(json.dumps(all_records, indent=2) + "\n")


if __name__ == "__main__":
    main()
