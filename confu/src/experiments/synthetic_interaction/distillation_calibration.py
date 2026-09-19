"""v4.3 B-branch: label-free D2 distillation-loss calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.identifiable_interaction import REGIMES, make_dataset
from src.experiments.synthetic_interaction.ipib import (
    MIN_JOINT_ADVANTAGE,
    _evaluate,
    _fit_predictors,
    _train_interaction,
)


VARIANTS = ("d2", "d2_mse", "d2_cos_mse_0.01", "d2_cos_mse_0.1")


def _validation_r2(features, target):
    best = -float("inf")
    for alpha in (1e-2, 1.0, 1e2, 1e4):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(features[0], target[0])
        best = max(best, float(r2_score(target[1], model.predict(features[1]), multioutput="variance_weighted")))
    return best


def _d2_targets(splits, predictions, identifiable):
    raw = [predictions["joint"][i] - predictions["additive"][i] for i in range(3)]
    return raw if identifiable else [np.zeros_like(value) for value in raw]


def run_seed(seed: int, output_dir: Path, variants=VARIANTS):
    records = []
    for regime in REGIMES:
        config = REGIMES[regime]
        splits = make_dataset(config, seed)
        predictions, predictor_metrics = _fit_predictors(splits, seed)
        identifiable = predictor_metrics["validation_joint_advantage"] > MIN_JOINT_ADVANTAGE
        target = _d2_targets(splits, predictions, identifiable)
        ground_truth = [split["joint_target"].numpy() for split in splits]
        candidates = {}
        for variant in variants:
            outputs, meta = _train_interaction(target, splits, seed + 22, variant)
            candidates[variant] = {
                "validation_target_r2": _validation_r2(outputs, target) if identifiable else 0.0,
                "training": meta,
                "test": _evaluate(variant, outputs, splits, target, ground_truth),
            }
        records.append(
            {
                "regime": regime,
                "seed": seed,
                "config": config.__dict__,
                "d2_target_identifiable_on_validation": identifiable,
                "predictors": predictor_metrics,
                "candidates": candidates,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"seed_{seed}.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_interaction/v43/calibration"))
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    args = parser.parse_args()
    all_records = []
    variants = (args.variant,) if args.variant else VARIANTS
    for seed in range(1, args.seeds + 1):
        records = run_seed(seed, args.out, variants)
        all_records.extend(records)
        print(json.dumps({"seed": seed, "regimes": len(records)}))
    (args.out / "summary.json").write_text(json.dumps(all_records, indent=2) + "\n")


if __name__ == "__main__":
    main()
