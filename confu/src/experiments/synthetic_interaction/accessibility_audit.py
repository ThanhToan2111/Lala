"""A0 target-versus-embedding accessibility audit for frozen v4.2 D0/D1/D2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.datasets.identifiable_interaction import REGIMES, make_dataset
from src.experiments.synthetic_interaction.ipib import (
    MIN_JOINT_ADVANTAGE,
    _component_diagnostics,
    _fit_predictors,
    _health,
    _linear_classification,
    _ridge_recovery,
    _train_interaction,
)


def _arrays(splits, key):
    return [split[key].numpy() for split in splits]


def _concat(left, right):
    return [np.concatenate((a, b), axis=1) for a, b in zip(left, right)]


def _feature_metrics(name, features, base, labels, g, g_tilde, task_score):
    base_accuracy = _linear_classification(base, labels)
    accuracy = _linear_classification(features, labels)
    augmented_accuracy = _linear_classification(_concat(base, features), labels)
    return {
        "feature": name,
        "r2_to_g12": _ridge_recovery(features, g),
        "r2_to_projected_joint": _ridge_recovery(features, g_tilde),
        "r2_to_task_score": _ridge_recovery(features, task_score),
        "linear_label_accuracy": accuracy,
        "base_linear_label_accuracy": base_accuracy,
        "augmented_linear_label_accuracy": augmented_accuracy,
        "delta12_linear": augmented_accuracy - base_accuracy,
        "health": _health(features[2]),
    }


def _retention(target, embedding, base, labels, task_score):
    target_acc = _linear_classification(target, labels)
    embedding_acc = _linear_classification(embedding, labels)
    base_acc = _linear_classification(base, labels)
    denominator = target_acc - base_acc
    target_score_r2 = _ridge_recovery(target, task_score)
    embedding_score_r2 = _ridge_recovery(embedding, task_score)
    base_score_r2 = _ridge_recovery(base, task_score)
    score_denominator = target_score_r2 - base_score_r2
    return {
        "target_label_accuracy": target_acc,
        "embedding_label_accuracy": embedding_acc,
        "lower_label_accuracy": base_acc,
        "label_accuracy_retention": None if denominator <= 1e-8 else (embedding_acc - base_acc) / denominator,
        "target_task_score_r2": target_score_r2,
        "embedding_task_score_r2": embedding_score_r2,
        "lower_task_score_r2": base_score_r2,
        "task_score_r2_retention": (
            None if score_denominator <= 1e-8 else (embedding_score_r2 - base_score_r2) / score_denominator
        ),
    }


def _predictor_diagnostics(splits, predictions, target):
    additive = _arrays(splits, "additive")
    joint = _arrays(splits, "joint")
    x3 = _arrays(splits, "x3")
    q_add, q_joint = predictions["additive"], predictions["joint"]
    error_add = [x - q for x, q in zip(x3, q_add)]
    error_joint = [x - q for x, q in zip(x3, q_joint)]
    d = [j - a for j, a in zip(q_joint, q_add)]
    centered_add = error_add[2] - error_add[2].mean(0)
    centered_joint = error_joint[2] - error_joint[2].mean(0)
    covariance = float(np.mean(centered_add * centered_joint))
    denominator = float(np.var(d[2] - target[2]))
    signal = float(np.var(target[2]))
    return {
        "q_add_to_additive_r2": _ridge_recovery(q_add, additive),
        "q_joint_to_additive_r2": _ridge_recovery(q_joint, additive),
        "q_joint_to_joint_r2": _ridge_recovery(q_joint, joint),
        "d2_to_joint_r2": _ridge_recovery(d, joint),
        "additive_joint_error_covariance_mean_diag": covariance,
        "d2_signal_variance": signal,
        "d2_error_variance": denominator,
        "d2_signal_to_error_ratio": None if denominator <= 1e-12 else signal / denominator,
    }


def audit(regime: str, seed: int, canonical_dir: Path | None = None):
    config = REGIMES[regime]
    splits = make_dataset(config, seed)
    predictions, predictor_metrics = _fit_predictors(splits, seed)
    raw_d2 = [predictions["joint"][i] - predictions["additive"][i] for i in range(3)]
    identifiable = predictor_metrics["validation_joint_advantage"] > MIN_JOINT_ADVANTAGE
    targets = {
        "d0_target": _arrays(splits, "x3"),
        "d1_target": [split["x3"].numpy() - predictions["additive"][i] for i, split in enumerate(splits)],
        "d2_target": raw_d2 if identifiable else [np.zeros_like(value) for value in raw_d2],
    }
    labels = _arrays(splits, "labels")
    base = [np.concatenate((split["x1"].numpy(), split["x2"].numpy()), axis=1) for split in splits]
    g = _arrays(splits, "g12")
    g_tilde = _arrays(splits, "joint")
    task_score = _arrays(splits, "task_score")
    embeddings = {}
    training = {}
    for offset, method in enumerate(("d0", "d1", "d2"), start=20):
        outputs, meta = _train_interaction(targets[f"{method}_target"], splits, seed + offset, method)
        embeddings[f"{method}_embedding"] = outputs
        training[method] = meta
    features = {"g": g, "g_tilde": g_tilde, **targets, **embeddings}
    metrics = {
        name: _feature_metrics(name, values, base, labels, g, g_tilde, task_score)
        for name, values in features.items()
    }
    retention = {
        method: _retention(
            targets[f"{method}_target"],
            embeddings[f"{method}_embedding"],
            base,
            labels,
            task_score,
        )
        for method in ("d0", "d1", "d2")
    }
    result = {
        "regime": regime,
        "seed": seed,
        "config": config.__dict__,
        "d2_target_identifiable_on_validation": identifiable,
        "predictors": predictor_metrics,
        "predictor_error_diagnostics": _predictor_diagnostics(
            splits, predictions, _arrays(splits, "joint_target")
        ),
        "component_diagnostics": _component_diagnostics(splits[2]),
        "features": metrics,
        "retention": retention,
        "training": training,
    }
    if canonical_dir is not None:
        path = canonical_dir / f"{regime}_seed_{seed}.json"
        if path.exists():
            result["canonical_v42"] = json.loads(path.read_text())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_interaction/v43"))
    parser.add_argument(
        "--canonical-dir", type=Path, default=Path("results/synthetic_interaction/confirm_seeded")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for regime in tuple(REGIMES):
        for seed in range(1, args.seeds + 1):
            record = audit(regime, seed, args.canonical_dir)
            records.append(record)
            (args.out / f"{regime}_seed_{seed}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({"regime": regime, "seed": seed, "J": record["predictors"]["joint_advantage"]}))
    (args.out / "a0_accessibility_audit.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
