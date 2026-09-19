"""C0 label-free predictor reliability audit for IPIB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.datasets.identifiable_interaction import REGIMES, make_dataset
from src.experiments.synthetic_interaction.ipib import _fit_predictors


def reliability_score(additive, joint, target):
    additive_error = np.mean((additive - target) ** 2, axis=0)
    joint_error = np.mean((joint - target) ** 2, axis=0)
    reliability = np.maximum(0.0, additive_error - joint_error) / (additive_error + 1e-8)
    return additive_error, joint_error, reliability


def _safe_correlation(left, right, method):
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(left, right).statistic)


def _quartile_metrics(reliability, fidelity):
    order = np.argsort(reliability)
    q = max(1, len(order) // 4)
    bottom, top = order[:q], order[-q:]
    middle = order[q:-q] if len(order) > 2 * q else order
    return {
        "bottom25_fidelity": float(np.mean(fidelity[bottom])),
        "middle50_fidelity": float(np.mean(fidelity[middle])),
        "top25_fidelity": float(np.mean(fidelity[top])),
        "top_bottom_fidelity_gap": float(np.mean(fidelity[top]) - np.mean(fidelity[bottom])),
        "bottom25_size": int(len(bottom)),
        "middle50_size": int(len(middle)),
        "top25_size": int(len(top)),
    }


def audit(regime: str, seed: int, canonical_dir: Path | None = None):
    config = REGIMES[regime]
    splits = make_dataset(config, seed)
    predictions, predictor_metrics = _fit_predictors(splits, seed)
    validation_target = splits[1]["x3"].numpy()
    additive_error, joint_error, reliability = reliability_score(
        predictions["additive"][1], predictions["joint"][1], validation_target
    )
    d = predictions["joint"][2] - predictions["additive"][2]
    joint_component = splits[2]["joint_target"].numpy()
    projected_joint = splits[2]["joint"].numpy()
    fidelity = 1 - np.mean((d - joint_component) ** 2, axis=0) / (np.var(joint_component, axis=0) + 1e-8)
    correlation = np.array(
        [
            np.corrcoef(d[:, index], joint_component[:, index])[0, 1]
            if np.std(d[:, index]) > 1e-12 and np.std(joint_component[:, index]) > 1e-12
            else 0.0
            for index in range(d.shape[1])
        ]
    )
    beta = config.beta
    scaled_fidelity = None
    if beta > 0:
        d_scaled = d / beta
        scaled_fidelity = 1 - np.mean((d_scaled - projected_joint) ** 2, axis=0) / (
            np.var(projected_joint, axis=0) + 1e-8
        )
    dimensions = [
        {
            "dimension": int(index),
            "E_add": float(additive_error[index]),
            "E_joint": float(joint_error[index]),
            "R": float(reliability[index]),
            "oracle_fidelity": float(fidelity[index]),
            "oracle_correlation": float(correlation[index]),
            "oracle_fidelity_scale_corrected": (
                None if scaled_fidelity is None else float(scaled_fidelity[index])
            ),
        }
        for index in range(d.shape[1])
    ]
    result = {
        "regime": regime,
        "seed": seed,
        "config": config.__dict__,
        "predictors": predictor_metrics,
        "dimensions": dimensions,
        "pearson_R_fidelity": _safe_correlation(reliability, fidelity, "pearson"),
        "spearman_R_fidelity": _safe_correlation(reliability, fidelity, "spearman"),
        "quartiles": _quartile_metrics(reliability, fidelity),
        "reliability_distribution": {
            "mean": float(np.mean(reliability)),
            "median": float(np.median(reliability)),
            "std": float(np.std(reliability)),
            "min": float(np.min(reliability)),
            "max": float(np.max(reliability)),
            "fraction_zero": float(np.mean(reliability <= 1e-12)),
            "top10_mass": float(np.sum(np.sort(reliability)[-max(1, len(reliability) // 10):]) / (np.sum(reliability) + 1e-8)),
        },
        "ground_truth_used_only_after_reliability": True,
    }
    if canonical_dir is not None:
        path = canonical_dir / f"{regime}_seed_{seed}.json"
        if path.exists():
            result["canonical_v42"] = json.loads(path.read_text())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_interaction/v44"))
    parser.add_argument("--canonical-dir", type=Path, default=Path("results/synthetic_interaction/confirm_seeded"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for regime in ("i1_weak", "i2_medium", "i3_strong", "i4_private_noise"):
        for seed in range(1, args.seeds + 1):
            record = audit(regime, seed, args.canonical_dir)
            records.append(record)
            (args.out / f"{regime}_seed_{seed}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({
                "regime": regime,
                "seed": seed,
                "pearson": record["pearson_R_fidelity"],
                "spearman": record["spearman_R_fidelity"],
                "gap": record["quartiles"]["top_bottom_fidelity_gap"],
            }))
    (args.out / "c0_reliability_audit.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
