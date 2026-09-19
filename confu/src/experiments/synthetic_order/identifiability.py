"""Audit whether v4.1 discovery targets contain a learnable signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.datasets.synthetic_order import REGIMES, make_dataset


MAPPINGS = (("12_to_3", "z1", "z2", "z3"), ("13_to_2", "z1", "z3", "z2"), ("23_to_1", "z2", "z3", "z1"))


def _fit(x_train, y_train, x_test, y_test):
    model = Ridge(alpha=1.0).fit(x_train, y_train)
    prediction = model.predict(x_test)
    return float(r2_score(y_test, prediction, multioutput="variance_weighted")), prediction


def audit(regime: str, seed: int):
    train, _, test = make_dataset(REGIMES[regime], seed)
    result = {"regime": regime, "seed": seed, "mappings": {}}
    for name, left, right, target in MAPPINGS:
        x_left_train = train[left].numpy()
        x_right_train = train[right].numpy()
        x_left_test = test[left].numpy()
        x_right_test = test[right].numpy()
        y_train, y_test = train[target].numpy(), test[target].numpy()
        _, left_prediction = _fit(x_left_train, y_train, x_left_test, y_test)
        _, right_prediction = _fit(x_right_train, y_train, x_right_test, y_test)
        additive = left_prediction + right_prediction
        additive_r2 = float(r2_score(y_test, additive, multioutput="variance_weighted"))
        joint_r2, joint_prediction = _fit(np.concatenate((x_left_train, x_right_train), 1), y_train, np.concatenate((x_left_test, x_right_test), 1), y_test)
        advantage = joint_prediction - additive
        result["mappings"][name] = {
            "additive_r2": additive_r2,
            "joint_r2": joint_r2,
            "joint_advantage_mean_variance": float(np.var(advantage, axis=0).mean()),
            "joint_advantage_norm": float(np.linalg.norm(advantage, axis=1).mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_order/discovery/identifiability.json"))
    args = parser.parse_args()
    values = [audit(regime, seed) for regime in ("s0_first_order", "s1_pair12", "s2_all_pairs", "s3_triple") for seed in range(1, args.seeds + 1)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(values, indent=2) + "\n")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
