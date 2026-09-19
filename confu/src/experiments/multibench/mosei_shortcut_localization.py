"""v5.2 S0: localize lower-order leakage in frozen MOSEI JAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src.experiments.multibench.mosei_identifiability import (
    MAPPINGS,
    _load_data,
    _predict,
    _standardize,
    capacity_config,
    fit_additive_predictor,
    fit_joint_predictor,
)
from src.experiments.multibench.mosei_r2 import _train_interaction


MAPPING = "VA_to_T"
SOURCE_A, SOURCE_B, TARGET = MAPPINGS[MAPPING]
REPRESENTATIONS = ("q_add", "q_joint", "d", "h_d2")
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(target, prediction, multioutput="variance_weighted"))


def _ridge_probe(train_x, train_y, valid_x, valid_y, test_x, test_y, seed):
    scaler = StandardScaler().fit(train_x)
    x_train, x_valid, x_test = (scaler.transform(x) for x in (train_x, valid_x, test_x))
    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(x_train, train_y)
        score = _r2(valid_y, model.predict(x_valid))
        if best is None or score > best[0]:
            best = score, alpha, model
    _, alpha, model = best
    return {
        "probe": "ridge",
        "alpha": alpha,
        "parameters": int(model.coef_.size + model.intercept_.size),
        "r2": {
            "train": _r2(train_y, model.predict(x_train)),
            "valid": _r2(valid_y, model.predict(x_valid)),
            "test": _r2(test_y, model.predict(x_test)),
        },
    }


def _capacity_probe(splits, target, kind, seed, epochs, patience):
    train, valid = splits["train"], splits["valid"]
    config = capacity_config(train[SOURCE_A].shape[1], train[SOURCE_B].shape[1], target["train"].shape[1])
    args = {
        "batch_size": 512,
        "epochs": epochs,
        "patience": patience,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }
    fit = fit_additive_predictor if kind == "additive" else fit_joint_predictor
    model, training = fit(
        train[SOURCE_A], train[SOURCE_B], target["train"],
        valid[SOURCE_A], valid[SOURCE_B], target["valid"], config, seed, **args,
    )
    outputs = {
        split: _predict(model, data[SOURCE_A], data[SOURCE_B])
        for split, data in splits.items()
    }
    return {
        "probe": kind,
        "parameters": training["parameters"],
        "capacity": config,
        "training": training,
        "r2": {split: _r2(target[split], outputs[split]) for split in splits},
    }


def _replay_frozen_targets(splits, seed):
    """Recreate frozen v5.1 q_add, q_joint, and d tensors."""
    train, valid = splits["train"], splits["valid"]
    config = capacity_config(train[SOURCE_A].shape[1], train[SOURCE_B].shape[1], train[TARGET].shape[1])
    args = {"batch_size": 512, "epochs": 80, "patience": 8, "lr": 1e-3, "weight_decay": 1e-4}
    additive, _ = fit_additive_predictor(
        train[SOURCE_A], train[SOURCE_B], train[TARGET],
        valid[SOURCE_A], valid[SOURCE_B], valid[TARGET], config, seed, **args,
    )
    joint, _ = fit_joint_predictor(
        train[SOURCE_A], train[SOURCE_B], train[TARGET],
        valid[SOURCE_A], valid[SOURCE_B], valid[TARGET], config, seed + 1, **args,
    )
    predictions = {
        split: {
            "q_add": _predict(additive, data[SOURCE_A], data[SOURCE_B]),
            "q_joint": _predict(joint, data[SOURCE_A], data[SOURCE_B]),
        }
        for split, data in splits.items()
    }
    target = {
        split: predictions[split]["q_joint"] - predictions[split]["q_add"]
        for split in splits
    }
    for split in splits:
        predictions[split]["d"] = target[split]
    return predictions


def _replay_frozen_seed(splits, seed, interaction_epochs, interaction_patience):
    """Recreate frozen v5.1 R2 tensors; no new method is trained."""
    predictions = _replay_frozen_targets(splits, seed)
    target = {split: predictions[split]["d"] for split in splits}
    interaction, _, _ = _train_interaction(
        splits, SOURCE_A, SOURCE_B, target, seed + 22, interaction_epochs, interaction_patience,
    )
    for split in splits:
        predictions[split]["h_d2"] = interaction[split]
    return predictions


def _audit_seed(splits, seed, args):
    representations = _replay_frozen_seed(splits, seed, args.interaction_epochs, args.interaction_patience)
    records = {}
    for index, name in enumerate(REPRESENTATIONS):
        target = {split: representations[split][name] for split in splits}
        source_v = _ridge_probe(
            splits["train"][SOURCE_A], target["train"],
            splits["valid"][SOURCE_A], target["valid"],
            splits["test"][SOURCE_A], target["test"], seed + index * 10,
        )
        source_a = _ridge_probe(
            splits["train"][SOURCE_B], target["train"],
            splits["valid"][SOURCE_B], target["valid"],
            splits["test"][SOURCE_B], target["test"], seed + index * 10 + 1,
        )
        additive = _capacity_probe(splits, target, "additive", seed + 100 + index, args.probe_epochs, args.probe_patience)
        joint = _capacity_probe(splits, target, "joint", seed + 200 + index, args.probe_epochs, args.probe_patience)
        records[name] = {"v_only": source_v, "a_only": source_a, "additive": additive, "joint": joint}
        records[name]["joint_advantage"] = {
            split: joint["r2"][split] - additive["r2"][split] for split in splits
        }
    return {"seed": seed, "mapping": MAPPING, "representations": records}


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def _summarize(records):
    summary = {}
    for name in REPRESENTATIONS:
        summary[name] = {}
        for probe in ("v_only", "a_only", "additive", "joint"):
            summary[name][probe] = {
                split: _mean_std([record["representations"][name][probe]["r2"][split] for record in records])
                for split in ("train", "valid", "test")
            }
        summary[name]["joint_advantage"] = {
            split: _mean_std([record["representations"][name]["joint_advantage"][split] for record in records])
            for split in ("train", "valid", "test")
        }
    return summary


def _decision(summary):
    d_values = [summary["d"][probe]["valid"]["mean"] for probe in ("v_only", "a_only")]
    h_values = [summary["h_d2"][probe]["valid"]["mean"] for probe in ("v_only", "a_only")]
    target_leakage = any(value >= 0.10 for value in d_values)
    distillation_amplification = any(h - d >= 0.05 for d, h in zip(d_values, h_values))
    if target_leakage and distillation_amplification:
        return "MIXED"
    if target_leakage:
        return "TARGET_LEVEL_LEAKAGE"
    if distillation_amplification:
        return "DISTILLATION_LEVEL_LEAKAGE"
    return "INCONCLUSIVE"


def _fmt(value):
    return f"{value['mean']:.3f} ± {value['std']:.3f}"


def _write_report(out_dir, summary, records, decision):
    lines = [
        "# MOSEI v5.2 S0 shortcut localization",
        "",
        "Run date: 2026-09-19. Frozen canonical raw MOSEI, pooled vision+audio → text, five R2 seeds.",
        "",
        "## Protocol",
        "",
        "S0 replays the frozen v5.1 predictor and rank-64 JAD settings to recover `q_add`, `q_joint`, `d=q_joint-q_add`, and `h_D2`. It does not introduce a new target, loss, architecture, task label, or hyperparameter sweep.",
        "",
        "V-only and A-only reconstruction use Ridge with alpha selected on validation. Additive and joint V+A references use the existing capacity-matched predictor classes (`AdditivePredictor`/`JointPredictor`, under 1% parameter mismatch), also selected by validation only. Test is reporting-only.",
        "",
        "## Required test-set table",
        "",
        "| Representation | V-only R² | A-only R² | Additive V+A R² | Joint V+A R² | J_z |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in REPRESENTATIONS:
        row = summary[name]
        lines.append(
            f"| `{name}` | {_fmt(row['v_only']['test'])} | {_fmt(row['a_only']['test'])} | {_fmt(row['additive']['test'])} | {_fmt(row['joint']['test'])} | {_fmt(row['joint_advantage']['test'])} |"
        )
    lines += [
        "",
        "Values are mean ± sample standard deviation over seeds 1–5. `J_z` is the joint reconstruction R² minus additive reconstruction R².",
        "",
        "## Validation gate and decision",
        "",
        f"The pre-registered S0 classification is **{decision}**. It is based on validation reconstruction, not test performance.",
        "",
        f"Validation single-modality R²: `d` is V-only {_fmt(summary['d']['v_only']['valid'])} and A-only {_fmt(summary['d']['a_only']['valid'])}; `h_D2` is V-only {_fmt(summary['h_d2']['v_only']['valid'])} and A-only {_fmt(summary['h_d2']['a_only']['valid'])}.",
        "",
        "Interpretation: S0 localizes whether lower-order leakage is already present in the JAD target or is introduced/amplified by distillation. It does not establish causality, PID synergy, or exact functional ANOVA interaction.",
        "",
        "## Frozen baseline context",
        "",
        "The prior five-seed R2 task baseline was lower-order accuracy `64.397 ± 0.231%`; D2 was `64.363 ± 0.288%` with audio-only shortcut R² `0.302 ± 0.048`. S0 is a mechanism audit, not a new accuracy benchmark.",
        "",
        "## Artifacts",
        "",
        "- `s0_shortcut_localization.json` — per-seed reconstruction metrics, probe hyperparameters, train/validation/test R².",
        "- `mosei_shortcut_localization.py` — reproducible S0 runner.",
        "",
    ]
    (out_dir / "S0_SHORTCUT_LOCALIZATION.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--cache", type=Path, default=Path("results/mosei/identifiability_exact/mosei_pooled.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/mosei/v52"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--interaction-epochs", type=int, default=70)
    parser.add_argument("--interaction-patience", type=int, default=8)
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--probe-patience", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits, metadata = _load_data(args.data_dir, args.cache)
    splits = _standardize(splits)
    records = [_audit_seed(splits, seed, args) for seed in args.seeds]
    summary = _summarize(records)
    decision = _decision(summary)
    payload = {
        "protocol": {
            "mapping": MAPPING,
            "seeds": args.seeds,
            "source": "canonical_raw_mosei",
            "data_metadata": metadata,
            "selection": "validation_only",
            "representations": list(REPRESENTATIONS),
            "ridge_alphas": list(ALPHAS),
        },
        "decision": decision,
        "summary": summary,
        "records": records,
    }
    (args.out / "s0_shortcut_localization.json").write_text(json.dumps(payload, indent=2))
    _write_report(args.out, summary, records, decision)


if __name__ == "__main__":
    main()
