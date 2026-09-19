"""v5.2 P1: label-free additive purification of the canonical JAD target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.experiments.multibench.mosei_identifiability import (
    _load_data,
    _predict,
    _standardize,
    capacity_config,
    fit_additive_predictor,
)
from src.experiments.multibench.mosei_shortcut_localization import (
    ALPHAS,
    MAPPING,
    SOURCE_A,
    SOURCE_B,
    _capacity_probe,
    _mean_std,
    _r2,
    _replay_frozen_targets,
    _ridge_probe,
)


def _health(values):
    centered = values - values.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    spectrum = singular**2 / max(float(np.sum(singular**2)), 1e-12)
    rank = float(np.exp(-np.sum(spectrum[spectrum > 0] * np.log(spectrum[spectrum > 0]))))
    return {
        "variance": float(np.var(values, axis=0).mean()),
        "mean_norm": float(np.linalg.norm(values, axis=1).mean()),
        "effective_rank": rank,
        "fraction_near_zero_dimensions": float(np.mean(np.std(values, axis=0) < 1e-4)),
    }


def _fit_purifier(splits, target, seed, epochs, patience):
    train, valid = splits["train"], splits["valid"]
    config = capacity_config(train[SOURCE_A].shape[1], train[SOURCE_B].shape[1], target["train"].shape[1])
    model, training = fit_additive_predictor(
        train[SOURCE_A], train[SOURCE_B], target["train"],
        valid[SOURCE_A], valid[SOURCE_B], target["valid"], config, seed,
        batch_size=512, epochs=epochs, patience=patience, lr=1e-3, weight_decay=1e-4,
    )
    prediction = {
        split: _predict(model, data[SOURCE_A], data[SOURCE_B])
        for split, data in splits.items()
    }
    return prediction, {"parameters": training["parameters"], "capacity": config, "training": training}


def _audit_target(splits, target, seed, args):
    purifier, purifier_meta = _fit_purifier(splits, target, seed + 500, args.purifier_epochs, args.purifier_patience)
    purified = {split: target[split] - purifier[split] for split in splits}
    result = {
        "canonical": {"health": _health(target["test"]), "probes": {}},
        "purified": {"health": _health(purified["test"]), "probes": {}, "purifier": purifier_meta},
    }
    for name, value in (("canonical", target), ("purified", purified)):
        result[name]["probes"]["v_only"] = _ridge_probe(
            splits["train"][SOURCE_A], value["train"], splits["valid"][SOURCE_A], value["valid"],
            splits["test"][SOURCE_A], value["test"], seed + (0 if name == "canonical" else 10),
        )
        result[name]["probes"]["a_only"] = _ridge_probe(
            splits["train"][SOURCE_B], value["train"], splits["valid"][SOURCE_B], value["valid"],
            splits["test"][SOURCE_B], value["test"], seed + (1 if name == "canonical" else 11),
        )
        result[name]["probes"]["additive"] = _capacity_probe(
            splits, value, "additive", seed + (20 if name == "canonical" else 30), args.probe_epochs, args.probe_patience,
        )
        result[name]["probes"]["joint"] = _capacity_probe(
            splits, value, "joint", seed + (40 if name == "canonical" else 50), args.probe_epochs, args.probe_patience,
        )
        result[name]["joint_advantage"] = {
            split: result[name]["probes"]["joint"]["r2"][split] - result[name]["probes"]["additive"]["r2"][split]
            for split in splits
        }
    result["target_variance_ratio_test"] = result["purified"]["health"]["variance"] / max(result["canonical"]["health"]["variance"], 1e-12)
    return result


def _summarize(records):
    summary = {}
    for name in ("canonical", "purified"):
        summary[name] = {"health": {}, "probes": {}, "joint_advantage": {}}
        for metric in ("variance", "mean_norm", "effective_rank", "fraction_near_zero_dimensions"):
            summary[name]["health"][metric] = _mean_std([r[name]["health"][metric] for r in records])
        for probe in ("v_only", "a_only", "additive", "joint"):
            summary[name]["probes"][probe] = {
                split: _mean_std([r[name]["probes"][probe]["r2"][split] for r in records])
                for split in ("train", "valid", "test")
            }
        summary[name]["joint_advantage"] = {
            split: _mean_std([r[name]["joint_advantage"][split] for r in records])
            for split in ("train", "valid", "test")
        }
    return summary


def _fmt(value):
    return f"{value['mean']:.3f} ± {value['std']:.3f}"


def _gate(summary):
    canonical, purified = summary["canonical"], summary["purified"]
    selectivity = all(
        purified["probes"][probe]["valid"]["mean"] < canonical["probes"][probe]["valid"]["mean"]
        for probe in ("v_only", "a_only", "additive")
    )
    joint_positive = purified["joint_advantage"]["valid"]["mean"] > 0
    healthy = purified["health"]["variance"]["mean"] > 0.05 and purified["health"]["effective_rank"]["mean"] > 2.0
    return {
        "selectivity_reduced": selectivity,
        "joint_structure_positive": joint_positive,
        "health_pass": healthy,
        "p1_target_audit_pass": bool(selectivity and joint_positive and healthy),
    }


def _report(out_dir, summary, gate):
    c, p = summary["canonical"], summary["purified"]
    lines = [
        "# MOSEI v5.2 P1 additive purification target audit",
        "",
        "Run date: 2026-09-19. Frozen canonical raw MOSEI, pooled vision+audio → text, five seeds.",
        "",
        "## Protocol",
        "",
        "The frozen target is `d=q_joint-q_add`. A label-free capacity-matched additive predictor is fit on train only, selected by validation MSE, and subtracted to form `d_perp=d-d_hat_add`. No sentiment, test score, architecture sweep, or purification strength is used.",
        "",
        "## Reconstruction audit",
        "",
        "| Target | V-only R² | A-only R² | Additive V+A R² | Joint V+A R² | J_d |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `d` | {_fmt(c['probes']['v_only']['test'])} | {_fmt(c['probes']['a_only']['test'])} | {_fmt(c['probes']['additive']['test'])} | {_fmt(c['probes']['joint']['test'])} | {_fmt(c['joint_advantage']['test'])} |",
        f"| `d_perp` | {_fmt(p['probes']['v_only']['test'])} | {_fmt(p['probes']['a_only']['test'])} | {_fmt(p['probes']['additive']['test'])} | {_fmt(p['probes']['joint']['test'])} | {_fmt(p['joint_advantage']['test'])} |",
        "",
        "All values are mean ± sample standard deviation over seeds 1–5. Primary probe selection is validation-only.",
        "",
        "## Health",
        "",
        f"| Target | Variance | Mean norm | Effective rank | Near-zero dimensions |",
        "|---|---:|---:|---:|---:|",
        f"| `d` | {_fmt(c['health']['variance'])} | {_fmt(c['health']['mean_norm'])} | {_fmt(c['health']['effective_rank'])} | {_fmt(c['health']['fraction_near_zero_dimensions'])} |",
        f"| `d_perp` | {_fmt(p['health']['variance'])} | {_fmt(p['health']['mean_norm'])} | {_fmt(p['health']['effective_rank'])} | {_fmt(p['health']['fraction_near_zero_dimensions'])} |",
        "",
        "## P1 gate",
        "",
        f"- Reduced V/A/additive reconstructability: **{'PASS' if gate['selectivity_reduced'] else 'FAIL'}**",
        f"- Positive joint residual advantage `J_d_perp`: **{'PASS' if gate['joint_structure_positive'] else 'FAIL'}**",
        f"- Nontrivial target health: **{'PASS' if gate['health_pass'] else 'FAIL'}**",
        f"- Target-side P1 audit: **{'PASS' if gate['p1_target_audit_pass'] else 'FAIL'}**",
        "",
        "This audit establishes only hypothesis-class-relative conditional selectivity; it does not establish causal interaction, PID synergy, or exact ANOVA interaction.",
        "",
        "## Decision",
        "",
        "If the P1 target audit passes, the next permitted step is the frozen seed-1 synthetic IPIB mechanism check comparing canonical D2 and `D2_perp`. If it fails, do not distill purified MOSEI or add another method.",
        "",
    ]
    (out_dir / "P1_TARGET_AUDIT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--cache", type=Path, default=Path("results/mosei/identifiability_exact/mosei_pooled.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/mosei/v52_p1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--purifier-epochs", type=int, default=80)
    parser.add_argument("--purifier-patience", type=int, default=8)
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--probe-patience", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits, metadata = _load_data(args.data_dir, args.cache)
    splits = _standardize(splits)
    records = []
    for seed in args.seeds:
        frozen = _replay_frozen_targets(splits, seed)
        target = {split: frozen[split]["d"] for split in splits}
        record = _audit_target(splits, target, seed, args)
        record["seed"] = seed
        records.append(record)
        (args.out / f"p1_target_audit_seed_{seed}.json").write_text(json.dumps(record, indent=2))
    summary = _summarize(records)
    gate = _gate(summary)
    payload = {"protocol": {"mapping": MAPPING, "seeds": args.seeds, "selection": "train_fit_validation_select_test_report", "data_metadata": metadata, "ridge_alphas": list(ALPHAS)}, "gate": gate, "summary": summary, "records": records}
    (args.out / "p1_target_audit.json").write_text(json.dumps(payload, indent=2))
    _report(args.out, summary, gate)


if __name__ == "__main__":
    main()
