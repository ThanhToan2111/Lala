"""v6.0 alignment-breaking null calibration for the operational J score."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

from src.datasets.identifiable_interaction import IPIBConfig, make_dataset
from src.experiments.hypothesis_class.h0_audit import (
    _fit_regression,
    _load_data,
    _standardize,
)
from src.experiments.multibench.meld_screening import load_meld
from src.experiments.multibench.mosei_identifiability import (
    AdditivePredictor,
    JointPredictor,
    MAPPINGS,
    capacity_config,
    _seed,
)


def _ipib_splits(seed: int):
    raw = make_dataset(IPIBConfig(beta=0.0), seed=seed)
    return [{"left": split["x1"].numpy(), "right": split["x2"].numpy(), "target": split["x3"].numpy()} for split in raw]


def _natural_splits(splits, mapping):
    left_name, right_name, target_name = mapping
    return [{"left": split[left_name], "right": split[right_name], "target": split[target_name]} for split in splits.values()]


def _shuffle_right(spec, replicate: int):
    result = []
    for split_index, split in enumerate(spec):
        rng = np.random.default_rng(16000 + replicate * 31 + split_index)
        shuffled = dict(split)
        shuffled["right"] = split["right"][rng.permutation(len(split["right"]))]
        result.append(shuffled)
    return result


def _fit_pair(splits, seed, args):
    train, valid, test = splits
    left_dim, right_dim, target_dim = train["left"].shape[1], train["right"].shape[1], train["target"].shape[1]
    capacity = capacity_config(left_dim, right_dim, target_dim)
    fit_args = argparse.Namespace(**vars(args), dataset="calibration", mapping_name="alignment_null")
    models = {}
    for name, model, model_seed in (
        ("additive", AdditivePredictor(left_dim, right_dim, target_dim, capacity["hidden_dim"]), seed),
        ("product", JointPredictor(left_dim, right_dim, target_dim, capacity["rank"]), seed + 1),
    ):
        _seed(model_seed)
        models[name], _, _ = _fit_regression(
            model,
            train["left"], train["right"], train["target"],
            valid["left"], valid["right"], valid["target"],
            model_seed, fit_args,
        )
    scores = {}
    for split_name, split in zip(("train", "valid", "test"), (train, valid, test)):
        predictions = {
            name: model(torch.from_numpy(split["left"]), torch.from_numpy(split["right"])).detach().numpy()
            for name, model in models.items()
        }
        scores[split_name] = {
            "additive_r2": float(r2_score(split["target"], predictions["additive"], multioutput="variance_weighted")),
            "product_r2": float(r2_score(split["target"], predictions["product"], multioutput="variance_weighted")),
        }
        scores[split_name]["J"] = scores[split_name]["product_r2"] - scores[split_name]["additive_r2"]
    return {"capacity": capacity, "scores": scores}


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "per_replicate": values.tolist(),
    }


def _historical_values(root: Path):
    mosei = json.loads((root / "results/mosei/identifiability_exact/mosei_identifiability.json").read_text())
    meld = json.loads((root / "results/meld/v55/meld_two_gate.json").read_text())
    mustard = json.loads((root / "results/mustard/v54/mustard_two_gate.json").read_text())
    rows = []
    rows.append(("MOSEI", "VA_to_T", [record["metrics"]["valid"]["joint_advantage"] for record in mosei["records"] if record["mapping"] == "VA_to_T"]))
    for dataset, payload in (("MELD", meld), ("MUStARD", mustard)):
        for mapping, record in payload["g1"].items():
            rows.append((dataset, mapping, record["summary"]["valid"]["joint_advantage"]["per_seed"]))
    return rows


def _write_threshold_sensitivity(out: Path, root: Path):
    thresholds = (0.005, 0.01, 0.02)
    rows = []
    for dataset, mapping, values in _historical_values(root):
        rows.append({"dataset": dataset, "mapping": mapping, "J_val_mean": float(np.mean(values)), "J_val_std": float(np.std(values, ddof=1)), "per_seed": values, "classification": {str(tau): [value > tau for value in values] for tau in thresholds}})
    (out / "threshold_sensitivity.json").write_text(json.dumps({"historical_gate": 0.01, "thresholds": thresholds, "rows": rows}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/calibration/v60"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--mosei-cache", type=Path, default=Path("results/mosei/identifiability_exact/mosei_pooled.npz"))
    parser.add_argument("--mosei-data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--meld-processed-dir", type=Path, default=Path("data/multibench/meld_download/processed"))
    parser.add_argument("--meld-annotation-dir", type=Path, default=Path("data/multibench/meld_download/annotations"))
    parser.add_argument("--meld-embedding", type=Path, default=Path("data/multibench/meld_download/processed/embedding.p"))
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mosei, mosei_meta = _load_data(args.mosei_data_dir, args.mosei_cache)
    mosei = _standardize(mosei)
    meld, meld_meta = load_meld(args.meld_processed_dir, args.meld_annotation_dir, args.meld_embedding)
    settings = {
        "IPIB_I0": (_ipib_splits, ("x1", "x2", "x3"), {"source": "src.datasets.identifiable_interaction", "regime": "i0_no_joint"}),
        "MOSEI_VA_to_T": (lambda seed: _natural_splits(mosei, MAPPINGS["VA_to_T"]), MAPPINGS["VA_to_T"], mosei_meta),
        "MELD_VA_to_T": (lambda seed: _natural_splits(meld, MAPPINGS["VA_to_T"]), MAPPINGS["VA_to_T"], meld_meta),
    }
    null_distributions = {"protocol": {"null": "shuffle right/source modality independently within each split", "replicates": args.replicates, "selection_split": "validation", "test_used_for_threshold": False}, "settings": {}}
    for name, (loader, mapping, metadata) in settings.items():
        values = []
        capacity = None
        for replicate in range(args.replicates):
            base = loader(replicate + 1)
            shuffled = _shuffle_right(base, replicate)
            result = _fit_pair(shuffled, 1000 + replicate, args)
            capacity = result["capacity"]
            values.append(result["scores"]["valid"]["J"])
        null_distributions["settings"][name] = {"metadata": metadata, "capacity": capacity, "summary": _summary(values)}
        print(json.dumps({"setting": name, "replicates": len(values), "q99": null_distributions["settings"][name]["summary"]["q99"]}))
    (args.out / "null_distributions.json").write_text(json.dumps(null_distributions, indent=2) + "\n")
    _write_threshold_sensitivity(args.out, args.project_root)
    report = ["# v6.0 null and threshold calibration", "", f"Alignment-breaking null: independently shuffle the right/source modality within each train/validation/test split; B={args.replicates} replicates per setting.", "", f"The historical gate is τ=0.01. B={args.replicates} is a compute-limited calibration replicate count; it is reported as a finite-sample audit, not as a replacement for a larger preregistered null.", "", "## Null distributions", "", "| Setting | Mean | Std | Median | q95 | q99 | Max | q99 > τ |", "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for name, value in null_distributions["settings"].items():
        summary = value["summary"]
        report.append(f"| `{name}` | {summary['mean']:.5f} | {summary['std']:.5f} | {summary['median']:.5f} | {summary['q95']:.5f} | {summary['q99']:.5f} | {summary['max']:.5f} | {'yes' if summary['q99'] > 0.01 else 'no'} |")
    report += ["", "## Interpretation", "", "The historical gate τ=0.01 is preserved for comparability; it is not recalibrated from these nulls. If q99 exceeds τ, the gate is not a 1% false-positive control under that setting and should be treated as descriptive until a larger preregistered null is run. Threshold sensitivity at 0.005 and 0.02 is descriptive and does not change historical labels. The shuffle is an operational alignment-breaking null; it does not preserve all lower-order dependencies.", "", "See `threshold_sensitivity.json` for MOSEI, MELD, and MUStARD historical values."]
    (args.out / "NULL_CALIBRATION_REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
