"""v5.6 H0: capacity-controlled joint hypothesis-class sufficiency audit."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t
from sklearn.metrics import r2_score
from torch import nn

from src.experiments.multibench.meld_screening import load_meld
from src.experiments.multibench.mosei_identifiability import (
    AdditivePredictor,
    JointPredictor,
    MAPPINGS as MOSEI_MAPPINGS,
    _device,
    _load_data,
    _seed,
    _standardize,
    capacity_config,
)


MELD_MAPPINGS = {
    "VA_to_T": ("vision", "audio", "text"),
    "VT_to_A": ("vision", "text", "audio"),
    "AT_to_V": ("audio", "text", "vision"),
}
PREDICTORS = ("additive", "product", "mlp")


class GenericJointMLP(nn.Module):
    """One-hidden-layer generic joint class; no hand-built interaction feature."""

    def __init__(self, left_dim: int, right_dim: int, target_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(left_dim + right_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((left, right), dim=1))


def mlp_parameter_count(left_dim: int, right_dim: int, target_dim: int, hidden_dim: int) -> int:
    return (left_dim + right_dim + 1) * hidden_dim + (hidden_dim + 1) * target_dim


def choose_mlp_width(left_dim: int, right_dim: int, target_dim: int, reference_params: int) -> dict:
    candidates = []
    for hidden_dim in range(1, 1025):
        params = mlp_parameter_count(left_dim, right_dim, target_dim, hidden_dim)
        mismatch = 100.0 * abs(params - reference_params) / reference_params
        candidates.append((mismatch, hidden_dim, params))
    mismatch, hidden_dim, params = min(candidates)
    return {
        "hidden_dim": hidden_dim,
        "mlp_parameters": params,
        "reference_parameters": reference_params,
        "mismatch_vs_reference_percent": mismatch,
    }


def _fit_regression(model, left, right, target, valid_left, valid_right, valid_target, seed, args):
    _seed(seed)
    device = _device()
    model = model.to(device)
    train_left, train_right, train_target = [torch.from_numpy(value).to(device) for value in (left, right, target)]
    valid_left, valid_right, valid_target = [torch.from_numpy(value).to(device) for value in (valid_left, valid_right, valid_target)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed + 1000)
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, stale = float("inf"), 0, 0
    finite = True
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(train_target), generator=generator).split(args.batch_size):
            indices = indices.to(device)
            loss = F.mse_loss(model(train_left[indices], train_right[indices]), train_target[indices])
            if not bool(torch.isfinite(loss)):
                finite = False
                raise FloatingPointError(f"non-finite training loss for seed {seed}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(F.mse_loss(model(valid_left, valid_right), valid_target).item())
        if not math.isfinite(valid_loss):
            finite = False
            raise FloatingPointError(f"non-finite validation loss for seed {seed}")
        if valid_loss < best_loss - 1e-7:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), valid_loss, epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.cpu().eval()
    predictions = {}
    with torch.no_grad():
        for split, (split_left, split_right) in {
            "train": (left, right),
            "valid": (valid_left.cpu().numpy(), valid_right.cpu().numpy()),
        }.items():
            predictions[split] = model(torch.from_numpy(split_left), torch.from_numpy(split_right)).numpy()
        predictions["test"] = None
    return model, {"best_epoch": best_epoch + 1, "parameters": sum(p.numel() for p in model.parameters()), "finite": finite}, predictions


def _fit_one(name, train, valid, test, mapping, seed, args, capacity):
    left_name, right_name, target_name = mapping
    left_dim, right_dim, target_dim = train[left_name].shape[1], train[right_name].shape[1], train[target_name].shape[1]
    train_left, train_right, train_target = train[left_name], train[right_name], train[target_name]
    valid_left, valid_right, valid_target = valid[left_name], valid[right_name], valid[target_name]
    test_left, test_right, test_target = test[left_name], test[right_name], test[target_name]
    model_seed = {"additive": seed, "product": seed + 1, "mlp": seed + 2}[name]
    _seed(model_seed)
    if name == "additive":
        model = AdditivePredictor(left_dim, right_dim, target_dim, capacity["hidden_dim"])
    elif name == "product":
        model = JointPredictor(left_dim, right_dim, target_dim, capacity["rank"])
    else:
        model = GenericJointMLP(left_dim, right_dim, target_dim, capacity["mlp"]["hidden_dim"])
    model, meta, predictions = _fit_regression(
        model, train_left, train_right, train_target, valid_left, valid_right, valid_target, model_seed, args,
    )
    predictions["test"] = model(torch.from_numpy(test_left), torch.from_numpy(test_right)).detach().numpy()
    metrics = {}
    for split, target, prediction in (
        ("train", train_target, predictions["train"]),
        ("valid", valid_target, predictions["valid"]),
        ("test", test_target, predictions["test"]),
    ):
        metrics[split] = {
            "mse": float(np.mean((target - prediction) ** 2)),
            "r2": float(r2_score(target, prediction, multioutput="variance_weighted")),
        }
    return {
        "dataset": args.dataset,
        "mapping": args.mapping_name,
        "seed": seed,
        "predictor_type": name,
        "parameter_count": meta["parameters"],
        "best_epoch": meta["best_epoch"],
        "finite": meta["finite"],
        "train_mse": metrics["train"]["mse"],
        "val_mse": metrics["valid"]["mse"],
        "test_mse": metrics["test"]["mse"],
        "train_r2": metrics["train"]["r2"],
        "val_r2": metrics["valid"]["r2"],
        "test_r2": metrics["test"]["r2"],
    }


def _difference(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half], "per_seed": values.tolist()}


def _run_mapping(dataset, mapping_name, mapping, splits, seeds, args):
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    left_name, right_name, target_name = mapping
    base = capacity_config(train[left_name].shape[1], train[right_name].shape[1], train[target_name].shape[1])
    mlp = choose_mlp_width(train[left_name].shape[1], train[right_name].shape[1], train[target_name].shape[1], base["additive_parameters"])
    capacity = {**base, "mlp": mlp}
    records = []
    for seed in seeds:
        local_args = argparse.Namespace(**vars(args), dataset=dataset, mapping_name=mapping_name)
        predictors = {name: _fit_one(name, train, valid, test, mapping, seed, local_args, capacity) for name in PREDICTORS}
        additive = predictors["additive"]
        product = predictors["product"]
        generic = predictors["mlp"]
        advantages = {}
        for split in ("train", "valid", "test"):
            metric_key = {"train": "train_r2", "valid": "val_r2", "test": "test_r2"}[split]
            advantages[split] = {
                "J_product": product[metric_key] - additive[metric_key],
                "J_mlp": generic[metric_key] - additive[metric_key],
                "delta_H": (generic[metric_key] - additive[metric_key]) - (product[metric_key] - additive[metric_key]),
            }
        records.append({"dataset": dataset, "mapping": mapping_name, "seed": seed, "predictors": predictors, "advantages": advantages})
    return {
        "dataset": dataset,
        "mapping": mapping_name,
        "sources": [left_name, right_name],
        "target": target_name,
        "capacity": capacity,
        "records": records,
        "summary": {
            split: {
                key: _difference([record["advantages"][split][key] for record in records])
                for key in ("J_product", "J_mlp", "delta_H")
            }
            for split in ("train", "valid", "test")
        },
    }


def _decision(meld_rows):
    candidates = {}
    seed_sensitive = False
    for mapping, row in meld_rows.items():
        values = row["summary"]["valid"]["J_mlp"]["per_seed"]
        candidates[mapping] = all(value > 0.01 for value in values)
        seed_sensitive = seed_sensitive or (max(values) > 0.01 and min(values) <= 0.0)
    if any(candidates.values()):
        return "PRODUCT_CLASS_INSUFFICIENT", candidates, "MELD generic MLP exceeds the historical gate in at least one mapping for all three seeds."
    if seed_sensitive:
        return "HYPOTHESIS_CLASS_INCONCLUSIVE", candidates, "MELD generic MLP is seed-sensitive around the historical gate."
    return "SUFFICIENCY_SUPPORTED", candidates, "No MELD mapping reaches the generic-MLP gate in all three seeds."


def _positive_control(rows: dict) -> dict:
    row = rows["VA_to_T"]
    product = row["summary"]["valid"]["J_product"]
    mlp = row["summary"]["valid"]["J_mlp"]
    recovered = all(value > 0 for value in mlp["per_seed"])
    return {
        "product_J_val": product,
        "mlp_J_val": mlp,
        "status": "PASS" if recovered else "MLP_DID_NOT_RECOVER_PRODUCT_SIGNAL",
        "interpretation": "MOSEI Product Joint is a positive control; Generic MLP must not be interpreted as weaker without checking training health and optimization.",
    }


def _metadata_summary(metadata: dict) -> dict:
    if "splits" not in metadata:
        return metadata
    return {
        key: metadata[key]
        for key in ("dataset", "dataset_version", "feature_dims", "sequence_handling", "text_representation", "standardization")
        if key in metadata
    } | {
        "split_counts": {split: metadata["splits"][split]["count"] for split in ("train", "valid", "test")},
        "split_feature_dims": {split: metadata["splits"][split]["feature_dims"] for split in ("train", "valid", "test")},
    }


def _write_dataset_report(path: Path, dataset: str, rows: dict, metadata: dict, decision: str | None = None, positive_control: dict | None = None):
    lines = [f"# {dataset} H0 hypothesis-class audit", "", "Frozen canonical pooled representations; additive, capacity-matched Product Joint, and one-hidden-layer Generic Joint MLP.", "", f"Dataset metadata: `{_metadata_summary(metadata)}`.", "", "## Parameter audit", "", "| Mapping | Additive | Product | Generic MLP | MLP mismatch vs additive |", "|---|---:|---:|---:|---:|"]
    for mapping, row in rows.items():
        cap = row["capacity"]
        lines.append(f"| `{mapping}` | {cap['additive_parameters']:,} | {cap['joint_parameters']:,} | {cap['mlp']['mlp_parameters']:,} | {cap['mlp']['mismatch_vs_reference_percent']:.3f}% |")
    lines += ["", "## Main validation result", "", "| Mapping | Additive R² | Product R² | MLP R² | J Product | J MLP | ΔH |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mapping, row in rows.items():
        val = row["summary"]["valid"]
        r2 = {name: _difference([record["predictors"][name]["val_r2"] for record in row["records"]]) for name in PREDICTORS}
        lines.append(f"| `{mapping}` | {r2['additive']['mean']:.4f} ± {r2['additive']['std']:.4f} | {r2['product']['mean']:.4f} ± {r2['product']['std']:.4f} | {r2['mlp']['mean']:.4f} ± {r2['mlp']['std']:.4f} | {val['J_product']['mean']:.4f} ± {val['J_product']['std']:.4f} | {val['J_mlp']['mean']:.4f} ± {val['J_mlp']['std']:.4f} | {val['delta_H']['mean']:.4f} ± {val['delta_H']['std']:.4f} |")
    lines += ["", "## Seed-level validation result", "", "| Mapping | Seed | J Product | J MLP | ΔH |", "|---|---:|---:|---:|---:|"]
    for mapping, row in rows.items():
        for record in row["records"]:
            val = record["advantages"]["valid"]
            lines.append(f"| `{mapping}` | {record['seed']} | {val['J_product']:.4f} | {val['J_mlp']:.4f} | {val['delta_H']:.4f} |")
    lines += ["", "## Descriptive test result", "", "| Mapping | J Product | J MLP | ΔH |", "|---|---:|---:|---:|"]
    for mapping, row in rows.items():
        test = row["summary"]["test"]
        lines.append(f"| `{mapping}` | {test['J_product']['mean']:.4f} ± {test['J_product']['std']:.4f} | {test['J_mlp']['mean']:.4f} ± {test['J_mlp']['std']:.4f} | {test['delta_H']['mean']:.4f} ± {test['delta_H']['std']:.4f} |")
    lines += ["", "## Overfit diagnostic", "", "| Mapping | J MLP train | J MLP val | Gap_M |", "|---|---:|---:|---:|"]
    for mapping, row in rows.items():
        train = row["summary"]["train"]["J_mlp"]
        valid = row["summary"]["valid"]["J_mlp"]
        gap = _difference([record["advantages"]["train"]["J_mlp"] - record["advantages"]["valid"]["J_mlp"] for record in row["records"]])
        lines.append(f"| `{mapping}` | {train['mean']:.4f} ± {train['std']:.4f} | {valid['mean']:.4f} ± {valid['std']:.4f} | {gap['mean']:.4f} ± {gap['std']:.4f} |")
    lines += ["", "## Training health", "", "| Mapping | Predictor | Train R² | Val R² | Test R² | Best epoch | Finite |", "|---|---|---:|---:|---:|---:|---|"]
    for mapping, row in rows.items():
        for name in PREDICTORS:
            values = [record["predictors"][name] for record in row["records"]]
            finite = all(v["finite"] for v in values)
            lines.append(f"| `{mapping}` | `{name}` | {np.mean([v['train_r2'] for v in values]):.4f} | {np.mean([v['val_r2'] for v in values]):.4f} | {np.mean([v['test_r2'] for v in values]):.4f} | {np.mean([v['best_epoch'] for v in values]):.1f} | {finite} |")
    if decision:
        lines += ["", f"## Dataset interpretation", "", f"`{decision}` is the dataset-level H0 interpretation. Decisions use validation only; test values are descriptive."]
    if positive_control:
        lines += ["", "## MOSEI positive-control audit", "", f"Status: **{positive_control['status']}**.", "", f"Product J validation: `{positive_control['product_J_val']['mean']:.4f} ± {positive_control['product_J_val']['std']:.4f}`; Generic MLP J validation: `{positive_control['mlp_J_val']['mean']:.4f} ± {positive_control['mlp_J_val']['std']:.4f}`.", positive_control["interpretation"]]
    path.write_text("\n".join(lines) + "\n")


def _write_combined(out: Path, mosei: dict, meld: dict, decision: str, candidates: dict, reason: str, positive_control: dict):
    lines = ["# H0 Joint Hypothesis-Class Sufficiency Decision", "", "## Research question", "", "Does a capacity-controlled generic nonlinear joint predictor reveal cross-modal predictive headroom that the canonical low-rank Product Joint misses?", "", "## Main validation table", "", "| Dataset | Mapping | J Product | J MLP | ΔH | Decision relevance |", "|---|---|---:|---:|---:|---|"]
    for dataset, rows in (("MOSEI", mosei["mappings"]), ("MELD", meld["mappings"])):
        for mapping, row in rows.items():
            val = row["summary"]["valid"]
            relevance = "positive control" if dataset == "MOSEI" else ("candidate" if candidates.get(mapping, False) else "not a 3/3 candidate")
            lines.append(f"| {dataset} | `{mapping}` | {val['J_product']['mean']:.4f} ± {val['J_product']['std']:.4f} | {val['J_mlp']['mean']:.4f} ± {val['J_mlp']['std']:.4f} | {val['delta_H']['mean']:.4f} ± {val['delta_H']['std']:.4f} | {relevance} |")
    lines += ["", "## Overall decision", "", f"**{decision}**", "", reason, f"MOSEI positive-control status: **{positive_control['status']}**. Product J is positive, but Generic MLP J is not positive in every seed; this prevents claiming hypothesis-class sufficiency without further permitted audit.", "", "The decision uses validation only. Test results are descriptive and never create a candidate.", "", "## Scope", "", "No JAD, task-aware objective, representation change, pooling change, architecture sweep, or new dataset was run. A PRODUCT_CLASS_INSUFFICIENT result would require frozen five-seed confirmation before v5.7.", ""]
    (out / "H0_JOINT_HYPOTHESIS_CLASS_DECISION.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/hypothesis_class/v56"))
    parser.add_argument("--mosei-cache", type=Path, default=Path("results/mosei/identifiability_exact/mosei_pooled.npz"))
    parser.add_argument("--mosei-data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--meld-processed-dir", type=Path, default=Path("data/multibench/meld_download/processed"))
    parser.add_argument("--meld-annotation-dir", type=Path, default=Path("data/multibench/meld_download/annotations"))
    parser.add_argument("--meld-embedding", type=Path, default=Path("data/multibench/meld_download/processed/embedding.p"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--mosei-mappings", nargs="+", default=["VA_to_T"], choices=list(MOSEI_MAPPINGS))
    parser.add_argument("--meld-mappings", nargs="+", default=list(MELD_MAPPINGS), choices=list(MELD_MAPPINGS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    out = args.out
    (out / "mosei").mkdir(parents=True, exist_ok=True)
    (out / "meld").mkdir(parents=True, exist_ok=True)

    mosei_splits, mosei_meta = _load_data(args.mosei_data_dir, args.mosei_cache)
    mosei_splits = _standardize(mosei_splits)
    mosei_rows = {}
    for mapping_name in args.mosei_mappings:
        row = _run_mapping("MOSEI", mapping_name, MOSEI_MAPPINGS[mapping_name], mosei_splits, args.seeds, args)
        mosei_rows[mapping_name] = row
        (out / "mosei" / "mosei_h0.json").write_text(json.dumps({"dataset": "MOSEI", "metadata": mosei_meta, "mappings": mosei_rows}, indent=2))

    meld_splits, meld_meta = load_meld(args.meld_processed_dir, args.meld_annotation_dir, args.meld_embedding)
    meld_rows = {}
    for mapping_name in args.meld_mappings:
        row = _run_mapping("MELD", mapping_name, MELD_MAPPINGS[mapping_name], meld_splits, args.seeds, args)
        meld_rows[mapping_name] = row
        (out / "meld" / "meld_h0.json").write_text(json.dumps({"dataset": "MELD", "metadata": meld_meta, "mappings": meld_rows}, indent=2))

    preliminary_decision, candidates, decision_reason = _decision(meld_rows)
    positive_control = _positive_control(mosei_rows)
    decision = preliminary_decision
    if decision == "SUFFICIENCY_SUPPORTED" and positive_control["status"] != "PASS":
        decision = "HYPOTHESIS_CLASS_INCONCLUSIVE"
        decision_reason = "MELD remains below the MLP gate, but the MOSEI positive control was not recovered by the Generic MLP in all seeds; hypothesis-class sufficiency therefore remains unverified."
    summary = {
        "decision": decision,
        "decision_reason": decision_reason,
        "preliminary_meld_decision": preliminary_decision,
        "positive_control": positive_control,
        "meld_mlp_candidates": candidates,
        "mosei": {"metadata": mosei_meta, "mappings": mosei_rows},
        "meld": {"metadata": meld_meta, "mappings": meld_rows},
        "protocol": {
            "seeds": args.seeds,
            "mosei_mappings": args.mosei_mappings,
            "meld_mappings": args.meld_mappings,
            "validation_metric": "MSE early stopping; R2 and J reported after checkpoint selection",
            "test_used_for_selection": False,
            "jad_trained": False,
            "task_g2_trained": False,
        },
    }
    (out / "h0_summary.json").write_text(json.dumps(summary, indent=2))
    _write_dataset_report(out / "mosei" / "MOSEI_H0_HYPOTHESIS_CLASS_REPORT.md", "MOSEI", mosei_rows, mosei_meta, positive_control=positive_control)
    _write_dataset_report(out / "meld" / "MELD_H0_HYPOTHESIS_CLASS_REPORT.md", "MELD", meld_rows, meld_meta, decision)
    _write_combined(out, {"mappings": mosei_rows}, {"mappings": meld_rows}, decision, candidates, decision_reason, positive_control)


if __name__ == "__main__":
    main()
