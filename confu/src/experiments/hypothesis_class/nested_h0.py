"""v5.6.1 final nested joint-class headroom audit."""

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

from src.experiments.hypothesis_class.h0_audit import (
    MELD_MAPPINGS,
    _fit_regression,
    _load_data,
    _metadata_summary,
    _seed,
    _standardize,
)
from src.experiments.hypothesis_class.nested_joint import (
    NestedJointPredictor,
    choose_residual_width,
)
from src.experiments.multibench.meld_screening import load_meld
from src.experiments.multibench.mosei_identifiability import (
    JointPredictor,
    MAPPINGS as MOSEI_MAPPINGS,
    _device,
    capacity_config,
)


PREDICTOR_FIELDS = ("train", "valid", "test")


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half], "per_seed": values.tolist()}


def _predict(model, left, right):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(left), torch.from_numpy(right)).numpy()


def _metrics(target, prediction):
    return {
        "mse": float(np.mean((target - prediction) ** 2)),
        "r2": float(r2_score(target, prediction, multioutput="variance_weighted")),
    }


def _load_baseline(path: Path, dataset: str, mapping: str, seed: int) -> dict:
    payload = json.loads(path.read_text())
    for record in payload["mappings"][mapping]["records"]:
        if record["seed"] == seed:
            return record["predictors"]["additive"]
    raise KeyError(f"missing additive baseline for {dataset} {mapping} seed {seed} in {path}")


def _product_path(dataset: str, mapping: str, seed: int, args) -> Path:
    if dataset == "MOSEI":
        return args.mosei_product_dir / f"{mapping}_seed_{seed}_joint.pt"
    return args.out / "checkpoints" / "meld" / f"{mapping}_seed_{seed}_product.pt"


def _load_product(path: Path, left_dim: int, right_dim: int, target_dim: int, rank: int):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = JointPredictor(left_dim, right_dim, target_dim, rank)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _fit_meld_product(train, valid, mapping_name, mapping, seed, capacity, args, path: Path):
    left_name, right_name, target_name = mapping
    model = JointPredictor(train[left_name].shape[1], train[right_name].shape[1], train[target_name].shape[1], capacity["rank"])
    fit_args = argparse.Namespace(**vars(args), dataset="MELD", mapping_name="MELD")
    model, meta, _ = _fit_regression(
        model,
        train[left_name], train[right_name], train[target_name],
        valid[left_name], valid[right_name], valid[target_name],
        seed + 1, fit_args,
    )
    for parameter in model.parameters():
        parameter.grad = None
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "dataset": "MELD", "mapping": mapping_name, "seed": seed, "config": capacity, "source": "recreated_v56_product_reference"}, path)
    return model, {"best_epoch": meta["best_epoch"], "source": "recreated_v56_product_reference"}


def _fit_nested(product, train, valid, mapping, seed, args, residual_config):
    left_name, right_name, target_name = mapping
    left, right, target = train[left_name], train[right_name], train[target_name]
    valid_left, valid_right, valid_target = valid[left_name], valid[right_name], valid[target_name]
    _seed(seed + 2)
    nested = NestedJointPredictor(product, left.shape[1], right.shape[1], target.shape[1], residual_config["hidden_dim"])
    for parameter in nested.product.parameters():
        parameter.grad = None
    device = _device()
    nested = nested.to(device)
    train_left, train_right, train_target = [torch.from_numpy(value).to(device) for value in (left, right, target)]
    valid_left, valid_right, valid_target = [torch.from_numpy(value).to(device) for value in (valid_left, valid_right, valid_target)]
    optimizer = torch.optim.AdamW(nested.residual.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed + 3000)
    with torch.no_grad():
        initial_val_mse = float(F.mse_loss(nested(valid_left, valid_right), valid_target).item())
    best_state = copy.deepcopy(nested.state_dict())
    best_loss, best_epoch, stale = initial_val_mse, 0, 0
    finite = math.isfinite(initial_val_mse)
    first_residual_grad_norm = 0.0
    max_product_grad_norm = 0.0
    for epoch in range(args.epochs):
        nested.train()
        nested.product.eval()
        for indices in torch.randperm(len(train_target), generator=generator).split(args.batch_size):
            indices = indices.to(device)
            loss = F.mse_loss(nested(train_left[indices], train_right[indices]), train_target[indices])
            if not bool(torch.isfinite(loss)):
                finite = False
                raise FloatingPointError(f"non-finite nested loss for seed {seed}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            residual_grad = max((float(parameter.grad.detach().norm().item()) for parameter in nested.residual.parameters() if parameter.grad is not None), default=0.0)
            product_grad = max((float(parameter.grad.detach().norm().item()) for parameter in nested.product.parameters() if parameter.grad is not None), default=0.0)
            first_residual_grad_norm = first_residual_grad_norm or residual_grad
            max_product_grad_norm = max(max_product_grad_norm, product_grad)
            optimizer.step()
        nested.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(nested(valid_left, valid_right), valid_target).item())
        if not math.isfinite(val_loss):
            finite = False
            raise FloatingPointError(f"non-finite nested validation loss for seed {seed}")
        if val_loss < best_loss - 1e-7:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(nested.state_dict()), val_loss, epoch + 1, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    nested.load_state_dict(best_state)
    nested.cpu().eval()
    return nested, {
        "initial_val_mse": initial_val_mse,
        "best_val_mse": best_loss,
        "best_epoch": best_epoch,
        "finite": finite,
        "first_residual_grad_norm": first_residual_grad_norm,
        "max_product_grad_norm": max_product_grad_norm,
    }


def _run_mapping(dataset, mapping_name, mapping, splits, seed, args, baseline_path: Path):
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    left_name, right_name, target_name = mapping
    capacity = capacity_config(train[left_name].shape[1], train[right_name].shape[1], train[target_name].shape[1])
    product_path = _product_path(dataset, mapping_name, seed, args)
    if product_path.exists():
        product, product_payload = _load_product(product_path, train[left_name].shape[1], train[right_name].shape[1], train[target_name].shape[1], capacity["rank"])
        product_source = product_payload.get("source", "frozen_existing_checkpoint")
    else:
        if dataset != "MELD":
            raise FileNotFoundError(product_path)
        product, product_meta = _fit_meld_product(train, valid, mapping_name, mapping, seed, capacity, args, product_path)
        product_source = product_meta["source"]
    product_params = sum(parameter.numel() for parameter in product.parameters())
    residual_config = choose_residual_width(train[left_name].shape[1], train[right_name].shape[1], train[target_name].shape[1], product_params)
    nested, train_meta = _fit_nested(product, train, valid, mapping, seed, args, residual_config)
    additive = _load_baseline(baseline_path, dataset, mapping_name, seed)
    predictions = {}
    for split, data in splits.items():
        product_prediction = _predict(product, data[left_name], data[right_name])
        nested_prediction = _predict(nested, data[left_name], data[right_name])
        residual_prediction = nested_prediction - product_prediction
        predictions[split] = {
            "product": _metrics(data[target_name], product_prediction),
            "nested": _metrics(data[target_name], nested_prediction),
            "residual_norm": float(np.linalg.norm(residual_prediction, axis=1).mean()),
        }
    product_r2 = {split: predictions[split]["product"]["r2"] for split in PREDICTOR_FIELDS}
    nested_r2 = {split: predictions[split]["nested"]["r2"] for split in PREDICTOR_FIELDS}
    additive_r2 = {split: additive["val_r2" if split == "valid" else f"{split}_r2"] for split in PREDICTOR_FIELDS}
    delta = {split: nested_r2[split] - product_r2[split] for split in PREDICTOR_FIELDS}
    j_product = {split: product_r2[split] - additive_r2[split] for split in PREDICTOR_FIELDS}
    j_nested = {split: nested_r2[split] - additive_r2[split] for split in PREDICTOR_FIELDS}
    finite = bool(train_meta["finite"] and all(math.isfinite(value) for split in predictions.values() for model in ("product", "nested") for value in (split[model]["mse"], split[model]["r2"])))
    nested_checkpoint = args.out / "checkpoints" / dataset.lower() / f"{mapping_name}_seed_{seed}_nested.pt"
    nested_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": nested.state_dict(), "dataset": dataset, "mapping": mapping_name, "seed": seed, "product_checkpoint": str(product_path), "residual_config": residual_config}, nested_checkpoint)
    return {
        "dataset": dataset,
        "mapping": mapping_name,
        "seed": seed,
        "product_checkpoint": str(product_path),
        "nested_checkpoint": str(nested_checkpoint),
        "product_params": product_params,
        "residual_params": residual_config["residual_parameters"],
        "residual_ratio": residual_config["residual_ratio"],
        "residual_config": residual_config,
        "initial_val_mse": train_meta["initial_val_mse"],
        "best_val_mse": train_meta["best_val_mse"],
        "best_epoch": train_meta["best_epoch"],
        "product_train_r2": product_r2["train"],
        "nested_train_r2": nested_r2["train"],
        "product_val_r2": product_r2["valid"],
        "nested_val_r2": nested_r2["valid"],
        "product_test_r2": product_r2["test"],
        "nested_test_r2": nested_r2["test"],
        "J_product": {"train": j_product["train"], "valid": j_product["valid"], "test": j_product["test"]},
        "J_nested": {"train": j_nested["train"], "valid": j_nested["valid"], "test": j_nested["test"]},
        "delta_nested": delta,
        "residual_norm": predictions["valid"]["residual_norm"],
        "residual_norm_by_split": {split: predictions[split]["residual_norm"] for split in PREDICTOR_FIELDS},
        "finite": finite,
        "product_source": product_source,
        "first_residual_grad_norm": train_meta["first_residual_grad_norm"],
        "max_product_grad_norm": train_meta["max_product_grad_norm"],
        "additive_r2": additive_r2,
        "product_mse": {split: predictions[split]["product"]["mse"] for split in PREDICTOR_FIELDS},
        "nested_mse": {split: predictions[split]["nested"]["mse"] for split in PREDICTOR_FIELDS},
    }


def _aggregate(records):
    return {
        split: {
            field: _stats([record[field][split] for record in records])
            for field in ("J_product", "J_nested", "delta_nested")
        }
        for split in PREDICTOR_FIELDS
    }


def _decision(rows):
    candidates = {}
    unstable = False
    for mapping, payload in rows.items():
        records = payload["records"]
        values = [record["delta_nested"]["valid"] for record in records]
        candidates[mapping] = all(value > 0.01 for value in values)
        unstable = unstable or (max(values) > 0.01 and min(values) <= 0.01)
        unstable = unstable or any(not record["finite"] for record in records)
    if unstable:
        return "NESTED_AUDIT_INCONCLUSIVE", candidates, "Nested validation headroom or numerical health is not consistent across the three seeds."
    if any(candidates.values()):
        return "NESTED_HEADROOM_PRESENT", candidates, "At least one MELD mapping exceeds Δ_nested=0.01 in all three seeds."
    return "NESTED_HEADROOM_ABSENT", candidates, "No MELD mapping exceeds Δ_nested=0.01 in all three seeds."


def _write_dataset_report(path, dataset, mappings, metadata, decision=None):
    lines = [f"# {dataset} nested joint-class headroom audit", "", "Frozen Product Joint plus one zero-initialized, one-hidden-layer concatenation residual. Product parameters are frozen.", "", f"Dataset metadata: `{_metadata_summary(metadata)}`."]
    sources = sorted({record["product_source"] for payload in mappings.values() for record in payload["records"]})
    lines += ["", "## Product reference provenance", "", *[f"- `{source}`" for source in sources]]
    lines += ["", "## Capacity and validation result", "", "| Mapping | Product params | Residual params | Residual ratio | J Product | J Nested | Δ Nested |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mapping, payload in mappings.items():
        val = payload["summary"]["valid"]
        lines.append(f"| `{mapping}` | {payload['records'][0]['product_params']:,} | {payload['records'][0]['residual_params']:,} | {payload['records'][0]['residual_ratio']:.3f} | {val['J_product']['mean']:.4f} ± {val['J_product']['std']:.4f} | {val['J_nested']['mean']:.4f} ± {val['J_nested']['std']:.4f} | {val['delta_nested']['mean']:.4f} ± {val['delta_nested']['std']:.4f} |")
    lines += ["", "## Seed-level validation result", "", "| Mapping | Seed | J Product | J Nested | Δ Nested |", "|---|---:|---:|---:|---:|"]
    for mapping, payload in mappings.items():
        for record in payload["records"]:
            lines.append(f"| `{mapping}` | {record['seed']} | {record['J_product']['valid']:.4f} | {record['J_nested']['valid']:.4f} | {record['delta_nested']['valid']:.4f} |")
    lines += ["", "## Training and residual health", "", "| Mapping | Seed | Initial val MSE | Best val MSE | Best epoch | Product R² val | Nested R² val | Residual norm val | Finite | Product grad max | Residual grad first |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for mapping, payload in mappings.items():
        for record in payload["records"]:
            lines.append(f"| `{mapping}` | {record['seed']} | {record['initial_val_mse']:.6f} | {record['best_val_mse']:.6f} | {record['best_epoch']} | {record['product_val_r2']:.4f} | {record['nested_val_r2']:.4f} | {record['residual_norm']:.4f} | {record['finite']} | {record['max_product_grad_norm']:.2e} | {record['first_residual_grad_norm']:.2e} |")
    lines += ["", "## Test result (descriptive)", "", "| Mapping | J Product | J Nested | Δ Nested |", "|---|---:|---:|---:|"]
    for mapping, payload in mappings.items():
        test = payload["summary"]["test"]
        lines.append(f"| `{mapping}` | {test['J_product']['mean']:.4f} ± {test['J_product']['std']:.4f} | {test['J_nested']['mean']:.4f} ± {test['J_nested']['std']:.4f} | {test['delta_nested']['mean']:.4f} ± {test['delta_nested']['std']:.4f} |")
    if decision:
        lines += ["", "## Interpretation", "", f"`{decision}`. Δ Nested measures additional predictable structure beyond the frozen Product Joint; it does not establish causal or PID synergy. Decisions use validation only."]
    path.write_text("\n".join(lines) + "\n")


def _write_combined(out, mosei, meld, decision, candidates, reason):
    lines = ["# Nested Joint-Class Headroom Decision", "", "## Research question", "", "Does a modest nonlinear residual on top of the frozen Product Joint reveal substantial additional cross-modal predictive headroom?", "", "## Validation result", "", "| Dataset | Mapping | J Product | J Nested | Δ Nested |", "|---|---|---:|---:|---:|"]
    for dataset, rows in (("MOSEI", mosei), ("MELD", meld)):
        for mapping, payload in rows.items():
            val = payload["summary"]["valid"]
            lines.append(f"| {dataset} | `{mapping}` | {val['J_product']['mean']:.4f} ± {val['J_product']['std']:.4f} | {val['J_nested']['mean']:.4f} ± {val['J_nested']['std']:.4f} | {val['delta_nested']['mean']:.4f} ± {val['delta_nested']['std']:.4f} |")
    lines += ["", "## Decision", "", f"**{decision}**", "", reason, f"MELD candidates: `{candidates}`.", "", "Product is frozen, the epoch-zero Product output is a validation candidate, and test metrics are descriptive only.", "", "## Scope", "", "No architecture sweep, optimizer sweep, representation change, JAD training, downstream task, synthetic reopen, or new dataset was run.", "", decision]
    (out / "NESTED_JOINT_CLASS_DECISION.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/hypothesis_class/v561"))
    parser.add_argument("--mosei-cache", type=Path, default=Path("results/mosei/identifiability_exact/mosei_pooled.npz"))
    parser.add_argument("--mosei-data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--meld-processed-dir", type=Path, default=Path("data/multibench/meld_download/processed"))
    parser.add_argument("--meld-annotation-dir", type=Path, default=Path("data/multibench/meld_download/annotations"))
    parser.add_argument("--meld-embedding", type=Path, default=Path("data/multibench/meld_download/processed/embedding.p"))
    parser.add_argument("--mosei-product-dir", type=Path, default=Path("results/mosei/identifiability_exact/checkpoints"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/hypothesis_class/v56"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--mosei-mappings", nargs="+", default=["VA_to_T"], choices=list(MOSEI_MAPPINGS))
    parser.add_argument("--meld-mappings", nargs="+", default=list(MELD_MAPPINGS), choices=list(MELD_MAPPINGS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    (args.out / "mosei").mkdir(parents=True, exist_ok=True)
    (args.out / "meld").mkdir(parents=True, exist_ok=True)
    mosei, mosei_meta = _load_data(args.mosei_data_dir, args.mosei_cache)
    mosei = _standardize(mosei)
    meld, meld_meta = load_meld(args.meld_processed_dir, args.meld_annotation_dir, args.meld_embedding)
    rows = {"MOSEI": {}, "MELD": {}}
    for dataset, splits, mappings, metadata, baseline_file in (
        ("MOSEI", mosei, args.mosei_mappings, mosei_meta, args.baseline_dir / "mosei" / "mosei_h0.json"),
        ("MELD", meld, args.meld_mappings, meld_meta, args.baseline_dir / "meld" / "meld_h0.json"),
    ):
        for mapping_name in mappings:
            mapping_rows = []
            for seed in args.seeds:
                mapping_rows.append(_run_mapping(dataset, mapping_name, MOSEI_MAPPINGS[mapping_name] if dataset == "MOSEI" else MELD_MAPPINGS[mapping_name], splits, seed, args, baseline_file))
            rows[dataset][mapping_name] = {"mapping": mapping_name, "records": mapping_rows, "summary": _aggregate(mapping_rows)}
            (args.out / dataset.lower() / f"{dataset.lower()}_nested_h0.json").write_text(json.dumps({"dataset": dataset, "metadata": metadata, "mappings": rows[dataset]}, indent=2))
        _write_dataset_report(args.out / dataset.lower() / f"{dataset}_NESTED_HEADROOM_REPORT.md", dataset, rows[dataset], metadata)
    decision, candidates, reason = _decision(rows["MELD"])
    summary = {
        "decision": decision,
        "decision_reason": reason,
        "meld_candidates": candidates,
        "mosei": {"metadata": mosei_meta, "mappings": rows["MOSEI"]},
        "meld": {"metadata": meld_meta, "mappings": rows["MELD"]},
        "protocol": {"seeds": args.seeds, "validation_selection": "MSE with epoch-zero Product baseline", "test_used_for_decision": False, "architecture_search": False, "jad_trained": False, "task_g2_trained": False},
    }
    (args.out / "nested_h0_summary.json").write_text(json.dumps(summary, indent=2))
    _write_combined(args.out, rows["MOSEI"], rows["MELD"], decision, candidates, reason)
    print(json.dumps({"decision": decision, "candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()
