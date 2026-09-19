"""v5.4 MUStARD S1/S2 two-gate screening; no JAD training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t
from sklearn.metrics import r2_score
from torch import nn

from src.experiments.multibench.mosei_identifiability import (
    _device,
    _predict,
    _seed,
    capacity_config,
    fit_additive_predictor,
    fit_joint_predictor,
)
from src.experiments.multibench.mosei_task_headroom import MODELS, _build, _metrics, parameter_counts


MAPPINGS = {
    "VA_to_T": ("vision", "audio", "text"),
    "VT_to_A": ("vision", "text", "audio"),
    "AT_to_V": ("audio", "text", "vision"),
}
FEATURES = ("vision", "audio", "text")


def _pool(sequence: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(sequence.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.abs(value).sum(axis=-1) > 1e-8
    count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return (value * mask[..., None]).sum(axis=1) / count


def load_mustard(path: Path):
    raw = pickle.loads(path.read_bytes())
    splits, audit = {}, {"source": str(path), "sequence_handling": "masked_mean_nonzero_frames"}
    all_ids = []
    for split in ("train", "valid", "test"):
        part = raw[split]
        ids = list(part["id"])
        labels = (np.asarray(part["labels"], dtype=np.float32).reshape(-1) > 0).astype(np.int64)
        pooled = {name: _pool(part[name]) for name in FEATURES}
        splits[split] = {**pooled, "labels": labels, "ids": ids}
        all_ids.extend(ids)
        audit.setdefault("splits", {})[split] = {
            "count": len(ids),
            "class_counts": {str(label): int((labels == label).sum()) for label in (0, 1)},
            "feature_dims": {name: int(pooled[name].shape[1]) for name in FEATURES},
            "zero_pooled_vectors": {name: int(np.sum(np.linalg.norm(pooled[name], axis=1) == 0)) for name in FEATURES},
            "nan_or_inf_after_pool": {name: int((~np.isfinite(pooled[name])).sum()) for name in FEATURES},
            "raw_shape": {name: list(part[name].shape) for name in FEATURES},
        }
    audit["unique_ids"] = len(set(all_ids)) == len(all_ids)
    audit["duplicate_ids"] = len(all_ids) - len(set(all_ids))
    audit["feature_dims"] = audit["splits"]["train"]["feature_dims"]
    stats = {}
    for name in FEATURES:
        mean = splits["train"][name].mean(0)
        std = np.maximum(splits["train"][name].std(0), 1e-6)
        stats[name] = {"mean": mean, "std": std}
        for split in splits:
            splits[split][name] = ((splits[split][name] - mean) / std).astype(np.float32)
    audit["standardization"] = "train mean/std per modality"
    return splits, audit


def _g1_one(splits, mapping, seed, args):
    source_a, source_b, target = MAPPINGS[mapping]
    train, valid = splits["train"], splits["valid"]
    config = capacity_config(train[source_a].shape[1], train[source_b].shape[1], train[target].shape[1])
    kwargs = {"batch_size": args.g1_batch_size, "epochs": args.g1_epochs, "patience": args.g1_patience, "lr": 1e-3, "weight_decay": 1e-4}
    additive, additive_meta = fit_additive_predictor(
        train[source_a], train[source_b], train[target], valid[source_a], valid[source_b], valid[target], config, seed, **kwargs,
    )
    joint, joint_meta = fit_joint_predictor(
        train[source_a], train[source_b], train[target], valid[source_a], valid[source_b], valid[target], config, seed + 1, **kwargs,
    )
    metrics = {}
    for split, data in splits.items():
        add = _predict(additive, data[source_a], data[source_b])
        jnt = _predict(joint, data[source_a], data[source_b])
        add_r2, joint_r2 = r2_score(data[target], add, multioutput="variance_weighted"), r2_score(data[target], jnt, multioutput="variance_weighted")
        metrics[split] = {"add_r2": float(add_r2), "joint_r2": float(joint_r2), "joint_advantage": float(joint_r2 - add_r2)}
    return {"mapping": mapping, "seed": seed, "sources": [source_a, source_b], "target": target, "capacity": config, "predictors": {"additive": additive_meta, "joint": joint_meta}, "metrics": metrics}


def _fit_task(name, left, right, labels, seed, epochs, patience, batch_size):
    _seed(seed)
    device = _device()
    model = _build(name, left["train"].shape[1], right["train"].shape[1]).to(device)
    tensors = {
        split: (torch.from_numpy(left[split]).to(device), torch.from_numpy(right[split]).to(device), torch.from_numpy(labels[split].astype(np.float32)).to(device))
        for split in left
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1000)
    best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    for epoch in range(epochs):
        model.train()
        left_train, right_train, y_train = tensors["train"]
        for indices in torch.randperm(len(y_train), generator=generator).split(batch_size):
            indices = indices.to(device)
            loss = F.binary_cross_entropy_with_logits(model(left_train[indices], right_train[indices]), y_train[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            left_valid, right_valid, y_valid = tensors["valid"]
            valid_loss = float(F.binary_cross_entropy_with_logits(model(left_valid, right_valid), y_valid).item())
        if valid_loss < best_loss - 1e-7:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), valid_loss, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.cpu().eval()
    logits = {}
    with torch.no_grad():
        for split in left:
            logits[split] = model(torch.from_numpy(left[split]), torch.from_numpy(right[split])).numpy()
    metrics = {split: _metrics(labels[split], logits[split]) for split in left}
    return {"parameter_count": sum(p.numel() for p in model.parameters()), "best_validation_epoch": best_epoch + 1, "train_loss": metrics["train"]["bce"], "val_loss": metrics["valid"]["bce"], "test_loss": metrics["test"]["bce"], "train": metrics["train"], "valid": metrics["valid"], "test": metrics["test"]}


def _diff(values):
    values = np.asarray(values, dtype=np.float64)
    mean, std = float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half], "per_seed": values.tolist()}


def _task_headroom(records, left_model="joint_product"):
    out = {}
    for split in ("valid", "test"):
        out[split] = {}
        for metric in ("accuracy", "macro_f1", "auc"):
            out[split][metric] = _diff([r["models"][left_model][split][metric] - r["models"]["additive"][split][metric] for r in records])
        out[split]["bce_improvement"] = _diff([r["models"]["additive"][split]["bce"] - r["models"][left_model][split]["bce"] for r in records])
    return out


def _g2_status(headroom):
    valid = headroom["valid"]
    supported = [metric for metric in valid if valid[metric]["mean"] > 0 and sum(value > 0 for value in valid[metric]["per_seed"]) >= 2]
    if len(supported) >= 3:
        return "PASS_3SEED_PROMISING"
    if len(supported) == 0:
        return "FAIL"
    return "WEAK_UNSTABLE"


def _g1_status(records):
    values = [r["metrics"]["valid"]["joint_advantage"] for r in records]
    return "PASS" if all(value > 0.01 for value in values) else ("WEAK_UNSTABLE" if any(value > 0.01 for value in values) else "FAIL")


def _final_type(g1_status, g2_status):
    if g1_status == "PASS" and g2_status == "PASS_3SEED_PROMISING":
        return "TYPE_IV"
    if g1_status == "PASS":
        return "TYPE_II"
    if g2_status == "PASS_3SEED_PROMISING":
        return "TYPE_III"
    if g1_status == "FAIL" and g2_status == "FAIL":
        return "TYPE_I"
    return "INCONCLUSIVE"


def _write_reports(out, audit, g1, g2, matrix):
    g1_path = out / "g1_cross_modal"
    g2_path = out / "g2_task_headroom"
    g1_path.mkdir(parents=True, exist_ok=True)
    g2_path.mkdir(parents=True, exist_ok=True)
    g1_lines = ["# MUStARD G1 cross-modal identifiability", "", "Processed MultiBench MUStARD, masked-mean pooled features, fixed split, three screening seeds.", "", "| Mapping | Capacity | Val J | Test J | G1 |", "|---|---:|---:|---:|---|"]
    for mapping, row in g1.items():
        vals = row["summary"]["valid"]["joint_advantage"]
        test = row["summary"]["test"]["joint_advantage"]
        g1_lines.append(f"| `{mapping}` | {row['capacity']['additive_parameters']:,}/{row['capacity']['joint_parameters']:,} ({row['capacity']['parameter_difference_percent']:.3f}%) | {vals['mean']:.4f} ± {vals['std']:.4f} | {test['mean']:.4f} ± {test['std']:.4f} | **{row['status']}** |")
    g1_lines += ["", "G1 pass requires validation J > 0.01 in all three screening seeds. Test J is reporting-only.", ""]
    (g1_path / "MUSTARD_G1_CROSS_MODAL_REPORT.md").write_text("\n".join(g1_lines))
    g2_lines = ["# MUStARD G2 task joint-headroom", "", "Frozen pooled V/A/T features; sarcasm label; three seeds; validation BCE early stopping.", "", "| Pair | Joint−Add Acc val | F1 val | AUC val | BCE improvement val | G2 |", "|---|---:|---:|---:|---:|---|"]
    for pair, row in g2.items():
        h = row["headroom"]
        g2_lines.append(f"| `{pair}` | {h['valid']['accuracy']['mean']*100:.3f} ± {h['valid']['accuracy']['std']*100:.3f} pp | {h['valid']['macro_f1']['mean']*100:.3f} ± {h['valid']['macro_f1']['std']*100:.3f} pp | {h['valid']['auc']['mean']*100:.3f} ± {h['valid']['auc']['std']*100:.3f} pp | {h['valid']['bce_improvement']['mean']:.4f} ± {h['valid']['bce_improvement']['std']:.4f} | **{row['status']}** |")
    g2_lines += ["", "G2 is a screening result; no JAD embedding is trained. A promising three-seed result requires confirmation before R2.", ""]
    (g2_path / "MUSTARD_G2_TASK_HEADROOM_REPORT.md").write_text("\n".join(g2_lines))
    decision_lines = ["# MUStARD two-gate decision", "", "## Data audit", "", f"Source: `{audit['source']}`; split sizes: " + ", ".join(f"{s}={audit['splits'][s]['count']}" for s in ("train", "valid", "test")) + ".", f"Feature dimensions: {audit['feature_dims']}; sequence handling: `{audit['sequence_handling']}`; unique IDs: `{audit['unique_ids']}`.", "", "## Two-gate matrix", "", "| Pair / mapping | G1 | G2 | Final type |", "|---|---|---|---|"]
    for pair, mapping in (("VA", "VA_to_T"), ("VT", "VT_to_A"), ("AT", "AT_to_V")):
        g1_status, g2_status = g1[mapping]["status"], g2[pair]["status"]
        final = _final_type(g1_status, g2_status)
        decision_lines.append(f"| `{pair}` / `{mapping}` | {g1_status} | {g2_status} | **{final}** |")
    decision_lines += ["", "## Hard decision", "", "No JAD training is allowed from this screening artifact unless the same pair reaches TYPE_IV after any permitted frozen-seed confirmation. This run does not change JAD, loss, rank, purifier, or architecture.", "", "MUStARD is small; preserve per-seed values and do not rank pairs by test performance.", ""]
    (out / "MUSTARD_TWO_GATE_DECISION.md").write_text("\n".join(decision_lines))


def _summary_g1(records):
    return {split: {metric: _diff([r["metrics"][split][metric] for r in records]) for metric in ("add_r2", "joint_r2", "joint_advantage")} for split in ("train", "valid", "test")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/multibench/mustard_download/sarcasm.pkl"))
    parser.add_argument("--out", type=Path, default=Path("results/mustard/v54"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--g1-epochs", type=int, default=100)
    parser.add_argument("--g1-patience", type=int, default=8)
    parser.add_argument("--g1-batch-size", type=int, default=128)
    parser.add_argument("--g2-epochs", type=int, default=80)
    parser.add_argument("--g2-patience", type=int, default=8)
    parser.add_argument("--g2-batch-size", type=int, default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits, audit = load_mustard(args.data)
    (args.out / "data_audit").mkdir(parents=True, exist_ok=True)
    (args.out / "data_audit" / "mustard_data_audit.json").write_text(json.dumps(audit, indent=2))
    (args.out / "data_audit" / "MUSTARD_DATA_AUDIT.md").write_text("\n".join([
        "# MUStARD data audit", "", f"Canonical source: `{args.data}`.", "", f"Feature dimensions: `{audit['feature_dims']}`.", f"Sequence handling: `{audit['sequence_handling']}`.", f"Unique sample IDs across splits: `{audit['unique_ids']}`; duplicates: `{audit['duplicate_ids']}`.", "", "| Split | N | Class 0 | Class 1 |", "|---|---:|---:|---:|", *[f"| {s} | {audit['splits'][s]['count']} | {audit['splits'][s]['class_counts']['0']} | {audit['splits'][s]['class_counts']['1']} |" for s in ("train", "valid", "test")], ""] + [f"- `{s}` zero pooled vectors: {audit['splits'][s]['zero_pooled_vectors']}" for s in ("train", "valid", "test")]))
    g1, g2 = {}, {}
    for mapping in MAPPINGS:
        records = [_g1_one(splits, mapping, seed, args) for seed in args.seeds]
        g1[mapping] = {"status": _g1_status(records), "capacity": records[0]["capacity"], "summary": _summary_g1(records), "records": records}
    labels = {split: splits[split]["labels"] for split in splits}
    for pair, (left_name, right_name, _) in (("VA", MAPPINGS["VA_to_T"]), ("VT", MAPPINGS["VT_to_A"]), ("AT", MAPPINGS["AT_to_V"])).__iter__():
        left = {split: splits[split][left_name] for split in splits}
        right = {split: splits[split][right_name] for split in splits}
        records = [{"seed": seed, "models": {name: _fit_task(name, left, right, labels, seed + index * 100, args.g2_epochs, args.g2_patience, args.g2_batch_size) for index, name in enumerate(MODELS)}} for seed in args.seeds]
        headroom = _task_headroom(records)
        g2[pair] = {"sources": [left_name, right_name], "architecture": parameter_counts(left["train"].shape[1], right["train"].shape[1]), "status": _g2_status(headroom), "headroom": headroom, "records": records}
    matrix = {}
    for pair, mapping in (("VA", "VA_to_T"), ("VT", "VT_to_A"), ("AT", "AT_to_V")):
        matrix[pair] = {"mapping": mapping, "g1": g1[mapping]["status"], "g2": g2[pair]["status"], "final_type": _final_type(g1[mapping]["status"], g2[pair]["status"])}
    payload = {"dataset": "MUStARD", "source": str(args.data), "audit": audit, "g1": g1, "g2": g2, "two_gate_matrix": matrix, "protocol": {"seeds": args.seeds, "g1_validation_gate": "J_val > 0.01 in every screening seed", "g2_selection": "validation BCE; no test-based gate", "jad_trained": False}}
    (args.out / "mustard_two_gate.json").write_text(json.dumps(payload, indent=2))
    _write_reports(args.out, audit, g1, g2, matrix)


if __name__ == "__main__":
    main()
