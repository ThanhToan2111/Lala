"""v5.3 T0: frozen-MOSEI sentiment task-headroom audit."""

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
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch import nn

from src.experiments.multibench.mosei_identifiability import _device, _load_data, _seed, _standardize


MODELS = ("linear", "additive", "joint_product", "concat_mlp")
HIDDEN = 16
RANK = 16


class LinearTaskProbe(nn.Module):
    def __init__(self, source_dim: int):
        super().__init__()
        self.output = nn.Linear(source_dim, 1)

    def forward(self, vision, audio):
        return self.output(torch.cat((vision, audio), dim=1)).squeeze(1)


class AdditiveTaskPredictor(nn.Module):
    def __init__(self, vision_dim: int, audio_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.vision = nn.Sequential(nn.Linear(vision_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.audio = nn.Sequential(nn.Linear(audio_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, vision, audio):
        return (self.vision(vision) + self.audio(audio)).squeeze(1)


class JointTaskPredictor(nn.Module):
    def __init__(self, vision_dim: int, audio_dim: int, rank: int = RANK):
        super().__init__()
        self.vision = nn.Linear(vision_dim, rank, bias=False)
        self.audio = nn.Linear(audio_dim, rank, bias=False)
        self.output = nn.Linear(3 * rank, 1)

    def forward(self, vision, audio):
        vision_factor = self.vision(vision)
        audio_factor = self.audio(audio)
        return self.output(torch.cat((vision_factor, audio_factor, vision_factor * audio_factor), dim=1)).squeeze(1)


class ConcatMLPTaskPredictor(nn.Module):
    def __init__(self, source_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(source_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, vision, audio):
        return self.network(torch.cat((vision, audio), dim=1)).squeeze(1)


def parameter_counts(vision_dim: int, audio_dim: int) -> dict:
    source_dim = vision_dim + audio_dim
    counts = {
        "linear": (source_dim + 1),
        "additive": (vision_dim + 1) * HIDDEN + (HIDDEN + 1) + (audio_dim + 1) * HIDDEN + (HIDDEN + 1),
        "joint_product": (vision_dim + audio_dim) * RANK + 3 * RANK + 1,
        "concat_mlp": (source_dim + 1) * HIDDEN + (HIDDEN + 1),
    }
    reference = counts["additive"]
    mismatch = {name: 100.0 * abs(value - reference) / reference for name, value in counts.items()}
    return {"counts": counts, "mismatch_vs_additive_percent": mismatch, "hidden": HIDDEN, "rank": RANK}


def _build(name: str, vision_dim: int, audio_dim: int):
    source_dim = vision_dim + audio_dim
    if name == "linear":
        return LinearTaskProbe(source_dim)
    if name == "additive":
        return AdditiveTaskPredictor(vision_dim, audio_dim)
    if name == "joint_product":
        return JointTaskPredictor(vision_dim, audio_dim)
    if name == "concat_mlp":
        return ConcatMLPTaskPredictor(source_dim)
    raise ValueError(name)


def _metrics(labels, logits):
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "auc": float(roc_auc_score(labels, probabilities)),
        "bce": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def _fit_model(name, splits, labels, seed, epochs, patience, batch_size):
    _seed(seed)
    vision_dim, audio_dim = splits["train"]["vision"].shape[1], splits["train"]["audio"].shape[1]
    model = _build(name, vision_dim, audio_dim).to(_device())
    tensors = {
        split: (
            torch.from_numpy(data["vision"]).to(_device()),
            torch.from_numpy(data["audio"]).to(_device()),
            torch.from_numpy(labels[split].astype(np.float32)).to(_device()),
        )
        for split, data in splits.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1000)
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(epochs):
        model.train()
        train_vision, train_audio, train_labels = tensors["train"]
        for indices in torch.randperm(len(train_labels), generator=generator).split(batch_size):
            indices = indices.to(_device())
            loss = F.binary_cross_entropy_with_logits(model(train_vision[indices], train_audio[indices]), train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_vision, valid_audio, valid_labels = tensors["valid"]
            valid_loss = float(F.binary_cross_entropy_with_logits(model(valid_vision, valid_audio), valid_labels).item())
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
        for split, data in splits.items():
            logits[split] = model(torch.from_numpy(data["vision"]), torch.from_numpy(data["audio"])).numpy()
    metrics = {split: _metrics(labels[split], logits[split]) for split in splits}
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_epoch": best_epoch + 1,
        "train_loss": metrics["train"]["bce"],
        "val_loss": metrics["valid"]["bce"],
        "test_loss": metrics["test"]["bce"],
        "train": metrics["train"],
        "valid": metrics["valid"],
        "test": metrics["test"],
    }


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def _paired(records, left, right, split):
    metrics = ("accuracy", "macro_f1", "auc")
    result = {}
    for metric in metrics:
        values = np.asarray([r["models"][left][split][metric] - r["models"][right][split][metric] for r in records])
        result[metric] = _difference(values)
    values = np.asarray([r["models"][right][split]["bce"] - r["models"][left][split]["bce"] for r in records])
    result["bce_improvement"] = _difference(values)
    return result


def _difference(values):
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half], "per_seed": values.tolist()}


def _summary(records):
    return {
        name: {
            split: {metric: _mean_std([r["models"][name][split][metric] for r in records]) for metric in ("accuracy", "macro_f1", "auc", "bce")}
            for split in ("train", "valid", "test")
        }
        for name in MODELS
    }


def _reliable(headroom):
    return all(headroom[metric]["mean"] > 0 and sum(value > 0 for value in headroom[metric]["per_seed"]) >= 4 for metric in headroom)


def _decision(paired):
    joint = paired["joint_minus_additive"]["valid"]
    mlp = paired["mlp_minus_additive"]["valid"]
    if _reliable(joint):
        return "EXPLICIT_JOINT_HEADROOM"
    if _reliable(mlp):
        return "GENERIC_NONLINEAR_HEADROOM_ONLY"
    # No method is promoted when validation signs are inconsistent.
    return "INCONCLUSIVE"


def _fmt(value, scale=1.0):
    return f"{value['mean'] * scale:.3f} ± {value['std'] * scale:.3f}"


def _write_report(out_dir, payload):
    summary, paired, decision = payload["summary"], payload["paired"], payload["decision"]
    lines = [
        "# MOSEI v5.3 T0 task-headroom audit",
        "",
        "Run date: 2026-09-19. Frozen canonical raw MOSEI, pooled vision+audio → sentiment, five seeds.",
        "",
        "## Protocol",
        "",
        "The canonical vision/audio features are frozen. T0 trains only task heads on `Y=1[sentiment>0]`; it does not use `h_D2`, does not retrain cross-modal predictors, and uses validation-only BCE for early stopping. All models use AdamW (`1e-3`, weight decay `1e-4`), batch size 512, maximum 80 epochs, patience 8.",
        "",
        "## Capacity audit",
        "",
        "| Model | Parameters | Mismatch vs additive |",
        "|---|---:|---:|",
    ]
    counts = payload["architecture"]["counts"]
    mismatch = payload["architecture"]["mismatch_vs_additive_percent"]
    labels = {"linear": "Linear VA", "additive": "Additive", "joint_product": "Joint Product", "concat_mlp": "Generic MLP"}
    for name in MODELS:
        lines.append(f"| {labels[name]} | {counts[name]:,} | {mismatch[name]:.3f}% |")
    lines += [
        "",
        "## Main five-seed test result",
        "",
        "| Model | Params | Accuracy (%) | Macro F1 (%) | AUC (%) | BCE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        row = summary[name]["test"]
        lines.append(f"| {labels[name]} | {counts[name]:,} | {_fmt(row['accuracy'], 100)} | {_fmt(row['macro_f1'], 100)} | {_fmt(row['auc'], 100)} | {_fmt(row['bce'])} |")
    lines += [
        "",
        "## Paired headroom",
        "",
        "Positive BCE improvement means the compared model has lower BCE than additive. Values are mean ± sample standard deviation; 95% CIs and per-seed differences are in JSON.",
        "",
        "| Comparison | Accuracy Δ (pp) | F1 Δ (pp) | AUC Δ (pp) | BCE improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, title in (("joint_minus_additive", "Joint − Additive"), ("mlp_minus_additive", "MLP − Additive")):
        row = paired[name]["test"]
        lines.append(f"| {title} | {_fmt(row['accuracy'], 100)} | {_fmt(row['macro_f1'], 100)} | {_fmt(row['auc'], 100)} | {_fmt(row['bce_improvement'])} |")
    lines += [
        "",
        "## Validation decision",
        "",
        f"T0 decision: **{decision}**.",
        "",
        "The decision is based only on validation paired differences. A headroom candidate must have positive mean and positive sign in at least 4/5 seeds for accuracy, macro F1, AUC, and BCE improvement. This prevents a single favorable test metric from opening a new method branch.",
        "",
        "## Interpretation",
        "",
        "T0 separates task-relevant non-additivity (`VA→sentiment`) from canonical JAD identifiability (`VA→T`). It does not claim causality, PID synergy, or exact interaction decomposition.",
        "",
        "## Artifacts",
        "",
        "- `t0_task_headroom.json` — per-model/per-seed train, validation, and test metrics plus paired differences.",
        "- `mosei_task_headroom.py` — reproducible T0 runner.",
        "",
    ]
    (out_dir / "T0_TASK_HEADROOM_AUDIT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--cache", type=Path, default=Path("results/mosei/identifiability_exact/mosei_pooled.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/mosei/v53"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits, metadata = _load_data(args.data_dir, args.cache)
    splits = _standardize(splits)
    labels = {split: (data["sentiment"] > 0).astype(np.int64) for split, data in splits.items()}
    architecture = parameter_counts(splits["train"]["vision"].shape[1], splits["train"]["audio"].shape[1])
    records = []
    for seed in args.seeds:
        models = {name: _fit_model(name, splits, labels, seed, args.epochs, args.patience, args.batch_size) for name in MODELS}
        record = {"seed": seed, "models": models}
        records.append(record)
        (args.out / f"t0_seed_{seed}.json").write_text(json.dumps(record, indent=2))
    paired = {
        comparison: {split: _paired(records, left, right, split) for split in ("valid", "test")}
        for comparison, left, right in (("joint_minus_additive", "joint_product", "additive"), ("mlp_minus_additive", "concat_mlp", "additive"))
    }
    payload = {
        "protocol": {"source": "canonical_raw_mosei", "mapping": "VA_to_sentiment", "seeds": args.seeds, "selection": "validation_bce_only", "data_metadata": metadata},
        "architecture": architecture,
        "decision": _decision(paired),
        "summary": _summary(records),
        "paired": paired,
        "records": records,
    }
    (args.out / "t0_task_headroom.json").write_text(json.dumps(payload, indent=2))
    _write_report(args.out, payload)


if __name__ == "__main__":
    main()
