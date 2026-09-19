"""v5.5 MELD utterance-level G1/G2 screening; no JAD training."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.experiments.multibench.mosei_identifiability import (
    _device,
    _predict,
    _seed,
    capacity_config,
    fit_additive_predictor,
    fit_joint_predictor,
)


FEATURES = ("vision", "audio", "text")
MAPPINGS = {
    "VA_to_T": ("vision", "audio", "text"),
    "VT_to_A": ("vision", "text", "audio"),
    "AT_to_V": ("audio", "text", "vision"),
}
MODELS = ("linear", "additive", "joint_product", "concat_mlp")
EMOTION_LABELS = ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger")
LABEL_TO_ID = {name: index for index, name in enumerate(EMOTION_LABELS)}
NUM_CLASSES = len(EMOTION_LABELS)
HIDDEN = 16
RANK = 16


def _masked_mean(sequence: np.ndarray, length: int | None = None) -> np.ndarray:
    """Mean only over valid sequence positions; invalid values are finite-safe."""
    value = np.nan_to_num(np.asarray(sequence, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError(f"expected a non-empty [L,D] sequence, got {value.shape}")
    valid_length = value.shape[0] if length is None else int(length)
    if not 0 < valid_length <= value.shape[0]:
        raise ValueError(f"invalid sequence length {valid_length} for {value.shape}")
    return value[:valid_length].mean(axis=0)


def _pool(sequence: np.ndarray) -> np.ndarray:
    """Compatibility helper used by tests and the loader for one utterance."""
    return _masked_mean(sequence)


def _read_annotations(annotation_dir: Path, split: str) -> dict[str, dict]:
    filename = "dev_sent_emo.csv" if split == "valid" else f"{split}_sent_emo.csv"
    with (annotation_dir / filename).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            f"{row['Dialogue_ID']}_{row['Utterance_ID']}": row
            for row in rows
        }


def _feature_path(processed_dir: Path, split: str) -> Path:
    return processed_dir / ("dev.pkl" if split == "valid" else f"{split}.pkl")


def load_meld(processed_dir: Path, annotation_dir: Path, embedding_path: Path):
    """Load official MELD annotations plus MM-Align's frozen utterance features."""
    embedding = np.asarray(pickle.loads(embedding_path.read_bytes()), dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[1] != 300:
        raise ValueError(f"expected a [vocab,300] embedding table, got {embedding.shape}")

    splits: dict[str, dict] = {}
    audit = {
        "dataset": "MELD",
        "dataset_version": "official declare-lab/MELD annotation CSVs plus MM-Align processed MELD pickle release",
        "task": "7-class emotion recognition",
        "feature_source": str(processed_dir),
        "annotation_source": str(annotation_dir),
        "embedding_source": str(embedding_path),
        "source_urls": {
            "annotations": "https://github.com/declare-lab/MELD/tree/master/data/MELD",
            "processed_features": "https://github.com/declare-lab/MM-Align",
            "processed_feature_release": "https://drive.google.com/file/d/1RjrYSMpXxg_6r_nUQaysaPyMsldLpMcb/view",
        },
        "sequence_handling": "masked_mean over the complete utterance sequence; no dialogue context",
        "text_representation": "provided 300-dimensional embedding.p lookup followed by masked mean",
        "label_mapping": LABEL_TO_ID,
        "embedding_shape": list(embedding.shape),
        "splits": {},
    }
    all_sample_ids: list[str] = []

    for split in ("train", "valid", "test"):
        raw = pickle.loads(_feature_path(processed_dir, split).read_bytes())
        annotations = _read_annotations(annotation_dir, split)
        feature_ids = list(raw)
        annotation_ids = list(annotations)
        missing_features = sorted(set(annotation_ids) - set(feature_ids))
        extra_features = sorted(set(feature_ids) - set(annotation_ids))
        rows = {name: [] for name in FEATURES}
        labels, ids, speakers, dialogue_ids, utterance_ids = [], [], [], [], []
        feature_label_mismatches = []
        sequence_lengths = []

        for local_id, item in raw.items():
            if local_id not in annotations:
                raise ValueError(f"feature id {split}:{local_id} has no canonical annotation")
            annotation = annotations[local_id]
            label_name = annotation["Emotion"].strip().lower()
            if label_name not in LABEL_TO_ID:
                raise ValueError(f"unknown MELD emotion {label_name!r}")
            label = LABEL_TO_ID[label_name]
            if int(item["label"]) != label:
                feature_label_mismatches.append(local_id)

            token_ids = np.asarray(item["token_ids"], dtype=np.int64).reshape(-1)
            if len(token_ids) == 0 or token_ids.min() < 0 or token_ids.max() >= len(embedding):
                raise ValueError(f"invalid token ids at {split}:{local_id}")
            audio = np.asarray(item["audio_features"])
            video = np.asarray(item["video_features"])
            if audio.ndim != 2 or video.ndim != 2 or len(token_ids) != len(audio) or len(token_ids) != len(video):
                raise ValueError(f"unaligned modality sequence at {split}:{local_id}")
            rows["text"].append(_masked_mean(embedding[token_ids], len(token_ids)))
            rows["audio"].append(_masked_mean(audio, len(token_ids)))
            rows["vision"].append(_masked_mean(video, len(token_ids)))
            labels.append(label)
            sample_id = f"{split}:{local_id}"
            ids.append(sample_id)
            speakers.append(annotation["Speaker"])
            dialogue_ids.append(int(annotation["Dialogue_ID"]))
            utterance_ids.append(int(annotation["Utterance_ID"]))
            sequence_lengths.append(len(token_ids))

        labels_array = np.asarray(labels, dtype=np.int64)
        pooled = {name: np.asarray(rows[name], dtype=np.float32) for name in FEATURES}
        splits[split] = {
            **pooled,
            "labels": labels_array,
            "ids": ids,
            "speakers": speakers,
            "dialogue_ids": np.asarray(dialogue_ids, dtype=np.int64),
            "utterance_ids": np.asarray(utterance_ids, dtype=np.int64),
        }
        all_sample_ids.extend(ids)
        audit["splits"][split] = {
            "count": len(ids),
            "annotation_count": len(annotation_ids),
            "feature_count": len(feature_ids),
            "missing_feature_ids": missing_features,
            "extra_feature_ids": extra_features,
            "class_counts": {str(index): int((labels_array == index).sum()) for index in range(NUM_CLASSES)},
            "class_names": {EMOTION_LABELS[index]: int((labels_array == index).sum()) for index in range(NUM_CLASSES)},
            "speaker_count": len(set(speakers)),
            "dialogue_count": len(set(dialogue_ids)),
            "feature_dims": {name: int(pooled[name].shape[1]) for name in FEATURES},
            "missing_modalities": {name: 0 for name in FEATURES},
            "sequence_length": {
                "min": int(min(sequence_lengths)),
                "mean": float(np.mean(sequence_lengths)),
                "max": int(max(sequence_lengths)),
            },
            "zero_pooled_vectors": {name: int(np.sum(np.linalg.norm(pooled[name], axis=1) == 0)) for name in FEATURES},
            "nan_or_inf_after_pool": {name: int((~np.isfinite(pooled[name])).sum()) for name in FEATURES},
            "feature_label_mismatch_ids": feature_label_mismatches,
            "sample_ids": ids,
            "dialogue_ids": dialogue_ids,
            "utterance_ids": utterance_ids,
            "speaker_ids": speakers,
        }

    audit["unique_sample_ids_across_splits"] = len(all_sample_ids) == len(set(all_sample_ids))
    audit["duplicate_sample_ids_across_splits"] = len(all_sample_ids) - len(set(all_sample_ids))
    audit["feature_dims"] = audit["splits"]["train"]["feature_dims"]
    audit["standardization"] = "train split mean/std per modality; validation and test never affect statistics"
    return _standardize(splits), audit


def _standardize(splits: dict) -> dict:
    result = {}
    stats = {
        name: (splits["train"][name].mean(axis=0), np.maximum(splits["train"][name].std(axis=0), 1e-6))
        for name in FEATURES
    }
    for split, data in splits.items():
        result[split] = dict(data)
        for name in FEATURES:
            mean, std = stats[name]
            result[split][name] = ((data[name] - mean) / std).astype(np.float32)
    return result


class LinearTaskProbe(nn.Module):
    def __init__(self, source_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.output = nn.Linear(source_dim, num_classes)

    def forward(self, left, right):
        return self.output(torch.cat((left, right), dim=1))


class AdditiveTaskPredictor(nn.Module):
    def __init__(self, left_dim: int, right_dim: int, hidden: int = HIDDEN, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.left = nn.Sequential(nn.Linear(left_dim, hidden), nn.GELU(), nn.Linear(hidden, num_classes))
        self.right = nn.Sequential(nn.Linear(right_dim, hidden), nn.GELU(), nn.Linear(hidden, num_classes))

    def forward(self, left, right):
        return self.left(left) + self.right(right)


class JointTaskPredictor(nn.Module):
    def __init__(self, left_dim: int, right_dim: int, rank: int = RANK, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.left = nn.Linear(left_dim, rank, bias=False)
        self.right = nn.Linear(right_dim, rank, bias=False)
        self.output = nn.Linear(3 * rank, num_classes)

    def forward(self, left, right):
        left_factor, right_factor = self.left(left), self.right(right)
        return self.output(torch.cat((left_factor, right_factor, left_factor * right_factor), dim=1))


class ConcatMLPTaskPredictor(nn.Module):
    def __init__(self, source_dim: int, hidden: int = HIDDEN, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(source_dim, hidden), nn.GELU(), nn.Linear(hidden, num_classes))

    def forward(self, left, right):
        return self.network(torch.cat((left, right), dim=1))


def parameter_counts(left_dim: int, right_dim: int) -> dict:
    source_dim = left_dim + right_dim
    counts = {
        "linear": (source_dim + 1) * NUM_CLASSES,
        "additive": ((left_dim + 1) * HIDDEN + (HIDDEN + 1) * NUM_CLASSES)
        + ((right_dim + 1) * HIDDEN + (HIDDEN + 1) * NUM_CLASSES),
        "joint_product": left_dim * RANK + right_dim * RANK + (3 * RANK + 1) * NUM_CLASSES,
        "concat_mlp": (source_dim + 1) * HIDDEN + (HIDDEN + 1) * NUM_CLASSES,
    }
    reference = counts["additive"]
    return {
        "counts": counts,
        "mismatch_vs_additive_percent": {name: 100.0 * abs(value - reference) / reference for name, value in counts.items()},
        "hidden": HIDDEN,
        "rank": RANK,
        "num_classes": NUM_CLASSES,
    }


def _build(name: str, left_dim: int, right_dim: int):
    if name == "linear":
        return LinearTaskProbe(left_dim + right_dim)
    if name == "additive":
        return AdditiveTaskPredictor(left_dim, right_dim)
    if name == "joint_product":
        return JointTaskPredictor(left_dim, right_dim)
    if name == "concat_mlp":
        return ConcatMLPTaskPredictor(left_dim + right_dim)
    raise ValueError(name)


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    logits_tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    labels_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    probabilities = torch.softmax(logits_tensor, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", labels=list(range(NUM_CLASSES)), zero_division=0)),
        "nll": float(F.cross_entropy(logits_tensor, labels_tensor).item()),
    }


def _fit_task(name, left, right, labels, seed, epochs, patience, batch_size):
    _seed(seed)
    device = _device()
    model = _build(name, left["train"].shape[1], right["train"].shape[1]).to(device)
    tensors = {
        split: (
            torch.from_numpy(left[split]).to(device),
            torch.from_numpy(right[split]).to(device),
            torch.from_numpy(labels[split].astype(np.int64)).to(device),
        )
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
            loss = F.cross_entropy(model(left_train[indices], right_train[indices]), y_train[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            left_valid, right_valid, y_valid = tensors["valid"]
            valid_loss = float(F.cross_entropy(model(left_valid, right_valid), y_valid).item())
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
        for split, data in left.items():
            logits[split] = model(torch.from_numpy(data), torch.from_numpy(right[split])).numpy()
    metrics = {split: _metrics(labels[split], logits[split]) for split in left}
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_epoch": best_epoch + 1,
        "train_loss": metrics["train"]["nll"],
        "validation_loss": metrics["valid"]["nll"],
        "test_loss": metrics["test"]["nll"],
        "train": metrics["train"],
        "valid": metrics["valid"],
        "test": metrics["test"],
    }


def _difference(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half], "per_seed": values.tolist()}


def _summary_g1(records):
    return {
        split: {metric: _difference([row["metrics"][split][metric] for row in records]) for metric in ("add_r2", "joint_r2", "joint_advantage")}
        for split in ("train", "valid", "test")
    }


def _g1_one(splits, mapping, seed, args):
    left_name, right_name, target_name = MAPPINGS[mapping]
    train, valid = splits["train"], splits["valid"]
    config = capacity_config(train[left_name].shape[1], train[right_name].shape[1], train[target_name].shape[1])
    kwargs = {
        "batch_size": args.g1_batch_size,
        "epochs": args.g1_epochs,
        "patience": args.g1_patience,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }
    additive, additive_meta = fit_additive_predictor(
        train[left_name], train[right_name], train[target_name],
        valid[left_name], valid[right_name], valid[target_name], config, seed, **kwargs,
    )
    joint, joint_meta = fit_joint_predictor(
        train[left_name], train[right_name], train[target_name],
        valid[left_name], valid[right_name], valid[target_name], config, seed + 1, **kwargs,
    )
    metrics = {}
    for split, data in splits.items():
        additive_prediction = _predict(additive, data[left_name], data[right_name])
        joint_prediction = _predict(joint, data[left_name], data[right_name])
        from sklearn.metrics import r2_score
        additive_r2 = r2_score(data[target_name], additive_prediction, multioutput="variance_weighted")
        joint_r2 = r2_score(data[target_name], joint_prediction, multioutput="variance_weighted")
        metrics[split] = {
            "add_r2": float(additive_r2),
            "joint_r2": float(joint_r2),
            "joint_advantage": float(joint_r2 - additive_r2),
        }
    return {
        "mapping": mapping,
        "seed": seed,
        "sources": [left_name, right_name],
        "target": target_name,
        "capacity": config,
        "predictors": {"additive": additive_meta, "joint": joint_meta},
        "metrics": metrics,
    }


def _g1_status(records):
    values = [record["metrics"]["valid"]["joint_advantage"] for record in records]
    return "PASS" if all(value > 0.01 for value in values) else ("WEAK_UNSTABLE" if any(value > 0.01 for value in values) else "FAIL")


def _headroom(records, model_name="joint_product"):
    result = {}
    for split in ("valid", "test"):
        result[split] = {}
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            result[split][metric] = _difference([
                record["models"][model_name][split][metric] - record["models"]["additive"][split][metric]
                for record in records
            ])
        result[split]["nll_improvement"] = _difference([
            record["models"]["additive"][split]["nll"] - record["models"][model_name][split]["nll"]
            for record in records
        ])
    return result


def _g2_status(headroom):
    macro = headroom["valid"]["macro_f1"]
    nll = headroom["valid"]["nll_improvement"]
    if all(value > 0 for value in macro["per_seed"]) and macro["mean"] > 0 and nll["mean"] > 0:
        return "PASS_3SEED_PROMISING"
    if macro["mean"] > 0 or nll["mean"] > 0:
        return "WEAK_UNSTABLE"
    return "FAIL"


def _final_type(g1_status: str, g2_status: str) -> str:
    if g1_status == "PASS" and g2_status == "PASS_3SEED_PROMISING":
        return "TYPE_IV"
    if g1_status == "PASS":
        return "TYPE_II"
    if g2_status == "PASS_3SEED_PROMISING":
        return "TYPE_III"
    if g1_status == "FAIL" and g2_status == "FAIL":
        return "TYPE_I"
    return "INCONCLUSIVE"


def _write_reports(out: Path, audit: dict, g1: dict, g2: dict, matrix: dict):
    (out / "data_audit").mkdir(parents=True, exist_ok=True)
    (out / "g1_cross_modal").mkdir(parents=True, exist_ok=True)
    (out / "g2_task_headroom").mkdir(parents=True, exist_ok=True)
    data_lines = [
        "# MELD data audit", "", "Official MELD emotion annotations joined to the processed MM-Align utterance features.",
        "", f"Feature source: `{audit['feature_source']}`.", f"Sequence handling: `{audit['sequence_handling']}`.",
        f"Text: `{audit['text_representation']}`.", f"Seven-class mapping: `{audit['label_mapping']}`.",
        f"Unique split-qualified sample IDs: `{audit['unique_sample_ids_across_splits']}`; duplicates: `{audit['duplicate_sample_ids_across_splits']}`.",
        "", "| Split | Features | Official rows | Dialogues | Speakers | Length min/mean/max |", "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "valid", "test"):
        row = audit["splits"][split]
        length = row["sequence_length"]
        data_lines.append(f"| {split} | {row['feature_count']} | {row['annotation_count']} | {row['dialogue_count']} | {row['speaker_count']} | {length['min']}/{length['mean']:.2f}/{length['max']} |")
    data_lines += ["", "| Split | Neutral | Surprise | Fear | Sadness | Joy | Disgust | Anger |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("train", "valid", "test"):
        counts = audit["splits"][split]["class_names"]
        data_lines.append("| " + split + " | " + " | ".join(str(counts[name]) for name in EMOTION_LABELS) + " |")
    data_lines += ["", "The feature package omits any annotation row without a matching feature. The official CSV label is authoritative; feature-pickle label discrepancies are reported, not used for training.", ""]
    for split in ("train", "valid", "test"):
        row = audit["splits"][split]
        data_lines.append(f"- `{split}` missing feature IDs: `{row['missing_feature_ids']}`; feature-label mismatch IDs: `{row['feature_label_mismatch_ids']}`; zero pooled vectors: `{row['zero_pooled_vectors']}`.")
    (out / "data_audit" / "MELD_DATA_AUDIT.md").write_text("\n".join(data_lines))

    labels = {"VA_to_T": "VA→T", "VT_to_A": "VT→A", "AT_to_V": "AT→V"}
    g1_lines = ["# MELD G1 cross-modal identifiability", "", "Utterance-level pooled V/A/T features; three screening seeds; validation-only model selection.", "", "| Mapping | Capacity additive/joint | Val J | Test J | G1 |", "|---|---:|---:|---:|---|"]
    for mapping, row in g1.items():
        val, test = row["summary"]["valid"]["joint_advantage"], row["summary"]["test"]["joint_advantage"]
        cap = row["capacity"]
        g1_lines.append(f"| `{labels[mapping]}` | {cap['additive_parameters']:,}/{cap['joint_parameters']:,} ({cap['parameter_difference_percent']:.3f}%) | {val['mean']:.4f} ± {val['std']:.4f} | {test['mean']:.4f} ± {test['std']:.4f} | **{row['status']}** |")
    g1_lines += ["", "G1 pass requires joint-vs-additive validation R² advantage J > 0.01 in every screening seed. Test J is reporting-only.", "", "## Per-seed validation J", "", "| Mapping | Seed 1 | Seed 2 | Seed 3 |", "|---|---:|---:|---:|"]
    for mapping, row in g1.items():
        values = {record["seed"]: record["metrics"]["valid"]["joint_advantage"] for record in row["records"]}
        g1_lines.append(f"| `{labels[mapping]}` | {values.get(1, float('nan')):.4f} | {values.get(2, float('nan')):.4f} | {values.get(3, float('nan')):.4f} |")
    (out / "g1_cross_modal" / "MELD_G1_CROSS_MODAL_REPORT.md").write_text("\n".join(g1_lines))

    g2_lines = ["# MELD G2 task headroom", "", "Frozen utterance-level features; seven-class emotion CE; validation NLL early stopping; three screening seeds.", "", "| Pair | Joint−Add Macro F1 val | NLL improvement val | Acc val | Weighted F1 val | G2 |", "|---|---:|---:|---:|---:|---|"]
    for pair, row in g2.items():
        val = row["headroom"]["valid"]
        g2_lines.append(f"| `{pair}` | {val['macro_f1']['mean']*100:.3f} ± {val['macro_f1']['std']*100:.3f} pp | {val['nll_improvement']['mean']:.5f} ± {val['nll_improvement']['std']:.5f} | {val['accuracy']['mean']*100:.3f} ± {val['accuracy']['std']*100:.3f} pp | {val['weighted_f1']['mean']*100:.3f} ± {val['weighted_f1']['std']*100:.3f} pp | **{row['status']}** |")
    g2_lines += ["", "Strong G2 requires joint-vs-additive validation Macro F1 positive in all three seeds and positive mean NLL improvement. Test values are reporting-only.", "", "## Test reporting", "", "| Pair | Joint−Add Macro F1 test | NLL improvement test | Acc test | Weighted F1 test |", "|---|---:|---:|---:|---:|"]
    for pair, row in g2.items():
        test = row["headroom"]["test"]
        g2_lines.append(f"| `{pair}` | {test['macro_f1']['mean']*100:.3f} ± {test['macro_f1']['std']*100:.3f} pp | {test['nll_improvement']['mean']:.5f} ± {test['nll_improvement']['std']:.5f} | {test['accuracy']['mean']*100:.3f} ± {test['accuracy']['std']*100:.3f} pp | {test['weighted_f1']['mean']*100:.3f} ± {test['weighted_f1']['std']*100:.3f} pp |")
    g2_lines += ["", "## Main test metrics", "", "| Pair | Model | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | NLL |", "|---|---|---:|---:|---:|---:|"]
    for pair, row in g2.items():
        for model in MODELS:
            values = {metric: _difference([record["models"][model]["test"][metric] for record in row["records"]]) for metric in ("accuracy", "macro_f1", "weighted_f1", "nll")}
            g2_lines.append(f"| `{pair}` | `{model}` | {values['accuracy']['mean']*100:.3f} ± {values['accuracy']['std']*100:.3f} | {values['macro_f1']['mean']*100:.3f} ± {values['macro_f1']['std']*100:.3f} | {values['weighted_f1']['mean']*100:.3f} ± {values['weighted_f1']['std']*100:.3f} | {values['nll']['mean']:.4f} ± {values['nll']['std']:.4f} |")
    g2_lines += ["", "## Per-seed validation primary metric", "", "| Pair | Seed 1 Δ Macro F1 / ΔNLL | Seed 2 Δ Macro F1 / ΔNLL | Seed 3 Δ Macro F1 / ΔNLL |", "|---|---:|---:|---:|"]
    for pair, row in g2.items():
        per_seed = []
        for record in row["records"]:
            delta_f1 = record["models"]["joint_product"]["valid"]["macro_f1"] - record["models"]["additive"]["valid"]["macro_f1"]
            delta_nll = record["models"]["additive"]["valid"]["nll"] - record["models"]["joint_product"]["valid"]["nll"]
            per_seed.append(f"{delta_f1*100:+.3f} / {delta_nll:+.4f}")
        g2_lines.append(f"| `{pair}` | " + " | ".join(per_seed) + " |")
    (out / "g2_task_headroom" / "MELD_G2_TASK_HEADROOM_REPORT.md").write_text("\n".join(g2_lines))

    decision_lines = ["# MELD two-gate decision", "", "## Two-gate matrix", "", "| Pair / mapping | G1 | G2 | Final type |", "|---|---|---|---|"]
    for pair, mapping in (("VA", "VA_to_T"), ("VT", "VT_to_A"), ("AT", "AT_to_V")):
        row = matrix[pair]
        decision_lines.append(f"| `{pair}` / `{mapping}` | {row['g1']} | {row['g2']} | **{row['final_type']}** |")
    decision_lines += ["", "## Hard stop", "", "JAD is not trained unless the same pair reaches TYPE_IV after permitted frozen-seed confirmation. This screening does not change pooling, rank, loss, purifier, context, or architecture.", "", "G1 and G2 are validation-gated; test metrics are descriptive only.", ""]
    (out / "MELD_TWO_GATE_DECISION.md").write_text("\n".join(decision_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/multibench/meld_download/processed"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/multibench/meld_download/annotations"))
    parser.add_argument("--embedding", type=Path, default=Path("data/multibench/meld_download/processed/embedding.p"))
    parser.add_argument("--out", type=Path, default=Path("results/meld/v55"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--g1-epochs", type=int, default=100)
    parser.add_argument("--g1-patience", type=int, default=8)
    parser.add_argument("--g1-batch-size", type=int, default=256)
    parser.add_argument("--g2-epochs", type=int, default=80)
    parser.add_argument("--g2-patience", type=int, default=8)
    parser.add_argument("--g2-batch-size", type=int, default=256)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits, audit = load_meld(args.processed_dir, args.annotation_dir, args.embedding)
    (args.out / "data_audit").mkdir(parents=True, exist_ok=True)
    (args.out / "data_audit" / "meld_data_audit.json").write_text(json.dumps(audit, indent=2))

    g1 = {}
    for mapping in MAPPINGS:
        records = [_g1_one(splits, mapping, seed, args) for seed in args.seeds]
        g1[mapping] = {"status": _g1_status(records), "capacity": records[0]["capacity"], "summary": _summary_g1(records), "records": records}

    labels = {split: splits[split]["labels"] for split in splits}
    g2 = {}
    for pair, (left_name, right_name, _) in (("VA", MAPPINGS["VA_to_T"]), ("VT", MAPPINGS["VT_to_A"]), ("AT", MAPPINGS["AT_to_V"])):
        left = {split: splits[split][left_name] for split in splits}
        right = {split: splits[split][right_name] for split in splits}
        records = [{"seed": seed, "models": {name: _fit_task(name, left, right, labels, seed, args.g2_epochs, args.g2_patience, args.g2_batch_size) for name in MODELS}} for seed in args.seeds]
        g2[pair] = {"sources": [left_name, right_name], "architecture": parameter_counts(left["train"].shape[1], right["train"].shape[1]), "status": _g2_status(_headroom(records)), "headroom": _headroom(records), "records": records}

    matrix = {}
    for pair, mapping in (("VA", "VA_to_T"), ("VT", "VT_to_A"), ("AT", "AT_to_V")):
        matrix[pair] = {"mapping": mapping, "g1": g1[mapping]["status"], "g2": g2[pair]["status"], "final_type": _final_type(g1[mapping]["status"], g2[pair]["status"])}
    payload = {
        "dataset": "MELD",
        "source": audit["feature_source"],
        "audit": audit,
        "g1": g1,
        "g2": g2,
        "two_gate_matrix": matrix,
        "protocol": {
            "seeds": args.seeds,
            "task": "utterance-level 7-class emotion recognition",
            "g1_validation_gate": "J_val > 0.01 in every screening seed",
            "g2_validation_gate": "joint Macro F1 positive in every screening seed and mean NLL improvement positive",
            "selection": "validation NLL only",
            "test_used_for_selection": False,
            "jad_trained": False,
        },
    }
    (args.out / "meld_two_gate.json").write_text(json.dumps(payload, indent=2))
    _write_reports(args.out, audit, g1, g2, matrix)


if __name__ == "__main__":
    main()
