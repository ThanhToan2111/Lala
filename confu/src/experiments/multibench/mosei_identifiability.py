"""R1: label-free CMU-MOSEI emotion identifiability screening."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn


EMOTIONS = ("happiness", "sadness", "anger", "surprise", "disgust", "fear")
MAPPINGS = {
    "VA_to_T": ("vision", "audio", "text"),
    "VT_to_A": ("vision", "text", "audio"),
    "AT_to_V": ("audio", "text", "vision"),
}


class AdditivePredictor(nn.Module):
    def __init__(self, source_a_dim: int, source_b_dim: int, target_dim: int, hidden_dim: int):
        super().__init__()
        self.left = nn.Sequential(nn.Linear(source_a_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, target_dim))
        self.right = nn.Sequential(nn.Linear(source_b_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, target_dim))

    def forward(self, source_a: torch.Tensor, source_b: torch.Tensor) -> torch.Tensor:
        return self.left(source_a) + self.right(source_b)


class JointPredictor(nn.Module):
    def __init__(self, source_a_dim: int, source_b_dim: int, target_dim: int, rank: int):
        super().__init__()
        self.left = nn.Linear(source_a_dim, rank)
        self.right = nn.Linear(source_b_dim, rank)
        self.output = nn.Linear(3 * rank, target_dim)

    def forward(self, source_a: torch.Tensor, source_b: torch.Tensor) -> torch.Tensor:
        left = self.left(source_a)
        right = self.right(source_b)
        return self.output(torch.cat((left, right, left * right), dim=1))


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def capacity_config(source_a_dim: int, source_b_dim: int, target_dim: int) -> dict:
    candidates = []
    for hidden_dim in range(16, 257):
        additive_parameters = (source_a_dim * hidden_dim + hidden_dim + hidden_dim * target_dim + target_dim) + (source_b_dim * hidden_dim + hidden_dim + hidden_dim * target_dim + target_dim)
        for rank in range(8, 257):
            joint_parameters = source_a_dim * rank + rank + source_b_dim * rank + rank + 3 * rank * target_dim + target_dim
            difference = 100.0 * abs(joint_parameters - additive_parameters) / additive_parameters
            if difference < 1.0:
                candidates.append((difference, hidden_dim, rank, additive_parameters, joint_parameters))
    if not candidates:
        raise RuntimeError("Could not find capacity-matched predictor configuration")
    additive_parameters, difference, hidden_dim, rank, joint_parameters = min(
        (item[3], item[0], item[1], item[2], item[4]) for item in candidates
    )
    return {
        "hidden_dim": hidden_dim,
        "rank": rank,
        "additive_parameters": additive_parameters,
        "joint_parameters": joint_parameters,
        "parameter_difference_percent": difference,
    }


def joint_advantage(additive_r2: float, joint_r2: float) -> float:
    return float(joint_r2 - additive_r2)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pool(sequence: np.ndarray) -> np.ndarray:
    sequence = np.nan_to_num(sequence.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.abs(sequence).sum(axis=-1) > 1e-8
    count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return (sequence * mask[..., None]).sum(axis=1) / count


def _align_emotions(part: dict, h5_audio: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> tuple[np.ndarray, float]:
    """Align HDF5 All Labels to processed rows using shared audio features."""
    sentiment = np.asarray(part["labels"], dtype=np.float32).reshape(-1)
    aligned = np.zeros((len(sentiment), 7), dtype=np.float32)
    sentiment_matches = 0
    for index, audio_row in enumerate(part["audio"]):
        base = str(part["id"][index, 0])
        audio_row = np.nan_to_num(audio_row.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        nonzero = np.where(np.abs(audio_row).sum(axis=1) > 1e-8)[0]
        sequence = audio_row[nonzero[0] :] if len(nonzero) else audio_row[:0]
        candidates = h5_audio[base]
        same_length = [candidate for candidate in candidates if candidate[0].shape[0] == sequence.shape[0]]
        candidates = same_length or candidates
        scored = []
        for features, label in candidates:
            length = min(len(sequence), len(features))
            score = float(np.nanmean(np.abs(np.nan_to_num(sequence[-length:]) - np.nan_to_num(features[-length:]))))
            scored.append((score, label))
        _, selected = min(scored, key=lambda item: item[0])
        if abs(float(selected[0]) - float(sentiment[index])) < 1e-4:
            sentiment_matches += 1
        aligned[index] = selected
    return aligned, sentiment_matches / max(len(sentiment), 1)


def _load_data(data_dir: Path, cache_path: Path | None = None) -> tuple[dict, dict]:
    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        splits = {
            split: {
                "vision": cached[f"{split}_vision"],
                "audio": cached[f"{split}_audio"],
                "text": cached[f"{split}_text"],
                "emotions": cached[f"{split}_emotions"],
                **({"sentiment": cached[f"{split}_sentiment"]} if f"{split}_sentiment" in cached.files else {}),
            }
            for split in ("train", "valid", "test")
        }
        return splits, {"cache": True, "label_alignment_sentiment_match": float(cached["alignment_rate"])}

    with open(data_dir / "mosei_senti_data.pkl", "rb") as handle:
        processed = pickle.load(handle)
    h5_audio: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    with h5py.File(data_dir / "mosei.hdf5", "r") as handle:
        for key in handle["COVAREP"].keys():
            base = key.rsplit("[", 1)[0]
            features = np.asarray(handle["COVAREP"][key]["features"][()], dtype=np.float32)
            labels = np.asarray(handle["All Labels"][key]["features"][()][0], dtype=np.float32)
            h5_audio.setdefault(base, []).append((features, labels))

    splits = {}
    rates = []
    for split, part in processed.items():
        emotions, rate = _align_emotions(part, h5_audio)
        rates.append(rate)
        splits[split] = {
            "vision": _pool(part["vision"]),
            "audio": _pool(part["audio"]),
            "text": _pool(part["text"]),
            "emotions": emotions[:, 1:],
            "sentiment": np.asarray(part["labels"], dtype=np.float32).reshape(-1),
        }
    metadata = {
        "cache": False,
        "label_alignment_sentiment_match": float(np.mean(rates)),
        "train_size": len(splits["train"]["text"]),
        "validation_size": len(splits["valid"]["text"]),
        "test_size": len(splits["test"]["text"]),
        "vision_dim": int(splits["train"]["vision"].shape[1]),
        "audio_dim": int(splits["train"]["audio"].shape[1]),
        "text_dim": int(splits["train"]["text"].shape[1]),
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            **{f"{split}_{key}": value for split, data in splits.items() for key, value in data.items()},
            alignment_rate=np.asarray(metadata["label_alignment_sentiment_match"], dtype=np.float32),
        )
    return splits, metadata


def _standardize(splits: dict) -> dict:
    stats = {}
    for modality in ("vision", "audio", "text"):
        stats[modality] = (splits["train"][modality].mean(0), np.maximum(splits["train"][modality].std(0), 1e-6))
    result = {}
    for split, data in splits.items():
        result[split] = dict(data)
        for modality in ("vision", "audio", "text"):
            mean, std = stats[modality]
            result[split][modality] = ((data[modality] - mean) / std).astype(np.float32)
    return result


def _fit(model: nn.Module, source_a: np.ndarray, source_b: np.ndarray, target: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray, valid_target: np.ndarray, seed: int, batch_size: int, epochs: int, patience: int, lr: float, weight_decay: float) -> tuple[nn.Module, dict]:
    _seed(seed)
    device = _device()
    model = model.to(device)
    train_a, train_b, train_target = [torch.from_numpy(value).to(device) for value in (source_a, source_b, target)]
    valid_a, valid_b, valid_target = [torch.from_numpy(value).to(device) for value in (valid_a, valid_b, valid_target)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed + 1000)
    best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(train_target), generator=generator).split(batch_size):
            indices = indices.to(device)
            loss = torch.nn.functional.mse_loss(model(train_a[indices], train_b[indices]), train_target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(torch.nn.functional.mse_loss(model(valid_a, valid_b), valid_target).item())
        if valid_loss < best_loss - 1e-7:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), valid_loss, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu(), {"best_epoch": best_epoch, "validation_mse": best_loss, "parameters": _parameter_count(model)}


def _predict(model: nn.Module, source_a: np.ndarray, source_b: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(source_a), torch.from_numpy(source_b)).numpy()


def fit_additive_predictor(source_a: np.ndarray, source_b: np.ndarray, target: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray, valid_target: np.ndarray, config: dict, seed: int, **kwargs) -> tuple[nn.Module, dict]:
    _seed(seed)
    model = AdditivePredictor(source_a.shape[1], source_b.shape[1], target.shape[1], config["hidden_dim"])
    return _fit(model, source_a, source_b, target, valid_a, valid_b, valid_target, seed, **kwargs)


def fit_joint_predictor(source_a: np.ndarray, source_b: np.ndarray, target: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray, valid_target: np.ndarray, config: dict, seed: int, **kwargs) -> tuple[nn.Module, dict]:
    _seed(seed)
    model = JointPredictor(source_a.shape[1], source_b.shape[1], target.shape[1], config["rank"])
    return _fit(model, source_a, source_b, target, valid_a, valid_b, valid_target, seed, **kwargs)


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(target, prediction, multioutput="variance_weighted"))


def _metrics(target: np.ndarray, additive: np.ndarray, joint: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        if int(mask.sum()) < 3:
            return {"n": int(mask.sum()), "add_r2": None, "joint_r2": None, "joint_advantage": None}
        target, additive, joint = target[mask], additive[mask], joint[mask]
    add_r2, joint_r2 = _r2(target, additive), _r2(target, joint)
    return {"n": int(len(target)), "add_r2": add_r2, "joint_r2": joint_r2, "joint_advantage": joint_advantage(add_r2, joint_r2)}


def _run_one(splits: dict, mapping: str, seed: int, args: argparse.Namespace, out_dir: Path) -> dict:
    source_a_name, source_b_name, target_name = MAPPINGS[mapping]
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    config = capacity_config(train[source_a_name].shape[1], train[source_b_name].shape[1], train[target_name].shape[1])
    kwargs = {"batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience, "lr": args.lr, "weight_decay": args.weight_decay}
    additive, additive_meta = fit_additive_predictor(train[source_a_name], train[source_b_name], train[target_name], valid[source_a_name], valid[source_b_name], valid[target_name], config, seed, **kwargs)
    joint, joint_meta = fit_joint_predictor(train[source_a_name], train[source_b_name], train[target_name], valid[source_a_name], valid[source_b_name], valid[target_name], config, seed + 1, **kwargs)
    predictions = {split: {"additive": _predict(additive, data[source_a_name], data[source_b_name]), "joint": _predict(joint, data[source_a_name], data[source_b_name])} for split, data in splits.items()}
    metrics = {split: _metrics(data[target_name], predictions[split]["additive"], predictions[split]["joint"]) for split, data in splits.items()}
    emotion_metrics = {}
    for emotion_index, emotion in enumerate(EMOTIONS):
        emotion_metrics[emotion] = {
            split: _metrics(data[target_name], predictions[split]["additive"], predictions[split]["joint"], data["emotions"][:, emotion_index] > 0)
            for split, data in splits.items()
        }
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": additive.state_dict(), "mapping": mapping, "seed": seed, "config": config}, checkpoint_dir / f"{mapping}_seed_{seed}_additive.pt")
    torch.save({"state_dict": joint.state_dict(), "mapping": mapping, "seed": seed, "config": config}, checkpoint_dir / f"{mapping}_seed_{seed}_joint.pt")
    return {
        "mapping": mapping,
        "seed": seed,
        "source_modalities": [source_a_name, source_b_name],
        "target_modality": target_name,
        "capacity": config,
        "predictors": {"additive": additive_meta, "joint": joint_meta},
        "metrics": metrics,
        "emotion_metrics": emotion_metrics,
        "selection": {"global_validation_joint_advantage": metrics["valid"]["joint_advantage"], "global_test_joint_advantage": metrics["test"]["joint_advantage"], "activation": metrics["valid"]["joint_advantage"] > 0.01},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--out", type=Path, default=Path("results/mosei/identifiability"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits, metadata = _load_data(args.data_dir, args.out / "mosei_pooled.npz")
    splits = _standardize(splits)
    records = []
    for seed in args.seeds:
        for mapping in MAPPINGS:
            record = _run_one(splits, mapping, seed, args, args.out)
            records.append(record)
            (args.out / f"{mapping.lower()}_seed_{seed}.json").write_text(json.dumps(record, indent=2))
            print(mapping, seed, "J_val=", record["metrics"]["valid"]["joint_advantage"], "J_test=", record["metrics"]["test"]["joint_advantage"])
    (args.out / "mosei_identifiability.json").write_text(json.dumps({"metadata": metadata, "records": records}, indent=2))


if __name__ == "__main__":
    main()
