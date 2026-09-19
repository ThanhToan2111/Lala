"""R2: canonical MOSEI D0/D1/D2 after the exact-alignment R1 pass."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.experiments.multibench.mosei_identifiability import (
    MAPPINGS,
    _device,
    _load_data,
    _predict,
    _seed,
    _standardize,
    capacity_config,
    fit_additive_predictor,
    fit_joint_predictor,
)


MAPPING = "VA_to_T"
RANK = 64


class CanonicalInteraction(nn.Module):
    def __init__(self, source_a_dim: int, source_b_dim: int, target_dim: int, rank: int = RANK):
        super().__init__()
        self.left = nn.Linear(source_a_dim, rank, bias=False)
        self.right = nn.Linear(source_b_dim, rank, bias=False)
        self.output = nn.Linear(rank, target_dim, bias=False)
        self.left_norm = nn.LayerNorm(rank)
        self.right_norm = nn.LayerNorm(rank)
        self.output_norm = nn.LayerNorm(target_dim)
        self.rank = rank

    def forward(self, source_a: torch.Tensor, source_b: torch.Tensor) -> torch.Tensor:
        left = self.left_norm(self.left(source_a))
        right = self.right_norm(self.right(source_b))
        return self.output_norm(self.output(left * right / self.rank**0.5))


def _cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(prediction, target, dim=1).mean()


def _cosine_score(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = np.sum(target * prediction, axis=1)
    denominator = np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-8)))


def _health(values: np.ndarray) -> dict:
    centered = values - values.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    probabilities = singular**2 / max(np.sum(singular**2), 1e-12)
    effective_rank = float(np.exp(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))))
    return {
        "variance": float(np.var(values, axis=0).mean()),
        "dimension_std": float(np.std(values, axis=0).mean()),
        "mean_norm": float(np.linalg.norm(values, axis=1).mean()),
        "effective_rank": effective_rank,
    }


def _train_interaction(splits: dict, source_a: str, source_b: str, target: dict[str, np.ndarray], seed: int, epochs: int, patience: int):
    _seed(seed)
    device = _device()
    model = CanonicalInteraction(splits["train"][source_a].shape[1], splits["train"][source_b].shape[1], target["train"].shape[1]).to(device)
    train_a = torch.from_numpy(splits["train"][source_a]).to(device)
    train_b = torch.from_numpy(splits["train"][source_b]).to(device)
    train_y = torch.from_numpy(target["train"]).to(device)
    valid_a = torch.from_numpy(splits["valid"][source_a]).to(device)
    valid_b = torch.from_numpy(splits["valid"][source_b]).to(device)
    valid_y = torch.from_numpy(target["valid"]).to(device)
    target_mean, target_std = train_y.mean(0), train_y.std(0).clamp_min(1e-6)
    train_y = (train_y - target_mean) / target_std
    valid_y = (valid_y - target_mean) / target_std
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(train_y), generator=generator).split(1024):
            indices = indices.to(device)
            loss = _cosine_loss(model(train_a[indices], train_b[indices]), train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(_cosine_loss(model(valid_a, valid_b), valid_y).item())
        if valid_loss < best_loss - 1e-6:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), valid_loss, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.cpu().eval()
    with torch.no_grad():
        outputs = {split: model(torch.from_numpy(data[source_a]), torch.from_numpy(data[source_b])).numpy() for split, data in splits.items()}
    return outputs, model, {"best_epoch": best_epoch, "validation_loss": best_loss, "parameters": sum(p.numel() for p in model.parameters())}


def _fit_linear_probe(train_x, train_y, valid_x, valid_y, seed):
    scaler = StandardScaler().fit(train_x)
    train_scaled, valid_scaled = scaler.transform(train_x), scaler.transform(valid_x)
    best_alpha, best_score = None, -float("inf")
    for alpha in (1e-5, 1e-4, 1e-3, 1e-2):
        model = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=80, tol=1e-3, class_weight="balanced", random_state=seed, average=True).fit(train_scaled, train_y)
        score = f1_score(valid_y, model.predict(valid_scaled), average="macro")
        if score > best_score:
            best_alpha, best_score = alpha, score
    model = SGDClassifier(loss="log_loss", alpha=best_alpha, max_iter=80, tol=1e-3, class_weight="balanced", random_state=seed, average=True).fit(train_scaled, train_y)
    return scaler, model, best_alpha, best_score


def _probe_metrics(train_x, train_y, valid_x, valid_y, test_x, test_y, seed):
    scaler, model, best_c, best_score = _fit_linear_probe(train_x, train_y, valid_x, valid_y, seed)
    predictions = {split: model.predict(scaler.transform(x)) for split, x in (("train", train_x), ("valid", valid_x), ("test", test_x))}
    probabilities = {split: model.predict_proba(scaler.transform(x))[:, 1] for split, x in (("train", train_x), ("valid", valid_x), ("test", test_x))}
    result = {"C": best_c, "validation_macro_f1": float(best_score)}
    for split, labels in (("train", train_y), ("valid", valid_y), ("test", test_y)):
        result[split] = {
            "accuracy": float(accuracy_score(labels, predictions[split])),
            "macro_f1": float(f1_score(labels, predictions[split], average="macro")),
            "auc": float(roc_auc_score(labels, probabilities[split])),
        }
    return result, scaler, model


def _nonlinear_probe(train_x, train_y, valid_x, valid_y, test_x, test_y, seed):
    scaler = StandardScaler().fit(train_x)
    train_x, valid_x, test_x = scaler.transform(train_x), scaler.transform(valid_x), scaler.transform(test_x)
    model = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, batch_size=1024, max_iter=15, tol=1e-2, early_stopping=True, validation_fraction=0.1, random_state=seed)
    model.fit(train_x, train_y)
    result = {"hidden": 32, "alpha": 1e-3}
    for split, x, labels in (("train", train_x, train_y), ("valid", valid_x, valid_y), ("test", test_x, test_y)):
        prediction = model.predict(x)
        result[split] = {"accuracy": float(accuracy_score(labels, prediction)), "macro_f1": float(f1_score(labels, prediction, average="macro"))}
    return result


def _run_seed(splits: dict, seed: int, args: argparse.Namespace, out_dir: Path) -> dict:
    source_a, source_b, target_name = MAPPINGS[MAPPING]
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    config = capacity_config(train[source_a].shape[1], train[source_b].shape[1], train[target_name].shape[1])
    kwargs = {"batch_size": args.predictor_batch_size, "epochs": args.predictor_epochs, "patience": args.predictor_patience, "lr": 1e-3, "weight_decay": 1e-4}
    additive, additive_meta = fit_additive_predictor(train[source_a], train[source_b], train[target_name], valid[source_a], valid[source_b], valid[target_name], config, seed, **kwargs)
    joint, joint_meta = fit_joint_predictor(train[source_a], train[source_b], train[target_name], valid[source_a], valid[source_b], valid[target_name], config, seed + 1, **kwargs)
    predictor_outputs = {split: {"additive": _predict(additive, data[source_a], data[source_b]), "joint": _predict(joint, data[source_a], data[source_b])} for split, data in splits.items()}
    raw_targets = {
        "d0": {split: data[target_name] for split, data in splits.items()},
        "d1": {split: data[target_name] - predictor_outputs[split]["additive"] for split, data in splits.items()},
        "d2": {split: predictor_outputs[split]["joint"] - predictor_outputs[split]["additive"] for split in splits},
    }
    labels = {split: (data["sentiment"] > 0).astype(np.int64) for split, data in splits.items()}
    lower = {split: np.concatenate((data[source_a], data[source_b]), axis=1) for split, data in splits.items()}
    lower_probe, _, _ = _probe_metrics(lower["train"], labels["train"], lower["valid"], labels["valid"], lower["test"], labels["test"], seed)
    results = {
        "seed": seed,
        "mapping": MAPPING,
        "task": "sentiment_positive",
        "source_modalities": [source_a, source_b],
        "target_modality": target_name,
        "capacity": config,
        "predictors": {"additive": additive_meta, "joint": joint_meta},
        "methods": {},
        "lower_order_probe": lower_probe,
    }
    for offset, method in enumerate(("d0", "d1", "d2"), start=20):
        outputs, interaction_model, training = _train_interaction(splits, source_a, source_b, raw_targets[method], seed + offset, args.interaction_epochs, args.interaction_patience)
        augmented = {split: np.concatenate((lower[split], outputs[split]), axis=1) for split in splits}
        probe, scaler, classifier = _probe_metrics(augmented["train"], labels["train"], augmented["valid"], labels["valid"], augmented["test"], labels["test"], seed + offset)
        nonlinear = _nonlinear_probe(augmented["train"], labels["train"], augmented["valid"], labels["valid"], augmented["test"], labels["test"], seed + offset)
        clean_prediction = classifier.predict(scaler.transform(augmented["test"]))
        rng = np.random.default_rng(seed + offset)
        shuffle = {}
        for source in (source_a, source_b):
            shuffled_a, shuffled_b = test[source_a], test[source_b]
            permutation = rng.permutation(len(test[source]))
            if source == source_a:
                shuffled_a = shuffled_a[permutation]
            else:
                shuffled_b = shuffled_b[permutation]
            with torch.no_grad():
                shuffled_output = interaction_model(torch.from_numpy(shuffled_a), torch.from_numpy(shuffled_b)).numpy()
            shuffled_features = np.concatenate((lower["test"], shuffled_output), axis=1)
            shuffled_prediction = classifier.predict(scaler.transform(shuffled_features))
            shuffle[source] = {
                "clean_accuracy": float(accuracy_score(labels["test"], clean_prediction)),
                "shuffled_accuracy": float(accuracy_score(labels["test"], shuffled_prediction)),
                "accuracy_drop": float(accuracy_score(labels["test"], clean_prediction) - accuracy_score(labels["test"], shuffled_prediction)),
            }
        shortcut = {source: float(r2_score(outputs["test"], Ridge(alpha=1.0).fit(train[source], outputs["train"]).predict(test[source]), multioutput="variance_weighted")) for source in (source_a, source_b)}
        results["methods"][method] = {
            "probe": probe,
            "linear_gain_over_lower_test_accuracy": probe["test"]["accuracy"] - lower_probe["test"]["accuracy"],
            "nonlinear_probe": nonlinear,
            "target_cosine_test": _cosine_score(raw_targets[method]["test"], outputs["test"]),
            "target_health": _health(outputs["test"]),
            "single_modality_shortcut_test_r2": shortcut,
            "shuffle": shuffle,
            "training": training,
        }
    (out_dir / f"mosei_r2_va_to_t_seed_{seed}.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--out", type=Path, default=Path("results/mosei/r2_va_to_t"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--predictor-batch-size", type=int, default=512)
    parser.add_argument("--predictor-epochs", type=int, default=80)
    parser.add_argument("--predictor-patience", type=int, default=8)
    parser.add_argument("--interaction-epochs", type=int, default=70)
    parser.add_argument("--interaction-patience", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits, _ = _load_data(args.data_dir, Path("results/mosei/identifiability_exact/mosei_pooled.npz"))
    splits = _standardize(splits)
    records = [_run_seed(splits, seed, args, args.out) for seed in args.seeds]
    (args.out / "mosei_r2_va_to_t.json").write_text(json.dumps({"records": records}, indent=2))


if __name__ == "__main__":
    main()
