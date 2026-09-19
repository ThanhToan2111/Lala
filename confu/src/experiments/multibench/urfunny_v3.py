"""ConFu++ v3 order-decomposed residual alignment on UR-FUNNY.

The script has two deliberately small stages:

``audit``
    Measures additive-vs-joint predictor headroom on frozen original ConFu
    representations.
``e4``
    Freezes first-order predictors and trains low-rank pair interactions to
    align with additive first-order residual targets.

No task loss, adversary, residual classifier, dynamic routing, or r123 is
included in this mechanism test.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
import torch.nn.functional as F

from src.experiments.multibench.urfunny_v2 import _linear_probe, _load_splits, _mlp_probe
from src.utils.log_reg import test_linear_probe, train_linear_probe


EMBED_DIM = 256
PAIR_NAMES = ("r12", "r13", "r23")
PAIR_MODALITIES = (("r1", "r2", "r3"), ("r1", "r3", "r2"), ("r2", "r3", "r1"))


def _standardize_fit(x: np.ndarray, y: np.ndarray):
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    return x_scaler.fit_transform(x), y_scaler.fit_transform(y), x_scaler, y_scaler


def _fit_first_order_predictor(train_x, valid_x, train_y, valid_y):
    train_x_z, train_y_z, x_scaler, y_scaler = _standardize_fit(train_x, train_y)
    valid_x_z = x_scaler.transform(valid_x)
    best, best_alpha, best_loss = None, None, float("inf")
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        predictor = Ridge(alpha=alpha, fit_intercept=False)
        predictor.fit(train_x_z, train_y_z)
        loss = mean_squared_error(valid_y, y_scaler.inverse_transform(predictor.predict(valid_x_z)))
        if loss < best_loss:
            best, best_alpha, best_loss = predictor, alpha, loss
    return best, x_scaler, y_scaler, {"alpha": best_alpha, "validation_mse": float(best_loss)}


def _fit_predictors(splits):
    predictors = {}
    for target in ("r1", "r2", "r3"):
        sources = tuple(value for value in ("r1", "r2", "r3") if value != target)
        target_predictors = {}
        target_predictors["_target_mean"] = splits[0][target].mean(0)
        for source in sources:
            predictor, x_scaler, y_scaler, audit = _fit_first_order_predictor(
                splits[0][source], splits[1][source], splits[0][target], splits[1][target]
            )
            target_predictors[source] = (predictor, x_scaler, y_scaler)
            audit["source"] = source
            audit["target"] = target
            target_predictors[f"{source}_audit"] = audit
        predictors[target] = target_predictors
    return predictors


def _predict(predictor, x_scaler, y_scaler, x):
    return y_scaler.inverse_transform(predictor.predict(x_scaler.transform(x)))


def _additive_prediction(split, target, predictors):
    sources = tuple(value for value in ("r1", "r2", "r3") if value != target)
    first = predictors[target][sources[0]]
    second = predictors[target][sources[1]]
    return _predict(*first, split[sources[0]]) + _predict(*second, split[sources[1]]) - predictors[target]["_target_mean"]


def _joint_prediction(train, valid, test, target):
    sources = tuple(value for value in ("r1", "r2", "r3") if value != target)
    train_x = np.concatenate((train[sources[0]], train[sources[1]]), axis=1)
    valid_x = np.concatenate((valid[sources[0]], valid[sources[1]]), axis=1)
    test_x = np.concatenate((test[sources[0]], test[sources[1]]), axis=1)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    train_x, valid_x, test_x = x_scaler.fit_transform(train_x), x_scaler.transform(valid_x), x_scaler.transform(test_x)
    train_y = y_scaler.fit_transform(train[target])
    model = MLPRegressor(
        hidden_layer_sizes=(128,),
        activation="relu",
        alpha=1e-4,
        batch_size=256,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=100,
        random_state=0,
    )
    model.fit(train_x, train_y)
    return y_scaler.inverse_transform(model.predict(test_x)), {
        "hidden_dim": 128,
        "iterations": int(model.n_iter_),
        "validation_score": float(model.best_validation_score_) if hasattr(model, "best_validation_score_") else None,
    }


def _predictor_health(target, split, predicted):
    residual = split[target] - predicted
    centered = residual - residual.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    probabilities = singular / max(singular.sum(), 1e-12)
    effective_rank = float(np.exp(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum()))
    cosine = np.sum(residual * split[target], axis=1) / (
        np.linalg.norm(residual, axis=1) * np.linalg.norm(split[target], axis=1) + 1e-12
    )
    return {
        "variance": float(residual.var(axis=0).mean()),
        "effective_rank": effective_rank,
        "mean_norm": float(np.linalg.norm(residual, axis=1).mean()),
        "mean_dimension_std": float(residual.std(axis=0).mean()),
        "cosine_with_target": float(cosine.mean()),
        "r2_from_additive": float(r2_score(split[target], predicted, multioutput="uniform_average")),
    }


def run_audit(checkpoint_dir: Path, cache_dir: Path, seed: int, output: Path):
    checkpoint = checkpoint_dir / f"seed_{seed}" / "best_model.ckpt"
    splits = _load_splits(checkpoint, seed, cache_dir)
    predictors = _fit_predictors(splits)
    result = {
        "protocol": {
            "stage": "v3_headroom_audit",
            "dataset": "UR-FUNNY",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "predictor": "independent linear first-order predictors vs joint MLP",
            "joint_predictor_is_diagnostic_only": True,
        },
        "targets": {},
    }
    for target in ("r1", "r2", "r3"):
        additive_valid = _additive_prediction(splits[1], target, predictors)
        additive_test = _additive_prediction(splits[2], target, predictors)
        joint_test, joint_meta = _joint_prediction(*splits, target)
        result["targets"][target] = {
            "predictors": {
                key: value for key, value in predictors[target].items() if key.endswith("_audit")
            },
            "additive_validation_mse": float(mean_squared_error(splits[1][target], additive_valid)),
            "additive_test_r2": float(r2_score(splits[2][target], additive_test, multioutput="uniform_average")),
            "joint_test_r2": float(r2_score(splits[2][target], joint_test, multioutput="uniform_average")),
            "joint_advantage": float(
                r2_score(splits[2][target], joint_test, multioutput="uniform_average")
                - r2_score(splits[2][target], additive_test, multioutput="uniform_average")
            ),
            "joint_meta": joint_meta,
            "residual_health": _predictor_health(target, splits[2], additive_test),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


class PairInteraction(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, rank: int = 64):
        super().__init__()
        self.left = nn.Linear(embed_dim, rank, bias=False)
        self.right = nn.Linear(embed_dim, rank, bias=False)
        self.output = nn.Linear(rank, embed_dim, bias=False)
        self.left_norm = nn.LayerNorm(rank)
        self.right_norm = nn.LayerNorm(rank)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.rank = rank

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        left = self.left_norm(self.left(left))
        right = self.right_norm(self.right(right))
        return self.output_norm(self.output((left * right) / self.rank**0.5))


class OrderTwo(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, rank: int = 64):
        super().__init__()
        self.pairs = nn.ModuleList([PairInteraction(embed_dim, rank) for _ in PAIR_NAMES])

    def forward(self, r1: Tensor, r2: Tensor, r3: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.pairs[0](r1, r2), self.pairs[1](r1, r3), self.pairs[2](r2, r3)


def _infonce(anchor: Tensor, target: Tensor, temperature: float = 0.07) -> Tensor:
    anchor = F.normalize(anchor, dim=1)
    target = F.normalize(target, dim=1)
    logits = anchor @ target.t() / temperature
    labels = torch.arange(len(anchor), device=anchor.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def _targets_from_predictors(splits, predictors):
    result = []
    for split in splits:
        residuals = {
            target: split[target] - _additive_prediction(split, target, predictors)
            for target in ("r1", "r2", "r3")
        }
        # h12 aligns to the residual of modality 3, h13 to modality 2,
        # and h23 to modality 1.
        result.append((residuals["r3"], residuals["r2"], residuals["r1"]))
    return result


def _tensor_split(split, device):
    return tuple(torch.from_numpy(split[key]).float().to(device) for key in ("r1", "r2", "r3"))


def _health_tensor(value: Tensor) -> dict[str, float]:
    value = value.detach()
    centered = value - value.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular / singular.sum().clamp_min(1e-12)
    rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return {
        "variance": float(value.var(0, unbiased=False).mean().item()),
        "effective_rank": float(rank.item()),
        "mean_norm": float(value.norm(dim=1).mean().item()),
        "mean_dimension_std": float(value.std(0, unbiased=False).mean().item()),
    }


def _features_from_model(model, split_tensors):
    with torch.no_grad():
        h12, h13, h23 = model(*split_tensors)
    return (h12.cpu().numpy(), h13.cpu().numpy(), h23.cpu().numpy())


def _pair_probe_features(splits, h_by_split):
    result = []
    for split, h_values in zip(splits, h_by_split):
        low = np.concatenate((split["r1"], split["r2"], split["r3"]), axis=1)
        result.append({
            "labels": split["labels"],
            "low": low,
            "low_h12": np.concatenate((low, h_values[0]), axis=1),
            "low_h13": np.concatenate((low, h_values[1]), axis=1),
            "low_h23": np.concatenate((low, h_values[2]), axis=1),
            "low_hall": np.concatenate((low, *h_values), axis=1),
        })
    return result


def _probe_summary(features, seed):
    names = ("low", "low_h12", "low_h13", "low_h23", "low_hall")
    linear = {name: _linear_probe(features, name) for name in names}
    linear_validation = {}
    for name in names:
        train, valid, test = features
        classifier = train_linear_probe(
            torch.from_numpy(train[name]), torch.from_numpy(train["labels"]).long(),
            torch.from_numpy(valid[name]), torch.from_numpy(valid["labels"]).long(),
            max_iter=100, combine_trainval=False, use_sklearn=True, fastsearch=True,
            logger=logging.getLogger("urfunny_v3_probe"),
        )
        linear_validation[name] = float(test_linear_probe(
            classifier, torch.from_numpy(valid[name]), torch.from_numpy(valid["labels"]).long(),
            True, use_sklearn=True
        )["acc1"])
    nonlinear = {name: _mlp_probe(features, name, seed) for name in names}
    return {
        "linear": linear,
        "linear_validation": linear_validation,
        "nonlinear": nonlinear,
        "linear_gains": {name: linear[name] - linear["low"] for name in names[1:]},
        "linear_validation_gains": {name: linear_validation[name] - linear_validation["low"] for name in names[1:]},
        "nonlinear_gains": {name: nonlinear[name]["accuracy"] - nonlinear["low"]["accuracy"] for name in names[1:]},
    }


def _single_r2(train, valid, test, source, target):
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    train_x = x_scaler.fit_transform(train[source])
    valid_x = x_scaler.transform(valid[source])
    test_x = x_scaler.transform(test[source])
    train_y = y_scaler.fit_transform(train[target])
    best, score = None, float("-inf")
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        predictor = Ridge(alpha=alpha, fit_intercept=False).fit(train_x, train_y)
        current = r2_score(valid[target], y_scaler.inverse_transform(predictor.predict(valid_x)), multioutput="uniform_average")
        if current > score:
            best, score = predictor, current
    prediction = y_scaler.inverse_transform(best.predict(test_x))
    return float(r2_score(test[target], prediction, multioutput="uniform_average"))


def _shuffle_drops(model, splits, features, seed):
    train, valid, test = splits
    test_tensors = _tensor_split(test, next(model.parameters()).device)
    result = {}
    rng = np.random.default_rng(seed)
    for pair_index, (pair_name, (left_index, right_index)) in enumerate(
        zip(PAIR_NAMES, ((0, 1), (0, 2), (1, 2)))
    ):
        base_name = f"low_h{pair_name[1:]}"
        clean_accuracy = _linear_probe(features, base_name)
        for modality_index, label in ((left_index, "left"), (right_index, "right")):
            shuffled = [value.clone() for value in test_tensors]
            permutation = rng.permutation(len(shuffled[modality_index]))
            shuffled[modality_index] = shuffled[modality_index][permutation]
            shuffled_h = _features_from_model(model, tuple(value for value in shuffled))
            corrupted = features[2][base_name].copy()
            corrupted[:, -EMBED_DIM:] = shuffled_h[pair_index]
            corrupted_split = {
                "labels": features[2]["labels"],
                "low": features[2]["low"],
                base_name: corrupted,
            }
            probe_features = [
                {"labels": item["labels"], "low": item["low"], base_name: item[base_name]}
                for item in features
            ]
            probe_features[2] = corrupted_split
            corrupted_accuracy = _linear_probe(probe_features, base_name)
            result[f"{pair_name}_shuffle_{label}"] = clean_accuracy - corrupted_accuracy
    return result


def run_e4(checkpoint_dir: Path, cache_dir: Path, seed: int, output: Path, rank: int = 64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint = checkpoint_dir / f"seed_{seed}" / "best_model.ckpt"
    splits = _load_splits(checkpoint, seed, cache_dir)
    predictors = _fit_predictors(splits)
    residuals = _targets_from_predictors(splits, predictors)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = [_tensor_split(split, device) for split in splits]
    target_tensors = [tuple(torch.from_numpy(value).float().to(device) for value in residual) for residual in residuals]
    model = OrderTwo(rank=rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 3000)
    best_state, best_value, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(40):
        model.train()
        for indices in torch.randperm(len(data[0][0]), generator=generator).split(512):
            indices = indices.to(device)
            outputs = model(*(value[indices] for value in data[0]))
            targets = tuple(value[indices].detach() for value in target_tensors[0])
            loss = sum(_infonce(output, target) for output, target in zip(outputs, targets)) / 3
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            outputs = model(*data[1])
            value = float(sum(_infonce(output, target_tensors[1][index]) for index, output in enumerate(outputs)).item() / 3)
        if value < best_value - 1e-5:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is None:
        raise RuntimeError("E4 did not produce a checkpoint")
    model.load_state_dict(best_state)
    h_by_split = [_features_from_model(model, values) for values in data]
    features = _pair_probe_features(splits, h_by_split)
    probes = _probe_summary(features, seed)
    single_r2 = {}
    h_splits = [
        dict(split, **{pair_name: h_values[index] for index, pair_name in enumerate(PAIR_NAMES)})
        for split, h_values in zip(splits, h_by_split)
    ]
    for pair_index, pair_name in enumerate(PAIR_NAMES):
        for source in PAIR_MODALITIES[pair_index][:2]:
            single_r2[f"{pair_name}_from_{source}"] = _single_r2(
                h_splits[0], h_splits[1], h_splits[2], source, pair_name
            )
    result = {
        "protocol": {
            "stage": "E4_order_decomposed_residual_alignment",
            "dataset": "UR-FUNNY",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "rank": rank,
            "temperature": 0.07,
            "max_epochs": 40,
            "patience": 5,
            "task_loss": False,
            "backbone_frozen": True,
        },
        "predictors": {
            target: {key: value for key, value in predictors[target].items() if key.endswith("_audit")}
            for target in predictors
        },
        "residual_health": {
            target: _predictor_health(target, splits[2], _additive_prediction(splits[2], target, predictors))
            for target in ("r1", "r2", "r3")
        },
        "training": {"best_epoch": best_epoch, "validation_loss": best_value},
        "probes": probes,
        "pair_gains": {
            pair_name: {
                "linear": probes["linear"][f"low_h{pair_name[1:]}"] - probes["linear"]["low"],
                "nonlinear": probes["nonlinear"][f"low_h{pair_name[1:]}"]["accuracy"] - probes["nonlinear"]["low"]["accuracy"],
            }
            for pair_name in PAIR_NAMES
        },
        "delta2": {
            "linear": probes["linear"]["low_hall"] - probes["linear"]["low"],
            "linear_validation": probes["linear_validation"]["low_hall"] - probes["linear_validation"]["low"],
            "nonlinear": probes["nonlinear"]["low_hall"]["accuracy"] - probes["nonlinear"]["low"]["accuracy"],
        },
        "single_modality_r2": single_r2,
        "h_health": {
            pair_name: _health_tensor(torch.from_numpy(h_by_split[2][index]))
            for index, pair_name in enumerate(PAIR_NAMES)
        },
        "shuffle_drops": _shuffle_drops(model, splits, features, seed),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output.with_suffix(".pt"))
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "e4"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/urfunny_confu/checkpoints/humor/confu"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/urfunny_v2/features"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "audit":
        run_audit(args.checkpoint_dir, args.cache_dir, args.seed, args.output)
    else:
        run_e4(args.checkpoint_dir, args.cache_dir, args.seed, args.output, args.rank)


if __name__ == "__main__":
    main()
