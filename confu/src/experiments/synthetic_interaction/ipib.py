"""IPIB predictors, Oracle, D0, D1, and D2 for 12 -> 3."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from src.datasets.identifiable_interaction import REGIMES, IPIBConfig, make_dataset


MIN_JOINT_ADVANTAGE = 0.01


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)


class LowRankInteraction(nn.Module):
    def __init__(self, input_dim: int = 64, output_dim: int = 64, rank: int = 64):
        super().__init__()
        self.left = nn.Linear(input_dim, rank, bias=False)
        self.right = nn.Linear(input_dim, rank, bias=False)
        self.output = nn.Linear(rank, output_dim, bias=False)
        self.left_norm = nn.LayerNorm(rank)
        self.right_norm = nn.LayerNorm(rank)
        self.output_norm = nn.LayerNorm(output_dim)
        self.rank = rank

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        left = self.left_norm(self.left(left))
        right = self.right_norm(self.right(right))
        return self.output_norm(self.output(left * right / self.rank**0.5))


def _device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _t(split, key, device):
    return split[key].float().to(device)


def _predict(model, split, device):
    model.eval()
    with torch.no_grad():
        return model(_t(split, "x1", device), _t(split, "x2", device)) if isinstance(model, PairPredictor) else model(_t(split, "x1", device))


class AdditivePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(64, 39), nn.GELU(), nn.Linear(39, 64))
        self.q2 = nn.Sequential(nn.Linear(64, 39), nn.GELU(), nn.Linear(39, 64))

    def forward(self, x1, x2):
        return self.q1(x1) + self.q2(x2)


class PairPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(64, 32, bias=False)
        self.right = nn.Linear(64, 32, bias=False)
        self.output = nn.Linear(96, 64, bias=False)

    def forward(self, x1, x2):
        left, right = self.left(x1), self.right(x2)
        return self.output(torch.cat((left, right, left * right), dim=1))


def _fit_predictor(model, train, valid, seed, additive):
    device = _device()
    model.to(device)
    train_x = (_t(train, "x1", device), _t(train, "x2", device))
    valid_x = (_t(valid, "x1", device), _t(valid, "x2", device))
    train_y, valid_y = _t(train, "x3", device), _t(valid, "x3", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    for epoch in range(150):
        model.train()
        for indices in torch.randperm(len(train_y), generator=generator).split(1024):
            indices = indices.to(device)
            prediction = model(train_x[0][indices], train_x[1][indices])
            loss = F.mse_loss(prediction, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(F.mse_loss(model(*valid_x), valid_y).item())
        if valid_loss < best_loss - 1e-6:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), valid_loss, epoch, 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best_state)
    return model.cpu(), {"best_epoch": best_epoch, "validation_mse": best_loss, "parameters": sum(p.numel() for p in model.parameters())}


def _predict_numpy(model, split):
    model.eval()
    with torch.no_grad():
        return model(split["x1"], split["x2"]).numpy()


def _r2(y_true, prediction):
    return float(r2_score(y_true.numpy() if isinstance(y_true, Tensor) else y_true, prediction, multioutput="variance_weighted"))


def _fit_predictors(splits, seed):
    _seed_torch(seed)
    additive, additive_meta = _fit_predictor(AdditivePredictor(), splits[0], splits[1], seed, True)
    _seed_torch(seed + 1)
    joint, joint_meta = _fit_predictor(PairPredictor(), splits[0], splits[1], seed + 1, False)
    predictions = {"additive": [_predict_numpy(additive, split) for split in splits], "joint": [_predict_numpy(joint, split) for split in splits]}
    additive_train_r2 = _r2(splits[0]["x3"], predictions["additive"][0])
    joint_train_r2 = _r2(splits[0]["x3"], predictions["joint"][0])
    additive_valid_r2 = _r2(splits[1]["x3"], predictions["additive"][1])
    joint_valid_r2 = _r2(splits[1]["x3"], predictions["joint"][1])
    metrics = {
        "additive_r2": _r2(splits[2]["x3"], predictions["additive"][2]),
        "joint_r2": _r2(splits[2]["x3"], predictions["joint"][2]),
        "joint_advantage": _r2(splits[2]["x3"], predictions["joint"][2]) - _r2(splits[2]["x3"], predictions["additive"][2]),
        "additive_train_r2": additive_train_r2,
        "joint_train_r2": joint_train_r2,
        "additive_validation_r2": additive_valid_r2,
        "joint_validation_r2": joint_valid_r2,
        "validation_joint_advantage": joint_valid_r2 - additive_valid_r2,
        "additive_parameters": additive_meta["parameters"],
        "joint_parameters": joint_meta["parameters"],
        "parameter_difference_percent": 100 * abs(joint_meta["parameters"] - additive_meta["parameters"]) / additive_meta["parameters"],
        "additive_training": additive_meta,
        "joint_training": joint_meta,
    }
    return predictions, metrics


def _ridge_recovery(features, target):
    if np.var(target[2]) <= 1e-12:
        return None
    best_alpha, best_score = None, -float("inf")
    for alpha in (1e-4, 1e-2, 1.0, 1e2, 1e4):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(features[0], target[0])
        score = r2_score(target[1], model.predict(features[1]), multioutput="variance_weighted")
        if score > best_score:
            best_alpha, best_score = alpha, score
    model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    model.fit(np.concatenate((features[0], features[1])), np.concatenate((target[0], target[1])))
    return float(r2_score(target[2], model.predict(features[2]), multioutput="variance_weighted"))


def _linear_classification(features, labels):
    best_c, best_score = None, -float("inf")
    for c in (1e-4, 1e-2, 1.0, 1e2, 1e4):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=0))
        model.fit(features[0], labels[0])
        score = model.score(features[1], labels[1])
        if score > best_score:
            best_c, best_score = c, score
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=1000, random_state=0))
    model.fit(np.concatenate((features[0], features[1])), np.concatenate((labels[0], labels[1])))
    return float(model.score(features[2], labels[2]))


def _health(values):
    centered = values - values.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False) ** 2
    if singular.sum() == 0:
        rank = 0.0
    else:
        spectrum = singular / singular.sum()
        rank = float(np.exp(-(spectrum[spectrum > 1e-12] * np.log(spectrum[spectrum > 1e-12])).sum()))
    return {"variance": float(np.var(values, axis=0).mean()), "dimension_std": float(np.std(values, axis=0).mean()), "mean_norm": float(np.linalg.norm(values, axis=1).mean()), "effective_rank": rank}


def _batch_standardize(value):
    return (value - value.mean(0, keepdim=True)) / value.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)


def _interaction_loss(prediction, target, objective):
    if objective == "d2":
        return 1 - F.cosine_similarity(prediction, target, dim=1).mean()
    if objective == "d2_mse":
        return F.mse_loss(_batch_standardize(prediction), _batch_standardize(target))
    if objective.startswith("d2_cos_mse_"):
        weight = float(objective.rsplit("_", 1)[1])
        cosine = 1 - F.cosine_similarity(prediction, target, dim=1).mean()
        mse = F.mse_loss(_batch_standardize(prediction), _batch_standardize(target))
        return cosine + weight * mse
    return F.mse_loss(prediction, target)


def _component_diagnostics(split):
    components = {key: split[key].numpy() for key in ("additive", "joint", "private")}
    result = {key: _health(value) for key, value in components.items()}
    result["cross_correlation"] = {
        left + "__" + right: float(np.corrcoef(components[left].reshape(-1), components[right].reshape(-1))[0, 1])
        for left, right in (("additive", "joint"), ("additive", "private"), ("joint", "private"))
    }
    return result


def _train_interaction(targets, splits, seed, objective):
    if np.var(targets[0]) <= 1e-12:
        return [np.zeros_like(value) for value in targets], {"best_epoch": 0, "validation_loss": 0.0, "parameters": 0, "degenerate_target": True}
    _seed_torch(seed)
    model = LowRankInteraction()
    device = _device()
    model.to(device)
    x1, x2 = (_t(splits[0], key, device) for key in ("x1", "x2"))
    vx1, vx2 = (_t(splits[1], key, device) for key in ("x1", "x2"))
    target = torch.from_numpy(targets[0]).float().to(device)
    valid_target = torch.from_numpy(targets[1]).float().to(device)
    mean, std = target.mean(0), target.std(0).clamp_min(1e-6)
    target = (target - mean) / std
    valid_target = (valid_target - mean) / std
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    for epoch in range(70):
        model.train()
        for indices in torch.randperm(len(x1), generator=generator).split(1024):
            indices = indices.to(device)
            prediction = model(x1[indices], x2[indices])
            value = _interaction_loss(prediction, target[indices], objective)
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(vx1, vx2)
            valid_loss = float(_interaction_loss(prediction, valid_target, objective).item())
        if valid_loss < best_loss - 1e-6:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), valid_loss, epoch, 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best_state)
    model.cpu()
    model.eval()
    with torch.no_grad():
        outputs = [model(split["x1"], split["x2"]).numpy() for split in splits]
    return outputs, {"best_epoch": best_epoch, "validation_loss": best_loss, "parameters": sum(p.numel() for p in model.parameters()), "degenerate_target": False}


def _oracle(splits, target, seed):
    outputs, meta = _train_interaction(target, splits, seed, "oracle")
    return outputs, meta


def _evaluate(method, outputs, splits, target, ground_truth):
    labels = [split["labels"].numpy() for split in splits]
    base = [np.concatenate((split["x1"].numpy(), split["x2"].numpy()), axis=1) for split in splits]
    augmented = [np.concatenate((base[index], outputs[index]), axis=1) for index in range(3)]
    target_quality = _ridge_recovery(target, [split["joint_target"].numpy() for split in splits])
    h_recovery = _ridge_recovery(outputs, [split["g12"].numpy() for split in splits])
    projected_recovery = _ridge_recovery(outputs, ground_truth)
    task_score_recovery = _ridge_recovery(outputs, [split["task_score"].numpy() for split in splits])
    base_accuracy = _linear_classification(base, labels)
    augmented_accuracy = _linear_classification(augmented, labels)
    return {
        "method": method,
        "target_to_joint_gt_r2": target_quality,
        "h_to_g12_r2": h_recovery,
        "h_to_projected_joint_r2": projected_recovery,
        "h_to_task_score_r2": task_score_recovery,
        "linear_base_accuracy": base_accuracy,
        "linear_augmented_accuracy": augmented_accuracy,
        "delta12_linear": augmented_accuracy - base_accuracy,
        "health": _health(outputs[2]),
    }


def run(regime: str, seed: int, output_dir: Path):
    config = REGIMES[regime]
    splits = make_dataset(config, seed)
    predictions, predictor_metrics = _fit_predictors(splits, seed)
    raw_d2 = [predictions["joint"][index] - predictions["additive"][index] for index in range(3)]
    d2_is_identifiable = predictor_metrics["validation_joint_advantage"] > MIN_JOINT_ADVANTAGE
    targets = {
        "d0": [split["x3"].numpy() for split in splits],
        "d1": [split["x3"].numpy() - predictions["additive"][index] for index, split in enumerate(splits)],
        "d2": raw_d2 if d2_is_identifiable else [np.zeros_like(value) for value in raw_d2],
    }
    ground_truth = [split["joint_target"].numpy() for split in splits]
    oracle_outputs, oracle_meta = _oracle(splits, ground_truth, seed + 10)
    methods = {"oracle": _evaluate("oracle", oracle_outputs, splits, ground_truth, ground_truth)}
    training = {"oracle": oracle_meta}
    for offset, method in enumerate(("d0", "d1", "d2"), start=20):
        outputs, meta = _train_interaction(targets[method], splits, seed + offset, method)
        methods[method] = _evaluate(method, outputs, splits, targets[method], ground_truth)
        training[method] = meta
    result = {
        "regime": regime, "seed": seed, "config": config.__dict__,
        "component_diagnostics": _component_diagnostics(splits[2]),
        "component_variance": {key: float(np.var(splits[2][key].numpy(), axis=0).mean()) for key in ("additive", "joint", "private", "joint_target")},
        "predictors": predictor_metrics,
        "d2_target_identifiable_on_validation": d2_is_identifiable,
        "methods": methods,
        "training": training,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{regime}_seed_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"regime": regime, "seed": seed, "J": predictor_metrics["joint_advantage"], "methods": {key: round(value["delta12_linear"] * 100, 3) for key, value in methods.items()}}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=tuple(REGIMES), default=None)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_interaction"))
    args = parser.parse_args()
    regimes = (args.regime,) if args.regime else ("i0_no_joint", "i2_medium", "i3_strong", "i4_private_noise")
    results = [run(regime, seed, args.out) for regime in regimes for seed in range(1, args.seeds + 1)]
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
