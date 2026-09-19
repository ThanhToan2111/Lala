"""ConFu++ v3.1 predictor-capacity audit.

P0 is linear additive, P1 is nonlinear additive with two independent
unimodal networks, and P2 is a capacity-matched joint nonlinear network.
This file intentionally stops before interaction retraining.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from src.experiments.multibench.urfunny_v2 import _load_splits
from src.experiments.multibench.urfunny_v3 import (
    OrderTwo,
    _features_from_model,
    _health_tensor,
    _infonce,
    _pair_probe_features,
    _probe_summary,
    _shuffle_drops,
    _single_r2,
    _tensor_split,
)


EMBED_DIM = 256
MODALITIES = ("r1", "r2", "r3")
TARGET_SOURCES = {
    "r1": ("r2", "r3"),
    "r2": ("r1", "r3"),
    "r3": ("r1", "r2"),
}


class LinearAdditivePredictor(nn.Module):
    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.left = nn.Linear(dim, dim)
        self.right = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return self.left(left) + self.right(right) + self.bias


class IndependentMLP(nn.Module):
    def __init__(self, dim: int = EMBED_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class NonlinearAdditivePredictor(nn.Module):
    def __init__(self, dim: int = EMBED_DIM, hidden: int = 128):
        super().__init__()
        self.left = IndependentMLP(dim, hidden)
        self.right = IndependentMLP(dim, hidden)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return self.left(left) + self.right(right) + self.bias


class JointMLPPredictor(nn.Module):
    def __init__(self, input_dim: int = 2 * EMBED_DIM, hidden: int = 171, output_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return self.net(torch.cat((left, right), dim=1))


class SingleLinearPredictor(nn.Module):
    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Linear(dim, dim)

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


def _parameter_counts() -> dict[str, int]:
    return {
        "p0_linear_additive": sum(value.numel() for value in LinearAdditivePredictor().parameters()),
        "p1_nonlinear_additive": sum(value.numel() for value in NonlinearAdditivePredictor().parameters()),
        "p2_joint_nonlinear": sum(value.numel() for value in JointMLPPredictor().parameters()),
    }


def _prepare(splits, sources, target):
    source_scalers = [StandardScaler().fit(splits[0][source]) for source in sources]
    target_scaler = StandardScaler().fit(splits[0][target])
    prepared = []
    for split in splits:
        values = [scaler.transform(split[source]) for scaler, source in zip(source_scalers, sources)]
        prepared.append((values, target_scaler.transform(split[target])))
    return prepared, source_scalers, target_scaler


def _fit_model(model, train_values, valid_values, seed: int):
    train_inputs, train_target = train_values
    valid_inputs, valid_target = valid_values
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_inputs = tuple(torch.from_numpy(value).float().to(device) for value in train_inputs)
    valid_inputs = tuple(torch.from_numpy(value).float().to(device) for value in valid_inputs)
    train_target = torch.from_numpy(train_target).float().to(device)
    valid_target = torch.from_numpy(valid_target).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(40):
        model.train()
        for indices in torch.randperm(len(train_target), generator=generator).split(512):
            indices = indices.to(device)
            prediction = model(*(value[indices] for value in train_inputs))
            loss = (prediction - train_target[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float((model(*valid_inputs) - valid_target).square().mean().item())
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is None:
        raise RuntimeError("predictor training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "validation_mse_standardized": best_loss}


def _predict(model, prepared_values, target_scaler):
    device = next(model.parameters()).device
    inputs = tuple(torch.from_numpy(value).float().to(device) for value in prepared_values[0])
    model.eval()
    with torch.no_grad():
        prediction = model(*inputs).cpu().numpy()
    return target_scaler.inverse_transform(prediction)


def _metrics(splits, predictions, target):
    result = {}
    for name, split, prediction in zip(("train", "valid", "test"), splits, predictions):
        result[name] = {
            "r2": float(r2_score(split[target], prediction, multioutput="uniform_average")),
            "mse": float(mean_squared_error(split[target], prediction)),
        }
    return result


def _fit_target(splits, target, seed):
    sources = TARGET_SOURCES[target]
    prepared, _, target_scaler = _prepare(splits, sources, target)
    models = {
        "p0": LinearAdditivePredictor(),
        "p1": NonlinearAdditivePredictor(),
        "p2": JointMLPPredictor(),
    }
    outputs, training, fitted_models = {}, {}, {}
    for index, (name, model) in enumerate(models.items()):
        fitted, fit_metrics = _fit_model(model, prepared[0], prepared[1], seed + index * 100)
        fitted_models[name] = fitted
        outputs[name] = [_predict(fitted, values, target_scaler) for values in prepared]
        training[name] = fit_metrics
    return outputs, training, fitted_models


def _fit_directed(splits, source, target, nonlinear, seed):
    prepared, _, target_scaler = _prepare(splits, (source,), target)
    model = IndependentMLP() if nonlinear else SingleLinearPredictor()
    fitted, training = _fit_model(model, prepared[0], prepared[1], seed)
    predictions = [_predict(fitted, values, target_scaler) for values in prepared]
    return _metrics(splits, predictions, target), training


def run(seed: int, checkpoint_dir: Path, cache_dir: Path, output: Path):
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint = checkpoint_dir / f"seed_{seed}" / "best_model.ckpt"
    splits = _load_splits(checkpoint, seed, cache_dir)
    parameter_counts = _parameter_counts()
    target_results, training = {}, {}
    checkpoint_root = output.parent / "checkpoints" / f"seed_{seed}"
    for target_index, target in enumerate(MODALITIES):
        outputs, fit_metrics, fitted_models = _fit_target(splits, target, seed + target_index * 1000)
        metrics = {name: _metrics(splits, values, target) for name, values in outputs.items()}
        target_checkpoints = {}
        for name, model in fitted_models.items():
            path = checkpoint_root / f"{target}_{name}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            target_checkpoints[name] = str(path)
        target_results[target] = {
            "p0": metrics["p0"],
            "p1": metrics["p1"],
            "p2": metrics["p2"],
            "p1_minus_p0": metrics["p1"]["test"]["r2"] - metrics["p0"]["test"]["r2"],
            "p2_minus_p1": metrics["p2"]["test"]["r2"] - metrics["p1"]["test"]["r2"],
            "validation_p2_minus_p1": metrics["p2"]["valid"]["r2"] - metrics["p1"]["valid"]["r2"],
            "checkpoints": target_checkpoints,
        }
        training[target] = fit_metrics
    directed = {}
    for source in MODALITIES:
        for target in MODALITIES:
            if source == target:
                continue
            linear, linear_training = _fit_directed(splits, source, target, False, seed + len(directed) * 100)
            nonlinear, nonlinear_training = _fit_directed(splits, source, target, True, seed + len(directed) * 100 + 1)
            directed[f"{source}_to_{target}"] = {
                "linear": linear,
                "mlp": nonlinear,
                "nonlinear_gain_test": nonlinear["test"]["r2"] - linear["test"]["r2"],
                "nonlinear_gain_validation": nonlinear["valid"]["r2"] - linear["valid"]["r2"],
                "linear_training": linear_training,
                "mlp_training": nonlinear_training,
            }
    result = {
        "protocol": {
            "stage": "E4.1_predictor_capacity_audit",
            "dataset": "UR-FUNNY",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "max_epochs": 40,
            "patience": 5,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "test_used_for_selection": False,
        },
        "parameter_counts": parameter_counts,
        "capacity_difference_p1_vs_p2": abs(parameter_counts["p1_nonlinear_additive"] - parameter_counts["p2_joint_nonlinear"]) / parameter_counts["p1_nonlinear_additive"],
        "targets": target_results,
        "training": training,
        "directed_unimodal": directed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def _fit_p1_residuals(splits, seed, checkpoint_root):
    predictions = {target: [] for target in MODALITIES}
    training = {}
    for index, target in enumerate(MODALITIES):
        sources = TARGET_SOURCES[target]
        prepared, source_scalers, target_scaler = _prepare(splits, sources, target)
        model, fit_metrics = _fit_model(
            NonlinearAdditivePredictor(), prepared[0], prepared[1], seed + index * 100
        )
        predictions[target] = [_predict(model, values, target_scaler) for values in prepared]
        training[target] = fit_metrics
        torch.save(model.state_dict(), checkpoint_root / f"{target}_p1.pt")
        np.savez(
            checkpoint_root / f"{target}_p1_scalers.npz",
            source_0_mean=source_scalers[0].mean_, source_0_scale=source_scalers[0].scale_,
            source_1_mean=source_scalers[1].mean_, source_1_scale=source_scalers[1].scale_,
            target_mean=target_scaler.mean_, target_scale=target_scaler.scale_,
        )
    residuals = []
    for split_index, split in enumerate(splits):
        residuals.append(tuple(
            split[target] - predictions[target][split_index]
            for target in ("r3", "r2", "r1")
        ))
    return residuals, predictions, training


def run_e42(seed: int, checkpoint_dir: Path, cache_dir: Path, output: Path, rank: int = 64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint = checkpoint_dir / f"seed_{seed}" / "best_model.ckpt"
    splits = _load_splits(checkpoint, seed, cache_dir)
    checkpoint_root = output.parent / "checkpoints" / f"e4_2_seed_{seed}"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    residuals, p1_predictions, p1_training = _fit_p1_residuals(splits, seed + 4000, checkpoint_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = [_tensor_split(split, device) for split in splits]
    target_tensors = [tuple(torch.from_numpy(value).float().to(device) for value in residual) for residual in residuals]
    model = OrderTwo(rank=rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 5000)
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
        raise RuntimeError("E4.2 did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), checkpoint_root / "order_two.pt")
    h_by_split = [_features_from_model(model, values) for values in data]
    features = _pair_probe_features(splits, h_by_split)
    probes = _probe_summary(features, seed)
    h_splits = [
        dict(split, **{pair_name: h_values[index] for index, pair_name in enumerate(("r12", "r13", "r23"))})
        for split, h_values in zip(splits, h_by_split)
    ]
    single_r2 = {}
    for pair_index, pair_name in enumerate(("r12", "r13", "r23")):
        pair_sources = (("r1", "r2"), ("r1", "r3"), ("r2", "r3"))[pair_index]
        for source in pair_sources:
            single_r2[f"{pair_name}_from_{source}"] = _single_r2(
                h_splits[0], h_splits[1], h_splits[2], source, pair_name
            )
    result = {
        "protocol": {
            "stage": "E4.2_nonlinear_additive_residual_alignment",
            "dataset": "UR-FUNNY",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "rank": rank,
            "temperature": 0.07,
            "max_epochs": 40,
            "patience": 5,
            "task_loss": False,
            "backbone_frozen": True,
            "target_mapping": {"h12": "e3", "h13": "e2", "h23": "e1"},
        },
        "p1_training": p1_training,
        "residual_health": {
            target: {
                "variance": float(np.var(splits[2][target] - p1_predictions[target][2], axis=0).mean()),
                "mean_norm": float(np.linalg.norm(splits[2][target] - p1_predictions[target][2], axis=1).mean()),
            }
            for target in MODALITIES
        },
        "training": {"best_epoch": best_epoch, "validation_loss": best_value},
        "probes": probes,
        "delta2": {
            "linear_validation": probes["linear_validation"]["low_hall"] - probes["linear_validation"]["low"],
            "linear_test": probes["linear"]["low_hall"] - probes["linear"]["low"],
            "nonlinear_test": probes["nonlinear"]["low_hall"]["accuracy"] - probes["nonlinear"]["low"]["accuracy"],
        },
        "linearization_gap": {
            "order1_test": probes["nonlinear"]["low"]["accuracy"] - probes["linear"]["low"],
            "order2_test": probes["nonlinear"]["low_hall"]["accuracy"] - probes["linear"]["low_hall"],
        },
        "pair_gains": {
            pair_name: {
                "linear_test": probes["linear"][f"low_h{pair_name[1:]}"] - probes["linear"]["low"],
                "nonlinear_test": probes["nonlinear"][f"low_h{pair_name[1:]}"]["accuracy"] - probes["nonlinear"]["low"]["accuracy"],
            }
            for pair_name in ("r12", "r13", "r23")
        },
        "single_modality_r2": single_r2,
        "h_health": {
            pair_name: _health_tensor(torch.from_numpy(h_by_split[2][index]))
            for index, pair_name in enumerate(("r12", "r13", "r23"))
        },
        "shuffle_drops": _shuffle_drops(model, splits, features, seed),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/urfunny_confu/checkpoints/humor/confu"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/urfunny_v2/features"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("e4_1", "e4_2"), default="e4_1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "e4_1":
        run(args.seed, args.checkpoint_dir, args.cache_dir, args.output)
    else:
        run_e42(args.seed, args.checkpoint_dir, args.cache_dir, args.output)


if __name__ == "__main__":
    main()
