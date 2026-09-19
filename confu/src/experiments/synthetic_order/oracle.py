"""Oracle order-recovery benchmark for ConFu++ v4."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
import torch.nn.functional as F

from src.datasets.synthetic_order import PAIR_KEYS, REGIMES, SyntheticOrderConfig, make_dataset


MODALITIES = ("1", "2", "3")


class LowRankPair(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.left = nn.Linear(dim, rank, bias=False)
        self.right = nn.Linear(dim, rank, bias=False)
        self.output = nn.Linear(rank, dim, bias=False)
        self.left_norm = nn.LayerNorm(rank)
        self.right_norm = nn.LayerNorm(rank)
        self.output_norm = nn.LayerNorm(dim)
        self.rank = rank

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        left = self.left_norm(self.left(left))
        right = self.right_norm(self.right(right))
        return self.output_norm(self.output(left * right / self.rank**0.5))


class LowRankTriple(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.first = nn.Linear(dim, rank, bias=False)
        self.second = nn.Linear(dim, rank, bias=False)
        self.third = nn.Linear(dim, rank, bias=False)
        self.norms = nn.ModuleList(nn.LayerNorm(rank) for _ in range(3))
        self.output = nn.Linear(rank, dim, bias=False)
        self.output_norm = nn.LayerNorm(dim)
        self.rank = rank

    def forward(self, first: Tensor, second: Tensor, third: Tensor) -> Tensor:
        values = [self.norms[0](self.first(first)), self.norms[1](self.second(second)), self.norms[2](self.third(third))]
        return self.output_norm(self.output(values[0] * values[1] * values[2] / self.rank**0.5))


class OracleOrderModel(nn.Module):
    def __init__(self, observation_dim: int, latent_dim: int, rank: int, active_pairs: tuple[str, ...], active_triple: bool):
        super().__init__()
        self.encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(observation_dim, 64), nn.GELU(), nn.Linear(64, latent_dim))
            for _ in range(3)
        )
        self.pairs = nn.ModuleDict({key: LowRankPair(latent_dim, rank) for key in PAIR_KEYS})
        self.triple = LowRankTriple(latent_dim, rank)
        self.active_pairs = frozenset(active_pairs)
        self.active_triple = active_triple

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> dict[str, Tensor]:
        r1, r2, r3 = (encoder(value) for encoder, value in zip(self.encoders, (x1, x2, x3)))
        zeros = torch.zeros_like(r1)
        return {
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "h12": self.pairs["12_to_3"](r1, r2) if "12_to_3" in self.active_pairs else zeros,
            "h13": self.pairs["13_to_2"](r1, r3) if "13_to_2" in self.active_pairs else zeros,
            "h23": self.pairs["23_to_1"](r2, r3) if "23_to_1" in self.active_pairs else zeros,
            "h123": self.triple(r1, r2, r3) if self.active_triple else zeros,
        }


def _as_tensor(split: dict[str, Tensor], key: str, device: torch.device) -> Tensor:
    return split[key].float().to(device)


def _train_oracle(model, train, valid, config: SyntheticOrderConfig, seed: int, device: torch.device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    train_inputs = tuple(_as_tensor(train, key, device) for key in ("x1", "x2", "x3"))
    valid_inputs = tuple(_as_tensor(valid, key, device) for key in ("x1", "x2", "x3"))
    train_z = tuple(_as_tensor(train, key, device) for key in ("z1", "z2", "z3"))
    valid_z = tuple(_as_tensor(valid, key, device) for key in ("z1", "z2", "z3"))
    targets = {key: _as_tensor(train, key, device) for key in ("g12", "g13", "g23", "g123")}
    valid_targets = {key: _as_tensor(valid, key, device) for key in targets}
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(50):
        model.train()
        for indices in torch.randperm(len(train_z[0]), generator=generator).split(512):
            indices = indices.to(device)
            output = model(*(value[indices] for value in train_inputs))
            loss = sum(F.mse_loss(output[key], train_z[index][indices]) for index, key in enumerate(("r1", "r2", "r3")))
            for output_key, target_key, pair_name in (("h12", "g12", "12_to_3"), ("h13", "g13", "13_to_2"), ("h23", "g23", "23_to_1")):
                if pair_name in config.active_pairs:
                    loss = loss + F.mse_loss(output[output_key], targets[target_key][indices])
            if config.active_triple:
                loss = loss + F.mse_loss(output["h123"], targets["g123"][indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(*valid_inputs)
            validation_loss = sum(F.mse_loss(output[key], valid_z[index]) for index, key in enumerate(("r1", "r2", "r3")))
            for output_key, target_key, pair_name in (("h12", "g12", "12_to_3"), ("h13", "g13", "13_to_2"), ("h23", "g23", "23_to_1")):
                if pair_name in config.active_pairs:
                    validation_loss = validation_loss + F.mse_loss(output[output_key], valid_targets[target_key])
            if config.active_triple:
                validation_loss = validation_loss + F.mse_loss(output["h123"], valid_targets["g123"])
            validation_loss = float(validation_loss.item())
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 6:
                break
    if best_state is None:
        raise RuntimeError("oracle training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "validation_loss": best_loss}


def _extract(model, split, device):
    model.eval()
    with torch.no_grad():
        values = model(*tuple(_as_tensor(split, key, device) for key in ("x1", "x2", "x3")))
    return {key: value.cpu().numpy() for key, value in values.items()}


def _linear_probe(train_x, train_y, valid_x, valid_y, test_x, test_y):
    best, best_score = None, float("-inf")
    for c in (1e-4, 1e-2, 1.0, 1e2, 1e4):
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=300, random_state=0))
        classifier.fit(train_x, train_y)
        score = classifier.score(valid_x, valid_y)
        if score > best_score:
            best, best_score = c, score
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=300, random_state=0))
    classifier.fit(np.concatenate((train_x, valid_x)), np.concatenate((train_y, valid_y)))
    return float(classifier.score(test_x, test_y)), float(best_score)


def _matched_hidden(input_dim: int, target_params: int = (96 + 1) * 128 + (128 + 1) * 2) -> int:
    return min(range(1, 512), key=lambda hidden: abs((input_dim + 1) * hidden + (hidden + 1) * 2 - target_params))


def _mlp_probe(train_x, train_y, valid_x, valid_y, test_x, test_y, seed: int):
    mean, std = train_x.mean(0), train_x.std(0).clip(min=1e-6)
    train_x, valid_x, test_x = ((value - mean) / std for value in (train_x, valid_x, test_x))
    hidden = _matched_hidden(train_x.shape[1])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(train_x.shape[1], hidden), nn.GELU(), nn.Linear(hidden, 2)).to(device)
    tensors = [torch.from_numpy(value).float().to(device) for value in (train_x, valid_x, test_x)]
    labels = [torch.from_numpy(value).long().to(device) for value in (train_y, valid_y, test_y)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, stale = copy.deepcopy(model.state_dict()), float("inf"), 0
    for _ in range(100):
        model.train()
        for indices in torch.randperm(len(tensors[0]), generator=generator).split(512):
            indices = indices.to(device)
            loss = F.cross_entropy(model(tensors[0][indices]), labels[0][indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = float(F.cross_entropy(model(tensors[1]), labels[1]).item())
        if loss < best_loss - 1e-5:
            best_loss, stale, best_state = loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 10:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        return float((model(tensors[2]).argmax(1) == labels[2]).float().mean().item()), hidden


def _features(representations: dict[str, np.ndarray]):
    z1 = np.concatenate((representations["r1"], representations["r2"], representations["r3"]), axis=1)
    z2 = np.concatenate((z1, representations["h12"], representations["h13"], representations["h23"]), axis=1)
    z3 = np.concatenate((z2, representations["h123"]), axis=1)
    return {"z1": z1, "z2": z2, "z3": z3}


def _target_metrics(representation, split, config):
    mapping = (("h12", "g12", "12_to_3"), ("h13", "g13", "13_to_2"), ("h23", "g23", "23_to_1"), ("h123", "g123", "triple"))
    metrics = {}
    for output_key, target_key, activation in mapping:
        predicted, target = representation[output_key], split[target_key].numpy()
        error = predicted - target
        metrics[output_key] = {
            "active": activation == "triple" and config.active_triple or activation in config.active_pairs,
            "mse": float(np.mean(error**2)),
            "cosine": float(np.mean(np.sum(predicted * target, axis=1) / (np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1)).clip(min=1e-8))),
        }
    return metrics


def _probe_all(representations, splits, seed, run_mlp=True):
    feature_splits = [_features(value) for value in representations]
    labels = [(split["labels"].numpy()) for split in splits]
    result = {"linear": {}, "linear_validation": {}, "probe_mode": "full" if run_mlp else "linear_only"}
    if run_mlp:
        result["mlp"] = {}
    for name in ("z1", "z2", "z3"):
        value = [feature_split[name] for feature_split in feature_splits]
        linear, validation = _linear_probe(value[0], labels[0], value[1], labels[1], value[2], labels[2])
        result["linear"][name] = linear
        result["linear_validation"][name] = validation
        if run_mlp:
            nonlinear, hidden = _mlp_probe(value[0], labels[0], value[1], labels[1], value[2], labels[2], seed)
            result["mlp"][name] = nonlinear
            result.setdefault("mlp_hidden", {})[name] = hidden
    result["delta2_linear"] = result["linear"]["z2"] - result["linear"]["z1"]
    result["delta3_linear"] = result["linear"]["z3"] - result["linear"]["z2"]
    result["delta2_linear_validation"] = result["linear_validation"]["z2"] - result["linear_validation"]["z1"]
    result["delta3_linear_validation"] = result["linear_validation"]["z3"] - result["linear_validation"]["z2"]
    if run_mlp:
        result["delta2_mlp"] = result["mlp"]["z2"] - result["mlp"]["z1"]
        result["delta3_mlp"] = result["mlp"]["z3"] - result["mlp"]["z2"]
        result["linearization_gap"] = {name: result["mlp"][name] - result["linear"][name] for name in ("z1", "z2", "z3")}
    result["pair_gains_linear"] = {}
    if run_mlp:
        result["pair_gains_mlp"] = {}
    for name, key in (("12", "h12"), ("13", "h13"), ("23", "h23")):
        pair_features = [np.concatenate((feature_splits[index]["z1"], representations[index][key]), axis=1) for index in range(3)]
        pair_linear, _ = _linear_probe(pair_features[0], labels[0], pair_features[1], labels[1], pair_features[2], labels[2])
        result["pair_gains_linear"][name] = pair_linear - result["linear"]["z1"]
        if run_mlp:
            pair_mlp, _ = _mlp_probe(pair_features[0], labels[0], pair_features[1], labels[1], pair_features[2], labels[2], seed)
            result["pair_gains_mlp"][name] = pair_mlp - result["mlp"]["z1"]

    raw_features = [np.concatenate((split["x1"].numpy(), split["x2"].numpy(), split["x3"].numpy()), axis=1) for split in splits]
    raw_linear, raw_validation = _linear_probe(raw_features[0], labels[0], raw_features[1], labels[1], raw_features[2], labels[2])
    result["raw"] = {"linear": raw_linear, "linear_validation": raw_validation}
    if run_mlp:
        raw_mlp, raw_hidden = _mlp_probe(raw_features[0], labels[0], raw_features[1], labels[1], raw_features[2], labels[2], seed)
        result["raw"].update({"mlp": raw_mlp, "mlp_hidden": raw_hidden})
    return result


def run_regime(regime: str, seed: int, output_dir: Path, config: SyntheticOrderConfig | None = None, run_mlp=True):
    config = REGIMES[regime] if config is None else config
    splits = make_dataset(config, seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = OracleOrderModel(config.observation_dim, config.latent_dim, rank=64, active_pairs=config.active_pairs, active_triple=config.active_triple).to(device)
    training = _train_oracle(model, splits[0], splits[1], config, seed, device)
    representations = [_extract(model, split, device) for split in splits]
    probes = _probe_all(representations, splits, seed, run_mlp=run_mlp)
    target_metrics = _target_metrics(representations[2], splits[2], config)
    health = {
        key: {"variance": float(np.var(representations[2][key], axis=0).mean()), "mean_norm": float(np.linalg.norm(representations[2][key], axis=1).mean())}
        for key in ("h12", "h13", "h23", "h123")
    }
    result = {
        "regime": regime,
        "seed": seed,
        "config": {key: value for key, value in config.__dict__.items()},
        "label_balance": float(splits[2]["labels"].float().mean().item()),
        "training": training,
        "oracle_targets": target_metrics,
        "probes": probes,
        "health": health,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"oracle_{regime}_seed_{seed}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.state_dict(), output_dir / f"oracle_{regime}_seed_{seed}.pt")
    print(json.dumps({"regime": regime, "seed": seed, "delta2": probes["delta2_linear"], "delta3": probes["delta3_linear"]}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=tuple(REGIMES), default=None)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_order/oracle"))
    parser.add_argument("--linear-only", action="store_true")
    args = parser.parse_args()
    regimes = (args.regime,) if args.regime else tuple(REGIMES)
    all_results = []
    for regime in regimes:
        for seed in range(1, args.seeds + 1):
            all_results.append(run_regime(regime, seed, args.out, run_mlp=not args.linear_only))
    (args.out / "oracle_summary.json").write_text(json.dumps(all_results, indent=2) + "\n")


if __name__ == "__main__":
    main()
