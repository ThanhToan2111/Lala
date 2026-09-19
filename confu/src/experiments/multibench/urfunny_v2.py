"""Utility-first ConFu++ v2 experiments for UR-FUNNY.

E0 is evaluation-only pair headroom on frozen original ConFu checkpoints.
E1 trains a small frozen-backbone logit residual correction with conservative
pair gates.  The script deliberately does not include S1, dynamic routing,
context reconstruction, or r123.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from src.datasets.affect import AffectDataModule
from src.experiments.multibench.mosi_audit import extract_split
from src.modules.models.confu import ConFu
from src.utils.log_reg import evaluate_linear_probe


FEATURES = ("low", "low_r12", "low_r13", "low_r23", "pairs")
PAIR_NAMES = ("r12", "r13", "r23")


def _add_features(splits):
    for split in splits:
        split["low"] = np.concatenate((split["r1"], split["r2"], split["r3"]), axis=1)
        split["low_r12"] = np.concatenate((split["low"], split["r12"]), axis=1)
        split["low_r13"] = np.concatenate((split["low"], split["r13"]), axis=1)
        split["low_r23"] = np.concatenate((split["low"], split["r23"]), axis=1)
        split["pairs"] = np.concatenate(
            (split["low"], split["r12"], split["r13"], split["r23"]), axis=1
        )
    return splits


def _load_splits(checkpoint: Path, seed: int, cache_dir: Path | None = None):
    cache_path = None if cache_dir is None else cache_dir / f"features_seed_{seed}.npz"
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:
            splits = []
            for split_name in ("train", "valid", "test"):
                splits.append({
                    key: cached[f"{split_name}_{key}"]
                    for key in ("r1", "r2", "r3", "r12", "r13", "r23", "labels")
                })
        return _add_features(splits)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ConFu.load_from_checkpoint(checkpoint, map_location=device).to(device)
    model.eval()
    dm = AffectDataModule(
        batch_size=256,
        num_workers=0,
        pickle_name="humor.pkl",
        dataset_name="humor",
    )
    dm.setup()
    splits = [
        extract_split(model, dm.train_dataloader(), device, seed),
        extract_split(model, dm.val_dataloader(), device, seed + 1),
        extract_split(model, dm.test_dataloader(), device, seed + 2),
    ]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            **{
                f"{split_name}_{key}": split[key]
                for split_name, split in zip(("train", "valid", "test"), splits)
                for key in ("r1", "r2", "r3", "r12", "r13", "r23", "labels")
            },
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _add_features(splits)


def _linear_probe(splits, name: str) -> float:
    train, valid, test = splits
    return float(
        evaluate_linear_probe(
            train_feats=torch.from_numpy(train[name]),
            train_labels=torch.from_numpy(train["labels"]).long(),
            val_feats=torch.from_numpy(valid[name]),
            val_labels=torch.from_numpy(valid["labels"]).long(),
            test_feats=torch.from_numpy(test[name]),
            test_labels=torch.from_numpy(test["labels"]).long(),
            fastsearch=True,
            use_sklearn=True,
        )
    )


def _probe_parameter_count(input_dim: int, hidden_dim: int, classes: int = 2) -> int:
    return (input_dim + 1) * hidden_dim + (hidden_dim + 1) * classes


def _matched_hidden(input_dim: int, target_parameters: int, classes: int = 2) -> int:
    candidates = range(1, 512)
    return min(
        candidates,
        key=lambda hidden: abs(_probe_parameter_count(input_dim, hidden, classes) - target_parameters),
    )


def _mlp_probe(splits, name: str, seed: int) -> dict[str, float | int]:
    train, valid, test = splits
    train_x = torch.from_numpy(train[name]).float()
    valid_x = torch.from_numpy(valid[name]).float()
    test_x = torch.from_numpy(test[name]).float()
    train_y = torch.from_numpy(train["labels"]).long()
    valid_y = torch.from_numpy(valid["labels"]).long()
    test_y = torch.from_numpy(test["labels"]).long()
    mean = train_x.mean(0)
    std = train_x.std(0, unbiased=False).clamp_min(1e-6)
    train_x, valid_x, test_x = ((value - mean) / std for value in (train_x, valid_x, test_x))

    low_target = _probe_parameter_count(768, 128)
    hidden = _matched_hidden(train_x.shape[1], low_target)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    probe = nn.Sequential(nn.Linear(train_x.shape[1], hidden), nn.GELU(), nn.Linear(hidden, 2)).to(device)
    train_x, valid_x, test_x = train_x.to(device), valid_x.to(device), test_x.to(device)
    train_y, valid_y, test_y = train_y.to(device), valid_y.to(device), test_y.to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state = copy.deepcopy(probe.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(100):
        probe.train()
        for indices in torch.randperm(len(train_x), generator=generator).split(512):
            indices = indices.to(device)
            logits = probe(train_x[indices])
            loss = F.cross_entropy(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            val_loss = float(F.cross_entropy(probe(valid_x), valid_y).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 10:
                break
    probe.load_state_dict(best_state)
    with torch.no_grad():
        test_accuracy = float((probe(test_x).argmax(1) == test_y).float().mean().item())
    return {
        "accuracy": test_accuracy,
        "validation_loss": best_val,
        "best_epoch": best_epoch,
        "hidden_dim": hidden,
        "parameters": _probe_parameter_count(train_x.shape[1], hidden),
    }


def run_e0(checkpoint_dir: Path, output_dir: Path, seed: int) -> dict[str, object]:
    checkpoint = checkpoint_dir / f"seed_{seed}" / "best_model.ckpt"
    splits = _load_splits(checkpoint, seed, output_dir.parent / "features")
    linear = {name: _linear_probe(splits, name) for name in FEATURES}
    nonlinear = {name: _mlp_probe(splits, name, seed) for name in FEATURES}
    result = {
        "protocol": {
            "stage": "E0",
            "dataset": "UR-FUNNY",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "probe": "same frozen split; linear sklearn fast-search and GPU MLP",
            "context_audit": "humor.pkl exposes aligned padded features only; no context/punchline boundary",
        },
        "linear": linear,
        "nonlinear": nonlinear,
        "gains": {
            "linear_pairs_over_low": linear["pairs"] - linear["low"],
            "nonlinear_pairs_over_low": nonlinear["pairs"]["accuracy"] - nonlinear["low"]["accuracy"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"e0_seed_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


class ResidualCorrection(nn.Module):
    """Small downstream classifier; the ConFu backbone remains frozen."""

    def __init__(self, embed_dim: int = 256, classes: int = 2) -> None:
        super().__init__()
        self.lower_head = nn.Linear(3 * embed_dim, classes)
        self.pair_heads = nn.ModuleList([nn.Linear(embed_dim, classes) for _ in PAIR_NAMES])
        self.gate_logits = nn.Parameter(torch.full((3,), math.log(0.1 / 0.9)))

    def lower(self, low: Tensor) -> Tensor:
        return self.lower_head(low)

    def forward(self, low: Tensor, interactions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        lower_logits = self.lower(low)
        deltas = torch.stack(
            [head(F.layer_norm(interactions[:, index], (interactions.shape[-1],)))
             for index, head in enumerate(self.pair_heads)], dim=1
        )
        gates = self.gate_logits.sigmoid()
        final_logits = lower_logits + (deltas * gates.view(1, -1, 1)).sum(1)
        return lower_logits, final_logits, gates


def _arrays(splits, device: torch.device):
    def tensor(split, key):
        return torch.from_numpy(split[key]).float().to(device)

    train, valid, test = splits
    return [
        (tensor(split, "low"), tensor(split, "r12"), tensor(split, "r13"), tensor(split, "r23"),
         torch.from_numpy(split["labels"]).long().to(device))
        for split in (train, valid, test)
    ]


def _metrics(lower: Tensor, final: Tensor, labels: Tensor) -> dict[str, object]:
    lower_pred, final_pred = lower.argmax(1), final.argmax(1)
    correction = (lower_pred != labels) & (final_pred == labels)
    regression = (lower_pred == labels) & (final_pred != labels)
    return {
        "lower_accuracy": float((lower_pred == labels).float().mean().item()),
        "final_accuracy": float((final_pred == labels).float().mean().item()),
        "correction": int(correction.sum().item()),
        "regression": int(regression.sum().item()),
        "net_correction": int(correction.sum().item() - regression.sum().item()),
    }


def _fit_lower(model, train_data, valid_data):
    """Fit the lower branch with the same sklearn family as the baseline probe."""
    train_x, train_y = train_data[0].cpu().numpy(), train_data[4].cpu().numpy()
    valid_x, valid_y = valid_data[0].cpu().numpy(), valid_data[4].cpu().numpy()
    costs = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6)
    scores = []
    for cost in costs:
        classifier = LogisticRegression(C=cost, max_iter=100, random_state=0)
        classifier.fit(train_x, train_y)
        scores.append(balanced_accuracy_score(valid_y, classifier.predict(valid_x)))
    selected_c = costs[int(np.argmax(scores))]
    _set_lower_classifier(model, train_data, valid_data, selected_c, combine_trainval=False)
    return {
        "selected_c": selected_c,
        "validation_accuracy": float(max(scores)),
        "train_samples": int(len(train_x)),
    }


def _set_lower_classifier(model, train_data, valid_data, selected_c: float, combine_trainval: bool):
    train_x, train_y = train_data[0].cpu().numpy(), train_data[4].cpu().numpy()
    valid_x, valid_y = valid_data[0].cpu().numpy(), valid_data[4].cpu().numpy()
    classifier = LogisticRegression(C=selected_c, max_iter=100, random_state=0)
    # Keep validation untouched while selecting/training correction; refit on
    # train+validation only after the correction checkpoint is selected.
    if combine_trainval:
        classifier.fit(np.concatenate((train_x, valid_x)), np.concatenate((train_y, valid_y)))
    else:
        classifier.fit(train_x, train_y)
    with torch.no_grad():
        # sklearn's binary classifier stores one decision function for class 1;
        # PyTorch CrossEntropy expects two logits, so use [0, decision].
        weight = torch.zeros_like(model.lower_head.weight)
        bias = torch.zeros_like(model.lower_head.bias)
        weight[1].copy_(torch.from_numpy(classifier.coef_[0]).to(weight))
        bias[1].copy_(torch.tensor(float(classifier.intercept_[0]), device=bias.device))
        model.lower_head.weight.copy_(weight)
        model.lower_head.bias.copy_(bias)
    return classifier


def _fit_correction(
    model,
    train_data,
    valid_data,
    seed: int,
    utility_weight: float = 0.0,
    preserve_weight: float = 0.0,
):
    for parameter in model.lower_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(model.pair_heads.parameters()) + [model.gate_logits], lr=1e-3, weight_decay=1e-4
    )
    best_state, best_val, best_epoch, stale = None, float("inf"), 0, 0
    generator = torch.Generator().manual_seed(seed + 1000)
    for epoch in range(40):
        model.train()
        for indices in torch.randperm(len(train_data[0]), generator=generator).split(512):
            indices = indices.to(train_data[0].device)
            interactions = torch.stack(train_data[1:4], dim=1)
            lower_logits, logits, _ = model(train_data[0][indices], interactions[indices])
            final_loss = F.cross_entropy(logits, train_data[4][indices], reduction="none")
            lower_loss = F.cross_entropy(lower_logits, train_data[4][indices], reduction="none")
            utility = F.relu(final_loss - lower_loss.detach()).mean()
            lower_probs = lower_logits.detach().softmax(-1)
            confidence = lower_probs.gather(1, train_data[4][indices].view(-1, 1)).squeeze(1)
            preserve_mask = (lower_logits.detach().argmax(1) == train_data[4][indices]) & (confidence > 0.8)
            preserve_values = F.kl_div(
                logits.log_softmax(-1), lower_probs, reduction="none"
            ).sum(1)
            preserve = preserve_values[preserve_mask].mean() if preserve_mask.any() else final_loss.new_zeros(())
            loss = final_loss.mean() + utility_weight * utility + preserve_weight * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            interactions = torch.stack(valid_data[1:4], dim=1)
            value = float(F.cross_entropy(model(valid_data[0], interactions)[1], valid_data[4]).item())
        if value < best_val - 1e-5:
            best_val, best_epoch, stale = value, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 5:
                break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "validation_loss": best_val}


@torch.no_grad()
def _pair_ablation(model, test_data, seed: int) -> dict[str, object]:
    low, interactions, labels = test_data[0], torch.stack(test_data[1:4], dim=1), test_data[4]
    lower, full, gates = model(low, interactions)
    result = {"gates": [float(value) for value in gates.cpu()]}
    generator = torch.Generator().manual_seed(seed)
    for index, name in enumerate(PAIR_NAMES):
        zeroed = interactions.clone()
        zeroed[:, index] = 0
        shuffled = interactions.clone()
        permutation = torch.randperm(len(interactions), generator=generator).to(interactions.device)
        shuffled[:, index] = interactions[permutation, index]
        zero_acc = float((model(low, zeroed)[1].argmax(1) == labels).float().mean().item())
        shuffle_acc = float((model(low, shuffled)[1].argmax(1) == labels).float().mean().item())
        result[name] = {
            "zero_drop": float((full.argmax(1) == labels).float().mean().item()) - zero_acc,
            "shuffle_drop": float((full.argmax(1) == labels).float().mean().item()) - shuffle_acc,
        }
    return result


def run_e1(
    checkpoint_dir: Path,
    output_dir: Path,
    seed: int,
    utility_weight: float = 0.0,
    preserve_weight: float = 0.0,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint = checkpoint_dir / f"seed_{seed}" / "best_model.ckpt"
    splits = _load_splits(checkpoint, seed, output_dir.parent / "features")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data, valid_data, test_data = _arrays(splits, device)
    model = ResidualCorrection().to(device)
    lower_stage = _fit_lower(model, train_data, valid_data)
    correction_stage = _fit_correction(
        model, train_data, valid_data, seed, utility_weight, preserve_weight
    )
    _set_lower_classifier(model, train_data, valid_data, lower_stage["selected_c"], combine_trainval=True)
    lower_stage["final_refit_trainval"] = True
    model.eval()
    with torch.no_grad():
        test_interactions = torch.stack(test_data[1:4], dim=1)
        lower_logits, final_logits, gates = model(test_data[0], test_interactions)
    stage = "E1" if utility_weight == 0 and preserve_weight == 0 else "E2" if preserve_weight == 0 else "E3"
    result = {
        "protocol": {
            "stage": stage,
            "dataset": "UR-FUNNY",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "backbone": "frozen original ConFu",
            "max_epochs": 40,
            "patience": 5,
            "gate_initial": 0.1,
            "utility_weight": utility_weight,
            "preserve_weight": preserve_weight,
        },
        "lower_stage": lower_stage,
        "correction_stage": correction_stage,
        "metrics": _metrics(lower_logits, final_logits, test_data[4]),
        "pair_ablation": _pair_ablation(model, test_data, seed),
        "parameters": {
            "lower_head": sum(parameter.numel() for parameter in model.lower_head.parameters()),
            "pair_heads": sum(parameter.numel() for parameter in model.pair_heads.parameters()),
            "gates": model.gate_logits.numel(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"{stage.lower()}_seed_{seed}.pt")
    (output_dir / f"{stage.lower()}_seed_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("e0", "e1", "e2", "e3"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/urfunny_confu/checkpoints/humor/confu"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/urfunny_v2"))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "e0":
        run_e0(args.checkpoint_dir, args.output_dir, args.seed)
    else:
        run_e1(
            args.checkpoint_dir,
            args.output_dir,
            args.seed,
            utility_weight=0.1 if args.mode in ("e2", "e3") else 0.0,
            preserve_weight=0.1 if args.mode == "e3" else 0.0,
        )


if __name__ == "__main__":
    main()
