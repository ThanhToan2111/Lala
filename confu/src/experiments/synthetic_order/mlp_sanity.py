"""Small, fixed-budget generic MLP sanity check for synthetic S3."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.datasets.synthetic_order import REGIMES, make_dataset
from src.experiments.synthetic_order.oracle import OracleOrderModel


def _accuracy(model, values, labels):
    with torch.no_grad():
        logits = model(values)
        return float((logits.argmax(1) == labels).float().mean().item()), float(F.cross_entropy(logits, labels).item())


def _run(train_x, train_y, valid_x, valid_y, test_x, test_y, hidden, seed):
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    values = [(value - mean) / std for value in (train_x, valid_x, test_x)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(values[0].shape[1], hidden), nn.GELU(), nn.Linear(hidden, 2)).to(device)
    values = [value.to(device) for value in values]
    labels = [value.to(device) for value in (train_y, valid_y, test_y)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    for epoch in range(100):
        model.train()
        for indices in torch.randperm(len(values[0]), generator=generator).split(512):
            indices = indices.to(device)
            loss = F.cross_entropy(model(values[0][indices]), labels[0][indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        valid_accuracy, valid_loss = _accuracy(model, values[1], labels[1])
        if valid_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale, best_state = valid_loss, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    train_accuracy, train_loss = _accuracy(model, values[0], labels[0])
    valid_accuracy, valid_loss = _accuracy(model, values[1], labels[1])
    test_accuracy, test_loss = _accuracy(model, values[2], labels[2])
    return {
        "hidden": hidden,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "train_loss": train_loss,
        "validation_loss": valid_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "validation_accuracy": valid_accuracy,
        "test_accuracy": test_accuracy,
    }


def _features(splits, checkpoint: Path | None):
    raw = [torch.cat((split["x1"], split["x2"], split["x3"]), dim=1) for split in splits]
    latent = [torch.cat((split["z1"], split["z2"], split["z3"]), dim=1) for split in splits]
    result = {"raw": raw, "latent": latent}
    if checkpoint is not None:
        model = OracleOrderModel(64, 32, rank=64, active_pairs=REGIMES["s3_triple"].active_pairs, active_triple=True)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            result["learned_first_order"] = [
                torch.cat(tuple(model(split["x1"], split["x2"], split["x3"])[key] for key in ("r1", "r2", "r3")), dim=1)
                for split in splits
            ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/synthetic_order/oracle/oracle_s3_triple_seed_1.pt"))
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_order/mlp_sanity_s3_seed_1.json"))
    args = parser.parse_args()
    splits = make_dataset(REGIMES["s3_triple"], args.seed)
    labels = [split["labels"] for split in splits]
    features = _features(splits, args.checkpoint if args.checkpoint.exists() else None)
    results = {"regime": "s3_triple", "seed": args.seed, "methods": {}}
    for name, values in features.items():
        results["methods"][name] = [_run(values[0], labels[0], values[1], labels[1], values[2], labels[2], hidden, args.seed) for hidden in (128, 256, 512)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
