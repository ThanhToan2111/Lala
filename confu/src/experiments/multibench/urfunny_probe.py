"""Fast frozen linear probes for UR-FUNNY ConFu checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.datasets.affect import AffectDataModule
from src.modules.models.confu import ConFu
from src.utils.log_reg import evaluate_linear_probe
from src.experiments.multibench.mosi_audit import extract_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-type", choices=("confu", "confu_plus"), default="confu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.model_type == "confu_plus":
        from src.experiments.multibench.urfunny_complementarity import ConFuPlus
        model = ConFuPlus.load_from_checkpoint(args.checkpoint, map_location=device).to(device)
    else:
        model = ConFu.load_from_checkpoint(args.checkpoint, map_location=device).to(device)
    dm = AffectDataModule(batch_size=args.batch_size, num_workers=0, pickle_name="humor.pkl", dataset_name="humor")
    dm.setup()
    train = extract_split(model, dm.train_dataloader(), device, args.seed)
    valid = extract_split(model, dm.val_dataloader(), device, args.seed + 1)
    test = extract_split(model, dm.test_dataloader(), device, args.seed + 2)

    def probe(features):
        train_x, valid_x, test_x = (torch.from_numpy(split[features]) for split in (train, valid, test))
        return float(evaluate_linear_probe(
            train_feats=train_x,
            train_labels=torch.from_numpy(train["labels"]).long(),
            val_feats=valid_x,
            val_labels=torch.from_numpy(valid["labels"]).long(),
            test_feats=test_x,
            test_labels=torch.from_numpy(test["labels"]).long(),
            fastsearch=True,
            use_sklearn=True,
        ))

    all_features = "all"
    for split in (train, valid, test):
        split[all_features] = np.concatenate((split["r1"], split["r2"], split["r3"]), axis=1)
    result = {
        "protocol": {"dataset": "UR-FUNNY", "checkpoint": str(args.checkpoint), "seed": args.seed},
        "test_accuracy": {name: probe(name) for name in ("r1", "r2", "r3", "r12", "r13", "r23", "all")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
