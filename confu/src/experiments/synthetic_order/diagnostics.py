"""Health and modality-intervention diagnostics for a saved Oracle run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.datasets.synthetic_order import REGIMES, make_dataset
from src.experiments.synthetic_order.oracle import OracleOrderModel, _linear_probe, _target_metrics


def _health(values):
    centered = values - values.mean(0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    spectrum = singular_values**2
    if spectrum.sum() <= 1e-12:
        effective_rank = 0.0
    else:
        spectrum = spectrum / spectrum.sum()
        effective_rank = float(np.exp(-(spectrum[spectrum > 1e-12] * np.log(spectrum[spectrum > 1e-12])).sum()))
    return {
        "variance": float(np.var(values, axis=0).mean()),
        "dimension_std": float(np.std(values, axis=0).mean()),
        "mean_norm": float(np.linalg.norm(values, axis=1).mean()),
        "effective_rank": effective_rank,
    }


def _extract(model, split):
    model.eval()
    with torch.no_grad():
        output = model(split["x1"], split["x2"], split["x3"])
    return {key: value.cpu().numpy() for key, value in output.items()}


def _shuffled_output(model, split, modality, seed):
    values = [split["x1"], split["x2"], split["x3"]]
    generator = torch.Generator().manual_seed(seed)
    values[modality] = values[modality][torch.randperm(len(values[modality]), generator=generator)]
    model.eval()
    with torch.no_grad():
        output = model(*values)
    return {key: value.cpu().numpy() for key, value in output.items()}


def _intervention_drop(model, splits, clean, key, modality, seed):
    changed = _shuffled_output(model, splits[2], modality, seed)
    base = [np.concatenate((value["r1"], value["r2"], value["r3"]), axis=1) for value in clean]
    features = [np.concatenate((base[index], clean[index][key]), axis=1) for index in range(3)]
    clean_test, _ = _linear_probe(features[0], splits[0]["labels"].numpy(), features[1], splits[1]["labels"].numpy(), features[2], splits[2]["labels"].numpy())
    shuffled_test, _ = _linear_probe(features[0], splits[0]["labels"].numpy(), features[1], splits[1]["labels"].numpy(), np.concatenate((base[2], changed[key]), axis=1), splits[2]["labels"].numpy())
    return {"clean_accuracy": clean_test, "shuffled_accuracy": shuffled_test, "drop": clean_test - shuffled_test}


def run(regime, seed, checkpoint):
    config = REGIMES[regime]
    splits = make_dataset(config, seed)
    model = OracleOrderModel(64, 32, rank=64, active_pairs=config.active_pairs, active_triple=config.active_triple)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    clean = [_extract(model, split) for split in splits]
    health = {key: _health(clean[2][key]) for key in ("h12", "h13", "h23", "h123")}
    interventions = {}
    source_map = {"h12": (0, 1), "h13": (0, 2), "h23": (1, 2), "h123": (0, 1, 2)}
    for key, modalities in source_map.items():
        if np.var(clean[2][key]) < 1e-12:
            continue
        interventions[key] = {str(modality + 1): _intervention_drop(model, splits, clean, key, modality, seed + modality + 1) for modality in modalities}
    return {"regime": regime, "seed": seed, "checkpoint": str(checkpoint), "health": health, "target_metrics": _target_metrics(clean[2], splits[2], config), "interventions": interventions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("s0_first_order", "s1_pair12", "s2_all_pairs", "s3_triple", "s4_mixed"), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.regime, args.seed, args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
