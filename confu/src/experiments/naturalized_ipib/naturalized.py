"""Naturalized IPIB: controlled interaction targets on MOSEI source geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.hypothesis_class.h0_audit import _fit_regression, _load_data, _standardize
from src.experiments.multibench.mosei_identifiability import AdditivePredictor, JointPredictor, _seed, capacity_config
from src.experiments.synthetic_interaction.ipib import (
    _health,
    _train_interaction,
)


BETA_VALUES = (0.0, 0.25, 0.5, 1.0)
SOURCE_DIM = 64
TARGET_DIM = 64
INTERACTION_RANK = 32
PRIVATE_STRENGTH = 0.25
PROJECTION_SEED = 6200
GENERATOR_SEED = 6201


def _projection(input_dim, output_dim, rng):
    matrix, _ = np.linalg.qr(rng.normal(size=(input_dim, output_dim)))
    return matrix[:, :output_dim].T.astype(np.float32)


def _fixed_generators(vision_dim, audio_dim):
    projection_rng = np.random.default_rng(PROJECTION_SEED)
    p_v = _projection(vision_dim, SOURCE_DIM, projection_rng)
    p_a = _projection(audio_dim, SOURCE_DIM, projection_rng)
    rng = np.random.default_rng(GENERATOR_SEED)
    a1 = rng.normal(size=(TARGET_DIM, SOURCE_DIM)).astype(np.float32)
    a2 = rng.normal(size=(TARGET_DIM, SOURCE_DIM)).astype(np.float32)
    p1 = rng.normal(size=(INTERACTION_RANK, SOURCE_DIM)).astype(np.float32)
    p2 = rng.normal(size=(INTERACTION_RANK, SOURCE_DIM)).astype(np.float32)
    b = rng.normal(size=(TARGET_DIM, INTERACTION_RANK)).astype(np.float32)
    private = rng.normal(size=(TARGET_DIM, TARGET_DIM)).astype(np.float32)
    task_weight = rng.normal(size=(INTERACTION_RANK,)).astype(np.float32)
    task_weight /= np.linalg.norm(task_weight)
    return {"vision_projection": p_v, "audio_projection": p_a, "a1": a1, "a2": a2, "p1": p1, "p2": p2, "b": b, "private": private, "task_weight": task_weight}


def _source_splits(raw, generators):
    train = raw["train"]
    projected = {}
    for split_name, split in raw.items():
        vision = split["vision"] @ generators["vision_projection"].T
        audio = split["audio"] @ generators["audio_projection"].T
        projected[split_name] = {"x1": vision.astype(np.float32), "x2": audio.astype(np.float32), "ids": [f"{split_name}:{i}" for i in range(len(vision))]}
    # Freeze source normalization after the fixed projection using train statistics only.
    for name in ("x1", "x2"):
        mean, std = projected["train"][name].mean(0), np.maximum(projected["train"][name].std(0), 1e-6)
        for split in projected.values():
            split[name] = ((split[name] - mean) / std).astype(np.float32)
    return projected


def _normalize_component(values, reference):
    scale = float(np.sqrt(np.mean(reference ** 2)))
    return values / max(scale, 1e-6), scale


def _make_targets(sources, generators, seed):
    rng = np.random.default_rng(70000 + seed)
    raw_components = {}
    for split_name, split in sources.items():
        x1, x2 = split["x1"], split["x2"]
        additive = x1 @ generators["a1"].T + x2 @ generators["a2"].T
        z1, z2 = x1 @ generators["p1"].T, x2 @ generators["p2"].T
        interaction = (z1 * z2) @ generators["b"].T
        private_latent = rng.normal(size=(len(x1), TARGET_DIM)).astype(np.float32)
        private = private_latent @ generators["private"].T
        raw_components[split_name] = {"additive": additive, "interaction": interaction, "private": private, "z1": z1, "z2": z2}
    a_scale = float(np.sqrt(np.mean(raw_components["train"]["additive"] ** 2)))
    j_scale = float(np.sqrt(np.mean(raw_components["train"]["interaction"] ** 2)))
    p_scale = float(np.sqrt(np.mean(raw_components["train"]["private"] ** 2)))
    outputs = {}
    for split_name, component in raw_components.items():
        additive = component["additive"] / max(a_scale, 1e-6)
        interaction = component["interaction"] / max(j_scale, 1e-6)
        private = component["private"] / max(p_scale, 1e-6)
        task_score = (component["z1"] * component["z2"]) @ generators["task_weight"]
        outputs[split_name] = {
            **sources[split_name],
            "target_components": {"additive": additive, "interaction": interaction, "private": private},
            "task_score": task_score.astype(np.float32),
            "labels": (task_score > 0).astype(np.int64),
        }
    return outputs, {"additive_scale": a_scale, "interaction_scale": j_scale, "private_scale": p_scale}


def _fit_predictors(splits, beta, seed, args):
    target_splits = {}
    for name, split in splits.items():
        components = split["target_components"]
        target_splits[name] = {"x1": split["x1"], "x2": split["x2"], "target": components["additive"] + beta * components["interaction"] + PRIVATE_STRENGTH * components["private"]}
    train, valid, test = target_splits["train"], target_splits["valid"], target_splits["test"]
    capacity = capacity_config(SOURCE_DIM, SOURCE_DIM, TARGET_DIM)
    fit_args = argparse.Namespace(**vars(args), dataset="NaturalizedIPIB", mapping_name=f"beta_{beta}")
    models = {}
    for name, model, model_seed in (
        ("additive", AdditivePredictor(SOURCE_DIM, SOURCE_DIM, TARGET_DIM, capacity["hidden_dim"]), seed),
        ("product", JointPredictor(SOURCE_DIM, SOURCE_DIM, TARGET_DIM, capacity["rank"]), seed + 1),
    ):
        _seed(model_seed)
        models[name], _, _ = _fit_regression(model, train["x1"], train["x2"], train["target"], valid["x1"], valid["x2"], valid["target"], model_seed, fit_args)
    predictions = {}
    for split_name, split in target_splits.items():
        predictions[split_name] = {name: model(torch.from_numpy(split["x1"]), torch.from_numpy(split["x2"])).detach().numpy() for name, model in models.items()}
    metrics = {}
    for split_name, split in target_splits.items():
        add_r2 = float(r2_score(split["target"], predictions[split_name]["additive"], multioutput="variance_weighted"))
        joint_r2 = float(r2_score(split["target"], predictions[split_name]["product"], multioutput="variance_weighted"))
        metrics[split_name] = {"additive_r2": add_r2, "product_r2": joint_r2, "J": joint_r2 - add_r2}
    return target_splits, predictions, metrics, capacity


def _cosine(left, right):
    if np.var(right) <= 1e-12 or np.var(left) <= 1e-12:
        return None
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-8)))


def _standardized_mse(prediction, target, train_target):
    mean, std = train_target.mean(0), np.maximum(train_target.std(0), 1e-6)
    return float(np.mean(((prediction - mean) / std - (target - mean) / std) ** 2))


def _fixed_ridge_recovery(features, target):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(features[0], target[0])
    return float(r2_score(target[2], model.predict(features[2]), multioutput="variance_weighted"))


def _fixed_classification(features, labels):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=200, solver="liblinear", random_state=0))
    model.fit(np.concatenate((features[0], features[1])), np.concatenate((labels[0], labels[1])))
    return float(model.score(features[2], labels[2]))


def run(seed, beta, sources, generators, scales, args):
    splits, _ = _make_targets(sources, generators, seed)
    target_splits, predictions, predictor_metrics, capacity = _fit_predictors(splits, beta, seed, args)
    d = [predictions[name]["product"] - predictions[name]["additive"] for name in ("train", "valid", "test")]
    ground_truth = [split["target_components"]["interaction"] * beta for split in (splits["train"], splits["valid"], splits["test"])]
    jad_splits = [{"x1": torch.from_numpy(split["x1"]), "x2": torch.from_numpy(split["x2"])} for split in (splits["train"], splits["valid"], splits["test"])]
    jad_outputs, jad_meta = _train_interaction(d, jad_splits, seed + 200, "d2")
    direct = {
        "d_to_ground_truth_r2": _fixed_ridge_recovery(d, ground_truth) if beta else None,
        "d_to_ground_truth_cosine_test": _cosine(d[2], ground_truth[2]) if beta else None,
        "d_to_ground_truth_standardized_mse_test": _standardized_mse(d[2], ground_truth[2], ground_truth[0]) if beta else None,
        "d_health": _health(d[2]),
        "ground_truth_health": _health(ground_truth[2]),
    }
    jad = {
        "training": jad_meta,
        "h_to_d_r2": _fixed_ridge_recovery(jad_outputs, d),
        "h_to_ground_truth_r2": _fixed_ridge_recovery(jad_outputs, ground_truth) if beta else None,
        "h_to_d_cosine_test": _cosine(jad_outputs[2], d[2]),
        "h_to_ground_truth_cosine_test": _cosine(jad_outputs[2], ground_truth[2]) if beta else None,
        "h_health": _health(jad_outputs[2]),
    }
    base = [np.concatenate((split["x1"], split["x2"]), axis=1) for split in (splits["train"], splits["valid"], splits["test"])]
    augmented = [np.concatenate((base[i], jad_outputs[i]), axis=1) for i in range(3)]
    labels = [split["labels"] for split in (splits["train"], splits["valid"], splits["test"])]
    base_accuracy = _fixed_classification(base, labels)
    augmented_accuracy = _fixed_classification(augmented, labels)
    return {
        "seed": seed,
        "beta": beta,
        "capacity": capacity,
        "predictor": predictor_metrics,
        "delta_nested_unused": None,
        "direct_recovery": direct,
        "jad_recovery": jad,
        "accessibility": {"base_test_accuracy": base_accuracy, "augmented_test_accuracy": augmented_accuracy, "delta_access": augmented_accuracy - base_accuracy},
        "finite": bool(np.isfinite([predictor_metrics[split]["J"] for split in ("train", "valid", "test")]).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/naturalized_ipib/v60"))
    parser.add_argument("--mosei-cache", type=Path, default=Path("results/mosei/identifiability_exact/mosei_pooled.npz"))
    parser.add_argument("--mosei-data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETA_VALUES))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw, metadata = _load_data(args.mosei_data_dir, args.mosei_cache)
    raw = _standardize(raw)
    generators = _fixed_generators(raw["train"]["vision"].shape[1], raw["train"]["audio"].shape[1])
    sources = _source_splits(raw, generators)
    np.savez_compressed(args.out / "fixed_generators.npz", **generators)
    spec = {"source_dataset": "MOSEI canonical V/A", "source_metadata": metadata, "source_dims": {"vision": int(raw["train"]["vision"].shape[1]), "audio": int(raw["train"]["audio"].shape[1])}, "projected_source_dim": SOURCE_DIM, "target_dim": TARGET_DIM, "interaction_rank": INTERACTION_RANK, "projection_seed": PROJECTION_SEED, "generator_seed": GENERATOR_SEED, "private_strength": PRIVATE_STRENGTH, "beta_values": args.betas, "seeds": args.seeds, "split_ids": {name: split["ids"] for name, split in sources.items()}, "normalization": "MOSEI train standardization, fixed projection, then projected-source train standardization"}
    (args.out / "data_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    records = []
    for seed in args.seeds:
        for beta in args.betas:
            # Scales are regenerated from the fixed generator and source geometry; keep them in each record for auditability.
            target_sources, scales = _make_targets(sources, generators, seed)
            record = run(seed, beta, sources, generators, scales, args)
            record["component_scales"] = scales
            records.append(record)
            print(json.dumps({"seed": seed, "beta": beta, "J_val": record["predictor"]["valid"]["J"], "delta_access": record["accessibility"]["delta_access"]}))
    predictor_payload = {"protocol": {"seeds": args.seeds, "betas": args.betas, "selection": "validation MSE", "test_used_for_selection": False}, "records": records}
    (args.out / "predictor_screen.json").write_text(json.dumps(predictor_payload, indent=2) + "\n")
    (args.out / "jad_recovery.json").write_text(json.dumps({"records": [{"seed": r["seed"], "beta": r["beta"], **r["direct_recovery"], **r["jad_recovery"]} for r in records]}, indent=2) + "\n")
    (args.out / "accessibility.json").write_text(json.dumps({"records": [{"seed": r["seed"], "beta": r["beta"], **r["accessibility"]} for r in records]}, indent=2) + "\n")
    report = ["# Naturalized IPIB v6.0", "", "Frozen MOSEI V/A representation geometry with a known additive component, product interaction component, private component, and no post-result tuning.", "", "## Locked construction", "", f"- Beta ladder: `{args.betas}`; seeds: `{args.seeds}`; interaction rank: `{INTERACTION_RANK}`; private strength: `{PRIVATE_STRENGTH}`.", "- The source is the canonical MOSEI V/A cache. Fixed projection and generator matrices are in `fixed_generators.npz`; all split IDs and seeds are in `data_spec.json`.", "", "## Predictor screen", "", "| Beta | J validation | J test | Direct d→ground-truth R² | JAD h→ground-truth R² | Accessibility Δ |", "|---:|---:|---:|---:|---:|---:|"]
    for beta in args.betas:
        subset = [r for r in records if r["beta"] == beta]
        mean = lambda values: float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None
        j_values = [r['predictor']['valid']['J'] for r in subset]
        j_std = float(np.std(j_values, ddof=1)) if len(j_values) > 1 else 0.0
        report.append(f"| {beta:.2f} | {np.mean(j_values):.5f} ± {j_std:.5f} | {np.mean([r['predictor']['test']['J'] for r in subset]):.5f} | {mean([r['direct_recovery']['d_to_ground_truth_r2'] for r in subset])} | {mean([r['jad_recovery']['h_to_ground_truth_r2'] for r in subset])} | {np.mean([r['accessibility']['delta_access'] for r in subset]):.5f} |")
    report += ["", "## Interpretation", "", "The benchmark is a controlled bridge, not a natural ground-truth claim. Beta=0 is the false-positive control for the representation-target score J: its target contains no injected interaction. The optional accessibility task is deliberately defined from the interaction score itself, so its beta=0 accessibility delta is not a null for task utility; it demonstrates that target-side absence and task-side interaction relevance are separate axes. Increasing beta is expected to increase predictive headroom and recovery, but monotonicity is evaluated descriptively. Accessibility labels are never used to train the predictor or JAD. Recovery uses one fixed Ridge(alpha=1) and accessibility uses one fixed LogisticRegression(C=1) fit on train+validation; no estimator sweep is performed."]
    (args.out / "NATURALIZED_IPIB_REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
