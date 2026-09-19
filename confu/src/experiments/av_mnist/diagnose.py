"""Run ConFu++ complementarity diagnostics on an existing AV-MNIST checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from src.datasets.av_mnist_datamodule import AVMNISTDataModule
from src.experiments.av_mnist.synergyformer import (
    AVMNISTBimodalExperiment,
    evaluate,
    git_output,
    interaction_predictability_probes,
    linear_probes,
    nonlinear_probes,
)


@hydra.main(config_path="../../../configs", config_name="av_mnist", version_base=None)
def main(cfg: DictConfig) -> None:
    source_experiment = cfg.get("source_experiment", "bimodal")
    output_experiment = cfg.get("diagnostic_experiment", "avmnist_base_diagnostics")
    source_path = Path(cfg.results_path) / source_experiment / f"synergy_seed_{cfg.seed}.json"
    source = json.loads(source_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AVMNISTBimodalExperiment.load_from_checkpoint(source["checkpoint"]).to(device)
    datamodule = AVMNISTDataModule(cfg.data_path, cfg.batch_size, cfg.num_workers)
    datamodule.setup()
    train_loader = DataLoader(
        datamodule.train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    started_at = time.perf_counter()
    metrics = evaluate(model, datamodule.test_dataloader(), tuple(cfg.gate_values))
    linear = linear_probes(
        model, train_loader, datamodule.test_dataloader(), cfg.probe_train_samples,
        cfg.get("probe_conditional_only", False),
    ) if cfg.get("run_linear_probes", True) else {}
    nonlinear = nonlinear_probes(
        model, train_loader, datamodule.test_dataloader(), cfg.probe_train_samples,
        cfg.nonlinear_probe_hidden_dim, cfg.nonlinear_probe_epochs, cfg.nonlinear_probe_lr,
        cfg.nonlinear_probe_weight_decay, cfg.seed,
    ) if cfg.get("run_nonlinear_probes", True) else {}
    predictability = interaction_predictability_probes(
        model, train_loader, datamodule.test_dataloader(), cfg.probe_train_samples,
        cfg.nonlinear_probe_hidden_dim, cfg.get("predictability_probe_epochs", 20),
        cfg.nonlinear_probe_lr, cfg.nonlinear_probe_weight_decay, cfg.seed,
    ) if cfg.get("run_predictability_probes", False) else {}
    correction = metrics["correction_regression"]
    result = {
        "experiment": output_experiment,
        "seed": cfg.seed,
        "accuracy": metrics["normal_accuracy"],
        "macro_f1": metrics["normal_macro_f1"],
        "weighted_f1": metrics["normal_weighted_f1"],
        "gain12": metrics["gain12"],
        "zero_drop12": metrics["interaction_zero_drop"],
        "shuffle_drop12": metrics["interaction_shuffle_drop"],
        "shuffle_modality1_drop": metrics["shuffle_modality1_drop"],
        "shuffle_modality2_drop": metrics["shuffle_modality2_drop"],
        "correction": correction["correction"],
        "regression": correction["regression"],
        "net_correction": correction["net_correction"],
        "linear_probe_gain": linear.get("gain_all_over_r1_r2"),
        "nonlinear_probe_gain": nonlinear.get("gain"),
        "interaction_predictability": predictability,
        "r12_variance": metrics["representations"]["r12_variance"],
        "r12_effective_rank": metrics["representations"]["r12_effective_rank"],
        "metrics": metrics,
        "linear_probes": linear,
        "nonlinear_probes": nonlinear,
        "source_result": str(source_path),
        "checkpoint": source["checkpoint"],
        "best_epoch": source["best_epoch"],
        "best_validation_accuracy": source["best_validation_accuracy"],
        "parameters": source.get("parameters", sum(p.numel() for p in model.parameters())),
        "dataset": source.get(
            "dataset", "MultiBench AV-MNIST (Google Drive id 1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp)"
        ),
        "split": "MultiBench official train/validation/test split",
        "evaluation_duration_seconds": time.perf_counter() - started_at,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
    }
    result_path = Path(cfg.results_path) / output_experiment / f"seed_{cfg.seed}.json"
    os.makedirs(result_path.parent, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
