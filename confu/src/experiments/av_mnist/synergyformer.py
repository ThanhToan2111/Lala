"""Fair bimodal AV-MNIST fusion benchmark with interaction diagnostics."""

from __future__ import annotations

from collections import defaultdict
import json
import os
import platform
import re
import subprocess
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.datasets.av_mnist_datamodule import AVMNISTDataModule
from src.modules.encoders.mlp import MLP
from src.modules.encoders.resnet import create_audio_encoder, create_image_encoder
from src.modules.models.synergyformer import (
    GatedComposition,
    LowRankInteraction,
    conditional_utility_loss,
    covariance_loss,
    dependence_ranking_loss,
    pair_cross_covariance_loss,
    variance_loss,
)


def effective_rank(representation: Tensor) -> float:
    centered = representation.float() - representation.float().mean(0)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    return torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()


class BimodalFusion(nn.Module):
    """Shared encoders plus one of the four fusion methods in the protocol."""

    def __init__(
        self,
        model_type: str,
        dim: int,
        rank: int,
        hidden_dim: int,
        temperature: float,
        interaction_factor_bias: bool = True,
        interaction_output_bias: bool = False,
        base_first_composition: bool = False,
    ) -> None:
        super().__init__()
        if model_type not in {"additive", "concat_mlp", "confu", "synergy"}:
            raise ValueError(f"unknown model_type: {model_type}")
        self.model_type = model_type
        self.temperature = temperature
        self.base_first_composition = base_first_composition
        self.image_encoder = create_image_encoder(dim)
        self.audio_encoder = create_audio_encoder(dim)
        self.prototypes = nn.Parameter(torch.randn(10, dim) / dim**0.5)
        self.unimodal_norms = nn.ModuleList((nn.LayerNorm(dim), nn.LayerNorm(dim)))
        self.additive_norm = nn.LayerNorm(dim)

        if model_type == "concat_mlp":
            self.fusion = nn.Sequential(
                nn.Linear(2 * dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim, bias=False), nn.LayerNorm(dim)
            )
        elif model_type == "confu":
            self.fusion = nn.Sequential(MLP(2 * dim, hidden_dim, dim, dropout=True), nn.LayerNorm(dim))
        elif model_type == "synergy":
            self.interaction = LowRankInteraction(
                dim,
                rank,
                2,
                normalize_factors=True,
                factor_bias=interaction_factor_bias,
                output_bias=interaction_output_bias,
            )
            self.composition = GatedComposition(dim)

    def classify(self, representation: Tensor) -> Tensor:
        return F.normalize(representation, dim=-1) @ F.normalize(self.prototypes, dim=-1).T / self.temperature

    def additive(self, r1: Tensor, r2: Tensor) -> Tensor:
        return self.additive_norm(self.unimodal_norms[0](r1) + self.unimodal_norms[1](r2))

    def compose(
        self,
        r1: Tensor,
        r2: Tensor,
        r12: Tensor | None = None,
        gate: Tensor | float | None = None,
    ) -> Tensor:
        if self.model_type == "additive":
            return self.additive(r1, r2)
        if self.model_type in {"concat_mlp", "confu"}:
            return self.fusion(torch.cat((r1, r2), dim=-1))
        if r12 is None:
            r12 = self.interaction(r1, r2)
        if self.base_first_composition:
            return self.composition.from_base(self.additive(r1, r2), r12, gate)
        return self.composition(r1, r2, r12, gate)

    def forward(self, images: Tensor, audios: Tensor) -> dict[str, Tensor]:
        r1 = self.image_encoder(images)[0]
        r2 = self.audio_encoder(audios)[0]
        r12 = self.interaction(r1, r2) if self.model_type == "synergy" else torch.zeros_like(r1)
        return {"r1": r1, "r2": r2, "r12": r12, "fused": self.compose(r1, r2, r12)}


class AVMNISTBimodalExperiment(pl.LightningModule):
    def __init__(
        self,
        model_type: str,
        embed_dim: int,
        rank: int,
        fusion_hidden_dim: int,
        temperature: float,
        lr: float,
        weight_decay: float,
        lambda_unimodal: float,
        lambda_variance: float,
        lambda_cross_covariance: float,
        variance_gamma: float,
        diagnostics_samples: int,
        lambda_covariance: float = 0.0,
        interaction_factor_bias: bool = True,
        interaction_output_bias: bool = False,
        lambda_utility: float = 0.0,
        utility_margin: float = 0.0,
        base_first_composition: bool = False,
        lambda_dependency: float = 0.0,
        dependency_margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = BimodalFusion(
            model_type,
            embed_dim,
            rank,
            fusion_hidden_dim,
            temperature,
            interaction_factor_bias,
            interaction_output_bias,
            base_first_composition,
        )
        self.validation_representations: dict[str, list[Tensor]] = defaultdict(list)

    def forward(self, images: Tensor, audios: Tensor) -> dict[str, Tensor]:
        return self.model(images, audios)

    def _loss(self, batch: tuple[Tensor, Tensor, list[str], Tensor], stage: str) -> tuple[Tensor, dict[str, Tensor]]:
        images, audios, _, labels = batch
        outputs = self(images, audios)
        full_logits = self.model.classify(outputs["fused"])
        full_losses = F.cross_entropy(full_logits, labels, reduction="none")
        base_losses = F.cross_entropy(
            self.model.classify(self.model.additive(outputs["r1"], outputs["r2"])), labels, reduction="none"
        )
        fusion_loss = full_losses.mean()
        base_loss = base_losses.mean()
        unimodal_loss = sum(
            F.cross_entropy(self.model.classify(outputs[name]), labels) for name in ("r1", "r2")
        )
        zero = fusion_loss.new_zeros(())
        variance = variance_loss(outputs["r12"], self.hparams.variance_gamma) if self.hparams.model_type == "synergy" else zero
        cross_covariance = (
            pair_cross_covariance_loss(outputs["r12"], outputs["r1"], outputs["r2"])
            if self.hparams.model_type == "synergy" else zero
        )
        covariance = covariance_loss(outputs["r12"]) if self.hparams.model_type == "synergy" else zero
        utility = (
            conditional_utility_loss(full_losses, base_losses, self.hparams.utility_margin)
            if self.hparams.model_type == "synergy" else zero
        )
        dependency = zero
        if self.hparams.model_type == "synergy" and self.hparams.lambda_dependency > 0:
            permutation1 = torch.randperm(labels.shape[0], device=labels.device)
            permutation2 = torch.randperm(labels.shape[0], device=labels.device)
            corrupted = (
                self.model.interaction(outputs["r1"][permutation1], outputs["r2"]),
                self.model.interaction(outputs["r1"], outputs["r2"][permutation2]),
            )
            corrupted_logits = tuple(
                self.model.classify(self.model.compose(outputs["r1"], outputs["r2"], value))
                for value in corrupted
            )
            dependency = dependence_ranking_loss(
                full_logits, corrupted_logits, labels, self.hparams.dependency_margin
            )
        loss = (
            fusion_loss
            + self.hparams.lambda_unimodal * unimodal_loss
            + self.hparams.lambda_variance * variance
            + self.hparams.lambda_cross_covariance * cross_covariance
            + self.hparams.lambda_covariance * covariance
            + self.hparams.lambda_utility * utility
            + self.hparams.lambda_dependency * dependency
        )
        accuracy = (self.model.classify(outputs["fused"]).argmax(1) == labels).float().mean()
        batch_size = labels.shape[0]
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/accuracy", accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for name, value in {
            "fusion": fusion_loss,
            "base": base_loss,
            "unimodal": unimodal_loss,
            "variance": variance,
            "cross_covariance": cross_covariance,
            "covariance": covariance,
            "utility": utility,
            "dependency": dependency,
        }.items():
            self.log(f"{stage}/loss_{name}", value, on_epoch=True, batch_size=batch_size)
        self.log(
            f"{stage}/loss_unimodal_weighted",
            self.hparams.lambda_unimodal * unimodal_loss,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/loss_variance_weighted",
            self.hparams.lambda_variance * variance,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/loss_cross_covariance_weighted",
            self.hparams.lambda_cross_covariance * cross_covariance,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/loss_covariance_weighted",
            self.hparams.lambda_covariance * covariance,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/loss_utility_weighted",
            self.hparams.lambda_utility * utility,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/loss_dependency_weighted",
            self.hparams.lambda_dependency * dependency,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss, outputs

    def training_step(self, batch: tuple[Tensor, Tensor, list[str], Tensor], batch_idx: int) -> Tensor:
        loss, outputs = self._loss(batch, "train")
        if self.hparams.model_type == "synergy":
            for name in ("r1", "r2", "r12"):
                outputs[name].retain_grad()
            self._gradient_representations = outputs
        return loss

    def on_after_backward(self) -> None:
        if self.hparams.model_type != "synergy" or not hasattr(self, "_gradient_representations"):
            return
        norms = {
            name: self._gradient_representations[name].grad.norm()
            for name in ("r1", "r2", "r12")
            if self._gradient_representations[name].grad is not None
        }
        for name, value in norms.items():
            self.log(f"grad/{name}", value)
        if len(norms) == 3:
            self.log("grad_ratio/r12_to_r1", norms["r12"] / norms["r1"].clamp_min(1e-12))
            self.log("grad_ratio/r12_to_r2", norms["r12"] / norms["r2"].clamp_min(1e-12))

    def validation_step(self, batch: tuple[Tensor, Tensor, list[str], Tensor], batch_idx: int) -> Tensor:
        loss, outputs = self._loss(batch, "val")
        collected = sum(chunk.shape[0] for chunk in self.validation_representations["r1"])
        if collected < self.hparams.diagnostics_samples:
            count = self.hparams.diagnostics_samples - collected
            for name in ("r1", "r2", "r12"):
                self.validation_representations[name].append(outputs[name][:count].detach().cpu())
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.validation_representations["r1"]:
            return
        representations = {name: torch.cat(chunks) for name, chunks in self.validation_representations.items()}
        for name, value in representations.items():
            self.log(f"representation/{name}_norm", value.norm(dim=1).mean())
            self.log(f"representation/{name}_variance", value.var(dim=0, unbiased=False).mean())
            self.log(f"representation/{name}_effective_rank", effective_rank(value))
        r1, r2, r12 = representations["r1"], representations["r2"], representations["r12"]
        self.log("representation/r12_r1_cosine", F.cosine_similarity(r12, r1).mean())
        self.log("representation/r12_r2_cosine", F.cosine_similarity(r12, r2).mean())
        if self.hparams.model_type == "synergy":
            self.log("composition/interaction_gate", self.model.composition.gate)
        self.validation_representations.clear()

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def f1_scores(predictions: Tensor, labels: Tensor, classes: int = 10) -> tuple[float, float]:
    scores, supports = [], []
    for label in range(classes):
        true_positive = ((predictions == label) & (labels == label)).sum().item()
        false_positive = ((predictions == label) & (labels != label)).sum().item()
        false_negative = ((predictions != label) & (labels == label)).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        supports.append((labels == label).sum().item())
    return sum(scores) / classes, sum(score * support for score, support in zip(scores, supports)) / len(labels)


def representation_metrics(representations: dict[str, Tensor]) -> dict[str, float]:
    result = {}
    for name, value in representations.items():
        result[f"{name}_norm"] = value.norm(dim=1).mean().item()
        result[f"{name}_variance"] = value.var(dim=0, unbiased=False).mean().item()
        result[f"{name}_effective_rank"] = effective_rank(value)
    result["r12_r1_cosine"] = F.cosine_similarity(representations["r12"], representations["r1"]).mean().item()
    result["r12_r2_cosine"] = F.cosine_similarity(representations["r12"], representations["r2"]).mean().item()
    weaker_variance = min(result["r1_variance"], result["r2_variance"])
    result["r12_to_weaker_variance_ratio"] = result["r12_variance"] / max(weaker_variance, 1e-12)
    return result


def correction_regression(base: Tensor, full: Tensor, labels: Tensor) -> dict[str, object]:
    correction = (base != labels) & (full == labels)
    regression = (base == labels) & (full != labels)
    both_correct = (base == labels) & (full == labels)
    both_wrong = (base != labels) & (full != labels)

    def counts(mask: Tensor) -> dict[str, int]:
        return {str(label): int((mask & (labels == label)).sum()) for label in range(10)}

    return {
        "correction": int(correction.sum()),
        "regression": int(regression.sum()),
        "net_correction": int(correction.sum() - regression.sum()),
        "both_correct": int(both_correct.sum()),
        "both_wrong": int(both_wrong.sum()),
        "per_class": {
            "correction": counts(correction),
            "regression": counts(regression),
            "net_correction": {
                str(label): counts(correction)[str(label)] - counts(regression)[str(label)]
                for label in range(10)
            },
        },
    }


@torch.no_grad()
def evaluate(
    model: AVMNISTBimodalExperiment,
    dataloader: object,
    gate_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, object]:
    model.eval()
    logits_all: dict[str, list[Tensor]] = defaultdict(list)
    labels_all: list[Tensor] = []
    stored: dict[str, list[Tensor]] = defaultdict(list)
    generator = torch.Generator(device=model.device).manual_seed(0)
    for images, audios, _, labels in dataloader:
        images, audios = images.to(model.device), audios.to(model.device)
        outputs = model(images, audios)
        additive = model.model.additive(outputs["r1"], outputs["r2"])
        candidates = {"normal": outputs["fused"], "additive": additive}
        if model.hparams.model_type == "synergy":
            candidates["zero"] = model.model.compose(outputs["r1"], outputs["r2"], torch.zeros_like(outputs["r12"]))
            shuffled = outputs["r12"][torch.randperm(labels.shape[0], device=model.device, generator=generator)]
            candidates["shuffle"] = model.model.compose(outputs["r1"], outputs["r2"], shuffled)
            permutation1 = torch.randperm(labels.shape[0], device=model.device, generator=generator)
            permutation2 = torch.randperm(labels.shape[0], device=model.device, generator=generator)
            shuffled_modality1 = model.model.interaction(outputs["r1"][permutation1], outputs["r2"])
            shuffled_modality2 = model.model.interaction(outputs["r1"], outputs["r2"][permutation2])
            shuffled_both = model.model.interaction(
                outputs["r1"][permutation1], outputs["r2"][permutation2]
            )
            candidates["shuffle_modality1"] = model.model.compose(
                outputs["r1"], outputs["r2"], shuffled_modality1
            )
            candidates["shuffle_modality2"] = model.model.compose(
                outputs["r1"], outputs["r2"], shuffled_modality2
            )
            candidates["shuffle_both_modalities"] = model.model.compose(
                outputs["r1"], outputs["r2"], shuffled_both
            )
            for gate in gate_values:
                candidates[f"gate_{gate:.2f}"] = model.model.compose(
                    outputs["r1"], outputs["r2"], outputs["r12"], gate=gate
                )
        for name, representation in candidates.items():
            logits_all[name].append(model.model.classify(representation).cpu())
        for name in ("r1", "r2", "r12"):
            stored[name].append(outputs[name].cpu())
        labels_all.append(labels)

    labels = torch.cat(labels_all)
    metrics: dict[str, object] = {"representations": representation_metrics({k: torch.cat(v) for k, v in stored.items()})}
    predictions = {}
    for name, chunks in logits_all.items():
        logits = torch.cat(chunks)
        predicted = logits.argmax(1)
        predictions[name] = predicted
        macro, weighted = f1_scores(predicted, labels)
        metrics[f"{name}_accuracy"] = (predicted == labels).float().mean().item()
        metrics[f"{name}_macro_f1"] = macro
        metrics[f"{name}_weighted_f1"] = weighted
        metrics[f"{name}_loss"] = F.cross_entropy(logits, labels).item()
    metrics["gain12"] = metrics["normal_accuracy"] - metrics["additive_accuracy"]
    if model.hparams.model_type == "synergy":
        metrics["interaction_shuffle_drop"] = metrics["normal_accuracy"] - metrics["shuffle_accuracy"]
        metrics["interaction_zero_drop"] = metrics["normal_accuracy"] - metrics["zero_accuracy"]
        metrics["shuffle_modality1_drop"] = metrics["normal_accuracy"] - metrics["shuffle_modality1_accuracy"]
        metrics["shuffle_modality2_drop"] = metrics["normal_accuracy"] - metrics["shuffle_modality2_accuracy"]
        metrics["shuffle_both_modalities_drop"] = (
            metrics["normal_accuracy"] - metrics["shuffle_both_modalities_accuracy"]
        )
        metrics["interaction_gate"] = model.model.composition.gate.item()
        metrics["correction_regression"] = correction_regression(
            predictions["additive"], predictions["normal"], labels
        )
        metrics["gate_sweep"] = {
            f"{gate:.2f}": {
                "accuracy": metrics[f"gate_{gate:.2f}_accuracy"],
                "macro_f1": metrics[f"gate_{gate:.2f}_macro_f1"],
                "loss": metrics[f"gate_{gate:.2f}_loss"],
                **correction_regression(
                    predictions["additive"], predictions[f"gate_{gate:.2f}"], labels
                ),
            }
            for gate in gate_values
        }
    return metrics


@torch.no_grad()
def extract_representations(model: AVMNISTBimodalExperiment, dataloader: object) -> tuple[dict[str, Tensor], Tensor]:
    model.eval()
    stored: dict[str, list[Tensor]] = defaultdict(list)
    labels_all = []
    for images, audios, _, labels in dataloader:
        outputs = model(images.to(model.device), audios.to(model.device))
        for name in ("r1", "r2", "r12"):
            stored[name].append(outputs[name].cpu())
        labels_all.append(labels)
    return {name: torch.cat(chunks) for name, chunks in stored.items()}, torch.cat(labels_all)


def linear_probes(
    model: AVMNISTBimodalExperiment,
    train_dataloader: object,
    test_dataloader: object,
    max_train_samples: int,
    conditional_only: bool = False,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, train_labels = extract_representations(model, train_dataloader)
    test, test_labels = extract_representations(model, test_dataloader)
    features = {
        "r1": (train["r1"], test["r1"]),
        "r2": (train["r2"], test["r2"]),
        "r12": (train["r12"], test["r12"]),
        "concat_r1_r2": (torch.cat((train["r1"], train["r2"]), 1), torch.cat((test["r1"], test["r2"]), 1)),
        "concat_all": (torch.cat(tuple(train.values()), 1), torch.cat(tuple(test.values()), 1)),
    }
    if conditional_only:
        features = {name: features[name] for name in ("concat_r1_r2", "concat_all")}
    size = min(max_train_samples, len(train_labels))
    indices = torch.randperm(len(train_labels), generator=torch.Generator().manual_seed(0))[:size]
    result = {}
    for name, (train_x, test_x) in features.items():
        probe = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, tol=1e-3, random_state=0)
        )
        probe.fit(train_x[indices].numpy(), train_labels[indices].numpy())
        result[name] = probe.score(test_x.numpy(), test_labels.numpy())
    result["gain_all_over_r1_r2"] = result["concat_all"] - result["concat_r1_r2"]
    return result


def nonlinear_probes(
    model: AVMNISTBimodalExperiment,
    train_dataloader: object,
    test_dataloader: object,
    max_train_samples: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, float | int]:
    train, train_labels = extract_representations(model, train_dataloader)
    test, test_labels = extract_representations(model, test_dataloader)
    size = min(max_train_samples, len(train_labels))
    indices = torch.randperm(len(train_labels), generator=torch.Generator().manual_seed(seed))[:size]
    zeros_train, zeros_test = torch.zeros_like(train["r12"]), torch.zeros_like(test["r12"])
    feature_pairs = {
        "base": (
            torch.cat((train["r1"], train["r2"], zeros_train), 1)[indices],
            torch.cat((test["r1"], test["r2"], zeros_test), 1),
        ),
        "full": (
            torch.cat((train["r1"], train["r2"], train["r12"]), 1)[indices],
            torch.cat((test["r1"], test["r2"], test["r12"]), 1),
        ),
    }
    selected_labels = train_labels[indices]
    result: dict[str, float | int] = {}
    for name, (train_x, test_x) in feature_pairs.items():
        mean, std = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6)
        train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
        torch.manual_seed(seed)
        probe = nn.Sequential(
            nn.Linear(train_x.shape[1], hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 10)
        ).to(model.device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        generator = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            permutation = torch.randperm(size, generator=generator)
            probe.train()
            for batch_indices in permutation.split(512):
                logits = probe(train_x[batch_indices].to(model.device))
                loss = F.cross_entropy(logits, selected_labels[batch_indices].to(model.device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        probe.eval()
        with torch.no_grad():
            predicted = torch.cat(
                [probe(chunk.to(model.device)).argmax(1).cpu() for chunk in test_x.split(512)]
            )
        result[name] = (predicted == test_labels).float().mean().item()
        result[f"{name}_parameters"] = sum(parameter.numel() for parameter in probe.parameters())
    result["gain"] = float(result["full"]) - float(result["base"])
    return result


def interaction_predictability_probes(
    model: AVMNISTBimodalExperiment,
    train_dataloader: object,
    test_dataloader: object,
    max_train_samples: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Measure how much of ``r12`` a nonlinear predictor can recover from lower orders."""
    train, _ = extract_representations(model, train_dataloader)
    test, _ = extract_representations(model, test_dataloader)
    size = min(max_train_samples, len(train["r12"]))
    indices = torch.randperm(len(train["r12"]), generator=torch.Generator().manual_seed(seed))[:size]
    target_train, target_test = train["r12"][indices], test["r12"]
    target_mean = target_train.mean(0)
    target_std = target_train.std(0, unbiased=False).clamp_min(1e-6)
    target_train = (target_train - target_mean) / target_std
    sources = {
        "r1": (train["r1"][indices], test["r1"]),
        "r2": (train["r2"][indices], test["r2"]),
        "r1_r2": (torch.cat((train["r1"], train["r2"]), 1)[indices], torch.cat((test["r1"], test["r2"]), 1)),
    }
    result = {}
    for name, (train_x, test_x) in sources.items():
        mean, std = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6)
        train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
        torch.manual_seed(seed)
        predictor = nn.Sequential(
            nn.Linear(train_x.shape[1], hidden_dim), nn.GELU(), nn.Linear(hidden_dim, target_train.shape[1])
        ).to(model.device)
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=weight_decay)
        generator = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            for batch_indices in torch.randperm(size, generator=generator).split(512):
                predicted = predictor(train_x[batch_indices].to(model.device))
                loss = F.mse_loss(predicted, target_train[batch_indices].to(model.device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        predictor.eval()
        with torch.no_grad():
            predicted = torch.cat([predictor(chunk.to(model.device)).cpu() for chunk in test_x.split(512)])
        target = (target_test - target_mean) / target_std
        mse = F.mse_loss(predicted, target).item()
        result[name] = {
            "normalized_mse": mse,
            "r2": 1.0 - mse,
            "cosine": F.cosine_similarity(
                predicted * target_std + target_mean, target_test, dim=1
            ).mean().item(),
        }
    return result


def git_output(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True, check=False).stdout.strip()


def checkpoint_epoch(path: str) -> int | None:
    match = re.search(r"epoch=(\d+)", path)
    return int(match.group(1)) if match else None


@hydra.main(config_path="../../../configs", config_name="av_mnist", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    datamodule = AVMNISTDataModule(cfg.data_path, cfg.batch_size, cfg.num_workers)
    datamodule.setup()
    checkpoint_directory = os.path.join(
        cfg.results_path, "checkpoints", cfg.experiment_name, cfg.model_type, f"seed_{cfg.seed}"
    )
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_directory,
        monitor="val/accuracy",
        mode="max",
        filename="best-{epoch:02d}",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val/accuracy", mode="max", patience=cfg.early_stopping_patience, min_delta=cfg.early_stopping_min_delta
    )
    logger = CSVLogger(cfg.log_path, name=f"{cfg.experiment_name}/{cfg.model_type}", version=f"seed_{cfg.seed}")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.precision,
        deterministic=True,
        gradient_clip_val=1.0,
        callbacks=[checkpoint, early_stopping],
        logger=logger,
        enable_progress_bar=False,
    )
    model = AVMNISTBimodalExperiment(
        model_type=cfg.model_type,
        embed_dim=cfg.embed_dim,
        rank=cfg.rank,
        fusion_hidden_dim=cfg.fusion_hidden_dim,
        temperature=cfg.temperature,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        lambda_unimodal=cfg.lambda_unimodal,
        lambda_variance=cfg.lambda_variance,
        lambda_cross_covariance=cfg.lambda_cross_covariance,
        variance_gamma=cfg.variance_gamma,
        diagnostics_samples=cfg.diagnostics_samples,
        lambda_covariance=cfg.lambda_covariance,
        interaction_factor_bias=cfg.interaction_factor_bias,
        interaction_output_bias=cfg.interaction_output_bias,
        lambda_utility=cfg.lambda_utility,
        utility_margin=cfg.utility_margin,
        base_first_composition=cfg.base_first_composition,
        lambda_dependency=cfg.lambda_dependency,
        dependency_margin=cfg.dependency_margin,
    )
    started_at = time.perf_counter()
    trainer.fit(model, datamodule)
    training_duration_seconds = time.perf_counter() - started_at
    best_model = AVMNISTBimodalExperiment.load_from_checkpoint(checkpoint.best_model_path).to(model.device)
    metrics = evaluate(best_model, datamodule.test_dataloader(), tuple(cfg.gate_values))
    probe_train_loader = DataLoader(
        datamodule.train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    probes = (
        linear_probes(
            best_model,
            probe_train_loader,
            datamodule.test_dataloader(),
            cfg.probe_train_samples,
            cfg.get("probe_conditional_only", False),
        )
        if cfg.run_linear_probes and cfg.model_type == "synergy" else {}
    )
    nonlinear = (
        nonlinear_probes(
            best_model,
            probe_train_loader,
            datamodule.test_dataloader(),
            cfg.probe_train_samples,
            cfg.nonlinear_probe_hidden_dim,
            cfg.nonlinear_probe_epochs,
            cfg.nonlinear_probe_lr,
            cfg.nonlinear_probe_weight_decay,
            cfg.seed,
        )
        if cfg.run_nonlinear_probes and cfg.model_type == "synergy" else {}
    )
    correction = metrics.get("correction_regression", {})
    result = {
        "experiment": cfg.experiment_name,
        "model": cfg.model_type,
        "seed": cfg.seed,
        "accuracy": metrics["normal_accuracy"],
        "macro_f1": metrics["normal_macro_f1"],
        "weighted_f1": metrics["normal_weighted_f1"],
        "gain12": metrics["gain12"],
        "zero_drop12": metrics.get("interaction_zero_drop"),
        "shuffle_drop12": metrics.get("interaction_shuffle_drop"),
        "shuffle_modality1_drop": metrics.get("shuffle_modality1_drop"),
        "shuffle_modality2_drop": metrics.get("shuffle_modality2_drop"),
        "correction": correction.get("correction"),
        "regression": correction.get("regression"),
        "net_correction": correction.get("net_correction"),
        "linear_probe_gain": probes.get("gain_all_over_r1_r2"),
        "nonlinear_probe_gain": nonlinear.get("gain"),
        "best_epoch": checkpoint_epoch(checkpoint.best_model_path),
        "best_validation_accuracy": checkpoint.best_model_score.item(),
        "metrics": metrics,
        "linear_probes": probes,
        "nonlinear_probes": nonlinear,
        "checkpoint": checkpoint.best_model_path,
        "parameters": sum(parameter.numel() for parameter in best_model.parameters()),
        "dataset": "MultiBench AV-MNIST (Google Drive id 1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp)",
        "split": "MultiBench official train/validation/test split",
        "training_duration_seconds": training_duration_seconds,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    result_path = os.path.join(
        cfg.results_path, cfg.experiment_name, f"{cfg.model_type}_seed_{cfg.seed}.json"
    )
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
