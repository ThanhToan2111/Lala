"""AV-MNIST Stage 1: S1 adversarial de-shortcutting (negative-control benchmark).

Extends the canonical frozen AV-MNIST synergy protocol with the
representation-level S1/S2 objectives from the ConFu++
Representation-Level Complementarity specification:

    S1  adversarial hinge penalty: r12 must be unpredictable from r1
        alone and from r2 alone (alternating minimax, manual
        optimization), warmup-scheduled.
    S2  per-factor variance floor on the low-rank interaction factors.

AV-MNIST audio carries little conditional label information, so the
expected HONEST outcome is reduced single-modality predictability
without fabricated audio dependence and without a large accuracy cost.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
import platform
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
from src.experiments.av_mnist.synergyformer import (
    BimodalFusion,
    checkpoint_epoch,
    effective_rank,
    evaluate,
    git_output,
    interaction_predictability_probes,
    linear_probes,
    nonlinear_probes,
)
from src.modules.models.complementarity import (
    AdversarialPredictor,
    EMAStandardizer,
    adversary_loss,
    factor_variance_loss,
    shortcut_hinge_loss,
)
from src.modules.models.synergyformer import (
    conditional_utility_loss,
    covariance_loss,
    pair_cross_covariance_loss,
    variance_loss,
)


class AVMNISTS1Experiment(pl.LightningModule):
    """Canonical synergy model + S1/S2 objectives with manual optimization."""

    def __init__(
        self,
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
        lambda_shortcut: float = 0.0,
        shortcut_tau: float = 0.5,
        k_adv: int = 1,
        adv_lr_scale: float = 0.5,
        adv_hidden: int = 256,
        adv_warmup_fraction: float = 0.2,
        lambda_factorvar: float = 0.5,
        lambda_hard: float = 0.0,
        hard_floor: float = 0.3,
        max_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams.model_type = "synergy"  # evaluate()/probes branch on this
        self.automatic_optimization = False
        self.model = BimodalFusion(
            "synergy",
            embed_dim,
            rank,
            fusion_hidden_dim,
            temperature,
            interaction_factor_bias,
            interaction_output_bias,
            base_first_composition,
        )
        self.adversary_left = AdversarialPredictor(embed_dim, adv_hidden)
        self.adversary_right = AdversarialPredictor(embed_dim, adv_hidden)
        self.standardizer = EMAStandardizer(embed_dim)
        self.validation_representations: dict[str, list[Tensor]] = defaultdict(list)

    def forward(self, images: Tensor, audios: Tensor) -> dict[str, Tensor]:
        return self.model(images, audios)

    def _forward_losses(self, batch: tuple) -> dict[str, Tensor]:
        images, audios, _, labels = batch
        outputs = self(images, audios)
        full_logits = self.model.classify(outputs["fused"])
        full_losses = F.cross_entropy(full_logits, labels, reduction="none")
        base_losses = F.cross_entropy(
            self.model.classify(self.model.additive(outputs["r1"], outputs["r2"])),
            labels,
            reduction="none",
        )
        unimodal_loss = sum(
            F.cross_entropy(self.model.classify(outputs[name]), labels) for name in ("r1", "r2")
        )
        return {
            "outputs": outputs,
            "labels": labels,
            "fusion": full_losses.mean(),
            "base": base_losses.mean(),
            "full_losses": full_losses,
            "base_losses": base_losses,
            "unimodal": unimodal_loss,
            "variance": variance_loss(outputs["r12"], self.hparams.variance_gamma),
            "cross_covariance": pair_cross_covariance_loss(outputs["r12"], outputs["r1"], outputs["r2"]),
            "covariance": covariance_loss(outputs["r12"]),
            "utility": conditional_utility_loss(full_losses, base_losses, self.hparams.utility_margin),
            "accuracy": (full_logits.argmax(1) == labels).float().mean(),
        }

    def training_step(self, batch: tuple, batch_idx: int) -> None:
        parts = self._forward_losses(batch)
        outputs, labels = parts["outputs"], parts["labels"]
        opt_gen, opt_adv = self.optimizers()
        hp = self.hparams

        # ---------------- adversary step (gradients to q1, q2 only) ----------------
        self.standardizer.update(outputs["r12"])
        shortcut_penalty = outputs["fused"].new_zeros(())
        gmse1 = gmse2 = outputs["fused"].new_zeros(())
        if hp.lambda_shortcut > 0:
            target_const = self.standardizer.standardize(outputs["r12"].detach())
            for _ in range(hp.k_adv):
                loss_adv, mse1, mse2 = adversary_loss(
                    self.adversary_left,
                    self.adversary_right,
                    outputs["r1"].detach(),
                    outputs["r2"].detach(),
                    target_const,
                )
                opt_adv.zero_grad()
                self.manual_backward(loss_adv)
                opt_adv.step()
            self.log("adv/mse1", mse1, batch_size=labels.shape[0])
            self.log("adv/mse2", mse2, batch_size=labels.shape[0])
            self.log("adv/r2_1", 1 - mse1, batch_size=labels.shape[0])
            self.log("adv/r2_2", 1 - mse2, batch_size=labels.shape[0])

            # ------------- generator-side hinge penalty (gradients via r12) -------------
            with torch.no_grad():
                pred_left = self.adversary_left(outputs["r1"])
                pred_right = self.adversary_right(outputs["r2"])
            target_grad = self.standardizer.standardize(outputs["r12"])
            shortcut_penalty, gmse1, gmse2 = shortcut_hinge_loss(
                pred_left, pred_right, target_grad, hp.shortcut_tau
            )
            warmup = min(1.0, (self.current_epoch + 1) / max(1.0, hp.adv_warmup_fraction * hp.max_epochs))
            shortcut_penalty = (hp.lambda_shortcut * warmup) * shortcut_penalty
            self.log("shortcut/warmup", warmup, batch_size=labels.shape[0])

        # ---------------- factor health (S2) ----------------
        factorvar = outputs["fused"].new_zeros(())
        if hp.lambda_shortcut > 0 and hp.lambda_factorvar > 0:
            factors = self.model.interaction.project_factors(outputs["r1"], outputs["r2"])
            factorvar = hp.lambda_factorvar * factor_variance_loss(factors)

        # ---------------- generator step ----------------
        if hp.lambda_hard > 0:
            # S4 hard-fraction emphasis: spend capacity where lower-order
            # evidence fails; stop-grad weights, floor keeps easy samples alive.
            w = parts["base_losses"].detach()
            w = w + hp.hard_floor * w.mean()
            hard_term = (w * parts["full_losses"]).sum() / w.sum().clamp_min(1e-6)
            task_term = (1 - hp.lambda_hard) * parts["fusion"] + hp.lambda_hard * hard_term
        else:
            task_term = parts["fusion"]
        loss = (
            task_term
            + hp.lambda_unimodal * parts["unimodal"]
            + hp.lambda_variance * parts["variance"]
            + hp.lambda_cross_covariance * parts["cross_covariance"]
            + hp.lambda_covariance * parts["covariance"]
            + hp.lambda_utility * parts["utility"]
            + shortcut_penalty
            + factorvar
        )
        opt_gen.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt_gen, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_gen.step()

        batch_size = labels.shape[0]
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/accuracy", parts["accuracy"], on_epoch=True, prog_bar=True, batch_size=batch_size)
        for name, value in {
            "fusion": parts["fusion"],
            "base": parts["base"],
            "unimodal": parts["unimodal"],
            "variance": parts["variance"],
            "cross_covariance": parts["cross_covariance"],
            "covariance": parts["covariance"],
            "utility": parts["utility"],
            "shortcut_weighted": shortcut_penalty,
            "factorvar_weighted": factorvar,
            "hard_task": task_term,
        }.items():
            self.log(f"train/loss_{name}", value, on_epoch=True, batch_size=batch_size)

    def on_train_epoch_end(self) -> None:
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        parts = self._forward_losses(batch)
        batch_size = parts["labels"].shape[0]
        self.log("val/loss", parts["fusion"], on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/accuracy", parts["accuracy"], on_epoch=True, prog_bar=True, batch_size=batch_size)
        collected = sum(chunk.shape[0] for chunk in self.validation_representations["r1"])
        if collected < self.hparams.diagnostics_samples:
            count = self.hparams.diagnostics_samples - collected
            for name in ("r1", "r2", "r12"):
                self.validation_representations[name].append(parts["outputs"][name][:count].detach().cpu())

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
        self.log("composition/interaction_gate", self.model.composition.gate)
        self.validation_representations.clear()

    def configure_optimizers(self):
        opt_gen = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        opt_adv = torch.optim.AdamW(
            list(self.adversary_left.parameters()) + list(self.adversary_right.parameters()),
            lr=self.hparams.lr * self.hparams.adv_lr_scale,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(opt_gen, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return [opt_gen, opt_adv], [{"scheduler": scheduler, "interval": "epoch"}]


@hydra.main(config_path="../../../configs", config_name="av_mnist_complementarity", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    datamodule = AVMNISTDataModule(cfg.data_path, cfg.batch_size, cfg.num_workers)
    datamodule.setup()
    checkpoint_directory = os.path.join(
        cfg.results_path, "checkpoints", cfg.experiment_name, "synergy_s1", f"seed_{cfg.seed}"
    )
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_directory,
        monitor="val/accuracy",
        mode="max",
        filename="best-{epoch:02d}",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val/accuracy", mode="max", patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta
    )
    logger = CSVLogger(cfg.log_path, name=f"{cfg.experiment_name}/synergy_s1", version=f"seed_{cfg.seed}")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.precision,
        deterministic=True,
        callbacks=[checkpoint, early_stopping],
        logger=logger,
        enable_progress_bar=False,
    )
    model = AVMNISTS1Experiment(
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
        lambda_shortcut=cfg.lambda_shortcut,
        shortcut_tau=cfg.shortcut_tau,
        k_adv=cfg.k_adv,
        adv_lr_scale=cfg.adv_lr_scale,
        adv_hidden=cfg.adv_hidden,
        adv_warmup_fraction=cfg.adv_warmup_fraction,
        lambda_factorvar=cfg.lambda_factorvar,
        lambda_hard=cfg.get("lambda_hard", 0.0),
        hard_floor=cfg.get("hard_floor", 0.3),
        max_epochs=cfg.max_epochs,
    )
    started_at = time.perf_counter()
    trainer.fit(model, datamodule)
    training_duration_seconds = time.perf_counter() - started_at
    best_model = AVMNISTS1Experiment.load_from_checkpoint(checkpoint.best_model_path).to(model.device)
    metrics = evaluate(best_model, datamodule.test_dataloader(), tuple(cfg.gate_values))
    probe_train_loader = DataLoader(
        datamodule.train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    probes = (
        linear_probes(
            best_model, probe_train_loader, datamodule.test_dataloader(),
            cfg.probe_train_samples, cfg.get("probe_conditional_only", False),
        )
        if cfg.run_linear_probes else {}
    )
    nonlinear = (
        nonlinear_probes(
            best_model, probe_train_loader, datamodule.test_dataloader(),
            cfg.probe_train_samples, cfg.nonlinear_probe_hidden_dim,
            cfg.nonlinear_probe_epochs, cfg.nonlinear_probe_lr,
            cfg.nonlinear_probe_weight_decay, cfg.seed,
        )
        if cfg.run_nonlinear_probes else {}
    )
    predictability = (
        interaction_predictability_probes(
            best_model, probe_train_loader, datamodule.test_dataloader(),
            cfg.probe_train_samples, cfg.nonlinear_probe_hidden_dim,
            cfg.predictability_probe_epochs, cfg.nonlinear_probe_lr,
            cfg.nonlinear_probe_weight_decay, cfg.seed,
        )
        if cfg.get("run_predictability_probes", False) else {}
    )
    correction = metrics.get("correction_regression", {})
    result = {
        "experiment": cfg.experiment_name,
        "model": "synergy_s1",
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
        "predictability": predictability,
        "best_epoch": checkpoint_epoch(checkpoint.best_model_path),
        "best_validation_accuracy": checkpoint.best_model_score.item(),
        "metrics": metrics,
        "linear_probes": probes,
        "nonlinear_probes": nonlinear,
        "checkpoint": checkpoint.best_model_path,
        "parameters_inference": sum(p.numel() for p in best_model.model.parameters()),
        "parameters_train_only": sum(
            p.numel() for p in
            list(best_model.adversary_left.parameters()) + list(best_model.adversary_right.parameters())
        ),
        "dataset": "MultiBench AV-MNIST",
        "split": "MultiBench official train/validation/test split",
        "training_duration_seconds": training_duration_seconds,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    result_path = os.path.join(cfg.results_path, cfg.experiment_name, f"synergy_s1_seed_{cfg.seed}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
