"""UR-FUNNY ConFu++: original ConFu plus S1 de-shortcutting."""

from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.datasets.affect import AffectDataModule
from src.modules.encoders.transformer_model import Transformer
from src.modules.models.complementarity import (
    AdversarialPredictor,
    EMAStandardizer,
    adversary_loss,
    shortcut_hinge_loss,
)
from src.modules.models.confu import ConFu


class ConFuPlus(pl.LightningModule):
    """ConFu with training-only S1 adversaries.

    The backbone and its outputs are unchanged. The six adversaries and three
    standardizers are discarded at inference, so downstream probes see the
    same three pair representations as the original ConFu.
    """

    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        input_dim3: int,
        transformer_hid_dim: int = 300,
        embed_dim: int = 256,
        lr: float = 5e-5,
        lambda_: float = 0.5,
        mask_ratio: float = 0.0,
        fusion_hidden_dim: int = 512,
        weight_decay: float = 1e-4,
        lambda_shortcut: float = 0.1,
        shortcut_tau: float = 0.5,
        k_adv: int = 1,
        adv_lr_scale: float = 0.5,
        adv_hidden: int = 256,
        adv_warmup_fraction: float = 0.2,
        lambda_task: float = 0.1,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.backbone = ConFu(
            Transformer(n_features=input_dim1, dim=transformer_hid_dim),
            Transformer(n_features=input_dim2, dim=transformer_hid_dim),
            Transformer(n_features=input_dim3, dim=transformer_hid_dim),
            embed_dim=embed_dim,
            lr=lr,
            lambda_=lambda_,
            mask_ratio=mask_ratio,
            fusion_hidden_dim=fusion_hidden_dim,
            weight_decay=weight_decay,
        )
        self.standardizers = torch.nn.ModuleList([EMAStandardizer(embed_dim) for _ in range(3)])
        self.adversaries = torch.nn.ModuleList(
            [AdversarialPredictor(embed_dim, adv_hidden) for _ in range(6)]
        )
        self.task_head = torch.nn.Linear(embed_dim * 3, 2)

    def forward(self, image: Tensor, audio: Tensor, text: Tensor):
        return self.backbone(image, audio, text)

    def _pairs(self, outputs):
        r12, r13, r23, r1, r2, r3 = outputs
        return ((r1, r2, r12), (r1, r3, r13), (r2, r3, r23))

    def training_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch
        outputs = self(images, audios, texts)
        base_loss, _ = self.backbone.pairwise_contrastive_loss(*outputs)
        task_logits = self.task_head(torch.cat(outputs[3:], dim=-1))
        task_loss = torch.nn.functional.cross_entropy(task_logits, rest[-1])
        opt_gen, opt_adv = self.optimizers()
        hp = self.hparams

        shortcut = base_loss.new_zeros(())
        if hp.lambda_shortcut > 0:
            pair_losses = []
            adversary_terms = []
            metrics = []
            for standardizer, (left, right, interaction), predictor_left, predictor_right in zip(
                self.standardizers,
                self._pairs(outputs),
                self.adversaries[0::2],
                self.adversaries[1::2],
            ):
                standardizer.update(interaction)
                target_const = standardizer.standardize(interaction.detach())
                for _ in range(hp.k_adv):
                    adv_loss, mse_left, mse_right = adversary_loss(
                        predictor_left,
                        predictor_right,
                        left.detach(),
                        right.detach(),
                        target_const,
                    )
                    adversary_terms.append(adv_loss)
                    metrics.append((mse_left, mse_right))

                with torch.no_grad():
                    pred_left = predictor_left(left)
                    pred_right = predictor_right(right)
                target_grad = standardizer.standardize(interaction)
                penalty, gen_mse_left, gen_mse_right = shortcut_hinge_loss(
                    pred_left, pred_right, target_grad, hp.shortcut_tau
                )
                pair_losses.append(penalty)
                self.log("shortcut/mse_left", gen_mse_left, on_epoch=True, batch_size=left.shape[0])
                self.log("shortcut/mse_right", gen_mse_right, on_epoch=True, batch_size=left.shape[0])

            opt_adv.zero_grad()
            self.manual_backward(torch.stack(adversary_terms).mean())
            opt_adv.step()
            self.log("adv/mse_left", torch.stack([item[0] for item in metrics]).mean(), on_epoch=True)
            self.log("adv/mse_right", torch.stack([item[1] for item in metrics]).mean(), on_epoch=True)

            warmup = min(
                1.0,
                (self.current_epoch + 1)
                / max(1.0, hp.adv_warmup_fraction * hp.max_epochs),
            )
            shortcut = hp.lambda_shortcut * warmup * torch.stack(pair_losses).mean()
            self.log("shortcut/warmup", warmup, on_epoch=True)

        loss = base_loss + hp.lambda_task * task_loss + shortcut
        opt_gen.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt_gen, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_gen.step()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        self.log("train_base_loss", base_loss, on_epoch=True, batch_size=images.shape[0])
        self.log("train_task_loss", task_loss, on_epoch=True, batch_size=images.shape[0])
        self.log("train_shortcut", shortcut, on_epoch=True, batch_size=images.shape[0])

    def validation_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch
        outputs = self(images, audios, texts)
        base_loss, _ = self.backbone.pairwise_contrastive_loss(*outputs)
        task_logits = self.task_head(torch.cat(outputs[3:], dim=-1))
        task_loss = torch.nn.functional.cross_entropy(task_logits, rest[-1])
        loss = base_loss + self.hparams.lambda_task * task_loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        self.log("val_task_accuracy", (task_logits.argmax(1) == rest[-1]).float().mean(), on_epoch=True, batch_size=images.shape[0])
        return loss

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

    def configure_optimizers(self):
        opt_gen = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.task_head.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        opt_adv = torch.optim.AdamW(
            self.adversaries.parameters(),
            lr=self.hparams.lr * self.hparams.adv_lr_scale,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(opt_gen, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return [opt_gen, opt_adv], [{"scheduler": scheduler, "interval": "epoch"}]


@hydra.main(config_path="../../../configs", config_name="multibench", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    model = ConFuPlus(
        input_dim1=cfg.dataset.embedding.input_dim1,
        input_dim2=cfg.dataset.embedding.input_dim2,
        input_dim3=cfg.dataset.embedding.input_dim3,
        transformer_hid_dim=cfg.dataset.embedding.transformer_hid_dim,
        embed_dim=cfg.dataset.embedding.common_dim,
        lr=cfg.training.lr,
        lambda_=cfg.lambda_,
        mask_ratio=cfg.mask_ratio,
        fusion_hidden_dim=cfg.fusion_hidden_dim,
        weight_decay=cfg.weight_decay,
        lambda_shortcut=cfg.lambda_shortcut,
        shortcut_tau=cfg.shortcut_tau,
        k_adv=cfg.k_adv,
        adv_lr_scale=cfg.adv_lr_scale,
        adv_hidden=cfg.adv_hidden,
        adv_warmup_fraction=cfg.adv_warmup_fraction,
        lambda_task=cfg.lambda_task,
        max_epochs=cfg.training.max_epochs,
    )
    dm = AffectDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pickle_name=cfg.dataset.pickle_name,
        dataset_name=cfg.dataset.train_dataset,
        samples_order=[0, 1, 2],
    )
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(
            cfg.results_path,
            "checkpoints",
            cfg.dataset.train_dataset,
            "confu_plus",
            f"seed_{cfg.seed}",
        ),
        monitor="val_loss",
        filename="best_model",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint],
        logger=False,
        deterministic=True,
        enable_progress_bar=False,
    )
    trainer.fit(model, dm)
    print(f"Best model saved at: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
