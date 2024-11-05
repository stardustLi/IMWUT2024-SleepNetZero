import os
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from pathlib import Path
import rich
import torch
import torch.nn as nn
import typing as t

from data.spo2 import Spo2Batch, Spo2NetInput
from metrics.regression import RegressionMetrics
from wuji_dl.optim import SingleOptimizerFactory
from wuji_dl.factory import create_instance
import numpy as np


class Spo2Task(L.LightningModule):
    def __init__(
        self,
        model: dict[str, t.Any],
        criterion: dict[str, t.Any],
        optimizer: dict[str, t.Any],
        lr_scheduler: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = create_instance(nn.Module, model)
        self.criterion = create_instance(nn.Module, criterion)
        self.optim = SingleOptimizerFactory(optimizer, lr_scheduler)
        self.metrics = RegressionMetrics()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optim(self.parameters())

    def forward(self, net_input: Spo2NetInput):
        return self.model(**net_input)

    def on_train_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self)

    def training_step(
        self, batch: Spo2Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        logits = self.forward(batch["net_input"])
        loss = self.criterion(logits, batch["spo2_burden"])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def setup(self, stage: str):
        PREFIX_MAPPING = {
            "fit": "val/",
            "test": "test/",
        }
        prefix = PREFIX_MAPPING.get(stage, "")
        self.metrics.prefix = prefix

    def validation_step(
        self, batch: Spo2Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._common_eval_step(batch)

    def test_step(
        self, batch: Spo2Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._common_eval_step(batch)


    def log_curve(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        if (
            isinstance(self.logger, WandbLogger)
            and self.logger.save_dir is not None
            and self.logger.name is not None
            and self.logger.version is not None
        ):
            path = Path(
                self.logger.save_dir, self.logger.name, self.logger.version, "npz"
            )
            os.makedirs(path, exist_ok=True)
            np.savez(
                Path(path, f"{self.current_epoch}"),
                **{
                    "pred": pred.cpu().to(dtype=torch.float32).numpy(),
                    "label": label.cpu().to(dtype=torch.float32).numpy(),
                },
            )

    @torch.no_grad()
    def _common_eval_step(self, batch: Spo2Batch) -> None:
        pred: torch.Tensor = self.forward(batch["net_input"])
        label = batch["spo2_burden"]
        gt = batch["spo2"]
        mask = ~torch.isnan(label)
        _ = self.metrics(pred[mask], label[mask])
        loss = self.criterion(pred, label)
        state = "val" if self.trainer.evaluating else "test"
        self.log_dict(
            {
                f"{state}/loss": loss,
            }
        )
        self.log_dict(self.metrics.loggable_items())
        self.log_dict(self.metrics.yield_event_based_matrix(pred, label))
        self.log_curve(pred, label)
