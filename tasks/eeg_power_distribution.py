import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn as nn
import typing as t
from wuji_dl.factory import create_instance
from wuji_dl.ops.functional import length_mask
from wuji_dl.optim import SingleOptimizerFactory

from data.eeg_power_distribution import EEGPowerDistributionBatch, EEGPowerDistributionNetInput


class EEGPowerDistributionTask(L.LightningModule):
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
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optim(self.parameters())

    def forward(self, net_input: EEGPowerDistributionNetInput):
        return self.model(**net_input)

    def on_train_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self)

    def training_step(
        self, batch: EEGPowerDistributionBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        loss = self._get_loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(
        self, batch: EEGPowerDistributionBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        loss = self._get_loss(batch)
        self.log("val/loss", loss)

    @torch.no_grad()
    def test_step(
        self, batch: EEGPowerDistributionBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        loss = self._get_loss(batch)
        self.log("test/loss", loss)

    def _get_loss(self, batch: EEGPowerDistributionBatch) -> torch.Tensor:
        net_output: torch.Tensor = self.forward(batch["net_input"])
        length = batch["net_input"]["length"]
        mask = length_mask(net_output.size(1), length)
        return self.criterion(net_output[mask], batch["eeg_power_distribution"][mask])
