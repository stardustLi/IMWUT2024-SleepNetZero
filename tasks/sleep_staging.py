import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import pandas as pd
import rich
import torch
import torch.nn as nn
import typing as t
from wuji_dl.display import df_to_rich, df_to_wandb
from wuji_dl.factory import create_instance
from wuji_dl.metrics import ClassificationMetrics
from wuji_dl.optim import SingleOptimizerFactory

from data.sleep_staging import SleepStagingBatch, SleepStagingNetInput, PAD_STAGE


class SleepStagingTask(L.LightningModule):
    def __init__(
        self,
        model: dict[str, t.Any],
        criterion: dict[str, t.Any],
        optimizer: dict[str, t.Any],
        lr_scheduler: dict[str, t.Any] | None = None,
        stages: list[str] = ["W", "N1", "N2", "N3", "R"],
        test_group: str = "test",
    ) -> None:
        super().__init__()
        self.model = create_instance(nn.Module, model)
        self.criterion = create_instance(nn.Module, criterion)
        self.optim = SingleOptimizerFactory(optimizer, lr_scheduler)
        self.metrics = ClassificationMetrics(stages, ignore_index=PAD_STAGE)
        self.test_group = test_group
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optim(self.parameters())

    def forward(self, net_input: SleepStagingNetInput):
        return self.model(**net_input)

    def on_train_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self)

    def training_step(
        self, batch: SleepStagingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        logits = self.forward(batch["net_input"])
        loss = self.criterion(logits, batch["stage"])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def setup(self, stage: str):
        PREFIX_MAPPING = {
            "fit": "val/",
            "test": self.test_group + "/",
        }
        prefix = PREFIX_MAPPING.get(stage, "")
        self.metrics.prefix = prefix

    def validation_step(
        self, batch: SleepStagingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._eval_step(batch)

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            self._on_eval_epoch_end()

    def test_step(
        self, batch: SleepStagingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._eval_step(batch)

    def on_test_epoch_end(self) -> None:
        self._on_eval_epoch_end()

    @torch.no_grad()
    def _eval_step(self, batch: SleepStagingBatch) -> None:
        logits: torch.Tensor = self.forward(batch["net_input"])
        label = batch["stage"]
        _loss = self.criterion(logits, label)
        self.log(f"{self.metrics.prefix}loss", _loss)
        _ = self.metrics(logits.flatten(0, 1), label.flatten(0, 1))
        self.log_dict(self.metrics.loggable_items())

    def _on_eval_epoch_end(self) -> None:
        self._report_precision()
        self._report_recall()
        self._report_confusion_matrix()

    @rank_zero_only
    def log_table(self, key: str, val: pd.DataFrame, corner: str) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.log_metrics(
                {key: df_to_wandb(val, corner=corner)},  # type: ignore
                step=self.global_step,
            )
        rich.print(df_to_rich(val, corner=corner))

    @rank_zero_only
    def _report_precision(self) -> None:
        key, val = self.metrics.yield_precision()
        self.log_table(key, val.to_frame().T, corner="stage")

    @rank_zero_only
    def _report_recall(self) -> None:
        key, val = self.metrics.yield_recall()
        self.log_table(key, val.to_frame().T, corner="stage")

    @rank_zero_only
    def _report_confusion_matrix(self) -> None:
        key, val = self.metrics.yield_confusion_matrix()
        self.log_table(key, val, corner="T\\P")
