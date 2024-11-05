from collections import defaultdict
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torchmetrics as tm
import typing as t
from wuji_dl.factory import create_instance
from wuji_dl.optim import SingleOptimizerFactory

from data.diagnosis import DiagnosisBatch, DiagnosisNetInput


class DiagnosisTask(L.LightningModule):
    strict_loading: bool = False

    def __init__(
        self,
        model: dict[str, t.Any],
        criterion: dict[str, t.Any],
        optimizer: dict[str, t.Any],
        lr_scheduler: dict[str, t.Any] | None = None,
        test_group: str = "test",
        results_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.model = create_instance(nn.Module, model)
        self.criterion = create_instance(nn.Module, criterion)
        self.optim = SingleOptimizerFactory(optimizer, lr_scheduler)
        self.metrics = tm.MetricCollection(
            {
                "accuracy": tm.Accuracy("binary"),
                "cohen_kappa": tm.CohenKappa("binary"),
                "f1": tm.F1Score("binary"),
                "precision": tm.Precision("binary"),
                "recall": tm.Recall("binary"),
                "specificity": tm.Specificity("binary"),
                "roc_auc": tm.AUROC("binary"),
            }
        )
        self.test_group = test_group
        self.results_dir = results_dir
        self.save_hyperparameters(ignore="results_dir")

    def configure_optimizers(self):
        return self.optim(self.parameters())

    def forward(self, net_input: DiagnosisNetInput) -> torch.Tensor:
        return self.model(**net_input)

    def on_train_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self)

    def training_step(
        self, batch: DiagnosisBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        logits = self.forward(batch["net_input"])
        loss = self.criterion(logits, batch["label"].float())
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.metrics.prefix = "val/"
        elif stage == "test":
            self.metrics.prefix = self.test_group + "/"

    def on_validation_epoch_start(self) -> None:
        self._on_eval_epoch_start()

    def validation_step(
        self, batch: DiagnosisBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._eval_step(batch)

    def on_validation_epoch_end(self) -> None:
        self._on_eval_epoch_end()

    def on_test_epoch_start(self) -> None:
        self._on_eval_epoch_start()

    def test_step(
        self, batch: DiagnosisBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._eval_step(batch)

    def on_test_epoch_end(self) -> None:
        self._on_eval_epoch_end()

    def _on_eval_epoch_start(self) -> None:
        self.pred = defaultdict(list)
        self.true = {}

    @torch.no_grad()
    def _eval_step(self, batch: DiagnosisBatch) -> None:
        logits = self.forward(batch["net_input"])
        probs = torch.sigmoid(logits).flatten().tolist()
        labels = batch["label"].flatten().tolist()
        for i, prob, label in zip(batch["id"], probs, labels):
            self.pred[i].append(prob)
            self.true[i] = label

    def _on_eval_epoch_end(self) -> None:
        ids = list(self.true.keys())
        preds = [np.mean(self.pred[i]) for i in ids]
        labels = list(self.true.values())
        preds_tensor = torch.tensor(preds, device=self.device)
        labels_tensor = torch.tensor(labels, device=self.device)
        self.metrics(preds_tensor, labels_tensor)
        self.log_dict(self.metrics)
        self._save_results(ids, preds, labels)

    def on_predict_epoch_start(self) -> None:
        self.pred = defaultdict(list)

    @torch.no_grad()
    def predict_step(self, batch: DiagnosisBatch) -> None:
        logits = self.forward(batch["net_input"])
        probs = torch.sigmoid(logits).flatten().tolist()
        for i, prob in zip(batch["id"], probs):
            self.pred[i].append(prob)

    def on_predict_epoch_end(self) -> None:
        ids = list(self.pred.keys())
        preds = [np.mean(self.pred[i]) for i in ids]
        self._save_results(ids, preds)

    def _save_results(self, ids: list, preds: list, labels: list | None = None) -> None:
        if self.results_dir is None:
            return
        path = Path(self.results_dir)
        path.mkdir(parents=True, exist_ok=True)
        data = {"id": ids, "pred": preds}
        if labels is not None:
            data["label"] = labels
            plots = tm.MetricCollection(
                {
                    "roc": tm.ROC("binary"),
                    "confusion_matrix": tm.ConfusionMatrix("binary", normalize="true"),
                }
            )
            plots(torch.tensor(preds), torch.tensor(labels))
            plots["roc"].plot()[0].savefig(path / "roc.png")
            plots["confusion_matrix"].plot(labels=["negative", "positive"], cmap="RdYlGn")[0].savefig(
                path / "confusion_matrix.png"
            )
        pd.DataFrame(data).to_csv(path / "prediction.csv", index_label=False)
