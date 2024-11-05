from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import typing as t
from wuji_dl import slicing
from wuji_dl.data import BaseDataset
import wuji_dl.ops.functional as wf


TOKEN_SEC = 30
TOKEN_LEN = 120


@dataclass
class SampleIndex:
    id: int | str
    path: str
    start: int
    end: int
    label: int


class DiagnosisSample(t.TypedDict):
    id: int | str
    length: int
    heartbeat: torch.Tensor
    breath: torch.Tensor
    label: torch.Tensor
    body_movement: torch.Tensor


class DiagnosisNetInput(t.TypedDict):
    length: torch.Tensor
    heartbeat: torch.Tensor
    breath: torch.Tensor
    body_movement: torch.Tensor


class DiagnosisBatch(t.TypedDict):
    id: list[int | str]
    net_input: DiagnosisNetInput
    label: torch.Tensor


class DiagnosisDataset(BaseDataset):
    def __init__(
        self,
        index: str,
        split: str,
        max_tokens: int,
        stride_tokens: int = 0,  # 0 for truncation
        label: str = "label",
        **kwargs: t.Any
    ) -> None:
        self.data: list[SampleIndex] = []
        csv = pd.read_csv(index)
        csv = csv[(csv["split"] == split) & csv[label].isin([0, 1])]
        for i, (_, row) in enumerate(csv.iterrows()):
            n = int(row["duration"] // TOKEN_SEC)
            for l, r in slicing.window(n, max_tokens, stride_tokens):
                self.data.append(SampleIndex(i, row["path"], l, r, int(row[label])))
        self.dataloader_config = kwargs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> DiagnosisSample:
        src = self.data[idx]
        with np.load(src.path) as npz:
            in_slice = slice(src.start * TOKEN_LEN, src.end * TOKEN_LEN)
            heartbeat = npz["heartbeat"][in_slice]
            breath = npz["breath"][in_slice]
            body_movement = npz["body_movement"][in_slice]
        return {
            "id": src.id,
            "length": src.end - src.start,
            "heartbeat": torch.from_numpy(heartbeat),
            "breath": torch.from_numpy(breath),
            "body_movement": torch.from_numpy(body_movement),
            "label": torch.tensor([src.label]),
        }

    @staticmethod
    def _collate_fn(samples: list[DiagnosisSample]) -> DiagnosisBatch:
        net_input: DiagnosisNetInput = {
            "length": torch.tensor([s["length"] for s in samples]),
            "heartbeat": wf.pad_batch([s["heartbeat"] for s in samples]),
            "breath": wf.pad_batch([s["breath"] for s in samples]),
            "body_movement": wf.pad_batch([s["body_movement"] for s in samples]),
        }
        return {
            "id": [s["id"] for s in samples],
            "net_input": net_input,
            "label": torch.stack([s["label"] for s in samples]),
        }

    def dataloader(self) -> DataLoader:
        return DataLoader(self, **self.dataloader_config, collate_fn=self._collate_fn)


class AppDiagnosisDataset(DiagnosisDataset):
    def __init__(
        self,
        index: str,
        max_tokens: int,
        stride_tokens: int = 0,  # 0 for truncation
        **kwargs: t.Any
    ) -> None:
        self.data: list[SampleIndex] = []
        csv = pd.read_csv(index)
        for i, (_, row) in enumerate(csv.iterrows()):
            n = int(row["duration"] // TOKEN_SEC)
            for l, r in slicing.window(n, max_tokens, stride_tokens):
                self.data.append(SampleIndex(row["path"], row["path"], l, r, -1))
        self.dataloader_config = kwargs
