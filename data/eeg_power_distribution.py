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
    path: str
    start: int
    end: int


class EEGPowerDistributionSample(t.TypedDict):
    length: float
    heartbeat: torch.Tensor
    breath: torch.Tensor
    body_movement: torch.Tensor
    eeg_power_distribution: torch.Tensor


class EEGPowerDistributionNetInput(t.TypedDict):
    length: torch.Tensor
    heartbeat: torch.Tensor
    breath: torch.Tensor
    body_movement: torch.Tensor


class EEGPowerDistributionBatch(t.TypedDict):
    net_input: EEGPowerDistributionNetInput
    eeg_power_distribution: torch.Tensor


class EEGPowerDistributionDataset(BaseDataset):
    def __init__(
        self,
        index: str,
        split: str,
        max_tokens: int,
        stride_tokens: int = 0,  # 0 for truncation
        **kwargs: t.Any
    ) -> None:
        self.data: list[SampleIndex] = []
        csv = pd.read_csv(index)
        csv = csv[(csv["split"] == split) & (csv["duration"] >= TOKEN_SEC)]
        for _, row in csv.iterrows():
            n = int(row["duration"] // TOKEN_SEC)
            for l, r in slicing.window(n, max_tokens, stride_tokens):
                self.data.append(SampleIndex(row["path"], l, r))
        self.dataloader_config = kwargs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> EEGPowerDistributionSample:
        src = self.data[idx]
        with np.load(src.path) as npz:
            in_slice = slice(src.start * TOKEN_LEN, src.end * TOKEN_LEN)
            out_slice = slice(src.start, src.end)
            heartbeat = npz["heartbeat"][in_slice]
            breath = npz["breath"][in_slice]
            body_movement = npz["body_movement"][in_slice]
            eeg_power_distribution = npz["eeg_power_distribution"].T[out_slice]
        return {
            "length": src.end - src.start,
            "heartbeat": torch.from_numpy(heartbeat),
            "breath": torch.from_numpy(breath),
            "body_movement": torch.from_numpy(body_movement),
            "eeg_power_distribution": torch.from_numpy(eeg_power_distribution),
        }

    @staticmethod
    def _collate_fn(
        samples: list[EEGPowerDistributionSample],
    ) -> EEGPowerDistributionBatch:
        net_input: EEGPowerDistributionNetInput = {
            "length": torch.tensor([s["length"] for s in samples]),
            "heartbeat": wf.pad_batch([s["heartbeat"] for s in samples]),
            "breath": wf.pad_batch([s["breath"] for s in samples]),
            "body_movement": wf.pad_batch([s["body_movement"] for s in samples]),
        }
        return {
            "net_input": net_input,
            "eeg_power_distribution": wf.pad_batch(
                [s["eeg_power_distribution"] for s in samples]
            ),
        }

    def dataloader(self) -> DataLoader:
        return DataLoader(self, **self.dataloader_config, collate_fn=self._collate_fn)
