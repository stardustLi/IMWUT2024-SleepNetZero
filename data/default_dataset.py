from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import DataLoader
import typing as t
from wuji_dl.data import BaseDataset
import wuji_dl.ops.functional as wf


@dataclass
class SampleIndex:
    id: int | str
    path: str
    start: int
    end: int
    payload: dict = field(default_factory=lambda: {})


@dataclass
class Sample:
    id: int | str
    length: int
    payload: dict


def default_extractor(name: str, frames_per_token: int):
    def extract(npz, start: int, end: int):
        return torch.from_numpy(
            npz[name][start * frames_per_token : end * frames_per_token]
        )

    return extract


def pad_collator(name: str, **kwargs):
    def collate(samples: list[Sample]):
        return wf.pad_batch([s.payload[name] for s in samples], **kwargs)

    return collate


class DefaultDataset(BaseDataset):
    """
    General dataset behavior comes here
    """

    def __init__(
        self,
        data: t.Sequence[SampleIndex],
        extractors: t.Mapping[str, t.Callable],
        collators: t.Mapping[str, tuple[bool, t.Callable]],
        dataloader_config: t.Mapping[str, t.Any],
    ) -> None:
        """
        Args:
            data: list of SampleIndex
            extractors: mapping, key : ((NpzFile, start, end) -> fetched value)
            collators: mapping, key : (is_input, (list[Sample]) -> batched value)
            dataloader_config: DataLoader kwargs except dataset and collate_fn
        """
        self.data = data
        self.extractors = extractors
        self.collators = collators
        self.dataloader_config = dataloader_config

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Sample:
        src = self.data[idx]
        with np.load(src.path) as npz:
            payload = {
                key: fn(npz, src.start, src.end) for key, fn in self.extractors.items()
            }
        payload.update(src.payload)
        return Sample(id=src.id, length=src.end - src.start, payload=payload)

    def dataloader(self) -> DataLoader:
        def collate_fn(samples: list[Sample]):
            batch: dict = {"id": [s.id for s in samples]}
            net_input = {
                "length": torch.tensor([s.length for s in samples]),
            }
            for key, (is_input, fn) in self.collators.items():
                if is_input:
                    net_input[key] = fn(samples)
                else:
                    batch[key] = fn(samples)
            batch["net_input"] = net_input
            return batch

        return DataLoader(self, **self.dataloader_config, collate_fn=collate_fn)
