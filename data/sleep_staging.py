import pandas as pd
import torch
import typing as t
from wuji_dl import slicing

from data.default_dataset import *
from data.default_dataset import Sample


class SleepStagingNetInput(t.TypedDict):
    """
    task interface
    """

    length: torch.Tensor
    heartbeat: t.NotRequired[torch.Tensor]
    breath: t.NotRequired[torch.Tensor]
    body_movement: t.NotRequired[torch.Tensor]
    eeg: t.NotRequired[torch.Tensor]


class SleepStagingBatch(t.TypedDict):
    """
    task interface
    """

    id: list[int | str]
    net_input: SleepStagingNetInput
    stage: torch.Tensor


TOKEN_SEC = 30
TOKEN_LEN = TOKEN_SEC * 4
PAD_STAGE = -1


class SleepStagingDataset(DefaultDataset):
    def __init__(
        self,
        index: str,
        split: str,
        max_tokens: int,
        stride_tokens: int = 0,  # 0 for truncation
        use_legacy_body_movement: bool = False,
        **kwargs: t.Any,
    ) -> None:
        csv = pd.read_csv(index)
        csv = csv[(csv["split"] == split) & (csv["duration"] >= TOKEN_SEC)]
        data: list[SampleIndex] = []
        for i, (_, row) in enumerate(csv.iterrows()):
            n = int(row["duration"] // TOKEN_SEC)
            for l, r in slicing.window(n, max_tokens, stride_tokens):
                data.append(SampleIndex(id=i, path=row["path"], start=l, end=r))
        body_movement_channel = "body_movement"
        if use_legacy_body_movement:
            body_movement_channel = "legacy_body_movement"
        super().__init__(
            data,
            extractors={
                "heartbeat": default_extractor("heartbeat", TOKEN_LEN),
                "breath": default_extractor("breath", TOKEN_LEN),
                "body_movement": default_extractor(body_movement_channel, TOKEN_LEN),
                "stage": default_extractor("stage", 1),
            },
            collators={
                "heartbeat": (True, pad_collator("heartbeat")),
                "breath": (True, pad_collator("breath")),
                "body_movement": (True, pad_collator("body_movement")),
                "stage": (False, pad_collator("stage", pad_value=PAD_STAGE)),
            },
            dataloader_config=kwargs,
        )


class EEGSleepStagingDataset(DefaultDataset):
    r_margin = 0

    def extract_eeg(self, npz: t.Mapping[str, np.ndarray], start: int, end: int):
        x: np.ndarray = npz[self.eeg_feature_channel]
        x = x.T[start * self.eeg_features_per_token : end * self.eeg_features_per_token]
        x = x.astype(np.float32)
        return torch.from_numpy(x)

    def __init__(
        self,
        index: str,
        split: str,
        max_tokens: int,
        stride_tokens: int = 0,  # 0 for truncation
        eeg_feature_channel: str = "eeg_power_distribution",
        eeg_features_per_token: int = 1,
        r_margin: float = 0,
        **kwargs: t.Any,
    ) -> None:
        self.eeg_feature_channel = eeg_feature_channel
        self.eeg_features_per_token = eeg_features_per_token
        csv = pd.read_csv(index)
        csv = csv[(csv["split"] == split) & (csv["duration"] >= TOKEN_SEC)]
        data: list[SampleIndex] = []
        for i, (_, row) in enumerate(csv.iterrows()):
            n = int((row["duration"] - r_margin) // TOKEN_SEC)
            for l, r in slicing.window(n, max_tokens, stride_tokens):
                data.append(SampleIndex(id=i, path=row["path"], start=l, end=r))
        super().__init__(
            data,
            extractors={
                "eeg": self.extract_eeg,
                "stage": default_extractor("stage", 1),
            },
            collators={
                "eeg": (True, pad_collator("eeg")),
                "stage": (False, pad_collator("stage", pad_value=PAD_STAGE)),
            },
            dataloader_config=kwargs,
        )


class EEG60SleepStagingDataset(EEGSleepStagingDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            eeg_feature_channel="eeg_power_distribution_60",
            eeg_features_per_token=60,
            r_margin=0.5,
            **kwargs,
        )
