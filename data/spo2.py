from dataclasses import dataclass
from termios import TOSTOP
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import typing as t
from wuji_dl import slicing
from wuji_dl.data import BaseDataset
import wuji_dl.ops.functional as wf
from scipy.ndimage import maximum_filter


TOKEN_SEC = 30
TOKEN_LEN = 120


def calc_dynamic_spo2_burden_old(arr, window=100, default_value=100.0) -> np.ndarray:
    # 初始化满长度的结果数组，所有值设置为100（因为窗口不足的最大值为100）
    max_values = np.full(arr.shape, default_value)
    arr_replace_nan = np.nan_to_num(arr, nan=default_value)
    # 只有当数组长度大于等于窗口长度时，才进行滑动窗口的最大值计算
    if len(arr_replace_nan) >= window:
        # 创建滑动窗口视图
        shape = arr_replace_nan.shape[:-1] + (
            arr_replace_nan.shape[-1] - window,
            window,
        )
        strides = arr_replace_nan.strides + (arr_replace_nan.strides[-1],)
        windows = np.lib.stride_tricks.as_strided(
            arr_replace_nan, shape=shape, strides=strides
        )
        # 计算每个窗口的最大值
        max_values[window:] = np.max(windows, axis=1)
    # 计算最大值和原数组对应值的差
    max_values[:window] = np.max(arr_replace_nan[:window])
    result = arr - max_values
    return result.astype(np.float32)


def calc_dynamic_spo2_burden(arr, window=100) -> np.ndarray:
    # 初始化满长度的结果数组，所有值设置为100（因为窗口不足的最大值为100）
    max_values = maximum_filter(arr, size=window+1, mode='nearest')
    arr_replace_nan = np.where(np.isnan(arr), max_values, arr)
    # 只有当数组长度大于等于窗口长度时，才进行滑动窗口的最大值计算
    if len(arr_replace_nan) >= window:
        # 创建滑动窗口视图
        shape = arr_replace_nan.shape[:-1] + (
            arr_replace_nan.shape[-1] - window,
            window,
        )
        strides = arr_replace_nan.strides + (arr_replace_nan.strides[-1],)
        windows = np.lib.stride_tricks.as_strided(
            arr_replace_nan, shape=shape, strides=strides
        )
        # 计算每个窗口的最大值
        max_values[window:] = np.max(windows, axis=1)
    # 计算最大值和原数组对应值的差
    result = arr - max_values
    return result.astype(np.float32)


def calc_spo2_burden_under_fixed_window(arr, window=50):
    filtered = maximum_filter(arr, size=2*window+1, mode='reflect')
    result = arr - filtered
    return result


@dataclass
class SampleIndex:
    path: str
    start: int
    end: int


class Spo2Sample(t.TypedDict):
    id: int
    length: int
    heartbeat: torch.Tensor
    breath: torch.Tensor
    body_movement: torch.Tensor
    spo2: torch.Tensor
    spo2_burden: torch.Tensor


class Spo2NetInput(t.TypedDict):
    length: torch.Tensor
    heartbeat: torch.Tensor
    breath: torch.Tensor
    body_movement: torch.Tensor


class Spo2Batch(t.TypedDict):
    id: torch.Tensor
    net_input: Spo2NetInput
    spo2: torch.Tensor
    spo2_burden: torch.Tensor


class Spo2Dataset(BaseDataset):
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

    def __getitem__(self, idx: int) -> Spo2Sample:
        src = self.data[idx]
        with np.load(src.path) as npz:
            in_slice = slice(src.start * TOKEN_LEN, src.end * TOKEN_LEN)
            out_slice = slice(src.start * TOKEN_SEC, src.end * TOKEN_SEC)
            heartbeat = npz["heartbeat"][in_slice]
            breath = npz["breath"][in_slice]
            body_movement = npz["body_movement"][in_slice]
            spo2 = npz["spo2"][out_slice]
            spo2_burden = calc_dynamic_spo2_burden(spo2)
            # spo2_burden = calc_spo2_burden_under_fixed_window(spo2)
            spo2 = spo2 - np.nanmean(spo2)
        return {
            "id": idx,
            "length": src.end - src.start,
            "heartbeat": torch.from_numpy(heartbeat),
            "breath": torch.from_numpy(breath),
            "body_movement": torch.from_numpy(body_movement),
            "spo2": torch.from_numpy(spo2),
            "spo2_burden": torch.from_numpy(spo2_burden),
        }

    @staticmethod
    def _collate_fn(samples: list[Spo2Sample]) -> Spo2Batch:
        net_input: Spo2NetInput = {
            "length": torch.tensor([s["length"] for s in samples]),
            "heartbeat": wf.pad_batch([s["heartbeat"] for s in samples]),
            "breath": wf.pad_batch([s["breath"] for s in samples]),
            "body_movement": wf.pad_batch([s["body_movement"] for s in samples]),
        }
        return {
            "id": torch.tensor([s["id"] for s in samples]),
            "net_input": net_input,
            "spo2": wf.pad_batch([s["spo2"] for s in samples], pad_value=torch.nan),
            "spo2_burden": wf.pad_batch([s["spo2_burden"] for s in samples], pad_value=torch.nan),
        }

    def dataloader(self) -> DataLoader:
        return DataLoader(self, **self.dataloader_config, collate_fn=self._collate_fn)
