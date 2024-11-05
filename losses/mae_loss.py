from math import log
import torch
import torch.nn as nn


class MAELossWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.criterion = nn.L1Loss(**kwargs)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        return self.criterion(logits[mask], target[mask])
