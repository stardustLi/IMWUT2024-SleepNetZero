from math import log
import torch
import torch.nn as nn


class MSELossWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.criterion = nn.MSELoss(**kwargs)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        return self.criterion(logits[mask], target[mask])
