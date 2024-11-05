import torch
import torch.nn as nn


class WassersteinLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xs = x.cumsum(dim=-1)
        ys = y.cumsum(dim=-1)
        return torch.mean(torch.abs(xs - ys))
