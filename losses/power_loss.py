from math import log
import torch
import torch.nn as nn


class PowerLoss(nn.Module):
    def __init__(self, power=3):
        super().__init__()
        self.power = power

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        loss = torch.abs(logits[mask] - target[mask]) ** self.power
        return torch.mean(loss)
