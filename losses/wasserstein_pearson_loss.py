from .wasserstein_w_nan import WassersteinLoss
from .pearson_loss import PearsonCorrelation
import torch.nn as nn
import torch


class WassersteinPearsonLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.pearson_weight = kwargs.pop("pearson_weight", 0.1)
        self.wasserstein_weight = 1 - self.pearson_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pearson_loss = PearsonCorrelation()(logits, target)
        wasserstein_loss = WassersteinLoss()(logits, target)
        return self.pearson_weight * pearson_loss + self.wasserstein_weight * wasserstein_loss