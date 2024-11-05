import torch
import torch.nn as nn

from .sleep_net_zero import SleepNetZeroFacade


class EEGPowerDistributionNet(nn.Module):
    def __init__(self, bins: int, **kwargs) -> None:
        super().__init__()
        self.backbone = SleepNetZeroFacade(**kwargs)
        self.proj = nn.Linear(kwargs["hidden_size"], bins)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, **kwargs):
        x: torch.Tensor = self.backbone(**kwargs)
        x = self.proj(x[:, 1:, :])
        return self.softmax(x)
