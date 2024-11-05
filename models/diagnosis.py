import torch
import torch.nn as nn
from wuji_dl.slicing import window

from .sleep_net_zero import SleepNetZeroFacade


class DiagnosisNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.backbone = SleepNetZeroFacade(**kwargs)
        self.fc = nn.Linear(self.backbone.hidden_size, 1)

    def forward(self, **kwargs):
        x: torch.Tensor = self.backbone(**kwargs)
        return self.fc(x[:, 0])


class DiagnosisNetMean(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.backbone = SleepNetZeroFacade(**kwargs)
        self.fc = nn.Linear(self.backbone.hidden_size, 1)

    def forward(self, **kwargs):
        x = self.backbone(**kwargs)
        return self.fc(x[:, 1:]).mean(dim=1)
