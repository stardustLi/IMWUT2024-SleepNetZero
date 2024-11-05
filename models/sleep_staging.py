import torch.nn as nn

from .sleep_net_zero import SleepNetZeroFacade
from .sleep_net_zero_mlp import SleepNetZeroFacade as SleepNetZeroFacadeMLP


class SleepStagingNet(nn.Module):
    def __init__(self, num_classes: int = 5, **kwargs) -> None:
        super().__init__()
        self.backbone = SleepNetZeroFacade(**kwargs)
        self.fc = nn.Linear(self.backbone.hidden_size, num_classes)

    def forward(self, **kwargs):
        x = self.backbone(**kwargs)[:, 1:, :]
        return self.fc(x)


class SleepStagingNetMLP(nn.Module):
    def __init__(self, num_classes: int = 5, **kwargs) -> None:
        super().__init__()
        self.backbone = SleepNetZeroFacadeMLP(**kwargs)
        self.fc = nn.Linear(self.backbone.hidden_size, num_classes)

    def forward(self, **kwargs):
        x = self.backbone(**kwargs)[:, 1:, :]
        return self.fc(x)
