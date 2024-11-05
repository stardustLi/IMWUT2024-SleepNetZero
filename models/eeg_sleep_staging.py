import torch
import torch.nn as nn
from transformers import RoFormerConfig, RoFormerModel
import typing as t
from wuji_dl.ops.functional import length_mask


class EEGSleepNetZeroBackbone(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        config = RoFormerConfig(vocab_size=1, **kwargs)
        self.roformer = RoFormerModel(config)
        self.cls = nn.Parameter(torch.empty(1, 1, config.embedding_size))
        nn.init.normal_(self.cls)
        self.hidden_size = config.hidden_size

    def _forward(self, eeg: torch.Tensor, length: torch.Tensor | None = None):
        x = torch.cat([self.cls.expand(eeg.size(0), -1, -1), eeg], dim=1)
        mask = None if length is None else length_mask(x.size(1), length + 1)
        return self.roformer(
            inputs_embeds=x, attention_mask=mask, output_hidden_states=True
        )

    def get_hidden_states(self, **kwargs):
        return self._forward(**kwargs).hidden_states

    def forward(self, **kwargs):
        return self._forward(**kwargs).last_hidden_state


class EEGSleepStagingNet(nn.Module):
    def __init__(
        self, num_classes: int = 5, log_transform: bool = False, **kwargs
    ) -> None:
        super().__init__()
        self.log_transform = log_transform
        self.backbone = EEGSleepNetZeroBackbone(**kwargs)
        self.fc = nn.Linear(self.backbone.hidden_size, num_classes)

    def forward(self, **kwargs):
        if self.log_transform:
            kwargs["eeg"] = (kwargs["eeg"] + 1e-9).log()
        x = self.backbone(**kwargs)[:, 1:, :]
        return self.fc(x)


class EEGSleepTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        head: dict[str, t.Any],
        backbone: dict[str, t.Any],
        num_classes: int = 5,
        log_transform: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.log_transform = log_transform
        self.head = EEGSleepNetZeroBackbone(**head)
        self.backbone = EEGSleepNetZeroBackbone(
            **backbone, embedding_size=self.head.hidden_size
        )
        self.fc = nn.Linear(self.backbone.hidden_size, num_classes)

    def forward(self, eeg: torch.Tensor, length: torch.Tensor):
        x: torch.Tensor

        n = eeg.size(0)
        if self.log_transform:
            eeg = (eeg + 1e-9).log()
        x = eeg.unflatten(1, (-1, self.patch_size)).flatten(0, 1)
        x = self.head(eeg=x)[:, 0, :]
        x = x.unflatten(0, (n, -1))
        x = self.backbone(eeg=x, length=length)[:, 1:, :]
        x = self.fc(x)
        return x


class EEGSleepStagingNet60(EEGSleepTransformer):
    def __init__(self, **kwargs) -> None:
        super().__init__(patch_size=60, **kwargs)
