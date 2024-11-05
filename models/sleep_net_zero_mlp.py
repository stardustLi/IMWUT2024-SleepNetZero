import lightning as L
import torch
import torch.nn as nn
from transformers import RoFormerConfig, RoFormerModel
from wuji_dl.factory import get_class
from wuji_dl.ops.functional import length_mask


class MLPFeatureExtractor(nn.Module):
    def __init__(self, *channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(channels) - 1):
            if i > 0:
                self.layers.append(nn.GELU())
            self.layers.append(nn.Linear(channels[i], channels[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SleepNetZeroHead(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        self.mode = mode
        self.feature_extractor = MLPFeatureExtractor(120, 512, 512)

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        x = kwargs[self.mode].unflatten(1, (-1, 120)).float()
        return self.feature_extractor(x)


class SleepNetZeroBackbone(nn.Module):
    def __init__(
        self, heads: list[str] = ["heartbeat", "breath", "body_movement"], **kwargs
    ) -> None:
        super().__init__()
        self.feature_extractors = nn.ModuleList(
            SleepNetZeroHead(mode) for mode in heads
        )
        config = RoFormerConfig(vocab_size=1, embedding_size=512, **kwargs)
        self.roformer = RoFormerModel(config)
        self.cls = nn.Parameter(torch.empty(1, 1, 512))
        nn.init.normal_(self.cls)
        self.hidden_size = config.hidden_size

    def _forward(self, length: torch.Tensor, **kwargs):
        feats = [head(**kwargs) for head in self.feature_extractors]
        x = torch.stack(feats).sum(dim=0)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        mask = length_mask(x.size(1), length + 1)
        return self.roformer(
            inputs_embeds=x, attention_mask=mask, output_hidden_states=True
        )

    def get_hidden_states(self, **kwargs):
        return self._forward(**kwargs).hidden_states

    def forward(self, **kwargs):
        return self._forward(**kwargs).last_hidden_state


class SleepNetZeroPretrained(nn.Module):
    def __init__(self, pretrain_task: str, pretrained: str):
        super().__init__()
        task_cls = get_class(L.LightningModule, pretrain_task)
        ckpt = task_cls.load_from_checkpoint(pretrained, map_location="cpu")
        self.pretrained: SleepNetZeroBackbone = ckpt.model.backbone
        assert isinstance(self.pretrained, SleepNetZeroBackbone)
        config = self.pretrained.roformer.config
        weights = [0.0] * config.num_hidden_layers + [1.0]
        self.weights = nn.Parameter(torch.tensor(weights))
        self.hidden_size = config.hidden_size

    def forward(self, **kwargs):
        feats = self.pretrained.get_hidden_states(**kwargs)
        return torch.sum(torch.stack(feats, dim=-1) * self.weights, dim=-1)


class SleepNetZeroFacade:
    def __new__(cls, **kwargs):
        if "pretrained" in kwargs:
            return SleepNetZeroPretrained(**kwargs)
        return SleepNetZeroBackbone(**kwargs)
