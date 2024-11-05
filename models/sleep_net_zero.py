import lightning as L
import torch
import torch.nn as nn
from transformers import RoFormerConfig, RoFormerModel
import typing as t
from wuji_dl.factory import get_class
from wuji_dl.ops.functional import length_mask
from wuji_dl.ops.res_block1d import ResBlock1d


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        channels: int,
        layers: list[int],
        norm_layer: t.Callable[[int], nn.Module] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.layer0 = nn.Sequential(
            nn.Conv1d(channels, self.inplanes, 7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=3)
        self.layer4 = self._make_layer(512, layers[3], stride=5)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, 1, stride=stride, bias=False),
                self._norm_layer(planes),
            )
        layers = [ResBlock1d(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResBlock1d(planes, planes, 1, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SleepNetZeroHead(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        self.mode = mode
        self.feature_extractor = ResNetFeatureExtractor(1, [2, 2, 2, 2])

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        x = kwargs[self.mode][:, None].float()
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
        x = torch.stack(feats).sum(dim=0).transpose(1, 2)
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
