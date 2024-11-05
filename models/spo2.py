import torch.nn as nn
import torch

from .sleep_net_zero import SleepNetZeroBackbone, SleepNetZeroPretrained, ResNetFeatureExtractor
from transformers import RoFormerConfig, RoFormerModel

class Spo2Net(nn.Module):
    def __init__(self, spo2_resolution_per_token: int = 30, **kwargs) -> None:
        super().__init__()
        self.backbone = SleepNetZeroBackbone(**kwargs)
        self.fc = nn.Linear(kwargs["hidden_size"], spo2_resolution_per_token)
        # nn.init.constant_(self.fc.bias, 95.0)

    def forward(self, **kwargs):
        print(kwargs)
        x = self.backbone(**kwargs).last_hidden_state[:, 1:, :]
        return self.fc(x).reshape(x.shape[0], -1)

class Spo2NetV2(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.heartbeat_feature_extractor = ResNetFeatureExtractor(1, [2, 2, 2, 2])
        self.breath_feature_extractor = ResNetFeatureExtractor(1, [2, 2, 2, 2])
        self.body_movement_feature_extractor = ResNetFeatureExtractor(1, [2, 2, 2, 2])
        self.roformer = RoFormerModel(RoFormerConfig(embedding_size=512, **kwargs))
        self.fc = nn.Linear(kwargs["hidden_size"], 30)
        # nn.init.constant_(self.fc.bias, 95.0)

    def forward(
        self,
        length: torch.Tensor,
        heartbeat: torch.Tensor,
        breath: torch.Tensor,
        body_movement: torch.Tensor,
        **kwargs,
    ):
        x: torch.Tensor = (
            self.heartbeat_feature_extractor(heartbeat[:, None])
            + self.breath_feature_extractor(breath[:, None])
            + self.body_movement_feature_extractor(
                body_movement[:, None].to(torch.float32)
            )
        ).transpose(1, 2)
        mask = torch.arange(x.size(1), device=x.device) < length[:, None]
        x = self.roformer(inputs_embeds=x, attention_mask=mask).last_hidden_state
        return self.fc(x).reshape(x.shape[0], -1)
