import torch
import torch.nn as nn
import torch.nn.functional as F


def label_smoothing(labels: torch.Tensor, num_classes: int, smoothing: float = 0.1):
    assert 0 <= smoothing < 1
    with torch.no_grad():
        true_dist = torch.empty(
            size=(labels.size(0), num_classes), device=labels.device
        )
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    return true_dist


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: list[float] | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.weight: torch.Tensor | None
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight))
        else:
            self.weight = None
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]

        targets_onehot = F.one_hot(targets, num_classes=inputs.size(-1))
        if self.label_smoothing > 0:
            targets_onehot = label_smoothing(
                targets, inputs.size(-1), self.label_smoothing
            )

        pos_prob = F.softmax(inputs, dim=-1) * targets_onehot
        pos_prob[pos_prob == 0] = 1  # 不考虑负样本的预测结果
        log_prob = F.log_softmax(inputs, dim=-1)
        # 核心是将正样本的log_prob数值变小，相当于放大了预测出错的程度，使得loss变大
        factor = torch.exp((1 - pos_prob) * self.gamma)
        loss = F.nll_loss(
            factor * log_prob, targets, weight=self.weight, reduction=self.reduction
        )

        return loss
