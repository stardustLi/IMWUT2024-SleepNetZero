import torch
import torch.nn as nn


def _masked_mean(x, mask=None, *, keepdim=False):
    if mask is None:
        return x.mean(dim=1, keepdim=keepdim)
    n = mask.sum(dim=1, keepdim=keepdim)
    return (x * mask).sum(dim=1, keepdim=keepdim) / torch.max(torch.ones_like(n), n)


def _masked_zscore(x, mask=None):
    x = x - _masked_mean(x, mask, keepdim=True)
    var = _masked_mean(x**2, mask, keepdim=True)
    return x / (var + 1e-8).sqrt()


class PearsonCorrelation(nn.Module):
    def __init__(self, reduction="mean"):
        super(PearsonCorrelation, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        mask = ~torch.isnan(y_true)
        y_true[torch.isnan(y_true)] = 0
        # Compute Pearson correlation coefficient
        z_pred = _masked_zscore(y_pred, mask)
        z_true = _masked_zscore(y_true, mask)
        # Pearson correlation loss is defined as 1 - Pearson correlation coefficient
        pearson_loss = 1.0 - _masked_mean(z_pred * z_true, mask)
        # Apply reduction
        if self.reduction == "sum":
            return pearson_loss.sum()
        elif self.reduction == "mean":
            return pearson_loss.mean()
        else:  # 'none'
            return pearson_loss
