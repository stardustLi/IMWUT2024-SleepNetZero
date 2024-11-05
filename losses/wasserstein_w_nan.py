import torch
import torch.nn as nn

class WassersteinLoss(nn.Module):
    def __init__(self, seq_length=10):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 创建掩码来忽略nan值
        mask = ~torch.isnan(target)
        logits_masked = logits[mask]
        target_masked = target[mask]

        # 确保长度足以被seq_length整除，否则填充到合适的长度
        remainder = logits_masked.size(-1) % self.seq_length
        if remainder != 0:
            padding_size = self.seq_length - remainder
            logits_masked = torch.nn.functional.pad(logits_masked, (0, padding_size), mode='constant', value=0)
            target_masked = torch.nn.functional.pad(target_masked, (0, padding_size), mode='constant', value=0)

        # 按seq_length拆分张量
        xs_split = logits_masked.unfold(dimension=-1, size=self.seq_length, step=self.seq_length)
        ys_split = target_masked.unfold(dimension=-1, size=self.seq_length, step=self.seq_length)

        # 计算每个子序列的累积和和损失
        xs_cum = xs_split.cumsum(dim=-1)
        ys_cum = ys_split.cumsum(dim=-1)
        loss_per_seq = torch.mean(torch.abs(xs_cum - ys_cum), dim=-1)

        # 对所有子序列损失取均值
        total_loss = torch.mean(loss_per_seq)
        return total_loss

# 示例用法
# logits = torch.tensor([1.0, 2.0, 3.0, 4, 5.0, 6.0, 7.0, 8.0])
# target = torch.tensor([1.1, 2.1, 3.1, 4.1, float('nan'), 6.1, 7.1, 8.1])
# seq_length = 2

# model = WassersteinLoss(seq_length=seq_length)
# loss = model(logits, target)
# print("Calculated loss:", loss.item())
