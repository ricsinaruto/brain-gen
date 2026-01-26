import torch
from torch import nn


class NTDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics = {}

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ):
        noise, pred_noise, mask = inputs
        loss = model.loss(noise, pred_noise, mask)
        return loss.mean()
