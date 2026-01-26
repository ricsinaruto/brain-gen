import torch
import torch.nn as nn


class ChannelLastLayerNorm(nn.Module):
    """LayerNorm over channels for (N, C, T) tensors without permanent permutes."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        x = x.transpose(1, 2)  # (N, T, C)
        x = self.ln(x)
        return x.transpose(1, 2)  # (N, C, T)
