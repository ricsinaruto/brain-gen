import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from typing import Literal, Optional, Type

from .norms import ChannelLastLayerNorm


def _choose_gn_groups(requested: int, num_channels: int) -> int:
    # pick the largest divisor of num_channels not exceeding requested
    g = min(requested, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return max(1, g)


def make_norm(
    norm_type: Literal["group", "batch", "layer"],
    num_channels: int,
    *,
    gn_groups: int = 16,
    eps: float = 1e-5,
) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm1d(num_channels, eps=eps)
    elif norm_type == "layer":
        # LayerNorm expects normalized last dim -> we wrap it for (N, C, T) input
        return ChannelLastLayerNorm(num_channels, eps=eps)
    else:  # "group"
        g = _choose_gn_groups(gn_groups, num_channels)
        return nn.GroupNorm(g, num_channels, eps=eps)


def time_group_norm(
    norm: nn.GroupNorm, x: torch.Tensor, *, time_dim: int = 2
) -> torch.Tensor:
    """Apply GroupNorm without mixing across the temporal dimension by reshaping (B, C,
    T, ...) -> (B*T, C, ...), normalising, then restoring shape."""
    if x.ndim < 3:
        return norm(x)

    time_dim = time_dim % x.dim()
    perm = [0, time_dim] + [d for d in range(1, x.dim()) if d != time_dim]
    x_perm = x.permute(perm).contiguous()

    B, T = x_perm.shape[:2]
    rest = x_perm.shape[2:]
    x_flat = x_perm.reshape(B * T, *rest)

    x_norm = norm(x_flat)
    x_norm = x_norm.view(B, T, *rest)

    inv_perm = [perm.index(i) for i in range(len(perm))]
    return x_norm.permute(inv_perm).contiguous()


class SE1D(nn.Module):
    """Squeeze-and-Excitation for (B, C, T)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)  # squeeze over time
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        z = self.pool(x).squeeze(-1)  # (B, C)
        w = self.fc(z).unsqueeze(-1)  # (B, C, 1)
        return x * w  # broadcast over T


class ECA1D(nn.Module):
    """Efficient Channel Attention for (B, C, T).

    No MLP.
    """

    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        k = k if k % 2 == 1 else k + 1  # ensure odd
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, C, 1)
        # conv over channel dimension: reshape to (B, 1, C)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)  # (B, C, 1)
        y = y.transpose(1, 2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = torch.sigmoid(y).transpose(1, 2)  # (B, C, 1)
        return x * y


class DownStage3D(nn.Module):
    """Spatial downsample by 2x, no temporal mixing."""

    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv = nn.Conv3d(
            C_in, C_out, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.norm = nn.GroupNorm(1, C_out)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = time_group_norm(self.norm, x)
        return self.act(x)


class UpStage3D(nn.Module):
    """Spatial upsample by 2x via transposed conv, no temporal mixing."""

    def __init__(self, C_in, C_out):
        super().__init__()
        # kernel 4, stride 2, pad 1 gives clean 2x upsample in H/W with T unchanged
        self.deconv = nn.ConvTranspose3d(
            C_in, C_out, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.norm = nn.GroupNorm(1, C_out)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.deconv(x)
        x = time_group_norm(self.norm, x)
        return self.act(x)


class Conv3dBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        kT: int,
        kH: int,
        kW: int,
        image_size: int,
        dropout: float,
        row_idx: torch.Tensor | None = None,
        col_idx: torch.Tensor | None = None,
    ):
        super().__init__()
        self.H = image_size
        self.W = image_size
        self.kT = kT
        self.kH = kH
        self.kW = kW

        # if row and col idx is none, load from tmp file
        tensors = np.load("tmp/img_inds.npy", allow_pickle=True).item()
        row_idx = torch.from_numpy(tensors["row_idx"])
        col_idx = torch.from_numpy(tensors["col_idx"])

        assert row_idx.shape == col_idx.shape
        self.register_buffer("row_idx", row_idx.long())
        self.register_buffer("col_idx", col_idx.long())

        self.conv3d = nn.Conv3d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=(kT, kH, kW),
            stride=1,
            padding=0,  # we will pad manually to enforce causal time
            bias=True,
        )
        self.pw = nn.Conv3d(d_model, d_model, kernel_size=1, bias=True)
        self.gn = nn.GroupNorm(d_model // 8, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def _to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """X: (B, C, T, M) -> y: (B, M, T, H, W) Places each sensor m at (row_idx[m],
        col_idx[m]) across the H×W grid."""
        B, C, T, M = x.shape

        x = rearrange(x, "b c t m -> b m t c")

        y = x.new_zeros((B, M, T, self.H, self.W))
        # Advanced indexing will produce a (B,C,T,M) view on the RHS
        y[..., self.row_idx, self.col_idx] = x
        return y

    def _from_grid(self, y: torch.Tensor) -> torch.Tensor:
        """Y: (B, M, T, H, W) -> x: (B, M, T, C) Gathers sensor positions back from the
        grid."""

        y = y[..., self.row_idx, self.col_idx]  # (B,M,T,C)
        y = rearrange(y, "b m t c -> b c t m")
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal 3D conv over (T,H,W) with channels=M.

        Causality enforced by left-padding T by (kT-1) and symmetric padding on H/W.
        """
        x = self._to_grid(x)

        pad_t = self.kT - 1
        pad_h = self.kH // 2
        pad_w = self.kW // 2
        # F.pad order for 5D: (W_left, W_right, H_top, H_bottom, D_front, D_back)
        y = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_t, 0))
        y = self.conv3d(y)
        y = time_group_norm(self.gn, y)
        y = self.act(y)
        y = self.pw(y)
        y = self.drop(y)
        y = y + x

        return self._from_grid(y)


class ConvBlock1D(nn.Module):
    """Residual 1D conv block with Conv -> Norm -> Act -> Drop -> (ChAttn) -> +
    Residual.

    Residual path matches channels and stride. padding='same'.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        norm: Literal["group", "batch", "layer"] = "group",
        gn_groups: int = 16,
        dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        eps: float = 1e-5,
        ch_attn: Optional[Literal["se", "eca"]] = "se",
        se_reduction: int = 8,
        eca_k: int = 3,
        padding: bool = True,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding else 0,
            bias=False,
        )
        self.norm = make_norm(norm, out_ch, gn_groups=gn_groups, eps=eps)
        self.act = activation()
        self.drop = nn.Dropout1d(p=dropout)

        # Optional channel attention on the main path output
        if ch_attn is None:
            self.ch_attn = None
        elif ch_attn == "se":
            self.ch_attn = SE1D(out_ch, reduction=se_reduction)
        elif ch_attn == "eca":
            self.ch_attn = ECA1D(out_ch, k=eca_k)
        else:
            raise ValueError(f"Unknown ch_attn: {ch_attn}")

        # Residual path matches channels and stride
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Conv1d(
                in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        if self.ch_attn is not None:
            x = self.ch_attn(x)
        return x + res
