import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput


def _as_bct(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    Accepts:
      [C, T] or [C, T, 1] or [B, C, T] or [B, C, T, 1]
    Returns:
      x_bct: [B, C, T]
      was_unbatched: bool (if input had no batch dim)
    """
    if x.dim() == 2:  # [C, T]
        return x.unsqueeze(0), True
    if x.dim() == 3:
        if x.shape[-1] == 1:  # [C, T, 1]
            return x[..., 0].unsqueeze(0), True
        # [B, C, T]
        return x, False
    if x.dim() == 4:
        if x.shape[-1] != 1:
            raise ValueError(f"Expected last dim=1 for [B,C,T,1], got {x.shape}")
        # [B, C, T, 1]
        return x[..., 0], False
    raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


def _restore_shape(
    x_bct1: torch.Tensor, was_unbatched: bool, want_last_dim_1: bool
) -> torch.Tensor:
    """
    x_bct1: [B, C, T] or [B, C, T, 1] (we'll accept either)
    returns either [C, T] / [C, T, 1] if was_unbatched else [B, C, T] / [B, C, T, 1]
    """
    if x_bct1.dim() == 3:
        x = x_bct1
        if want_last_dim_1:
            x = x.unsqueeze(-1)  # [B,C,T,1]
    elif x_bct1.dim() == 4:
        x = x_bct1 if want_last_dim_1 else x_bct1[..., 0]
    else:
        raise ValueError

    if was_unbatched:
        return x[0]
    return x


def _causal_pad_1d(
    x: torch.Tensor, kernel_size: int, dilation: int = 1
) -> torch.Tensor:
    pad = dilation * (kernel_size - 1)
    if pad <= 0:
        return x
    return F.pad(x, (pad, 0))


def _binomial_kernel(kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return torch.tensor([1.0], dtype=torch.float32)
    coeffs = [math.comb(kernel_size - 1, i) for i in range(kernel_size)]
    kernel = torch.tensor(coeffs, dtype=torch.float32)
    return kernel / kernel.sum()


def _normalize_factors(
    total: int, factors: Optional[Tuple[int, ...]], name: str
) -> Tuple[int, ...]:
    if total < 1:
        raise ValueError(f"{name} must be >= 1, got {total}")
    if factors is None:
        if total == 1:
            return ()
        remaining = total
        out = []
        while remaining % 2 == 0 and remaining > 1:
            out.append(2)
            remaining //= 2
        if remaining > 1:
            out.append(remaining)
        return tuple(out)

    factors = tuple(int(f) for f in factors)
    if any(f < 1 for f in factors):
        raise ValueError(f"All {name} factors must be >= 1, got {factors}")
    if math.prod(factors) != total:
        raise ValueError(
            f"{name} factors product {math.prod(factors)} != total {total}"
        )
    return factors


def _align_factors(
    t_factors: Tuple[int, ...], c_factors: Tuple[int, ...]
) -> Tuple[Tuple[int, int], ...]:
    n = max(len(t_factors), len(c_factors))
    t = list(t_factors) + [1] * (n - len(t_factors))
    c = list(c_factors) + [1] * (n - len(c_factors))
    return tuple(zip(t, c))


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _causal_pad_1d(x, self.kernel_size, self.dilation)
        return self.conv(x)


class CausalLowpass1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = int(kernel_size)
        kernel = _binomial_kernel(self.kernel_size)
        weight = kernel.view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x
        weight = self.weight.to(dtype=x.dtype)
        x = _causal_pad_1d(x, self.kernel_size, dilation=1)
        return F.conv1d(x, weight, bias=None, stride=1, groups=self.channels)


class ChannelMix(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return _apply_channel_mix_weight(h, self.weight)


def _apply_channel_mix_weight(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    h_perm = h.permute(0, 2, 3, 1).contiguous()
    h_mix = F.linear(h_perm, weight)
    return h_mix.permute(0, 3, 1, 2).contiguous()


def _channel_mlp_hidden_dim(c_in: int, c_out: int, ratio: float) -> int:
    if ratio <= 0:
        raise ValueError(f"channel_mlp_ratio must be > 0, got {ratio}")
    base = min(c_in, c_out)
    return max(1, int(round(base * ratio)))


class ChannelMixMLP(nn.Module):
    """Two-layer channel MLP with a bottleneck for more expressive mixing."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        bottleneck_ratio: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_mid = _channel_mlp_hidden_dim(c_in, c_out, bottleneck_ratio)
        self.w1 = nn.Parameter(torch.empty(self.c_mid, c_in))
        self.w2 = nn.Parameter(torch.empty(c_out, self.c_mid))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = _apply_channel_mix_weight(h, self.w1)
        h = F.gelu(h)
        h = self.drop(h)
        return _apply_channel_mix_weight(h, self.w2)


class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize per time-step over channels only.
        x = x.transpose(1, 2)
        x = self.ln(x)
        return x.transpose(1, 2)


class TemporalDownsampleBlock(nn.Module):
    """
    Depthwise temporal conv (per feature channel)
     + grouped 1x1 mixing within each ROI group.
    Input:  [B, C*F, T]
    Output: [B, C*F, ceil(T/stride)] (causal left padding)
    """

    def __init__(
        self,
        c: int,
        f: int,
        stride: int = 2,
        k: int = 5,
        dropout: float = 0.0,
        aa_kernel_size: int = 0,
    ):
        super().__init__()
        assert stride >= 1
        self.c = c
        self.f = f
        ch = c * f

        self.aa = (
            CausalLowpass1d(ch, aa_kernel_size)
            if stride > 1 and aa_kernel_size > 1
            else nn.Identity()
        )

        # Depthwise conv over time for each feature channel.
        self.dw = CausalConv1d(
            ch, ch, kernel_size=k, stride=stride, groups=ch, bias=False
        )

        # Channel-only normalization per time-step.
        self.gn1 = ChannelLayerNorm(num_channels=ch, eps=1e-5)

        # Mix features *within* each ROI (grouped 1x1 conv).
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, groups=c, bias=False)
        self.gn2 = ChannelLayerNorm(num_channels=ch, eps=1e-5)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aa(x)
        x = self.dw(x)
        x = self.gn1(x)
        x = F.gelu(x)
        x = self.pw(x)
        x = self.gn2(x)
        x = F.gelu(x)
        x = self.drop(x)
        return x


class TemporalUpsampleBlock(nn.Module):
    """
    Nearest-neighbor upsample + causal depthwise conv
     + grouped 1x1 mixing within each ROI group.
    Input:  [B, C*F, T]
    Output: [B, C*F, T*stride]
    """

    def __init__(
        self,
        c: int,
        f: int,
        stride: int = 2,
        k: int = 5,
        dropout: float = 0.0,
        aa_kernel_size: int = 0,
    ):
        super().__init__()
        assert stride >= 1
        self.c = c
        self.f = f
        self.stride = stride
        ch = c * f

        self.aa = (
            CausalLowpass1d(ch, aa_kernel_size)
            if stride > 1 and aa_kernel_size > 1
            else nn.Identity()
        )
        self.dwt = CausalConv1d(ch, ch, kernel_size=k, stride=1, groups=ch, bias=False)
        self.gn1 = ChannelLayerNorm(num_channels=ch, eps=1e-5)

        self.pw = nn.Conv1d(ch, ch, kernel_size=1, groups=c, bias=False)
        self.gn2 = ChannelLayerNorm(num_channels=ch, eps=1e-5)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = x.repeat_interleave(self.stride, dim=-1)
            x = self.aa(x)
        x = self.dwt(x)
        x = self.gn1(x)
        x = F.gelu(x)
        x = self.pw(x)
        x = self.gn2(x)
        x = F.gelu(x)
        x = self.drop(x)
        return x


class MEGFactorizedEncoder(nn.Module):
    """
    Encoder for MEG ROI timeseries (C=68 typical).
    - Temporal feature lifting per ROI (depthwise/grouped)
    - Interleaved temporal downsample + channel reduction stages
    - Per-token feature projection: F -> D

    Output latent: z ∈ [B, C_red, T_red, D]
    """

    def __init__(
        self,
        c_in: int = 68,
        c_down: int = 4,
        t_down: int = 4,
        t_down_factors: Optional[Tuple[int, ...]] = None,
        c_down_factors: Optional[Tuple[int, ...]] = None,
        c_stages: Optional[Tuple[int, ...]] = None,
        channel_mlp_ratio: Optional[float] = None,
        f_hidden: int = 32,
        d_latent: int = 128,
        k_stem: int = 7,
        k_down: int = 5,
        dropout: float = 0.0,
        aa_kernel_size: int = 3,
    ):
        super().__init__()
        self.c_in = c_in
        self.t_down = t_down
        self.f_hidden = f_hidden
        self.d_latent = d_latent
        self.channel_mlp_ratio = channel_mlp_ratio
        self.t_factors = _normalize_factors(t_down, t_down_factors, "t_down")
        self.c_stages = None if c_stages is None else tuple(int(c) for c in c_stages)

        if self.c_stages is not None:
            if any(c < 1 for c in self.c_stages):
                raise ValueError(f"c_stages must be >= 1, got {self.c_stages}")
            if len(self.t_factors) != len(self.c_stages):
                raise ValueError(
                    "t_down_factors length must match c_stages length when "
                    "c_stages is provided."
                )
            self.stage_t_factors = self.t_factors
            self.stage_channels = [c_in] + list(self.c_stages)
        else:
            if c_in % c_down != 0:
                raise ValueError(
                    f"Channels {c_in} must be divisible by c_down={c_down}"
                )
            self.c_down = c_down
            self.c_factors = _normalize_factors(c_down, c_down_factors, "c_down")
            self.stage_factors = _align_factors(self.t_factors, self.c_factors)
            self.stage_t_factors = tuple(t for t, _ in self.stage_factors)

            self.stage_channels = [c_in]
            for _, c_factor in self.stage_factors:
                c_prev = self.stage_channels[-1]
                if c_prev % c_factor != 0:
                    raise ValueError(
                        f"Stage channel reduction factor {c_factor} does not divide {c_prev}"
                    )
                self.stage_channels.append(c_prev // c_factor)

        self.c_red = self.stage_channels[-1]

        # Lift 1 feature per ROI -> F features per ROI (grouped by ROI).
        # Input [B, C, T] treated as Conv1d with channels=C.
        # groups=C =>
        # each ROI independently maps 1 -> F (implemented as out_channels=C*F).
        self.stem = CausalConv1d(
            in_channels=c_in,
            out_channels=c_in * f_hidden,
            kernel_size=k_stem,
            stride=1,
            groups=c_in,
            bias=False,
        )
        self.stem_gn = ChannelLayerNorm(num_channels=c_in * f_hidden, eps=1e-5)

        self.down_blocks = nn.ModuleList()
        self.channel_reducers = nn.ModuleList()
        for stage_idx, t_factor in enumerate(self.stage_t_factors):
            c_stage = self.stage_channels[stage_idx]
            self.down_blocks.append(
                TemporalDownsampleBlock(
                    c=c_stage,
                    f=f_hidden,
                    stride=t_factor,
                    k=k_down,
                    dropout=dropout,
                    aa_kernel_size=aa_kernel_size,
                )
            )
            c_next = self.stage_channels[stage_idx + 1]
            if c_next == c_stage:
                self.channel_reducers.append(nn.Identity())
            elif self.channel_mlp_ratio is None:
                self.channel_reducers.append(ChannelMix(c_stage, c_next))
            else:
                self.channel_reducers.append(
                    ChannelMixMLP(
                        c_stage,
                        c_next,
                        bottleneck_ratio=self.channel_mlp_ratio,
                        dropout=dropout,
                    )
                )

        # After reduction:
        # [B, C_red, F, T_red] -> project F -> D per reduced channel (grouped 1x1).
        self.proj_fd = nn.Conv1d(
            in_channels=self.c_red * f_hidden,
            out_channels=self.c_red * d_latent,
            kernel_size=1,
            groups=self.c_red,
            bias=False,
        )
        self.proj_gn = ChannelLayerNorm(num_channels=self.c_red * d_latent, eps=1e-5)

    def get_reduce_weights(self) -> Tuple[Optional[torch.Tensor], ...]:
        weights: list[Optional[torch.Tensor]] = []
        for reducer in self.channel_reducers:
            if isinstance(reducer, ChannelMix):
                weights.append(reducer.weight)
            else:
                weights.append(None)
        return tuple(weights)

    @torch.no_grad()
    def expected_latent_shape(self, T: int) -> Tuple[int, int, int]:
        T_red = T
        for t_factor in self.stage_t_factors:
            if t_factor > 1:
                T_red = (T_red + t_factor - 1) // t_factor
        return (self.c_red, T_red, self.d_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bct, was_unbatched = _as_bct(x)  # [B,C,T]
        B, C, T = x_bct.shape
        assert C == self.c_in, f"Expected C={self.c_in}, got {C}"

        # Stem: [B, C, T] -> [B, C*F, T]
        h = self.stem(x_bct)
        h = self.stem_gn(h)
        h = F.gelu(h)

        # Interleaved temporal downsample + channel reduction
        h_bct = h
        T_red = h_bct.shape[-1]
        for stage_idx, t_factor in enumerate(self.stage_t_factors):
            h_bct = self.down_blocks[stage_idx](h_bct)
            T_red = h_bct.shape[-1]

            c_stage = self.stage_channels[stage_idx]
            h_stage = h_bct.view(B, c_stage, self.f_hidden, T_red)
            reducer = self.channel_reducers[stage_idx]
            if not isinstance(reducer, nn.Identity):
                h_stage = reducer(h_stage)
            c_next = self.stage_channels[stage_idx + 1]
            h_bct = h_stage.reshape(B, c_next * self.f_hidden, T_red)

        # Project F -> D per reduced channel:
        # [B, C_red, F, T_red] -> [B, C_red*F, T_red] -> [B, C_red*D, T_red]
        h_red_flat = h_bct.reshape(B, self.c_red * self.f_hidden, T_red)
        z = self.proj_fd(h_red_flat)
        z = self.proj_gn(z)
        z = F.gelu(z)

        # Return [B, C_red, T_red, D]
        z = z.view(B, self.c_red, self.d_latent, T_red).permute(0, 1, 3, 2).contiguous()
        return z


class MEGFactorizedDecoder(nn.Module):
    """
    Decoder inverse of MEGFactorizedEncoder:
    - Per-token projection D -> F
    - Interleaved channel expansion + temporal upsample stages
    - Per-ROI projection F -> 1

    Input:  z ∈ [B, C_red, T_red, D]
    Output: x̂ ∈ [B, C, T, 1] (or unbatched if input was unbatched)
    """

    def __init__(
        self,
        c_out: int = 68,
        c_down: int = 4,
        t_up: int = 4,
        t_up_factors: Optional[Tuple[int, ...]] = None,
        c_up_factors: Optional[Tuple[int, ...]] = None,
        c_up_stages: Optional[Tuple[int, ...]] = None,
        channel_mlp_ratio: Optional[float] = None,
        f_hidden: int = 32,
        d_latent: int = 128,
        k_up: int = 4,
        k_refine: int = 7,
        dropout: float = 0.0,
        aa_kernel_size: int = 3,
        tie_channel_mats: bool = False,
    ):
        super().__init__()
        self.c_out = c_out
        self.t_up = t_up
        self.f_hidden = f_hidden
        self.d_latent = d_latent
        self.tie_channel_mats = tie_channel_mats
        self.channel_mlp_ratio = channel_mlp_ratio
        self.t_factors = _normalize_factors(t_up, t_up_factors, "t_up")
        self.c_up_stages = (
            None if c_up_stages is None else tuple(int(c) for c in c_up_stages)
        )

        if self.channel_mlp_ratio is not None and self.tie_channel_mats:
            raise ValueError(
                "tie_channel_mats requires linear ChannelMix; set channel_mlp_ratio=None."
            )

        if self.c_up_stages is not None:
            if any(c < 1 for c in self.c_up_stages):
                raise ValueError(f"c_up_stages must be >= 1, got {self.c_up_stages}")
            if len(self.t_factors) != len(self.c_up_stages):
                raise ValueError(
                    "t_up_factors length must match c_up_stages length when "
                    "c_up_stages is provided."
                )
            if c_out % c_down != 0:
                raise ValueError(
                    f"Channels {c_out} must be divisible by c_down={c_down}"
                )
            self.c_down = c_down
            self.c_red = c_out // c_down
            self.stage_t_factors = self.t_factors
            self.stage_channels = [self.c_red] + list(self.c_up_stages)
            if self.stage_channels[-1] != c_out:
                raise ValueError("c_up_stages must end with c_out when provided.")
        else:
            if c_out % c_down != 0:
                raise ValueError(
                    f"Channels {c_out} must be divisible by c_down={c_down}"
                )
            self.c_down = c_down
            self.c_red = c_out // c_down
            self.c_factors = _normalize_factors(c_down, c_up_factors, "c_up")
            self.stage_factors = _align_factors(self.t_factors, self.c_factors)
            self.stage_t_factors = tuple(t for t, _ in self.stage_factors)

            self.stage_channels = [self.c_red]
            for _, c_factor in self.stage_factors:
                c_prev = self.stage_channels[-1]
                self.stage_channels.append(c_prev * c_factor)
            if self.stage_channels[-1] != c_out:
                raise ValueError(
                    f"Channel expansion factors lead to C={self.stage_channels[-1]}, "
                    f"expected {c_out}."
                )

        # D -> F per reduced channel
        self.proj_df = nn.Conv1d(
            in_channels=self.c_red * d_latent,
            out_channels=self.c_red * f_hidden,
            kernel_size=1,
            groups=self.c_red,
            bias=False,
        )
        self.proj_gn = ChannelLayerNorm(num_channels=self.c_red * f_hidden, eps=1e-5)

        if not tie_channel_mats:
            self.channel_expanders = nn.ModuleList()
            for stage_idx, _ in enumerate(self.stage_t_factors):
                c_in = self.stage_channels[stage_idx]
                c_out_stage = self.stage_channels[stage_idx + 1]
                if c_in == c_out_stage:
                    self.channel_expanders.append(nn.Identity())
                elif self.channel_mlp_ratio is None:
                    self.channel_expanders.append(ChannelMix(c_in, c_out_stage))
                else:
                    self.channel_expanders.append(
                        ChannelMixMLP(
                            c_in,
                            c_out_stage,
                            bottleneck_ratio=self.channel_mlp_ratio,
                            dropout=dropout,
                        )
                    )
        else:
            self.channel_expanders = None

        self.up_blocks = nn.ModuleList()
        for stage_idx, t_factor in enumerate(self.stage_t_factors):
            c_stage = self.stage_channels[stage_idx + 1]
            self.up_blocks.append(
                TemporalUpsampleBlock(
                    c=c_stage,
                    f=f_hidden,
                    stride=t_factor,
                    k=k_up,
                    dropout=dropout,
                    aa_kernel_size=aa_kernel_size,
                )
            )

        # Refine + final projection F -> 1 per ROI
        # First keep [B, C*F, T] then grouped convs.
        self.refine = nn.Sequential(
            CausalConv1d(
                c_out * f_hidden,
                c_out * f_hidden,
                kernel_size=k_refine,
                groups=c_out * f_hidden,
                bias=False,
            ),
            ChannelLayerNorm(num_channels=c_out * f_hidden, eps=1e-5),
            nn.GELU(),
            nn.Conv1d(
                c_out * f_hidden,
                c_out * f_hidden,
                kernel_size=1,
                groups=c_out,
                bias=False,
            ),
            ChannelLayerNorm(num_channels=c_out * f_hidden, eps=1e-5),
            nn.GELU(),
        )
        self.to_x = nn.Conv1d(
            in_channels=c_out * f_hidden,
            out_channels=c_out,
            kernel_size=1,
            groups=c_out,
            bias=True,
        )

    def forward(
        self,
        z: torch.Tensor,
        T_target: Optional[int] = None,
        W_reduce_from_encoder: Optional[
            Tuple[Optional[torch.Tensor], ...] | torch.Tensor
        ] = None,
        return_last_dim_1: bool = True,
    ) -> torch.Tensor:
        """
        z: [B, C_red, T_red, D] or unbatched [C_red, T_red, D]
        T_target: if provided, crop/pad output time to match exactly.
        W_reduce_from_encoder: pass encoder.get_reduce_weights() if tie_channel_mats=True
        """
        was_unbatched = z.dim() == 3
        if was_unbatched:
            z = z.unsqueeze(0)
        if z.dim() != 4:
            raise ValueError(
                f"Expected z as [B,C_red,T_red,D] or [C_red,T_red,D], "
                f"got {tuple(z.shape)}"
            )

        B, C_red, T_red, D = z.shape
        assert C_red == self.c_red, f"Expected C_red={self.c_red}, got {C_red}"
        assert D == self.d_latent, f"Expected D={self.d_latent}, got {D}"

        # [B, C_red, T_red, D] -> [B, C_red*D, T_red]
        h = z.permute(0, 1, 3, 2).contiguous().view(B, C_red * D, T_red)

        # D -> F per reduced channel => [B, C_red*F, T_red]
        h = self.proj_df(h)
        h = self.proj_gn(h)
        h = F.gelu(h)

        # reshape to [B, C_red, F, T_red]
        h = h.view(B, C_red, self.f_hidden, T_red)

        if self.tie_channel_mats:
            if W_reduce_from_encoder is None:
                raise ValueError(
                    "tie_channel_mats=True but W_reduce_from_encoder was not provided."
                )
            if isinstance(W_reduce_from_encoder, torch.Tensor):
                if len(self.stage_t_factors) != 1:
                    raise ValueError(
                        "Single W_reduce provided but decoder has multiple stages."
                    )
                W_reduce_from_encoder = (W_reduce_from_encoder,)
            else:
                W_reduce_from_encoder = tuple(W_reduce_from_encoder)
            if len(W_reduce_from_encoder) != len(self.stage_t_factors):
                raise ValueError(
                    "W_reduce_from_encoder length does not match decoder stages."
                )

        T_cur = T_red
        c_cur = self.c_red
        n_stages = len(self.stage_t_factors)
        for stage_idx, _ in enumerate(self.stage_t_factors):
            c_next = self.stage_channels[stage_idx + 1]
            if c_next != c_cur:
                if self.tie_channel_mats:
                    enc_idx = n_stages - 1 - stage_idx
                    W_reduce = W_reduce_from_encoder[enc_idx]
                    if W_reduce is None:
                        raise ValueError(f"Missing encoder weight for stage {enc_idx}.")
                    h = _apply_channel_mix_weight(h, W_reduce.t())
                else:
                    reducer = self.channel_expanders[stage_idx]
                    if not isinstance(reducer, nn.Identity):
                        h = reducer(h)
            c_cur = c_next
            h = h.reshape(B, c_cur * self.f_hidden, T_cur)
            h = self.up_blocks[stage_idx](h)
            T_cur = h.shape[-1]
            h = h.view(B, c_cur, self.f_hidden, T_cur)

        h = h.reshape(B, c_cur * self.f_hidden, T_cur)

        # refine + project to [B, C, T]
        h = self.refine(h)
        x_hat = self.to_x(h)  # [B, C, T_hat]

        # match target length if needed
        if T_target is not None:
            T_hat = x_hat.shape[-1]
            if T_hat > T_target:
                x_hat = x_hat[..., :T_target]
            elif T_hat < T_target:
                x_hat = F.pad(x_hat, (0, T_target - T_hat))

        # return with last dim 1 if desired
        x_hat = _restore_shape(
            x_hat, was_unbatched=was_unbatched, want_last_dim_1=return_last_dim_1
        )
        return x_hat


@dataclass
class MEGFactorizedOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    recon_loss: Optional[torch.Tensor] = None

    latents: Optional[torch.Tensor] = None
    x_hat: Optional[torch.Tensor] = None


class MEGFactorizedConfig(PretrainedConfig):
    model_type = "meg-factorized"

    def __init__(
        self,
        c_in: int = 68,
        c_out: Optional[int] = None,
        c_down: int = 4,
        t_down: int = 4,
        t_down_factors: Optional[Tuple[int, ...]] = None,
        c_down_factors: Optional[Tuple[int, ...]] = None,
        c_stages: Optional[Tuple[int, ...]] = None,
        t_up: Optional[int] = None,
        t_up_factors: Optional[Tuple[int, ...]] = None,
        c_up_factors: Optional[Tuple[int, ...]] = None,
        c_up_stages: Optional[Tuple[int, ...]] = None,
        channel_mlp_ratio: Optional[float] = None,
        f_hidden: int = 32,
        d_latent: int = 128,
        k_stem: int = 7,
        k_down: int = 5,
        k_up: int = 4,
        k_refine: int = 7,
        dropout: float = 0.0,
        aa_kernel_size: int = 3,
        aa_kernel_size_up: Optional[int] = None,
        tie_channel_mats: bool = False,
        recon_loss: str = "mse",  # "mse" | "l1" | "huber"
        huber_delta: float = 1.0,
        return_last_dim_1: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.c_in = c_in
        self.c_out = c_out if c_out is not None else c_in
        self.c_down = c_down
        self.t_down = t_down
        self.t_down_factors = t_down_factors
        self.c_down_factors = c_down_factors
        self.c_stages = c_stages
        self.t_up = t_down if t_up is None else t_up
        self.t_up_factors = t_up_factors
        self.c_up_factors = c_up_factors
        self.c_up_stages = c_up_stages
        self.channel_mlp_ratio = channel_mlp_ratio
        self.f_hidden = f_hidden
        self.d_latent = d_latent
        self.k_stem = k_stem
        self.k_down = k_down
        self.k_up = k_up
        self.k_refine = k_refine
        self.dropout = dropout
        self.aa_kernel_size = aa_kernel_size
        self.aa_kernel_size_up = (
            aa_kernel_size if aa_kernel_size_up is None else aa_kernel_size_up
        )
        self.tie_channel_mats = tie_channel_mats
        self.recon_loss = recon_loss
        self.huber_delta = huber_delta
        self.return_last_dim_1 = return_last_dim_1


class MEGFactorizedAutoencoder(PreTrainedModel):
    config_class = MEGFactorizedConfig
    base_model_prefix = "meg_factorized"

    def __init__(self, config: MEGFactorizedConfig | dict):
        if isinstance(config, dict):
            config = MEGFactorizedConfig(**config)
        super().__init__(config)

        self.encoder = MEGFactorizedEncoder(
            c_in=config.c_in,
            c_down=config.c_down,
            t_down=config.t_down,
            t_down_factors=config.t_down_factors,
            c_down_factors=config.c_down_factors,
            c_stages=config.c_stages,
            channel_mlp_ratio=config.channel_mlp_ratio,
            f_hidden=config.f_hidden,
            d_latent=config.d_latent,
            k_stem=config.k_stem,
            k_down=config.k_down,
            dropout=config.dropout,
            aa_kernel_size=config.aa_kernel_size,
        )
        t_up_factors = config.t_up_factors
        if t_up_factors is None and config.t_down_factors is not None:
            t_up_factors = tuple(reversed(config.t_down_factors))
        c_up_factors = config.c_up_factors
        if c_up_factors is None and config.c_down_factors is not None:
            c_up_factors = tuple(reversed(config.c_down_factors))
        c_up_stages = config.c_up_stages
        if c_up_stages is None and config.c_stages is not None:
            c_up_stages = tuple(list(reversed(config.c_stages[:-1])) + [config.c_out])
        self.decoder = MEGFactorizedDecoder(
            c_out=config.c_out,
            c_down=config.c_down,
            t_up=config.t_up,
            t_up_factors=t_up_factors,
            c_up_factors=c_up_factors,
            c_up_stages=c_up_stages,
            channel_mlp_ratio=config.channel_mlp_ratio,
            f_hidden=config.f_hidden,
            d_latent=config.d_latent,
            k_up=config.k_up,
            k_refine=config.k_refine,
            dropout=config.dropout,
            aa_kernel_size=config.aa_kernel_size_up,
            tie_channel_mats=config.tie_channel_mats,
        )

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _recon_loss(self, x_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mode = self.config.recon_loss.lower()
        if mode == "mse":
            return F.mse_loss(x_hat, target)
        if mode == "l1":
            return F.l1_loss(x_hat, target)
        if mode == "huber":
            return F.huber_loss(x_hat, target, delta=float(self.config.huber_delta))
        raise ValueError(f"Unknown recon_loss: {self.config.recon_loss}")

    @staticmethod
    def _infer_target_time_and_shape(
        target: torch.Tensor,
        explicit_t: Optional[int],
        channels_expected: Tuple[int, ...],
    ) -> Tuple[Optional[int], bool]:
        is_unbatched_trailing = (
            target.dim() == 3
            and target.shape[-1] == 1
            and target.shape[0] in channels_expected
            and target.shape[1] not in channels_expected
        )
        has_trailing_dim = target.dim() == 4 or is_unbatched_trailing
        if explicit_t is not None:
            return explicit_t, has_trailing_dim

        if target.dim() >= 3:
            t_dim = target.shape[-2] if has_trailing_dim else target.shape[-1]
        elif target.dim() == 2:
            t_dim = target.shape[-1]
        else:
            t_dim = None
        return t_dim, has_trailing_dim

    @torch.no_grad()
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        T_target: Optional[int] = None,
        return_last_dim_1: Optional[bool] = None,
    ) -> torch.Tensor:
        return_last_dim_1 = (
            self.config.return_last_dim_1
            if return_last_dim_1 is None
            else return_last_dim_1
        )
        decoder_kwargs = dict(
            T_target=T_target,
            return_last_dim_1=return_last_dim_1,
        )
        if self.config.tie_channel_mats:
            decoder_kwargs["W_reduce_from_encoder"] = self.encoder.get_reduce_weights()
        return self.decoder(latents, **decoder_kwargs)

    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        T_target: Optional[int] = None,
        return_dict: Optional[bool] = True,
    ) -> MEGFactorizedOutput:
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]

        return_dict = True if return_dict is None else return_dict
        target = inputs if labels is None else labels
        input_was_unbatched = inputs.dim() == 2 or (
            inputs.dim() == 3
            and inputs.shape[-1] == 1
            and inputs.shape[0] in (self.config.c_in, self.config.c_out)
        )

        latents = self.encoder(inputs)
        tgt_time, want_last_dim_1 = self._infer_target_time_and_shape(
            target, T_target, (self.config.c_out, self.config.c_in)
        )
        decoder_kwargs = dict(
            T_target=tgt_time,
            return_last_dim_1=want_last_dim_1,
        )
        if self.config.tie_channel_mats:
            decoder_kwargs["W_reduce_from_encoder"] = self.encoder.get_reduce_weights()
        latents_for_decode = latents[0] if input_was_unbatched else latents
        x_hat = self.decoder(latents_for_decode, **decoder_kwargs)

        recon_loss = self._recon_loss(x_hat, target)
        loss = recon_loss

        if not return_dict:
            return loss, recon_loss, x_hat, latents

        return MEGFactorizedOutput(
            loss=loss,
            recon_loss=recon_loss,
            latents=latents,
            x_hat=x_hat,
        )
