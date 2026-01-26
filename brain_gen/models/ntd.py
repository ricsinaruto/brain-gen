# ADAPTED FROM: https://github.com/mackelab/neural_timeseries_diffusion

import torch
import torch.nn as nn
from typing import Callable, Optional


import numpy as np
import math

import torch.nn.functional as F
from einops import rearrange, repeat
from scipy.linalg import cholesky_banded, solve_banded
from tqdm import tqdm

import logging

log = logging.getLogger(__name__)


def _off_diag(ell, sigma_squared):
    """Helper function of banded OU precision matrix."""
    return (1.0 / sigma_squared) * (np.exp(-(1 / ell))) / (np.exp(-(2 / ell)) - 1.0)


def _corner_diag(ell, sigma_squared):
    """Helper function of banded OU precision matrix."""
    return (1.0 / sigma_squared) * (1.0 / (1.0 - np.exp(-(2 / ell))))


def _mid_diag(ell, sigma_squared):
    """Helper function of banded OU precision matrix."""
    return (1.0 / sigma_squared) * (
        (1.0 + np.exp(-(2 / ell))) / (1.0 - np.exp(-(2 / ell)))
    )


def get_in_mask(
    signal_channel,
    hidden_channel,
    cond_channel=0,
):
    """Returns the input mask for the specified mode.

    Args:     signal_channel: Number of signal channels.     hidden_channel: Number of
    hidden channels.     time_channel: Number of diffusion time embedding channels.
    cond_channel: Number of conditioning channels. Returns:     Input mask as torch
    tensor.
    """
    np_mask = np.concatenate(
        (
            get_restricted(signal_channel, 1, hidden_channel),
            get_full(cond_channel, signal_channel * hidden_channel),
        ),
        axis=1,
    )
    return torch.from_numpy(np.float32(np_mask))


def get_mid_mask(signal_channel, hidden_channel, off_diag, num_heads=1):
    """Returns the hidden mask for the specified mode.

    Args:     signal_channel: Number of signal channels.     hidden_channel: Number of
    hidden channels.     off_diag: Number of off-diagonal interactions.     num_heads:
    Number of heads.

    Returns:     Mid mask as torch tensor.
    """
    np_mask = np.maximum(
        get_restricted(signal_channel, hidden_channel, hidden_channel),
        get_sub_interaction(signal_channel, hidden_channel, off_diag),
    )

    return torch.from_numpy(np.float32(np.repeat(np_mask, num_heads, axis=1)))


def get_out_mask(signal_channel, hidden_channel):
    """Returns the output mask for the specified mode.

    Args:     signal_channel: Number of signal channels.     hidden_channel: Number of
    hidden channels.

    Returns:     Output mask as torch tensor.
    """
    np_mask = get_restricted(signal_channel, hidden_channel, 1)
    return torch.from_numpy(np.float32(np_mask))


def get_full(num_in, num_out):
    """Get full mask containing all ones."""
    return np.ones((num_out, num_in))


def get_restricted(num_signal, num_in, num_out):
    """Get mask with ones only on the block diagonal."""
    return np.repeat(np.repeat(np.eye(num_signal), num_out, axis=0), num_in, axis=1)


def get_sub_interaction(num_signal, size_hidden, num_sub_interaction):
    """Get off-diagonal interactions."""
    sub_interaction = np.zeros((size_hidden, size_hidden))
    sub_interaction[:num_sub_interaction, :num_sub_interaction] = 1.0
    return np.tile(sub_interaction, (num_signal, num_signal))


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class WhiteNoiseProcess(nn.Module):
    """White noise process. Provides a sample method and a Mahalabonis distance method.
    In the case of white noise, this is just the (scaled) L2 distance.

    Args:     sigma_squared: Variance of the white noise.     signal_length: Length of
    the signal to sample and compute the distance on.
    """

    def __init__(self, sigma_squared, signal_length):
        super().__init__()
        self.sigma_squared = sigma_squared
        self.signal_length = signal_length  # needs to be implemented
        self.device = "cpu"

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        return self

    # Expects and returns tensor with shape (B, C, L).
    def sample(self, sample_shape, device="cpu"):
        return np.sqrt(self.sigma_squared) * torch.randn(*sample_shape, device=device)

    def sqrt_mal(self, train_batch):
        return (1 / self.sigma_squared) * train_batch


class OUProcess(nn.Module):
    """Ornstein-Uhlenbeck process. Provides a sample method and a Mahalabonis distance
    method. Supports linear time operations.

    Args:     sigma_squared: Variance of the process.     ell: Length scale of the
    process.     signal_length: Length of the signal to sample and compute the distance
    on.
    """

    def __init__(self, sigma_squared, ell, signal_length):
        super().__init__()
        self.sigma_squared = sigma_squared
        self.ell = ell
        self.signal_length = signal_length
        self.device = "cpu"  # default

        # Build banded precision (only diag and lower diag) because of symmetry.
        lower_banded = np.zeros((2, signal_length))
        lower_banded[0, 1:-1] = _mid_diag(ell, sigma_squared)
        lower_banded[0, 0] = _corner_diag(ell, sigma_squared)
        lower_banded[0, -1] = _corner_diag(ell, sigma_squared)
        lower_banded[1, :-1] = _off_diag(ell, sigma_squared)

        banded_lower_prec_numpy = cholesky_banded(lower_banded, lower=True)
        # Transpose as needed, matrix now in upper notation as a result.
        self.banded_upper_prec_numpy = np.zeros((2, signal_length))
        self.banded_upper_prec_numpy[0, 1:] = banded_lower_prec_numpy[1, :-1]
        self.banded_upper_prec_numpy[1, :] = banded_lower_prec_numpy[0, :]

        # Convert to torch tensor
        self.register_buffer(
            "banded_upper_prec",
            torch.from_numpy(np.float32(self.banded_upper_prec_numpy)),
        )

        self.register_buffer(
            "dense_upper_matrix",
            torch.diag(self.banded_upper_prec[0, 1:], diagonal=1)
            + torch.diag(self.banded_upper_prec[1, :], diagonal=0),
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        return self

    def sqrt_mal(self, train_batch):  # (B, C, L)
        assert self.signal_length == train_batch.shape[2]
        # Use lower-triangular application for temporal causality:
        # y_t depends on x_{<=t}
        superdiag_as_subdiag = self.banded_upper_prec[0]  # length L, 0 at index 0
        main_diag = self.banded_upper_prec[1]
        sub_mult = torch.einsum("l, bcl -> bcl", superdiag_as_subdiag, train_batch)
        main_mult = torch.einsum("l, bcl -> bcl", main_diag, train_batch)
        # add sub-diagonal contribution: shift left by one (from t-1 to t)
        main_mult[:, :, 1:] += sub_mult[:, :, :-1]
        return main_mult

    def sample(self, sample_shape, device="cpu"):
        return self.sample_gpu_dense(sample_shape, device)

    # O(n)
    def sample_numpy_banded(self, sample_shape):
        normal_samples = np.random.randn(*sample_shape)
        ou_samples = solve_banded(
            (0, 1),  # Upper triangular matrix.
            self.banded_upper_prec_numpy,
            np.transpose(normal_samples, (2, 1, 0)),
        )
        return torch.from_numpy(np.float32(np.transpose(ou_samples, (2, 1, 0)))).to(
            self.device
        )

    # This is not O(n), but for shorter sequences,
    # the theoretical advantage is dwarfed by GPU acceleration.
    def sample_gpu_dense(self, sample_shape, device="cpu"):
        """Draw samples from the OU process using a dense (upper triangular) precision
        matrix.  This path leverages GPU acceleration for speed but falls back to a CPU
        implementation on Apple M-series GPUs (``mps`` device) because
        ``torch.linalg.solve_triangular`` is known to segfault on that backend for
        certain input sizes (observed on M4 MacBook, see
        https://github.com/pytorch/pytorch/issues/98292).

        Parameters ---------- sample_shape : Tuple[int, ...]     Desired output shape
        ``(B, C, L)`` matching the call signature of     ``torch.randn``.
        """

        # draw standard normal noise on the *target* device so the final output
        # stays there regardless of which backend performs the triangular solve
        normal_samples = torch.randn(*sample_shape, device=device)

        res = torch.linalg.solve_triangular(
            self.dense_upper_matrix,
            torch.transpose(normal_samples, 1, 2),
            upper=True,
        )

        return torch.transpose(res, 1, 2)


class MaskedConv1d(nn.Module):
    """1D Convolutional layer with masking."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        assert (out_channels, in_channels) == mask.shape

        # Enforce causal (left-only) zero padding regardless of padding_mode
        # so no future timestep influences the present.
        self.padding_mode = "constant"
        total_padding = kernel_size - 1
        self.pad = [total_padding, 0]

        init_k = np.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = nn.Parameter(
            data=torch.FloatTensor(out_channels, in_channels, kernel_size).uniform_(
                -init_k, init_k
            ),
            requires_grad=True,
        )
        self.register_buffer("mask", mask)
        self.bias = (
            nn.Parameter(
                data=torch.FloatTensor(out_channels).uniform_(-init_k, init_k),
                requires_grad=True,
            )
            if bias
            else None
        )

    def forward(self, x):
        return F.conv1d(
            F.pad(x, self.pad, mode=self.padding_mode),
            self.weight * self.mask.unsqueeze(-1),
            self.bias,
        )


class EfficientMaskedConv1d(nn.Module):
    """1D Convolutional layer with masking."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask=None,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        if mask is None:
            # Build a standard conv without internal padding; we will apply
            # left-only zero padding in forward to ensure causality.
            self.layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                padding=0,
                padding_mode="zeros",
            )
            self.kernel_size = kernel_size
        else:
            self.layer = MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                mask,
                bias=bias,
                padding_mode=padding_mode,
            )

    def forward(self, x):
        if isinstance(self.layer, nn.Conv1d):
            left_pad = self.kernel_size - 1
            x = F.pad(x, [left_pad, 0], mode="constant")
            return self.layer(x)
        return self.layer.forward(x)


class GeneralEmbedder(nn.Module):
    def __init__(self, cond_channel, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_channel, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, cond):
        cond = rearrange(cond, "b c l -> b l c")
        cond = self.mlp(cond)
        return rearrange(cond, "b l c -> b c l")


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings.

        :param t: a 1-D Tensor of N indices, one per batch element. These may be
            fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SLConv(nn.Module):
    """Structured Long Convolutional layer. Adapted from
    https://github.com/ctlllll/SGConv.

    Args:     kernel_size: Kernel size used to build convolution.     num_channels:
    Number of channels.     num_scales: Number of scales.         Overall length will
    be: kernel_size * (2 ** (num_scales - 1))     decay_min: Minimum decay. Advanced
    option.     decay_max: Maximum decay. Advanced option.     heads: Number of heads.
    padding_mode: Padding mode. Either "zeros" or "circular".     use_fft_conv: Whether
    to use FFT convolution.     interpolate_mode: Interpolation mode. Either "nearest"
    or "linear".
    """

    def __init__(
        self,
        kernel_size,
        num_channels,
        num_scales,
        decay_min=2.0,
        decay_max=2.0,
        heads=1,
        padding_mode="zeros",
        use_fft_conv=False,
        interpolate_mode="nearest",
    ):
        super().__init__()
        assert decay_min <= decay_max

        self.h = num_channels
        self.num_scales = num_scales
        self.kernel_length = kernel_size * (2 ** (num_scales - 1))

        self.heads = heads

        # Causal semantics are enforced irrespective of the padding_mode argument.
        # We preserve the attribute but always perform left-only zero padding.
        self.padding_mode = "constant"
        self.use_fft_conv = use_fft_conv
        self.interpolate_mode = interpolate_mode

        self.D = nn.Parameter(torch.randn(self.heads, self.h))

        total_padding = self.kernel_length - 1
        # Left-only padding for causality
        self.pad = [total_padding, 0]

        # Init of conv kernels. There are more options here.
        # Full kernel is always normalized by initial kernel norm.
        self.kernel_list = nn.ParameterList()
        for _ in range(self.num_scales):
            kernel = nn.Parameter(torch.randn(self.heads, self.h, kernel_size))
            self.kernel_list.append(kernel)

        # Support multiple scales. Only makes sense in non-sparse setting.
        self.register_buffer(
            "multiplier",
            torch.linspace(decay_min, decay_max, self.h).view(1, -1, 1),
        )
        self.register_buffer("kernel_norm", torch.ones(self.heads, self.h, 1))
        self.register_buffer(
            "kernel_norm_initialized", torch.tensor(0, dtype=torch.bool)
        )

    def forward(self, x):
        signal_length = x.size(-1)

        kernel_list = []
        for i in range(self.num_scales):
            kernel = F.interpolate(
                self.kernel_list[i],
                scale_factor=2 ** (max(0, i - 1)),
                mode=self.interpolate_mode,
            ) * self.multiplier ** (self.num_scales - i - 1)
            kernel_list.append(kernel)
        k = torch.cat(kernel_list, dim=-1)

        if not self.kernel_norm_initialized:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(
                1, dtype=torch.bool, device=k.device
            )
            log.debug(f"Kernel norm: {self.kernel_norm.mean()}")
            log.debug(f"Kernel size: {k.size()}")

        assert k.size(-1) < signal_length
        if self.use_fft_conv:
            k = F.pad(k, (0, signal_length - k.size(-1)))

        k = k / self.kernel_norm

        # Convolution
        if self.use_fft_conv:
            if self.padding_mode == "constant":
                factor = 2
            elif self.padding_mode == "circular":
                factor = 1

            k_f = torch.fft.rfft(k, n=factor * signal_length)  # (C H L)
            u_f = torch.fft.rfft(x, n=factor * signal_length)  # (B H L)
            y_f = torch.einsum("bhl,chl->bchl", u_f, k_f)
            slice_start = self.kernel_length // 2
            y = torch.fft.irfft(y_f, n=factor * signal_length)

            if self.padding_mode == "constant":
                y = y[..., slice_start : slice_start + signal_length]  # (B C H L)
            elif self.padding_mode == "circular":
                y = torch.roll(y, -slice_start, dims=-1)
            y = rearrange(y, "b c h l -> b (h c) l")
        else:
            # Pytorch implements convolutions as cross-correlations! flip necessary
            padded = F.pad(x, self.pad, mode=self.padding_mode)
            k_flip = rearrange(k.flip(-1), "c h l -> (h c) 1 l")
            y = F.conv1d(padded, k_flip, groups=self.h)

        # Compute D term in state space equation - essentially a skip connection
        y = y + rearrange(
            torch.einsum("bhl,ch->bchl", x, self.D),
            "b c h l -> b (h c) l",
        )

        return y


class ChannelLayerNorm1d(nn.Module):
    """LayerNorm applied across channels per time step for inputs of shape (B, C, L).

    This avoids mixing statistics across the temporal dimension, preserving causality.
    """

    def __init__(self, num_channels: int, affine: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(
            normalized_shape=num_channels, elementwise_affine=affine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        x_perm = rearrange(x, "b c l -> b l c")
        x_norm = self.norm(x_perm)
        return rearrange(x_norm, "b l c -> b c l")


class AdaConvBlock(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_scales = num_scales
        self.mid_mask = mid_mask

        self.conv = SLConv(
            self.kernel_size,
            channel,
            num_scales=self.num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.mlp = nn.Sequential(
            EfficientMaskedConv1d(channel, channel, 1, mask=self.mid_mask),
            nn.GELU(),
            EfficientMaskedConv1d(channel, channel, 1, mask=self.mid_mask),
        )

        # Channel-only normalization per time step to maintain temporal causality
        self.norm1 = ChannelLayerNorm1d(channel, affine=False)
        self.norm2 = ChannelLayerNorm1d(channel, affine=False)

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel // 3, channel * 6, bias=True),
        )

        self.ada_ln[-1].weight.data.zero_()
        self.ada_ln[-1].bias.data.zero_()

    def forward(self, x, t_cond):
        y = x
        y = self.norm1(y)
        temp = self.ada_ln(rearrange(t_cond, "b c l -> b l c"))
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = rearrange(
            temp, "b l c -> b c l"
        ).chunk(6, dim=1)
        y = modulate(y, shift_tm, scale_tm)
        y = self.conv(y)
        y = x + gate_tm * y

        x = y
        y = self.norm2(y)
        y = modulate(y, shift_cm, scale_cm)
        y = x + gate_cm * self.mlp(y)
        return y


class AdaConv(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channel,
        cond_dim=0,
        hidden_channel=8,
        in_kernel_size=1,
        out_kernel_size=1,
        slconv_kernel_size=17,
        num_scales=5,
        num_blocks=3,
        num_off_diag=8,
        use_pos_emb=False,
        padding_mode="circular",
        use_fft_conv=False,
        mask_channel=0,
    ):
        super().__init__()
        self.signal_channel = signal_channel
        signal_channel += mask_channel

        self.signal_length = signal_length
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = signal_channel * hidden_channel

        in_mask = get_in_mask(signal_channel, hidden_channel, 0)
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, 1)
        )

        self.conv_in = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                AdaConvBlock(
                    slconv_kernel_size,
                    hidden_channel_full,
                    num_scales,
                    signal_length=signal_length,
                    mid_mask=mid_mask,
                    padding_mode=padding_mode,
                    use_fft_conv=use_fft_conv,
                )
                for _ in range(num_blocks)
            ]
        )

        out_mask = get_out_mask(signal_channel, hidden_channel)
        out_mask = out_mask[: signal_channel - mask_channel]

        self.conv_out = EfficientMaskedConv1d(
            hidden_channel_full,
            signal_channel - mask_channel,
            out_kernel_size,
            out_mask,
            padding_mode=padding_mode,
        )

        self.t_emb = TimestepEmbedder(hidden_channel_full // 3)
        if self.use_pos_emb:
            self.pos_emb = TimestepEmbedder(hidden_channel_full // 3)
        if cond_dim > 0:
            self.cond_emb = GeneralEmbedder(cond_dim, hidden_channel_full // 3)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """TODO: add channel embedding.

        Args:     x: (B, C, L)     t: (B,)     cond: (B, C_cond, L)

        Returns:     (B, C, L)
        """
        x = self.conv_in(x)

        t_emb = self.t_emb(t.to(x.device))
        t_emb = repeat(t_emb, "b c -> b c l", l=self.signal_length)

        pos_emb = 0
        if self.use_pos_emb:
            pos_emb = self.pos_emb(torch.arange(self.signal_length).to(x.device))
            pos_emb = repeat(pos_emb, "l c -> b c l", b=x.shape[0])

        cond_emb = 0
        if cond is not None:
            cond_emb = self.cond_emb(cond)

        emb = t_emb + pos_emb + cond_emb

        for block in self.blocks:
            x = block(x, emb)

        x = self.conv_out(x)
        return x


class NTD(nn.Module):
    """Neurophysiological Time-series Diffusion (NTD)

    Parameters ---------- signal_length : int     Length (L) of the training signals.
    signal_channel : int     Number of channels (C). cond_dim : int, default 0
    Dimensionality of optional conditioning channels (C_cond). If you pass     extra
    channels for conditioning, they must match (B, cond_dim, L). diffusion_time_steps :
    int     Number of diffusion steps T. schedule : {"linear", "quad"}     Noise
    schedule type for betas. start_beta, end_beta : float     Range for beta schedule.
    ou_sigma2 : float     OU process variance. ou_ell : float     OU process length
    scale. net_hidden_channel : int     Hidden channels per signal channel for AdaConv.
    net_in_kernel_size, net_out_kernel_size : int net_slconv_kernel_size : int
    net_num_scales : int net_num_blocks : int net_num_off_diag : int net_use_pos_emb :
    bool net_padding_mode : str net_use_fft_conv : bool
    """

    def __init__(
        self,
        *,
        signal_length: int,
        signal_channel: int,
        cond_dim: int = 0,
        diffusion_time_steps: int = 1000,
        schedule: str = "linear",
        start_beta: float = 1e-4,
        end_beta: float = 2e-2,
        ou_sigma2: float = 1.0,
        ou_ell: float = 10.0,
        net_hidden_channel: int = 8,
        net_in_kernel_size: int = 1,
        net_out_kernel_size: int = 1,
        net_slconv_kernel_size: int = 17,
        net_num_scales: int = 5,
        net_num_blocks: int = 3,
        net_num_off_diag: int = 8,
        net_use_pos_emb: bool = False,
        net_padding_mode: str = "circular",
        net_use_fft_conv: bool = False,
        noise_process: str = "ou",
        mask_channel: int = 0,
        p_forecast: float = 0.0,
    ):
        super().__init__()

        # --- Noise process (sampler + Mahalanobis) ---
        if noise_process == "ou":
            ou_process = OUProcess(ou_sigma2, ou_ell, signal_length)
        elif noise_process == "white":
            ou_process = WhiteNoiseProcess(ou_sigma2, signal_length)
        else:
            raise ValueError(f"Unknown noise process: {noise_process}")

        # --- Denoiser network ---
        self.network = AdaConv(
            signal_length=signal_length,
            mask_channel=mask_channel,
            signal_channel=signal_channel,
            cond_dim=cond_dim,
            hidden_channel=net_hidden_channel,
            in_kernel_size=net_in_kernel_size,
            out_kernel_size=net_out_kernel_size,
            slconv_kernel_size=net_slconv_kernel_size,
            num_scales=net_num_scales,
            num_blocks=net_num_blocks,
            num_off_diag=net_num_off_diag,
            use_pos_emb=net_use_pos_emb,
            padding_mode=net_padding_mode,
            use_fft_conv=net_use_fft_conv,
        )
        assert self.network.signal_length == ou_process.signal_length

        self.noise_sampler = ou_process
        self.mal_dist_computer = ou_process
        self.schedule = schedule
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.mask_channel = mask_channel
        self.p_forecast = p_forecast

        self._create_schedule(diffusion_time_steps)

    def _create_schedule(self, diffusion_time_steps: int):
        self.diffusion_time_steps = diffusion_time_steps

        if self.schedule == "linear":
            betas = torch.linspace(self.start_beta, self.end_beta, diffusion_time_steps)
        elif self.schedule == "quad":
            betas = (
                torch.linspace(
                    self.start_beta**0.5, self.end_beta**0.5, diffusion_time_steps
                )
                ** 2.0
            )
        else:
            raise ValueError("Unknown schedule type.")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("unormalized_probs", torch.ones(self.diffusion_time_steps))

    def _get_beta(self, timestep: int) -> torch.Tensor:
        return self.betas[timestep] ** 0.5

    def _get_alpha_beta(self, timestep: int) -> torch.Tensor:
        if timestep == 0:
            return self._get_beta(timestep)
        return (
            ((1.0 - self.alpha_bars[timestep - 1]) / (1.0 - self.alpha_bars[timestep]))
            * self.betas[timestep]
        ) ** 0.5

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device_ = args[0] if args else kwargs.get("device", "cpu")
        self.noise_sampler.to(*args, **kwargs)
        self.mal_dist_computer.to(*args, **kwargs)
        return self

    def sample_mask(
        self,
        x: torch.Tensor,
        min_L: int = 1,
        max_L: int = -1,
    ) -> torch.Tensor:
        """Shape  : (B, C, L_total) Returns a binary mask tensor with the same shape.

        Args:     shape: (B, C, L_total)     p_forecast: probability of forecast
        scenario     min_L: minimum length of forecast scenario     max_L: maximum
        length of forecast scenario

        Returns:     mask: (B, C, L_total)
        """
        B, C, L = x.shape
        max_L = L - 1 if max_L == -1 else max_L

        # Keep sampling in the graph to avoid Dynamo breaks on .item()/Python control flow.
        forecast = torch.rand((), device=x.device) < self.p_forecast
        Lf = torch.randint(min_L, max_L + 1, (), device=x.device)
        cutoff = L - Lf
        prefix = (torch.arange(L, device=x.device) < cutoff).to(dtype=x.dtype)
        mask = prefix.view(1, 1, L).expand(B, C, L) * forecast.to(dtype=x.dtype)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One forward diffusion step loss per sample (returns per-item losses).

        Args:     batch: (B, C, L)     cond : (B, C_cond, L) or None     mask : (B, C,
        L) -> 1 for observed (loss computed), 0 ignored

        Returns:     torch.Tensor: Per-sample losses.
        """
        if isinstance(x, (tuple, list)):
            x = x[0]
        B = x.shape[0]

        if mask is None and self.mask_channel > 0:
            mask = self.sample_mask(x)

        t_idx = self.unormalized_probs.multinomial(num_samples=B, replacement=True)
        a_bar = self.alpha_bars[t_idx].unsqueeze(-1).unsqueeze(-1)
        noise = self.noise_sampler.sample(
            sample_shape=(B, self.network.signal_channel, self.network.signal_length),
            device=x.device,
        )
        assert noise.shape == x.shape

        # forward diffusion
        noisy = torch.sqrt(a_bar) * x + torch.sqrt(1.0 - a_bar) * noise

        if mask is not None:
            noisy = torch.cat([noisy, mask[:, :1, :]], dim=1)  # suggested by chatgpt

        pred_noise = self.network(noisy, t_idx, cond=cond)

        return noise, pred_noise, mask

    def loss(
        self,
        noise: torch.Tensor,
        pred_noise: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduce: str = "mean",
    ) -> torch.Tensor:
        diff = noise - pred_noise
        maha = self.mal_dist_computer.sqrt_mal(diff)
        if mask is not None:
            maha = maha * (1 - mask)

        # per-sample loss
        # return torch.einsum("bcl,bcl->b", maha, maha)

        if reduce == "mean":
            return (maha**2).mean()
        elif reduce == "sum":
            return (maha**2).sum()

        return maha**2

    def sample(
        self,
        B: int,
        device: torch.device = "cuda",
        cond: Optional[torch.Tensor] = None,
        sample_length: Optional[int] = None,
        sampler: Optional[OUProcess] = None,
        noise_type: str = "alpha_beta",
    ) -> torch.Tensor:
        channels = self.network.signal_channel
        mask = None
        if sampler is None:
            sampler = self.noise_sampler
        if sample_length is None:
            sample_length = sampler.signal_length

        if cond is not None:
            b_cond, c_cond, l_cond = cond.shape
            assert l_cond == sample_length
            if b_cond == 1:
                cond = cond.repeat(B, 1, 1)
            else:
                assert b_cond == B

        self.eval()
        with torch.no_grad():
            state = sampler.sample((B, channels, sample_length), device=device)
            if self.mask_channel > 0:
                mask = self.sample_mask(state)

            for i in range(self.diffusion_time_steps):
                t = self.diffusion_time_steps - i - 1
                t_vec = torch.full((B,), t, device=device, dtype=torch.long)
                eps = sampler.sample((B, channels, sample_length), device=device)

                x_in = state
                if mask is not None:
                    x_in = torch.cat([x_in, mask[:, :1, :]], dim=1)

                res = self.network(x_in, t_vec, cond=cond)

                state = (1 / torch.sqrt(self.alphas[t])) * (
                    state
                    - ((1.0 - self.alphas[t]) / torch.sqrt(1.0 - self.alpha_bars[t]))
                    * res
                )

                if t > 0:
                    sigma = (
                        self._get_alpha_beta(t)
                        if noise_type == "alpha_beta"
                        else self._get_beta(t)
                    )
                    state = state + sigma * eps
            return state

    def impute(
        self,
        signal: torch.Tensor,
        mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Signal : (B,C,L) observed target values (fill anywhere mask==1) mask   :

        (B,C,L) 1 = observed (keep), 0 = missing (sample) cond   : optional conditioning
        channels (B,Cc,L) Returns completed signal.
        """
        B, C, L = signal.shape
        assert mask.shape == signal.shape
        if cond is not None:
            assert cond.shape[0] == B and cond.shape[2] == L

        self.eval()
        with torch.no_grad():
            # ---- before the loop ----
            sqrt_alphas = torch.sqrt(self.alphas).to(signal.device)
            sqrt_alpha_bars = torch.sqrt(self.alpha_bars).to(signal.device)
            sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bars).to(signal.device)

            # Optional: initialise observed coords at x_T
            eps_init = self.noise_sampler.sample((B, C, L), device=signal.device)
            state = sqrt_alpha_bars[-1] * signal + sqrt_one_minus_ab[-1] * eps_init

            for i in range(self.diffusion_time_steps):
                t = self.diffusion_time_steps - i - 1
                t_vec = torch.full((B,), t, dtype=torch.long, device=signal.device)

                # independent noises
                eps_pred = self.noise_sampler.sample((B, C, L), device=signal.device)
                eps_obs = self.noise_sampler.sample((B, C, L), device=signal.device)

                x_in = state
                if mask is not None:
                    x_in = torch.cat([x_in, mask[:, :1, :]], dim=1)

                # network prediction ε_θ(x_t,t)
                eps_theta = self.network(x_in, t_vec, cond=cond)

                # ------ unknown / missing coordinates ------
                mu_tilde = (1.0 / sqrt_alphas[t]) * (
                    state - ((1.0 - self.alphas[t]) / sqrt_one_minus_ab[t]) * eps_theta
                )

                if t > 0:
                    sigma = torch.sqrt(
                        self.betas[t]
                        * (1.0 - self.alpha_bars[t - 1])
                        / (1.0 - self.alpha_bars[t])
                    )
                    mu_tilde += sigma * eps_pred

                # ------ known / observed coordinates ------
                noisy_obs = sqrt_alpha_bars[t] * signal + sqrt_one_minus_ab[t] * eps_obs
                mu_known = (1.0 / sqrt_alphas[t]) * (
                    noisy_obs - ((1.0 - self.alphas[t]) / sqrt_one_minus_ab[t]) * signal
                )

                # combine
                state = mask * mu_known + (1.0 - mask) * mu_tilde

        return state

    @torch.inference_mode()
    def forecast(
        self,
        past: torch.Tensor | tuple[torch.Tensor, ...],
        horizon: int,
        sample_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        *,
        cond: Optional[torch.Tensor] = None,
        sliding_window_overlap: float | int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Recursive diffusion forecast compatible with eval_runner.generate.

        Args:     past: Context tensor (B, C, L_ctx) or tuple whose first element is the
        context. Extra tuple elements (e.g., positions) are ignored.     horizon: Number
        of future steps to generate.     sample_fn: Unused placeholder for API parity
        with autoregressive models.     cond: Optional conditioning channels matching
        the context.     noise_type: Diffusion noise type for reverse sampling.
        sliding_window_overlap: Overlap ratio (float) or stride (int) between
        consecutive diffusion windows. Defaults to 0.5 overlap.     **kwargs: Extra
        arguments are accepted for interface compatibility.

        Returns:     Tensor of shape (B, C, horizon) containing the generated future
        samples.
        """
        _ = sample_fn
        _ = kwargs
        steps_total = int(horizon)
        if steps_total <= 0:
            raise ValueError("horizon must be positive for forecast.")

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        context = past[0] if isinstance(past, (tuple, list)) else past
        cond_tensor = cond
        if cond_tensor is None and isinstance(past, (tuple, list)) and len(past) > 1:
            maybe_cond = past[1]
            if torch.is_tensor(maybe_cond):
                cond_tensor = maybe_cond

        if context.dim() == 1:
            context = context.unsqueeze(0).unsqueeze(0)
        elif context.dim() == 2:
            context = context.unsqueeze(0)
        context = context.to(device)

        cond_window = None
        supports_cond = hasattr(self.network, "cond_emb")
        if cond_tensor is not None and supports_cond:
            cond_window = cond_tensor
            if cond_window.dim() == 2:
                cond_window = cond_window.unsqueeze(0)
            cond_window = cond_window.to(device)
        elif cond_tensor is not None:
            cond_tensor = None

        seg_len = self.network.signal_length
        # Limit initial context to the most recent segment
        if context.shape[-1] > seg_len:
            context = context[..., -seg_len:]
        if cond_window is not None and cond_window.shape[-1] > context.shape[-1]:
            cond_window = cond_window[..., -context.shape[-1] :]

        overlap = 0.5 if sliding_window_overlap is None else sliding_window_overlap
        if isinstance(overlap, int):
            stride = max(1, min(seg_len, int(overlap)))
        else:
            overlap_ratio = float(overlap)
            if overlap_ratio < 0.0:
                raise ValueError("sliding_window_overlap must be non-negative.")
            overlap_ratio = min(overlap_ratio, 0.999)
            stride = max(1, seg_len - int(round(seg_len * overlap_ratio)))
        past_len = seg_len - stride

        generated: list[torch.Tensor] = []
        total_generated = 0
        window_ctx = context
        pbar = tqdm(total=steps_total, desc="Diffusion forecast", leave=False)
        while total_generated < steps_total:
            if past_len > 0:
                ctx_slice = window_ctx[..., -past_len:]
            else:
                ctx_slice = window_ctx[..., :0]
            ctx_len = ctx_slice.shape[-1]

            B, C, _ = context.shape
            signal = torch.zeros((B, C, seg_len), device=device, dtype=context.dtype)
            mask = torch.zeros_like(signal)
            if ctx_len > 0:
                signal[..., :ctx_len] = ctx_slice
                mask[..., :ctx_len] = 1.0

            cond_full = None
            if cond_window is not None:
                cond_slice = (
                    cond_window[..., -ctx_len:] if ctx_len > 0 else cond_window[..., :0]
                )
                cond_full = torch.zeros(
                    (cond_slice.shape[0], cond_slice.shape[1], seg_len),
                    device=device,
                    dtype=cond_slice.dtype,
                )
                if ctx_len > 0:
                    cond_full[..., : cond_slice.shape[-1]] = cond_slice

            completed = self.impute(signal, mask, cond=cond_full)
            available_future = seg_len - ctx_len
            step = min(stride, available_future, steps_total - total_generated)
            new_segment = completed[..., ctx_len : ctx_len + step]
            generated.append(new_segment)
            total_generated += int(step)
            pbar.update(int(step))

            window_ctx = torch.cat([window_ctx, new_segment], dim=-1)
            if window_ctx.shape[-1] > seg_len:
                window_ctx = window_ctx[..., -seg_len:]

            if cond_window is not None:
                pad = torch.zeros(
                    (cond_window.shape[0], cond_window.shape[1], step),
                    device=cond_window.device,
                    dtype=cond_window.dtype,
                )
                cond_window = torch.cat([cond_window, pad], dim=-1)
                if cond_window.shape[-1] > seg_len:
                    cond_window = cond_window[..., -seg_len:]

        out = torch.cat(generated, dim=-1)
        pbar.close()
        if was_training:
            self.train()
        return out
