from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from einops import rearrange

from ...layers.brainomni.seanet import SEANetDecoder, SEANetEncoder
from .brainomni import BrainOmniCausalTokenizer


class _ZeroSensorEmbed(nn.Module):
    """Return a zero-valued sensor embedding for each channel."""

    def __init__(self, n_dim: int) -> None:
        super().__init__()
        self.n_dim = int(n_dim)

    def forward(self, pos: torch.Tensor, sensor_type: torch.Tensor) -> torch.Tensor:
        batch, channels = pos.shape[:2]
        return torch.zeros(
            (batch, channels, self.n_dim), device=pos.device, dtype=pos.dtype
        )


class CausalSEANetChannelMixEncoder(nn.Module):
    """Channel-mixing SEANet encoder with spatial token splitting."""

    def __init__(
        self,
        *,
        n_filters: int,
        ratios: Iterable[int],
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_neuro: int,
        input_channels: int,
        n_residual_layers: int = 1,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.n_neuro = n_neuro
        self.n_dim = n_dim
        self.seanet_encoder = SEANetEncoder(
            channels=input_channels,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=list(ratios),
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            causal=True,
            bidirectional=False,
            true_skip=True,
            n_residual_layers=n_residual_layers,
            lstm=lstm_layers,
        )

    def forward(
        self, x: torch.Tensor, sensor_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode causal windows into latent tokens.

        Args:
            x: B C N L (windows already unfolded along time).
            sensor_embedding: unused, included for interface compatibility.
        Returns:
            Latent tokens shaped B C N T D.
        """
        B, C, N, _ = x.shape
        if C != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {C}.")
        x = rearrange(x, "B C N L -> (B N) C L")
        x = self.seanet_encoder(x)
        x = rearrange(x, "(B N) D T -> B N T D", B=B, N=N)
        return rearrange(x, "B N T (C D) -> B C N T D", C=self.n_neuro)


class CausalSEANetChannelMixDecoder(nn.Module):
    """Channel-mixing SEANet decoder that mirrors the encoder."""

    def __init__(
        self,
        *,
        n_dim: int,
        n_neuro: int,
        n_filters: int,
        ratios: Iterable[int],
        kernel_size: int,
        last_kernel_size: int,
        output_channels: int,
        n_residual_layers: int = 1,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.n_neuro = n_neuro
        self.n_dim = n_dim
        self.seanet_decoder = SEANetDecoder(
            channels=output_channels,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=list(ratios),
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            causal=True,
            trim_right_ratio=1.0,
            bidirectional=False,
            true_skip=True,
            n_residual_layers=n_residual_layers,
            lstm=lstm_layers,
        )

    def forward(
        self, x: torch.Tensor, sensor_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Decode latent tokens back into windows.

        Args:
            x: B C N T D.
            sensor_embedding: unused, included for interface compatibility.
        Returns:
            Reconstructed windows (B, C, N, L).
        """
        B, C, N, T, D = x.shape
        if C != self.n_neuro:
            raise ValueError(f"Expected {self.n_neuro} latent channels, got {C}.")
        x = rearrange(x, "B C N T D -> (B N) (C D) T")
        x = self.seanet_decoder(x)
        return rearrange(x, "(B N) C L -> B C N L", B=B, N=N)


class BrainOmniCausalTokenizerSEANetChannelMix(BrainOmniCausalTokenizer):
    """BrainOmni tokenizer variant using channel-mixing SEANet and spatial splitting."""

    def __init__(
        self,
        window_length: int,
        n_filters,
        ratios,
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_neuro: int,
        n_head: int,
        codebook_dim: int,
        codebook_size: int,
        n_residual_layers: int = 1,
        lstm_layers: int = 2,
        dropout: float = 0.0,
        num_quantizers: int = 4,
        rotation_trick: bool = True,
        mask_ratio: float = 0.0,
        noise_std: float = 0.0,
        num_sensors: int = 68,
        normalize: bool = False,
        sensor_space: str = "source",
        shuffle_channels: bool = False,
        compile: bool = False,
    ):
        if n_dim % n_neuro != 0:
            raise ValueError(
                f"n_dim ({n_dim}) must be divisible by n_neuro ({n_neuro})."
            )
        token_dim = n_dim // n_neuro
        self.seanet_dim = n_dim
        self.n_neuro = n_neuro

        super().__init__(
            window_length=window_length,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_dim=token_dim,
            n_neuro=n_neuro,
            n_head=n_head,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            n_residual_layers=n_residual_layers,
            lstm_layers=lstm_layers,
            dropout=dropout,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            mask_ratio=mask_ratio,
            noise_std=noise_std,
            num_sensors=num_sensors,
            normalize=normalize,
            sensor_space=sensor_space,
            shuffle_channels=shuffle_channels,
            compile=False,
        )

        self.sensor_embed = _ZeroSensorEmbed(token_dim)
        self.encoder = CausalSEANetChannelMixEncoder(
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_dim=n_dim,
            n_neuro=n_neuro,
            input_channels=num_sensors,
            n_residual_layers=n_residual_layers,
            lstm_layers=lstm_layers,
        )
        self.decoder = CausalSEANetChannelMixDecoder(
            n_dim=n_dim,
            n_neuro=n_neuro,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            output_channels=num_sensors,
            n_residual_layers=n_residual_layers,
            lstm_layers=lstm_layers,
        )

        self.encoder.apply(self._init_weights)
        self.decoder.apply(self._init_weights)

        if compile:
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)
            self.sensor_embed = torch.compile(self.sensor_embed)
