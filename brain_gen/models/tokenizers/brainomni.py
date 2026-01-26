# ADAPTED FROM: https://github.com/OpenTSLab/BrainOmni/

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...layers.brainomni.attn import RMSNorm
from ...layers.brainomni.loss import (
    get_frequency_domain_loss,
    get_pcc,
    get_time_loss,
    get_entropy,
)
from ...layers.brainomni.module import (
    BackWardSolution,
    BrainQuantizer,
    BrainSensorModuleFixed,
    BrainSensorModule,
    ForwardSolution,
)
from ...layers.brainomni.seanet import SEANetDecoder, SEANetEncoder


@dataclass
class CausalTokenSequence:
    """Container for flattened causal latent tokens."""

    embeddings: torch.Tensor  # (B, C, W, D)
    indices: torch.Tensor  # (B, C, W, Q)
    tokens_per_window: int
    num_windows: int
    overlap_ratio: float

    def detach(self) -> "CausalTokenSequence":
        return CausalTokenSequence(
            embeddings=self.embeddings.detach(),
            indices=self.indices.detach(),
            tokens_per_window=self.tokens_per_window,
            num_windows=self.num_windows,
            overlap_ratio=self.overlap_ratio,
        )


class CausalBrainTokenizerEncoder(nn.Module):
    """Same as the reference BrainTokenizerEncoder but with strictly causal convolutions
    so that each latent token depends only on current/past samples."""

    def __init__(
        self,
        n_filters: int,
        ratios,
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_neuro: int,
        n_head: int,
        dropout: float,
        n_residual_layers: int = 1,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.seanet_encoder = SEANetEncoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            causal=True,
            bidirectional=False,
            true_skip=True,
            n_residual_layers=n_residual_layers,
            lstm=lstm_layers,
        )
        self.neuros = nn.Parameter(torch.randn(n_neuro, n_dim))
        torch.nn.init.trunc_normal_(self.neuros, std=0.02)

        self.backwardsolution = BackWardSolution(
            n_dim=n_dim, n_head=n_head, dropout=dropout
        )
        self.k_proj = nn.Linear(n_dim, n_dim)

    def forward(self, x: torch.Tensor, sensor_embedding: torch.Tensor) -> torch.Tensor:
        """Args:

        x: B C N L  (windows already unfolded along time)     sensor_embedding: B C D
        Returns:     Latent tokens shaped B C N T D
        """
        B, C, N, _ = x.shape
        x = rearrange(x, "B C N L -> (B C N) 1 L")
        x = self.seanet_encoder(x)
        x = rearrange(x, "(B C N) D T -> B C (N T) D", B=B, C=C, N=N)

        B, C, W, _ = x.shape
        sensor_embedding = sensor_embedding.unsqueeze(2)
        x_flat = rearrange(x, "B C W D -> (B W) C D")
        k_flat = self.k_proj(rearrange(x + sensor_embedding, "B C W D -> (B W) C D"))
        neuros = (
            self.neuros.type_as(x_flat).unsqueeze(0).expand(x_flat.shape[0], -1, -1)
        )
        x = self.backwardsolution(neuros, k_flat, x_flat)
        x = rearrange(x, "(B N T) C D -> B C (N T) D", B=B, N=N)
        return rearrange(x, "B C (N T) D -> B C N T D", N=N)


class CausalBrainTokenizerDecoder(nn.Module):
    """Mirror of the reference BrainTokenizerDecoder with causal transposed convolutions
    to preserve autoregressive semantics at decode time."""

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        n_filters,
        ratios,
        kernel_size: int,
        last_kernel_size: int,
        dropout: float,
        n_residual_layers: int = 1,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.forwardsolution = ForwardSolution(n_dim, n_head, dropout)
        self.seanet_decoder = SEANetDecoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            causal=True,
            trim_right_ratio=1.0,
            bidirectional=False,
            true_skip=True,
            n_residual_layers=n_residual_layers,
            lstm=lstm_layers,
        )

    def forward(self, x: torch.Tensor, sensor_embedding: torch.Tensor) -> torch.Tensor:
        """Args:

        x: B C N T D     sensor_embedding: B C D Returns:     Reconstructed windows (B,
        C, N, L)
        """
        B, C, N, T, D = x.shape
        x = rearrange(x, "B C N T D -> (B N T) C D")
        sensor_flat = rearrange(
            sensor_embedding.view(B, -1, 1, 1, D).expand(-1, -1, N, T, -1),
            "B C N T D -> (B N T) C D",
        )
        x = self.forwardsolution(sensor_flat, x)
        x = rearrange(x, "(B N T) C D -> (B C N) D T", B=B, N=N, T=T)
        x = self.seanet_decoder(x)
        return rearrange(x, "(B C N) 1 L -> B C N L", B=B, N=N)


class BrainOmniCausalTokenizer(nn.Module):
    """Causal variant of BrainOmni tokenizer producing latency-aligned VQ tokens."""

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
        sensor_space: str = "source",  # source or sensor
        shuffle_channels: bool = False,
        compile: bool = False,
        forecast_decode_window_chunk: int | None = None,
        forecast_decode_max_tokens: int = 4096,
    ):
        super().__init__()
        self.window_length = window_length
        self.n_dim = n_dim
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.normalize = normalize
        self.codebook_size = codebook_size
        self.sensor_space = sensor_space
        self.shuffle_channels = shuffle_channels
        self.forecast_decode_window_chunk = forecast_decode_window_chunk
        self.forecast_decode_max_tokens = forecast_decode_max_tokens

        if sensor_space == "source":
            self.sensor_embed = BrainSensorModuleFixed(n_dim, num_sensors)
        elif sensor_space == "sensor":
            self.sensor_embed = BrainSensorModule(n_dim)
        else:
            raise ValueError(f"Invalid sensor space: {sensor_space}")

        self.encoder = CausalBrainTokenizerEncoder(
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_dim=n_dim,
            n_neuro=n_neuro,
            n_head=n_head,
            dropout=dropout,
            n_residual_layers=n_residual_layers,
            lstm_layers=lstm_layers,
        )
        self.quantizer = BrainQuantizer(
            n_dim=n_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            normalize=normalize,
        )
        self.decoder = CausalBrainTokenizerDecoder(
            n_dim=n_dim,
            n_head=n_head,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            dropout=dropout,
            n_residual_layers=n_residual_layers,
            lstm_layers=lstm_layers,
        )
        self.apply(self._init_weights)

        if compile:
            self.encoder = torch.compile(self.encoder)
            # self.quantizer = torch.compile(self.quantizer)
            self.decoder = torch.compile(self.decoder)
            self.sensor_embed = torch.compile(self.sensor_embed)

    # ----------------------------- helpers ----------------------------- #
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            if isinstance(m.weight, nn.Parameter):
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

    def _stride(self, overlap_ratio: float) -> int:
        stride = int(self.window_length * (1 - overlap_ratio))
        return max(stride, 1)

    def _prepare_windows(
        self, x: torch.Tensor, allow_shorter_windows: bool
    ) -> torch.Tensor:
        """Unfold inputs into fixed windows without padding.

        When enabled, shorter inputs are treated as a single window (no padding) as
        long as the length is divisible by the temporal reduction factor.
        """
        if x.shape[-1] < self.window_length:
            if not allow_shorter_windows:
                raise ValueError(
                    f"x must be at least window_length long, got {x.shape[-1]}"
                )
            hop_length = int(self.encoder.seanet_encoder.hop_length)
            if x.shape[-1] % hop_length != 0:
                raise ValueError(
                    f"x must be divisible by hop_length={hop_length} for short windows,"
                    f"got {x.shape[-1]} % {hop_length} != 0"
                )
            return x.unsqueeze(2)

        if x.shape[-1] % self.window_length != 0:
            raise ValueError(
                f"x must be divisible by window_length, got {x.shape[-1]} % "
                f"{self.window_length} != 0"
            )
        return x.unfold(dimension=-1, size=self.window_length, step=self.window_length)

    def unfold(self, x: torch.Tensor, overlap_ratio: float = 0.0) -> torch.Tensor:
        if x.shape[-1] < self.window_length:
            x = F.pad(x, pad=(0, self.window_length - x.shape[-1]))

        stride = self._stride(overlap_ratio)
        if stride < self.window_length:
            right_remain = (x.shape[-1] - self.window_length) % stride
            if right_remain > 0:
                x = F.pad(x, pad=(0, stride - right_remain))
        return x.unfold(dimension=-1, size=self.window_length, step=stride)

    def norm_target(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-6)
        return x

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.noise_std

    def _reshape_indices(
        self, indices: torch.Tensor, tokens_per_window: int | None = None
    ) -> Tuple[torch.Tensor, int]:
        """Args:

        indices: B C W Q (flattened) Returns:     indices_5d: B C N T Q and the inferred
        number of windows.
        """
        B, C, W, Q = indices.shape
        if tokens_per_window is None:
            raise ValueError("tokens_per_window must be provided for reshaping.")
        num_windows = math.ceil(W / tokens_per_window)
        pad_tokens = num_windows * tokens_per_window - W
        if pad_tokens > 0:
            pad_vals = indices[..., -1:, :].detach().expand(B, C, pad_tokens, Q)
            indices = torch.cat([indices, pad_vals], dim=2)
        indices = indices.view(B, C, num_windows, tokens_per_window, Q)
        return indices, num_windows

    def indices_to_embeddings(
        self, indices: torch.Tensor, tokens_per_window: int
    ) -> torch.Tensor:
        """Decode RVQ indices back into quantized embeddings."""
        total_levels = len(self.quantizer.rvq.layers)
        if indices.shape[-1] < total_levels:
            pad_q = total_levels - indices.shape[-1]
            pad = indices[..., -1:].expand(*indices.shape[:-1], pad_q)
            indices = torch.cat([indices, pad], dim=-1)
        indices, num_windows = self._reshape_indices(indices, tokens_per_window)
        quantized = 0.0
        for q, layer in enumerate(self.quantizer.rvq.layers):
            quantized = quantized + layer.decode(indices[..., q])
        return quantized.view(
            indices.shape[0], indices.shape[1], num_windows, tokens_per_window, -1
        )

    def decode_windows(
        self,
        indices: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        tokens_per_window: int,
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings is None:
            embeddings = self.indices_to_embeddings(indices, tokens_per_window)
        sensor_embedding = self.sensor_embed(pos, sensor_type)
        return self.decoder(embeddings, sensor_embedding)

    def _forecast_decode_chunk_size(self, tokens_per_window: int) -> int | None:
        """Determine the number of windows to decode per chunk during forecast."""
        if self.forecast_decode_window_chunk is not None:
            chunk = int(self.forecast_decode_window_chunk)
            return max(chunk, 1)
        max_tokens = int(self.forecast_decode_max_tokens)
        if tokens_per_window <= 0 or max_tokens <= 0:
            return None
        return max(1, max_tokens // tokens_per_window)

    def _decode_windows_chunked(
        self,
        indices_5d: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        tokens_per_window: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """Decode windows in chunks to cap peak memory during forecasting."""
        B, C, num_windows, _, Q = indices_5d.shape
        if chunk_size >= num_windows:
            flat_indices = indices_5d.reshape(B, C, -1, Q)
            return self.decode_windows(
                flat_indices, pos, sensor_type, tokens_per_window=tokens_per_window
            )

        sensor_embedding = self.sensor_embed(pos, sensor_type)
        windows = []
        for start in range(0, num_windows, chunk_size):
            end = min(start + chunk_size, num_windows)
            chunk_indices = indices_5d[:, :, start:end]
            chunk_flat = chunk_indices.reshape(B, C, -1, Q)
            embeddings = self.indices_to_embeddings(chunk_flat, tokens_per_window)
            windows.append(self.decoder(embeddings, sensor_embedding))
        return torch.cat(windows, dim=2)

    @staticmethod
    def overlap_add(windows: torch.Tensor, stride: int) -> torch.Tensor:
        """Convert overlapping windows back to a contiguous timeseries."""
        B, C, N, L = windows.shape
        total_len = stride * (N - 1) + L
        out = windows.new_zeros(B, C, total_len)
        counts = windows.new_zeros(B, C, total_len)
        for i in range(N):
            start = i * stride
            out[..., start : start + L] += windows[:, :, i]
            counts[..., start : start + L] += 1
        counts = torch.clamp(counts, min=1)
        return out / counts

    def set_eval_mode(self):
        self.shuffle_channels = False

    # ----------------------------- forward ----------------------------- #
    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs
    ):
        """Args:

        x: B C T pos: B C 2 (canonical 2-D layout) sensor_type: B C

        Keyword args:
            allow_shorter_windows: allow x shorter than window_length (no padding).
        """
        x, pos, sensor_type = inputs

        allow_shorter_windows = bool(kwargs.get("allow_shorter_windows", False))
        x = self._prepare_windows(x, allow_shorter_windows)

        sensor_embedding = self.sensor_embed(pos, sensor_type)

        if self.shuffle_channels:
            random_index = torch.randperm(x.shape[1], device=x.device)
            x = x.index_select(dim=1, index=random_index)
            sensor_embedding = sensor_embedding.index_select(dim=1, index=random_index)

        if self.noise_std > 0.0:
            x = self.add_noise(x)

        # cast to bf16
        # sensor_embedding = sensor_embedding.to(torch.bfloat16)

        feature = self.encoder(x, sensor_embedding)
        feature, indices, commitment_loss = self.quantizer(feature)
        x_rec = self.decoder(feature, sensor_embedding)

        x_rec = x_rec.float()
        if self.normalize:
            x = self.norm_target(x)

        time_loss = get_time_loss(x_rec, x)
        pcc = get_pcc(x_rec, x)
        amp_loss, phase_loss = get_frequency_domain_loss(x_rec, x)
        ppl = get_entropy(indices, self.codebook_size)
        return {
            "loss": time_loss
            + torch.exp(-pcc)
            + commitment_loss
            + amp_loss
            + 0.5 * phase_loss,
            "l1": time_loss.detach(),
            "pcc": pcc.detach(),
            "amp": amp_loss.detach(),
            "phi": phase_loss.detach(),
            "commit": commitment_loss.detach(),
            "ppl": ppl.detach(),
            "logits": x_rec.reshape(x_rec.shape[0], x_rec.shape[1], -1),
        }

    def tokenize(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs
    ) -> Tuple[CausalTokenSequence, torch.Tensor]:
        """Deterministic tokenization for inference/forecasting.

        Keyword args:
            allow_shorter_windows: allow x shorter than window_length (no padding).
        """
        x, pos, sensor_type = inputs

        allow_shorter_windows = bool(kwargs.get("allow_shorter_windows", False))
        x = self._prepare_windows(x, allow_shorter_windows)

        sensor_embedding = self.sensor_embed(pos, sensor_type)
        feature = self.encoder(x, sensor_embedding)
        feature, indices, commitment_loss = self.quantizer(feature)

        tokens_per_window = feature.shape[3]
        num_windows = feature.shape[2]
        tokens = CausalTokenSequence(
            embeddings=rearrange(feature, "B C N T D->B C (N T) D"),
            indices=rearrange(indices, "B C N T Q -> B C (N T) Q"),
            tokens_per_window=tokens_per_window,
            num_windows=num_windows,
            overlap_ratio=0.0,
        )

        self.pos = pos[0]
        self.sensor_type = sensor_type[0]
        self.tokens_per_window = tokens_per_window
        self.num_latents = feature.shape[1]

        return tokens

    def encode(self, *args, **kwargs) -> Tuple[CausalTokenSequence, torch.Tensor]:
        tokens = self.tokenize(*args, **kwargs)

        indices = rearrange(tokens.indices, "B C T Q -> B (T C) Q")
        return {
            "codes": indices.reshape(indices.shape[0], -1),
            # "rvq_codes": indices,
        }

    def forecast_strip_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        # if seq is > 2D reshape to 2D
        if seq.dim() > 2:
            seq = seq.reshape(seq.shape[0], -1)

        pos = self.pos.unsqueeze(0).repeat(seq.shape[0], 1, 1)
        sensor_type = self.sensor_type.unsqueeze(0).repeat(seq.shape[0], 1)

        num_quantizers = len(self.quantizer.rvq.layers)

        # decode the tokens
        indices = rearrange(
            seq, "B (T C Q) -> B C T Q", C=self.num_latents, Q=num_quantizers
        )
        chunk_size = self._forecast_decode_chunk_size(self.tokens_per_window)
        if chunk_size is None:
            x_rec = self.decode_windows(
                indices, pos, sensor_type, tokens_per_window=self.tokens_per_window
            )
        else:
            num_tokens = int(indices.shape[2])
            num_windows = math.ceil(num_tokens / self.tokens_per_window)
            if chunk_size >= num_windows:
                x_rec = self.decode_windows(
                    indices, pos, sensor_type, tokens_per_window=self.tokens_per_window
                )
            else:
                indices_5d, _ = self._reshape_indices(indices, self.tokens_per_window)
                x_rec = self._decode_windows_chunked(
                    indices_5d,
                    pos,
                    sensor_type,
                    self.tokens_per_window,
                    chunk_size,
                )

        return x_rec.reshape(x_rec.shape[0], x_rec.shape[1], -1)

    def decode(self, *args, **kwargs) -> torch.Tensor:
        return self.forecast_strip_tokens(*args, **kwargs)

    @torch.no_grad()
    def visualize(
        self, x: torch.Tensor, pos: torch.Tensor, sensor_type: torch.Tensor, **kwargs
    ):
        allow_shorter_windows = bool(kwargs.get("allow_shorter_windows", False))
        x = self._prepare_windows(x, allow_shorter_windows)
        sensor_embedding = self.sensor_embed(pos, sensor_type)
        feature = self.encoder(x, sensor_embedding)
        feature, indices, _ = self.quantizer(feature)
        x_rec = self.decoder(feature, sensor_embedding)
        return {
            "x": self.norm_target(x),
            "x_rec": x_rec.float(),
            "sensor_type": sensor_type,
            "indices": indices,
        }

    def get_finetune_parameter_groups(self, weight_decay, layer_decay):
        del self.decoder
        del self.quantizer
        parameter_groups = {}

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue

            this_weight_decay = weight_decay
            group_name = "decay"

            if group_name not in parameter_groups:
                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": layer_decay,
                }

            parameter_groups[group_name]["params"].append(p)

        return list(parameter_groups.values())
