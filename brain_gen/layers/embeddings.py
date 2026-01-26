import torch
import math

from typing import Optional
from torch import Tensor
from torch.nn import Embedding, Module


def _apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    if x.ndim == 3:
        # (tokens, channels, head_dim)
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)
    elif x.ndim == 4:
        # (batch, tokens, channels, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(0).unsqueeze(-2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)
    else:
        raise ValueError(
            "Unexpected tensor rank for rotary embedding: expected 3D or 4D"
        )


def rope_inv_freq(d_rot, base=10000.0, device=None):
    if d_rot == 0:
        return torch.empty(0, device=device)
    return 1.0 / (
        base ** (torch.arange(0, d_rot, 2, device=device, dtype=torch.float32) / d_rot)
    )


def apply_rope_1d(q, k, inv_freq, pos_start, L):
    # q,k: [B,h,L,d]; inv_freq: [d_rot/2]
    if inv_freq.numel() == 0:
        return q, k
    d = q.shape[-1]
    d_rot = (d // 2) * 2
    if d_rot == 0:
        return q, k
    t = torch.arange(
        pos_start, pos_start + L, device=q.device, dtype=torch.float32
    )  # [L]
    freqs = torch.einsum("l,f->lf", t, inv_freq)  # [L, d_rot/2]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).view(1, 1, L, d_rot).to(q.dtype)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).view(1, 1, L, d_rot).to(q.dtype)

    def rot(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q1, k1 = q[..., :d_rot], k[..., :d_rot]
    q2, k2 = q[..., d_rot:], k[..., d_rot:]
    q1 = q1 * cos + rot(q1) * sin
    k1 = k1 * cos + rot(k1) * sin
    return torch.cat([q1, q2], dim=-1), torch.cat([k1, k2], dim=-1)


class Embeddings(Module):
    """Handles various embeddings for conditioning, quantization, channels, and
    subjects."""

    def __init__(
        self,
        num_channels: int,
        quant_levels: int,
        quant_emb: int,
        num_classes: int = 0,
        class_emb: int = 0,
        channel_emb: int = 0,
        subjects: int = 0,
        sub_emb: int = 0,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels

        # embeddings for various conditioning
        self.quant_emb = Embedding(quant_levels, quant_emb)
        self.cond_emb = Embedding(num_classes, class_emb) if class_emb > 0 else None
        self.channel_emb = (
            Embedding(num_channels, channel_emb) if channel_emb > 0 else None
        )
        self.subject_emb = Embedding(subjects, sub_emb) if sub_emb > 0 else None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: Module) -> None:
        if isinstance(module, Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: Tensor,
        ids: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
        sid: Optional[Tensor] = None,
    ) -> Tensor:
        """Args: x: Input tensor of shape (B, C, T) ids: Channel IDs of shape (C,) cond:
        Conditional tensor of shape (B, 1, T) sid: Subject IDs of shape (B,)

        Returns:     Output tensor of shape (B*C, T, E_q + E_ch + E_c + E_s)
        """
        # if x is a tuple, unpack it
        if isinstance(x, tuple) or isinstance(x, list):
            x, cond = x

        channels, timesteps = x.shape[1], x.shape[2]

        # apply quantization embedding
        x = self.quant_emb(x)  # B x C x T x E_q

        embeddings = []

        # channel embeddings
        if self.channel_emb is not None:
            if ids is None:
                ids = torch.arange(self.num_channels, device=x.device)
            ch_ids = torch.arange(self.num_channels, device=x.device)
            ch_emb = self.channel_emb(ch_ids[ids])  # C x E_ch
            # repeat across batch and time:  B x T x C x E_ch
            ch_emb = ch_emb.expand(1, timesteps, -1, -1)
            ch_emb = ch_emb.permute(0, 2, 1, 3)  # B x C x T x E_ch
            embeddings.append(ch_emb)

        # conditional embeddings
        if self.cond_emb is not None and cond is not None:
            cond_emb = self.cond_emb(cond)  # B x 1 x T x E_c
            # set elements of cond to 0 where id is 0
            mask = cond.unsqueeze(-1) > 0
            cond_emb = cond_emb * mask.to(cond_emb.dtype)
            cond_emb = cond_emb.expand(-1, channels, -1, -1)  # B x C x T x E_c
            embeddings.append(cond_emb)

        # subject embeddings
        if self.subject_emb is not None and sid is not None:
            semb = self.subject_emb(sid)
            semb = semb.expand(-1, channels, -1, -1)  # B x C x T x E_s
            embeddings.append(semb)

        # Sum all embeddings
        if embeddings:
            x = x + sum(embeddings)

        return x.reshape(-1, x.shape[-2], x.shape[-1])  # B*C x T x E


class RotaryEmbedding(Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(
        self, device: torch.device | None = None
    ) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071."""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int, device: torch.device | None = None):
        concentration, inv_freq = self._compute_concentration_and_inv_freq(device)
        t = torch.arange(num_tokens, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Supports either flattened tokens-first tensors or batched tensors.
        # query/key shapes:
        # - Flattened: (tokens, ..., head_dim)
        # - Batched: (batch, tokens, ..., head_dim)
        if query.ndim == 4:
            # (tokens, channels, head_dim) after view in caller
            num_tokens = query.shape[0]
            cos, sin = self._compute_cos_sin(num_tokens, device=query.device)

            query_shape = query.shape
            query = query.view(num_tokens, -1, self.head_dim)
            query = _apply_rotary_emb(query, cos, sin)
            query = query.reshape(query_shape)

            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_dim)
            key = _apply_rotary_emb(key, cos, sin)
            key = key.reshape(key_shape)
            return query, key
        elif query.ndim == 5:
            # (batch, tokens, channels, head_dim) after view in caller
            bsz, num_tokens = query.shape[0], query.shape[1]
            cos, sin = self._compute_cos_sin(num_tokens, device=query.device)

            query_shape = query.shape
            query = query.view(bsz, num_tokens, -1, self.head_dim)
            query = _apply_rotary_emb(query, cos, sin)
            query = query.reshape(query_shape)

            key_shape = key.shape
            key = key.view(bsz, num_tokens, -1, self.head_dim)
            key = _apply_rotary_emb(key, cos, sin)
            key = key.reshape(key_shape)
            return query, key
        else:
            raise ValueError(
                "Unexpected query/key rank for RotaryEmbedding: expected"
                " 4D or 5D after view"
            )
