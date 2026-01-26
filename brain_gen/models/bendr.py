import torch
import torch.nn as nn
import copy
import numpy as np

from math import ceil
from typing import Sequence

from ..layers.transformer_blocks import TransformerBlock


class DebugModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        print(x.shape)
        return x


class CausalTransposeDecoder(nn.Module):
    def __init__(
        self,
        enc_width: Sequence[int],
        enc_stride: Sequence[int],
        in_ch: int,
        channels_out: int,
    ):
        """Construct a causal transpose‑convolutional stack that inverts the encoder.

        Each ConvTranspose1d layer mirrors a corresponding Conv1d layer in the encoder
        but in *reverse* order.  We deliberately use `padding=0` so the decoder remains
        *causal* (i.e. the output at time *t* depends **only** on inputs ≤ *t*).  To
        compensate for the receptive‑field shift introduced by strides > 1 we crop the
        extra samples at the very end of `forward()`.
        """
        super().__init__()
        self.enc_width = enc_width
        self.enc_stride = enc_stride
        self.in_ch = in_ch
        self.channels_out = channels_out

        layers = []
        self.kernels = list(enc_width)[::-1]
        self.strides = list(enc_stride)[::-1]

        # Internal feature dimension stays constant (in_ch) throughout the decoder.
        ch = in_ch
        for k, s in zip(self.kernels, self.strides):
            # layers.append(nn.ConstantPad1d((k - 1, 0), 0))
            layers.append(
                nn.ConvTranspose1d(
                    in_channels=ch,
                    out_channels=ch,
                    kernel_size=k,
                    stride=s,  # *causal* – no future leakage
                    padding=0,
                    output_padding=s - 1,  # guarantees length doubling by stride
                ),
            )
            layers.append(nn.GELU())

        self.conv_stack = nn.ModuleList(layers)

        # Final 1 × 1 projection back to raw channel space
        self.to_raw = nn.Conv1d(in_ch, channels_out, kernel_size=1)

    def forward(self, z: torch.Tensor):
        x = z
        layer_idx = 0
        for k, s in zip(self.kernels, self.strides):
            # Apply transposed convolution
            conv = self.conv_stack[layer_idx]
            act = self.conv_stack[layer_idx + 1]
            layer_idx += 2

            x = conv(x)
            # Remove padding replicas (first k−1 samples)
            if k > 1:
                x = x[..., (k - 1) :]
            x = act(x)

        # Final projection (no cropping needed, kernel=1)
        x = self.to_raw(x)
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class ConvEncoderBENDR(nn.Module):
    def __init__(
        self,
        in_features,
        encoder_h=256,
        enc_width=(3, 3, 3, 3, 3, 3),
        enc_downsample=(2, 2, 2, 1, 1, 1),
        dropout=0.0,
    ):
        super().__init__()
        self.encoder_h = encoder_h
        self.in_features = in_features
        assert len(enc_downsample) == len(enc_width)

        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module(
                "Encoder_{}".format(i),
                nn.Sequential(
                    # add left causal padding
                    nn.ConstantPad1d((width - downsample, 0), 0),
                    nn.Conv1d(
                        in_features,
                        encoder_h,
                        width,
                        stride=downsample,
                        padding=0,
                    ),
                    nn.Dropout1d(dropout),
                    nn.GroupNorm(encoder_h // 2, encoder_h),
                    nn.GELU(),
                ),
            )
            in_features = encoder_h

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = ceil(samples / factor)
        return samples

    def forward(self, x):
        return self.encoder(x)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze


class BENDRContextualizer(nn.Module):
    def __init__(
        self,
        in_features: int,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",
        mlp_type: str = "standard",
        layers: int = 8,
        dropout: float = 0.15,
        position_encoder: int = 25,
        layer_drop: float = 0.0,
        mask_p_t: float = 0.1,
        mask_p_c: float = 0.004,
        mask_t_span: int = 6,
        mask_c_span: int = 64,
        start_token: int = -5,
        finetuning: bool = False,
    ):
        super().__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3
        attn_args.setdefault("d_model", in_features * 3)
        mlp_args.setdefault("d_model", in_features * 3)

        encoder = TransformerBlock(
            attn_args=attn_args,
            mlp_args=mlp_args,
            attn_type=attn_type,
            mlp_type=mlp_type,
        )

        self.transformer_layers = nn.ModuleList(
            [copy.deepcopy(encoder) for _ in range(layers)]
        )
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(
                in_features,
                in_features,
                position_encoder,
                padding=position_encoder // 2,
                groups=16,
            )
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = (
                0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data
            )

    def forward(self, x: torch.Tensor):
        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(
                x.device
            ).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x, causal=True)

        return self.output_layer(x.permute([1, 2, 0]))

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False


def compute_downsample_factor(strides):
    """Utility to compute the cumulative down‑sampling factor of ConvEncoderBENDR.

    Args:     strides (Sequence[int]): strides used in the encoder (enc_downsample)
    Returns:     int: cumulative down‑sampling factor
    """
    factor = 1
    for s in strides:
        factor *= s
    return factor


class BENDRForecast(nn.Module):
    """Forecast the next raw EEG sample for *all* channels.

    The model reuses the *core* BENDR architecture – a stack of 1‑D convolutions
    followed by a Transformer encoder – but adds a lightweight *projection* head that
    maps the contextualised embeddings back to the raw signal space.

    During training the model receives a sequence **x** with shape ``(batch, channels,
    samples)`` and is optimised with *mean‑squared error* to predict **x** *one timestep
    into the future*.
    """

    def __init__(
        self,
        channels: int,
        samples: int,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",
        mlp_type: str = "standard",
        encoder_h: int = 512,
        enc_width=(3, 2, 2, 2, 2, 2),
        enc_downsample=(2, 2, 2, 1, 1, 1),
        transformer_layers: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Encoder (identical to original BENDR)
        # ------------------------------------------------------------------
        self.encoder = ConvEncoderBENDR(
            in_features=channels,
            encoder_h=encoder_h,
            enc_width=enc_width,
            enc_downsample=enc_downsample,
            dropout=dropout,  # keep feature extractor deterministic
        )

        # ------------------------------------------------------------------
        # Transformer contextualiser (identical hyper‑params to paper)
        # ------------------------------------------------------------------
        self.contextualiser = BENDRContextualizer(
            in_features=encoder_h,
            attn_args=attn_args,
            mlp_args=mlp_args,
            attn_type=attn_type,
            mlp_type=mlp_type,
            layers=transformer_layers,
            dropout=dropout,
            finetuning=False,  # training from scratch
            mask_p_t=0.0,  # *disable* masking for forecasting objective
            mask_p_c=0.0,
            position_encoder=25,
        )

        # ------------------------------------------------------------------
        # NEW: projection layer back to raw sample space
        # ------------------------------------------------------------------
        self.project = CausalTransposeDecoder(
            enc_width=enc_width,
            enc_stride=enc_downsample,
            in_ch=encoder_h,
            channels_out=channels,
        )

        # The cumulative down‑sampling factor so that we know which target step
        # each encoded element should predict.
        self._ds_factor = compute_downsample_factor(enc_downsample)
        self._encoded_len = samples // self._ds_factor
        self.receptive_field = (self._encoded_len - 1) * self._ds_factor
        self.output_dim = encoder_h

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor):
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]

        # Inputs use only the context window up to the last encoded step.
        inputs = x[..., : (self._encoded_len - 1) * self._ds_factor]

        # Convolutional feature extractor
        z = self.encoder(inputs)  # (B, F, Tenc)
        # Contextualisation
        c = self.contextualiser(z)  # (B, F, Tenc)
        # Project back to channels
        return c

    def forward(self, x: torch.Tensor):
        """Run a forward pass.

        Args:     x (Tensor): raw EEG with shape *(B, C, T)*.

        Returns:     Tensor: prediction of shape *(B, C, Tenc)* corresponding to the
        *next*     raw sample that starts *after* each encoded window.
        """

        c = self.encode(x)
        y_hat = self.project(c)  # (B, C, T_raw)
        return y_hat

    @torch.inference_mode()
    def forecast(self, past: torch.Tensor, horizon: int) -> torch.Tensor:
        """Autoregressive forecast using the trained 1‑step predictor.

        Args:     past: (B, C, Lp) observed context     horizon: number of future steps
        to generate (N)

        Returns:     (B, C, Lp+N) concatenation of past and generated samples
        """
        device = next(self.parameters()).device
        seq = past.to(device)
        B, C, Lp = seq.shape

        # Effective receptive window the model was trained with
        win = (self._encoded_len - 1) * self._ds_factor
        win = max(int(win), 1)

        generated: list[torch.Tensor] = []
        for _ in range(horizon):
            ctx = seq[..., -win:]
            y_next = self((ctx, None, None))[..., -1]  # (B,C)
            generated.append(y_next.unsqueeze(-1))
            seq = torch.cat([seq, y_next.unsqueeze(-1)], dim=-1)

        return seq
