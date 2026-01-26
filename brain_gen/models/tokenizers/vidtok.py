from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from brain_gen.layers.vidtok_layers import (
    EncoderCausal3DPadding,
    DecoderCausal3DPadding,
)
from brain_gen.layers.quantizers import FSQRegularizer, RVQRegularizer


class Vidtok(nn.Module):
    def __init__(
        self,
        encoder: Dict[str, Any],
        regularizer: Dict[str, Any],
    ):
        super().__init__()
        self.encoder = EncoderCausal3DPadding(**encoder)
        self.decoder = DecoderCausal3DPadding(**encoder)
        self.regularization = self._build_regularizer(encoder, regularizer)

    def _build_regularizer(
        self, encoder: Dict[str, Any], regularizer: Dict[str, Any]
    ) -> nn.Module:
        return FSQRegularizer(**regularizer)

    def encode(self, x: Any, return_reg_log: bool = False, global_step: int = 0) -> Any:
        z = self.encoder(x)
        z, reg_log = self.regularization(z, n_steps=global_step // 2)

        if return_reg_log:
            return z, reg_log
        return z

    def indices_to_latent(self, token_indices: torch.Tensor) -> torch.Tensor:
        assert token_indices.dim() == 4, "token_indices should be of shape (b, t, h, w)"
        b, t, h, w = token_indices.shape
        token_indices = token_indices.unsqueeze(-1).reshape(b, -1, 1)
        codes = self.regularization.indices_to_codes(token_indices)
        codes = codes.permute(0, 2, 3, 1).reshape(b, codes.shape[2], -1)
        z = self.regularization.project_out(codes)
        return z.reshape(b, t, h, w, -1).permute(0, 4, 1, 2, 3)

    def decode(self, z: Any, decode_from_indices: bool = False) -> torch.Tensor:
        if decode_from_indices:
            z = self.indices_to_latent(z)
        x = self.decoder(z)
        return x

    def get_input(self, batch: tuple) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        return batch

    def forward(
        self, x: Any, global_step: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.get_input(x)
        if self.encoder.fix_encoder:
            with torch.no_grad():
                z, reg_log = self.encode(
                    x, return_reg_log=True, global_step=global_step
                )
        else:
            z, reg_log = self.encode(x, return_reg_log=True, global_step=global_step)
        dec = self.decode(z)
        if dec.shape[2] != x.shape[2]:
            dec = dec[:, :, -x.shape[2] :, ...]
        return dec, reg_log


class VidtokRVQ(Vidtok):
    def _build_regularizer(
        self, encoder: Dict[str, Any], regularizer: Dict[str, Any]
    ) -> nn.Module:
        rvq_cfg = dict(regularizer)
        rvq_cfg.setdefault("dim", encoder["z_channels"])
        return RVQRegularizer(**rvq_cfg)

    def indices_to_latent(self, token_indices: torch.Tensor) -> torch.Tensor:
        if token_indices.dim() == 4:
            token_indices = token_indices.unsqueeze(-1)
        assert (
            token_indices.dim() == 5
        ), "token_indices should be of shape (b, t, h, w, q)"
        b, t, h, w, q = token_indices.shape
        flat_indices = token_indices.reshape(b, -1, q)
        quantized = None
        for level, layer in enumerate(self.regularization.rvq.layers):
            decoded = layer.decode(flat_indices[..., level])
            quantized = decoded if quantized is None else quantized + decoded
        quantized = quantized.reshape(b, t, h, w, -1).permute(0, 4, 1, 2, 3)
        return quantized
