import torch
import torch.nn as nn
from typing import Union
from einops import rearrange

from brain_gen.layers.brainomni.loss import (
    get_pcc,
    compute_l1_loss,
    get_frequency_domain_loss,
)


class VidtokLoss(nn.Module):
    def __init__(
        self,
        logvar_init: float = 0.0,
        scale_input_to_tgt_size: bool = False,
        learn_logvar: bool = False,
        regularization_weights: Union[None, dict] = None,
        temporal_pcc_weight: float = 0.0,
        temporal_freq_amp_weight: float = 0.0,
        temporal_freq_phase_weight: float = 0.0,
        temporal_cov_weight: float = 0.0,
    ):
        super().__init__()
        self.metrics = {"pcc": self.pcc, "l1": self.l1_loss}

        self.scale_input_to_tgt_size = scale_input_to_tgt_size

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.learn_logvar = learn_logvar

        self.regularization_weights = regularization_weights or {}

        # BrainOmni temporal losses (computed across time dimension)
        self.temporal_pcc_weight = temporal_pcc_weight
        self.temporal_freq_amp_weight = temporal_freq_amp_weight
        self.temporal_freq_phase_weight = temporal_freq_phase_weight
        self.temporal_cov_weight = temporal_cov_weight

    def get_input(self, inputs: torch.Tensor):
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]
        return inputs

    def pcc(self, rec: torch.Tensor, raw: torch.Tensor):
        rec = self.get_input(rec)
        if rec.ndim == 3:
            rec = rec.unsqueeze(-2)
            raw = raw.unsqueeze(-2)
        return get_pcc(rec, raw)

    def l1_loss(self, rec, raw):
        rec = self.get_input(rec)
        return compute_l1_loss(rec, raw)

    def _covariance_loss(self, rec: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        rec = rec.float()
        raw = raw.float()
        rec = rec - rec.mean(dim=-1, keepdim=True)
        raw = raw - raw.mean(dim=-1, keepdim=True)
        denom = rec.shape[-1] - 1
        rec_cov = rec @ rec.transpose(-1, -2) / denom
        raw_cov = raw @ raw.transpose(-1, -2) / denom
        return torch.mean(torch.abs(rec_cov - raw_cov))

    def forward(self, outputs, inputs, **kwargs):
        reconstructions, regularization_log = outputs

        if self.scale_input_to_tgt_size:
            inputs = torch.nn.functional.interpolate(
                inputs, reconstructions.shape[2:], mode="bicubic", antialias=True
            )

        # Compute BrainOmni temporal losses before rearranging
        # These losses operate on the time dimension (T)
        # with H, W as batch dimensions
        # Input shape: (B, C, T, H, W) -> (B*H*W, C, 1, T) for brainomni loss format
        temporal_pcc_loss = torch.tensor(0.0, device=inputs.device)
        temporal_freq_amp_loss = torch.tensor(0.0, device=inputs.device)
        temporal_freq_phase_loss = torch.tensor(0.0, device=inputs.device)
        temporal_cov_loss = torch.tensor(0.0, device=inputs.device)

        use_temporal_losses = (
            self.temporal_pcc_weight > 0
            or self.temporal_freq_amp_weight > 0
            or self.temporal_freq_phase_weight > 0
        )
        use_covariance_loss = self.temporal_cov_weight > 0
        if use_temporal_losses or use_covariance_loss:
            # Rearrange (B, C, T, H, W) -> (B*H*W, C, 1, T)
            inp_temporal = rearrange(inputs, "b c t h w -> b (h w) c t")
            rec_temporal = rearrange(reconstructions, "b c t h w -> b (h w) c t")

            # randomly pick 64 pixels from h*w
            num_pixels = 64
            random_indices = torch.randint(
                0, inp_temporal.shape[1], (num_pixels,), device=inputs.device
            )
            inp_temporal = inp_temporal[:, random_indices, ...]
            rec_temporal = rec_temporal[:, random_indices, ...]

            if use_temporal_losses:
                inp_temporal_loss = rearrange(inp_temporal, "b s c t -> (b s) c 1 t")
                rec_temporal_loss = rearrange(rec_temporal, "b s c t -> (b s) c 1 t")

                if self.temporal_pcc_weight > 0:
                    pcc = get_pcc(rec_temporal_loss, inp_temporal_loss)
                    temporal_pcc_loss = torch.exp(-pcc)

                if (
                    self.temporal_freq_amp_weight > 0
                    or self.temporal_freq_phase_weight > 0
                ):
                    amp_loss, phase_loss = get_frequency_domain_loss(
                        rec_temporal_loss, inp_temporal_loss
                    )
                    temporal_freq_amp_loss = amp_loss
                    temporal_freq_phase_loss = phase_loss

            if use_covariance_loss:
                inp_cov = inp_temporal.reshape(inp_temporal.shape[0], num_pixels, -1)
                rec_cov = rec_temporal.reshape(rec_temporal.shape[0], num_pixels, -1)
                temporal_cov_loss = self._covariance_loss(rec_cov, inp_cov)

        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.mean(nll_loss)

        loss = nll_loss

        # Add BrainOmni temporal losses
        loss = loss + self.temporal_pcc_weight * temporal_pcc_loss
        loss = loss + self.temporal_freq_amp_weight * temporal_freq_amp_loss
        loss = loss + self.temporal_freq_phase_weight * temporal_freq_phase_loss
        loss = loss + self.temporal_cov_weight * temporal_cov_loss

        for k in regularization_log:
            if k in self.regularization_weights:
                loss = loss + self.regularization_weights[k] * regularization_log[k]

        return loss
