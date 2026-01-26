import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Any, List, Optional, Tuple


class VQVAEHF(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics: dict[str, Any] = {
            "recon": self._recon_loss,
            "vq": self._vq_loss,
            "ppl": self._perplexity,
        }

    def forward(self, outputs, target: Tensor, **kwargs) -> Tensor:
        return outputs.loss

    @staticmethod
    def _recon_loss(outputs: Tensor, target: Tensor, **kwargs) -> Tensor:
        return outputs.recon_loss

    @staticmethod
    def _vq_loss(outputs: Tensor, target: Tensor, **kwargs) -> Tensor:
        return outputs.vq_loss

    @staticmethod
    def _perplexity(outputs: Tensor, target: Tensor, **kwargs) -> Tensor:
        return outputs.perplexity


class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        # keep consistent interface with CrossEntropy
        self.metrics: dict[str, Any] = {}

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        **kwargs,
    ) -> Tensor:
        """Mean-squared-error loss with a flexible signature.

        Accepts arbitrary additional keyword arguments (ignored) so that it can be
        called in the same way as :class:`CrossEntropy` within the training loop, which
        always forwards the current *model* instance.
        """
        return F.mse_loss(logits, targets).mean()


class NLL(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics: dict[str, Any] = {"nll": self._nll, "logdet": self._logdet}

    @staticmethod
    def _nll(losses: Tensor, target: Tensor, **kwargs) -> Tensor:
        return losses[0]

    @staticmethod
    def _logdet(losses: Tensor, target: Tensor, **kwargs) -> Tensor:
        return losses[1]

    def forward(self, losses: Tensor, target: Tensor, **kwargs) -> Tensor:
        nll, logdet = losses
        return nll + logdet


class ChronoFlowLoss(nn.Module):
    """Wrapper around ChronoFlowSSM outputs for Lightning training.

    Expects the model forward pass to return a dictionary with at least the key
    ``"nll"`` and optionally a ``"stats"`` sub-dictionary containing ``"bits_per_dim"``
    and ``"avg_boundary"`` tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.metrics: dict[str, Any] = {
            "bits_per_dim": self._bits_per_dim,
            "avg_boundary": self._avg_boundary,
        }

    def forward(
        self,
        outputs: dict[str, Any],
        targets: Tensor | tuple | None,
        **kwargs,
    ) -> Tensor:
        if not isinstance(outputs, dict) or "nll" not in outputs:
            raise ValueError(
                "ChronoFlowLoss expects the model to return a dict with an 'nll' key."
            )
        return outputs["nll"]

    @staticmethod
    def _bits_per_dim(outputs: dict[str, Any], *_) -> Tensor:
        stats = outputs.get("stats", {})
        val = stats.get("bits_per_dim")
        if val is None:
            return torch.tensor(float("nan"), device=outputs["nll"].device)
        return (
            val.detach()
            if isinstance(val, torch.Tensor)
            else torch.as_tensor(val, device=outputs["nll"].device)
        )

    @staticmethod
    def _avg_boundary(outputs: dict[str, Any], *_) -> Tensor:
        stats = outputs.get("stats", {})
        val = stats.get("avg_boundary")
        if val is None:
            return torch.tensor(float("nan"), device=outputs["nll"].device)
        return (
            val.detach()
            if isinstance(val, torch.Tensor)
            else torch.as_tensor(val, device=outputs["nll"].device)
        )


class VQVAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics: dict[str, Any] = {
            "codebook_perplexity": lambda outputs, *_: self._stat(
                outputs, "codebook_perplexity"
            ),
            "unused_codes": lambda outputs, *_: self._stat(outputs, "unused_codes"),
            "active_codes": lambda outputs, *_: self._stat(outputs, "active_codes"),
            "revived_codes": lambda outputs, *_: self._stat(outputs, "revived_codes"),
        }

    def forward(self, outputs: Tensor, target: Tensor, **kwargs) -> Tensor:
        x_recon, vq_output = outputs

        if isinstance(vq_output, dict) and "total_loss" in vq_output:
            loss_val = vq_output["total_loss"]
            if not torch.is_tensor(loss_val):
                loss_val = torch.as_tensor(loss_val, device=x_recon.device)
            return loss_val

        recon_loss = F.mse_loss(x_recon, target)
        loss = recon_loss + vq_output["commitment_loss"]
        return loss * 2

    @staticmethod
    def _stat(outputs: Tensor | tuple, key: str) -> Tensor:
        # metrics are derived from the auxiliary dict returned by the model
        if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
            rec = outputs[0]
            vq_output = outputs[1]
            device = rec.device if torch.is_tensor(rec) else torch.device("cpu")
            if isinstance(vq_output, dict):
                val = vq_output.get(key)
                if val is None:
                    return torch.tensor(float("nan"), device=device)
                return (
                    val.detach()
                    if torch.is_tensor(val)
                    else torch.as_tensor(val, device=device)
                )
        # fallback if shape unexpected
        device = outputs.device if torch.is_tensor(outputs) else torch.device("cpu")
        return torch.tensor(float("nan"), device=device)


class VQNSPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics: dict[str, Any] = {}

    def calculate_rec_loss(self, rec: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(rec, target)

    def forward(
        self,
        inputs: tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
        ],
        targets: Tensor,
        **kwargs,
    ) -> Tensor:
        xrec, xrec_angle, amplitude, angle, emb_loss = inputs

        rec_loss = self.calculate_rec_loss(xrec, amplitude)
        rec_angle_loss = self.calculate_rec_loss(xrec_angle, angle)

        loss = emb_loss + rec_loss + rec_angle_loss

        return loss


class SpectralLoss(nn.Module):
    """Time-domain MSE + log-PSD band loss.

    Uses STFT with Hann window; compares mean log power per band.
    """

    def __init__(
        self,
        sample_rate: float,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        eps: float = 1e-8,
        band_edges: Optional[List[Tuple[float, float]]] = None,
        band_weight: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = hop_length or (self.n_fft // 4)
        self.eps = eps
        self.band_edges = band_edges or [
            (1, 4),
            (4, 8),
            (8, 13),
            (13, 30),
            (30, 50),
        ]  # delta..gamma up to 50hz
        self.band_weight = band_weight

        self.metrics: dict[str, Any] = {}

    def _power_spectrogram(self, x: torch.Tensor):
        """X: (B, T, C) returns: psd (B*C, F), freqs (F,)"""
        B, C, T = x.shape
        xc = x.reshape(B * C, T)  # (B*C, T)
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        X = torch.stft(
            xc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )  # (B*C, F, frames)
        power = (X.real**2 + X.imag**2).mean(dim=-1)  # average over frames -> (B*C, F)
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate).to(x.device)
        return power, freqs

    @staticmethod
    def _bandpower(psd: torch.Tensor, freqs: torch.Tensor, f_lo: float, f_hi: float):
        idx = (freqs >= f_lo) & (freqs < f_hi)
        if idx.sum().item() == 0:
            return psd.mean(dim=-1)  # fallback
        return psd[:, idx].mean(dim=-1)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, model: nn.Module | None = None
    ) -> torch.Tensor:
        # Time-domain MSE
        td = F.mse_loss(pred, target)

        # Spectral band loss on log-PSD
        psd_p, freqs = self._power_spectrogram(pred)
        psd_t, _ = self._power_spectrogram(target)
        lp = torch.log(psd_p + self.eps)
        lt = torch.log(psd_t + self.eps)

        band_losses = []
        for flo, fhi in self.band_edges:
            bp = self._bandpower(lp, freqs, flo, fhi)
            bt = self._bandpower(lt, freqs, flo, fhi)
            band_losses.append(F.mse_loss(bp, bt))
        spec = sum(band_losses) / max(1, len(band_losses))

        return td + self.band_weight * spec
