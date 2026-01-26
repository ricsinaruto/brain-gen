import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal


class DiscretizedMixLogisticLoss(nn.Module):
    """Negative log-likelihood for Discretized Mixture of Logistics (DMoL).

    Modes ----- - 'independent': each channel is modeled with its own K-component
    mixture.     Expected output channels: channels * 3*K       (mix_logits, means,
    log_scales per channel)     Joint prob = ∏_c [ Σ_k π_{c,k} * P_bin(x_c | μ_{c,k},
    s_{c,k}) ]. - 'pixelcnnpp': shared K-component mixture over RGB with coupling.
    Expected output channels: 10*K laid out as:         [K mix_logits | 3K means | 3K
    log_scales | 3K coeffs]     Joint prob = Σ_k π_k ∏_{c∈{R,G,B}} P_bin(x_c | μ'_c,k,
    s_{c,k})     with μ'_G,k = μ_G,k + a_k * R,  μ'_B,k = μ_B,k + b_k * R + c_k * G.

    Inputs ------ pred : float tensor of shape     - independent: [B, channels*3K,
    *spatial]     - pixelcnnpp: [B, 10K, *spatial] target : int tensor of shape [B,
    channels, *spatial], with values in [0, bins-1]. mask : optional broadcastable mask
    over [B, 1 or channels, *spatial]   (1=keep, 0=ignore).

    Args ---- num_mixtures: K bins: number of quantization bins (e.g., 256 for 8-bit,
    65536 for 16-bit PCM) value_range: tuple (min, max) for the continuous scale you map
    bins to.              Common choices: (-1, 1) for images/audio, or (0, 1). channels:
    number of channels in the target (1 for audio/grayscale, 3 for RGB) mode:
    'independent' or 'pixelcnnpp' reduction: 'mean' | 'sum' | 'none' clamp_log_scale:
    (lo, hi) clamp for predicted log_scales to stabilize training tanh_coeffs: if True
    (default), apply tanh to PixelCNN++ coupling coeffs eps: numerical epsilon

    Utility ------- - use DiscretizedMixLogisticLoss.expected_out_channels(...)   to
    size your output head.
    """

    def __init__(
        self,
        num_mixtures: int,
        bins: int = 256,
        value_range: Tuple[float, float] = (-1.0, 1.0),
        channels: int = 1,
        mode: Literal["independent", "pixelcnnpp"] = "independent",
        reduction: Literal["mean", "sum", "none"] = "mean",
        clamp_log_scale: Tuple[float, float] = (-7.0, 7.0),
        tanh_coeffs: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        assert bins >= 2
        assert num_mixtures >= 1
        assert mode in ("independent", "pixelcnnpp")
        assert channels in (1, 3) if mode == "pixelcnnpp" else channels >= 1

        self.K = num_mixtures
        self.bins = bins
        self.vmin, self.vmax = float(value_range[0]), float(value_range[1])
        self.channels = channels
        self.mode = mode
        self.reduction = reduction
        self.clamp_log_scale = clamp_log_scale
        self.tanh_coeffs = tanh_coeffs
        self.eps = eps

        span = self.vmax - self.vmin
        # With integer bins 0..(bins-1),
        # the sample points sit Δ apart across [vmin, vmax].
        self.bin_size = span / float(bins - 1)
        self.half_bin = 0.5 * self.bin_size

    # --------- public API ---------

    @staticmethod
    def expected_out_channels(
        num_mixtures: int,
        channels: int = 1,
        mode: Literal["independent", "pixelcnnpp"] = "independent",
    ) -> int:
        if mode == "independent":
            return channels * (3 * num_mixtures)
        else:  # pixelcnnpp
            assert channels == 3, "pixelcnnpp mode expects RGB targets (channels=3)."
            return 10 * num_mixtures

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns a scalar (mean/sum) or per-position loss depending on `reduction`."""
        if self.mode == "independent":
            nll = self._nll_independent(pred, target)
        else:
            nll = self._nll_pixelcnnpp(pred, target)

        if mask is not None:
            nll = nll * mask.squeeze(1) if (mask.dim() == nll.dim() + 1) else nll * mask
        if self.reduction == "mean":
            denom = (
                mask.sum()
                if mask is not None
                else torch.tensor(nll.numel(), device=nll.device)
            )
            # If mask provided, denom is count of ones;
            # otherwise mean over all positions
            return nll.sum() / (denom.clamp_min(1.0))
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll  # shape [B, *spatial]

    # --------- core math ---------

    def _log_bin_mass(
        self,
        x_center: torch.Tensor,  # [..., K] broadcast OK
        q: torch.Tensor,  # [...] integer bins for lower/upper handling
        means: torch.Tensor,  # [..., K]
        log_scales: torch.Tensor,  # [..., K]
    ) -> (
        torch.Tensor
    ):  # [...], log probability per bin for each position (mixture not summed)
        lo, hi = self.clamp_log_scale
        inv_scales = torch.exp(-log_scales.clamp(min=lo, max=hi))  # 1/s

        # CDF at bin edges
        z_plus = (x_center + self.half_bin - means) * inv_scales
        z_minus = (x_center - self.half_bin - means) * inv_scales
        cdf_plus = torch.sigmoid(z_plus)
        cdf_minus = torch.sigmoid(z_minus)

        # Standard bin probability, with boundary handling
        is_lower = q == 0
        is_upper = q == (self.bins - 1)
        probs = torch.where(
            is_lower,
            cdf_plus,
            torch.where(is_upper, 1.0 - cdf_minus, cdf_plus - cdf_minus),
        )

        # Fallback for very small masses: use PDF(x) * Δ
        z_mid = (x_center - means) * inv_scales
        sigm_mid = torch.sigmoid(z_mid)
        pdf_mid = sigm_mid * (1.0 - sigm_mid) * inv_scales  # logistic pdf at x
        approx = pdf_mid * (self.bin_size)

        probs = torch.clamp(probs, min=self.eps)
        probs = torch.where(probs > 1e-5, probs, approx + self.eps)

        return torch.log(probs)

    def _nll_independent(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Pred: [B, channels*3K, *spatial] target: [B, channels, *spatial] integers
        0..bins-1 returns: [B, *spatial] position-wise NLL."""
        B, Cout, *sp = pred.shape
        C = self.channels
        K = self.K
        expected = C * (3 * K)
        if Cout != expected:
            raise ValueError(
                f"Expected pred with {expected} channels (got {Cout}). "
                f"Use DiscretizedMixLogisticLoss.expected_out_channels to size head."
            )
        if target.dtype.is_floating_point:
            # If the caller provided normalized floats, map to nearest bin.
            q = (
                torch.round((target - self.vmin) / self.bin_size)
                .clamp(0, self.bins - 1)
                .long()
            )
        else:
            q = target.long().clamp(0, self.bins - 1)

        # Reshape and split parameters
        pred = pred.view(B, C, 3 * K, *sp)
        mix_logits = pred[:, :, 0:K, ...]
        means = pred[:, :, K : 2 * K, ...]
        log_scales = pred[:, :, 2 * K : 3 * K, ...]
        log_mix = F.log_softmax(mix_logits, dim=2)  # over K

        # Map integer bins to continuous centers
        x_center = self.vmin + q.to(pred.dtype) * self.bin_size  # [B, C, *sp]
        x_center = x_center.unsqueeze(2)  # [B, C, 1, *sp] -> broadcast to K
        q_exp = q.unsqueeze(2)  # same expansion for boundary handling

        # log P(x_c | k) for each channel and mixture
        log_mass = self._log_bin_mass(
            x_center, q_exp, means, log_scales
        )  # [B, C, K, *sp]

        # log p(x_c) = logsumexp_k (log π_ck + log_mass_ck)
        log_p_c = torch.logsumexp(log_mix + log_mass, dim=2)  # [B, C, *sp]

        # Independent channels → sum logs across channels
        log_p = log_p_c.sum(dim=1)  # [B, *sp]
        return -log_p

    def _nll_pixelcnnpp(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """PixelCNN++ style shared mixture with RGB coupling.

        pred: [B, 10K, *spatial] laid out as:     [K mix_logits | 3K means | 3K
        log_scales | 3K coeffs] target: [B, 3, *spatial] with integer bins in [0,
        bins-1] returns: [B, *spatial] position-wise NLL
        """
        B, Cout, *sp = pred.shape
        K = self.K
        expected = 10 * K
        if Cout != expected:
            raise ValueError(
                f"Expected pred with {expected} channels (got {Cout}). "
                f"Use DiscretizedMixLogisticLoss.expected_out_channels to sizehead."
            )
        if self.channels != 3:
            raise ValueError("pixelcnnpp mode expects channels=3 (RGB).")
        if target.dtype.is_floating_point:
            q = (
                torch.round((target - self.vmin) / self.bin_size)
                .clamp(0, self.bins - 1)
                .long()
            )
        else:
            q = target.long().clamp(0, self.bins - 1)

        # Split parameters
        off = 0
        mix_logits = pred[:, off : off + K, ...]
        off += K
        means = pred[:, off : off + 3 * K, ...]
        off += 3 * K
        log_scales = pred[:, off : off + 3 * K, ...]
        off += 3 * K
        coeffs = pred[:, off : off + 3 * K, ...]
        off += 3 * K

        mix_logits = mix_logits.view(B, K, *sp)  # [B,K,*]
        means = means.view(B, 3, K, *sp)  # [B,3,K,*]
        log_scales = log_scales.view(B, 3, K, *sp)  # [B,3,K,*]
        coeffs = coeffs.view(B, 3, K, *sp)  # [B,3,K,*]
        if self.tanh_coeffs:
            coeffs = torch.tanh(coeffs)

        log_mix = F.log_softmax(mix_logits, dim=1)  # [B,K,*]

        # Map bins to centers
        q_r, q_g, q_b = q[:, 0], q[:, 1], q[:, 2]  # [B,*]
        x_r = self.vmin + q_r.to(pred.dtype) * self.bin_size
        x_g = self.vmin + q_g.to(pred.dtype) * self.bin_size
        x_b = self.vmin + q_b.to(pred.dtype) * self.bin_size

        # Expand to mixture dimension
        def _expK(x):
            return x.unsqueeze(1)  # [B,1,*] -> [B,1,*] (will broadcast vs K)

        x_rK, x_gK, x_bK = _expK(x_r), _expK(x_g), _expK(x_b)
        q_rK, q_gK, q_bK = _expK(q_r), _expK(q_g), _expK(q_b)

        # Per-component parameters
        mu_r, mu_g, mu_b = means[:, 0], means[:, 1], means[:, 2]  # [B,K,*]
        ls_r, ls_g, ls_b = log_scales[:, 0], log_scales[:, 1], log_scales[:, 2]

        # Coupled means: μ'_G = μ_G + a*R ; μ'_B = μ_B + b*R + c*G
        a, b, c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]  # [B,K,*]
        mu_g_p = mu_g + a * x_rK
        mu_b_p = mu_b + b * x_rK + c * x_gK

        # log mass per channel & component
        log_m_r = self._log_bin_mass(x_rK, q_rK, mu_r, ls_r)  # [B,K,*]
        log_m_g = self._log_bin_mass(x_gK, q_gK, mu_g_p, ls_g)
        log_m_b = self._log_bin_mass(x_bK, q_bK, mu_b_p, ls_b)

        # Joint within each component: sum channel log-masses
        log_p_k = log_m_r + log_m_g + log_m_b  # [B,K,*]

        # Mixture: logsumexp over K
        log_p = torch.logsumexp(log_mix + log_p_k, dim=1)  # [B,*]
        return -log_p
