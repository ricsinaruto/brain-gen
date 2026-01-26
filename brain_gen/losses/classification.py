import torch
from torch import nn

from ..utils.eval import accuracy, top_k_accuracy, f1_score, mse_loss
from typing import Any


class CrossEntropy(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        l1: float = 0.0,
        l2: float = 0.0,
        use_f1: bool = False,
        use_mse: bool = True,
        half_window: bool = False,
        spectral_weight: float = 0.0,
        spectral_freq_range: tuple[float, float] = (8.0, 12.0),
        spectral_n_fft: int = 256,
        spectral_eps: float = 1.0e-8,
        sfreq: float = 200.0,
        spectral_signal_clip: float | None = None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.l1 = l1
        self.l2 = l2
        self.half_window = half_window
        self.spectral_weight = spectral_weight
        self.spectral_freq_range = spectral_freq_range
        self.spectral_n_fft = spectral_n_fft
        self.spectral_eps = spectral_eps
        self.sfreq = sfreq
        self.spectral_signal_clip = spectral_signal_clip

        self.metrics = {"acc": accuracy}

        if use_f1:
            self.metrics["f1"] = f1_score

        if use_mse:
            self.metrics["mse"] = mse_loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Cross entropy supporting soft targets and regularisation."""
        mask = None

        if isinstance(targets, tuple) or isinstance(targets, list):
            targets = targets[0]

        if self.half_window:
            targets = targets[..., targets.shape[-1] // 2 :]
            logits = logits[..., logits.shape[-2] // 2 :, :]

        if targets.dim() == logits.dim():
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = (-targets * log_probs).sum(dim=-1)
            if reduction == "mean":
                loss = loss.mean()
            elif reduction == "sum":
                loss = loss.sum()
            elif reduction != "none":
                raise ValueError(f"Invalid reduction: {reduction}")
        else:
            target_shape = targets.shape
            if mask is not None:
                mask = mask.reshape(-1, 1).expand(-1, targets.size(-1)).reshape(-1)
                logits = logits.view(-1, logits.size(-1))[mask]
                targets = targets.view(-1)[mask]

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                weight=weight,
                label_smoothing=self.label_smoothing,
                reduction=reduction,
            )
            if reduction == "none" and mask is None:
                loss = loss.view(target_shape)

        if model is not None and self.l1 > 0 and reduction != "none":
            l1_pen = sum(p.abs().sum() for p in model.parameters())
            loss = loss + self.l1 * l1_pen
        return loss

    def _expected_signal(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to expected quantized signal values."""
        values = torch.arange(logits.size(-1), device=logits.device, dtype=logits.dtype)
        probs = torch.softmax(logits, dim=-1)
        expected = (probs * values).sum(dim=-1)
        if self.spectral_signal_clip is not None:
            expected = torch.clamp(expected, 0.0, float(self.spectral_signal_clip))
        return expected

    def _band_indices(self, n_freq: int) -> slice:
        freqs = torch.linspace(0, self.sfreq / 2, steps=n_freq, device="cpu")
        lo, hi = self.spectral_freq_range
        mask = (freqs >= lo) & (freqs <= hi)
        idx = mask.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return slice(0, n_freq)
        return slice(int(idx.min().item()), int(idx.max().item()) + 1)

    def _spectral_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.spectral_weight <= 0:
            return logits.new_tensor(0.0)

        # shape handling: assume logits [..., T, V]
        pred_signal = self._expected_signal(logits)
        tgt_signal = targets.float()

        # align trailing time dim
        time_dim = pred_signal.dim() - 1
        if tgt_signal.dim() == pred_signal.dim() - 1:
            tgt_signal = tgt_signal.unsqueeze(time_dim - 1)
        if tgt_signal.shape[time_dim] != pred_signal.shape[time_dim]:
            min_t = min(tgt_signal.shape[time_dim], pred_signal.shape[time_dim])
            pred_signal = pred_signal.narrow(time_dim, 0, min_t)
            tgt_signal = tgt_signal.narrow(time_dim, 0, min_t)

        def _psd(x: torch.Tensor) -> torch.Tensor:
            # flatten batch/channel dims
            x_flat = x.reshape(-1, x.shape[-1])
            spec = torch.fft.rfft(x_flat, n=self.spectral_n_fft, dim=-1)
            power = (spec.abs() ** 2).mean(dim=0)  # average over batch*channels
            return power

        pred_psd = _psd(pred_signal)
        tgt_psd = _psd(tgt_signal)
        band = self._band_indices(pred_psd.numel())
        pred_band = pred_psd[band]
        tgt_band = tgt_psd[band]

        pred_log = torch.log(pred_band + self.spectral_eps)
        tgt_log = torch.log(tgt_band + self.spectral_eps)
        return torch.nn.functional.l1_loss(pred_log, tgt_log)


class CrossEntropySpectral(CrossEntropy):
    """CrossEntropy with auxiliary spectral matching in a target band."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        ce = super().forward(logits, targets, model, weight, reduction=reduction)
        if reduction == "none":
            return ce
        spec = self._spectral_loss(logits, targets)
        return ce + self.spectral_weight * spec


class CrossEntropyMasked(CrossEntropy):
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        targets, mask = targets

        new_logits = []
        new_targets = []
        for i in range(logits.shape[0]):
            new_logits.append(logits[i, mask[i]])
            new_targets.append(targets[i, mask[i]])

        logits = torch.cat(new_logits, dim=0)
        targets = torch.cat(new_targets, dim=0)
        return super().forward(logits, targets, model, reduction=reduction)


class CrossEntropyWeighted(CrossEntropy):
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        weight = None
        if isinstance(targets, tuple) or isinstance(targets, list):
            targets, weight = targets[0], targets[1][0]

        return super().forward(logits, targets, model, weight, reduction=reduction)


class CrossEntropyBalanced(CrossEntropy):
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if isinstance(targets, tuple) or isinstance(targets, list):
            targets, weight = targets[0], targets[1][0]
            logits = logits + weight
        return super().forward(logits, targets, model, reduction=reduction)


class CrossEntropyWithCodes(CrossEntropy):
    """Cross entropy loss with codes returned by model instead of dataloader."""

    def __init__(self, label_smoothing: float = 0.0, l1: float = 0.0, l2: float = 0.0):
        super().__init__(label_smoothing, l1, l2)
        # self.metrics = {
        #    "acc": lambda logits, targets: accuracy(logits[0], logits[1]),
        #    "top5": lambda logits, targets: top_k_accuracy(logits[0], logits[1]),
        # }
        self.metrics = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> torch.Tensor:
        return super().forward(
            logits[0],
            logits[1],
            model,
            reduction=reduction,
            **kwargs,
        )
