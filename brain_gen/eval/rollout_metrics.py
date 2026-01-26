from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)

_JSD_MAX = np.sqrt(np.log(2.0))


def _default_bands() -> dict[str, tuple[float, float]]:
    """Return the default frequency bands in Hz."""
    return {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }


def resolve_frequency_bands(
    bands_cfg: dict[str, Any] | None,
) -> dict[str, tuple[float, float]]:
    """Normalize band definitions from config into a {name: (lo, hi)} mapping."""
    if bands_cfg is None:
        return _default_bands()

    bands: dict[str, tuple[float, float]] = {}
    for name, bounds in bands_cfg.items():
        if bounds is None or len(bounds) < 2:
            continue
        bands[str(name)] = (float(bounds[0]), float(bounds[1]))
    return bands


def _resolve_log_scales(min_scale: int, max_scale: int, n_scales: int) -> np.ndarray:
    """Return integer log-spaced scales within [min_scale, max_scale]."""
    if n_scales < 2 or max_scale <= min_scale:
        return np.array([], dtype=np.int64)
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), int(n_scales))
    scales = np.unique(np.round(scales).astype(np.int64))
    return scales[(scales >= min_scale) & (scales <= max_scale)]


def _resolve_scale_bound(
    cfg: dict[str, Any],
    steps_key: str,
    seconds_key: str,
    fs_val: float | None,
    fallback: int,
) -> int:
    """Resolve an integer scale bound from steps/seconds config."""
    val = cfg.get(steps_key)
    if val is not None:
        return max(1, int(val))
    sec_val = cfg.get(seconds_key)
    if sec_val is not None and fs_val is not None:
        return max(1, int(float(sec_val) * fs_val))
    return max(1, int(fallback))


def _safe_nanmean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    """Compute nanmean without warnings when all values are NaN."""
    mask = np.isfinite(values)
    count = np.sum(mask, axis=axis)
    summed = np.sum(np.where(mask, values, 0.0), axis=axis)
    denom = np.where(count > 0, count, np.nan)
    return summed / denom


@dataclass(frozen=True)
class RolloutMetricsConfig:
    """Configuration for rollout metric computations."""

    fs: float | None = None
    fmin: float = 0.5
    fmax: float | None = None
    welch_nperseg_s: float = 1.0
    welch_noverlap_frac: float = 0.5
    stft_nperseg_s: float = 1.0
    stft_noverlap_frac: float = 0.75
    tail_sigma: float = 3.0
    dfa_scales: int = 8
    dfa_min_scale_s: float | None = 0.1
    dfa_max_scale_s: float | None = 2
    dfa_min_scale: int | None = None
    dfa_max_scale: int | None = None
    hurst_lags: int = 8
    hurst_min_lag_s: float | None = 0.1
    hurst_max_lag_s: float | None = 2
    hurst_min_lag: int | None = None
    hurst_max_lag: int | None = None
    bands: dict[str, tuple[float, float]] = field(default_factory=_default_bands)

    @classmethod
    def from_params(
        cls, params: dict[str, Any] | None, sfreq: float | None
    ) -> "RolloutMetricsConfig":
        """Build a config from evaluation params and a fallback sampling rate."""
        defaults = cls()
        ts_cfg = params.get("timeseries_divergence", {}) if params else {}
        bands_cfg = None
        if params:
            bands_cfg = params.get("fft_bands")
            if bands_cfg is None:
                bands_cfg = params.get("coherence_bands")
            if bands_cfg is None and isinstance(ts_cfg, dict):
                bands_cfg = ts_cfg.get("bands")

        fs_val = ts_cfg.get("fs", sfreq)
        fs = float(fs_val) if fs_val is not None else None
        fmin_val = ts_cfg.get("fmin", defaults.fmin)
        fmin = float(fmin_val) if fmin_val is not None else defaults.fmin
        fmax_val = ts_cfg.get("fmax", defaults.fmax)
        fmax = float(fmax_val) if fmax_val is not None else None
        return cls(
            fs=fs,
            fmin=fmin,
            fmax=fmax,
            welch_nperseg_s=float(
                ts_cfg.get("welch_nperseg_s", defaults.welch_nperseg_s)
            ),
            welch_noverlap_frac=float(
                ts_cfg.get("welch_noverlap_frac", defaults.welch_noverlap_frac)
            ),
            stft_nperseg_s=float(ts_cfg.get("stft_nperseg_s", defaults.stft_nperseg_s)),
            stft_noverlap_frac=float(
                ts_cfg.get("stft_noverlap_frac", defaults.stft_noverlap_frac)
            ),
            tail_sigma=float(ts_cfg.get("tail_sigma", defaults.tail_sigma)),
            dfa_scales=int(ts_cfg.get("dfa_scales", defaults.dfa_scales)),
            dfa_min_scale_s=ts_cfg.get("dfa_min_scale_s", defaults.dfa_min_scale_s),
            dfa_max_scale_s=ts_cfg.get("dfa_max_scale_s", defaults.dfa_max_scale_s),
            dfa_min_scale=ts_cfg.get("dfa_min_scale", defaults.dfa_min_scale),
            dfa_max_scale=ts_cfg.get("dfa_max_scale", defaults.dfa_max_scale),
            hurst_lags=int(ts_cfg.get("hurst_lags", defaults.hurst_lags)),
            hurst_min_lag_s=ts_cfg.get("hurst_min_lag_s", defaults.hurst_min_lag_s),
            hurst_max_lag_s=ts_cfg.get("hurst_max_lag_s", defaults.hurst_max_lag_s),
            hurst_min_lag=ts_cfg.get("hurst_min_lag", defaults.hurst_min_lag),
            hurst_max_lag=ts_cfg.get("hurst_max_lag", defaults.hurst_max_lag),
            bands=resolve_frequency_bands(bands_cfg),
        )

    def resolve_dfa_scales(self, window_len: int) -> np.ndarray:
        """Compute DFA scales for a given window length."""
        cfg = {
            "dfa_min_scale": self.dfa_min_scale,
            "dfa_min_scale_s": self.dfa_min_scale_s,
            "dfa_max_scale": self.dfa_max_scale,
            "dfa_max_scale_s": self.dfa_max_scale_s,
        }
        dfa_min = _resolve_scale_bound(
            cfg,
            "dfa_min_scale",
            "dfa_min_scale_s",
            self.fs,
            max(4, int(window_len * 0.05)),
        )
        dfa_max_default = max(dfa_min + 1, int(window_len * 0.5))
        dfa_max = _resolve_scale_bound(
            cfg,
            "dfa_max_scale",
            "dfa_max_scale_s",
            self.fs,
            dfa_max_default,
        )
        dfa_max = min(dfa_max, max(2, window_len // 2))
        return _resolve_log_scales(dfa_min, dfa_max, self.dfa_scales)

    def resolve_hurst_lags(self, window_len: int) -> np.ndarray:
        """Compute Hurst lags for a given window length."""
        cfg = {
            "hurst_min_lag": self.hurst_min_lag,
            "hurst_min_lag_s": self.hurst_min_lag_s,
            "hurst_max_lag": self.hurst_max_lag,
            "hurst_max_lag_s": self.hurst_max_lag_s,
        }
        hurst_min = _resolve_scale_bound(
            cfg,
            "hurst_min_lag",
            "hurst_min_lag_s",
            self.fs,
            max(2, int(window_len * 0.02)),
        )
        hurst_max_default = max(hurst_min + 1, int(window_len * 0.2))
        hurst_max = _resolve_scale_bound(
            cfg,
            "hurst_max_lag",
            "hurst_max_lag_s",
            self.fs,
            hurst_max_default,
        )
        hurst_max = min(hurst_max, max(2, window_len // 2))
        return _resolve_log_scales(hurst_min, hurst_max, self.hurst_lags)


class RolloutMetrics:
    """Compute per-example, per-channel metrics for rollout divergence."""

    def __init__(
        self,
        cfg: RolloutMetricsConfig,
        spatial_weights: np.ndarray | None = None,
    ) -> None:
        """Initialize metric computation for rollout windows."""
        self.cfg = cfg
        self.spatial_weights = spatial_weights

    def compute(self, data: np.ndarray) -> dict[str, Any]:
        """Compute all metrics for a batch shaped (B, C, T)."""
        if data.ndim != 3:
            raise ValueError("Metrics expect data shaped as (B, C, T).")
        data = np.asarray(data, dtype=np.float32)
        batch, channels, steps = data.shape
        metrics: dict[str, Any] = {"timeseries": data, "bands": self.cfg.bands}

        if steps < 2:
            cov = np.full((batch, channels, channels), np.nan, dtype=np.float32)
        else:
            centered = data - np.mean(data, axis=2, keepdims=True)
            denom = max(1, steps - 1)
            cov = np.einsum("bct,bdt->bcd", centered, centered) / denom
        metrics["covariance"] = cov

        corr = self._corr_from_cov(cov)
        metrics["spatial_connectivity"] = self._spatial_connectivity(corr)

        metrics["amplitude_kurtosis"] = self._excess_kurtosis(data)
        metrics["amplitude_tail_fraction"] = self._tail_fraction(data)

        metrics["fft_magnitude"], metrics["fft_angle"] = self._fft_metrics(data)

        psd_freqs, psd = self._psd_metrics(data)
        metrics["psd_freqs"] = psd_freqs
        metrics["psd"] = psd
        metrics["bandpower_ratios"] = self._bandpower_ratios(psd_freqs, psd)

        stft_mag, stft_angle = self._stft_metrics(data)
        metrics["stft_magnitude"] = stft_mag
        metrics["stft_angle"] = stft_angle

        metrics["dfa_exponent"] = self._dfa_exponent(data)
        metrics["hurst_exponent"] = self._hurst_exponent(data)
        metrics["one_over_f_exponent"] = self._one_over_f_exponent(psd_freqs, psd)

        return metrics

    def compute_summaries(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """Compute per-example summary metrics for a batch shaped (B, C, T)."""
        metrics = self.compute(data)
        summaries: dict[str, np.ndarray] = {}

        summaries["spatial_connectivity"] = metrics["spatial_connectivity"]
        summaries["one_over_f_exponent"] = self._summarize_channels(
            metrics["one_over_f_exponent"]
        )
        summaries["dfa_exponent"] = self._summarize_channels(metrics["dfa_exponent"])
        summaries["hurst_exponent"] = self._summarize_channels(
            metrics["hurst_exponent"]
        )
        summaries["amplitude_kurtosis"] = self._summarize_channels(
            metrics["amplitude_kurtosis"]
        )
        summaries["amplitude_tail_fraction"] = self._summarize_channels(
            metrics["amplitude_tail_fraction"]
        )

        ratios = metrics["bandpower_ratios"]
        bands = list(self.cfg.bands.keys())
        if ratios.size == 0 or not bands:
            for name in bands:
                summaries[f"bandpower_ratio_{name}"] = np.full(
                    metrics["spatial_connectivity"].shape[0], np.nan, dtype=np.float32
                )
        else:
            for idx, name in enumerate(bands):
                summaries[f"bandpower_ratio_{name}"] = self._summarize_channels(
                    ratios[:, :, idx]
                )

        cov_entropy, cov_top_frac = self._covariance_eigensummary(metrics["covariance"])
        summaries["cov_eig_entropy"] = cov_entropy
        summaries["cov_eig_top_frac"] = cov_top_frac

        psd_entropy, psd_centroid = self._psd_summary_stats(
            metrics["psd_freqs"], metrics["psd"]
        )
        summaries["psd_entropy"] = psd_entropy
        summaries["psd_centroid"] = psd_centroid

        summaries.update(self._token_summary_stats(data))

        return summaries

    def _summarize_channels(self, values: np.ndarray) -> np.ndarray:
        """Summarize per-channel values into a per-example statistic."""
        if values.size == 0:
            return np.full(values.shape[0], np.nan, dtype=np.float32)
        return _safe_nanmean(values, axis=1).astype(np.float32)

    def _covariance_eigensummary(
        self, cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return entropy and top-eigenvalue fraction of covariance spectra."""
        if cov.size == 0:
            nan = np.full(cov.shape[0], np.nan, dtype=np.float32)
            return nan, nan
        eigvals = np.linalg.eigvalsh(cov).astype(np.float32)
        eigvals = np.clip(eigvals, 0.0, None)
        total = np.sum(eigvals, axis=1, keepdims=True)
        total = np.clip(total, 1e-12, None)
        probs = eigvals / total
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        if cov.shape[1] > 1:
            entropy = entropy / np.log(float(cov.shape[1]))
        top_frac = np.max(probs, axis=1)
        return entropy.astype(np.float32), top_frac.astype(np.float32)

    def _psd_summary_stats(
        self, freqs: np.ndarray, psd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute PSD entropy and centroid per example."""
        if freqs.size == 0 or psd.size == 0:
            nan = np.full(psd.shape[0], np.nan, dtype=np.float32)
            return nan, nan
        psd_mean = _safe_nanmean(psd, axis=1)
        psd_mean = np.clip(psd_mean, 0.0, None)
        total = np.sum(psd_mean, axis=1, keepdims=True)
        total = np.clip(total, 1e-12, None)
        probs = psd_mean / total
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        if freqs.size > 1:
            entropy = entropy / np.log(float(freqs.size))
        centroid = np.sum(probs * freqs[None, :], axis=1)
        return entropy.astype(np.float32), centroid.astype(np.float32)

    def _token_summary_stats(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """Compute token-level summary stats when data are integer tokens."""
        tokens = self._maybe_integer_tokens(data)
        if tokens is None:
            return {
                "token_entropy": np.full(data.shape[0], np.nan, dtype=np.float32),
                "token_unique_rate": np.full(data.shape[0], np.nan, dtype=np.float32),
                "token_repetition_rate": np.full(
                    data.shape[0], np.nan, dtype=np.float32
                ),
            }
        entropy = self._token_entropy_per_step(tokens)
        unique_rate, repetition_rate = self._token_repeat_rates(tokens)
        return {
            "token_entropy": entropy,
            "token_unique_rate": unique_rate,
            "token_repetition_rate": repetition_rate,
        }

    def _maybe_integer_tokens(self, data: np.ndarray) -> np.ndarray | None:
        """Return integer tokens if the data appear tokenized."""
        if data.size == 0:
            return None
        if np.issubdtype(data.dtype, np.integer):
            return data.astype(np.int64, copy=False)
        if not np.isfinite(data).all():
            return None
        rounded = np.rint(data)
        if np.max(np.abs(data - rounded)) > 1e-3:
            return None
        return rounded.astype(np.int64)

    def _token_entropy_per_step(self, tokens: np.ndarray) -> np.ndarray:
        """Compute per-step entropy of token distributions."""
        batch, _, steps = tokens.shape
        channels = tokens.shape[1]
        if steps == 0 or channels == 0 or batch == 0:
            return np.full(batch, np.nan, dtype=np.float32)

        tokens_bt = tokens.transpose(0, 2, 1).reshape(-1)
        bt_index = np.repeat(np.arange(batch * steps, dtype=np.int64), channels)
        pairs = np.empty(tokens_bt.size, dtype=[("bt", np.int64), ("val", np.int64)])
        pairs["bt"] = bt_index
        pairs["val"] = tokens_bt
        unique_pairs, counts = np.unique(pairs, return_counts=True)
        if counts.size == 0:
            return np.full(batch, np.nan, dtype=np.float32)
        bt_vals = unique_pairs["bt"]
        group_start = np.r_[0, np.flatnonzero(np.diff(bt_vals)) + 1]
        counts_f = counts.astype(np.float64)
        sum_c_log2_c = np.add.reduceat(counts_f * np.log2(counts_f), group_start)
        entropy_per_group = np.log2(float(channels)) - (sum_c_log2_c / float(channels))
        entropy_per_bt = np.full(batch * steps, np.nan, dtype=np.float32)
        entropy_per_bt[bt_vals[group_start]] = entropy_per_group.astype(np.float32)
        entropies = entropy_per_bt.reshape(batch, steps)
        return _safe_nanmean(entropies, axis=1).astype(np.float32)

    def _token_repeat_rates(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute unique and repetition rates for token sequences."""
        batch = tokens.shape[0]
        unique_rate = np.full(batch, np.nan, dtype=np.float32)
        repetition_rate = np.full(batch, np.nan, dtype=np.float32)
        for b in range(batch):
            flat = tokens[b].ravel()
            if flat.size == 0:
                continue
            unique = np.unique(flat).size
            rate = unique / float(flat.size)
            unique_rate[b] = rate
            repetition_rate[b] = 1.0 - rate
        return unique_rate, repetition_rate

    def _corr_from_cov(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance matrices to correlation matrices."""
        if cov.size == 0:
            return cov
        std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
        denom = std[:, :, None] * std[:, None, :]
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.arange(corr.shape[1])
        corr[:, idx, idx] = 1.0
        return corr

    def _spatial_connectivity(self, corr: np.ndarray) -> np.ndarray:
        """Compute mean absolute off-diagonal correlation per example."""
        if corr.size == 0 or corr.shape[1] < 2:
            return np.full(corr.shape[0], np.nan, dtype=np.float32)
        mask = ~np.eye(corr.shape[1], dtype=bool)
        values = np.abs(corr[:, mask])
        if values.size == 0:
            return np.full(corr.shape[0], np.nan, dtype=np.float32)
        if (
            self.spatial_weights is not None
            and self.spatial_weights.shape == corr.shape[1:]
        ):
            w = self.spatial_weights[mask]
            w_sum = float(np.sum(w))
            if np.isfinite(w_sum) and w_sum > 0:
                return np.sum(values * w, axis=1) / w_sum
        return np.mean(values, axis=1)

    def _excess_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Compute per-channel excess kurtosis for each example."""
        if data.size == 0:
            return np.full(data.shape[:2], np.nan, dtype=np.float32)
        mean = np.mean(data, axis=2, keepdims=True)
        centered = data - mean
        var = np.mean(centered**2, axis=2)
        m4 = np.mean(centered**4, axis=2)
        kurt = np.full_like(var, np.nan)
        valid = var > 1e-12
        kurt[valid] = m4[valid] / (var[valid] ** 2) - 3.0
        return kurt

    def _tail_fraction(self, data: np.ndarray) -> np.ndarray:
        """Compute fraction of samples beyond tail_sigma * std per channel."""
        if data.size == 0:
            return np.full(data.shape[:2], np.nan, dtype=np.float32)
        std = np.std(data, axis=2)
        thresh = self.cfg.tail_sigma * std
        fraction = np.mean(np.abs(data) > thresh[..., None], axis=2)
        fraction = np.where(std > 1e-12, fraction, np.nan)
        return fraction

    def _fft_metrics(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return FFT magnitude and angle for each example and channel."""
        steps = data.shape[2]
        if steps < 2:
            shape = (data.shape[0], data.shape[1], 0)
            return np.full(shape, np.nan, dtype=np.float32), np.full(
                shape, np.nan, dtype=np.float32
            )
        norm = float(max(1, steps))
        fft = np.fft.rfft(data, axis=2) / norm
        return np.abs(fft).astype(np.float32), np.angle(fft).astype(np.float32)

    def _psd_metrics(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute Welch PSD for each example/channel and return (freqs, psd)."""
        fs = self.cfg.fs
        steps = data.shape[2]
        if fs is None or fs <= 0 or steps < 2:
            return np.array([], dtype=np.float32), np.full(
                (data.shape[0], data.shape[1], 0), np.nan, dtype=np.float32
            )
        nperseg = int(round(self.cfg.welch_nperseg_s * fs))
        nperseg = max(8, nperseg)
        if nperseg < 2 or nperseg > steps:
            return np.array([], dtype=np.float32), np.full(
                (data.shape[0], data.shape[1], 0), np.nan, dtype=np.float32
            )
        noverlap = int(round(self.cfg.welch_noverlap_frac * nperseg))
        noverlap = min(noverlap, max(0, nperseg - 1))
        flat = data.reshape(data.shape[0] * data.shape[1], steps)
        freqs, psd = signal.welch(
            flat,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=1,
        )
        fmax = (
            float(self.cfg.fmax)
            if self.cfg.fmax is not None
            else 0.95 * (float(fs) / 2.0)
        )
        mask = (freqs >= self.cfg.fmin) & (freqs <= fmax)
        if not np.any(mask):
            return np.array([], dtype=np.float32), np.full(
                (data.shape[0], data.shape[1], 0), np.nan, dtype=np.float32
            )
        freqs = freqs[mask].astype(np.float32)
        psd = psd[:, mask].reshape(data.shape[0], data.shape[1], -1).astype(np.float32)
        return freqs, psd

    def _bandpower_ratios(self, freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
        """Compute per-band power ratios for each example/channel."""
        bands = list(self.cfg.bands.items())
        if not bands:
            return np.full((psd.shape[0], psd.shape[1], 0), np.nan, dtype=np.float32)
        if freqs.size == 0 or psd.size == 0:
            return np.full(
                (psd.shape[0], psd.shape[1], len(bands)), np.nan, dtype=np.float32
            )
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        total = trapz(psd, freqs, axis=2)
        total = np.clip(total, 1e-12, None)
        ratios = np.full(
            (psd.shape[0], psd.shape[1], len(bands)), np.nan, dtype=np.float32
        )
        for idx, (_, (lo, hi)) in enumerate(bands):
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                continue
            band_power = trapz(psd[:, :, mask], freqs[mask], axis=2)
            ratios[:, :, idx] = band_power / total
        return ratios

    def _stft_metrics(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return STFT magnitude and angle for each example/channel."""
        fs = self.cfg.fs
        steps = data.shape[2]
        if fs is None or fs <= 0 or steps < 2:
            shape = (data.shape[0], data.shape[1], 0, 0)
            return np.full(shape, np.nan, dtype=np.float32), np.full(
                shape, np.nan, dtype=np.float32
            )
        nperseg = int(round(self.cfg.stft_nperseg_s * fs))
        nperseg = max(8, nperseg)
        if nperseg < 2 or nperseg > steps:
            shape = (data.shape[0], data.shape[1], 0, 0)
            return np.full(shape, np.nan, dtype=np.float32), np.full(
                shape, np.nan, dtype=np.float32
            )
        noverlap = int(round(self.cfg.stft_noverlap_frac * nperseg))
        noverlap = min(noverlap, max(0, nperseg - 1))
        flat = data.reshape(data.shape[0] * data.shape[1], steps)
        freqs, _, spec = signal.stft(
            flat,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=-1,
            boundary=None,
            padded=False,
        )
        fmax = (
            float(self.cfg.fmax)
            if self.cfg.fmax is not None
            else 0.95 * (float(fs) / 2.0)
        )
        mask = (freqs >= self.cfg.fmin) & (freqs <= fmax)
        if not np.any(mask):
            shape = (data.shape[0], data.shape[1], 0, 0)
            return np.full(shape, np.nan, dtype=np.float32), np.full(
                shape, np.nan, dtype=np.float32
            )
        spec = spec[:, mask, :]
        spec = spec.reshape(data.shape[0], data.shape[1], spec.shape[1], spec.shape[2])
        return np.abs(spec).astype(np.float32), np.angle(spec).astype(np.float32)

    def _dfa_exponent(self, data: np.ndarray) -> np.ndarray:
        """Estimate DFA exponent per channel."""
        scales = self.cfg.resolve_dfa_scales(data.shape[2])
        if scales.size < 2:
            return np.full(data.shape[:2], np.nan, dtype=np.float32)
        steps = data.shape[2]
        if steps < 8:
            return np.full(data.shape[:2], np.nan, dtype=np.float32)
        series = data.reshape(data.shape[0] * data.shape[1], steps).astype(
            np.float64, copy=False
        )
        series = series - np.mean(series, axis=1, keepdims=True)
        std = np.std(series, axis=1, keepdims=True)
        scale = np.where(std > 1e-6, std, 1.0)
        series = series / scale
        y = np.cumsum(series, axis=1)

        fluctuations: list[np.ndarray] = []
        valid_scales: list[int] = []
        for scale in scales:
            scale = int(scale)
            n_seg = steps // scale
            if n_seg < 2:
                continue
            y_seg = y[:, : n_seg * scale].reshape(y.shape[0], n_seg, scale)
            x = np.arange(scale, dtype=np.float64)
            x_mean = np.mean(x)
            x_centered = x - x_mean
            denom = np.sum(x_centered**2)
            if denom <= 0:
                continue
            y_mean = np.mean(y_seg, axis=2)
            cov = np.sum(x_centered * (y_seg - y_mean[..., None]), axis=2)
            slope = cov / denom
            intercept = y_mean - slope * x_mean
            trend = slope[..., None] * x + intercept[..., None]
            residuals = y_seg - trend
            fluct = np.sqrt(np.mean(residuals**2, axis=(1, 2)))
            fluct[~np.isfinite(fluct) | (fluct <= 0)] = np.nan
            fluctuations.append(fluct)
            valid_scales.append(scale)

        if len(valid_scales) < 2:
            return np.full(data.shape[:2], np.nan, dtype=np.float32)

        x = np.log(np.asarray(valid_scales, dtype=np.float32))
        y_vals = np.log(np.stack(fluctuations, axis=0))
        mask = np.isfinite(y_vals)
        n = np.sum(mask, axis=0)
        sum_x = np.sum(mask * x[:, None], axis=0)
        sum_y = np.sum(np.where(mask, y_vals, 0.0), axis=0)
        sum_xx = np.sum(mask * (x[:, None] ** 2), axis=0)
        sum_xy = np.sum(np.where(mask, y_vals * x[:, None], 0.0), axis=0)
        denom = n * sum_xx - sum_x**2
        slope = np.full(y.shape[0], np.nan, dtype=np.float32)
        good = (n >= 2) & np.isfinite(denom) & (np.abs(denom) > 0)
        slope[good] = (n[good] * sum_xy[good] - sum_x[good] * sum_y[good]) / denom[good]
        return slope.reshape(data.shape[0], data.shape[1])

    def _hurst_exponent(self, data: np.ndarray) -> np.ndarray:
        """Estimate Hurst exponent per channel."""
        lags = self.cfg.resolve_hurst_lags(data.shape[2])
        if lags.size < 2:
            return np.full(data.shape[:2], np.nan, dtype=np.float32)
        steps = data.shape[2]
        series = data.reshape(data.shape[0] * data.shape[1], steps)

        tau_vals: list[np.ndarray] = []
        valid_lags: list[int] = []
        for lag in lags:
            lag = int(lag)
            if lag <= 0 or lag >= steps:
                continue
            diff = series[:, lag:] - series[:, :-lag]
            std = np.std(diff, axis=1)
            std[~np.isfinite(std) | (std <= 0)] = np.nan
            tau_vals.append(std)
            valid_lags.append(lag)

        if len(valid_lags) < 2:
            return np.full(data.shape[:2], np.nan, dtype=np.float32)

        x = np.log(np.asarray(valid_lags, dtype=np.float32))
        y_vals = np.log(np.stack(tau_vals, axis=0))
        mask = np.isfinite(y_vals)
        n = np.sum(mask, axis=0)
        sum_x = np.sum(mask * x[:, None], axis=0)
        sum_y = np.sum(np.where(mask, y_vals, 0.0), axis=0)
        sum_xx = np.sum(mask * (x[:, None] ** 2), axis=0)
        sum_xy = np.sum(np.where(mask, y_vals * x[:, None], 0.0), axis=0)
        denom = n * sum_xx - sum_x**2
        slope = np.full(series.shape[0], np.nan, dtype=np.float32)
        good = (n >= 2) & np.isfinite(denom) & (np.abs(denom) > 0)
        slope[good] = (n[good] * sum_xy[good] - sum_x[good] * sum_y[good]) / denom[good]
        return slope.reshape(data.shape[0], data.shape[1])

    def _one_over_f_exponent(self, freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
        """Estimate 1/f slope per channel from the PSD."""
        if freqs.size < 2 or psd.size == 0:
            return np.full(psd.shape[:2], np.nan, dtype=np.float32)
        mask = (freqs > 0) & np.isfinite(freqs)
        if np.count_nonzero(mask) < 2:
            return np.full(psd.shape[:2], np.nan, dtype=np.float32)
        x = np.log10(freqs[mask])
        y = np.log10(psd[:, :, mask])
        valid = np.isfinite(y) & (psd[:, :, mask] > 0)
        n = np.sum(valid, axis=2)
        sum_x = np.sum(valid * x[None, None, :], axis=2)
        sum_y = np.sum(np.where(valid, y, 0.0), axis=2)
        sum_xx = np.sum(valid * (x[None, None, :] ** 2), axis=2)
        sum_xy = np.sum(np.where(valid, y * x[None, None, :], 0.0), axis=2)
        denom = n * sum_xx - sum_x**2
        slope = np.full(psd.shape[:2], np.nan, dtype=np.float32)
        good = (n >= 2) & np.isfinite(denom) & (np.abs(denom) > 0)
        slope[good] = (n[good] * sum_xy[good] - sum_x[good] * sum_y[good]) / denom[good]
        return -slope


def build_pair_metrics(
    metrics: dict[str, Any],
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> dict[str, Any]:
    """Stack metrics into (pair, 2, ...) arrays for distance computation."""
    pair_metrics: dict[str, Any] = {
        "psd_freqs": metrics.get("psd_freqs"),
        "bands": metrics.get("bands", {}),
    }
    for key, value in metrics.items():
        if key in {"psd_freqs", "bands"}:
            continue
        if value is None:
            pair_metrics[key] = None
            continue
        pair_metrics[key] = np.stack(
            [value[left_indices], value[right_indices]], axis=1
        )
    return pair_metrics


class MetricDistances:
    """Compute per-pair distances from paired metrics."""

    def __init__(self, cfg: RolloutMetricsConfig) -> None:
        """Initialize distance computations for paired metrics."""
        self.cfg = cfg

    def compute(self, pair_metrics: dict[str, Any]) -> dict[str, np.ndarray]:
        """Compute all distances for the provided paired metrics."""
        distances: dict[str, np.ndarray] = {}
        timeseries = pair_metrics["timeseries"]
        distances["correlation"] = self._correlation_divergence(timeseries)
        distances["covariance"] = self._relative_l2(
            pair_metrics["covariance"][:, 0], pair_metrics["covariance"][:, 1]
        )

        distances["amplitude_kurtosis"] = self._channel_abs_diff(
            pair_metrics["amplitude_kurtosis"]
        )
        distances["amplitude_tail_fraction"] = self._channel_abs_diff(
            pair_metrics["amplitude_tail_fraction"]
        )
        distances["dfa_exponent"] = self._channel_abs_diff(pair_metrics["dfa_exponent"])
        distances["hurst_exponent"] = self._channel_abs_diff(
            pair_metrics["hurst_exponent"]
        )
        distances["one_over_f_exponent"] = self._channel_abs_diff(
            pair_metrics["one_over_f_exponent"]
        )
        distances["spatial_connectivity"] = np.abs(
            pair_metrics["spatial_connectivity"][:, 0]
            - pair_metrics["spatial_connectivity"][:, 1]
        )

        psd_freqs = pair_metrics.get("psd_freqs")
        psd = pair_metrics["psd"]
        distances["psd_jsd"] = self._psd_jsd(psd)
        distances["psd_corr"] = self._psd_corr(psd)
        distances["band_jsd"] = self._band_jsd(psd_freqs, psd)
        distances.update(self._psd_jsd_per_band(psd_freqs, psd))
        distances.update(self._psd_corr_per_band(psd_freqs, psd))
        distances.update(
            self._bandpower_ratio_distances(pair_metrics["bandpower_ratios"])
        )

        distances.update(self._stft_wasserstein_distances(pair_metrics))
        distances.update(self._fft_wasserstein_distances(pair_metrics))
        distances.update(self._coherence_distances(pair_metrics))

        return distances

    def _relative_l2(self, generated: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Relative L2 distance per pair."""
        if generated.size == 0 or target.size == 0:
            return np.full(generated.shape[0], np.nan, dtype=np.float32)
        gen_flat = generated.reshape(generated.shape[0], -1)
        tgt_flat = target.reshape(target.shape[0], -1)
        gen_norm = np.linalg.norm(gen_flat, axis=1)
        tgt_norm = np.linalg.norm(tgt_flat, axis=1)
        denom = np.maximum.reduce([gen_norm, tgt_norm, np.full_like(gen_norm, 1e-12)])
        diff_norm = np.linalg.norm(gen_flat - tgt_flat, axis=1)
        return diff_norm / denom

    def _correlation_divergence(self, pair_ts: np.ndarray) -> np.ndarray:
        """Channel-averaged correlation divergence in [0, 1]."""
        gen = pair_ts[:, 0]
        tgt = pair_ts[:, 1]
        gen = gen - gen.mean(axis=2, keepdims=True)
        tgt = tgt - tgt.mean(axis=2, keepdims=True)
        gen_std = gen.std(axis=2)
        tgt_std = tgt.std(axis=2)
        denom = gen_std * tgt_std
        corr = np.zeros_like(gen_std)
        valid = denom >= 1e-8
        corr[valid] = np.mean(gen[valid] * tgt[valid], axis=1) / denom[valid]
        corr = np.clip(corr, -1.0, 1.0, out=corr)
        divergence = (1.0 - corr) * 0.5
        small = ~valid
        if np.any(small):
            close = np.all(
                np.isclose(gen[small], tgt[small], atol=1e-8, rtol=0.0),
                axis=1,
            )
            divergence[small] = np.where(close, 0.0, 1.0)
        return np.mean(divergence, axis=1)

    def _channel_abs_diff(self, values: np.ndarray) -> np.ndarray:
        """Mean absolute per-channel difference for paired scalars."""
        diff = np.abs(values[:, 0] - values[:, 1])
        return _safe_nanmean(diff, axis=1)

    def _jsd(self, p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Compute Jensen-Shannon distance for distributions."""
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        p = p / np.sum(p, axis=-1, keepdims=True)
        q = q / np.sum(q, axis=-1, keepdims=True)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=-1)
        kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=-1)
        jsd = 0.5 * (kl_pm + kl_qm)
        return np.sqrt(np.maximum(jsd, 0.0))

    def _psd_jsd(self, psd: np.ndarray) -> np.ndarray:
        """Compute normalized PSD JSD per pair (channel-averaged)."""
        if psd.size == 0:
            return np.full(psd.shape[0], np.nan, dtype=np.float32)
        gen = psd[:, 0]
        tgt = psd[:, 1]
        gen_total = np.sum(gen, axis=2, keepdims=True)
        tgt_total = np.sum(tgt, axis=2, keepdims=True)
        gen_dist = gen / np.clip(gen_total, 1e-12, None)
        tgt_dist = tgt / np.clip(tgt_total, 1e-12, None)
        jsd = self._jsd(tgt_dist, gen_dist)
        return _safe_nanmean(jsd, axis=1) / _JSD_MAX

    def _psd_corr(self, psd: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Compute PSD correlation per pair (channel-averaged)."""
        if psd.size == 0:
            return np.full(psd.shape[0], np.nan, dtype=np.float32)
        gen = psd[:, 0]
        tgt = psd[:, 1]
        if mask is not None:
            gen = gen[:, :, mask]
            tgt = tgt[:, :, mask]
        if gen.shape[2] < 2 or tgt.shape[2] < 2:
            return np.full(psd.shape[0], np.nan, dtype=np.float32)
        gen_centered = gen - np.mean(gen, axis=2, keepdims=True)
        tgt_centered = tgt - np.mean(tgt, axis=2, keepdims=True)
        denom = np.linalg.norm(gen_centered, axis=2) * np.linalg.norm(
            tgt_centered, axis=2
        )
        corr = np.divide(
            np.sum(gen_centered * tgt_centered, axis=2),
            denom,
            out=np.full_like(denom, np.nan),
            where=denom > 0,
        )
        corr = np.where(np.isfinite(corr), corr, np.nan)
        return _safe_nanmean(corr, axis=1)

    def _psd_jsd_per_band(
        self, freqs: np.ndarray | None, psd: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute normalized PSD JSD for each band."""
        bands = self.cfg.bands
        if freqs is None or freqs.size == 0 or psd.size == 0 or not bands:
            return {f"psd_jsd_{name}": np.full(psd.shape[0], np.nan) for name in bands}
        results: dict[str, np.ndarray] = {}
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                results[f"psd_jsd_{name}"] = np.full(psd.shape[0], np.nan)
                continue
            band_idx = np.flatnonzero(mask)
            gen = np.take(psd[:, 0], band_idx, axis=2)
            tgt = np.take(psd[:, 1], band_idx, axis=2)
            gen_total = np.sum(gen, axis=2, keepdims=True)
            tgt_total = np.sum(tgt, axis=2, keepdims=True)
            gen_dist = gen / np.clip(gen_total, 1e-12, None)
            tgt_dist = tgt / np.clip(tgt_total, 1e-12, None)
            jsd = self._jsd(tgt_dist, gen_dist)
            results[f"psd_jsd_{name}"] = _safe_nanmean(jsd, axis=1) / _JSD_MAX
        return results

    def _psd_corr_per_band(
        self, freqs: np.ndarray | None, psd: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute PSD correlation per band."""
        bands = self.cfg.bands
        if freqs is None or freqs.size == 0 or psd.size == 0 or not bands:
            return {f"psd_corr_{name}": np.full(psd.shape[0], np.nan) for name in bands}
        results: dict[str, np.ndarray] = {}
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            results[f"psd_corr_{name}"] = self._psd_corr(psd, mask=mask)
        return results

    def _band_jsd(self, freqs: np.ndarray | None, psd: np.ndarray) -> np.ndarray:
        """Compute normalized bandpower JSD per pair (channel-averaged)."""
        bands = list(self.cfg.bands.values())
        if freqs is None or freqs.size == 0 or psd.size == 0 or not bands:
            return np.full(psd.shape[0], np.nan, dtype=np.float32)
        gen = psd[:, 0]
        tgt = psd[:, 1]
        band_edges = np.array(bands, dtype=float)
        band_pow_t = np.zeros((gen.shape[0], gen.shape[1], band_edges.shape[0]))
        band_pow_g = np.zeros_like(band_pow_t)
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        for bi, (lo, hi) in enumerate(band_edges):
            bm = (freqs >= lo) & (freqs < hi)
            if not np.any(bm):
                band_pow_t[:, :, bi] = 1e-12
                band_pow_g[:, :, bi] = 1e-12
                continue
            band_pow_t[:, :, bi] = trapz(tgt[:, :, bm], freqs[bm], axis=2)
            band_pow_g[:, :, bi] = trapz(gen[:, :, bm], freqs[bm], axis=2)
        bt = band_pow_t + 1e-12
        bg = band_pow_g + 1e-12
        bt = bt / np.sum(bt, axis=2, keepdims=True)
        bg = bg / np.sum(bg, axis=2, keepdims=True)
        jsd = self._jsd(bt, bg)
        return _safe_nanmean(jsd, axis=1) / _JSD_MAX

    def _bandpower_ratio_distances(self, ratios: np.ndarray) -> dict[str, np.ndarray]:
        """Compute per-band relative distance between bandpower ratios."""
        bands = list(self.cfg.bands.keys())
        if ratios.size == 0 or not bands:
            return {
                f"bandpower_ratio_{name}": np.full(ratios.shape[0], np.nan)
                for name in bands
            }
        diffs = np.abs(ratios[:, 0] - ratios[:, 1])
        means = 0.5 * (ratios[:, 0] + ratios[:, 1])
        denom = np.clip(means, 1e-6, None)
        rel = np.clip(diffs / denom, 0.0, 1.0)
        results: dict[str, np.ndarray] = {}
        for idx, name in enumerate(bands):
            results[f"bandpower_ratio_{name}"] = _safe_nanmean(rel[:, :, idx], axis=1)
        return results

    def _stft_wasserstein_distances(
        self, pair_metrics: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Compute normalized STFT magnitude/angle distances."""
        mag = pair_metrics["stft_magnitude"]
        angle = pair_metrics["stft_angle"]
        if mag.size == 0 or angle.size == 0:
            nan = np.full(mag.shape[0], np.nan, dtype=np.float32)
            return {"stft_magnitude": nan, "stft_angle": nan}

        log_mag, mag_weights = self._normalize_magnitude_pair(mag[:, 0], mag[:, 1])
        distances = {
            "stft_magnitude": self._wasserstein_distance(log_mag[:, 0], log_mag[:, 1]),
            "stft_angle": self._circular_distance(
                angle[:, 0], angle[:, 1], weights=mag_weights
            ),
        }
        return distances

    def _fft_wasserstein_distances(
        self, pair_metrics: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Compute normalized FFT magnitude/angle distances."""
        mag = pair_metrics["fft_magnitude"]
        angle = pair_metrics["fft_angle"]
        if mag.size == 0 or angle.size == 0:
            nan = np.full(mag.shape[0], np.nan, dtype=np.float32)
            return {"fft_magnitude": nan, "fft_angle": nan}
        norm_mag, mag_weights = self._normalize_magnitude_pair(mag[:, 0], mag[:, 1])
        distances = {
            "fft_magnitude": self._wasserstein_distance(norm_mag[:, 0], norm_mag[:, 1]),
            "fft_angle": self._circular_distance(
                angle[:, 0], angle[:, 1], weights=mag_weights
            ),
        }
        return distances

    def _wasserstein_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute per-pair Wasserstein-1 distance for flattened features."""
        if a.size == 0 or b.size == 0:
            return np.full(a.shape[0], np.nan, dtype=np.float32)
        a_flat = a.reshape(a.shape[0], a.shape[1], -1)
        b_flat = b.reshape(b.shape[0], b.shape[1], -1)
        min_len = min(a_flat.shape[2], b_flat.shape[2])
        if min_len <= 0:
            return np.full(a.shape[0], np.nan, dtype=np.float32)
        a_flat = a_flat[:, :, :min_len]
        b_flat = b_flat[:, :, :min_len]
        a_sorted = np.sort(a_flat, axis=2)
        b_sorted = np.sort(b_flat, axis=2)
        distances = np.mean(np.abs(a_sorted - b_sorted), axis=2)
        return _safe_nanmean(distances, axis=1)

    def _coherence_distances(
        self, pair_metrics: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Compute per-band coherence distance from within-signal coherence."""
        bands = self.cfg.bands
        if not bands:
            return {"coherence": np.full(pair_metrics["timeseries"].shape[0], np.nan)}
        pair_ts = pair_metrics["timeseries"]
        steps = pair_ts.shape[3]
        fs = self.cfg.fs
        if fs is None or fs <= 0 or steps < 2:
            results = {
                f"coherence_{name}": np.full(pair_ts.shape[0], np.nan) for name in bands
            }
            results["coherence"] = np.full(pair_ts.shape[0], np.nan)
            return results
        nperseg = int(round(self.cfg.welch_nperseg_s * fs))
        nperseg = max(8, nperseg)
        if nperseg < 2 or nperseg > steps:
            results = {
                f"coherence_{name}": np.full(pair_ts.shape[0], np.nan) for name in bands
            }
            results["coherence"] = np.full(pair_ts.shape[0], np.nan)
            return results
        noverlap = int(round(self.cfg.welch_noverlap_frac * nperseg))
        noverlap = min(noverlap, max(0, nperseg - 1))

        freqs, coh_gen = self._coherence_matrix(pair_ts[:, 0], fs, nperseg, noverlap)
        _, coh_tgt = self._coherence_matrix(pair_ts[:, 1], fs, nperseg, noverlap)

        psd_freqs = pair_metrics.get("psd_freqs")
        if psd_freqs is not None and psd_freqs.size > 0 and freqs.size > 0:
            fmin = float(psd_freqs[0])
            fmax = float(psd_freqs[-1])
            mask = (freqs >= fmin) & (freqs <= fmax)
            if np.any(mask):
                freqs = freqs[mask]
                coh_gen = coh_gen[..., mask]
                coh_tgt = coh_tgt[..., mask]

        results: dict[str, np.ndarray] = {}
        off_diag = ~np.eye(coh_gen.shape[1], dtype=bool)
        diff = np.abs(coh_gen - coh_tgt)
        diff_vals = diff[:, off_diag]
        results["coherence"] = _safe_nanmean(_safe_nanmean(diff_vals, axis=2), axis=1)
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                results[f"coherence_{name}"] = np.full(pair_ts.shape[0], np.nan)
                continue
            gen_band = _safe_nanmean(coh_gen[..., mask], axis=3)
            tgt_band = _safe_nanmean(coh_tgt[..., mask], axis=3)
            band_diff = np.abs(gen_band - tgt_band)
            band_vals = band_diff[:, off_diag]
            results[f"coherence_{name}"] = _safe_nanmean(band_vals, axis=1)
        return results

    def _normalize_magnitude_pair(
        self, a: np.ndarray, b: np.ndarray, eps: float = 1e-12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize magnitudes per pair/channel and return magnitude weights."""
        a_flat = a.reshape(a.shape[0], a.shape[1], -1)
        b_flat = b.reshape(b.shape[0], b.shape[1], -1)
        scale = 0.5 * (
            np.mean(np.abs(a_flat), axis=2) + np.mean(np.abs(b_flat), axis=2)
        )
        scale = np.clip(scale, eps, None)
        a_norm = (a_flat / scale[:, :, None]).reshape(a.shape)
        b_norm = (b_flat / scale[:, :, None]).reshape(b.shape)
        log_norm = np.log(np.stack([a_norm, b_norm], axis=1) + eps)
        weights = 0.5 * (a + b)
        return log_norm, weights

    def _circular_distance(
        self,
        a: np.ndarray,
        b: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute mean wrapped angular distance, normalized to [0, 1]."""
        wrapped = np.angle(np.exp(1j * (a - b)))
        abs_wrapped = np.abs(wrapped)
        axes = tuple(range(2, abs_wrapped.ndim))
        if weights is None:
            dist = np.mean(abs_wrapped, axis=axes)
        else:
            w = np.clip(weights, 0.0, None)
            try:
                w = np.broadcast_to(w, abs_wrapped.shape)
            except ValueError:
                dist = np.mean(abs_wrapped, axis=axes)
            else:
                w_sum = np.sum(w, axis=axes)
                weighted = np.sum(w * abs_wrapped, axis=axes)
                dist = weighted / np.where(w_sum > 0, w_sum, np.nan)
        return _safe_nanmean(dist, axis=1) / np.pi

    def _coherence_matrix(
        self,
        data: np.ndarray,
        fs: float,
        nperseg: int,
        noverlap: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute within-signal coherence matrix per example."""
        step = nperseg - noverlap
        if step <= 0 or data.shape[2] < nperseg:
            return np.array([], dtype=np.float32), np.full(
                (data.shape[0], data.shape[1], data.shape[1], 0),
                np.nan,
                dtype=np.float32,
            )
        windows = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=nperseg, axis=2
        )
        segments = windows[:, :, ::step, :]
        if segments.shape[2] == 0:
            return np.array([], dtype=np.float32), np.full(
                (data.shape[0], data.shape[1], data.shape[1], 0),
                np.nan,
                dtype=np.float32,
            )
        window = signal.get_window("hann", nperseg, fftbins=True).astype(np.float32)
        segments = segments * window[None, None, None, :]
        spec = np.fft.rfft(segments, axis=3)
        pxx = np.mean(np.abs(spec) ** 2, axis=2)
        pxy = np.einsum("bcsf,bdsf->bcdf", spec, np.conj(spec), optimize=True) / float(
            segments.shape[2]
        )
        denom = pxx[:, :, None, :] * pxx[:, None, :, :]
        coh = np.full_like(pxy, np.nan, dtype=np.float32)
        valid = denom > 0
        coh[valid] = (np.abs(pxy[valid]) ** 2) / denom[valid]
        coh = np.clip(coh, 0.0, 1.0)
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs).astype(np.float32)
        return freqs, coh


def compute_window_distances(
    window_indices: list[int],
    window_points: list[int],
    combined: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    cfg: RolloutMetricsConfig,
    spatial_weights: np.ndarray | None,
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Compute distance curves for a subset of rollout windows."""
    print(f"[rollout_metrics] Worker processing {len(window_indices)} windows.")
    metrics_computer = RolloutMetrics(cfg, spatial_weights=spatial_weights)
    distance_computer = MetricDistances(cfg)
    pair_count = left_indices.shape[0]
    results: dict[str, np.ndarray] = {}
    for local_idx, window_idx in enumerate(window_indices):
        step = window_points[window_idx]
        window_data = combined[:, :, :step]
        metrics = metrics_computer.compute(window_data)
        pair_metrics = build_pair_metrics(metrics, left_indices, right_indices)
        distances = distance_computer.compute(pair_metrics)
        if not results:
            for name in distances:
                results[name] = np.full(
                    (pair_count, len(window_indices)),
                    np.nan,
                    dtype=np.float32,
                )
        for name, vals in distances.items():
            results[name][:, local_idx] = np.asarray(vals, dtype=np.float32)
    print(f"[rollout_metrics] Worker finished {len(window_indices)} windows.")
    return window_indices, results


def compute_window_summaries(
    window_indices: list[int],
    window_starts: list[int],
    window_steps: int,
    combined: np.ndarray,
    cfg: RolloutMetricsConfig,
    spatial_weights: np.ndarray | None,
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Compute summary metrics for a subset of sliding windows."""
    print(f"[rollout_metrics] Worker processing {len(window_indices)} windows.")
    metrics_computer = RolloutMetrics(cfg, spatial_weights=spatial_weights)
    run_count = combined.shape[0]
    results: dict[str, np.ndarray] = {}
    for local_idx, window_idx in enumerate(window_indices):
        start = window_starts[window_idx]
        end = start + window_steps
        window_data = combined[:, :, start:end]
        summaries = metrics_computer.compute_summaries(window_data)
        if not results:
            for name in summaries:
                results[name] = np.full(
                    (run_count, len(window_indices)),
                    np.nan,
                    dtype=np.float32,
                )
        for name, vals in summaries.items():
            results[name][:, local_idx] = np.asarray(vals, dtype=np.float32)
    print(f"[rollout_metrics] Worker finished {len(window_indices)} windows.")
    return window_indices, results
