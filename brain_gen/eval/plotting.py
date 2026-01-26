from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
import torch
import scipy.signal as signal


class EvaluationPlotting:
    """Shared plotting utilities for evaluation outputs."""

    def __init__(self, *, sfreq: float | None, val_dataset: Any | None = None) -> None:
        """Initialize plotting helpers with dataset context."""
        self.sfreq = sfreq
        self.val_dataset = val_dataset

    def _normalise_timeseries(self, tensor: torch.Tensor | Any) -> torch.Tensor | None:
        """Convert arbitrary tensors to a (C, T) float tensor without a batch dim."""
        if tensor is None:
            return None

        try:
            arr = (
                tensor.detach() if torch.is_tensor(tensor) else torch.as_tensor(tensor)
            )
        except Exception:
            return None

        if arr.ndim >= 3:
            arr = arr[0]

        if arr.ndim == 0:
            return None

        if arr.ndim == 1:
            arr = arr.unsqueeze(0)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        if arr.shape[0] > arr.shape[1]:
            arr = arr.transpose(0, 1)

        return arr.to(torch.float32)

    def _resolve_plot_steps(
        self, max_steps: int, max_plot_seconds: float | None
    ) -> int:
        """Resolve the number of steps to display when a seconds cap is provided."""
        if max_plot_seconds is None:
            return max_steps
        max_seconds = float(max_plot_seconds)
        if max_seconds <= 0:
            return max_steps
        cap = int(max_seconds * float(self.sfreq)) if self.sfreq else int(max_seconds)
        if cap <= 0:
            return max_steps
        return min(max_steps, cap)

    def _plot_timeseries_pair(
        self,
        generated: np.ndarray,
        target: np.ndarray,
        prefix: str,
        out_dir: Path,
        channel_indices: np.ndarray | list[int] | None = None,
        context_len: int | None = None,
    ) -> None:
        """Plot generated vs target time series pairs."""
        prepared = self._prepare_plot_pair(
            generated,
            target,
            min_steps=1,
            channel_indices=channel_indices,
        )
        if prepared is None:
            return
        gen_data, tgt_data, indices, max_steps = prepared
        n_channels = gen_data.shape[0]
        fig, axes = plt.subplots(
            n_channels,
            2,
            figsize=(60, max(5, int(n_channels * 5))),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        palette = sns.color_palette("husl", n_channels)
        label_fontsize = 14
        time_axis = (
            np.arange(max_steps) / float(self.sfreq)
            if self.sfreq
            else np.arange(max_steps)
        )
        xlabel = "Time (s)" if self.sfreq else "Samples"
        fig.suptitle("Generated (left) vs Target (right)")

        for idx in range(n_channels):
            label = int(indices[idx])
            color = palette[idx % len(palette)]
            ax_gen = axes[idx, 0]
            ax_tgt = axes[idx, 1]
            ax_gen.plot(
                time_axis,
                gen_data[idx],
                color=color,
                linewidth=0.8,
                alpha=0.9,
            )
            ax_tgt.plot(
                time_axis,
                tgt_data[idx],
                color=color,
                linewidth=0.8,
                alpha=0.9,
            )
            ax_gen.set_ylabel(f"ch {label}", fontsize=label_fontsize)
            ax_tgt.set_ylabel(f"ch {label}", fontsize=label_fontsize)
            ax_gen.set_title(f"Generated channel {label}", loc="left", fontsize=10)
            ax_tgt.set_title(f"Target channel {label}", loc="left", fontsize=10)
            ax_gen.grid(False)
            ax_tgt.grid(False)

            ymin = min(np.min(gen_data[idx]), np.min(tgt_data[idx]))
            ymax = max(np.max(gen_data[idx]), np.max(tgt_data[idx]))
            if np.isclose(ymin, ymax):
                pad = 1e-6 if ymin == 0 else abs(ymin) * 0.05
                ymin -= pad
                ymax += pad
            ax_gen.set_ylim(ymin, ymax)
            ax_tgt.set_ylim(ymin, ymax)

            self._plot_context_marker(
                ax_gen,
                context_len,
                max_steps,
                color="red",
                linestyle="--",
                linewidth=4.0,
                alpha=0.9,
            )
            self._plot_context_marker(
                ax_tgt,
                context_len,
                max_steps,
                color="red",
                linestyle="--",
                linewidth=4.0,
                alpha=0.9,
            )

        axes[-1, 0].set_xlabel(xlabel, fontsize=label_fontsize)
        axes[-1, 1].set_xlabel(xlabel, fontsize=label_fontsize)
        fig.savefig(out_dir / f"{prefix}_timeseries.png", bbox_inches="tight")
        plt.close(fig)

    def plot_timeseries_pair(
        self,
        generated: np.ndarray,
        target: np.ndarray,
        prefix: str,
        out_dir: Path,
        channel_indices: np.ndarray | list[int] | None = None,
        context_len: int | None = None,
    ) -> None:
        """Public wrapper for plotting matched time series."""
        self._plot_timeseries_pair(
            generated,
            target,
            prefix,
            out_dir,
            channel_indices=channel_indices,
            context_len=context_len,
        )

    def _plot_stacked_timeseries_pair(
        self,
        generated: np.ndarray,
        target: np.ndarray,
        prefix: str,
        out_dir: Path,
        channel_indices: np.ndarray | list[int] | None = None,
        context_len: int | None = None,
        max_plot_seconds: float | None = None,
    ) -> None:
        """Plot generated vs target signals with channels overlaid per pair."""
        gen_arr = np.asarray(generated)
        tgt_arr = np.asarray(target)
        if gen_arr.ndim == 2:
            gen_arr = gen_arr[None, ...]
        if tgt_arr.ndim == 2:
            tgt_arr = tgt_arr[None, ...]
        if gen_arr.ndim != 3 or tgt_arr.ndim != 3:
            return

        max_pairs = min(gen_arr.shape[0], tgt_arr.shape[0])
        max_channels = min(gen_arr.shape[1], tgt_arr.shape[1])
        max_steps = min(gen_arr.shape[2], tgt_arr.shape[2])
        if max_pairs <= 0 or max_channels <= 0 or max_steps <= 0:
            return

        indices = (
            np.asarray(channel_indices)
            if channel_indices is not None
            else np.arange(max_channels)
        )
        indices = indices[indices < max_channels]
        if indices.size == 0:
            return

        max_steps = self._resolve_plot_steps(max_steps, max_plot_seconds)
        gen_data = gen_arr[:max_pairs, indices, :max_steps]
        tgt_data = tgt_arr[:max_pairs, indices, :max_steps]
        n_pairs = gen_data.shape[0]
        n_channels = gen_data.shape[1]

        fig_height = max(6, int(n_pairs * 3.2))
        fig, axes = plt.subplots(
            n_pairs,
            2,
            figsize=(60, fig_height),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        palette = sns.color_palette("husl", n_channels)
        label_fontsize = 14
        time_axis = (
            np.arange(max_steps) / float(self.sfreq)
            if self.sfreq
            else np.arange(max_steps)
        )
        xlabel = "Time (s)" if self.sfreq else "Samples"

        for row_idx in range(n_pairs):
            for col_idx, data in enumerate((gen_data[row_idx], tgt_data[row_idx])):
                ax = axes[row_idx, col_idx]
                for ch_idx in range(n_channels):
                    color = palette[ch_idx % len(palette)]
                    ax.plot(
                        time_axis,
                        data[ch_idx],
                        color=color,
                        linewidth=0.8,
                        alpha=0.8,
                    )
                if n_pairs > 1:
                    label = f"run {row_idx}"
                else:
                    label = ""
                title = "Generated" if col_idx == 0 else "Target"
                if label:
                    title = f"{title} ({label})"
                ax.set_title(title, loc="left", fontsize=11)
                ax.grid(False)
                self._plot_context_marker(
                    ax,
                    context_len,
                    max_steps,
                    color="red",
                    linestyle="--",
                    linewidth=3.0,
                    alpha=0.8,
                )

        axes[-1, 0].set_xlabel(xlabel, fontsize=label_fontsize)
        axes[-1, 1].set_xlabel(xlabel, fontsize=label_fontsize)
        for row_idx in range(n_pairs):
            axes[row_idx, 0].set_ylabel("Amplitude", fontsize=label_fontsize)
            axes[row_idx, 1].set_ylabel("Amplitude", fontsize=label_fontsize)

        fig.savefig(out_dir / f"{prefix}_stacked_timeseries.png", bbox_inches="tight")
        plt.close(fig)

    def plot_stacked_timeseries_pair(
        self,
        generated: np.ndarray,
        target: np.ndarray,
        prefix: str,
        out_dir: Path,
        channel_indices: np.ndarray | list[int] | None = None,
        context_len: int | None = None,
        max_plot_seconds: float | None = None,
    ) -> None:
        """Public wrapper for stacked generated vs target plots."""
        self._plot_stacked_timeseries_pair(
            generated,
            target,
            prefix,
            out_dir,
            channel_indices=channel_indices,
            context_len=context_len,
            max_plot_seconds=max_plot_seconds,
        )

    def _plot_stft_pair(
        self,
        generated: np.ndarray,
        target: np.ndarray,
        prefix: str,
        out_dir: Path,
        channel_indices: np.ndarray | list[int] | None = None,
        context_len: int | None = None,
    ) -> None:
        """Plot STFT pairs for generated vs target signals."""
        prepared = self._prepare_plot_pair(
            generated,
            target,
            min_steps=2,
            channel_indices=channel_indices,
        )
        if prepared is None:
            return
        gen_data, tgt_data, indices, max_steps = prepared
        n_channels = gen_data.shape[0]
        nfft = min(128, max_steps)
        noverlap = int(nfft * 0.75) if nfft > 1 else 0
        fs = float(self.sfreq) if self.sfreq else 1.0
        eps = 1e-12
        fig, axes = plt.subplots(
            n_channels,
            2,
            figsize=(60, max(5, int(n_channels * 5))),
            sharex=True,
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
        last_im = None
        fig.suptitle("Generated (left) vs Target (right)")
        label_fontsize = 14

        for row_idx, ch_idx in enumerate(indices):
            f_gen, t_gen, Pxx_gen = signal.spectrogram(
                gen_data[row_idx],
                fs=fs,
                nperseg=nfft,
                noverlap=noverlap,
                scaling="density",
                mode="psd",
            )
            f_tgt, t_tgt, Pxx_tgt = signal.spectrogram(
                tgt_data[row_idx],
                fs=fs,
                nperseg=nfft,
                noverlap=noverlap,
                scaling="density",
                mode="psd",
            )
            Pxx_gen_db = 10.0 * np.log10(Pxx_gen + eps)
            Pxx_tgt_db = 10.0 * np.log10(Pxx_tgt + eps)
            vmin = min(float(np.min(Pxx_gen_db)), float(np.min(Pxx_tgt_db)))
            vmax = max(float(np.max(Pxx_gen_db)), float(np.max(Pxx_tgt_db)))

            for col_idx, (data_db, freqs, times, label) in enumerate(
                (
                    (Pxx_gen_db, f_gen, t_gen, "Generated"),
                    (Pxx_tgt_db, f_tgt, t_tgt, "Target"),
                )
            ):
                ax = axes[row_idx, col_idx]
                if times.size >= 2 and freqs.size >= 2:
                    extent = (times[0], times[-1], freqs[0], freqs[-1])
                else:
                    duration = max_steps / fs
                    extent = (0.0, duration, freqs[0], freqs[-1])
                im = ax.imshow(
                    data_db,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax,
                )
                last_im = im
                ax.set_ylabel(f"ch {int(ch_idx)}", fontsize=label_fontsize)
                ax.set_title(f"{label} STFT ch {int(ch_idx)}", loc="left", fontsize=10)
                ax.grid(False)
                self._plot_context_marker(
                    ax,
                    context_len,
                    max_steps,
                    color="black",
                    linestyle="--",
                    linewidth=4.0,
                    alpha=0.9,
                )

        xlabel = "Time (s)" if self.sfreq else "Samples"
        axes[-1, 0].set_xlabel(xlabel, fontsize=label_fontsize)
        axes[-1, 1].set_xlabel(xlabel, fontsize=label_fontsize)
        if last_im is not None:
            fig.colorbar(
                last_im,
                ax=axes.ravel().tolist(),
                orientation="vertical",
                fraction=0.02,
                pad=0.01,
                label="Power (dB)",
            )

        fig.savefig(out_dir / f"{prefix}_stft.png", bbox_inches="tight")
        plt.close(fig)

    def plot_stft_pair(
        self,
        generated: np.ndarray,
        target: np.ndarray,
        prefix: str,
        out_dir: Path,
        channel_indices: np.ndarray | list[int] | None = None,
        context_len: int | None = None,
    ) -> None:
        """Public wrapper for plotting STFT pairs."""
        self._plot_stft_pair(
            generated,
            target,
            prefix,
            out_dir,
            channel_indices=channel_indices,
            context_len=context_len,
        )

    def _plot_metric_grid(
        self, losses: list[float], metrics: dict[str, list[float]]
    ) -> plt.Figure:
        """Plot each metric on its own subplot within a single figure."""
        metric_items = [("loss", losses)] + list(metrics.items())
        n = len(metric_items)
        fig, axes = plt.subplots(1, n, figsize=(4 * max(2, n), 4))
        # Normalize axes handling for the n==1 case
        if isinstance(axes, np.ndarray):
            axes_list = axes.flatten().tolist()
        else:
            axes_list = [axes]

        for ax, (name, values) in zip(axes_list, metric_items):
            if values:
                sns.violinplot(y=values, cut=0, inner="box", ax=ax)
                ax.set_ylabel(name)
                ax.set_title(f"{name} distribution")
            else:
                ax.set_title(f"No data for {name}")
                ax.axis("off")
                continue
            ax.grid(False)

        fig.tight_layout()
        return fig

    def plot_metric_grid(
        self, losses: list[float], metrics: dict[str, list[float]]
    ) -> plt.Figure:
        """Public wrapper for metric grid plots."""
        return self._plot_metric_grid(losses, metrics)

    def _compute_psd(
        self, tensor: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute PSD for a tensor and return (freqs, psd) arrays."""
        seq = self._normalise_timeseries(tensor)
        if seq is None or seq.numel() == 0:
            return None

        arr = seq.cpu().numpy()
        fs = float(self.sfreq) if self.sfreq is not None else 1.0
        nperseg = min(arr.shape[-1], max(1, int(fs)))
        if nperseg <= 0:
            return None

        freqs, psd = signal.welch(
            arr,
            fs=fs,
            axis=-1,
            nperseg=nperseg,
            scaling="density",
        )
        return freqs, psd

    def _compute_psd_cov(
        self, data_runs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Compute mean PSD and covariance across runs."""
        if not data_runs or self.sfreq is None:
            return None

        psd_list: list[np.ndarray] = []
        cov_list: list[np.ndarray] = []
        freqs_ref: np.ndarray | None = None

        for data in data_runs:
            freqs, psd = signal.welch(
                data,
                fs=self.sfreq,
                axis=-1,
                nperseg=self.sfreq,
                scaling="density",
            )
            freqs_ref = freqs if freqs_ref is None else freqs_ref
            psd_list.append(psd)
            cov_list.append(np.cov(data))

        psd_mean = np.mean(np.stack(psd_list, axis=0), axis=0)
        cov_mean = np.mean(np.stack(cov_list, axis=0), axis=0)
        return freqs_ref if freqs_ref is not None else np.array([]), psd_mean, cov_mean

    def _save_psd_pair(
        self, preds: torch.Tensor, target: torch.Tensor, prefix: str, out_dir: Path
    ) -> None:
        """Save side-by-side PSD plots for predictions and targets."""
        pred_res = self._compute_psd(preds)
        tgt_res = self._compute_psd(target)
        if pred_res is None or tgt_res is None:
            return

        pred_freqs, pred_psd = pred_res
        tgt_freqs, tgt_psd = tgt_res

        fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
        plots = [
            ("Preds", pred_freqs, pred_psd, axes[0]),
            ("Target", tgt_freqs, tgt_psd, axes[1]),
        ]
        ylims = self._psd_ylim(pred_psd, tgt_psd)

        for title, freqs, psd, ax in plots:
            ax.plot(freqs, psd.T, alpha=0.3)
            ax.set_xlabel("Hz")
            ax.set_ylabel("Power")
            ax.set_title(f"{title} PSD - {prefix}")
            ax.set_yscale("log")
            if ylims:
                ax.set_ylim(ylims)

        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_psd.png", bbox_inches="tight")
        plt.close(fig)

        pred_psd_mean = pred_psd.mean(axis=0)
        tgt_psd_mean = tgt_psd.mean(axis=0)
        fig_mean, ax_mean = plt.subplots(1, 1, figsize=(10, 6))
        ax_mean.plot(pred_freqs, pred_psd_mean, label="Preds")
        ax_mean.plot(tgt_freqs, tgt_psd_mean, label="Target")
        ax_mean.set_xlabel("Hz")
        ax_mean.set_ylabel("Power")
        ax_mean.set_title(f"Mean PSD - {prefix}")
        ax_mean.set_yscale("log")
        ax_mean.legend()

        mean_ylim = self._psd_ylim(pred_psd_mean, tgt_psd_mean)
        if mean_ylim:
            ax_mean.set_ylim(mean_ylim)

        fig_mean.tight_layout()
        fig_mean.savefig(out_dir / f"{prefix}_psd_mean.png", bbox_inches="tight")
        plt.close(fig_mean)

    def _save_example_summaries(
        self,
        example_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        out_dir: Path,
        prefix: str = "examples",
    ) -> None:
        """Save aggregated PSD and covariance summaries across examples."""
        if not example_pairs:
            return

        self._save_example_psd_summary(example_pairs, out_dir, prefix)
        self._save_example_cov_summary(example_pairs, out_dir, prefix)

    def _save_example_psd_summary(
        self,
        example_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        out_dir: Path,
        prefix: str,
    ) -> None:
        """Plot mean +/- std PSD across examples for reconstructions vs targets."""
        pred_psd_means: list[np.ndarray] = []
        tgt_psd_means: list[np.ndarray] = []
        freqs_ref: np.ndarray | None = None

        for preds, target in example_pairs:
            pred_res = self._compute_psd(preds)
            tgt_res = self._compute_psd(target)
            if pred_res is None or tgt_res is None:
                continue

            pred_freqs, pred_psd = pred_res
            tgt_freqs, tgt_psd = tgt_res

            if freqs_ref is None:
                freqs_ref = pred_freqs

            if freqs_ref is None:
                continue

            if pred_freqs.shape != freqs_ref.shape or not np.allclose(
                pred_freqs, freqs_ref
            ):
                continue
            if tgt_freqs.shape != freqs_ref.shape or not np.allclose(
                tgt_freqs, freqs_ref
            ):
                continue

            pred_psd_means.append(pred_psd.mean(axis=0))
            tgt_psd_means.append(tgt_psd.mean(axis=0))

        if not pred_psd_means or not tgt_psd_means or freqs_ref is None:
            return

        pred_stack = np.stack(pred_psd_means, axis=0)
        tgt_stack = np.stack(tgt_psd_means, axis=0)
        pred_mean = pred_stack.mean(axis=0)
        pred_std = pred_stack.std(axis=0)
        tgt_mean = tgt_stack.mean(axis=0)
        tgt_std = tgt_stack.std(axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # Keep bands positive for log-scaled PSD plots.
        eps = np.finfo(float).tiny
        pred_lower = np.maximum(pred_mean - pred_std, eps)
        pred_upper = np.maximum(pred_mean + pred_std, eps)
        tgt_lower = np.maximum(tgt_mean - tgt_std, eps)
        tgt_upper = np.maximum(tgt_mean + tgt_std, eps)

        ax.plot(freqs_ref, pred_mean, color="tab:blue", label="Reconstructed")
        ax.fill_between(freqs_ref, pred_lower, pred_upper, color="tab:blue", alpha=0.2)
        ax.plot(freqs_ref, tgt_mean, color="tab:orange", label="Ground truth")
        ax.fill_between(freqs_ref, tgt_lower, tgt_upper, color="tab:orange", alpha=0.2)
        ax.set_xlabel("Hz")
        ax.set_ylabel("Power")
        ax.set_title("Example PSD summary (mean +/- std)")
        ax.set_yscale("log")
        ax.legend()

        ylims = self._psd_ylim(pred_mean, tgt_mean)
        if ylims:
            ax.set_ylim(ylims)

        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_psd_summary.png", bbox_inches="tight")
        plt.close(fig)

    def _save_example_cov_summary(
        self,
        example_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        out_dir: Path,
        prefix: str,
    ) -> None:
        """Plot mean cov matrices across examples for reconstructions vs targets."""
        pred_covs: list[np.ndarray] = []
        tgt_covs: list[np.ndarray] = []
        n_channels: int | None = None

        for preds, target in example_pairs:
            pred_ts = self._normalise_timeseries(preds)
            tgt_ts = self._normalise_timeseries(target)
            if pred_ts is None or tgt_ts is None:
                continue

            if pred_ts.shape[0] != tgt_ts.shape[0]:
                continue

            if n_channels is None:
                n_channels = pred_ts.shape[0]

            if pred_ts.shape[0] != n_channels:
                continue

            pred_covs.append(np.cov(pred_ts.cpu().numpy()))
            tgt_covs.append(np.cov(tgt_ts.cpu().numpy()))

        if not pred_covs or not tgt_covs:
            return

        pred_cov_mean = np.mean(np.stack(pred_covs, axis=0), axis=0)
        tgt_cov_mean = np.mean(np.stack(tgt_covs, axis=0), axis=0)

        vmin = min(pred_cov_mean.min(), tgt_cov_mean.min())
        vmax = max(pred_cov_mean.max(), tgt_cov_mean.max())

        fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharex=True, sharey=True)
        im = axes[0].imshow(pred_cov_mean, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title("Reconstructed covariance (mean)")
        axes[1].imshow(tgt_cov_mean, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Ground truth covariance (mean)")
        for ax in axes:
            ax.grid(False)

        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_cov_summary.png", bbox_inches="tight")
        plt.close(fig)

    def _plot_psd_cov_pair(
        self,
        gen_runs: list[np.ndarray],
        tgt_runs: list[np.ndarray],
        out_dir: Path,
        prefix: str = "gen_target",
    ) -> None:
        """Plot PSD and covariance comparisons for generated vs target runs."""
        gen_res = self._compute_psd_cov(gen_runs)
        tgt_res = self._compute_psd_cov(tgt_runs)
        if gen_res is None or tgt_res is None:
            return

        gen_freqs, gen_psd, gen_cov = gen_res
        tgt_freqs, tgt_psd, tgt_cov = tgt_res

        # PSD side-by-side
        fig_psd, ax_psd = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
        psd_sets = [
            ("Generated PSD", gen_freqs, gen_psd, ax_psd[0]),
            ("Target PSD", tgt_freqs, tgt_psd, ax_psd[1]),
        ]
        ylims = self._psd_ylim(gen_psd, tgt_psd)

        for title, freqs, psd, ax in psd_sets:
            ax.plot(freqs, psd.T, alpha=0.3)
            ax.set_xlabel("Hz")
            ax.set_ylabel("Power")
            ax.set_title(title)
            ax.set_yscale("log")
            if ylims:
                ax.set_ylim(ylims)

        fig_psd.tight_layout()
        fig_psd.savefig(out_dir / f"{prefix}_psd.png", bbox_inches="tight")
        plt.close(fig_psd)

        # Covariance side-by-side
        fig_cov, ax_cov = plt.subplots(1, 2, figsize=(16, 16), sharex=True, sharey=True)
        cov_sets = [
            ("Generated Covariance", gen_cov, ax_cov[0]),
            ("Target Covariance", tgt_cov, ax_cov[1]),
        ]
        vmin = min(float(np.nanmin(gen_cov)), float(np.nanmin(tgt_cov)))
        vmax = max(float(np.nanmax(gen_cov)), float(np.nanmax(tgt_cov)))
        for title, cov_mat, ax in cov_sets:
            im = ax.imshow(cov_mat, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.grid(False)
        fig_cov.colorbar(im, ax=ax_cov, fraction=0.046, pad=0.04)
        fig_cov.savefig(out_dir / f"{prefix}_cov.png", bbox_inches="tight")
        plt.close(fig_cov)

    def plot_psd_cov_pair(
        self,
        gen_runs: list[np.ndarray],
        tgt_runs: list[np.ndarray],
        out_dir: Path,
        prefix: str = "gen_target",
    ) -> None:
        """Public wrapper for PSD/covariance plots."""
        self._plot_psd_cov_pair(gen_runs, tgt_runs, out_dir, prefix=prefix)

    def _psd_ylim(self, *psd_arrays: np.ndarray) -> tuple[float, float] | None:
        """Compute robust y-limits for PSD plots."""
        if not psd_arrays:
            return None
        psd_flat = np.concatenate([arr.flatten() for arr in psd_arrays])
        psd_flat = psd_flat[psd_flat > 0]
        if psd_flat.size == 0:
            return None
        lower = np.percentile(psd_flat, 0.1)
        upper = np.percentile(psd_flat, 99.9)
        if lower > 0 and upper > lower:
            return (lower, upper)
        return None

    def _prepare_plot_pair(
        self,
        generated: np.ndarray,
        target: np.ndarray,
        min_steps: int,
        channel_indices: np.ndarray | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
        """Normalize and align generated/target arrays for plotting."""
        if generated is None or target is None:
            return None

        max_channels = min(generated.shape[0], target.shape[0])
        max_steps = min(generated.shape[1], target.shape[1])
        if max_channels <= 0 or max_steps < min_steps:
            return None

        indices = (
            np.asarray(channel_indices)
            if channel_indices is not None
            else np.arange(max_channels)
        )
        indices = indices[indices < max_channels]
        if indices.size == 0:
            return None

        gen_data = generated[:max_channels, :max_steps][indices]
        tgt_data = target[:max_channels, :max_steps][indices]
        return gen_data, tgt_data, indices, max_steps

    def _plot_context_marker(
        self,
        ax: plt.Axes,
        context_len: int | None,
        max_steps: int,
        *,
        color: str,
        linestyle: str,
        linewidth: float,
        alpha: float,
    ) -> None:
        """Draw a vertical marker at the context boundary."""
        if context_len and context_len < max_steps:
            cutoff = (
                context_len / float(self.sfreq) if self.sfreq else float(context_len)
            )
            ax.axvline(
                cutoff,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )

    def _images_to_channels(self, img: torch.Tensor) -> torch.Tensor:
        """Convert (B,T,H,W) to (B,T,C) by indexing sensor pixels."""
        row_idx = torch.as_tensor(self.val_dataset.row_idx, device=img.device)
        col_idx = torch.as_tensor(self.val_dataset.col_idx, device=img.device)

        img = img.squeeze()

        # gather per time slice
        return img[..., row_idx, col_idx]

    def _plot_examples(
        self,
        examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        out_dir: Path,
    ) -> list[plt.Figure]:
        """Plot model input/output/target triplets and save summary plots."""
        figures: list[plt.Figure] = []
        summary_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, (inp, out, tgt) in enumerate(examples):
            fig, axes = plt.subplots(3, 1, figsize=(100, 12), sharex=True)

            # convert to channels if shape[-2] = shape[-1], i.e. images
            if inp.shape[-2] == inp.shape[-1]:
                inp = self._images_to_channels(inp)
                out = self._images_to_channels(out)
                tgt = self._images_to_channels(tgt)

            self._plot_sequence(inp, axes[0], "Input")

            preds = out
            if torch.is_tensor(out) and out.dim() == tgt.dim() + 1:
                preds = out.argmax(dim=-1)
            self._plot_sequence(preds, axes[1], "Model output")
            self._plot_sequence(tgt, axes[2], "Target")
            fig.suptitle(f"Example {idx}")
            axes[-1].set_xlabel("timestep")

            try:
                self._save_psd_pair(preds, tgt, f"example{idx}", out_dir)
            except Exception as exc:  # pragma: no cover - logging only
                print(f"[eval_runner] Failed to save PSD for example {idx}: {exc}")

            summary_pairs.append((preds, tgt))
            figures.append(fig)

        try:
            self._save_example_summaries(summary_pairs, out_dir)
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[eval_runner] Failed to save example summaries: {exc}")

        return figures

    def plot_examples(
        self,
        examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        out_dir: Path,
    ) -> list[plt.Figure]:
        """Public wrapper for example plots with PSD/covariance summaries."""
        return self._plot_examples(examples, out_dir)

    def _plot_sequence(self, tensor: torch.Tensor, ax: plt.Axes, title: str) -> None:
        """Plot a subset of time series for a single tensor."""
        arr = tensor.detach().cpu()
        if arr.dim() == 1:
            arr = arr.unsqueeze(0)
        if arr.dim() > 2:
            arr = arr.reshape(arr.shape[0], -1)

        max_series = min(arr.shape[0], 6)
        for i in range(max_series):
            ax.plot(arr[i].numpy(), alpha=0.5, linewidth=0.5)
        ax.set_title(title)
        ax.grid(False)

    def plot_prefix_divergence_curves(
        self,
        metrics: dict[str, dict[str, Any]],
        out_dir: Path,
        x: np.ndarray,
        *,
        line_order: list[str] | None = None,
        title: str | None = None,
        base_name: str = "rollout_divergence",
        formats: Iterable[str] = ("png",),
    ) -> None:
        """Plot prefix divergence curves for multiple conditions."""
        if not metrics:
            return

        band_order = ["delta", "theta", "alpha", "beta", "gamma"]
        band_rank = {name: idx for idx, name in enumerate(band_order)}
        banded_prefixes = [
            "coherence_",
            "psd_jsd_",
            "psd_corr_",
            "bandpower_ratio_",
        ]
        non_banded_order = [
            "correlation",
            "covariance",
            "psd_jsd",
            "psd_corr",
            "band_jsd",
            "coherence",
            "stft_magnitude",
            "stft_angle",
            "fft_magnitude",
            "fft_angle",
            "amplitude_kurtosis",
            "amplitude_tail_fraction",
            "dfa_exponent",
            "hurst_exponent",
            "one_over_f_exponent",
            "spatial_connectivity",
        ]

        metric_labels = {
            "psd_jsd": "PSD JSD",
            "covariance": "Covariance distance",
            "band_jsd": "Bandpower JSD",
            "one_over_f_exponent": "1/f exponent distance",
            "spatial_connectivity": "Spatial connectivity distance",
            "amplitude_kurtosis": "Amplitude kurtosis distance",
            "amplitude_tail_fraction": "Amplitude tail fraction distance",
            "psd_corr": "PSD correlation",
            "coherence": "Coherence distance",
        }
        ylabel_map = {
            "psd_corr": "Correlation",
        }
        conditions = line_order or [
            "correct",
            "target_swap",
            "prompt_swap",
            "real_real",
        ]
        color_map = {
            "correct": "#1B6DA8",
            "target_swap": "#E67E22",
            "prompt_swap": "#C0392B",
            "real_real": "#5F5F5F",
        }
        label_map = {
            "correct": "Correct",
            "target_swap": "Target swap",
            "prompt_swap": "Prompt swap",
            "real_real": "Real-real",
        }

        metric_names = list(metrics.keys())
        non_banded = [
            name
            for name in metric_names
            if not any(name.startswith(prefix) for prefix in banded_prefixes)
        ]
        ordered_non_banded = [m for m in non_banded_order if m in non_banded] + [
            m for m in non_banded if m not in non_banded_order
        ]

        banded_groups: list[tuple[str, list[str]]] = []
        for prefix in banded_prefixes:
            names = [m for m in metric_names if m.startswith(prefix)]
            if not names:
                continue

            def _band_key(name: str) -> tuple[int, str]:
                suffix = name.replace(prefix, "")
                if suffix in band_rank:
                    return (0, band_rank[suffix])
                return (1, suffix)

            banded_groups.append((prefix, sorted(names, key=_band_key)))

        cols = 5 if len(metric_names) > 1 else 1
        metric_order: list[str | None] = list(ordered_non_banded)
        if banded_groups:
            remainder = len(metric_order) % cols
            if remainder:
                metric_order.extend([None] * (cols - remainder))
        for _, names in banded_groups:
            if metric_order and len(metric_order) % cols:
                metric_order.extend([None] * (cols - (len(metric_order) % cols)))
            metric_order.extend(names)
            remainder = len(metric_order) % cols
            if remainder:
                metric_order.extend([None] * (cols - remainder))

        rows = int(np.ceil(len(metric_order) / cols)) if metric_order else 0
        xlabel = "Prefix duration (s)" if self.sfreq else "Prefix steps"
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(6.0 * cols, 4.2 * rows),
            squeeze=False,
            constrained_layout=False,
        )
        axes_list = axes.flatten().tolist()

        for ax, metric in zip(axes_list, metric_order):
            if metric is None:
                ax.axis("off")
                continue
            data = metrics.get(metric)
            if data is None:
                ax.axis("off")
                continue
            any_plotted = False
            for condition in conditions:
                cond_data = data.get(condition)
                if cond_data is None:
                    continue
                median = np.asarray(cond_data.get("median", []), dtype=np.float32)
                q25 = np.asarray(cond_data.get("q25", []), dtype=np.float32)
                q75 = np.asarray(cond_data.get("q75", []), dtype=np.float32)
                if median.size == 0 or not np.isfinite(median).any():
                    continue
                x_vals = x[: median.shape[0]]
                color = color_map.get(condition, "C0")
                label = label_map.get(condition, condition)
                ax.plot(
                    x_vals,
                    median[: x_vals.shape[0]],
                    color=color,
                    linewidth=2.2,
                    label=label,
                )
                if q25.size == median.size and q75.size == median.size:
                    ax.fill_between(
                        x_vals,
                        q25[: x_vals.shape[0]],
                        q75[: x_vals.shape[0]],
                        color=color,
                        alpha=0.18,
                        linewidth=0.0,
                    )
                any_plotted = True

            if not any_plotted:
                ax.axis("off")
                continue

            ax.set_title(
                metric_labels.get(metric, metric.replace("_", " ")), loc="left"
            )
            ax.set_xlabel(xlabel)
            ylabel = "Correlation" if metric.startswith("psd_corr") else None
            ax.set_ylabel(ylabel or ylabel_map.get(metric, "Distance"))
            ax.grid(True, axis="y", alpha=0.2)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        for ax in axes_list[len(metric_order) :]:
            ax.axis("off")

        handles, labels = axes_list[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=len(handles),
                frameon=False,
                bbox_to_anchor=(0.5, 1.02),
                fontsize=12,
                columnspacing=1.6,
                handlelength=2.6,
            )
        if title:
            fig.suptitle(title, y=0.995)
        top_margin = 0.84 if rows <= 2 else 0.9
        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            bottom=0.1,
            top=top_margin,
            hspace=0.28,
            wspace=0.24,
        )
        for fmt in formats:
            fmt_str = str(fmt).lower().strip(".")
            if not fmt_str:
                continue
            fig.savefig(out_dir / f"{base_name}.{fmt_str}")
        plt.close(fig)

    def plot_rollout_window_metrics(
        self,
        metrics: dict[str, dict[str, Any]],
        out_dir: Path,
        x: np.ndarray,
        out_of_envelope: dict[str, Any] | None = None,
        summary_curves: list[dict[str, Any]] | None = None,
    ) -> None:
        """Plot sliding-window metric envelopes for generated vs target data."""
        if not metrics:
            return

        metric_order = [
            "spatial_connectivity",
            "one_over_f_exponent",
            "dfa_exponent",
            "hurst_exponent",
            "amplitude_kurtosis",
            "amplitude_tail_fraction",
            "cov_eig_entropy",
            "cov_eig_top_frac",
            "psd_entropy",
            "psd_centroid",
            "token_entropy",
            "token_unique_rate",
            "token_repetition_rate",
        ]
        bandpower_names = [
            k for k in metrics.keys() if k.startswith("bandpower_ratio_")
        ]
        metric_names = [m for m in metric_order if m in metrics] + [
            m for m in metrics.keys() if m not in metric_order
        ]
        metric_names.extend([m for m in bandpower_names if m not in metric_names])

        metric_names = [m for m in metric_names if m in metrics]
        if not metric_names:
            return
        summary_entries: list[dict[str, Any]] = []
        if summary_curves:
            summary_entries.extend(summary_curves)
        if out_of_envelope is not None:
            summary_entries.append(
                {
                    "key": "out_of_envelope_rate",
                    "label": out_of_envelope.get("label", "Out-of-envelope rate"),
                    "curve": out_of_envelope.get("curve"),
                    "y_label": out_of_envelope.get("y_label", "Fraction outside"),
                    "ylim": out_of_envelope.get("ylim", (0.0, 1.0)),
                }
            )
        cleaned_summary: list[dict[str, Any]] = []
        for entry in summary_entries:
            curve = entry.get("curve")
            if curve is None:
                continue
            curve_arr = np.asarray(curve, dtype=np.float32)
            if curve_arr.size == 0 or not np.isfinite(curve_arr).any():
                continue
            cleaned = dict(entry)
            cleaned["curve"] = curve_arr
            cleaned_summary.append(cleaned)

        plot_entries: list[dict[str, Any]] = []
        for entry in cleaned_summary:
            plot_entries.append({"type": "summary", "entry": entry})
        plot_entries.extend([{"type": "metric", "name": name} for name in metric_names])

        total_panels = len(plot_entries)
        cols = 3 if total_panels > 8 else 2 if total_panels > 1 else 1
        rows = int(np.ceil(total_panels / cols))
        xlabel = "Rollout time (s)" if self.sfreq else "Rollout steps"
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(4.8 * cols, 3.4 * rows),
            squeeze=False,
            constrained_layout=False,
        )
        axes_list = axes.flatten().tolist()

        def _title(name: str) -> str:
            if name.startswith("bandpower_ratio_"):
                band = name.replace("bandpower_ratio_", "")
                return f"bandpower ratio ({band})"
            return name.replace("_", " ")

        real_color = "#5F5F5F"
        gen_color = "#1B6DA8"
        summary_colors = ["#D1495B", "#2E8B57", "#6A4C93", "#FF9F1C"]

        summary_count = 0
        for idx, entry in enumerate(plot_entries):
            ax = axes_list[idx]
            row_idx = idx // cols
            show_xlabel = row_idx == (rows - 1)

            if entry["type"] == "summary":
                data = entry["entry"]
                curve = data["curve"]
                label = data.get("label") or data.get("key", "summary")
                y_label = data.get("y_label")
                ylim = data.get("ylim")
                color = summary_colors[summary_count % len(summary_colors)]
                summary_count += 1
                ax.plot(
                    x,
                    curve,
                    color=color,
                    linewidth=2.2,
                    label=label,
                )
                ax.set_title(label, loc="left")
                if y_label:
                    ax.set_ylabel(y_label)
                if ylim is not None:
                    ax.set_ylim(*ylim)
                    ax.yaxis.set_major_locator(MaxNLocator(5))
            else:
                name = entry["name"]
                data = metrics[name]
                gen = data["generated"]
                tgt = data["target"]
                if gen["median"].size == 0 or not np.isfinite(gen["median"]).any():
                    ax.set_title(f"No data for {name}")
                    ax.axis("off")
                    continue

                ax.fill_between(
                    x,
                    tgt["q05"],
                    tgt["q95"],
                    color=real_color,
                    alpha=0.18,
                    label="real 5-95%",
                )
                ax.fill_between(
                    x,
                    tgt["q25"],
                    tgt["q75"],
                    color=real_color,
                    alpha=0.35,
                    label="real 25-75%",
                )
                ax.plot(
                    x,
                    tgt["q50"],
                    color=real_color,
                    linewidth=1.6,
                    linestyle="--",
                    label="real median",
                )
                ax.plot(
                    x,
                    gen["median"],
                    color=gen_color,
                    linewidth=2.1,
                    label="generated median",
                )
                ax.fill_between(
                    x,
                    gen["q25"],
                    gen["q75"],
                    color=gen_color,
                    alpha=0.22,
                    label="generated 25-75%",
                )
                ax.set_title(_title(name), loc="left")
                ax.set_ylabel(name.replace("_", " "))

            if show_xlabel:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            ax.grid(False)

        for ax in axes_list[len(plot_entries) :]:
            ax.axis("off")

        legend_items = [
            Patch(facecolor=real_color, alpha=0.18, label="Real 5–95%"),
            Patch(facecolor=real_color, alpha=0.35, label="Real 25–75%"),
            Line2D(
                [0],
                [0],
                color=real_color,
                linestyle="--",
                linewidth=1.6,
                label="Real median",
            ),
            Line2D([0], [0], color=gen_color, linewidth=2.1, label="Generated median"),
            Patch(facecolor=gen_color, alpha=0.22, label="Generated 25–75%"),
        ]
        for idx, entry in enumerate(cleaned_summary):
            label = entry.get("label") or entry.get("key", "summary")
            color = summary_colors[idx % len(summary_colors)]
            legend_items.append(
                Line2D([0], [0], color=color, linewidth=2.2, label=label)
            )
        fig.legend(
            handles=legend_items,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=12,
            columnspacing=1.6,
            handlelength=2.6,
        )
        top_margin = 0.84 if rows <= 2 else 0.88
        fig.subplots_adjust(top=top_margin, wspace=0.25, hspace=0.35)
        fig.savefig(
            out_dir / "rollout_window_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
