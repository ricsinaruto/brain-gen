import inspect
import math
import json
from pathlib import Path
from typing import Any
from einops import rearrange

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..utils.quantizers import mulaw_inv_torch


class TokenSummaryPlotter:
    """Track and plot token-level summary metrics during eval."""

    def __init__(
        self,
        cfg: dict[str, Any] | None,
        model: torch.nn.Module,
        *,
        loss_fn: torch.nn.Module | None = None,
        sfreq: float | None,
    ) -> None:
        """Initialize the token summary plotter."""
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.tokens_per_second = cfg.get("tokens_per_second", cfg.get("tokens_per_sec"))
        self.tokens_per_step = cfg.get("tokens_per_step")
        self.model = model
        self.tokenizer = getattr(model, "tokenizer", None)
        self.sfreq = sfreq
        self.mu = cfg.get("mu", None)
        self.loss_fn = loss_fn
        self.vocab_size = getattr(model, "vocab_size", None)
        self._bits_curves: list[np.ndarray] = []
        self._ppl_curves: list[np.ndarray] = []
        self._unigram_ppl_curves: list[np.ndarray] = []
        self._mse_curves: list[np.ndarray] = []

        self.tokens_per_step = self._infer_tokens_per_step()

    def update(self, outputs: tuple[torch.Tensor, torch.Tensor], *args: Any) -> None:
        """Update curves from model outputs.

        Accepts either the current signature (outputs only) or a legacy
        `(inputs, targets, outputs, batch_idx)` call pattern and extracts the
        `outputs` tuple from positional arguments when provided.
        """
        if args:
            outputs = args[1] if len(args) > 1 else args[0]
        if not self.enabled:
            return

        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
            return

        logits = outputs[0]
        target_tokens = outputs[1]
        if not torch.is_tensor(logits) or not torch.is_tensor(target_tokens):
            return

        if logits.shape[-1] <= 1:
            return

        tokens_per_step = int(self.tokens_per_step)
        usable_len = (logits.shape[1] // tokens_per_step) * tokens_per_step
        start = logits.shape[1] - usable_len
        if usable_len <= 0:
            return
        logits = logits[:, start:]
        target_tokens = target_tokens[:, start:]

        token_loss = None
        if self.loss_fn is not None:
            token_loss = self.loss_fn(
                (logits, target_tokens),
                target_tokens,
                reduction="none",
                model=self.model,
            )

        if token_loss is not None:
            steps = logits.shape[1] // tokens_per_step
            bits_curve = token_loss.float().reshape(
                token_loss.shape[0], steps, tokens_per_step
            )
            bits_curve = bits_curve.mean(dim=-1) / math.log(2.0)
            self._bits_curves.extend(bits_curve.detach().cpu().numpy())

            ppl_curve = 2.0**bits_curve
            self._ppl_curves.extend(ppl_curve.detach().cpu().numpy())

            unigram_bits = self._compute_unigram_bits_curve(
                target_tokens, tokens_per_step
            )
            if unigram_bits is not None:
                self._unigram_ppl_curves.extend(
                    (2.0**unigram_bits).detach().cpu().numpy()
                )

        mse_curve = self._compute_mse_curve(target_tokens, logits, tokens_per_step)
        if mse_curve is not None:
            self._mse_curves.extend(mse_curve.detach().cpu().numpy())

    def finalize(self, out_dir: Path) -> None:
        """Write plots and JSON summaries to disk."""
        if not self.enabled:
            return

        metric_curves: dict[str, tuple[list[np.ndarray], str]] = {}
        if self._bits_curves:
            metric_curves["bits_per_token"] = (self._bits_curves, "Bits per token")
            metric_curves["perplexity"] = (self._ppl_curves, "Perplexity")
        if self._unigram_ppl_curves:
            metric_curves["unigram_perplexity"] = (
                self._unigram_ppl_curves,
                "Unigram perplexity",
            )

        if self.tokens_per_second and self._bits_curves:
            bps_curves = [
                curve * float(self.tokens_per_second) for curve in self._bits_curves
            ]
            metric_curves["bits_per_second"] = (bps_curves, "Bits per second")

        if self._mse_curves:
            metric_curves["decoded_mse"] = (self._mse_curves, "Decoded MSE")

        if not metric_curves:
            return

        context_label = "seconds" if self.sfreq else self._context_label()
        plot_groups = self._build_plot_groups(metric_curves)
        rows = len(plot_groups)
        fig, axes = plt.subplots(rows, 1, figsize=(12, 12 * rows), squeeze=False)

        payload = {
            "context_unit": context_label,
            "tokens_per_second": self.tokens_per_second,
            "tokens_per_step": self.tokens_per_step,
            "vocab_size": self.vocab_size,
            "metrics": {},
        }

        stats_by_name: dict[str, dict[str, np.ndarray | list[int]]] = {}
        for name, (curves, _ylabel) in metric_curves.items():
            stats_by_name[name] = self._aggregate_curves(curves)

        for name in metric_curves:
            stats = stats_by_name[name]
            payload["metrics"][name] = {
                "mean": stats["mean"].tolist(),
                "median": stats["median"].tolist(),
                "q25": stats["q25"].tolist(),
                "q75": stats["q75"].tolist(),
                "std": stats["std"].tolist(),
                "lengths": stats["lengths"],
            }

        for row_idx, group in enumerate(plot_groups):
            ax = axes[row_idx, 0]
            xlabel = None
            for series in group["series"]:
                stats = stats_by_name[series["name"]]
                x, xlabel = self._time_axis(stats["median"].shape[0], group["x_mode"])
                median = stats["median"]
                q25 = stats["q25"]
                q75 = stats["q75"]
                window = self._smoothing_window_steps(x)
                if window is not None:
                    x = self._window_mean(x, window)
                    median = self._window_mean(median, window)
                    q25 = self._window_mean(q25, window)
                    q75 = self._window_mean(q75, window)
                (line,) = ax.plot(x, median, label=series["label"])
                ax.fill_between(
                    x,
                    q25,
                    q75,
                    color=line.get_color(),
                    alpha=0.2,
                )
            ax.set_title(group["title"])
            ax.set_ylabel(group["ylabel"])
            if xlabel is None:
                xlabel = f"Context length ({context_label})"
            ax.set_xlabel(xlabel)
            ax.grid(False)
            ax.legend()

        fig.tight_layout()
        fig.savefig(out_dir / "token_summary.png", bbox_inches="tight")
        plt.close(fig)

        with open(out_dir / "token_summary.json", "w") as f:
            json.dump(payload, f, indent=2)

    def _compute_unigram_bits_curve(
        self, targets: torch.Tensor, tokens_per_step: int
    ) -> torch.Tensor | None:
        """Compute a unigram baseline bits-per-token curve."""
        flat = targets.detach().to(torch.long).reshape(-1).cpu()
        if flat.numel() == 0:
            return None

        vocab_size = self.vocab_size
        if vocab_size is None:
            return None
        vocab_size = int(vocab_size)
        counts = torch.bincount(flat, minlength=vocab_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        probs = probs.clamp_min(1.0e-12)

        nll = -torch.log(probs[flat]).reshape(targets.shape)
        steps = targets.shape[1] // int(tokens_per_step)
        nll = nll.reshape(nll.shape[0], steps, int(tokens_per_step)).mean(dim=-1)
        return nll / math.log(2.0)

    def _compute_mse_curve(
        self, targets: torch.Tensor, logits: torch.Tensor, tokens_per_step: int
    ) -> torch.Tensor | None:
        """Compute decoded MSE curves when a tokenizer is available."""
        if self.tokenizer is None or not hasattr(
            self.tokenizer, "forecast_strip_tokens"
        ):
            return None

        if logits.dim() == 3:
            pred_tokens = logits.argmax(dim=-1)
        else:
            pred_tokens = logits

        steps = int(targets.shape[1]) // int(tokens_per_step)
        if steps <= 0:
            return None

        pred_decoded = self._decode_tokens(pred_tokens, tokens_per_step)
        tgt_decoded = self._decode_tokens(targets, tokens_per_step)
        if pred_decoded is None or tgt_decoded is None:
            return None

        diff = (pred_decoded - tgt_decoded) ** 2

        temporal_reduction = self._temporal_reduction()
        diff = rearrange(diff, "b c (t w) -> b t (c w)", w=int(temporal_reduction))

        return diff.mean(dim=-1)[:, :steps]

    def _decode_tokens(
        self, tokens: torch.Tensor, tokens_per_step: int
    ) -> torch.Tensor | None:
        """Decode token sequences back to continuous values when possible."""
        if not torch.is_tensor(tokens):
            return None

        decoded = self._apply_forecast_strip(tokens.long(), tokens_per_step)
        if decoded is None:
            return None
        if torch.is_floating_point(decoded):
            return decoded.float()
        if self.mu is None:
            return None
        return mulaw_inv_torch(decoded, int(self.mu))

    def _apply_forecast_strip(
        self, tokens: torch.Tensor, tokens_per_step: int
    ) -> torch.Tensor | None:
        """Call forecast_strip_tokens with the right arity."""
        if self.tokenizer is None or not hasattr(
            self.tokenizer, "forecast_strip_tokens"
        ):
            return None

        try:
            signature = inspect.signature(self.tokenizer.forecast_strip_tokens)
        except (TypeError, ValueError):
            signature = None

        if signature is not None and len(signature.parameters) >= 3:
            return self.tokenizer.forecast_strip_tokens(tokens, int(tokens_per_step))
        return self.tokenizer.forecast_strip_tokens(tokens)

    def _align_decoded_for_steps(
        self, decoded: torch.Tensor, steps: int
    ) -> torch.Tensor | None:
        """Return decoded values shaped as (B, steps, features)."""
        if not torch.is_tensor(decoded) or steps <= 0:
            return None

        batch = decoded.shape[0]
        if decoded.dim() == 2:
            if decoded.shape[1] % steps != 0:
                return None
            return decoded.reshape(batch, steps, -1)

        if decoded.dim() == 3:
            if decoded.shape[1] == steps:
                return decoded
            if decoded.shape[2] == steps:
                return decoded.transpose(1, 2)

        flat = decoded.reshape(batch, -1)
        if flat.shape[1] % steps != 0:
            return None
        return flat.reshape(batch, steps, -1)

    def _infer_tokens_per_step(self) -> int | None:
        """Infer tokens-per-step from model or tokenizer metadata."""
        if self.tokens_per_step is not None:
            return int(self.tokens_per_step)

        if hasattr(self.model, "reduced_shape"):
            reduced_shape = getattr(self.model, "reduced_shape")
            if isinstance(reduced_shape, (tuple, list)) and len(reduced_shape) >= 3:
                if all(isinstance(val, (int, np.integer)) for val in reduced_shape[1:]):
                    spatial = int(np.prod(reduced_shape[1:]))
                else:
                    spatial = 0
                if spatial > 0:
                    print(f"inferred tokens_per_step: {spatial}")
                    self.tokens_per_step = spatial
                    return spatial

        if self.tokenizer is not None and hasattr(self.tokenizer, "tokens_per_window"):
            tokens_per_window = getattr(self.tokenizer, "tokens_per_window", None)
            if tokens_per_window:
                self.tokens_per_step = int(tokens_per_window)
                return self.tokens_per_step

        if self.tokens_per_second is not None and self.sfreq:
            temporal_reduction = getattr(self.model, "temporal_reduction", 1)
            steps_per_second = float(self.sfreq) / float(temporal_reduction or 1)
            if steps_per_second > 0:
                val = int(round(float(self.tokens_per_second) / steps_per_second))
                self.tokens_per_step = max(val, 1)
                return self.tokens_per_step

        return None

    def _aggregate_curves(
        self, curves: list[np.ndarray]
    ) -> dict[str, np.ndarray | list[int]]:
        """Aggregate variable-length curves into summary statistics."""
        lengths = [int(curve.shape[0]) for curve in curves]
        max_len = max(lengths)
        data = np.full((len(curves), max_len), np.nan, dtype=np.float32)
        for idx, curve in enumerate(curves):
            data[idx, : curve.shape[0]] = curve
        with np.errstate(all="ignore"):
            mean = np.nanmean(data, axis=0)
            median = np.nanmedian(data, axis=0)
            q25 = np.nanpercentile(data, 25, axis=0)
            q75 = np.nanpercentile(data, 75, axis=0)
            std = np.nanstd(data, axis=0)
        return {
            "mean": mean,
            "median": median,
            "q25": q25,
            "q75": q75,
            "std": std,
            "lengths": lengths,
        }

    def _smoothing_window_steps(self, x: np.ndarray) -> int | None:
        """Return the window size in steps for 1-second averaging."""
        # divide tokens_per_second by tokens_per_step to get the window size in steps
        return int(self.tokens_per_second / self.tokens_per_step)

    def _window_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Average 1D arrays into non-overlapping windows."""
        if window <= 1 or arr.size == 0:
            return arr
        n = arr.shape[0]
        means: list[float] = []
        with np.errstate(all="ignore"):
            for start in range(0, n, window):
                means.append(float(np.nanmean(arr[start : start + window])))
        return np.asarray(means, dtype=np.float32)

    def _build_plot_groups(
        self, metric_curves: dict[str, tuple[list[np.ndarray], str]]
    ) -> list[dict[str, Any]]:
        """Build grouped plot specs for token summary metrics."""
        groups: list[dict[str, Any]] = []
        if "bits_per_token" in metric_curves:
            groups.append(
                {
                    "title": "bits per token",
                    "ylabel": metric_curves["bits_per_token"][1],
                    "x_mode": "tokens",
                    "series": [{"name": "bits_per_token", "label": "median"}],
                }
            )
        if "perplexity" in metric_curves:
            series = [{"name": "perplexity", "label": "perplexity"}]
            if "unigram_perplexity" in metric_curves:
                series.append({"name": "unigram_perplexity", "label": "unigram"})
            groups.append(
                {
                    "title": "perplexity",
                    "ylabel": metric_curves["perplexity"][1],
                    "x_mode": "tokens",
                    "series": series,
                }
            )
        if "bits_per_second" in metric_curves:
            groups.append(
                {
                    "title": "bits per second",
                    "ylabel": metric_curves["bits_per_second"][1],
                    "x_mode": "tokens",
                    "series": [{"name": "bits_per_second", "label": "median"}],
                }
            )
        if "decoded_mse" in metric_curves:
            groups.append(
                {
                    "title": "decoded mse",
                    "ylabel": metric_curves["decoded_mse"][1],
                    "x_mode": "mse",
                    "series": [{"name": "decoded_mse", "label": "median"}],
                }
            )
        return groups

    def _time_axis(self, length: int, mode: str) -> tuple[np.ndarray, str]:
        """Convert curve indices to seconds using sfreq and reduced_shape."""
        x = np.arange(1, length + 1, dtype=np.float32)
        if not self.sfreq or float(self.sfreq) <= 0:
            return x, f"Context length ({self._context_label()})"
        sfreq = float(self.sfreq)
        temporal_reduction = self._temporal_reduction()
        if temporal_reduction is None:
            return x / sfreq, "Time (s)"
        return x * float(temporal_reduction) / sfreq, "Time (s)"

    def _temporal_reduction(self) -> float | None:
        """Infer temporal reduction factor from model shapes."""
        reduced_shape = getattr(self.model, "reduced_shape", None)
        input_shape = getattr(self.model, "input_shape", None)
        if (
            isinstance(input_shape, (tuple, list))
            and isinstance(reduced_shape, (tuple, list))
            and input_shape
            and reduced_shape
        ):
            if isinstance(input_shape[0], (int, np.integer)) and isinstance(
                reduced_shape[0], (int, np.integer)
            ):
                numerator = float(input_shape[0])
                denominator = float(reduced_shape[0])
                if denominator > 0:
                    return numerator / denominator
        temporal_reduction = getattr(self.model, "temporal_reduction", None)
        if isinstance(temporal_reduction, (int, float, np.integer, np.floating)):
            temporal_reduction = float(temporal_reduction)
            if temporal_reduction > 0:
                return temporal_reduction
        return None

    def _context_label(self) -> str:
        """Return the label used for context-length x-axes."""
        if self.tokens_per_step is not None and int(self.tokens_per_step) > 1:
            return "steps"
        return "tokens"
