from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


from .rollout_metrics import (
    RolloutMetricsConfig,
    compute_window_summaries,
)
from .rollout_base import RolloutAnalysisBase


class RolloutSlidingWindowAnalysis(RolloutAnalysisBase):
    """Compute and plot sliding-window rollout metrics."""

    def _resolve_window_length(self, params: dict[str, Any]) -> int:
        """Determine the window length (in steps) for sliding-window metrics."""
        default = (
            max(5, int(float(self.sfreq) * 30.0)) if self.sfreq is not None else 30
        )
        candidates = [
            ("window_length_steps", "window_length_s"),
            ("window_length_steps", "window_length_seconds"),
            ("window_metrics_window_steps", "window_metrics_window_seconds"),
            ("divergence_window_steps", "divergence_window_seconds"),
        ]
        for steps_key, seconds_key in candidates:
            if params.get(steps_key) is not None or params.get(seconds_key) is not None:
                return self._resolve_steps_from_params(
                    params, steps_key, seconds_key, default
                )
        return int(default)

    def _resolve_window_stride(self, params: dict[str, Any], window_steps: int) -> int:
        """Determine stride (in steps) between sliding windows."""
        default = int(window_steps)
        candidates = [
            ("stride_steps", "stride_s"),
            ("stride_steps", "stride_seconds"),
            ("divergence_stride_steps", "divergence_stride_seconds"),
        ]
        for steps_key, seconds_key in candidates:
            if params.get(steps_key) is not None or params.get(seconds_key) is not None:
                return self._resolve_steps_from_params(
                    params, steps_key, seconds_key, default
                )
        return default

    def _coerce_list(self, value: Any) -> list[Any]:
        """Return a list from a scalar/list-like value."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        return [value]

    def _plot_rollout_window_metrics(
        self,
        metrics: dict[str, dict[str, Any]],
        out_dir: Path,
        x: np.ndarray,
        out_of_envelope: dict[str, Any] | None = None,
        summary_curves: list[dict[str, Any]] | None = None,
    ) -> None:
        """Delegate sliding-window plotting to the shared plotting helper."""
        self.plotting.plot_rollout_window_metrics(
            metrics,
            out_dir,
            x,
            out_of_envelope=out_of_envelope,
            summary_curves=summary_curves,
        )

    def _out_of_envelope_rate(
        self, generated: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        """Compute fraction of generated runs outside the target envelope."""
        if generated.size == 0 or lower.size == 0 or upper.size == 0:
            return np.array([], dtype=np.float32)
        low = np.asarray(lower, dtype=np.float32)[None, :]
        high = np.asarray(upper, dtype=np.float32)[None, :]
        finite = np.isfinite(generated) & np.isfinite(low) & np.isfinite(high)
        if not np.any(finite):
            return np.full(generated.shape[1], np.nan, dtype=np.float32)
        outside = (generated < low) | (generated > high)
        outside = np.where(finite, outside, False)
        denom = np.sum(finite, axis=0)
        rate = np.divide(
            np.sum(outside, axis=0),
            denom,
            out=np.full_like(denom, np.nan, dtype=np.float32),
            where=denom > 0,
        )
        return rate.astype(np.float32)

    def _rollout_window_metrics(
        self,
        generated: np.ndarray,
        target: np.ndarray,
        window_steps: int,
        stride_steps: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute sliding-window summary metrics for generated vs target runs."""
        if generated.shape != target.shape:
            raise ValueError(
                "Generated and target shapes must match,"
                f"got {generated.shape} vs {target.shape}"
            )
        if generated.ndim != 3:
            raise ValueError("Sliding window metrics expect data shaped as (R, C, T).")

        run_count, _, total_steps = generated.shape
        if run_count == 0:
            return {"generated": {}, "target": {}, "window_starts": []}

        stride = max(1, int(stride_steps) if stride_steps is not None else window_steps)
        window_steps = max(1, int(window_steps))
        if total_steps < window_steps:
            return {"generated": {}, "target": {}, "window_starts": []}

        window_starts = list(range(0, total_steps - window_steps + 1, stride))
        num_windows = len(window_starts)
        if num_windows == 0:
            return {"generated": {}, "target": {}, "window_starts": []}

        params = params or {}
        combined = np.concatenate([generated, target], axis=0)
        cfg = RolloutMetricsConfig.from_params(params, getattr(self, "sfreq", None))
        num_workers, chunk_size, chunks = self._resolve_window_chunks(
            num_windows, params
        )
        print(
            f"[eval_runner] Rollout window metrics: runs={run_count} "
            f"windows={num_windows} workers={num_workers} chunk={chunk_size}"
        )

        results = self._run_window_tasks(
            chunks,
            num_workers,
            compute_window_summaries,
            window_starts,
            window_steps,
            combined,
            cfg,
            self.spatial_weights,
        )

        metric_curves: dict[str, np.ndarray] = {}
        total_runs = combined.shape[0]
        for window_idx_list, chunk_metrics in results:
            for name, values in chunk_metrics.items():
                if name not in metric_curves:
                    metric_curves[name] = np.full(
                        (total_runs, num_windows), np.nan, dtype=np.float32
                    )
                metric_curves[name][:, window_idx_list] = values

        gen_curves = {name: vals[:run_count] for name, vals in metric_curves.items()}
        tgt_curves = {name: vals[run_count:] for name, vals in metric_curves.items()}
        print("[eval_runner] Rollout window metrics computation complete.")
        return {
            "generated": gen_curves,
            "target": tgt_curves,
            "window_starts": window_starts,
        }

    def run(
        self,
        generated: np.ndarray | list[np.ndarray],
        target: np.ndarray | list[np.ndarray],
        out_dir: Path,
    ) -> None:
        """Run sliding-window analysis and emit plots/json."""
        params = dict(self.cfg)
        generated_runs = (
            list(generated) if isinstance(generated, np.ndarray) else list(generated)
        )
        target_runs = list(target) if isinstance(target, np.ndarray) else list(target)

        stacked = self._stack_rollout_runs(generated_runs, target_runs)
        if stacked is None:
            return
        gen_stack, tgt_stack = stacked

        if self.context_steps is not None and self.context_steps < gen_stack.shape[2]:
            gen_stack = gen_stack[:, :, self.context_steps :]
            tgt_stack = tgt_stack[:, :, self.context_steps :]

        window_steps = self._resolve_window_length(params)
        stride_steps = self._resolve_window_stride(params, window_steps)
        curves = self._rollout_window_metrics(
            gen_stack,
            tgt_stack,
            window_steps,
            stride_steps=stride_steps,
            params=params,
        )

        gen_curves = curves.get("generated", {})
        tgt_curves = curves.get("target", {})
        window_starts = np.asarray(curves.get("window_starts", []), dtype=np.float32)
        if not gen_curves or window_starts.size == 0:
            print("[eval_runner] No valid windows for rollout window metrics.")
            return

        x = window_starts + float(window_steps)
        if self.sfreq:
            x = x / float(self.sfreq)

        def _quantiles(values: np.ndarray, qs: list[int]) -> dict[str, np.ndarray]:
            if values.size == 0:
                return {f"q{q:02d}": np.array([]) for q in qs}
            q_vals = np.nanpercentile(values, qs, axis=0)
            return {f"q{q:02d}": q_vals[i] for i, q in enumerate(qs)}

        aggregated: dict[str, dict[str, Any]] = {}
        out_of_envelope_mode = params.get("out_of_envelope_rate")
        if isinstance(out_of_envelope_mode, str):
            out_of_envelope_mode = out_of_envelope_mode.strip().lower()
        if out_of_envelope_mode in (False, None, "none", "off", "false"):
            out_of_envelope_mode = None
        if out_of_envelope_mode is True:
            out_of_envelope_mode = "mean"
        out_of_envelope_by_metric: dict[str, np.ndarray] = {}
        for name, gen_vals in gen_curves.items():
            tgt_vals = tgt_curves.get(name)
            if tgt_vals is None:
                continue
            if not (np.isfinite(gen_vals).any() or np.isfinite(tgt_vals).any()):
                continue
            gen_stats = _quantiles(gen_vals, [25, 50, 75])
            tgt_stats = _quantiles(tgt_vals, [5, 25, 50, 75, 95])
            aggregated[name] = {
                "generated": {
                    "median": gen_stats["q50"],
                    "q25": gen_stats["q25"],
                    "q75": gen_stats["q75"],
                },
                "target": {
                    "q05": tgt_stats["q05"],
                    "q25": tgt_stats["q25"],
                    "q50": tgt_stats["q50"],
                    "q75": tgt_stats["q75"],
                    "q95": tgt_stats["q95"],
                },
            }
            if out_of_envelope_mode is not None:
                rate = self._out_of_envelope_rate(
                    gen_vals, tgt_stats["q05"], tgt_stats["q95"]
                )
                if rate.size > 0:
                    out_of_envelope_by_metric[name] = rate

        if not aggregated:
            print("[eval_runner] No finite metrics for rollout window summaries.")
            return

        diversity_by_metric: dict[str, np.ndarray] = {}
        for name, data in aggregated.items():
            gen_iqr = data["generated"]["q75"] - data["generated"]["q25"]
            tgt_iqr = data["target"]["q75"] - data["target"]["q25"]
            ratio = np.divide(
                gen_iqr,
                tgt_iqr,
                out=np.full_like(gen_iqr, np.nan, dtype=np.float32),
                where=tgt_iqr > 0,
            )
            if ratio.size > 0 and np.isfinite(ratio).any():
                diversity_by_metric[name] = ratio.astype(np.float32)

        diversity_mean: np.ndarray | None = None
        if diversity_by_metric:
            stacked_ratios = np.stack(list(diversity_by_metric.values()), axis=0)
            finite = np.isfinite(stacked_ratios)
            count = np.sum(finite, axis=0)
            summed = np.nansum(stacked_ratios, axis=0)
            diversity_mean = np.full(stacked_ratios.shape[1], np.nan, dtype=np.float32)
            valid = count > 0
            diversity_mean[valid] = (summed[valid] / count[valid]).astype(np.float32)

        out_of_envelope_summary: dict[str, Any] | None = None
        if out_of_envelope_mode is not None and out_of_envelope_by_metric:
            out_of_envelope_payload = {
                "mode": out_of_envelope_mode,
                "per_metric": {
                    name: rate.tolist()
                    for name, rate in out_of_envelope_by_metric.items()
                },
                "mean": None,
            }
            mean_curve = None
            if out_of_envelope_mode in ("mean", "avg", "average"):
                stacked_rates = np.stack(
                    list(out_of_envelope_by_metric.values()), axis=0
                )
                mean_curve = np.nanmean(stacked_rates, axis=0).astype(np.float32)
                out_of_envelope_payload["mean"] = mean_curve.tolist()
            out_of_envelope_summary = out_of_envelope_payload

        payload = {
            "window_steps": int(window_steps),
            "window_seconds": (
                float(window_steps) / float(self.sfreq)
                if self.sfreq is not None
                else None
            ),
            "stride_steps": int(stride_steps),
            "stride_seconds": (
                float(stride_steps) / float(self.sfreq)
                if self.sfreq is not None
                else None
            ),
            "sfreq": self.sfreq,
            "x": x.tolist(),
            "metrics": {
                name: {
                    "generated": {
                        "median": data["generated"]["median"].tolist(),
                        "q25": data["generated"]["q25"].tolist(),
                        "q75": data["generated"]["q75"].tolist(),
                    },
                    "target": {
                        "q05": data["target"]["q05"].tolist(),
                        "q25": data["target"]["q25"].tolist(),
                        "q50": data["target"]["q50"].tolist(),
                        "q75": data["target"]["q75"].tolist(),
                        "q95": data["target"]["q95"].tolist(),
                    },
                }
                for name, data in aggregated.items()
            },
        }
        if out_of_envelope_summary is not None:
            payload["out_of_envelope_rate"] = out_of_envelope_summary
        if diversity_by_metric:
            payload["diversity_proxy"] = {
                "metric": "iqr_ratio",
                "per_metric": {
                    name: curve.tolist() for name, curve in diversity_by_metric.items()
                },
                "mean": diversity_mean.tolist() if diversity_mean is not None else [],
            }

        with open(out_dir / "rollout_window_metrics.json", "w") as f:
            json.dump(payload, f, indent=2)

        if params.get("plot_rollout_window_metrics", True):
            plot_cfg: dict[str, Any] = {}
            plots_cfg = params.get("plots")
            if isinstance(plots_cfg, list) and plots_cfg:
                plot_cfg = dict(plots_cfg[0])
            elif isinstance(plots_cfg, dict):
                plot_cfg = dict(plots_cfg)

            metrics_cfg = plot_cfg.get("metrics")
            metrics_filter = None
            if metrics_cfg is not None:
                metrics_filter = [str(m) for m in self._coerce_list(metrics_cfg)]

            # Filter plotted metrics without altering the JSON payload.
            plot_metrics = aggregated
            plot_metric_names: list[str] | None = None
            if metrics_filter is not None:
                plot_metric_names = [
                    name for name in metrics_filter if name in aggregated
                ]
                plot_metrics = {name: aggregated[name] for name in plot_metric_names}

            summary_curves: list[dict[str, Any]] = []
            if out_of_envelope_summary is not None:
                curve = out_of_envelope_summary.get("mean")
                if curve is not None and plot_metric_names is not None:
                    if plot_metric_names:
                        rates = [
                            out_of_envelope_by_metric[name]
                            for name in plot_metric_names
                            if name in out_of_envelope_by_metric
                        ]
                        if rates:
                            curve = np.nanmean(np.stack(rates, axis=0), axis=0).astype(
                                np.float32
                            )
                        else:
                            curve = None
                    else:
                        curve = None
                if curve is not None:
                    summary_curves.append(
                        {
                            "key": "out_of_envelope_rate",
                            "label": "Out-of-envelope rate (mean)",
                            "curve": np.asarray(curve, dtype=np.float32),
                            "y_label": "Fraction outside",
                            "ylim": (0.0, 1.0),
                        }
                    )
            plot_diversity_mean = diversity_mean
            if plot_diversity_mean is not None and plot_metric_names is not None:
                if plot_metric_names:
                    ratios = [
                        diversity_by_metric[name]
                        for name in plot_metric_names
                        if name in diversity_by_metric
                    ]
                    if ratios:
                        stacked = np.stack(ratios, axis=0)
                        finite = np.isfinite(stacked)
                        count = np.sum(finite, axis=0)
                        summed = np.nansum(stacked, axis=0)
                        plot_diversity_mean = np.full(
                            stacked.shape[1], np.nan, dtype=np.float32
                        )
                        valid = count > 0
                        plot_diversity_mean[valid] = (
                            summed[valid] / count[valid]
                        ).astype(np.float32)
                    else:
                        plot_diversity_mean = None
                else:
                    plot_diversity_mean = None

            if plot_diversity_mean is not None and plot_diversity_mean.size > 0:
                summary_curves.append(
                    {
                        "key": "iqr_ratio_mean",
                        "label": "IQR(gen)/IQR(real) mean",
                        "curve": plot_diversity_mean,
                        "y_label": "IQR ratio",
                    }
                )
            self._plot_rollout_window_metrics(
                plot_metrics, out_dir, x, summary_curves=summary_curves
            )
