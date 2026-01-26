from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


from .rollout_metrics import (
    RolloutMetricsConfig,
    compute_window_distances,
)

from .rollout_base import RolloutAnalysisBase


class RolloutDivergenceAnalysis(RolloutAnalysisBase):
    """Compute and plot rollout divergence metrics."""

    def _resolve_divergence_window(self, params: dict[str, Any]) -> int:
        """Determine the window (in steps) used for rollout divergence."""
        default = max(5, int(float(self.sfreq) * 0.5)) if self.sfreq is not None else 20
        return self._resolve_steps_from_params(
            params,
            "divergence_window_steps",
            "divergence_window_seconds",
            default,
        )

    def _resolve_timeseries_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return params dict with timeseries divergence settings resolved."""
        merged = dict(params)
        eval_cfg = params.get("evaluation")
        if isinstance(eval_cfg, dict) and "timeseries_divergence" not in merged:
            ts_cfg = eval_cfg.get("timeseries_divergence_params")
            if ts_cfg is None:
                ts_cfg = eval_cfg.get("timeseries_divergence")
            if ts_cfg is not None:
                merged["timeseries_divergence"] = ts_cfg
        return merged

    def _coerce_list(self, value: Any) -> list[Any]:
        """Return a list from a scalar/list-like value."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        return [value]

    def _resolve_prefix_points(
        self, params: dict[str, Any], total_steps: int
    ) -> tuple[list[int], int | None, int | None]:
        """Resolve prefix endpoints for divergence curves."""
        eval_cfg = params.get("evaluation")
        prefix_steps = params.get("prefix_steps")
        prefix_times = params.get("prefix_times_s")
        if prefix_steps is None and isinstance(eval_cfg, dict):
            prefix_steps = eval_cfg.get("prefix_steps")
        if prefix_times is None and isinstance(eval_cfg, dict):
            prefix_times = eval_cfg.get("prefix_times_s") or eval_cfg.get(
                "prefix_times_seconds"
            )

        include_max = params.get("include_max_prefix")
        if include_max is None and isinstance(eval_cfg, dict):
            include_max = eval_cfg.get("include_max_prefix")
        if include_max is None:
            include_max = True

        points: list[int] = []
        prefix_steps_list = self._coerce_list(prefix_steps)
        prefix_times_list = self._coerce_list(prefix_times)
        if prefix_steps_list or prefix_times_list:
            for step in prefix_steps_list:
                if step is None:
                    continue
                points.append(int(round(float(step))))
            for t in prefix_times_list:
                if t is None:
                    continue
                if self.sfreq is None:
                    points.append(int(round(float(t))))
                else:
                    points.append(int(round(float(t) * float(self.sfreq))))

        if not points:
            window_steps = self._resolve_divergence_window(params)
            stride_steps = self._resolve_divergence_stride(params, window_steps)
            window_points = list(range(window_steps, total_steps + 1, stride_steps))
            return window_points, int(window_steps), int(stride_steps)

        if include_max:
            points.append(int(total_steps))

        cleaned = sorted({p for p in points if p > 0})
        cleaned = [p for p in cleaned if p <= total_steps]
        return cleaned, None, None

    def _resolve_control_cfg(
        self, params: dict[str, Any], name: str, *, default_enabled: bool
    ) -> dict[str, Any]:
        """Resolve control configuration into a dict."""
        controls = params.get("controls")
        raw = None
        if isinstance(controls, dict):
            raw = controls.get(name)
        if raw is None:
            raw = params.get(name)
        if raw is None:
            return {"enabled": default_enabled}
        if isinstance(raw, bool):
            return {"enabled": raw}
        if isinstance(raw, dict):
            cfg = dict(raw)
            cfg.setdefault("enabled", True)
            return cfg
        return {"enabled": bool(raw)}

    def _is_similarity_metric(self, metric_name: str) -> bool:
        """Return True when a metric is higher-is-better."""
        name = metric_name.strip().lower()
        return name == "psd_corr" or name.startswith("psd_corr_")

    def _resolve_match_keys(
        self, params: dict[str, Any], control_cfg: dict[str, Any]
    ) -> list[str] | None:
        """Resolve metadata keys used to match swap/real-real pairs."""
        match_keys = control_cfg.get("match_keys")
        if match_keys is None:
            selection = control_cfg.get("selection")
            if isinstance(selection, dict):
                match_keys = selection.get("match_keys")
        if match_keys is None:
            pairing = control_cfg.get("pairing")
            if isinstance(pairing, dict):
                match_keys = pairing.get("match_keys")
        if match_keys is None:
            data_cfg = params.get("data")
            if isinstance(data_cfg, dict):
                task_match = data_cfg.get("task_matching")
                if isinstance(task_match, dict):
                    match_keys = task_match.get("strata_keys")
        if match_keys is None and self.metadata:
            candidate = ["dataset_key", "task_type"]
            if all(
                isinstance(self.metadata[0], dict) and k in self.metadata[0]
                for k in candidate
            ):
                match_keys = candidate
        if match_keys is None:
            return None
        if isinstance(match_keys, str):
            return [match_keys]
        return [str(k) for k in match_keys]

    def _build_metadata_groups(
        self, run_count: int, match_keys: list[str] | None
    ) -> dict[tuple[Any, ...], list[int]]:
        """Group run indices by metadata keys."""
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        if not match_keys or not self.metadata:
            groups[(None,)] = list(range(run_count))
            return groups
        for idx in range(run_count):
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            if not isinstance(meta, dict):
                meta = {}
            key = tuple(meta.get(k) for k in match_keys)
            groups[key].append(idx)
        return groups

    def _select_partner_indices(
        self,
        run_count: int,
        num_pairs: int,
        *,
        rng: np.random.Generator,
        match_keys: list[str] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pairs, context_ids) for swap/real-real controls."""
        if run_count < 2 or num_pairs <= 0:
            return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=int)
        groups = self._build_metadata_groups(run_count, match_keys)
        pairs: list[tuple[int, int]] = []
        context_ids: list[int] = []
        for ctx_idx in range(run_count):
            key = None
            if match_keys and self.metadata and ctx_idx < len(self.metadata):
                meta = self.metadata[ctx_idx]
                if not isinstance(meta, dict):
                    meta = {}
                key = tuple(meta.get(k) for k in match_keys)
            group = groups.get(key) if key in groups else groups.get((None,), [])
            if not group:
                group = list(range(run_count))
            candidates = [idx for idx in group if idx != ctx_idx]
            if not candidates:
                continue
            replace = num_pairs > len(candidates)
            chosen = rng.choice(candidates, size=num_pairs, replace=replace)
            for partner in np.atleast_1d(chosen):
                pairs.append((ctx_idx, int(partner)))
                context_ids.append(ctx_idx)
        if not pairs:
            return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=int)
        return np.array(pairs, dtype=int), np.array(context_ids, dtype=int)

    def _rollout_divergence_curve(
        self,
        generated: np.ndarray,
        target: np.ndarray,
        window_steps: int,
        stride_steps: int | None = None,
        params: dict[str, Any] | None = None,
        window_points: list[int] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Compute rollout divergence curves for generated vs target runs."""
        if generated.shape != target.shape:
            raise ValueError(
                "Generated and target shapes must match,"
                f"got {generated.shape} vs {target.shape}"
            )
        if generated.ndim != 3:
            raise ValueError("Batched divergence expects data shaped as (R, C, T).")

        run_count, _, total_steps = generated.shape
        if run_count == 0:
            return {"gen": {}, "baseline": {}}

        if window_points is None:
            # Build cumulative window endpoints based on stride/window length.
            stride = max(
                1, int(stride_steps) if stride_steps is not None else window_steps
            )
            window_points = list(range(window_steps, total_steps + 1, stride))
        else:
            stride = 0
            window_points = [int(p) for p in window_points]

        num_windows = len(window_points)
        if num_windows == 0:
            return {"gen": {}, "baseline": {}}

        params = params or {}
        params = self._resolve_timeseries_params(params)
        combined = np.concatenate([generated, target], axis=0)
        target_cfg = self._resolve_control_cfg(
            params, "target_swap", default_enabled=False
        )
        prompt_cfg = self._resolve_control_cfg(
            params, "prompt_swap", default_enabled=False
        )
        real_cfg = self._resolve_control_cfg(params, "real_real", default_enabled=True)

        pair_specs: list[dict[str, Any]] = []
        # Correct pairs (generated vs target).
        correct_left = np.arange(run_count, dtype=int)
        correct_right = np.arange(run_count, dtype=int) + run_count
        pair_specs.append(
            {
                "name": "gen",
                "left": correct_left,
                "right": correct_right,
                "context_ids": np.arange(run_count, dtype=int),
            }
        )

        # Target-swap control (generated i vs target j).
        if target_cfg.get("enabled", False):
            target_seed = target_cfg.get("seed")
            selection_cfg = target_cfg.get("selection")
            if isinstance(selection_cfg, dict) and target_seed is None:
                target_seed = selection_cfg.get("seed")
            rng = np.random.default_rng(target_seed)
            num_swaps = int(
                target_cfg.get(
                    "num_swaps_per_context",
                    target_cfg.get("num_pairs_per_context", 1),
                )
            )
            match_keys = self._resolve_match_keys(params, target_cfg)
            pairs, context_ids = self._select_partner_indices(
                run_count, num_swaps, rng=rng, match_keys=match_keys
            )
            if pairs.size > 0:
                pair_specs.append(
                    {
                        "name": "target_swap",
                        "left": pairs[:, 0],
                        "right": pairs[:, 1] + run_count,
                        "context_ids": context_ids,
                        "aggregate": target_cfg.get("aggregate", "mean"),
                    }
                )
            else:
                print("[eval_runner] No valid target-swap pairs; skipping target_swap.")

        # Prompt-swap control (target i vs generated j).
        if prompt_cfg.get("enabled", False):
            prompt_seed = prompt_cfg.get("seed")
            selection_cfg = prompt_cfg.get("selection")
            if isinstance(selection_cfg, dict) and prompt_seed is None:
                prompt_seed = selection_cfg.get("seed")
            rng = np.random.default_rng(prompt_seed)
            num_swaps = int(
                prompt_cfg.get(
                    "num_swaps_per_context",
                    prompt_cfg.get("num_pairs_per_context", 1),
                )
            )
            match_keys = self._resolve_match_keys(params, prompt_cfg)
            pairs, context_ids = self._select_partner_indices(
                run_count, num_swaps, rng=rng, match_keys=match_keys
            )
            if pairs.size > 0:
                pair_specs.append(
                    {
                        "name": "prompt_swap",
                        "left": pairs[:, 1],
                        "right": pairs[:, 0] + run_count,
                        "context_ids": context_ids,
                        "aggregate": prompt_cfg.get("aggregate", "mean"),
                    }
                )
            else:
                print("[eval_runner] No valid prompt-swap pairs; skipping prompt_swap.")

        # Real-real control.
        baseline_valid = run_count > 1
        if real_cfg.get("enabled", baseline_valid):
            real_seed = real_cfg.get("seed")
            pairing_cfg = real_cfg.get("pairing")
            if isinstance(pairing_cfg, dict) and real_seed is None:
                real_seed = pairing_cfg.get("seed")
            rng = np.random.default_rng(real_seed)
            num_pairs = int(
                real_cfg.get(
                    "num_real_real_pairs_per_context",
                    real_cfg.get("num_pairs_per_context", 1),
                )
            )
            match_keys = self._resolve_match_keys(params, real_cfg)
            pairs, context_ids = self._select_partner_indices(
                run_count, num_pairs, rng=rng, match_keys=match_keys
            )
            if pairs.size > 0:
                pair_specs.append(
                    {
                        "name": "baseline",
                        "left": pairs[:, 0] + run_count,
                        "right": pairs[:, 1] + run_count,
                        "context_ids": context_ids,
                        "aggregate": real_cfg.get("aggregate", "mean"),
                    }
                )
            else:
                baseline_valid = False
        else:
            baseline_valid = False

        if not baseline_valid and run_count <= 1:
            print(
                "[eval_runner] Only one run available; baseline distances set to NaN."
            )

        left_indices = np.concatenate([spec["left"] for spec in pair_specs])
        right_indices = np.concatenate([spec["right"] for spec in pair_specs])
        pair_count = left_indices.shape[0]

        cfg = RolloutMetricsConfig.from_params(params, getattr(self, "sfreq", None))
        num_workers, chunk_size, chunks = self._resolve_window_chunks(
            num_windows, params
        )
        print(
            f"[eval_runner] Rollout divergence: runs={run_count} "
            f"windows={num_windows} workers={num_workers} chunk={chunk_size}"
        )

        results = self._run_window_tasks(
            chunks,
            num_workers,
            compute_window_distances,
            window_points,
            combined,
            left_indices,
            right_indices,
            cfg,
            self.spatial_weights,
        )

        distance_curves: dict[str, np.ndarray] = {}
        for window_idx_list, chunk_distances in results:
            for name, values in chunk_distances.items():
                if name not in distance_curves:
                    distance_curves[name] = np.full(
                        (pair_count, num_windows), np.nan, dtype=np.float32
                    )
                distance_curves[name][:, window_idx_list] = values

        cond_curves: dict[str, dict[str, np.ndarray]] = {}
        cond_context_ids: dict[str, np.ndarray] = {}
        offset = 0
        for spec in pair_specs:
            count = len(spec["left"])
            if count == 0:
                continue
            cond_curves[spec["name"]] = {
                name: values[offset : offset + count]
                for name, values in distance_curves.items()
            }
            cond_context_ids[spec["name"]] = np.asarray(
                spec.get("context_ids", np.arange(count)), dtype=int
            )
            offset += count

        if not baseline_valid and "baseline" in cond_curves:
            for name in cond_curves["baseline"]:
                cond_curves["baseline"][name] = np.full_like(
                    cond_curves["baseline"][name], np.nan
                )

        print("[eval_runner] Rollout divergence computation complete.")
        cond_curves["context_ids"] = cond_context_ids
        return cond_curves

    def _aggregate_curve_set(
        self,
        curves: np.ndarray,
        context_ids: np.ndarray | None,
        *,
        reduce: str = "mean",
    ) -> dict[str, Any]:
        """Aggregate curves by context ids and compute summary stats."""
        if curves.size == 0:
            return {
                "runs": [],
                "mean": np.array([], dtype=np.float32),
                "median": np.array([], dtype=np.float32),
                "q25": np.array([], dtype=np.float32),
                "q75": np.array([], dtype=np.float32),
                "std": np.array([], dtype=np.float32),
                "lengths": [],
                "context_ids": [],
            }

        grouped: dict[int, list[np.ndarray]] = defaultdict(list)
        if context_ids is None or len(context_ids) != curves.shape[0]:
            for idx in range(curves.shape[0]):
                grouped[idx].append(curves[idx])
        else:
            for idx, ctx in enumerate(context_ids):
                grouped[int(ctx)].append(curves[idx])

        reduce = str(reduce).lower()

        def _reduce_stack(values: np.ndarray) -> np.ndarray:
            if reduce == "median":
                return np.nanmedian(values, axis=0)
            mask = np.isfinite(values)
            count = np.sum(mask, axis=0)
            summed = np.sum(np.where(mask, values, 0.0), axis=0)
            out = np.full(values.shape[1], np.nan, dtype=np.float32)
            valid = count > 0
            out[valid] = summed[valid] / count[valid]
            return out

        runs: list[np.ndarray] = []
        ctx_ids: list[int] = []
        for ctx in sorted(grouped.keys()):
            stacked = np.stack(grouped[ctx], axis=0)
            if not np.isfinite(stacked).any():
                runs.append(np.full(stacked.shape[1], np.nan, dtype=np.float32))
            else:
                runs.append(_reduce_stack(stacked))
            ctx_ids.append(ctx)

        max_len = max(len(c) for c in runs)
        stacked = np.full((len(runs), max_len), np.nan, dtype=np.float32)
        for idx, curve in enumerate(runs):
            stacked[idx, : curve.shape[0]] = curve
        finite_mask = np.isfinite(stacked)
        valid_cols = np.any(finite_mask, axis=0)
        if not np.any(valid_cols):
            nan = np.full(max_len, np.nan, dtype=np.float32)
            return {
                "runs": runs,
                "mean": nan,
                "median": nan,
                "q25": nan,
                "q75": nan,
                "std": nan,
                "lengths": [int(curve.shape[0]) for curve in runs],
                "context_ids": ctx_ids,
            }
        mean_vals = np.full(max_len, np.nan, dtype=np.float32)
        count = np.sum(finite_mask, axis=0)
        summed = np.sum(np.where(finite_mask, stacked, 0.0), axis=0)
        denom = np.where(count > 0, count, np.nan)
        mean_vals[valid_cols] = summed[valid_cols] / denom[valid_cols]

        valid_data = stacked[:, valid_cols]
        median_vals = np.full(max_len, np.nan, dtype=np.float32)
        q25_vals = np.full(max_len, np.nan, dtype=np.float32)
        q75_vals = np.full(max_len, np.nan, dtype=np.float32)
        std_vals = np.full(max_len, np.nan, dtype=np.float32)
        median_vals[valid_cols] = np.nanmedian(valid_data, axis=0)
        q25_vals[valid_cols] = np.nanpercentile(valid_data, 25, axis=0)
        q75_vals[valid_cols] = np.nanpercentile(valid_data, 75, axis=0)
        std_vals[valid_cols] = np.nanstd(valid_data, axis=0)

        return {
            "runs": runs,
            "mean": mean_vals,
            "median": median_vals,
            "q25": q25_vals,
            "q75": q75_vals,
            "std": std_vals,
            "lengths": [int(curve.shape[0]) for curve in runs],
            "context_ids": ctx_ids,
        }

    def _resolve_stats_cfg(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """Return normalized stats config or None."""
        stats_cfg = params.get("stats")
        if stats_cfg is None:
            eval_cfg = params.get("evaluation")
            if isinstance(eval_cfg, dict):
                stats_cfg = eval_cfg.get("stats")
        if stats_cfg is None:
            return None
        if isinstance(stats_cfg, dict):
            return dict(stats_cfg)
        return None

    def _resolve_stats_horizons(
        self,
        stats_cfg: dict[str, Any],
        x_vals: np.ndarray,
    ) -> list[tuple[int, float]]:
        """Resolve horizon indices for stats."""
        horizons = stats_cfg.get("horizons")
        if horizons is None:
            primary = stats_cfg.get("primary_horizon_s")
            if primary is None:
                primary = stats_cfg.get("primary_horizon")
            horizons = primary
        horizons_list = self._coerce_list(horizons)
        if not horizons_list or (
            len(horizons_list) == 1
            and isinstance(horizons_list[0], str)
            and horizons_list[0].lower() == "all"
        ):
            return [(idx, float(x_vals[idx])) for idx in range(len(x_vals))]

        indices: list[tuple[int, float]] = []
        for h in horizons_list:
            try:
                target = float(h)
            except (TypeError, ValueError):
                continue
            idx = int(np.nanargmin(np.abs(x_vals - target)))
            indices.append((idx, float(x_vals[idx])))
        return indices

    def _resolve_stats_comparisons(
        self, stats_cfg: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Normalize comparison definitions for stats reporting."""
        comparisons = stats_cfg.get("comparisons")
        if comparisons is None:
            return []
        resolved: list[dict[str, Any]] = []
        if (
            isinstance(comparisons, list)
            and comparisons
            and isinstance(comparisons[0], dict)
        ):
            for entry in comparisons:
                name = entry.get("name")
                stat_cfg = entry.get("statistic", {})
                report = stat_cfg.get("report") or entry.get("report")
                delta = stat_cfg.get("delta") or entry.get("delta")
                resolved.append(
                    {
                        "name": name,
                        "paired": entry.get("paired", stats_cfg.get("paired", True)),
                        "report": report or stats_cfg.get("report", []),
                        "delta": delta,
                        "inference": entry.get("inference", stats_cfg.get("inference")),
                    }
                )
            return resolved

        if isinstance(comparisons, list):
            for name in comparisons:
                resolved.append(
                    {
                        "name": str(name),
                        "paired": stats_cfg.get("paired", True),
                        "report": stats_cfg.get("report", []),
                        "delta": None,
                        "inference": stats_cfg.get("inference"),
                    }
                )
        return resolved

    def _comparison_conditions(self, name: str) -> tuple[str, str] | None:
        """Map comparison names to (left, right) condition keys."""
        name = name.strip().lower()
        if "prompt" in name and ("real" in name or "baseline" in name):
            return ("baseline", "prompt_swap")
        if "target" in name and ("real" in name or "baseline" in name):
            return ("baseline", "target_swap")
        if "prompt" in name:
            return ("prompt_swap", "gen")
        if "target" in name:
            return ("target_swap", "gen")
        if "real" in name or "baseline" in name:
            return ("baseline", "gen")
        return None

    def _resolve_delta_direction(self, delta_cfg: Any) -> str | None:
        """Return explicit delta direction override when provided."""
        if delta_cfg is None:
            return None
        token = str(delta_cfg).strip().lower()
        if token in {
            "right_minus_left",
            "right",
            "correct_minus_control",
            "gen_minus_control",
        }:
            return "right"
        if token in {
            "left_minus_right",
            "left",
            "control_minus_correct",
            "control_minus_gen",
        }:
            return "left"
        return None

    def _resolve_delta(
        self,
        metric_name: str,
        left_vals: np.ndarray,
        right_vals: np.ndarray,
        delta_cfg: Any,
    ) -> np.ndarray:
        """Compute deltas with similarity/distance semantics."""
        override = self._resolve_delta_direction(delta_cfg)
        if override == "right":
            return right_vals - left_vals
        if override == "left":
            return left_vals - right_vals
        if self._is_similarity_metric(metric_name):
            return right_vals - left_vals
        return left_vals - right_vals

    def _compute_stats(
        self,
        params: dict[str, Any],
        metrics: dict[str, dict[str, Any]],
        x_vals: np.ndarray,
        out_dir: Path,
    ) -> None:
        """Compute and save stats comparisons for divergence curves."""
        stats_cfg = self._resolve_stats_cfg(params)
        if stats_cfg is None:
            return

        comparisons = self._resolve_stats_comparisons(stats_cfg)
        if not comparisons:
            return

        horizons = self._resolve_stats_horizons(stats_cfg, x_vals)
        if not horizons:
            return

        metrics_filter = stats_cfg.get("metrics")
        if metrics_filter is None:
            eval_cfg = params.get("evaluation")
            if isinstance(eval_cfg, dict):
                dist_cfg = eval_cfg.get("metrics_distances")
                if isinstance(dist_cfg, dict):
                    metrics_filter = dist_cfg.get("primary")
        metrics_list = (
            [str(m) for m in self._coerce_list(metrics_filter)]
            if metrics_filter
            else list(metrics.keys())
        )

        results: list[dict[str, Any]] = []
        for metric_name in metrics_list:
            if metric_name not in metrics:
                continue
            metric_entry = metrics[metric_name]
            for comp in comparisons:
                conds = self._comparison_conditions(comp["name"])
                if conds is None:
                    continue
                left_key, right_key = conds
                left_entry = (
                    metric_entry if left_key == "gen" else metric_entry.get(left_key)
                )
                right_entry = (
                    metric_entry if right_key == "gen" else metric_entry.get(right_key)
                )
                if left_entry is None or right_entry is None:
                    continue
                left_runs = left_entry.get("runs", [])
                right_runs = right_entry.get("runs", [])
                left_ids = left_entry.get("context_ids", [])
                right_ids = right_entry.get("context_ids", [])
                if not left_runs or not right_runs:
                    continue

                right_map = {int(ctx): idx for idx, ctx in enumerate(right_ids or [])}
                left_map = {int(ctx): idx for idx, ctx in enumerate(left_ids or [])}
                if right_map and left_map:
                    common = sorted(set(right_map) & set(left_map))
                    right_idx = [right_map[c] for c in common]
                    left_idx = [left_map[c] for c in common]
                else:
                    n = min(len(right_runs), len(left_runs))
                    right_idx = list(range(n))
                    left_idx = list(range(n))
                if not right_idx or not left_idx:
                    continue

                right_arr = np.stack([right_runs[i] for i in right_idx], axis=0)
                left_arr = np.stack([left_runs[i] for i in left_idx], axis=0)

                for h_idx, horizon_val in horizons:
                    if h_idx >= right_arr.shape[1] or h_idx >= left_arr.shape[1]:
                        continue
                    deltas = self._resolve_delta(
                        metric_name,
                        left_arr[:, h_idx],
                        right_arr[:, h_idx],
                        comp.get("delta"),
                    )
                    deltas = deltas[np.isfinite(deltas)]
                    if deltas.size == 0:
                        continue
                    median_delta = float(np.nanmedian(deltas))
                    prob_improvement = float(np.mean(deltas > 0))

                    entry = {
                        "metric": metric_name,
                        "comparison": comp["name"],
                        "control": left_key,
                        "baseline": right_key,
                        "left": left_key,
                        "right": right_key,
                        "horizon": float(horizon_val),
                        "n": int(deltas.size),
                        "median_delta": median_delta,
                        "prob_improvement": prob_improvement,
                    }

                    inference = comp.get("inference") or stats_cfg.get("inference")
                    if isinstance(inference, dict):
                        ci_cfg = inference.get("ci")
                        if (
                            isinstance(ci_cfg, dict)
                            and ci_cfg.get("method") == "bootstrap"
                        ):
                            n_boot = int(ci_cfg.get("n_boot", 1000))
                            seed = ci_cfg.get("seed", 0)
                            rng = np.random.default_rng(seed)
                            boot = []
                            for _ in range(n_boot):
                                sample = rng.choice(
                                    deltas, size=deltas.size, replace=True
                                )
                                boot.append(float(np.nanmedian(sample)))
                            low, high = np.nanpercentile(boot, [2.5, 97.5])
                            entry["ci"] = {
                                "method": "bootstrap",
                                "n_boot": n_boot,
                                "low": float(low),
                                "high": float(high),
                            }

                        test_cfg = inference.get("test")
                        if (
                            isinstance(test_cfg, dict)
                            and test_cfg.get("type") == "wilcoxon_signed_rank"
                        ):
                            try:
                                from scipy import stats as scipy_stats

                                stat = scipy_stats.wilcoxon(deltas)
                                entry["test"] = {
                                    "type": "wilcoxon_signed_rank",
                                    "statistic": float(stat.statistic),
                                    "p_value": float(stat.pvalue),
                                }
                            except Exception:
                                entry["test"] = {
                                    "type": "wilcoxon_signed_rank",
                                    "statistic": np.nan,
                                    "p_value": np.nan,
                                }

                    results.append(entry)

        payload = {
            "unit_of_replication": stats_cfg.get("unit_of_replication", "context"),
            "horizon_units": "seconds" if self.sfreq else "steps",
            "comparisons": results,
        }

        json_path = out_dir / "rollout_divergence_stats.json"
        json_path.write_text(json.dumps(payload, indent=2))

        if not results:
            return

        header = "| metric | comparison | horizon | n | median_delta | prob_improvement | ci_low | ci_high | p_value |"
        sep = "|---|---|---|---|---|---|---|---|---|"
        lines = [header, sep]
        for entry in results:
            ci = entry.get("ci", {})
            test = entry.get("test", {})
            lines.append(
                "| {metric} | {comparison} | {horizon:.3f} | {n} | {median_delta:.4f} | {prob_improvement:.3f} | {ci_low} | {ci_high} | {p_value} |".format(
                    metric=entry.get("metric", ""),
                    comparison=entry.get("comparison", ""),
                    horizon=entry.get("horizon", np.nan),
                    n=entry.get("n", 0),
                    median_delta=entry.get("median_delta", np.nan),
                    prob_improvement=entry.get("prob_improvement", np.nan),
                    ci_low=(f"{ci.get('low', np.nan):.4f}" if "ci" in entry else ""),
                    ci_high=(f"{ci.get('high', np.nan):.4f}" if "ci" in entry else ""),
                    p_value=(
                        f"{test.get('p_value', np.nan):.3g}" if "test" in entry else ""
                    ),
                )
            )

        md_path = out_dir / "rollout_divergence_stats.md"
        md_path.write_text("\n".join(lines))

    def run(
        self,
        generated: np.ndarray | list[np.ndarray],
        target: np.ndarray | list[np.ndarray],
        out_dir: Path,
    ) -> None:
        """Run divergence analysis and emit plots/json."""
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

        continuation_steps = int(gen_stack.shape[2])
        if continuation_steps <= 0:
            print("[eval_runner] No continuation steps for divergence metrics.")
            return

        prefix_points, window_steps, stride_steps = self._resolve_prefix_points(
            params, continuation_steps
        )
        if not prefix_points:
            print("[eval_runner] No valid prefix points for divergence metrics.")
            return

        window_steps_payload = window_steps
        stride_steps_payload = stride_steps
        window_steps = window_steps or int(prefix_points[0])
        stride_steps = stride_steps or int(prefix_points[0])

        curves = self._rollout_divergence_curve(
            gen_stack,
            tgt_stack,
            window_steps,
            stride_steps=stride_steps,
            params=params,
            window_points=prefix_points,
        )
        cond_context_ids = curves.pop("context_ids", {})
        gen_curves = curves.get("gen", {})
        target_curves = curves.get("target_swap", {})
        prompt_curves = curves.get("prompt_swap", {})
        baseline_curves = curves.get("baseline", {})

        if not gen_curves:
            print("[eval_runner] No valid runs for divergence metric.")
            return

        target_cfg = self._resolve_control_cfg(
            params, "target_swap", default_enabled=False
        )
        prompt_cfg = self._resolve_control_cfg(
            params, "prompt_swap", default_enabled=False
        )
        real_cfg = self._resolve_control_cfg(params, "real_real", default_enabled=True)
        target_reduce = target_cfg.get("aggregate", "mean")
        prompt_reduce = prompt_cfg.get("aggregate", "mean")
        real_reduce = real_cfg.get("aggregate", "mean")

        x_vals = np.asarray(prefix_points, dtype=np.float32)
        if self.sfreq:
            x_vals = x_vals / float(self.sfreq)

        aggregated: dict[str, dict[str, Any]] = {}
        for name, curve_batch in gen_curves.items():
            data = self._aggregate_curve_set(
                curve_batch,
                cond_context_ids.get("gen"),
                reduce="mean",
            )
            data["x"] = x_vals.tolist()
            data["context_ids"] = data.get("context_ids", [])
            target_batch = target_curves.get(name)
            if target_batch is not None:
                data["target_swap"] = self._aggregate_curve_set(
                    target_batch,
                    cond_context_ids.get("target_swap"),
                    reduce=target_reduce,
                )
            prompt_batch = prompt_curves.get(name)
            if prompt_batch is not None:
                data["prompt_swap"] = self._aggregate_curve_set(
                    prompt_batch,
                    cond_context_ids.get("prompt_swap"),
                    reduce=prompt_reduce,
                )
            baseline_batch = baseline_curves.get(name)
            if baseline_batch is not None:
                data["baseline"] = self._aggregate_curve_set(
                    baseline_batch,
                    cond_context_ids.get("baseline"),
                    reduce=real_reduce,
                )
            aggregated[name] = data

        payload = {
            "window_steps": (
                int(window_steps_payload) if window_steps_payload is not None else None
            ),
            "window_seconds": (
                float(window_steps_payload) / float(self.sfreq)
                if window_steps_payload and self.sfreq is not None
                else None
            ),
            "stride_steps": (
                int(stride_steps_payload) if stride_steps_payload is not None else None
            ),
            "stride_seconds": (
                float(stride_steps_payload) / float(self.sfreq)
                if stride_steps_payload and self.sfreq is not None
                else None
            ),
            "prefix_steps": [int(p) for p in prefix_points],
            "prefix_seconds": x_vals.tolist() if self.sfreq else None,
            "sfreq": self.sfreq,
            "metrics": {
                name: {
                    "per_run_lengths": data["lengths"],
                    "mean": data["mean"].tolist(),
                    "median": data["median"].tolist(),
                    "q25": data["q25"].tolist(),
                    "q75": data["q75"].tolist(),
                    "std": data["std"].tolist(),
                    "context_ids": data.get("context_ids", []),
                    "x": data.get("x"),
                    "target_swap": (
                        {
                            "per_run_lengths": data["target_swap"]["lengths"],
                            "mean": data["target_swap"]["mean"].tolist(),
                            "median": data["target_swap"]["median"].tolist(),
                            "q25": data["target_swap"]["q25"].tolist(),
                            "q75": data["target_swap"]["q75"].tolist(),
                            "std": data["target_swap"]["std"].tolist(),
                            "context_ids": data["target_swap"].get("context_ids", []),
                        }
                        if "target_swap" in data
                        else None
                    ),
                    "prompt_swap": (
                        {
                            "per_run_lengths": data["prompt_swap"]["lengths"],
                            "mean": data["prompt_swap"]["mean"].tolist(),
                            "median": data["prompt_swap"]["median"].tolist(),
                            "q25": data["prompt_swap"]["q25"].tolist(),
                            "q75": data["prompt_swap"]["q75"].tolist(),
                            "std": data["prompt_swap"]["std"].tolist(),
                            "context_ids": data["prompt_swap"].get("context_ids", []),
                        }
                        if "prompt_swap" in data
                        else None
                    ),
                    "baseline": (
                        {
                            "per_run_lengths": data["baseline"]["lengths"],
                            "mean": data["baseline"]["mean"].tolist(),
                            "median": data["baseline"]["median"].tolist(),
                            "q25": data["baseline"]["q25"].tolist(),
                            "q75": data["baseline"]["q75"].tolist(),
                            "std": data["baseline"]["std"].tolist(),
                            "context_ids": data["baseline"].get("context_ids", []),
                        }
                        if "baseline" in data
                        else None
                    ),
                }
                for name, data in aggregated.items()
            },
        }

        with open(out_dir / "rollout_divergence.json", "w") as f:
            json.dump(payload, f, indent=2)

        if params.get("plot_rollout_divergence", True):
            plot_cfg: dict[str, Any] = {}
            if isinstance(params.get("plots"), list) and params["plots"]:
                plot_cfg = dict(params["plots"][0])
            elif isinstance(params.get("plots"), dict):
                plot_cfg = dict(params["plots"])

            metrics_cfg = plot_cfg.get("metrics")
            if metrics_cfg is None:
                eval_cfg = params.get("evaluation")
                if isinstance(eval_cfg, dict):
                    dist_cfg = eval_cfg.get("metrics_distances")
                    if isinstance(dist_cfg, dict):
                        metrics_cfg = dist_cfg.get("primary")
            if metrics_cfg:
                metrics_to_plot = [str(m) for m in self._coerce_list(metrics_cfg)]
            else:
                metrics_to_plot = list(aggregated.keys())
            if not metrics_to_plot:
                metrics_to_plot = list(aggregated.keys())

            lines_cfg = plot_cfg.get("lines")
            line_order = (
                [str(l) for l in self._coerce_list(lines_cfg)]
                if lines_cfg
                else ["correct", "target_swap", "prompt_swap", "real_real"]
            )
            save_formats = params.get("plot_formats")
            if save_formats is None:
                io_cfg = params.get("io")
                if isinstance(io_cfg, dict):
                    save_cfg = io_cfg.get("save")
                    if isinstance(save_cfg, dict):
                        save_formats = save_cfg.get("formats")
            formats = [str(fmt) for fmt in self._coerce_list(save_formats)] or ["png"]

            plot_name = plot_cfg.get("name", "rollout_divergence")
            plot_name = str(plot_name)

            plot_metrics: dict[str, dict[str, Any]] = {}
            for metric in metrics_to_plot:
                entry = aggregated.get(metric)
                if entry is None:
                    continue
                plot_metrics[metric] = {
                    "correct": entry,
                    "target_swap": entry.get("target_swap"),
                    "prompt_swap": entry.get("prompt_swap"),
                    "real_real": entry.get("baseline"),
                }

            self.plotting.plot_prefix_divergence_curves(
                plot_metrics,
                out_dir,
                x_vals,
                line_order=line_order,
                title=plot_cfg.get("title"),
                base_name=plot_name,
                formats=formats,
            )

            if plot_name != "rollout_divergence":
                self.plotting.plot_prefix_divergence_curves(
                    plot_metrics,
                    out_dir,
                    x_vals,
                    line_order=line_order,
                    title=plot_cfg.get("title"),
                    base_name="rollout_divergence",
                    formats=formats,
                )

        self._compute_stats(params, aggregated, x_vals, out_dir)
