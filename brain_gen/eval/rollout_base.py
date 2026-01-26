import math
from typing import Any, Callable
import numpy as np
import dask


class RolloutAnalysisBase:
    """Shared helpers for rollout analysis classes."""

    def __init__(
        self,
        cfg: dict[str, Any] | None,
        *,
        sfreq: float | None,
        plotting: Any,
        spatial_weights: np.ndarray | None = None,
    ) -> None:
        """Initialize analysis helpers with shared config and plotting."""
        self.cfg = cfg or {}
        self.sfreq = sfreq
        self.plotting = plotting
        self.spatial_weights = spatial_weights
        self.metadata: list[dict[str, Any]] | None = None
        self.context_steps: int | None = None
        self.total_steps: int | None = None

    def set_rollout_info(
        self,
        *,
        metadata: list[dict[str, Any]] | None = None,
        context_steps: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        """Attach rollout metadata and context/total lengths."""
        self.metadata = metadata
        self.context_steps = context_steps
        self.total_steps = total_steps

    def _filter_metadata_by_indices(self, indices: list[int]) -> None:
        """Filter attached metadata to match the kept run indices."""
        if self.metadata is None:
            return
        filtered: list[dict[str, Any]] = []
        for idx in indices:
            if idx < len(self.metadata):
                filtered.append(self.metadata[idx])
            else:
                # Preserve alignment even if metadata is short.
                filtered.append({})
        self.metadata = filtered

    def _stack_rollout_runs(
        self,
        generated_runs: list[np.ndarray],
        target_runs: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Stack rollout pairs after filtering incompatible shapes."""
        if not generated_runs or not target_runs:
            return None

        pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
        for idx, (gen, tgt) in enumerate(zip(generated_runs, target_runs)):
            if gen.shape != tgt.shape:
                print(
                    "[eval_runner] Skipping divergence: shape mismatch "
                    f"{gen.shape} vs {tgt.shape}"
                )
                continue
            pairs.append((idx, gen, tgt))

        if not pairs:
            print("[eval_runner] No valid runs for divergence metric.")
            return None

        max_len = max(gen.shape[1] for _, gen, _ in pairs)
        length_filtered = [pair for pair in pairs if pair[1].shape[1] == max_len]
        if not length_filtered:
            print("[eval_runner] No runs matched the longest length.")
            return None

        chan_counts = [pair[1].shape[0] for pair in length_filtered]
        common_channels = max(set(chan_counts), key=chan_counts.count)
        filtered_pairs = [
            pair for pair in length_filtered if pair[1].shape[0] == common_channels
        ]
        if not filtered_pairs:
            print("[eval_runner] No runs matched the common channel count.")
            return None
        if len(filtered_pairs) < len(pairs):
            print(
                f"[eval_runner] Dropped {len(pairs) - len(filtered_pairs)} runs "
                "to keep a consistent shape."
            )

        kept_indices = [pair[0] for pair in filtered_pairs]
        self._filter_metadata_by_indices(kept_indices)

        gen_stack = np.stack([pair[1] for pair in filtered_pairs], axis=0)
        tgt_stack = np.stack([pair[2] for pair in filtered_pairs], axis=0)
        return gen_stack, tgt_stack

    def _resolve_window_chunks(
        self, num_windows: int, params: dict[str, Any]
    ) -> tuple[int, int, list[list[int]]]:
        """Resolve worker count + window chunks for rollout metrics."""
        num_workers = int(params.get("rollout_workers", 8))
        num_workers = max(1, num_workers)
        chunk_param = params.get("rollout_window_chunk")
        if chunk_param is None:
            chunk_size = (
                max(1, int(math.ceil(num_windows / num_workers)))
                if num_workers > 1
                else num_windows
            )
        else:
            chunk_size = max(1, int(chunk_param))
        window_indices = list(range(num_windows))
        chunks = [
            window_indices[i : i + chunk_size]
            for i in range(0, num_windows, chunk_size)
        ]
        return num_workers, chunk_size, chunks

    def _run_window_tasks(
        self,
        chunks: list[list[int]],
        num_workers: int,
        compute_fn: Callable[..., tuple[list[int], dict[str, np.ndarray]]],
        *args: Any,
    ) -> list[tuple[list[int], dict[str, np.ndarray]]]:
        """Execute per-window computations in parallel when available."""
        if num_workers > 1 and len(chunks) > 1:
            tasks = [dask.delayed(compute_fn)(chunk, *args) for chunk in chunks]
            results = dask.compute(
                *tasks, scheduler="processes", num_workers=num_workers
            )
            return list(results)

        return [compute_fn(chunk, *args) for chunk in chunks]

    def _resolve_divergence_stride(
        self, params: dict[str, Any], window_steps: int
    ) -> int:
        """Determine stride (in steps) between divergence windows."""
        return self._resolve_steps_from_params(
            params,
            "divergence_stride_steps",
            "divergence_stride_seconds",
            int(window_steps),
        )

    def _resolve_steps_from_params(
        self,
        params: dict[str, Any],
        steps_key: str,
        seconds_key: str,
        default_steps: int,
    ) -> int:
        """Resolve a step count from steps/seconds params with a fallback."""
        steps = params.get(steps_key)
        if steps is not None:
            steps_int = int(steps)
            if steps_int > 0:
                return steps_int

        seconds = params.get(seconds_key)
        if seconds is not None and self.sfreq is not None:
            sec_val = float(seconds)
            if sec_val > 0:
                approx_steps = int(sec_val * float(self.sfreq))
                if approx_steps > 0:
                    return approx_steps

        return int(default_steps)
