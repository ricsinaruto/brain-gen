"""Utilities for window-based cleaning of chunked sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_gen.utils.session_stats import list_chunk_files


@dataclass(frozen=True)
class CleaningSummary:
    """Summary statistics for a cleaned session."""

    total_windows: int
    bad_windows: int
    bad_window_pct: float
    kept_windows: int
    clipped_pct: float
    segments_kept: int
    discarded: bool
    discard_reason: str | None


def load_session_data(session_dir: Path) -> tuple[np.ndarray, dict]:
    """Load chunked session data and concatenate into a (C, T) array."""
    chunk_files = list_chunk_files(session_dir)
    if not chunk_files:
        raise ValueError(f"No chunk files found in {session_dir}")

    chunks: list[np.ndarray] = []
    metadata: dict | None = None
    for chunk_path in chunk_files:
        chunk = np.load(chunk_path, allow_pickle=True).item()
        if metadata is None:
            metadata = {key: value for key, value in chunk.items() if key != "data"}
        chunks.append(np.asarray(chunk["data"]))

    data = np.concatenate(chunks, axis=1)
    return data, metadata or {}


def window_session(data: np.ndarray, window_samples: int) -> np.ndarray:
    """Split a (C, T) session array into non-overlapping windows (N, C, T)."""
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")

    total_samples = data.shape[1]
    num_windows = total_samples // window_samples
    if num_windows <= 0:
        return np.empty((0, data.shape[0], window_samples), dtype=data.dtype)

    trimmed = data[:, : num_windows * window_samples]
    windows = trimmed.reshape(data.shape[0], num_windows, window_samples)
    return windows.transpose(1, 0, 2)


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start, end) positions for contiguous runs in indices."""
    if indices.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    run_start = 0
    for pos in range(1, len(indices)):
        if indices[pos] != indices[pos - 1] + 1:
            runs.append((run_start, pos - 1))
            run_start = pos
    runs.append((run_start, len(indices) - 1))
    return runs


def clean_session_array(
    data: np.ndarray,
    sfreq: float,
    window_seconds: float,
    std_threshold: float,
    max_bad_pct: float,
    clip_range: tuple[float, float],
    *,
    max_segment_seconds: float = 600.0,
    min_segment_seconds: float = 0.0,
    min_first_segment_seconds: float = 0.0,
) -> tuple[list[np.ndarray], CleaningSummary]:
    """Apply window-based cleaning to a (C, T) session array."""
    if data.ndim != 2:
        raise ValueError("Expected data shaped as (C, T)")

    window_samples = int(round(window_seconds * sfreq))
    if window_samples <= 0:
        raise ValueError("window_seconds must produce at least one sample")

    windows = window_session(data, window_samples)
    total_windows = windows.shape[0]
    if total_windows == 0:
        summary = CleaningSummary(
            total_windows=0,
            bad_windows=0,
            bad_window_pct=100.0,
            kept_windows=0,
            clipped_pct=0.0,
            segments_kept=0,
            discarded=True,
            discard_reason="too_short",
        )
        return [], summary

    # Compute std across channels + time in one vectorized call.
    window_stds = np.std(windows, axis=(1, 2))
    bad_mask = window_stds > std_threshold
    bad_windows = int(np.count_nonzero(bad_mask))
    bad_window_pct = 100.0 * bad_windows / total_windows

    if bad_window_pct > max_bad_pct:
        summary = CleaningSummary(
            total_windows=total_windows,
            bad_windows=bad_windows,
            bad_window_pct=bad_window_pct,
            kept_windows=0,
            clipped_pct=0.0,
            segments_kept=0,
            discarded=True,
            discard_reason="too_many_bad_windows",
        )
        return [], summary

    good_mask = ~bad_mask
    kept_windows = windows[good_mask]
    kept_count = kept_windows.shape[0]
    if kept_count == 0:
        summary = CleaningSummary(
            total_windows=total_windows,
            bad_windows=bad_windows,
            bad_window_pct=bad_window_pct,
            kept_windows=0,
            clipped_pct=0.0,
            segments_kept=0,
            discarded=True,
            discard_reason="no_good_windows",
        )
        return [], summary

    clip_min, clip_max = sorted(clip_range)
    clipped_samples = int(
        np.count_nonzero((kept_windows < clip_min) | (kept_windows > clip_max))
    )
    total_samples = int(kept_windows.size)
    clipped_pct = 100.0 * clipped_samples / total_samples if total_samples else 0.0
    kept_windows = np.clip(kept_windows, clip_min, clip_max)

    good_indices = np.flatnonzero(good_mask)
    runs = _contiguous_runs(good_indices)

    window_seconds = window_samples / float(sfreq)
    if max_segment_seconds <= 0:
        max_windows_per_segment = kept_count
    else:
        max_windows_per_segment = max(1, int(max_segment_seconds / window_seconds))

    min_segment_samples = int(max(0.0, min_segment_seconds) * sfreq)
    segments: list[np.ndarray] = []

    for run_start, run_end in runs:
        for start in range(run_start, run_end + 1, max_windows_per_segment):
            end = min(run_end, start + max_windows_per_segment - 1)
            segment_windows = kept_windows[start : end + 1]
            segment = segment_windows.transpose(1, 0, 2).reshape(
                segment_windows.shape[1], -1
            )
            if segment.shape[1] >= min_segment_samples:
                segments.append(segment)

    if min_first_segment_seconds > 0 and segments:
        min_first_samples = int(min_first_segment_seconds * sfreq)
        if segments[0].shape[1] < min_first_samples:
            summary = CleaningSummary(
                total_windows=total_windows,
                bad_windows=bad_windows,
                bad_window_pct=bad_window_pct,
                kept_windows=kept_count,
                clipped_pct=clipped_pct,
                segments_kept=0,
                discarded=True,
                discard_reason="first_segment_too_short",
            )
            return [], summary

    if not segments:
        summary = CleaningSummary(
            total_windows=total_windows,
            bad_windows=bad_windows,
            bad_window_pct=bad_window_pct,
            kept_windows=kept_count,
            clipped_pct=clipped_pct,
            segments_kept=0,
            discarded=True,
            discard_reason="no_segments",
        )
        return [], summary

    summary = CleaningSummary(
        total_windows=total_windows,
        bad_windows=bad_windows,
        bad_window_pct=bad_window_pct,
        kept_windows=kept_count,
        clipped_pct=clipped_pct,
        segments_kept=len(segments),
        discarded=False,
        discard_reason=None,
    )
    return segments, summary


def save_session_segments(
    session_dir: Path,
    segments: list[np.ndarray],
    metadata: dict,
    *,
    overwrite: bool = True,
) -> None:
    """Write cleaned segments to disk using 0..N .npy naming."""
    session_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for chunk_path in session_dir.glob("*.npy"):
            chunk_path.unlink()

    for idx, segment in enumerate(segments):
        chunk = {"data": segment, **metadata}
        np.save(session_dir / f"{idx}.npy", chunk)
