"""Utilities for summarizing chunked session data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class SessionStats:
    """Summary statistics for a single session."""

    session: str
    num_chunks: int
    num_samples: int
    mean: float
    std: float
    min: float
    max: float
    edge_range: float
    near_min_pct: float
    near_max_pct: float
    near_edge_pct: float


def _chunk_sort_key(path: Path) -> tuple[int, object]:
    """Return a numeric sort key for chunk filenames."""
    stem = path.stem
    if stem.isdigit():
        return 0, int(stem)
    return 1, stem


def list_chunk_files(session_dir: Path) -> list[Path]:
    """List chunk files for a session in numeric order."""
    chunk_files = [
        path
        for path in session_dir.iterdir()
        if path.suffix == ".npy"
        and "events" not in path.name
        and "session_" not in path.name
    ]
    chunk_files.sort(key=_chunk_sort_key)
    return chunk_files


def list_session_dirs(root: Path) -> list[Path]:
    """Return session directories under a root in sorted order."""
    sessions = [
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    sessions.sort(key=lambda path: path.name)
    return sessions


def _load_chunk_data(chunk_path: Path) -> np.ndarray:
    """Load a chunk dict and return the data array."""
    chunk = np.load(chunk_path, allow_pickle=True).item()
    return np.asarray(chunk["data"])


def compute_session_stats(session_dir: Path, edge_range: float) -> SessionStats | None:
    """Compute mean/std/min/max and edge occupancy for a session."""
    chunk_files = list_chunk_files(session_dir)
    if not chunk_files:
        return None

    edge_range = float(abs(edge_range))

    total_samples = 0
    sum_values = 0.0
    sum_squares = 0.0
    min_value = float("inf")
    max_value = float("-inf")

    # First pass: gather mean/std/min/max without holding the full session in memory.
    for chunk_path in chunk_files:
        data = _load_chunk_data(chunk_path)
        total_samples += data.size
        sum_values += float(np.sum(data, dtype=np.float64))
        sum_squares += float(
            np.sum(np.square(data, dtype=np.float64), dtype=np.float64)
        )
        min_value = min(min_value, float(np.min(data)))
        max_value = max(max_value, float(np.max(data)))

    if total_samples == 0:
        return None

    mean_value = sum_values / total_samples
    variance = max(sum_squares / total_samples - mean_value**2, 0.0)
    std_value = float(np.sqrt(variance))

    # Second pass: count samples near the global extrema.
    min_threshold = min_value + edge_range
    max_threshold = max_value - edge_range
    near_min = 0
    near_max = 0
    for chunk_path in chunk_files:
        data = _load_chunk_data(chunk_path)
        near_min += int(np.count_nonzero(data <= min_threshold))
        near_max += int(np.count_nonzero(data >= max_threshold))

    near_min_pct = 100.0 * near_min / total_samples
    near_max_pct = 100.0 * near_max / total_samples
    near_edge_pct = 100.0 * (near_min + near_max) / total_samples

    return SessionStats(
        session=session_dir.name,
        num_chunks=len(chunk_files),
        num_samples=total_samples,
        mean=float(mean_value),
        std=std_value,
        min=min_value,
        max=max_value,
        edge_range=edge_range,
        near_min_pct=near_min_pct,
        near_max_pct=near_max_pct,
        near_edge_pct=near_edge_pct,
    )


def summarize_sessions(
    root: Path,
    edge_range: float,
    *,
    show_progress: bool = True,
) -> list[SessionStats]:
    """Compute stats for every session directory under a root."""
    stats: list[SessionStats] = []
    session_dirs = list_session_dirs(root)
    iterator = (
        tqdm(session_dirs, desc="Sessions", leave=False)
        if show_progress
        else session_dirs
    )
    for session_dir in iterator:
        session_stats = compute_session_stats(session_dir, edge_range=edge_range)
        if session_stats is not None:
            stats.append(session_stats)
    return stats


def _format_float(value: float) -> str:
    """Format floats consistently for Markdown output."""
    return f"{value:.6g}"


def render_session_stats_markdown(
    root: Path,
    stats: Sequence[SessionStats],
    *,
    edge_range: float,
) -> str:
    """Render stats into a Markdown table."""
    lines = [
        "# Session chunk statistics",
        "",
        f"Root: `{root}`",
        f"Edge range: `{edge_range}`",
        "",
        "| Session | Chunks | Samples | Mean | Std | Min | Max | % near min | % near max | % near edges |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    if not stats:
        lines.append("| _No sessions found_ | - | - | - | - | - | - | - | - | - |")
        return "\n".join(lines)

    for entry in stats:
        lines.append(
            "| "
            + " | ".join(
                [
                    entry.session,
                    str(entry.num_chunks),
                    str(entry.num_samples),
                    _format_float(entry.mean),
                    _format_float(entry.std),
                    _format_float(entry.min),
                    _format_float(entry.max),
                    _format_float(entry.near_min_pct),
                    _format_float(entry.near_max_pct),
                    _format_float(entry.near_edge_pct),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def write_session_stats_markdown(
    root: Path,
    stats: Sequence[SessionStats],
    *,
    edge_range: float,
    output_name: str,
) -> Path:
    """Write a Markdown report for session stats and return the path."""
    output_path = root / output_name
    markdown = render_session_stats_markdown(root, stats, edge_range=edge_range)
    output_path.write_text(markdown)
    return output_path


def _parse_edge_range(lines: Sequence[str]) -> float:
    """Extract the edge range from a session stats Markdown header."""
    for line in lines:
        if line.startswith("Edge range:"):
            parts = line.split("`")
            if len(parts) >= 2:
                return float(parts[1])
    return 0.0


def parse_session_stats_markdown(markdown: str) -> list[SessionStats]:
    """Parse the Markdown produced by render_session_stats_markdown."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    edge_range = _parse_edge_range(lines)

    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("| Session |"):
            header_index = index
            break

    if header_index is None:
        return []

    rows: list[SessionStats] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or cells[0].startswith("_No sessions"):
            continue

        if len(cells) < 10:
            continue

        (
            session,
            num_chunks,
            num_samples,
            mean,
            std,
            min_val,
            max_val,
            near_min,
            near_max,
            near_edges,
        ) = cells[:10]

        # The table is assumed to match the exact column order of SessionStats.
        rows.append(
            SessionStats(
                session=session,
                num_chunks=int(num_chunks),
                num_samples=int(num_samples),
                mean=float(mean),
                std=float(std),
                min=float(min_val),
                max=float(max_val),
                edge_range=edge_range,
                near_min_pct=float(near_min),
                near_max_pct=float(near_max),
                near_edge_pct=float(near_edges),
            )
        )

    return rows


def read_session_stats_markdown(path: Path) -> list[SessionStats]:
    """Read a session stats Markdown file and return parsed rows."""
    return parse_session_stats_markdown(path.read_text())


def filter_sessions_by_std(
    stats: Sequence[SessionStats],
    threshold: float,
) -> list[SessionStats]:
    """Return sessions whose std exceeds the provided threshold."""
    return [entry for entry in stats if entry.std > threshold]


def filter_sessions_by_edge_pct(
    stats: Sequence[SessionStats],
    threshold: float,
) -> list[SessionStats]:
    """Return sessions whose near-edge percentage exceeds the threshold."""
    return [entry for entry in stats if entry.near_edge_pct > threshold]
