from __future__ import annotations

from pathlib import Path

import numpy as np

from brain_gen.utils.session_stats import (
    compute_session_stats,
    filter_sessions_by_edge_pct,
    filter_sessions_by_std,
    list_chunk_files,
    read_session_stats_markdown,
)


def _write_chunk(path: Path, data: np.ndarray) -> None:
    np.save(path, {"data": data})


def test_list_chunk_files_orders_numeric(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    _write_chunk(session_dir / "10.npy", np.zeros((1, 1)))
    _write_chunk(session_dir / "2.npy", np.zeros((1, 1)))
    _write_chunk(session_dir / "1.npy", np.zeros((1, 1)))
    _write_chunk(session_dir / "events.npy", np.zeros((1, 1)))
    _write_chunk(session_dir / "session_meta.npy", np.zeros((1, 1)))

    chunk_files = list_chunk_files(session_dir)
    names = [path.name for path in chunk_files]

    assert names == ["1.npy", "2.npy", "10.npy"]


def test_compute_session_stats(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessionA"
    session_dir.mkdir()

    data0 = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    data1 = np.array([[6, 7], [8, 9]], dtype=np.float32)
    _write_chunk(session_dir / "0.npy", data0)
    _write_chunk(session_dir / "1.npy", data1)

    stats = compute_session_stats(session_dir, edge_range=0.5)

    assert stats is not None
    assert stats.session == "sessionA"
    assert stats.num_chunks == 2
    assert stats.num_samples == 10
    assert np.isclose(stats.mean, 4.5)
    assert np.isclose(stats.std, np.sqrt(8.25))
    assert stats.min == 0.0
    assert stats.max == 9.0
    assert np.isclose(stats.near_min_pct, 10.0)
    assert np.isclose(stats.near_max_pct, 10.0)
    assert np.isclose(stats.near_edge_pct, 20.0)


def test_filter_sessions_by_std_from_markdown(tmp_path: Path) -> None:
    markdown = """# Session chunk statistics

Root: `/tmp`
Edge range: `0.01`

| Session | Chunks | Samples | Mean | Std | Min | Max | % near min | % near max | % near edges |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| session_a | 1 | 10 | 0.0 | 1.6 | -1 | 1 | 0.0 | 0.0 | 0.0 |
| session_b | 2 | 20 | 0.0 | 1.4 | -1 | 1 | 0.0 | 0.0 | 0.0 |
"""
    report_path = tmp_path / "session_stats.md"
    report_path.write_text(markdown)

    stats = read_session_stats_markdown(report_path)
    matches = filter_sessions_by_std(stats, threshold=1.5)

    assert [entry.session for entry in matches] == ["session_a"]
    assert stats[0].edge_range == 0.01


def test_filter_sessions_by_edge_pct_from_markdown(tmp_path: Path) -> None:
    markdown = """# Session chunk statistics

Root: `/tmp`
Edge range: `0.01`

| Session | Chunks | Samples | Mean | Std | Min | Max | % near min | % near max | % near edges |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| session_a | 1 | 10 | 0.0 | 1.6 | -1 | 1 | 0.0 | 0.0 | 2.5 |
| session_b | 2 | 20 | 0.0 | 1.4 | -1 | 1 | 0.0 | 0.0 | 1.0 |
"""
    report_path = tmp_path / "session_stats.md"
    report_path.write_text(markdown)

    stats = read_session_stats_markdown(report_path)
    matches = filter_sessions_by_edge_pct(stats, threshold=2.0)

    assert [entry.session for entry in matches] == ["session_a"]
