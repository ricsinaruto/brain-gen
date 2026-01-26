from __future__ import annotations

import numpy as np
import pytest

from brain_gen.utils.session_cleaning import clean_session_array


def _stack_windows(windows: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(windows, axis=1)


def test_clean_session_array_segments_and_clipping() -> None:
    sfreq = 10.0
    window_seconds = 1.0

    window0 = np.zeros((2, 10), dtype=np.float32)
    window1 = np.arange(20, dtype=np.float32).reshape(2, 10)
    window2 = np.ones((2, 10), dtype=np.float32)
    window3 = np.ones((2, 10), dtype=np.float32) * 2

    data = _stack_windows([window0, window1, window2, window3])

    segments, summary = clean_session_array(
        data,
        sfreq,
        window_seconds,
        std_threshold=1.0,
        max_bad_pct=50.0,
        clip_range=(-0.5, 0.5),
        max_segment_seconds=600.0,
        min_segment_seconds=0.0,
    )

    assert not summary.discarded
    assert summary.total_windows == 4
    assert summary.bad_windows == 1
    assert summary.bad_window_pct == pytest.approx(25.0)
    assert summary.clipped_pct == pytest.approx(66.6667, rel=1e-4)
    assert len(segments) == 2
    assert segments[0].shape == (2, 10)
    assert segments[1].shape == (2, 20)
    assert np.allclose(segments[0], 0.0)
    assert np.allclose(segments[1], 0.5)


def test_clean_session_array_splits_long_segments() -> None:
    sfreq = 10.0
    window_seconds = 1.0

    windows = [np.zeros((2, 10), dtype=np.float32) for _ in range(6)]
    data = _stack_windows(windows)

    segments, summary = clean_session_array(
        data,
        sfreq,
        window_seconds,
        std_threshold=0.5,
        max_bad_pct=50.0,
        clip_range=(-1.0, 1.0),
        max_segment_seconds=2.0,
        min_segment_seconds=0.0,
    )

    assert not summary.discarded
    assert len(segments) == 3
    assert all(segment.shape == (2, 20) for segment in segments)


def test_clean_session_array_discards_when_too_many_bad_windows() -> None:
    sfreq = 10.0
    window_seconds = 1.0

    data = np.arange(40, dtype=np.float32).reshape(2, 20)

    segments, summary = clean_session_array(
        data,
        sfreq,
        window_seconds,
        std_threshold=0.1,
        max_bad_pct=50.0,
        clip_range=(-1.0, 1.0),
    )

    assert summary.discarded
    assert summary.discard_reason == "too_many_bad_windows"
    assert segments == []
