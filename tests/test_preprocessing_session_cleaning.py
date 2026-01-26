"""Tests for session cleaning integration in preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from brain_gen.preprocessing import base as base_module


def test_chunk_and_save_runs_session_cleaning(tmp_path):
    """Session cleaning runs after chunking and writes a summary + cleaned chunks."""

    class DummyPreprocessing(base_module.Preprocessing):
        """Minimal preprocessing stub for exercising chunk_and_save."""

        def load(self):
            # No real data loading needed for this unit test.
            self.batch_args = {"files": [], "subjects": []}

        def extract_raw(self, data):
            # Not used in this test, but required by the abstract base class.
            return data

    data_root = tmp_path / "data"
    data_root.mkdir()
    save_root = tmp_path / "preprocessed"

    loader = DummyPreprocessing(
        str(data_root),
        save_path=str(save_root),
        preproc_config={},
        use_dask=False,
        gen_report=False,
        delete_fif=False,
        chunk_seconds=1,
        session_cleaning={
            "enabled": True,
            "window_seconds": 1.0,
            "std_threshold": 1.0,
            "max_bad_pct": 50.0,
            "clip_range": (-1.0, 1.0),
            "max_segment_seconds": 10.0,
            "min_segment_seconds": 0.0,
            "min_first_segment_seconds": 0.0,
            "overwrite": True,
        },
    )

    # Two channels, 25 samples at 10 Hz -> 2 full windows + remainder.
    data = {
        "raw_array": np.zeros((2, 25), dtype=np.float32),
        "sfreq": 10.0,
        "ch_names": ["A", "B"],
        "ch_types": ["mag", "grad"],
        "pos_2d": np.zeros((2, 2), dtype=np.float32),
        "session": "sub-01",
    }

    loader.chunk_and_save(data, "sub-01")

    session_dir = Path(loader.save_folder) / "sub-01"
    summary_path = session_dir / "cleaning_summary.json"

    assert summary_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["discarded"] is False
    assert summary["segments_kept"] == 1

    cleaned_chunks = [path for path in session_dir.iterdir() if path.suffix == ".npy"]
    assert len(cleaned_chunks) == summary["segments_kept"]
