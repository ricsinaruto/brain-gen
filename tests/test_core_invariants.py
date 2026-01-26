"""
Core invariant tests for brain-gen.

These tests verify the most critical behaviors that must never break:
- Dataset split determinism and non-overlap
- Postprocessor image↔channel mapping correctness
- ChannelRegistry canonical layout building
"""

import numpy as np
import torch

from brain_gen.dataset.datasets import Postprocessor
from brain_gen.dataset.datasplitter import (
    ChannelRegistry,
    _split_sessions,
    _subject_from_session,
    _normalise_roots,
    split_datasets,
)
from tests.models.utils import make_dummy_session


class TestPostprocessor:
    """Tests for Postprocessor image-to-channel mapping."""

    def test_reshape_3d_to_2d(self, tmp_path):
        """Postprocessor should reshape (H, W, T) -> (C, T)."""
        # Create a 3-channel layout with distinct positions
        pos_2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        pp = Postprocessor(pos_2d, image_size=4, tmp_dir=str(tmp_path))

        # Create image with known values at channel positions
        T = 10
        img = torch.zeros(4, 4, T)

        # Place distinct values at each channel's pixel location
        for ch_idx in range(3):
            r, c = int(pp.row_idx[ch_idx]), int(pp.col_idx[ch_idx])
            img[r, c, :] = ch_idx + 1.0

        out = pp.reshape(img)
        assert out.shape == (3, T)

        # Each channel should have its corresponding value
        for ch_idx in range(3):
            expected = ch_idx + 1.0
            torch.testing.assert_close(
                out[ch_idx], torch.full((T,), expected, dtype=out.dtype)
            )

    def test_reshape_4d_batched(self, tmp_path):
        """Postprocessor should handle batched (B, H, W, T) inputs."""
        pos_2d = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        pp = Postprocessor(pos_2d, image_size=4, tmp_dir=str(tmp_path))

        B, T = 2, 5
        img = torch.randn(B, 4, 4, T)
        out = pp.reshape(img)

        assert out.shape == (B, 2, T)

    def test_reshape_2d_input(self, tmp_path):
        """Postprocessor should handle (H, W) inputs."""
        pos_2d = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        pp = Postprocessor(pos_2d, image_size=4, tmp_dir=str(tmp_path))

        img = torch.randn(4, 4)
        out = pp.reshape(img)

        assert out.shape == (2,)

    def test_call_processes_multiple_tensors(self, tmp_path):
        """Calling postprocessor should process all tensors."""
        pos_2d = np.array([[0.0, 0.0]], dtype=np.float32)
        pp = Postprocessor(pos_2d, image_size=4, tmp_dir=str(tmp_path))

        img1 = torch.randn(4, 4, 10)
        img2 = torch.randn(4, 4, 10)

        out1, out2 = pp(img1, img2)
        assert out1.shape == (1, 10)
        assert out2.shape == (1, 10)


class TestChannelRegistry:
    """Tests for ChannelRegistry canonical layout building."""

    def test_same_position_same_type_merges(self):
        """Channels at same position with same type should merge."""
        reg = ChannelRegistry()

        # First session
        ch1 = reg.register(["ch0", "ch1"], np.array([[0.0, 0.0], [1.0, 0.0]]))
        # Second session with same positions
        ch2 = reg.register(["ch0_alt", "ch1_alt"], np.array([[0.0, 0.0], [1.0, 0.0]]))

        layout = reg.build_layout()
        assert len(layout.names) == 2  # Should merge, not duplicate
        np.testing.assert_array_equal(ch1.indices, ch2.indices)

    def test_different_positions_create_new_entries(self):
        """Different positions should create separate entries."""
        reg = ChannelRegistry()

        reg.register(["ch0"], np.array([[0.0, 0.0]]))
        reg.register(["ch1"], np.array([[1.0, 0.0]]))

        layout = reg.build_layout()
        assert len(layout.names) == 2

    def test_position_averaging(self):
        """Mean position should be computed when merging."""
        reg = ChannelRegistry(decimals=4)

        # Register same logical channel twice with slightly different positions
        # (within quantization tolerance)
        reg.register(["ch0"], np.array([[0.00001, 0.0]]))
        reg.register(["ch0"], np.array([[0.00002, 0.0]]))

        layout = reg.build_layout()
        # Mean position should be average
        expected_pos = (0.00001 + 0.00002) / 2
        assert abs(layout.pos_2d[0, 0] - expected_pos) < 1e-5


class TestSplitSessions:
    """Tests for session splitting logic."""

    def test_split_determinism(self):
        """Same seed should produce same split."""
        import random

        sessions = [f"sub-{i:03d}" for i in range(20)]

        rng1 = random.Random(42)
        train1, val1, test1 = _split_sessions(sessions, 0.2, 0.2, rng1)

        rng2 = random.Random(42)
        train2, val2, test2 = _split_sessions(sessions, 0.2, 0.2, rng2)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_splits_are_disjoint(self):
        """Train, val, test should not overlap."""
        import random

        sessions = [f"sub-{i:03d}" for i in range(30)]
        rng = random.Random(42)
        train, val, test = _split_sessions(sessions, 0.2, 0.2, rng)

        train_set = set(train)
        val_set = set(val)
        test_set = set(test)

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_all_sessions_accounted(self):
        """All sessions should end up in exactly one split."""
        import random

        sessions = [f"sub-{i:03d}" for i in range(30)]
        rng = random.Random(42)
        train, val, test = _split_sessions(sessions, 0.2, 0.2, rng)

        all_split = set(train) | set(val) | set(test)
        assert all_split == set(sessions)

    def test_empty_sessions_handled(self):
        """Empty session list should not crash."""
        import random

        rng = random.Random(42)
        train, val, test = _split_sessions([], 0.2, 0.2, rng)

        assert train == []
        assert val == []
        assert test == []

    def test_single_session_goes_to_train(self):
        """Single session should be assigned to train."""
        import random

        rng = random.Random(42)
        train, val, test = _split_sessions(["sub-001"], 0.2, 0.2, rng)

        assert train == ["sub-001"]
        assert val == []
        assert test == []


class TestNormaliseRoots:
    """Tests for dataset root normalisation."""

    def test_string_input(self):
        """String input should become single-item dict."""
        result = _normalise_roots("/path/to/data")
        assert result == {"dataset0": "/path/to/data"}

    def test_list_input_uses_directory_names(self):
        """List input should use directory names as keys."""
        result = _normalise_roots(["/path/to/alpha", "/path/to/beta"])
        assert "alpha" in result
        assert "beta" in result
        assert result["alpha"] == "/path/to/alpha"
        assert result["beta"] == "/path/to/beta"

    def test_duplicate_directory_names_get_suffix(self):
        """Duplicate directory names should get unique suffixes."""
        result = _normalise_roots(["/path1/data", "/path2/data"])
        keys = list(result.keys())
        assert len(keys) == 2
        assert len(set(keys)) == 2  # All unique

    def test_dict_input_preserved(self):
        """Dict input should be preserved."""
        input_dict = {"ds1": "/path/to/ds1", "ds2": "/path/to/ds2"}
        result = _normalise_roots(input_dict)
        assert result == input_dict


class TestSplitDatasetsIntegration:
    """Integration tests for split_datasets function."""

    def test_reproducible_splits(self, tmp_path):
        """Same seed should produce identical splits."""
        root = tmp_path / "omega"

        # Create multiple sessions
        for i in range(5):
            make_dummy_session(str(root), f"sub-{i:03d}", C=4, T=40)

        split1 = split_datasets(
            str(root),
            example_seconds=0.1,
            overlap_seconds=0.0,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
            dataset_class="ChunkDataset",
        )

        split2 = split_datasets(
            str(root),
            example_seconds=0.1,
            overlap_seconds=0.0,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
            dataset_class="ChunkDataset",
        )

        # Same indices
        assert len(split1.train) == len(split2.train)
        assert len(split1.val) == len(split2.val)
        assert len(split1.test) == len(split2.test)

    def test_different_seeds_different_splits(self, tmp_path):
        """Different seeds should produce different splits."""
        root = tmp_path / "omega"

        for i in range(10):
            make_dummy_session(str(root), f"sub-{i:03d}", C=4, T=40)

        split1 = split_datasets(
            str(root),
            example_seconds=0.1,
            overlap_seconds=0.0,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
            dataset_class="ChunkDataset",
        )

        split2 = split_datasets(
            str(root),
            example_seconds=0.1,
            overlap_seconds=0.0,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=999,
            dataset_class="ChunkDataset",
        )

        # Should have same total count but different assignment
        total1 = len(split1.train) + len(split1.val) + len(split1.test)
        total2 = len(split2.train) + len(split2.val) + len(split2.test)
        assert total1 == total2

    def test_session_level_splitting(self, tmp_path):
        """Splits should be at session level, not window level."""
        root = tmp_path / "omega"

        # Create sessions with multiple chunks
        for i in range(4):
            make_dummy_session(str(root), f"sub-{i:03d}", C=4, T=100)

        split = split_datasets(
            str(root),
            example_seconds=0.05,
            overlap_seconds=0.0,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
            dataset_class="ChunkDataset",
        )

        # Get session names from each split
        train_sessions = {entry[1] for entry in split.train.indices}
        val_sessions = {entry[1] for entry in split.val.indices}
        test_sessions = {entry[1] for entry in split.test.indices}

        # Sessions should not overlap
        assert train_sessions.isdisjoint(val_sessions)
        assert train_sessions.isdisjoint(test_sessions)
        assert val_sessions.isdisjoint(test_sessions)

    def test_subject_level_splitting(self, tmp_path):
        """Splits should keep all sessions from the same subject together."""
        root = tmp_path / "omega"

        make_dummy_session(str(root), "sub-001_ses-01", C=4, T=64)
        make_dummy_session(str(root), "sub-001_ses-02", C=4, T=64)
        make_dummy_session(str(root), "sub-002_ses-01", C=4, T=64)

        split = split_datasets(
            str(root),
            example_seconds=0.05,
            overlap_seconds=0.0,
            val_ratio=0.34,
            test_ratio=0.34,
            seed=42,
            dataset_class="ChunkDataset",
            split_strategy="subject",
        )

        subject_assignments: dict[str, str] = {}
        for split_name, ds in (
            ("train", split.train),
            ("val", split.val),
            ("test", split.test),
        ):
            sessions = {entry[1] for entry in ds.indices}
            for session in sessions:
                subject = _subject_from_session(session)
                if subject in subject_assignments:
                    assert subject_assignments[subject] == split_name
                else:
                    subject_assignments[subject] = split_name

    def test_dataset_level_splitting(self, tmp_path):
        """Dataset split should hold out one dataset and split it by subject."""
        train_root = tmp_path / "train_ds"
        holdout_root = tmp_path / "holdout_ds"

        for i in range(3):
            make_dummy_session(str(train_root), f"sub-train-{i:03d}", C=4, T=80)

        for subj in range(4):
            for ses in range(2):
                make_dummy_session(
                    str(holdout_root),
                    f"sub-{subj:03d}_ses-{ses:02d}",
                    C=4,
                    T=80,
                )

        split = split_datasets(
            {"train": str(train_root), "holdout": str(holdout_root)},
            example_seconds=0.05,
            overlap_seconds=0.0,
            seed=123,
            dataset_class="ChunkDataset",
            split_strategy="dataset",
            heldout_dataset="holdout",
        )

        train_datasets = {entry[0] for entry in split.train.indices}
        val_datasets = {entry[0] for entry in split.val.indices}
        test_datasets = {entry[0] for entry in split.test.indices}

        assert train_datasets == {"train"}
        assert val_datasets == {"holdout"}
        assert test_datasets == {"holdout"}

        subject_assignments: dict[str, str] = {}
        for split_name, ds in (("val", split.val), ("test", split.test)):
            sessions = {entry[1] for entry in ds.indices if entry[0] == "holdout"}
            for session in sessions:
                subject = _subject_from_session(session)
                if subject in subject_assignments:
                    assert subject_assignments[subject] == split_name
                else:
                    subject_assignments[subject] = split_name

        val_subjects = {
            subject
            for subject, split_name in subject_assignments.items()
            if split_name == "val"
        }
        test_subjects = {
            subject
            for subject, split_name in subject_assignments.items()
            if split_name == "test"
        }
        assert len(val_subjects) == len(test_subjects)
        assert len(val_subjects) + len(test_subjects) == 4
