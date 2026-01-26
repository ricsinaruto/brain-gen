"""
Regression tests for brain-gen.

These tests document and guard against specific bugs or edge cases found
during code review. Each test includes a description of the issue.
"""

import numpy as np
import pytest
import torch

from brain_gen.dataset.datasplitter import (
    _chunk_sort_key,
    split_datasets,
)
from brain_gen.training.lightning import LitModel
from tests.models.utils import make_dummy_session


class TestChunkSortingRegression:
    """
    Regression: Chunk files should be loaded in numeric order.

    Issue: If chunks are sorted lexicographically, "10.npy" comes before
    "2.npy". This breaks temporal continuity assumptions.
    """

    def test_numeric_chunks_sorted_correctly(self):
        """Numeric chunk names should be sorted numerically."""
        files = ["10.npy", "2.npy", "1.npy", "100.npy"]
        sorted_files = sorted(files, key=_chunk_sort_key)
        assert sorted_files == ["1.npy", "2.npy", "10.npy", "100.npy"]

    def test_mixed_numeric_and_alpha_chunks(self):
        """Numeric chunks should come before alphabetic chunks."""
        files = ["b.npy", "10.npy", "a.npy", "2.npy"]
        sorted_files = sorted(files, key=_chunk_sort_key)
        # Numeric first (2, 10), then alpha (a, b)
        assert sorted_files == ["2.npy", "10.npy", "a.npy", "b.npy"]


class TestMissingChannelZeroPadRegression:
    """
    Regression: Missing channels must be zero-padded, not skipped.

    Issue: If a session has fewer channels than the canonical layout, the
    missing channels should be filled with zeros, not cause index errors
    or corrupt data alignment.
    """

    def test_missing_channels_are_zeroed(self, tmp_path):
        """Sessions with fewer channels should have zeros for missing."""
        root = tmp_path / "omega"

        # Session 1: full channel set
        full_pos = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        make_dummy_session(
            str(root),
            "sub-001",
            C=3,
            T=40,
            pos_2d=full_pos,
            ch_names=["c0", "c1", "c2"],
        )

        # Session 2: missing one channel
        partial_pos = full_pos[:2]
        partial_data = np.random.randn(2, 40).astype(np.float32)
        make_dummy_session(
            str(root),
            "sub-002",
            data=partial_data,
            pos_2d=partial_pos,
            ch_names=["c0", "c1"],
        )

        # Use split_datasets which properly sets up session_indices
        split = split_datasets(
            str(root),
            example_seconds=0.1,
            overlap_seconds=0.0,
            val_ratio=0.0,
            test_ratio=0.0,
            seed=42,
            dataset_class="ChunkDataset",
        )
        ds = split.train

        # Find an index from sub-002
        partial_idx = next(
            i for i, (_, sess, _, _) in enumerate(ds.indices) if sess == "sub-002"
        )
        x, _ = ds[partial_idx]

        # Should have 3 channels (canonical count)
        assert x.shape[0] == 3

        # The missing channel should be all zeros (or fill_value)
        # Get the dataset key from the index entry
        dataset_key = ds.indices[partial_idx][0]
        present = set(ds.session_indices[(dataset_key, "sub-002")].tolist())
        missing = [i for i in range(3) if i not in present]
        if missing:
            assert torch.all(x[missing[0]] == ds.fill_value)


class TestConfigMutationRegression:
    """
    Regression: configure_optimizers should not mutate user config.

    Issue: If lr_scheduler config dict is modified in place (e.g., via pop()),
    subsequent calls or serialization will fail with missing keys.
    """

    def test_scheduler_config_not_mutated(self):
        """User config should remain intact after configure_optimizers."""
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        class TinyLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.metrics = {}

            def forward(self, out, tgt, model=None):
                return out.sum()

        trainer_cfg = {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "lr_scheduler": {
                "class_name": "StepLR",
                "step_size": 10,
                "gamma": 0.1,
            },
        }

        # Deep copy to check mutation
        original_sched_cfg = dict(trainer_cfg["lr_scheduler"])

        lit = LitModel(
            model_class=TinyModel,
            loss_class=TinyLoss,
            model_cfg={},
            loss_cfg={},
            trainer_cfg=trainer_cfg,
        )

        _ = lit.configure_optimizers()

        # Config should not have been mutated
        assert trainer_cfg["lr_scheduler"] == original_sched_cfg


class TestEmptyDatasetRegression:
    """
    Regression: Empty datasets should raise clear errors.

    Issue: If a dataset root has no valid sessions or chunks, the code
    should fail with a clear error rather than returning empty iterables
    that cause cryptic failures during training.
    """

    def test_empty_root_raises_error(self, tmp_path):
        """Empty dataset root should raise FileNotFoundError."""
        from brain_gen.dataset.datasplitter import _prepare_datasets

        empty_root = tmp_path / "empty"
        empty_root.mkdir()

        with pytest.raises((FileNotFoundError, ValueError)):
            _prepare_datasets(str(empty_root), 0.1, 0.0)


class TestOverlapValidationRegression:
    """
    Regression: overlap_seconds >= example_seconds should be rejected.

    Issue: If overlap is >= example length, windows would overlap completely
    or have negative step, causing infinite loops or nonsensical data.
    """

    def test_overlap_equal_to_example_raises(self, tmp_path):
        """overlap_seconds = example_seconds should raise ValueError."""
        from brain_gen.dataset.datasplitter import _prepare_datasets

        root = tmp_path / "omega"
        make_dummy_session(str(root), "sub-001", C=4, T=100)

        with pytest.raises(ValueError, match="overlap"):
            _prepare_datasets(str(root), example_seconds=0.1, overlap_seconds=0.1)

    def test_overlap_greater_than_example_raises(self, tmp_path):
        """overlap_seconds > example_seconds should raise ValueError."""
        from brain_gen.dataset.datasplitter import _prepare_datasets

        root = tmp_path / "omega"
        make_dummy_session(str(root), "sub-001", C=4, T=100)

        with pytest.raises(ValueError, match="overlap"):
            _prepare_datasets(str(root), example_seconds=0.1, overlap_seconds=0.2)


class TestWindowLengthValidationRegression:
    """
    Regression: Zero or negative window lengths should be rejected.

    Issue: Very small example_seconds values could result in zero-length
    windows after rounding to samples, causing empty tensors.
    """

    def test_zero_window_length_raises(self, tmp_path):
        """example_seconds resulting in 0 samples should raise."""
        from brain_gen.dataset.datasplitter import _prepare_datasets

        root = tmp_path / "omega"
        make_dummy_session(str(root), "sub-001", C=4, T=100, sfreq=200)

        # With sfreq=200, example_seconds=0.001 gives 0.2 samples -> rounds to 0
        with pytest.raises(ValueError, match="zero-length"):
            _prepare_datasets(str(root), example_seconds=0.001, overlap_seconds=0.0)


class TestFreeRunConfigValidationRegression:
    """
    Regression: Invalid free-run configs should be rejected early.

    Issue: Invalid warmup_range or rollout_range values could cause runtime
    errors during training rather than being caught at configuration time.
    """

    def test_zero_warmup_rejected(self):
        """warmup_range with zeros should raise ValueError."""
        from brain_gen.training.lightning import LitModelFreerun
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        class TinyLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.metrics = {}

            def forward(self, out, tgt, model=None):
                return out.sum()

        lit = LitModelFreerun(
            model_class=TinyModel,
            loss_class=TinyLoss,
            model_cfg={},
            loss_cfg={},
            trainer_cfg={"lr": 1e-3, "weight_decay": 0.0},
            free_run_cfg=None,
        )

        with pytest.raises(ValueError):
            lit._prepare_free_run_cfg(
                {"enabled": True, "warmup_range": 0, "rollout_range": 5}
            )

    def test_inverted_range_rejected(self):
        """warmup_range with high < low should raise ValueError."""
        from brain_gen.training.lightning import LitModelFreerun
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        class TinyLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.metrics = {}

            def forward(self, out, tgt, model=None):
                return out.sum()

        lit = LitModelFreerun(
            model_class=TinyModel,
            loss_class=TinyLoss,
            model_cfg={},
            loss_cfg={},
            trainer_cfg={"lr": 1e-3, "weight_decay": 0.0},
            free_run_cfg=None,
        )

        with pytest.raises(ValueError):
            lit._prepare_free_run_cfg(
                {"enabled": True, "warmup_range": [10, 5], "rollout_range": 5}
            )
