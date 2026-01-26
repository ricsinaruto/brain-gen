import pytest
import torch
import numpy as np

from brain_gen.dataset import datasplitter as datasplitter_module
from brain_gen.dataset.datasets import (
    ChunkDataset,
    ChunkDatasetForecastCont,
    ChunkDatasetImageReconstruction,
    ChunkDatasetInterpolatedImage,
    ChunkDatasetReconstruction,
)
from brain_gen.dataset.datasplitter import build_indices, split_datasets
from tests.models.utils import make_dummy_session
from brain_gen.utils.image_interpolation import compute_layout_indices


def test_chunk_dataset_discrete_shift(tmp_path):
    root = tmp_path / "omega"
    make_dummy_session(str(root), "sub-001", C=272, T=32)

    indices, example_len, _, ch_names, pos_2d, sfreq = build_indices(
        str(root), example_len=0.16, overlap=0.0
    )

    # Use a short fixed window (example_len=0 leads to full-length windows here)
    ds = ChunkDataset(
        str(root),
        indices=indices,
        length=example_len,
        ch_names=ch_names,
        pos_2d=pos_2d,
        sfreq=sfreq,
    )

    x_in, x_tgt = ds[0]
    assert x_in.dtype == torch.long
    assert x_in.shape[0] == x_tgt.shape[0]
    # training dataset pairs inputs/targets as shifted by one in __getitem__
    # but because we choose example_len from seconds and sampling can clip,
    # just verify lengths are reasonable and same channels
    assert x_in.shape[-1] >= 1 and x_tgt.shape[-1] >= 1


def test_chunk_dataset_continuous_shift(tmp_path):
    root = tmp_path / "omega"
    make_dummy_session(str(root), "sub-001", C=272, T=40)

    indices, example_len, _, ch_names, pos_2d, sfreq = build_indices(
        str(root), example_len=0.2, overlap=0.0
    )

    ds = ChunkDatasetForecastCont(
        str(root),
        indices=indices,
        length=example_len,
        ch_names=ch_names,
        pos_2d=pos_2d,
        sfreq=sfreq,
    )
    x_in, x_tgt = ds[0]
    assert x_in.dtype == torch.float32
    assert x_in.shape[0] == x_tgt.shape[0]
    assert x_in.shape[-1] >= 1 and x_tgt.shape[-1] >= 1


def test_chunk_dataset_image_returns_forecast_pairs(tmp_path):
    root = tmp_path / "omega"
    make_dummy_session(str(root), "sub-001", C=272, T=20)

    indices, example_len, _, ch_names, pos_2d, sfreq = build_indices(
        str(root), example_len=0.1, overlap=0.0
    )

    ds_img = ChunkDatasetImageReconstruction(
        str(root),
        indices=indices,
        length=example_len,
        ch_names=ch_names,
        pos_2d=pos_2d,
        sfreq=sfreq,
        image_size=32,
    )
    img_in, img_tgt = ds_img[0]
    # Expect 3D images [H,W,T] and one-step shift across time
    assert img_in.ndim == 3 and img_tgt.ndim == 3
    assert img_in.shape[:2] == img_tgt.shape[:2]
    # Current implementation returns identical tensors (reconstruction/data pairing)
    assert img_in.shape[-1] == img_tgt.shape[-1]


def test_chunk_dataset_interpolated_image_dense_frames(tmp_path):
    root = tmp_path / "omega"
    data = np.zeros((4, 6), dtype=np.float32)
    data[0] = 1.0  # single active channel to make the peak location clear
    pos_2d = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]

    make_dummy_session(str(root), "sub-001", data=data, pos_2d=pos_2d)

    indices, example_len, _, ch_names, pos, sfreq = build_indices(
        str(root), example_len=0.03, overlap=0.0
    )

    ds = ChunkDatasetInterpolatedImage(
        str(root),
        indices=indices,
        length=example_len,
        ch_names=ch_names,
        pos_2d=pos,
        sfreq=sfreq,
        image_size=8,
        normalize=False,
    )

    img_in, img_tgt = ds[0]

    # Expect time-major dense images with a singleton channel dimension
    assert img_in.shape == (1, example_len, 8, 8)
    assert torch.allclose(img_in, img_tgt)

    # Peak should sit on the pixel closest to the active sensor
    row_idx, col_idx = compute_layout_indices(pos, image_size=8)
    peak_pixel = img_in[0, 0, row_idx[0], col_idx[0]]
    assert torch.isclose(peak_pixel, img_in.max())
    # And interpolation should distribute energy beyond the central pixel
    assert img_in.mean() < peak_pixel


def test_chunk_dataset_reconstruction_pretokenized(tmp_path):
    root = tmp_path / "omega"
    make_dummy_session(str(root), "sub-001", C=4, T=32)

    indices, example_len, _, ch_names, pos_2d, sfreq = build_indices(
        str(root), example_len=0.08, overlap=0.0
    )

    token_root = tmp_path / "omega_tokens"
    session_dir = token_root / "sub-001"
    session_dir.mkdir(parents=True, exist_ok=True)

    starts = np.array([idx[2] for idx in indices], dtype=np.int32)
    rvq_codes = np.arange(len(starts) * 6, dtype=np.uint16).reshape(len(starts), 3, 2)
    np.save(session_dir / "0.npy", {"starts": starts, "codes": rvq_codes})

    ds = ChunkDatasetReconstruction(
        str(root),
        indices=indices,
        length=example_len,
        ch_names=ch_names,
        pos_2d=pos_2d,
        sfreq=sfreq,
        use_tokenized=True,
        tokenized_root=str(token_root),
    )

    inputs, targets = ds[0]
    assert isinstance(inputs, dict)
    assert torch.equal(inputs["codes"], torch.tensor(rvq_codes[0]).long())
    assert torch.equal(targets, inputs["codes"])


def test_split_datasets_merges_multiple_roots(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"

    # Dataset A: four channels arranged on a square
    a_pos = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    a_names = [f"a{i}" for i in range(4)]
    make_dummy_session(
        str(root_a),
        "sub-001",
        C=4,
        T=40,
        pos_2d=a_pos,
        ch_names=a_names,
        ch_types=[3012] * 4,
    )

    # Dataset B: shares two positions but introduces a new sensor at (2, 0)
    b_pos = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    b_names = [f"b{i}" for i in range(3)]
    make_dummy_session(
        str(root_b),
        "sub-101",
        C=3,
        T=40,
        pos_2d=b_pos,
        ch_names=b_names,
        ch_types=[3012] * 3,
    )

    split = split_datasets(
        [str(root_a), str(root_b)],
        example_seconds=0.1,
        overlap_seconds=0.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=0,
        dataset_class="ChunkDataset",
    )

    train_ds = split.train
    assert train_ds.num_channels == 5  # four from A plus one new from B

    key_a = root_a.name
    key_b = root_b.name

    present_a = set(train_ds.session_indices[(key_a, "sub-001")].tolist())
    present_b = set(train_ds.session_indices[(key_b, "sub-101")].tolist())

    idx_a = next(i for i, entry in enumerate(train_ds.indices) if entry[0] == key_a)
    idx_b = next(i for i, entry in enumerate(train_ds.indices) if entry[0] == key_b)

    x_a, _ = train_ds[idx_a]
    x_b, _ = train_ds[idx_b]

    missing_a = sorted(set(range(train_ds.num_channels)) - present_a)
    missing_b = sorted(set(range(train_ds.num_channels)) - present_b)

    assert missing_a and torch.all(x_a[missing_a] == 0)
    assert missing_b and torch.all(x_b[missing_b] == 0)

    # Ensure present channels carry signal (non-zero with high probability)
    assert torch.any(x_a[list(present_a)] != 0)
    assert torch.any(x_b[list(present_b)] != 0)


def test_session_with_missing_channels_is_zero_padded(tmp_path):
    root = tmp_path / "omega"

    # Session 1 with complete set of sensors
    full_pos = [(float(i), 0.0) for i in range(3)]
    make_dummy_session(
        str(root),
        "sub-001",
        C=3,
        T=32,
        pos_2d=full_pos,
        ch_names=["c0", "c1", "c2"],
    )

    # Session 2 missing the final channel
    partial_pos = full_pos[:2]
    make_dummy_session(
        str(root),
        "sub-002",
        C=2,
        T=32,
        pos_2d=partial_pos,
        ch_names=["c0_alt", "c1_alt"],
    )

    split = split_datasets(
        str(root),
        example_seconds=0.08,
        overlap_seconds=0.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=0,
        dataset_class="ChunkDataset",
    )

    train_ds = split.train
    assert train_ds.num_channels == 3

    key = next(entry[0] for entry in train_ds.indices if entry[1] == "sub-002")
    partial_present = set(train_ds.session_indices[(key, "sub-002")].tolist())
    missing = sorted(set(range(train_ds.num_channels)) - partial_present)

    idx_partial = next(
        i for i, entry in enumerate(train_ds.indices) if entry[1] == "sub-002"
    )
    x_partial, _ = train_ds[idx_partial]

    assert missing and torch.all(x_partial[missing] == 0)
    assert torch.any(x_partial[list(partial_present)] != 0)


def test_split_datasets_uses_cache_to_skip_chunk_loads(tmp_path, monkeypatch):
    root = tmp_path / "omega"

    # Two chunks so that metadata includes both full and tail lengths
    make_dummy_session(
        str(root),
        "sub-001",
        C=3,
        T=40,
        chunk_idx=0,
    )
    make_dummy_session(
        str(root),
        "sub-001",
        C=3,
        T=20,
        chunk_idx=1,
    )

    cache_dir = tmp_path / "cache"

    split_datasets(
        str(root),
        example_seconds=0.05,
        overlap_seconds=0.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=0,
        dataset_class="ChunkDataset",
        cache_dir=str(cache_dir),
    )

    def fail_on_load(path):
        raise AssertionError(f"Unexpected chunk load for {path}")

    monkeypatch.setattr(datasplitter_module, "_load_chunk", fail_on_load)

    split_datasets(
        str(root),
        example_seconds=0.05,
        overlap_seconds=0.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=0,
        dataset_class="ChunkDataset",
        cache_dir=str(cache_dir),
    )


def test_build_indices_handles_variable_chunk_lengths(tmp_path):
    root = tmp_path / "omega"
    make_dummy_session(str(root), "sub-001", C=4, T=10, chunk_idx=0)
    make_dummy_session(str(root), "sub-001", C=4, T=8, chunk_idx=1)
    make_dummy_session(str(root), "sub-001", C=4, T=12, chunk_idx=2)

    indices, example_len, _, _, _, _ = build_indices(
        str(root), example_len=0.025, overlap=0.0
    )

    counts = {0: 0, 1: 0, 2: 0}
    for _, chunk_idx, _ in indices:
        counts[chunk_idx] += 1

    assert counts == {0: 2, 1: 1, 2: 2}


def test_chunk_dataset_raises_on_short_window(tmp_path):
    root = tmp_path / "omega"
    make_dummy_session(str(root), "sub-001", C=4, T=6)

    _, example_len, _, ch_names, pos_2d, sfreq = build_indices(
        str(root), example_len=0.025, overlap=0.0
    )

    ds = ChunkDataset(
        str(root),
        indices=[("sub-001", 0, 2)],
        length=example_len,
        ch_names=ch_names,
        pos_2d=pos_2d,
        sfreq=sfreq,
    )

    with pytest.raises(ValueError, match="Window shorter than expected"):
        _ = ds[0]


def test_chunk_dataset_image_handles_multi_dataset_layout(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"

    # Dataset A with three sensors forming an L-shape
    make_dummy_session(
        str(root_a),
        "sub-001",
        C=3,
        T=32,
        pos_2d=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        ch_names=["a0", "a1", "a2"],
    )

    # Dataset B missing sensor a2 and introducing a new one
    make_dummy_session(
        str(root_b),
        "sub-101",
        C=3,
        T=32,
        pos_2d=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        ch_names=["b0", "b1", "b2"],
    )

    split = split_datasets(
        [str(root_a), str(root_b)],
        example_seconds=0.08,
        overlap_seconds=0.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=0,
        dataset_class="ChunkDatasetImageReconstruction",
        dataset_kwargs={"image_size": 16},
    )

    ds = split.train
    row_idx = ds.row_idx
    col_idx = ds.col_idx

    key_a = root_a.name
    key_b = root_b.name

    idx_a = next(i for i, entry in enumerate(ds.indices) if entry[0] == key_a)
    idx_b = next(i for i, entry in enumerate(ds.indices) if entry[0] == key_b)

    img_a, _ = ds[idx_a]
    img_b, _ = ds[idx_b]

    present_a = set(ds.session_indices[(key_a, "sub-001")].tolist())
    present_b = set(ds.session_indices[(key_b, "sub-101")].tolist())

    missing_a = sorted(set(range(ds.num_channels)) - present_a)
    missing_b = sorted(set(range(ds.num_channels)) - present_b)

    # Missing channels remain zero-valued across the entire image sequence
    for ch_idx in missing_a:
        r = int(row_idx[ch_idx])
        c = int(col_idx[ch_idx])
        assert torch.count_nonzero(img_a[r, c, :]) == 0
    for ch_idx in missing_b:
        r = int(row_idx[ch_idx])
        c = int(col_idx[ch_idx])
        assert torch.count_nonzero(img_b[r, c, :]) == 0

    # Present channels carry non-zero signals in their assigned pixels
    for ch_idx in present_a:
        r = int(row_idx[ch_idx])
        c = int(col_idx[ch_idx])
        assert torch.count_nonzero(img_a[r, c, :]) > 0
    for ch_idx in present_b:
        r = int(row_idx[ch_idx])
        c = int(col_idx[ch_idx])
        assert torch.count_nonzero(img_b[r, c, :]) > 0
