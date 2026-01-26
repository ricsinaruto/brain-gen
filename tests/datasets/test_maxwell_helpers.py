import pytest

from brain_gen.preprocessing.maxwell import (
    chunk_pairs,
    filter_maxwell_pairs,
    maxwell_output_path,
)


def test_filter_maxwell_pairs_skips_non_fif_and_existing(tmp_path):
    save_folder = tmp_path / "out"
    save_folder.mkdir()

    file_ok = tmp_path / "sub-01_raw.fif"
    file_ok.touch()
    file_skip = tmp_path / "sub-02_raw.ds"
    file_skip.touch()
    file_done = tmp_path / "sub-03_raw.fif"
    file_done.touch()

    maxwell_output_path(save_folder, "sub-03").touch()

    pairs = [
        (str(file_ok), "sub-01"),
        (str(file_skip), "sub-02"),
        (str(file_done), "sub-03"),
    ]

    filtered = filter_maxwell_pairs(pairs, save_folder, skip_done=True)

    assert filtered == [(str(file_ok), "sub-01")]


def test_chunk_pairs_by_num_chunks():
    pairs = [(str(i), f"sub-{i:02d}") for i in range(5)]
    chunks = chunk_pairs(pairs, num_chunks=2)

    assert [len(chunk) for chunk in chunks] == [3, 2]
    assert chunks[0][0] == pairs[0]


def test_chunk_pairs_by_chunk_size():
    pairs = [(str(i), f"sub-{i:02d}") for i in range(5)]
    chunks = chunk_pairs(pairs, chunk_size=2)

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]


def test_chunk_pairs_requires_single_limit():
    pairs = [("file", "sub-01")]

    with pytest.raises(ValueError):
        chunk_pairs(pairs)

    with pytest.raises(ValueError):
        chunk_pairs(pairs, num_chunks=1, chunk_size=1)
