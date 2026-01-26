from pathlib import Path

from brain_gen.dataset import datasplitter


def test_normalise_roots_deduplicates_sequence_names(tmp_path):
    root_a = tmp_path / "dataset"
    root_b = tmp_path / "dataset" / "nested"
    roots = [str(root_a), str(root_b)]

    normalised = datasplitter._normalise_roots(roots)

    # Uses directory names and guarantees uniqueness with suffixes
    assert set(normalised.keys()) == {"dataset", "nested"}
    assert normalised["dataset"] == str(root_a)
    assert normalised["nested"] == str(root_b)


def test_chunk_sort_key_orders_numeric_before_lexical():
    files = ["10.npy", "2.npy", "1a.npy", "b.npy"]
    ordered = sorted(files, key=datasplitter._chunk_sort_key)
    assert ordered == ["2.npy", "10.npy", "1a.npy", "b.npy"]


def test_list_chunk_files_respects_numeric_order(tmp_path):
    session = tmp_path / "sub-001"
    session.mkdir()
    for name in ["10.npy", "2.npy", "foo.npy", "ignore.txt"]:
        Path(session, name).write_text("stub")

    listed = datasplitter._list_chunk_files(session)
    # Numeric stems first, then other stems alphabetically
    assert listed == ["2.npy", "10.npy", "foo.npy"]
