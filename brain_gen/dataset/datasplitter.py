import json
import os
import random
from tqdm import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .datasets import (
    ChunkDataset,
    ChunkDatasetForecastCont,
    ChunkDatasetImage01,
    ChunkDatasetImageQuantized,
    ChunkDatasetJIT,
    ChunkDatasetReconstruction,
    ChunkDatasetMasked,
    ChunkDatasetSubset,
    ChunkDatasetImageReconstruction,
    ChunkDataset3D,
    ChunkDatasetSensor3D,
    ChunkDatasetInterpolatedImage,
    BPEDataset,
)

DATASET_CLASSES = {
    "ChunkDataset": ChunkDataset,
    "ChunkDatasetReconstruction": ChunkDatasetReconstruction,
    "ChunkDatasetForecastCont": ChunkDatasetForecastCont,
    "ChunkDatasetImageQuantized": ChunkDatasetImageQuantized,
    "ChunkDatasetJIT": ChunkDatasetJIT,
    "ChunkDatasetImage01": ChunkDatasetImage01,
    "ChunkDatasetImageReconstruction": ChunkDatasetImageReconstruction,
    "ChunkDatasetInterpolatedImage": ChunkDatasetInterpolatedImage,
    "ChunkDatasetMasked": ChunkDatasetMasked,
    "ChunkDatasetSubset": ChunkDatasetSubset,
    "ChunkDataset3D": ChunkDataset3D,
    "ChunkDatasetSensor3D": ChunkDatasetSensor3D,
    "BPEDataset": BPEDataset,
}


@dataclass
class Split:
    train: ChunkDataset
    val: ChunkDataset
    test: ChunkDataset


@dataclass(frozen=True)
class WindowSpec:
    """Describes a fixed-length window inside a chunked MEG recording."""

    dataset: str
    session: str
    chunk: str
    start: int


@dataclass
class SessionChannels:
    """Channel mapping for a single session relative to the canonical layout."""

    indices: np.ndarray  # maps session channel order → canonical order


@dataclass
class ChannelEntry:
    """Aggregated information about a canonical channel across datasets."""

    pos_sum: np.ndarray
    count: int
    ch_type: Union[int, str]
    pos3d_sum: Optional[np.ndarray] = None
    ori_sum: Optional[np.ndarray] = None
    names: set[str] = field(default_factory=set)

    def update(
        self,
        pos: np.ndarray,
        name: str,
        *,
        pos_3d: Optional[np.ndarray] = None,
        ori_3d: Optional[np.ndarray] = None,
    ) -> None:
        self.pos_sum += pos
        self.count += 1
        self.names.add(name)
        if pos_3d is not None:
            if self.pos3d_sum is None:
                self.pos3d_sum = np.zeros_like(pos_3d, dtype=np.float64)
            self.pos3d_sum += pos_3d
        if ori_3d is not None:
            if self.ori_sum is None:
                self.ori_sum = np.zeros_like(ori_3d, dtype=np.float64)
            self.ori_sum += ori_3d

    @property
    def mean_pos(self) -> np.ndarray:
        return (self.pos_sum / max(self.count, 1)).astype(np.float32)

    @property
    def mean_pos3d(self) -> Optional[np.ndarray]:
        if self.pos3d_sum is None:
            return None
        return (self.pos3d_sum / max(self.count, 1)).astype(np.float32)

    @property
    def mean_ori(self) -> Optional[np.ndarray]:
        if self.ori_sum is None:
            return None
        return (self.ori_sum / max(self.count, 1)).astype(np.float32)

    @property
    def canonical_name(self) -> str:
        if not self.names:
            return "unknown"
        return sorted(self.names)[0]


@dataclass
class ChannelLayout:
    """Canonical channel ordering shared across all datasets."""

    names: List[str]
    types: List[Union[int, str]]
    pos_2d: np.ndarray
    pos_3d: Optional[np.ndarray] = None
    ori_3d: Optional[np.ndarray] = None

    def as_numpy_types(self) -> np.ndarray:
        return np.array(self.types)


@dataclass
class PreparedDatasets:
    """Result of scanning dataset roots prior to splitting."""

    indices: List[WindowSpec]
    layout: ChannelLayout
    session_channels: Dict[Tuple[str, str], SessionChannels]
    dataset_roots: Dict[str, str]
    window_size: int
    overlap: int
    sfreq: float
    sessions_per_dataset: Dict[str, List[str]]


@dataclass
class SessionMetadata:
    chunk_files: List[str]
    chunk_lengths: Optional[List[int]]
    chunk_length: int
    last_chunk_length: int
    ch_names: List[str]
    pos_2d: np.ndarray
    ch_types: Optional[List[Union[int, str]]]
    sfreq: float
    pos_3d: Optional[np.ndarray] = None
    ori_3d: Optional[np.ndarray] = None


def _normalise_roots(
    dataset_root: Union[str, Sequence[str], Mapping[str, str]],
) -> Dict[str, str]:
    if isinstance(dataset_root, str):
        return {"dataset0": dataset_root}

    if isinstance(dataset_root, Mapping):
        return {str(key): str(value) for key, value in dataset_root.items()}

    if isinstance(dataset_root, Sequence):
        normalised: Dict[str, str] = {}
        for idx, root in enumerate(dataset_root):
            key_base = Path(root).name or f"dataset{idx}"
            key = key_base
            # Guarantee unique keys when directory names collide
            suffix = 1
            while key in normalised:
                key = f"{key_base}_{suffix}"
                suffix += 1
            normalised[key] = str(root)
        return normalised

    raise TypeError(
        "dataset_root must be a string, sequence of strings, or mapping of name → path"
    )


def _list_sessions(root_dir: str) -> List[str]:
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_dir}")

    sessions = [
        entry.name
        for entry in root_path.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    sessions.sort()
    return sessions


def _chunk_sort_key(filename: str) -> Tuple[int, Union[int, str]]:
    stem = Path(filename).stem
    if stem.isdigit():
        return 0, int(stem)
    return 1, stem


def _list_chunk_files(session_path: Path) -> List[str]:
    files = [
        f.name
        for f in session_path.iterdir()
        if f.suffix == ".npy" and "events" not in f.name and "session_" not in f.name
    ]
    files.sort(key=_chunk_sort_key)
    return files


def _load_chunk(path: Path) -> dict:
    try:
        chunk = np.load(path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading chunk {path}: {e}")
        raise e
    return chunk


def _load_chunk_lengths(
    session_path: Path,
    chunk_files: Sequence[str],
    *,
    first_chunk: Optional[dict] = None,
) -> List[int]:
    """Load per-chunk sample counts for a session."""
    lengths: List[int] = []
    for idx, chunk_name in enumerate(chunk_files):
        if idx == 0 and first_chunk is not None:
            chunk = first_chunk
        else:
            chunk = _load_chunk(session_path / chunk_name)
        lengths.append(int(chunk["data"].shape[1]))
    return lengths


def _cache_file(cache_dir: Path, dataset_key: str) -> Path:
    safe_key = dataset_key.replace(os.sep, "_")
    return cache_dir / f"{safe_key}_index.json"


def _load_dataset_cache(
    cache_dir: Optional[Path], dataset_key: str
) -> Dict[str, Dict[str, object]]:
    if cache_dir is None:
        return {}
    cache_path = _cache_file(cache_dir, dataset_key)
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            raw: Dict[str, Dict[str, object]] = json.load(handle)
    except Exception:
        return {}
    return raw


def _save_dataset_cache(
    cache_dir: Optional[Path], dataset_key: str, data: Dict[str, Dict[str, object]]
) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file(cache_dir, dataset_key)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


def _metadata_from_cache(entry: Dict[str, object]) -> SessionMetadata:
    pos_array = np.asarray(entry["pos_2d"], dtype=np.float32)
    pos3d_entry = entry.get("pos_3d")
    ori_entry = entry.get("ori_3d")
    pos3d_array = (
        np.asarray(pos3d_entry, dtype=np.float32) if pos3d_entry is not None else None
    )
    ori_array = (
        np.asarray(ori_entry, dtype=np.float32) if ori_entry is not None else None
    )
    ch_types = entry.get("ch_types")
    chunk_lengths = entry.get("chunk_lengths")
    if chunk_lengths is not None:
        chunk_lengths = [int(length) for length in chunk_lengths]
    return SessionMetadata(
        chunk_files=list(entry["chunk_files"]),
        chunk_lengths=chunk_lengths,
        chunk_length=int(entry["chunk_length"]),
        last_chunk_length=int(entry.get("last_chunk_length", entry["chunk_length"])),
        ch_names=list(entry["ch_names"]),
        pos_2d=pos_array,
        ch_types=list(ch_types) if ch_types is not None else None,
        sfreq=float(entry["sfreq"]),
        pos_3d=pos3d_array,
        ori_3d=ori_array,
    )


def _metadata_to_cache(metadata: SessionMetadata) -> Dict[str, object]:
    payload = {
        "chunk_files": metadata.chunk_files,
        "chunk_lengths": metadata.chunk_lengths,
        "chunk_length": metadata.chunk_length,
        "last_chunk_length": metadata.last_chunk_length,
        "ch_names": metadata.ch_names,
        "pos_2d": metadata.pos_2d.tolist(),
        "ch_types": metadata.ch_types,
        "sfreq": metadata.sfreq,
    }
    payload["pos_3d"] = (
        metadata.pos_3d.tolist() if metadata.pos_3d is not None else None
    )
    payload["ori_3d"] = (
        metadata.ori_3d.tolist() if metadata.ori_3d is not None else None
    )
    return payload


def _prepare_datasets(
    dataset_root: Union[str, Sequence[str], Mapping[str, str]],
    example_seconds: float,
    overlap_seconds: float,
    cache_dir: Optional[Union[str, Path]] = None,
    refresh_cache: bool = False,
) -> PreparedDatasets:
    dataset_roots = _normalise_roots(dataset_root)
    if not dataset_roots:
        raise ValueError("No dataset roots were provided")

    cache_dir_path: Optional[Path]
    if cache_dir is None:
        cache_dir_path = None
    else:
        cache_dir_path = Path(cache_dir)

    registry = ChannelRegistry()
    session_channels: Dict[Tuple[str, str], SessionChannels] = {}
    sessions_per_dataset: Dict[str, List[str]] = {}
    indices: List[WindowSpec] = []

    window_size: Optional[int] = None
    overlap_size: Optional[int] = None
    sfreq_ref: Optional[float] = None

    for dataset_key, root in dataset_roots.items():
        dataset_cache: Dict[str, Dict[str, object]]
        if cache_dir_path is None or refresh_cache:
            dataset_cache = {}
        else:
            dataset_cache = _load_dataset_cache(cache_dir_path, dataset_key)
        session_names = _list_sessions(root)
        valid_sessions: List[str] = []

        for session in tqdm(session_names, desc=f"Sessions in {dataset_key}"):
            session_path = Path(root) / session

            metadata: Optional[SessionMetadata] = None
            if cache_dir_path is not None and not refresh_cache:
                cached_entry = dataset_cache.get(session)
                if cached_entry:  # and cached_entry.get("chunk_files") == chunk_files:
                    missing_geom = (
                        "pos_3d" not in cached_entry or "ori_3d" not in cached_entry
                    )
                    if not missing_geom:
                        metadata = _metadata_from_cache(cached_entry)

            if metadata is None:
                chunk_files = _list_chunk_files(session_path)
                if not chunk_files:
                    continue

                first_chunk = _load_chunk(session_path / chunk_files[0])
                ch_names = list(first_chunk.get("ch_names", []))
                if not ch_names:
                    raise ValueError(
                        f"Session {session} in dataset {dataset_key} has no chn names"
                    )

                pos_2d = np.asarray(first_chunk.get("pos_2d"))
                if pos_2d.size == 0:
                    raise ValueError(
                        f"Session {session} in dataset {dataset_key} has no 2D pos"
                    )

                pos_3d = first_chunk.get("pos_3d")
                pos_3d_arr = (
                    np.asarray(pos_3d, dtype=np.float32) if pos_3d is not None else None
                )
                if pos_3d_arr is not None:
                    if pos_3d_arr.shape[0] != len(ch_names) or pos_3d_arr.shape[1] != 3:
                        raise ValueError(
                            f"Session {session} in {dataset_key} has mismatched pos_3d."
                        )
                ori_3d = first_chunk.get("ori_3d")
                ori_3d_arr = (
                    np.asarray(ori_3d, dtype=np.float32) if ori_3d is not None else None
                )
                if ori_3d_arr is not None:
                    if ori_3d_arr.shape[0] != len(ch_names) or ori_3d_arr.shape[1] != 3:
                        raise ValueError(
                            f"Session {session} in {dataset_key} has mismatched ori_3d."
                        )

                ch_types = first_chunk.get("ch_types")
                chunk_lengths = _load_chunk_lengths(
                    session_path, chunk_files, first_chunk=first_chunk
                )
                chunk_length = int(chunk_lengths[0])
                last_chunk_length = int(chunk_lengths[-1])

                metadata = SessionMetadata(
                    chunk_files=chunk_files,
                    chunk_lengths=chunk_lengths,
                    chunk_length=chunk_length,
                    last_chunk_length=last_chunk_length,
                    ch_names=ch_names,
                    pos_2d=pos_2d,
                    ch_types=list(ch_types) if ch_types is not None else None,
                    sfreq=float(first_chunk.get("sfreq")),
                    pos_3d=pos_3d_arr,
                    ori_3d=ori_3d_arr,
                )
            else:
                ch_types = metadata.ch_types
                if metadata.chunk_lengths is None or len(metadata.chunk_lengths) != len(
                    metadata.chunk_files
                ):
                    chunk_lengths = _load_chunk_lengths(
                        session_path, metadata.chunk_files
                    )
                    metadata.chunk_lengths = chunk_lengths
                    metadata.chunk_length = int(chunk_lengths[0])
                    metadata.last_chunk_length = int(chunk_lengths[-1])
                else:
                    chunk_lengths = metadata.chunk_lengths

            session_channels[(dataset_key, session)] = registry.register(
                metadata.ch_names,
                metadata.pos_2d,
                ch_types,
                pos_3d=metadata.pos_3d,
                ori_3d=metadata.ori_3d,
            )

            sfreq = float(metadata.sfreq)
            if sfreq_ref is None:
                sfreq_ref = sfreq
            elif not np.isclose(sfreq_ref, sfreq, rtol=0, atol=1e-6):
                raise ValueError(
                    "All datasets must share the same sampling frequency to build"
                    " consistent chunks"
                )

            example_len_samples = int(round(example_seconds * sfreq))
            if example_len_samples <= 0:
                raise ValueError("example_seconds results in zero-length windows")

            overlap_samples = int(round(overlap_seconds * sfreq))
            if overlap_samples >= example_len_samples:
                raise ValueError("overlap_seconds must be smaller than example_seconds")

            if window_size is None:
                window_size = example_len_samples
                overlap_size = overlap_samples
            else:
                if example_len_samples != window_size:
                    raise ValueError(
                        "All datasets must share the same window length in samples."
                        " Adjust example_seconds or resample the data."
                    )
                if overlap_samples != overlap_size:
                    raise ValueError(
                        "All datasets must share the same overlap in samples."
                        " Adjust overlap_seconds accordingly."
                    )

            step = example_len_samples - overlap_samples
            chunk_files_order = metadata.chunk_files

            for idx_chunk, chunk_name in enumerate(chunk_files_order):
                total_samples = chunk_lengths[idx_chunk]

                if total_samples < example_len_samples:
                    # Skip chunks too short for the requested window
                    continue

                for start in range(0, total_samples - example_len_samples + 1, step):
                    indices.append(WindowSpec(dataset_key, session, chunk_name, start))

            valid_sessions.append(session)

            if cache_dir_path is not None:
                dataset_cache[session] = _metadata_to_cache(metadata)

        if valid_sessions:
            sessions_per_dataset[dataset_key] = valid_sessions
        if cache_dir_path is not None:
            _save_dataset_cache(cache_dir_path, dataset_key, dataset_cache)

    if not indices:
        raise ValueError("No windows were generated from the provided datasets")

    assert (
        window_size is not None and overlap_size is not None and sfreq_ref is not None
    )

    layout = registry.build_layout()

    return PreparedDatasets(
        indices=indices,
        layout=layout,
        session_channels=session_channels,
        dataset_roots=dataset_roots,
        window_size=window_size,
        overlap=overlap_size,
        sfreq=sfreq_ref,
        sessions_per_dataset=sessions_per_dataset,
    )


def _split_sessions(
    sessions: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    sessions = list(sessions)
    if not sessions:
        return [], [], []

    rng.shuffle(sessions)
    n_total = len(sessions)

    val_count = int(round(n_total * val_ratio))
    test_count = int(round(n_total * test_ratio))

    val_count = min(val_count, n_total)
    test_count = min(test_count, n_total - val_count)

    train_count = n_total - val_count - test_count
    if train_count <= 0 and n_total > 0:
        # Borrow one session back from validation/test to ensure train is non-empty
        if val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        train_count = n_total - val_count - test_count

    train = sessions[:train_count]
    val = sessions[train_count : train_count + val_count]
    test = sessions[train_count + val_count : train_count + val_count + test_count]

    return train, val, test


def _subject_from_session(session: str) -> str:
    """Extract a subject label from a session name (BIDS-style by default)."""
    for part in session.split("_"):
        if part.startswith("sub-"):
            return part
    return session


def _split_subjects(
    sessions: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    sessions = list(sessions)
    if not sessions:
        return [], [], []

    subject_to_sessions: Dict[str, List[str]] = {}
    for session in sessions:
        subject = _subject_from_session(session)
        subject_to_sessions.setdefault(subject, []).append(session)

    subjects = list(subject_to_sessions)
    train_subjects, val_subjects, test_subjects = _split_sessions(
        subjects, val_ratio, test_ratio, rng
    )

    def expand(subject_list: Sequence[str]) -> List[str]:
        return [
            session
            for subject in subject_list
            for session in subject_to_sessions[subject]
        ]

    return expand(train_subjects), expand(val_subjects), expand(test_subjects)


def _split_subjects_two_way(
    sessions: Sequence[str], rng: random.Random
) -> Tuple[List[str], List[str]]:
    sessions = list(sessions)
    if not sessions:
        return [], []

    subject_to_sessions: Dict[str, List[str]] = {}
    for session in sessions:
        subject = _subject_from_session(session)
        subject_to_sessions.setdefault(subject, []).append(session)

    subjects = list(subject_to_sessions)
    rng.shuffle(subjects)

    val_count = len(subjects) // 2
    val_subjects = subjects[:val_count]
    test_subjects = subjects[val_count:]

    def expand(subject_list: Sequence[str]) -> List[str]:
        return [
            session
            for subject in subject_list
            for session in subject_to_sessions[subject]
        ]

    return expand(val_subjects), expand(test_subjects)


class ChannelRegistry:
    """Keeps a canonical ordering of sensors across multiple datasets."""

    def __init__(self, *, decimals: int = 4) -> None:
        self.decimals = decimals
        self._entries: List[ChannelEntry] = []
        self._key_to_index: Dict[Tuple[Tuple[int, int], Union[int, str]], int] = {}

    def _make_key(
        self, pos: np.ndarray, ch_type: Union[int, str]
    ) -> Tuple[Tuple[int, int], Union[int, str]]:
        scale = 10**self.decimals
        quantised = tuple(int(round(coord * scale)) for coord in pos[:2])
        return quantised, ch_type

    def register(
        self,
        ch_names: Sequence[str],
        pos_2d: np.ndarray,
        ch_types: Optional[Sequence[Union[int, str]]] = None,
        *,
        pos_3d: Optional[np.ndarray] = None,
        ori_3d: Optional[np.ndarray] = None,
    ) -> SessionChannels:
        if ch_types is None:
            ch_types = ["unknown"] * len(ch_names)

        canonical_indices = np.empty(len(ch_names), dtype=np.int64)
        pos3d_arr = np.asarray(pos_3d, dtype=np.float64) if pos_3d is not None else None
        ori_arr = np.asarray(ori_3d, dtype=np.float64) if ori_3d is not None else None

        for idx, (name, pos, ch_type) in enumerate(zip(ch_names, pos_2d, ch_types)):
            pos3d = pos3d_arr[idx] if pos3d_arr is not None else None
            ori_vec = ori_arr[idx] if ori_arr is not None else None
            key = self._make_key(np.asarray(pos), ch_type)
            if key not in self._key_to_index:
                entry = ChannelEntry(
                    pos_sum=np.asarray(pos, dtype=np.float64),
                    count=1,
                    ch_type=ch_type,
                    pos3d_sum=(
                        np.asarray(pos3d, dtype=np.float64)
                        if pos3d is not None
                        else None
                    ),
                    ori_sum=(
                        np.asarray(ori_vec, dtype=np.float64)
                        if ori_vec is not None
                        else None
                    ),
                    names={name},
                )
                self._key_to_index[key] = len(self._entries)
                self._entries.append(entry)
            else:
                entry = self._entries[self._key_to_index[key]]
                entry.update(
                    np.asarray(pos, dtype=np.float64),
                    name,
                    pos_3d=(
                        np.asarray(pos3d, dtype=np.float64)
                        if pos3d is not None
                        else None
                    ),
                    ori_3d=(
                        np.asarray(ori_vec, dtype=np.float64)
                        if ori_vec is not None
                        else None
                    ),
                )

            canonical_indices[idx] = self._key_to_index[key]

        return SessionChannels(indices=canonical_indices)

    def build_layout(self) -> ChannelLayout:
        names = [entry.canonical_name for entry in self._entries]
        types = [entry.ch_type for entry in self._entries]
        pos = np.stack([entry.mean_pos for entry in self._entries], axis=0)
        pos3d = None
        if any(entry.mean_pos3d is not None for entry in self._entries):
            pos3d = np.zeros((len(self._entries), 3), dtype=np.float32)
            for idx, entry in enumerate(self._entries):
                if entry.mean_pos3d is not None:
                    pos3d[idx] = entry.mean_pos3d

        ori = None
        if any(entry.mean_ori is not None for entry in self._entries):
            ori = np.zeros((len(self._entries), 3), dtype=np.float32)
            for idx, entry in enumerate(self._entries):
                if entry.mean_ori is not None:
                    vec = entry.mean_ori
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    ori[idx] = vec

        return ChannelLayout(
            names=names, types=types, pos_2d=pos, pos_3d=pos3d, ori_3d=ori
        )


def build_indices(session_dir: str, example_len: float, overlap: float) -> Tuple[
    List[Tuple[str, int, int]],
    int,
    int,
    List[str],
    np.ndarray,
    float,
]:
    """Legacy helper that mirrors the historical build_indices API.

    This function now routes through the multi-dataset-aware preparation logic but only
    supports a single `session_dir`. It is kept for backwards compatibility with older
    tests and utilities.
    """

    prepared = _prepare_datasets(session_dir, example_len, overlap)
    if len(prepared.dataset_roots) != 1:
        raise ValueError(
            "build_indices only supports a single dataset root; please use"
            " split_datasets for multi-dataset scenarios"
        )

    dataset_key = next(iter(prepared.dataset_roots))

    legacy_indices: List[Tuple[str, int, int]] = []
    for spec in prepared.indices:
        if spec.dataset != dataset_key:
            continue
        stem = Path(spec.chunk).stem
        if not stem.isdigit():
            raise ValueError(
                "Legacy build_indices expects chunk filenames to be numeric"
            )
        legacy_indices.append((spec.session, int(stem), spec.start))

    return (
        legacy_indices,
        prepared.window_size,
        prepared.overlap,
        prepared.layout.names,
        prepared.layout.pos_2d,
        prepared.sfreq,
    )


def split_datasets(
    dataset_root: Union[str, Sequence[str], Mapping[str, str]],
    example_seconds: float,
    overlap_seconds: float,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    dataset_class: str = "ChunkDataset",
    dataset_kwargs: Optional[dict] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    refresh_cache: bool = False,
    split_strategy: str = "session",
    heldout_dataset: Optional[str] = None,
) -> Split:
    """Create train/val/test splits across one or more MEG datasets.

    Parameters mirror the legacy interface but now support multiple dataset roots. Each
    dataset is split independently before concatenating indices so that distributional
    differences do not leak between splits. All channel layouts are merged into a
    canonical ordering that downstream datasets use to produce consistent tensors
    regardless of missing sensors.

    Args:
        cache_dir: Optional directory to persist session metadata between runs.
            When provided, the datasplitter will avoid reloading chunk files on
            subsequent startups by replaying cached metadata. Use
            ``refresh_cache=True`` if new data has been added and the cache should be
            rebuilt.
        split_strategy: ``session`` to split per-session (default), ``subject`` to keep
            all sessions for a subject in the same split, or ``dataset`` to hold out a
            full dataset for validation/testing. Subject labels are inferred from
            BIDS-style ``sub-`` tokens in session names.
        heldout_dataset: Dataset key to hold out when using `split_strategy="dataset"`.
            All other datasets are used for training and the held-out dataset is split
            50/50 into validation and test at the subject level.
    """

    dataset_kwargs = dataset_kwargs or {}

    prepared = _prepare_datasets(
        dataset_root,
        example_seconds,
        overlap_seconds,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
    )
    rng = random.Random(seed)

    split_strategy = split_strategy.lower()
    if split_strategy not in {"session", "subject", "dataset"}:
        raise ValueError("split_strategy must be 'session', 'subject', or 'dataset'")

    assignments: Dict[Tuple[str, str], str] = {}
    if split_strategy == "dataset":
        dataset_keys = list(prepared.dataset_roots)
        if len(dataset_keys) < 2:
            raise ValueError("split_strategy='dataset' requires multiple dataset roots")
        if heldout_dataset is None:
            raise ValueError(
                "split_strategy='dataset' requires heldout_dataset to be set"
            )
        heldout_key = str(heldout_dataset)
        if heldout_key not in dataset_keys:
            raise KeyError(
                "heldout_dataset must match a dataset_root key; "
                f"got '{heldout_key}', expected one of {sorted(dataset_keys)}"
            )

        for dataset_key, sessions in prepared.sessions_per_dataset.items():
            if dataset_key == heldout_key:
                if not sessions:
                    raise ValueError(
                        f"Held-out dataset '{heldout_key}' has no valid sessions"
                    )
                val_s, test_s = _split_subjects_two_way(sessions, rng)
                for session in val_s:
                    assignments[(dataset_key, session)] = "val"
                for session in test_s:
                    assignments[(dataset_key, session)] = "test"

                print(
                    f"Dataset {dataset_key}: "
                    f"train=0 val={len(val_s)} test={len(test_s)}"
                )
            else:
                for session in sessions:
                    assignments[(dataset_key, session)] = "train"

                print(f"Dataset {dataset_key}: train={len(sessions)} val=0 test=0")
    else:
        splitter = _split_subjects if split_strategy == "subject" else _split_sessions

        for dataset_key, sessions in prepared.sessions_per_dataset.items():
            train_s, val_s, test_s = splitter(sessions, val_ratio, test_ratio, rng)

            for session in train_s:
                assignments[(dataset_key, session)] = "train"
            for session in val_s:
                assignments[(dataset_key, session)] = "val"
            for session in test_s:
                assignments[(dataset_key, session)] = "test"

            print(
                f"Dataset {dataset_key}: "
                f"train={len(train_s)} val={len(val_s)} test={len(test_s)}"
            )

    expected_pairs = {
        (dataset_key, session)
        for dataset_key, sessions in prepared.sessions_per_dataset.items()
        for session in sessions
    }
    assignment_pairs = set(assignments)
    missing_pairs = expected_pairs - assignment_pairs
    extra_pairs = assignment_pairs - expected_pairs
    if missing_pairs or extra_pairs:
        raise ValueError(
            "Split assignments do not cover the prepared sessions. "
            f"missing={sorted(missing_pairs)} extra={sorted(extra_pairs)}"
        )

    def collect(split: str) -> List[WindowSpec]:
        return [
            spec
            for spec in prepared.indices
            if assignments[(spec.dataset, spec.session)] == split
        ]

    train_idx = collect("train")
    val_idx = collect("val")
    test_idx = collect("test")

    dataset_class_impl = DATASET_CLASSES[dataset_class]

    common_kwargs = dict(
        root_dir=prepared.dataset_roots,
        length=prepared.window_size,
        ch_names=prepared.layout.names,
        pos_2d=prepared.layout.pos_2d,
        pos_3d=prepared.layout.pos_3d,
        ori_3d=prepared.layout.ori_3d,
        sfreq=prepared.sfreq,
        ch_types=prepared.layout.types,
        session_channels=prepared.session_channels,
    )
    common_kwargs.update(dataset_kwargs)

    train_ds = dataset_class_impl(indices=train_idx, **common_kwargs)
    val_ds = dataset_class_impl(indices=val_idx, **common_kwargs)
    test_ds = dataset_class_impl(indices=test_idx, **common_kwargs)

    return Split(train_ds, val_ds, test_ds)
