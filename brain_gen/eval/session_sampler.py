from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from ..dataset.datasets import _load_chunk_cached


@dataclass(frozen=True)
class SessionSample:
    """Container for a sampled session segment."""

    dataset_key: str
    session: str
    task_type: str | None
    data: np.ndarray
    condition: np.ndarray | None
    context_steps: int
    total_steps: int
    pos: np.ndarray | None
    sensor_type: np.ndarray | None


class SessionSampler:
    """Sample session sequences from the first chunk file only."""

    def __init__(self, dataset: Any, cfg: dict[str, Any]) -> None:
        """Initialize the session sampler with dataset and config."""
        self.dataset = dataset
        self.cfg = cfg
        self.sfreq = float(getattr(dataset, "sfreq", 1.0))
        self.rng = np.random.default_rng(cfg.get("seed"))
        self.pos, self.sensor_type = self._resolve_sensor_metadata(dataset)

        self.split = str(cfg.get("split", "val"))
        self.task_type = cfg.get("task_type")
        self.dataset_filter = _normalise_dataset_filter(
            cfg.get("dataset_key") or cfg.get("dataset_keys") or cfg.get("dataset")
        )
        self.dataset_keys = _resolve_dataset_keys(
            self.dataset.root_dirs, self.dataset_filter
        )
        self.num_sessions = int(cfg.get("num_sessions", cfg.get("num_runs", 1)))

        self.context_steps = self._resolve_steps(
            cfg,
            "context_length_steps",
            "context_length_s",
        )
        self.total_steps = self._resolve_steps(
            cfg,
            "total_length_steps",
            "total_length_s",
        )
        if self.total_steps <= 0:
            raise ValueError("total_length_s/steps must be set and > 0.")
        if self.context_steps <= 0:
            raise ValueError("context_length_s/steps must be set and > 0.")
        if self.context_steps >= self.total_steps:
            raise ValueError(
                "context_length must be smaller than total_length "
                f"(got {self.context_steps} >= {self.total_steps})."
            )

    def sample_sessions(self) -> list[SessionSample]:
        """Return sampled session segments as (C, T) arrays."""
        allowed = self._allowed_sessions()
        candidates = self._collect_sessions(allowed)
        if not candidates:
            return []

        if self.num_sessions > 0:
            self.rng.shuffle(candidates)
            candidates = candidates[: min(self.num_sessions, len(candidates))]
        session_names = [
            f"{meta['dataset_key']}/{meta['session']}" for meta in candidates
        ]
        print(
            "[session_sampler] Sampled sessions "
            f"({len(session_names)}):\n" + "\n".join(session_names)
        )

        samples: list[SessionSample] = []
        for meta in candidates:
            data, condition = self._load_session(meta)
            if data is None:
                continue
            data = data[:, : self.total_steps]
            cond = condition[: self.total_steps] if condition is not None else None
            samples.append(
                SessionSample(
                    dataset_key=meta["dataset_key"],
                    session=meta["session"],
                    task_type=meta["task_type"],
                    data=data,
                    condition=cond,
                    context_steps=self.context_steps,
                    total_steps=self.total_steps,
                    pos=self.pos,
                    sensor_type=self.sensor_type,
                )
            )
        return samples

    def _resolve_sensor_metadata(
        self, dataset: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return canonical sensor metadata for tokenizer-aware models."""
        pos = getattr(dataset, "_pos_orientation", None)
        sensor_type = getattr(dataset, "_sensor_type", None)
        if pos is None:
            pos = getattr(dataset, "pos_2d", None)
        if sensor_type is None:
            sensor_type = getattr(dataset, "ch_type", None)

        def _to_numpy(value: Any) -> np.ndarray | None:
            if value is None:
                return None
            if torch.is_tensor(value):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        return _to_numpy(pos), _to_numpy(sensor_type)

    def _resolve_steps(
        self,
        cfg: dict[str, Any],
        steps_key: str,
        seconds_key: str,
    ) -> int:
        """Resolve step counts from config using seconds when needed."""
        steps = cfg.get(steps_key)
        if steps is not None:
            return max(0, int(steps))
        seconds = cfg.get(seconds_key)
        if seconds is None:
            return 0
        return max(0, int(round(float(seconds) * self.sfreq)))

    def _allowed_sessions(self) -> set[tuple[str, str]]:
        """Return (dataset_key, session) keys permitted by the split."""
        indices = getattr(self.dataset, "indices", [])
        allowed: set[tuple[str, str]] = set()
        for entry in indices:
            spec = _unpack_index_entry(entry)
            if spec is None:
                continue
            dataset_key, session, _, _ = spec
            if self.dataset_keys and dataset_key not in self.dataset_keys:
                continue
            allowed.add((dataset_key, session))
        return allowed

    def _collect_sessions(
        self,
        allowed: Iterable[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        """Collect candidate sessions matching filters and length constraints."""
        allowed = set(allowed)
        task_filter = _normalise_task_filter(self.task_type)
        candidates: list[dict[str, Any]] = []
        for dataset_key in self.dataset_keys:
            root = self.dataset.root_dirs[dataset_key]
            root_path = Path(root)
            if not root_path.exists():
                continue
            session_dirs = [
                session_dir
                for session_dir in root_path.iterdir()
                if session_dir.is_dir() and not session_dir.name.startswith(".")
            ]
            session_dirs.sort(key=lambda p: p.name)
            for session_dir in session_dirs:
                session = session_dir.name
                if allowed and (dataset_key, session) not in allowed:
                    continue
                task_type = _resolve_task_type(session, task_filter)
                if task_filter and task_type not in task_filter:
                    continue
                chunk_files = _list_chunk_files(session_dir)
                if not chunk_files:
                    continue
                first_chunk = chunk_files[0]
                first_len = _chunk_length(first_chunk)
                if first_len < self.total_steps:
                    continue
                candidates.append(
                    {
                        "dataset_key": dataset_key,
                        "session": session,
                        "task_type": task_type,
                        "chunk_files": [first_chunk],
                    }
                )
        return candidates

    def _load_session(
        self, meta: dict[str, Any]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load and map the session's first chunk to canonical channels."""
        segments: list[np.ndarray] = []
        conditions: list[np.ndarray] = []
        dataset_key = meta["dataset_key"]
        session = meta["session"]
        has_condition = bool(getattr(self.dataset, "has_condition", False))
        ch_names = getattr(self.dataset, "ch_names", None)
        if ch_names is None:
            return None, None

        for chunk_path in meta["chunk_files"]:
            chunk_dict = _load_chunk_cached(str(chunk_path))
            data = chunk_dict["data"]

            cond = None
            if has_condition:
                if "rest" in session:
                    cond = np.zeros(data.shape[1], dtype=data.dtype)
                    window = data
                else:
                    cond = data[-1]
                    window = data[:-1]
            else:
                window = data

            mapped = np.ones((len(ch_names), window.shape[1]), dtype=window.dtype)
            mapped *= getattr(self.dataset, "fill_value", 0)
            indices = self.dataset._get_session_indices(
                dataset_key, session, window.shape[0]
            )
            if len(indices) != window.shape[0]:
                raise ValueError(
                    f"Channel mismatch for session {session} ({dataset_key}): "
                    f"expected {len(indices)}, got {window.shape[0]}"
                )
            mapped[indices, :] = window
            segments.append(mapped)
            if cond is not None:
                conditions.append(cond.astype(window.dtype, copy=False))

        if not segments:
            return None, None
        data = np.concatenate(segments, axis=1)
        condition = np.concatenate(conditions, axis=0) if conditions else None
        return data, condition


def _chunk_sort_key(filename: str) -> tuple[int, object]:
    """Return a sortable key for chunk filenames."""
    stem = Path(filename).stem
    if stem.isdigit():
        return 0, int(stem)
    return 1, stem


def _list_chunk_files(session_path: Path) -> list[Path]:
    """List chunk files for a session in numeric order."""
    files = [
        f
        for f in session_path.iterdir()
        if f.suffix == ".npy" and "events" not in f.name and "session_" not in f.name
    ]
    files.sort(key=lambda p: _chunk_sort_key(p.name))
    return files


def _chunk_length(chunk_file: Path) -> int:
    """Return the number of samples in a single chunk file."""
    chunk_dict = _load_chunk_cached(str(chunk_file))
    return int(chunk_dict["data"].shape[1])


def _unpack_index_entry(entry: Any) -> tuple[str, str, str, int] | None:
    """Normalize an index entry into a tuple form."""
    if entry is None:
        return None

    if isinstance(entry, tuple) and len(entry) >= 4:
        dataset_key, session, chunk, start = entry[:4]
        return str(dataset_key), str(session), str(chunk), int(start)

    if all(hasattr(entry, attr) for attr in ("dataset", "session", "chunk", "start")):
        return (
            str(getattr(entry, "dataset")),
            str(getattr(entry, "session")),
            str(getattr(entry, "chunk")),
            int(getattr(entry, "start")),
        )

    return None


def _normalise_task_filter(task_type: Any) -> list[str]:
    """Normalize task type filters into a list of lowercase strings."""
    if task_type is None:
        return []
    if isinstance(task_type, (list, tuple)):
        return [str(t).lower() for t in task_type]
    return [str(task_type).lower()]


def _normalise_dataset_filter(dataset_key: Any) -> list[str]:
    """Normalize dataset filters into a list of lowercase strings."""
    if dataset_key is None:
        return []
    if isinstance(dataset_key, (list, tuple, set, np.ndarray)):
        return [str(key).lower() for key in dataset_key]
    return [str(dataset_key).lower()]


def _resolve_dataset_keys(
    root_dirs: Mapping[str, str], dataset_filter: list[str]
) -> list[str]:
    """Return dataset keys selected by the filter (or all keys if empty)."""
    keys = sorted(root_dirs)
    if not dataset_filter:
        return keys
    selected = [key for key in keys if key.lower() in dataset_filter]
    if not selected:
        available = ", ".join(keys) if keys else "<none>"
        requested = ", ".join(dataset_filter)
        raise ValueError(
            "No dataset roots matched the sampler dataset filter "
            f"({requested}). Available: {available}."
        )
    return selected


def _resolve_task_type(session_name: str, task_filter: list[str]) -> str | None:
    """Infer task type from a session folder name."""
    name = session_name.lower()
    if task_filter:
        for task in task_filter:
            if task in name:
                return task
    for sep in ("_", "-"):
        if sep in name:
            return name.split(sep)[0]
    return name or None
