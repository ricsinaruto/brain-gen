import os
import re
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import json
from torch.utils.data import Dataset

from ..utils.quantizers import mulaw_torch
from ..utils.image_interpolation import GaussianSensorInterpolator

SENSOR_TYPES = {
    "CTF_AXIAL_GRAD": 0,
    "MAG": 1,
    "ELEKTA_GRAD_PLANAR_X": 2,
    "ELEKTA_GRAD_PLANAR_Y": 3,
}

DATASET_NAMES = {
    "omega": "CTF_AXIAL_GRAD",
    "mous": "CTF_AXIAL_GRAD",
    "camcan": ["MAG", "ELEKTA_GRAD_PLANAR_X", "ELEKTA_GRAD_PLANAR_Y"],
}


# Module-level LRU cache for chunk loading to avoid repeated disk I/O
# Caches the raw dict from np.load; default 64 chunks (~4-8 GB typical)
@lru_cache(maxsize=64)
def _load_chunk_cached(file_path: str) -> dict:
    """Load and cache a chunk file from disk.

    Using module-level function with lru_cache enables sharing across dataset instances
    and dataloader workers (within same process).
    """
    try:
        return np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading chunk {file_path}: {e}")
        raise e


def clear_chunk_cache() -> None:
    """Clear the chunk loading cache (useful for memory management)."""
    _load_chunk_cached.cache_clear()


def get_chunk_cache_info():
    """Get cache statistics (hits, misses, size, maxsize)."""
    return _load_chunk_cached.cache_info()


# Tokenized chunk cache to reuse precomputed codes across samples
@lru_cache(maxsize=64)
def _load_tokenized_chunk_cached(file_path: str) -> dict:
    """Load and cache a tokenized chunk file from disk."""
    return np.load(file_path, allow_pickle=True).item()


def clear_tokenized_cache() -> None:
    """Clear the tokenized chunk loading cache."""
    _load_tokenized_chunk_cached.cache_clear()


def get_tokenized_cache_info():
    """Get cache statistics for tokenized chunks."""
    return _load_tokenized_chunk_cached.cache_info()


class Postprocessor:
    def __init__(
        self, pos_2d: List[Tuple[float, float]], image_size: int, tmp_dir: str = "tmp"
    ):
        # --- Pre-compute the (row, col) pixel for each channel -----------------
        pos = pos_2d.astype(np.float32)
        # Normalise to the unit square [0,1]²
        pos_min = pos.min(axis=0)
        pos_max = pos.max(axis=0)
        span = pos_max - pos_min
        span[span == 0] = 1.0  # avoid divide-by-zero
        pos_norm = (pos - pos_min) / span

        col_idx = np.round(pos_norm[:, 0] * (image_size - 1)).astype(np.int64)
        row_idx = np.round(pos_norm[:, 1] * (image_size - 1)).astype(np.int64)

        self.row_idx = torch.from_numpy(row_idx)
        self.col_idx = torch.from_numpy(col_idx)

        # save these to a tmp file
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        tmp_file = os.path.join(tmp_dir, "img_inds.npy")
        np.save(tmp_file, {"row_idx": row_idx, "col_idx": col_idx})

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape an image into a channel-wise tensor. This is the opposite of the
        vectorised scatter operation in __getitem__. It should create a tensor.

        with shape (C, T), where C is the number of channels, always lower than H * W.

        x: [H, W, T] -> [C, T]
        """
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]  # kind of hacky assuming tensor is first element of tuple/list

        # Add a batch dimension if it's not present to unify logic
        squeeze_batch = False
        squeeze_all = False
        if x.ndim == 2:  # [H, W] -> [1, H, W, 1]
            x = x.unsqueeze(0).unsqueeze(-1)
            squeeze_all = True
        elif x.ndim == 3:  # [H, W, T]
            x = x.unsqueeze(0)  # [1, H, W, T]
            squeeze_batch = True

        # x: [B, H, W, T] -> out: [B, C, T]
        out = x[:, self.row_idx, self.col_idx, ...]

        # If original input was 3D, remove batch dimension
        if squeeze_batch:
            out = out.squeeze(0)  # [C, T]
        if squeeze_all:
            out = out.squeeze(-1).squeeze(0)

        return out

    def __call__(self, *tensors):
        return tuple(map(self.reshape, tensors))


class ChunkDataset(Dataset):
    """Dataset for chunked numpy files with canonicalised channel layouts."""

    IndexEntry = Tuple[str, str, str, int]

    def __init__(
        self,
        root_dir: Union[str, Mapping[str, str]],
        indices: Sequence[Union[IndexEntry, Tuple[str, int, int], object]],
        length: int,
        ch_names: Sequence[str],
        pos_2d: Sequence[Tuple[float, float]],
        sfreq: Union[int, float],
        name: Optional[str] = None,
        image_size: int = 32,
        aug_cfg: Optional[dict] = None,
        tmp_dir: str = "tmp",
        fill_value: int = 0,
        layout_path: Optional[str] = None,
        *,
        ch_types: Optional[Sequence[Union[int, str]]] = None,
        session_channels: Optional[Mapping[Tuple[str, str], object]] = None,
        has_condition: bool = False,
        pos_3d: Optional[Sequence[Tuple[float, float, float]]] = None,
        ori_3d: Optional[Sequence[Tuple[float, float, float]]] = None,
    ) -> None:
        self.root_dirs = self._normalise_roots(root_dir)
        self.default_dataset_key = next(iter(self.root_dirs))
        self.root_dir = self.root_dirs[self.default_dataset_key]
        self.indices = self._normalise_indices(indices)
        self.length = int(length)
        self.sfreq = float(sfreq)
        self.image_size = image_size
        self.tmp_dir = tmp_dir
        self.fill_value = fill_value
        self.has_condition = has_condition

        self.ch_names = [str(name) for name in ch_names]
        self.pos_3d = None if pos_3d is None else np.asarray(pos_3d, dtype=np.float32)
        self.ori_3d = None if ori_3d is None else np.asarray(ori_3d, dtype=np.float32)
        self.num_channels = len(self.ch_names)

        if layout_path:
            self.pos_2d = np.load(layout_path)
        else:
            self.pos_2d = np.asarray(pos_2d, dtype=np.float32)
        Postprocessor(self.pos_2d, image_size, tmp_dir)

        if self.pos_3d is not None:
            if self.pos_3d.shape[0] != self.num_channels or self.pos_3d.shape[1] != 3:
                raise ValueError("pos_3d must have shape (num_channels, 3).")
        if self.ori_3d is not None:
            if self.ori_3d.shape[0] != self.num_channels or self.ori_3d.shape[1] != 3:
                raise ValueError("ori_3d must have shape (num_channels, 3).")

        self.ch_type_labels = self._resolve_channel_labels(ch_types, name)
        self.ch_type = self._encode_channel_types(self.ch_type_labels)

        self.session_indices = self._prepare_session_channels(
            session_channels, len(self.ch_names)
        )

        print(
            "Canonical channels:",
            len(self.ch_names),
            "| sessions:",
            len({(d, s) for d, s, _, _ in self.indices}),
        )

    @staticmethod
    def _normalise_roots(root_dir: Union[str, Mapping[str, str]]) -> Dict[str, str]:
        if isinstance(root_dir, Mapping):
            roots = {str(k): str(v) for k, v in root_dir.items()}
        else:
            roots = {"dataset0": str(root_dir)}

        for key, path in roots.items():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Dataset root not found for {key}: {path}")
        return roots

    def _normalise_indices(
        self, indices: Sequence[Union[IndexEntry, Tuple[str, int, int], object]]
    ) -> List[IndexEntry]:
        normalised: List[ChunkDataset.IndexEntry] = []

        for item in indices:
            if hasattr(item, "dataset") and hasattr(item, "session"):
                dataset_key = str(getattr(item, "dataset"))
                session = str(getattr(item, "session"))
                chunk = str(getattr(item, "chunk"))
                start = int(getattr(item, "start"))
            else:
                session, chunk_idx, start = item  # type: ignore[misc]
                dataset_key = self.default_dataset_key
                chunk = f"{int(chunk_idx)}.npy"
                start = int(start)

            if dataset_key not in self.root_dirs:
                raise KeyError(
                    f"Unknown dataset key '{dataset_key}' for index (session={session})"
                )

            normalised.append((dataset_key, session, chunk, start))

        return normalised

    def _resolve_channel_labels(
        self,
        ch_types: Optional[Sequence[Union[int, str]]],
        name: Optional[str],
    ) -> List[Union[int, str]]:
        if ch_types is not None:
            return list(ch_types)

        if name and name in DATASET_NAMES:
            sensor_key = DATASET_NAMES[name]
            mapped = SENSOR_TYPES.get(sensor_key)
            if mapped is not None:
                return [mapped] * len(self.ch_names)

        return ["unknown"] * len(self.ch_names)

    @staticmethod
    def _encode_channel_types(labels: Sequence[Union[int, str]]) -> np.ndarray:
        arr = np.asarray(labels)
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype(np.int64)

        unique = {label: idx for idx, label in enumerate(sorted(set(arr.tolist())))}
        return np.array([unique[label] for label in arr], dtype=np.int64)

    def _prepare_session_channels(
        self,
        session_channels: Optional[Mapping[Tuple[str, str], object]],
        canonical_size: int,
    ) -> Dict[Tuple[str, str], np.ndarray]:
        identity = np.arange(canonical_size, dtype=np.int64)
        mapping: Dict[Tuple[str, str], np.ndarray] = {}

        if session_channels:
            for key, value in session_channels.items():
                indices = getattr(value, "indices", value)
                arr = np.asarray(indices, dtype=np.int64)
                mapping[(str(key[0]), str(key[1]))] = arr

        self._identity_indices = identity
        self._session_present: Dict[Tuple[str, str], np.ndarray] = {}
        for key, arr in mapping.items():
            mask = np.zeros(canonical_size, dtype=bool)
            mask[arr] = True
            self._session_present[key] = mask

        return mapping

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.indices)

    def make_postprocessor(self):
        return Postprocessor(self.pos_2d, self.image_size, self.tmp_dir)

    def _get_session_indices(
        self, dataset_key: str, session: str, n_channels: int
    ) -> np.ndarray:
        key = (dataset_key, session)
        if key in self.session_indices:
            return self.session_indices[key]
        if n_channels > len(self._identity_indices):
            raise ValueError(
                f"Session {session} ({dataset_key}) "
                "has more channels than canonical layout"
            )
        return self._identity_indices[:n_channels]

    def _resolve_index(self, idx: int) -> IndexEntry:
        return self.indices[idx]

    def _load_data(self, idx: int):
        dataset_key, session, chunk, start = self._resolve_index(idx)
        root = self.root_dirs[dataset_key]
        file_path = os.path.join(root, session, chunk)
        # Use cached loader to avoid repeated disk I/O for same chunk
        data_dict = _load_chunk_cached(file_path)

        data = data_dict["data"]

        # check if condition contains a few integers, or continuous values
        condition = None
        window = data[:, start : start + self.length]
        if self.has_condition:
            if "rest" in session:
                condition = np.zeros(self.length, dtype=window.dtype)
            else:
                window = data[:-1, start : start + self.length]
                condition = data[-1, start : start + self.length]

            data_dict["condition"] = torch.from_numpy(condition).long()

        if window.shape[1] < self.length:
            raise ValueError(
                f"Window shorter than expected for session {session} ({dataset_key}) "
                f"chunk={chunk} start={start}: "
                f"expected {self.length}, got {window.shape[1]}"
            )

        mapped = np.ones((len(self.ch_names), self.length), dtype=window.dtype)
        mapped *= self.fill_value
        indices = self._get_session_indices(dataset_key, session, window.shape[0])

        if len(indices) != window.shape[0]:
            raise ValueError(
                f"Channel count mismatch for session {session} ({dataset_key}):"
                f" expected {len(indices)}, got {window.shape[0]}"
            )

        mapped[indices, :] = window
        x = torch.from_numpy(mapped)

        data_dict["indices"] = torch.from_numpy(indices)
        return x, data_dict

    def __getitem__(self, idx: int, return_dict: bool = False, long: bool = True):
        x, data_dict = self._load_data(idx)
        x = x.long() if long else x.float()

        inputs = x[:, :-1]
        targets = x[:, 1:]

        input_ret = inputs
        if self.has_condition:
            cond = data_dict["condition"][None, :-1]
            input_ret = (inputs, cond)

        if return_dict:
            return input_ret, targets, data_dict

        return input_ret, targets


class ChunkDataset3D(ChunkDataset):
    def __getitem__(self, idx: int, return_dict: bool = False, long: bool = True):
        x, _ = self._load_data(idx)
        x = x.long() if long else x.float()

        # reshape x to [T, H, 1]
        x = x.permute(1, 0)
        x = x.unsqueeze(-1)

        return x, x


class BPEDataset(ChunkDataset3D):
    def __init__(self, *args, group_size: int = 50, escape_value: int = 63, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size
        self.escape_value = escape_value
        lookup = json.load(open(f"{self.root_dir}/char_lookup_codepoints.json"))

        self.lookup = [chr(x) for x in lookup]

    def __getitem__(self, idx: int):
        x, _ = super().__getitem__(idx)
        x = x.squeeze(-1)
        T, C = x.shape

        x[x == self.escape_value] = 0

        groups: list[str] = []

        for start in range(0, T, self.group_size):
            end = min(start + self.group_size, T)
            tokens = [
                "".join(self.lookup[val] for val in x[start:end, ch]) for ch in range(C)
            ]

            groups.append(tokens)

        return groups, x


class ChunkDatasetMasked(ChunkDataset):
    def __getitem__(self, idx: int):  # type: ignore[override]
        input_ret, targets, data_dict = super().__getitem__(idx, return_dict=True)

        indices = data_dict["indices"]
        mask = torch.zeros(self.num_channels, dtype=torch.bool, device=indices.device)
        mask[indices] = True

        return input_ret, (targets, mask)


class ChunkDatasetSubset(ChunkDataset):
    _PRESET_KEYS = ("visual",)

    def __init__(
        self,
        *args,
        channel_subset: Union[str, Sequence[str]] = "visual",
        **kwargs,
    ) -> None:
        if channel_subset is None:
            raise ValueError("channel_subset must be provided for ChunkDatasetSubset.")

        self._requested_subset = channel_subset

        super().__init__(*args, **kwargs)

        self._canonical_ch_names = list(self.ch_names)
        self._canonical_pos_2d = np.array(self.pos_2d, copy=True)
        self._canonical_pos_3d = (
            None if self.pos_3d is None else np.array(self.pos_3d, copy=True)
        )
        self._canonical_ori_3d = (
            None if self.ori_3d is None else np.array(self.ori_3d, copy=True)
        )
        self._canonical_ch_type = np.array(self.ch_type, copy=True)
        self._canonical_ch_type_labels = list(self.ch_type_labels)

        subset_names = self._normalise_subset(channel_subset)
        subset_indices = self._indices_from_names(subset_names)
        if subset_indices.size == 0:
            raise ValueError("Resolved channel subset is empty.")

        self._subset_indices = subset_indices
        self._subset_names = [self._canonical_ch_names[i] for i in subset_indices]

        self.ch_names = list(self._subset_names)
        self.pos_2d = self._canonical_pos_2d[subset_indices]
        if self._canonical_pos_3d is not None:
            self.pos_3d = self._canonical_pos_3d[subset_indices]
        if self._canonical_ori_3d is not None:
            self.ori_3d = self._canonical_ori_3d[subset_indices]
        self.ch_type_labels = [
            self._canonical_ch_type_labels[i] for i in subset_indices
        ]
        self.ch_type = self._canonical_ch_type[subset_indices]
        self.num_channels = len(self.ch_names)

        # Regenerate postprocessor artefacts for the active subset layout.
        Postprocessor(self.pos_2d, self.image_size, self.tmp_dir)

    # ------------------------------------------------------------------ #
    def _normalise_subset(self, channel_subset: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(channel_subset, str):
            option = channel_subset.strip().lower()
            if option in self._PRESET_KEYS:
                resolved = self._resolve_preset(option)
                if resolved:
                    return resolved
                raise ValueError(
                    f"Preset '{channel_subset}' did not match any channels."
                )
            # Treat as single explicit channel name.
            return [channel_subset]

        if not isinstance(channel_subset, Sequence):
            raise TypeError(
                "channel_subset must be a string preset or a sequence of channel names."
            )

        subset_list = [str(name) for name in channel_subset]
        if not subset_list:
            raise ValueError("channel_subset must contain at least one channel name.")
        return subset_list

    def _indices_from_names(self, subset_names: Sequence[str]) -> np.ndarray:
        name_to_index = {name: idx for idx, name in enumerate(self._canonical_ch_names)}
        indices: List[int] = []
        missing: List[str] = []
        seen = set()

        for name in subset_names:
            key = str(name)
            idx = name_to_index.get(key)
            if idx is None:
                missing.append(key)
                continue
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)

        if missing:
            raise KeyError(
                "Unknown channel names in subset: " + ", ".join(sorted(missing))
            )

        if not indices:
            raise ValueError("No valid channel names resolved for the subset.")

        return np.array(indices, dtype=np.int64)

    def _resolve_preset(self, option: str) -> List[str]:
        if option == "visual":
            return self._resolve_visual_channels()
        raise ValueError(f"Unknown channel subset preset '{option}'.")

    def _resolve_visual_channels(self) -> List[str]:
        prefixes = ("MLP", "MLO", "MRO", "MZP", "MZO")
        matches = [
            name
            for name in self._canonical_ch_names
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
        if matches:
            return matches

        # Fallback: pattern-based matching for generic CTF posterior sensors.
        pattern = re.compile(r"^M[LRZ][OP].*")
        matches = [name for name in self._canonical_ch_names if pattern.match(name)]
        if matches:
            return matches

        # Last resort: pick channels in the posterior third based on 2D layout.
        if self._canonical_pos_2d.size == 0:
            return []
        y_coords = self._canonical_pos_2d[:, 1]
        threshold = np.quantile(y_coords, 0.33)
        posterior_mask = y_coords <= threshold
        return [
            name for name, keep in zip(self._canonical_ch_names, posterior_mask) if keep
        ]

    # ------------------------------------------------------------------ #
    def _load_data(self, idx: int):
        dataset_key, session, chunk, start = self._resolve_index(idx)
        root = self.root_dirs[dataset_key]
        file_path = os.path.join(root, session, chunk)
        # Use cached loader to avoid repeated disk I/O for same chunk
        data_dict = _load_chunk_cached(file_path)

        data = data_dict["data"]
        window = data[:, start : start + self.length]
        if window.shape[1] < self.length:
            pad_width = self.length - window.shape[1]
            window = np.pad(window, ((0, 0), (0, pad_width)), mode="constant")

        canonical_size = len(self._canonical_ch_names)
        mapped = np.ones((canonical_size, self.length), dtype=window.dtype)
        mapped *= self.fill_value
        indices = self._get_session_indices(dataset_key, session, window.shape[0])

        if len(indices) != window.shape[0]:
            raise ValueError(
                f"Channel count mismatch for session {session} ({dataset_key}):"
                f" expected {len(indices)}, got {window.shape[0]}"
            )

        mapped[indices, :] = window

        subset_idx = self._subset_indices
        # missing_mask = ~np.isin(subset_idx, indices)
        # if np.any(missing_mask):
        # missing = [self._canonical_ch_names[i] for i in subset_idx[missing_mask]]

        subset_data = mapped[subset_idx, :]
        x = torch.from_numpy(subset_data)

        data_dict["canonical_indices"] = torch.from_numpy(indices)
        data_dict["indices"] = torch.arange(x.shape[0], dtype=torch.int64)
        return x, data_dict


class ChunkDatasetJIT(ChunkDataset):
    def __init__(self, *args, quant_levels: int = 256, max_val: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_levels = quant_levels
        self.max_val = max_val

    def __getitem__(self, idx: int):  # type: ignore[override]
        x, _ = self._load_data(idx)

        # need to clip
        x = x / self.max_val
        x = mulaw_torch(x, self.quant_levels - 1)

        inputs = x[:, :-1]
        targets = x[:, 1:]

        return inputs, targets


class ChunkDatasetForecastCont(ChunkDataset):
    def __getitem__(self, idx: int):  # type: ignore[override]
        return super().__getitem__(idx, long=False)


class ChunkDatasetReconstruction(ChunkDataset):
    def __init__(
        self,
        *args,
        transpose: bool = False,
        use_tokenized: bool = False,
        tokenized_root: Optional[Union[str, Mapping[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transpose = transpose
        self.use_tokenized = bool(use_tokenized)
        self.tokenized_roots: Optional[Dict[str, str]] = None
        if self.use_tokenized:
            self.tokenized_roots = self._normalise_tokenized_roots(tokenized_root)

    def _normalise_tokenized_roots(
        self, tokenized_root: Optional[Union[str, Mapping[str, str]]]
    ) -> Dict[str, str]:
        if tokenized_root is None:
            raise ValueError("tokenized_root must be provided when use_tokenized=True.")

        if isinstance(tokenized_root, Mapping):
            roots = {str(k): str(v) for k, v in tokenized_root.items()}
        else:
            base = str(tokenized_root)
            if len(self.root_dirs) == 1:
                key = next(iter(self.root_dirs))
                roots = {key: base}
            else:
                roots = {key: os.path.join(base, key) for key in self.root_dirs}

        for key in self.root_dirs:
            if key not in roots:
                raise KeyError(
                    f"Missing tokenized_root for dataset key '{key}'. "
                    "Provide a mapping or a base directory."
                )

        for key, path in roots.items():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Tokenized root not found for {key}: {path}")
        return roots

    def _tokenized_path(self, dataset_key: str, session: str, chunk: str) -> str:
        if self.tokenized_roots is None:
            raise RuntimeError("Tokenized roots are not configured.")
        root = self.tokenized_roots[dataset_key]
        return os.path.join(root, session, chunk)

    def _load_tokenized_codes(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_key, session, chunk, start = self._resolve_index(idx)
        file_path = self._tokenized_path(dataset_key, session, chunk)
        token_dict = _load_tokenized_chunk_cached(file_path)

        starts = np.asarray(token_dict.get("starts"))
        if starts.ndim != 1:
            raise ValueError("Tokenized starts must be a 1D array.")
        pos = int(np.searchsorted(starts, start))
        if pos >= starts.size or int(starts[pos]) != int(start):
            raise KeyError(f"Start {start} not found in tokenized chunk {file_path}.")

        codes_arr = token_dict.get("codes")
        if codes_arr is None:
            raise KeyError(f"No codes found in tokenized chunk {file_path}.")

        codes = torch.from_numpy(codes_arr[pos]).long()
        return {"codes": codes}

    def __getitem__(self, idx: int):  # type: ignore[override]
        if self.use_tokenized:
            inputs = self._load_tokenized_codes(idx)
            return inputs, inputs["codes"]

        x, _ = self._load_data(idx)
        x = x.float()
        if self.transpose:
            x = x.permute(1, 0)

        pos = torch.from_numpy(self.pos_2d).float()
        ch_type = torch.from_numpy(self.ch_type).long()

        inputs = (x, pos, ch_type)

        return inputs, x


class ChunkDatasetSensor3D(ChunkDatasetReconstruction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pos_3d is None or self.ori_3d is None:
            raise ValueError(
                "pos_3d and ori_3d must be provided for sensor geometry datasets."
            )
        pos_orientation = np.concatenate([self.pos_3d, self.ori_3d], axis=1)
        self._pos_orientation = torch.from_numpy(pos_orientation.astype(np.float32))
        unique_types = {
            label: idx for idx, label in enumerate(sorted(set(self.ch_type_labels)))
        }
        mapped_types = np.array(
            [unique_types[label] for label in self.ch_type_labels], dtype=np.int64
        )
        self._sensor_type = torch.from_numpy(mapped_types)

    def __getitem__(self, idx: int):  # type: ignore[override]
        if self.use_tokenized:
            inputs = self._load_tokenized_codes(idx)
            return inputs, inputs["codes"]

        x, _ = self._load_data(idx)
        x = x.float()
        if self.transpose:
            x = x.permute(1, 0)

        inputs = (x, self._pos_orientation, self._sensor_type)
        return inputs, x


class ChunkDatasetSensorPos(ChunkDataset):
    def __getitem__(self, idx):
        dataset_key, session, chunk, _ = self._resolve_index(idx)
        file_path = os.path.join(self.root_dirs[dataset_key], session, chunk)
        data_dict = np.load(file_path, allow_pickle=True).item()

        return data_dict["ch_names"], data_dict["pos_2d"]


class ChunkDatasetImageReconstruction(ChunkDataset):
    """Dataset that maps sensor channels into a sparse HxW image (default 32x32). Each
    channel's value is written to the pixel closest to its 2-D position in ``pos_2d``.
    The spatial layout of the MEG helmet is thus roughly preserved inside an image that
    can be processed by vision models.

    Input  : x ∈ R^{CxT} Output : img ∈ R^{HxWxT}  (sparse - most pixels are zero)
    """

    def __init__(
        self, *args, return_mask: bool = False, transpose: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.postprocessor = self.make_postprocessor()
        self.row_idx = self.postprocessor.row_idx
        self.col_idx = self.postprocessor.col_idx
        self.return_mask = return_mask
        self.transpose = transpose

    def __getitem__(self, idx: int):
        x, data_dict = self._load_data(idx)  # x: [C, T]
        x = x.float()

        H = W = self.image_size
        _, T = x.shape
        img = torch.ones((H, W, T), dtype=x.dtype) * self.fill_value

        # Vectorised scatter – assign each channel to its pixel across time
        img[self.row_idx, self.col_idx, :] = x

        if self.transpose:
            img = img.permute(2, 0, 1)

        input_ret = img
        if self.has_condition:
            cond = data_dict["condition"][None, None, :]
            input_ret = (img, cond)

        if self.return_mask:
            input_dict = {
                "inputs": input_ret,
                "row_idx": self.row_idx,
                "col_idx": self.col_idx,
            }
            return input_dict, img

        return input_ret, img


class ChunkDatasetImageQuantized(ChunkDatasetImageReconstruction):
    def __getitem__(self, idx: int):
        input_ret, img = super().__getitem__(idx)
        inputs = img[..., :-1].long()
        target = img[..., 1:].long()

        if isinstance(input_ret, tuple):
            cond = input_ret[1][..., :-1]
            return (inputs, cond), target

        return inputs, target


class ChunkDatasetImage01(ChunkDatasetImageReconstruction):
    """Dataset that maps sensor channels into a sparse HxW image (default 32x32) and
    quantizes the values to 0-256. Each channel's value is written to the pixel closest
    to its 2-D position in ``pos_2d``.  The spatial layout of the MEG helmet is thus
    roughly preserved inside an image that can be processed by vision models.

    Input  : x ∈ R^{CxT} Output : img ∈ {0,1}^{HxWxT}  (sparse - most pixels are zero)

    The dataset returns (*img*[:, :, :-1], *img*[:, :, 1:]) so that models can learn to
    predict the next timestep given the past.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        postprocessor = self.make_postprocessor()
        self.row_idx = postprocessor.row_idx
        self.col_idx = postprocessor.col_idx

    def __getitem__(self, idx: int):
        input_ret, img = super().__getitem__(idx)

        # rescale to 0-1
        img = (img + 10) / 20  # assume 5std clipping

        img = img.permute(2, 0, 1).unsqueeze(1)  # T, 1, H, W

        if isinstance(input_ret, tuple):
            cond = input_ret[1].permute(2, 0, 1).unsqueeze(1)
            return (img, cond), img

        return img, img


class PostprocessorTHW(Postprocessor):
    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # gather per time slice
        return x[..., self.row_idx, self.col_idx].transpose(-1, -2)


class PostprocessorInterpolator:
    def __init__(
        self, interpolator: GaussianSensorInterpolator, postprocessor: Postprocessor
    ):
        self.interpolator = interpolator
        self.postprocessor = postprocessor

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        return self.interpolator.inverse(x)

    def __call__(self, *tensors, gaussian: bool = True):
        if gaussian:
            return tuple(map(self.reshape, tensors))
        else:
            return self.postprocessor(*tensors)


class ChunkDatasetInterpolatedImage(ChunkDataset):
    """Dataset that interpolates sensor values into dense image frames using a Gaussian
    kernel over the sensor layout.

    This smooths sparse helmet measurements into a continuous 2-D map suitable for
    vision models.
    """

    def __init__(
        self,
        *args,
        image_size: int = 32,
        sigma_scale: float = 0.75,
        r_max_factor: float = 4.0,
        normalize: bool = False,
        min_std: float = 1e-6,
        **kwargs,
    ):
        super().__init__(*args, image_size=image_size, **kwargs)
        self.interpolator = GaussianSensorInterpolator(
            self.pos_2d,
            image_size=image_size,
            sigma_scale=sigma_scale,
            r_max_factor=r_max_factor,
        )
        self.normalize = normalize
        self.min_std = float(min_std)

        postprocessor = PostprocessorTHW(self.pos_2d, image_size)
        self.row_idx = postprocessor.row_idx
        self.col_idx = postprocessor.col_idx

        self.postprocessor = PostprocessorInterpolator(self.interpolator, postprocessor)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        mean = x.mean()
        std = x.std().clamp_min(self.min_std)
        return (x - mean) / std

    def __getitem__(self, idx: int):
        x, _ = self._load_data(idx)  # [C, T]
        x = self._normalize(x.float())

        img = self.interpolator(x)  # [T, H, W]
        img = img.unsqueeze(0)  # [1, T, H, W]

        return img, img
