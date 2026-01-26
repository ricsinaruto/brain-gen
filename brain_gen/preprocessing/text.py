from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from wordfreq import get_frequency_dict


class TextProcessor:
    """Utility class for turning quantized arrays into text."""

    def __init__(
        self,
        num_bins: int = 256,
        collapse_bin: int | None = None,
    ) -> None:
        if num_bins <= 0:
            raise ValueError("num_bins must be positive")

        self.num_bins = int(num_bins)
        if collapse_bin is None:
            collapse_bin = self.num_bins - 1

        if collapse_bin < 0 or collapse_bin >= self.num_bins:
            raise ValueError("collapse_bin must be within [0, num_bins)")

        self.collapse_bin = int(collapse_bin)

    @staticmethod
    def chunk_sort_key(filename: str) -> tuple[int, Any]:
        """Numeric-aware sort key for chunk filenames."""
        stem = Path(filename).stem
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    def _target_dtype(self) -> np.dtype:
        """Choose an integer dtype capable of representing num_bins."""
        if self.num_bins <= np.iinfo(np.uint8).max + 1:
            return np.uint8
        if self.num_bins <= np.iinfo(np.uint16).max + 1:
            return np.uint16
        return np.int64

    def validate_array(self, array: np.ndarray) -> np.ndarray:
        """Validate that array is (C, T) of integers within [0, num_bins)."""
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array (C, T); got shape {arr.shape}")

        if not np.issubdtype(arr.dtype, np.integer):
            if not np.all(np.isfinite(arr)):
                raise ValueError("Array contains non-finite values")
            if not np.all(arr == np.floor(arr)):
                raise ValueError("Array must contain integer-valued samples")
            arr = arr.astype(self._target_dtype())

        if np.any(arr < 0) or np.any(arr >= self.num_bins):
            raise ValueError(
                f"Array values must be in [0, {self.num_bins - 1}] "
                f"but found range [{arr.min()}, {arr.max()}]"
            )

        return arr.astype(self._target_dtype(), copy=False)

    def normalize_array(self, array: np.ndarray) -> np.ndarray:
        """Validate and collapse the reserved bin into 0."""
        arr = self.validate_array(array)
        if self.collapse_bin is None:
            return arr
        arr = arr.copy()
        arr[arr == self.collapse_bin] = 0
        return arr

    def count_bins(self, array: np.ndarray) -> np.ndarray:
        """Count values across the array after collapsing the reserved bin."""
        arr = self.normalize_array(array)
        return np.bincount(arr.ravel(), minlength=self.num_bins)

    def normalize_counts(self, counts: np.ndarray) -> np.ndarray:
        """Pad/merge counts so they are compatible with this processor."""
        counts_arr = np.asarray(counts, dtype=np.int64)
        if counts_arr.ndim != 1:
            counts_arr = counts_arr.reshape(-1)
        if counts_arr.size < self.num_bins:
            counts_arr = np.pad(
                counts_arr, (0, self.num_bins - counts_arr.size), constant_values=0
            )

        counts_arr = counts_arr.copy()
        if self.collapse_bin is not None:
            counts_arr[0] += counts_arr[self.collapse_bin]
            counts_arr[self.collapse_bin] = 0
        return counts_arr

    @staticmethod
    @lru_cache(maxsize=None)
    def byte_frequency_ranking(
        lang: str = "en", wordlist: str = "best", k: int = 256, add_space: bool = False
    ) -> list[str]:
        """Returns:

        chars: list of length k with the most frequent characters
        """
        word_freqs = get_frequency_dict(lang, wordlist=wordlist)
        counts = Counter()

        for word, pw in word_freqs.items():
            for ch in word:
                counts[ch] += pw

            if add_space:
                counts[" "] += pw

        if not add_space:
            counts.pop(" ", None)

        most_common = counts.most_common(k)
        return [ch for ch, _ in most_common]

    def build_char_lookup(self, counts: np.ndarray) -> list[str]:
        """Build bin→character lookup based on bin counts and global char
        frequencies."""
        counts = counts.copy()
        counts[0] += counts[self.collapse_bin]  # merge 255 into 0
        counts[self.collapse_bin] = 0

        bin_order = np.argsort(-counts[: self.num_bins], kind="stable")

        char_ranking = self.byte_frequency_ranking(k=max(len(bin_order), self.num_bins))
        if len(char_ranking) < len(bin_order):
            raise RuntimeError(
                f"Char ranking must provide at least {len(bin_order)} symbols."
            )

        lookup = [""] * self.num_bins
        for rank, bin_idx in enumerate(bin_order):
            lookup[bin_idx] = char_ranking[rank]

        if self.collapse_bin is not None:
            lookup[self.collapse_bin] = " "

        return lookup

    def array_to_text(self, array: np.ndarray, lookup: list[str]) -> str:
        """Convert a (C, T) array into a space-separated text document."""
        if len(lookup) < self.num_bins:
            raise ValueError(
                f"Lookup must provide at least {self.num_bins}; got {len(lookup)}"
            )

        arr = self.normalize_array(array)
        words = ["".join(lookup[val] for val in arr[:, t]) for t in range(arr.shape[1])]
        return " ".join(words)

    def _char_to_bin_map(self, lookup: list[str]) -> dict[str, int]:
        """Build a mapping from character to bin index, favoring the first
        occurrence."""
        char_to_bin: dict[str, int] = {}
        for idx, ch in enumerate(lookup):
            if ch not in char_to_bin:
                char_to_bin[ch] = idx
        return char_to_bin

    def text_to_array(
        self, text: str, lookup: list[str], channels: int | None = None
    ) -> np.ndarray:
        """Convert space-separated text back into a (C, T) array of bin indices.

        Args:     text: Text document produced by `array_to_text`.     lookup: Lookup
        table used to encode the text.     channels: Expected number of channels. If
        omitted, inferred from the first token.
        """
        if len(lookup) < self.num_bins:
            raise ValueError(
                f"Lookup must provide at least {self.num_bins}; got {len(lookup)}"
            )

        tokens = text.strip().split()
        if not tokens:
            raise ValueError("Text is empty; nothing to decode.")

        first_token_len = len(tokens[0])
        if channels is None:
            channels = first_token_len

        if channels <= 0:
            raise ValueError("channels must be positive")

        char_to_bin = self._char_to_bin_map(lookup)
        dtype = self._target_dtype()
        arr = np.zeros((channels, len(tokens)), dtype=dtype)

        for t, token in enumerate(tokens):
            if len(token) != channels:
                raise ValueError(
                    f"Inconsistent token length at position {t}: "
                    f"expected {channels}, got {len(token)}"
                )
            for c, ch in enumerate(token):
                if ch not in char_to_bin:
                    raise ValueError(f"Character '{ch}' not found in lookup.")
                arr[c, t] = char_to_bin[ch]

        return arr.astype(dtype, copy=False)


class GroupedTextProcessor(TextProcessor):
    """Text processor that groups time steps per channel before rendering."""

    def __init__(
        self,
        num_bins: int = 256,
        collapse_bin: int | None = None,
        group_size: int = 5,
        channels_per_line: int = 68,
    ) -> None:
        super().__init__(num_bins=num_bins, collapse_bin=collapse_bin)
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if channels_per_line <= 0:
            raise ValueError("channels_per_line must be positive")
        self.group_size = int(group_size)
        self.channels_per_line = int(channels_per_line)

    def array_to_text(self, array: np.ndarray, lookup: list[str]) -> str:
        """Render grouped channel slices with spaces and line breaks."""
        if len(lookup) < self.num_bins:
            raise ValueError(
                f"Lookup must provide at least {self.num_bins}; got {len(lookup)}"
            )

        arr = self.normalize_array(array)
        channels, timesteps = arr.shape
        groups: list[str] = []

        for start in range(0, timesteps, self.group_size):
            end = min(start + self.group_size, timesteps)
            channel_tokens = [
                "".join(lookup[val] for val in arr[ch, start:end])
                for ch in range(channels)
            ]

            lines = []
            for block_start in range(0, len(channel_tokens), self.channels_per_line):
                lines.append(
                    " ".join(
                        channel_tokens[
                            block_start : block_start + self.channels_per_line
                        ]
                    )
                )

            groups.append("\n".join(lines))

        return "\n".join(groups)

    def text_to_array(
        self, text: str, lookup: list[str], channels: int | None = None
    ) -> np.ndarray:
        """Decode grouped text back into a (C, T) array.

        Args:     text: Grouped text produced by `array_to_text`.     lookup: Lookup
        table used to encode the text.     channels: Number of channels. This must be
        provided for grouped decoding.
        """
        if len(lookup) < self.num_bins:
            raise ValueError(
                f"Lookup must provide at least {self.num_bins}; got {len(lookup)}"
            )

        if channels is None:
            raise ValueError(
                "channels must be provided when decoding grouped text because "
                "it cannot be inferred reliably from formatting."
            )
        if channels <= 0:
            raise ValueError("channels must be positive")

        tokens = text.strip().split()
        if not tokens:
            raise ValueError("Text is empty; nothing to decode.")

        if len(tokens) % channels != 0:
            raise ValueError(
                f"Token count {len(tokens)} is not divisible by channels {channels}."
            )

        char_to_bin = self._char_to_bin_map(lookup)
        dtype = self._target_dtype()
        num_groups = len(tokens) // channels
        channel_series = [[] for _ in range(channels)]

        for g in range(num_groups):
            group_tokens = tokens[g * channels : (g + 1) * channels]
            lengths = {len(tok) for tok in group_tokens}
            if len(lengths) != 1:
                raise ValueError(
                    f"Inconsistent token lengths within group {g}: {sorted(lengths)}"
                )
            for ch_idx, tok in enumerate(group_tokens):
                for ch in tok:
                    if ch not in char_to_bin:
                        raise ValueError(f"Character '{ch}' not found in lookup.")
                    channel_series[ch_idx].append(char_to_bin[ch])

        lengths = {len(seq) for seq in channel_series}
        if len(lengths) != 1:
            raise ValueError(
                f"Channel sequences differ in length after decoding: {sorted(lengths)}"
            )

        total_timesteps = lengths.pop()
        arr = np.zeros((channels, total_timesteps), dtype=dtype)
        for ch_idx, seq in enumerate(channel_series):
            arr[ch_idx] = np.asarray(seq, dtype=dtype)

        return arr


class TemporalTextProcessor(GroupedTextProcessor):
    """Text processor that groups time steps per channel before rendering."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def array_to_text(self, array: np.ndarray, lookup: list[str]) -> str:
        """Render grouped channel slices with spaces and line breaks."""
        if len(lookup) < self.num_bins:
            raise ValueError(
                f"Lookup must provide at least {self.num_bins}; got {len(lookup)}"
            )

        arr = self.normalize_array(array)
        channels = arr.shape[0]

        channel_tokens = [
            "".join(lookup[val] for val in arr[ch]) for ch in range(channels)
        ]

        return "\n".join(channel_tokens)

    def text_to_array(
        self, text: str, lookup: list[str], channels: int | None = None
    ) -> np.ndarray:
        """Decode grouped text back into a (C, T) array.

        Args:     text: Grouped text produced by `array_to_text`.     lookup: Lookup
        table used to encode the text.     channels: Number of channels. This must be
        provided for grouped decoding.
        """
        if len(lookup) < self.num_bins:
            raise ValueError(
                f"Lookup must provide at least {self.num_bins}; got {len(lookup)}"
            )

        if channels is None:
            raise ValueError(
                "channels must be provided when decoding grouped text because "
                "it cannot be inferred reliably from formatting."
            )
        if channels <= 0:
            raise ValueError("channels must be positive")

        tokens = text.strip().split()
        if not tokens:
            raise ValueError("Text is empty; nothing to decode.")

        if len(tokens) % channels != 0:
            raise ValueError(
                f"Token count {len(tokens)} is not divisible by channels {channels}."
            )

        char_to_bin = self._char_to_bin_map(lookup)
        dtype = self._target_dtype()
        num_groups = len(tokens) // channels
        channel_series = [[] for _ in range(channels)]

        for g in range(num_groups):
            group_tokens = tokens[g * channels : (g + 1) * channels]
            lengths = {len(tok) for tok in group_tokens}
            if len(lengths) != 1:
                raise ValueError(
                    f"Inconsistent token lengths within group {g}: {sorted(lengths)}"
                )
            for ch_idx, tok in enumerate(group_tokens):
                for ch in tok:
                    if ch not in char_to_bin:
                        raise ValueError(f"Character '{ch}' not found in lookup.")
                    channel_series[ch_idx].append(char_to_bin[ch])

        lengths = {len(seq) for seq in channel_series}
        if len(lengths) != 1:
            raise ValueError(
                f"Channel sequences differ in length after decoding: {sorted(lengths)}"
            )

        total_timesteps = lengths.pop()
        arr = np.zeros((channels, total_timesteps), dtype=dtype)
        for ch_idx, seq in enumerate(channel_series):
            arr[ch_idx] = np.asarray(seq, dtype=dtype)

        return arr
