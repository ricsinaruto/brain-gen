import math
import torch
from torch import nn
from typing import Tuple
import numpy as np
import json
from transformers import AutoTokenizer
from einops import rearrange


class AmplitudeTokenizer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, _, self.C, _ = x.shape
        x = x.reshape(x.shape[0], -1)  # (B, L)
        return {"codes": x}

    def forecast_tokens_per_step(
        self,
        _encoded: torch.Tensor,
        _raw_input: torch.Tensor,
        reduced_shape: Tuple[int, int, int],
    ) -> int:
        # For flattened grid, one timestep = H'*W' tokens.
        return int(np.prod(reduced_shape[1:]))

    def forecast_strip_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        return seq.reshape(seq.shape[0], -1, self.C)


class AmplitudeTokenizerMix(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, _, self.C, _ = x.shape
        x = rearrange(x, "b t c w -> (b c w) t", c=self.C)
        return {"codes": x}

    def forecast_tokens_per_step(self, *args, **kwargs) -> int:
        return 1

    def forecast_strip_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        return rearrange(seq, "(b c) t -> b t c", c=self.C)


class DelimitedTokenizer(nn.Module):
    def __init__(
        self,
        delimiter_id: int = 256,
        input_shape: Tuple[int, ...] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.delimiter_id = int(delimiter_id)
        self.input_shape = input_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor of shape (B, T, ...) where the trailing dimensions encode C
        channels for each timestep.

        Returns:     Tensor of shape (B, T * C + T - 1) where a delimiter token
        separates     every timestep's C-channel block.
        """
        if x.ndim < 3:
            raise ValueError(
                f"DelimitedTokenizer expects input with at least 3 dims (B, T, C...), "
                f"got shape {tuple(x.shape)}."
            )

        batch, timesteps = x.shape[:2]
        per_timestep = x.reshape(batch, timesteps, -1)

        delimiter = per_timestep.new_full((batch, timesteps, 1), self.delimiter_id)
        tokens = torch.cat([per_timestep, delimiter], dim=2).reshape(batch, -1)

        # add one more delimiter to the start
        delimiter = delimiter.new_full((batch, 1), self.delimiter_id)
        tokens = torch.cat([delimiter, tokens], dim=1)

        # Drop the final delimiter so delimiters appear strictly between timesteps.
        return {"codes": tokens}

    def forecast_tokens_per_step(
        self,
        encoded: torch.Tensor,
        raw_input: torch.Tensor,
        reduced_shape: Tuple[int, int, int],
    ) -> int:
        """Number of tokens per timestep, including the delimiter."""
        # Prefer inferring from raw input to preserve exact flattening semantics
        if raw_input is not None and raw_input.ndim >= 3:
            per_timestep = int(np.prod(raw_input.shape[2:]))
        elif self.input_shape is not None:
            per_timestep = int(np.prod(self.input_shape[1:]))
        else:
            per_timestep = int(np.prod(reduced_shape[1:]))
        return per_timestep + 1  # +1 for delimiter

    def forecast_strip_tokens(
        self, seq: torch.Tensor, tokens_per_step: int
    ) -> torch.Tensor:
        """Remove delimiters from a flattened forecast sequence."""
        if tokens_per_step <= 1:
            raise ValueError(
                "DelimitedTokenizer expects at least one data token per timestep."
            )

        delimiter_id = self.delimiter_id
        has_start_delim = seq.shape[1] > 0 and torch.all(seq[:, 0] == delimiter_id)
        offset = 1 if has_start_delim else 0
        usable_len = seq.shape[1] - offset
        if usable_len < 0 or usable_len % tokens_per_step != 0:
            raise ValueError(
                "Delimited sequence length not divisible by tokens_per_step; "
                "cannot strip delimiters cleanly."
            )

        steps_total = usable_len // tokens_per_step
        if steps_total == 0:
            return seq.new_zeros(seq.shape[0], 0, dtype=seq.dtype)

        body = seq[:, offset : offset + steps_total * tokens_per_step]
        grouped = body.view(seq.shape[0], steps_total, tokens_per_step)
        return grouped[:, :, :-1].reshape(seq.shape[0], -1)


class BPETokenizer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.tokenizer = None
        self.lookup: list[str] = []
        self._char_to_idx: dict[str, int] = {}
        self.num_channels: int | None = None
        self.group_size: int = int(max(1, kwargs.get("group_size", 1)))

        input_shape = kwargs.get("input_shape")
        if input_shape is not None and len(input_shape) >= 2:
            self.num_channels = int(np.prod(input_shape[1:]))

        if kwargs.get("tokenizer_path") is not None:
            self.load_tokenizer(kwargs["tokenizer_path"])

    def load_tokenizer(self, path: str):
        self.tokenizer_path = path
        self.tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")

        with open(f"{path}/char_lookup_codepoints.json", "r") as f:
            lookup_codes = json.load(f)

        self.lookup = [chr(x) for x in lookup_codes]
        for idx, ch in enumerate(self.lookup):
            if ch not in self._char_to_idx:
                self._char_to_idx[ch] = idx

    def encode(
        self, x: tuple[torch.Tensor, ...] | list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        codes = x[0]
        attention_mask = x[1] if len(x) > 1 else None
        channel_ids = x[2] if len(x) > 2 else None

        out = {"codes": codes}
        if attention_mask is not None:
            out["attention_mask"] = attention_mask
        if channel_ids is not None:
            out["channel_ids"] = channel_ids
        return out

    def _strip_padding(self, seq: torch.Tensor) -> torch.Tensor:
        """Remove trailing pad tokens to avoid decoding padding."""

        if self.tokenizer is None:
            return seq

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            return seq

        # Find last non-pad token (1D seq expected)
        non_pad = (seq != pad_id).nonzero(as_tuple=True)[0]
        if non_pad.numel() == 0:
            return seq.new_zeros(0, dtype=seq.dtype, device=seq.device)

        last = int(non_pad.max().item())
        return seq[: last + 1]

    def _decode_to_array(
        self, seq: torch.Tensor, text_path: str | None = None
    ) -> torch.Tensor:
        """Invert the BPEDataset packing:

        decoded text → channels × timesteps characters → integer bins.
        """
        channels = self.num_channels
        if channels is None:
            raise ValueError("Number of channels must be provided and positive.")

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for decoding BPE tokens.")
        if not self.lookup:
            raise RuntimeError("Lookup table is missing; cannot map characters back.")

        if channels is None or channels <= 0:
            raise ValueError("Number of channels must be provided and positive.")

        seq = self._strip_padding(seq)
        if seq.numel() == 0:
            return seq.new_zeros((0, channels), dtype=torch.long)

        text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True).strip()
        text = text[:1000000]

        path = f"{self.tokenizer_path}/text{len(text)}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        # optionally read a text file
        if text_path is not None:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()

        tokens = text.split()
        if not tokens:
            return seq.new_zeros((0, channels), dtype=torch.long)

        flat_chars = "".join(tokens)
        block_size = channels * max(1, self.group_size)
        usable_len = (len(flat_chars) // block_size) * block_size
        if usable_len == 0:
            return seq.new_zeros((0, channels), dtype=torch.long)
        flat_chars = flat_chars[:usable_len]

        channel_strings = ["" for _ in range(channels)]
        for offset in range(0, usable_len, block_size):
            block = flat_chars[offset : offset + block_size]
            for ch in range(channels):
                start = ch * self.group_size
                end = start + self.group_size
                channel_strings[ch] += block[start:end]

        timesteps = len(channel_strings[0])
        decoded = seq.new_zeros((timesteps, channels), dtype=torch.long)

        for ch_idx, ch_seq in enumerate(channel_strings):
            if len(ch_seq) != timesteps:
                raise ValueError(
                    "Channel sequence length mismatch after trimming to blocks."
                )

            indices = []
            for ch in ch_seq:
                if ch not in self._char_to_idx:
                    print(f"Character '{ch}' not found in lookup.")
                    indices.append(len(self.lookup) // 2 - 1)
                    continue
                indices.append(self._char_to_idx[ch])

            decoded[:, ch_idx] = seq.new_tensor(indices, dtype=torch.long)

        return decoded

    def forecast_tokens_per_step(
        self,
        _encoded: torch.Tensor,
        _raw_input: torch.Tensor,
        reduced_shape: Tuple[int, int, int],
    ) -> int:
        # Estimate how many token IDs correspond to one original timestep by
        # decoding the current context and comparing sequence lengths.
        channels = self.num_channels
        if channels is None:
            channels = int(np.prod(reduced_shape[1:]))
            self.num_channels = channels

        encoded = _encoded
        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(0)

        try:
            trimmed = self._strip_padding(encoded[0])
            decoded = self._decode_to_array(trimmed, channels=channels)
            timesteps = decoded.shape[0]
            if timesteps > 0:
                tokens_len = int(trimmed.shape[0])
                tokens_per_step = int(math.ceil(tokens_len / timesteps))
                return max(tokens_per_step, 1)
        except Exception:
            pass

        # Fallback to channel count if decoding fails.
        return int(np.prod(reduced_shape[1:]))

    def forecast_strip_tokens(
        self, seq: torch.Tensor, _tokens_per_step: int
    ) -> torch.Tensor:
        # Forecasting only uses batch size 1; simplify accordingly.
        if seq.ndim == 1:
            seq = seq.unsqueeze(0)
        if seq.shape[0] != 1:
            raise ValueError(
                f"BPETokenizer.forecast_strip_tokens expects batch size 1, "
                f"batch size 1, got {seq.shape[0]}"
            )

        decoded = self._decode_to_array(seq[0])
        return decoded.unsqueeze(0)
