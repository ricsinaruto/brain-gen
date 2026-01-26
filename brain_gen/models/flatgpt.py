import torch

from torch import nn
from typing import Callable, Tuple
import numpy as np
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F

from ..training.lightning import LitModel
from .hf_adapters.llm import SmoLLM3, MiniMax  # noqa: F401
from .hf_adapters.vlm import (  # noqa: F401
    Qwen2_5_Video,
    Qwen2_5_VideoText,
    Qwen3_Video,
)
from ..layers.flatgpt_layers import (
    QuantizerEmbedding,
    JointRVQHead,
    TiedRVQHead,
    ListEmbedding,
    ChannelHead,
    MixEmbedding,
    MixHead,
    MixQuantizerEmbedding,
    MixRVQHead,
    EmbeddingCorruptor,
)
from .tokenizers.flat_tokenizers import (  # noqa: F401
    AmplitudeTokenizer,
    AmplitudeTokenizerMix,
    DelimitedTokenizer,
    BPETokenizer,
)


def _compute_reduced_shape_from_init(
    args: tuple, kwargs: dict
) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
    input_shape = kwargs.get("input_shape")
    if input_shape is None and len(args) > 4:
        input_shape = args[4]
    if input_shape is None:
        raise ValueError("input_shape must be provided for RVQ reduced shape override.")

    spatial_reduction = kwargs.get("spatial_reduction")
    if spatial_reduction is None and len(args) > 6:
        spatial_reduction = args[6]
    if spatial_reduction is None:
        spatial_reduction = 1
    if isinstance(spatial_reduction, int):
        spatial_reduction = (spatial_reduction, spatial_reduction)

    temporal_reduction = kwargs.get("temporal_reduction")
    if temporal_reduction is None and len(args) > 7:
        temporal_reduction = args[7]
    if temporal_reduction is None:
        temporal_reduction = 1

    reduced_shape = (
        input_shape[0] // temporal_reduction,
        input_shape[1] // spatial_reduction[0],
        input_shape[2] // spatial_reduction[1],
    )
    return reduced_shape, spatial_reduction


class FlatGPT(nn.Module):
    def __init__(
        self,
        trf_class: str,
        trf_args: dict,
        hidden_size: int,
        vocab_size: int,
        input_shape: Tuple[int, int, int],  # T, H, W
        input_type: str = "vector",  # vector, image
        spatial_reduction: int | Tuple[int, int] = 1,
        temporal_reduction: int = 1,
        tok_class: str = "AmplitudeTokenizer",  # amplitde, cosmos, brainomni, etc.
        tok_args: dict = None,
        tokenizer_path: str = None,
        train_tokenizer: bool = False,
        token_corruption_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(spatial_reduction, int):
            spatial_reduction = (spatial_reduction, spatial_reduction)

        self.train_tokenizer = train_tokenizer
        self.input_shape = input_shape
        self.input_type = input_type
        self.spatial_reduction = spatial_reduction
        self.temporal_reduction = temporal_reduction
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.block_size = trf_args.get("block_size", 1)
        self.reduced_shape = (
            input_shape[0] // temporal_reduction,
            input_shape[1] // spatial_reduction[0],
            input_shape[2] // spatial_reduction[1],
        )

        self.max_context_tokens = int(np.prod(self.reduced_shape))

        tok_args = tok_args or {}
        tok_args["temporal_reduction"] = temporal_reduction
        tok_args["spatial_reduction"] = spatial_reduction
        tok_args["input_shape"] = input_shape
        tok_args["vocab_size"] = vocab_size
        tok_args["hidden_size"] = hidden_size

        trf_args["hidden_size"] = hidden_size
        trf_args["vocab_size"] = vocab_size
        transformer_reduced_shape = trf_args.get("reduced_shape", self.reduced_shape)
        trf_args["reduced_shape"] = transformer_reduced_shape
        self.transformer_reduced_shape = transformer_reduced_shape
        if trf_class == "Qwen2_5_Video_TASA3D" and "input_shape" not in trf_args:
            trf_args["input_shape"] = input_shape

        # Load tokenizer if path is given
        if tokenizer_path is not None:
            lit = LitModel.load_from_checkpoint(tokenizer_path, strict=False)
            self.tokenizer = lit.model

            # check if model is compiled
            if hasattr(self.tokenizer, "_orig_mod"):
                self.tokenizer = self.tokenizer._orig_mod

        else:
            self.tokenizer = globals()[tok_class](**tok_args)

        # freeze tokenizer during autoregressive training (optional)
        if not self.train_tokenizer:
            for p in self.tokenizer.parameters():
                p.requires_grad_(False)

        self.tokenizer.eval()
        self.transformer = globals()[trf_class](**trf_args)
        self.pre_embedding = None

        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

        # tie embeddings
        self.head.weight = self.transformer.get_embed_layer().weight
        self._pre_embedding_accepts_chid = None
        self._head_accepts_chid = None
        self._transformer_accepts_cache = None
        self.token_corruption_cfg = self._init_token_corruption_cfg(
            token_corruption_cfg
        )
        if self.token_corruption_cfg.get("enabled", False):
            print(
                f"Token corruption enabled with probability "
                f"{self.token_corruption_cfg.get('p_end', 0.0)}"
            )
        self._validate_vlm_rope_setup()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # L corresponds to T'*H'*W', the product of reduced dimensions
        inputs = self._encode_tokens(x)  # (B, L)
        codes = inputs.pop("codes")
        model_codes = self._apply_token_corruption(codes)

        if self.pre_embedding is not None:
            embeds = self.pre_embedding(model_codes)
            trf_out = self.transformer(inputs_embeds=embeds, **inputs)
        else:
            trf_out = self.transformer(model_codes, **inputs)

        hidden = trf_out
        if isinstance(trf_out, tuple):
            hidden = trf_out[0]
        elif hasattr(trf_out, "last_hidden_state"):
            hidden = trf_out.last_hidden_state

        logits = self.head(hidden)
        logits = logits[:, : -self.block_size]
        targets = codes[:, self.block_size :]

        if logits.dim() > 3:
            logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
            targets = targets.reshape(targets.shape[0], -1)

        return logits, targets

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, dict) and ("codes" in x):
            return x

        if self.train_tokenizer:
            return self.tokenizer.encode(x)

        self.tokenizer.eval()
        with torch.no_grad():
            return self.tokenizer.encode(x)

    def _init_token_corruption_cfg(self, cfg: dict | None) -> dict:
        """Normalise corruption settings; disabled by default."""
        base_cfg = {"enabled": False, "p_start": 0.0, "p_end": 0.0}
        if not cfg:
            return base_cfg

        parsed = dict(base_cfg)
        parsed.update(cfg)
        parsed["p_start"] = max(0.0, float(parsed.get("p_start", base_cfg["p_start"])))
        parsed["p_end"] = max(0.0, float(parsed.get("p_end", base_cfg["p_end"])))
        parsed["enabled"] = bool(parsed.get("enabled", True)) and (
            parsed["p_start"] > 0.0 or parsed["p_end"] > 0.0
        )
        return parsed

    def update_token_corruption_cfg(self, cfg: dict | None) -> None:
        """Update token corruption settings after init (e.g., resume overrides)."""
        if cfg is None:
            return

        # Re-parse to keep defaults and normalisation consistent.
        self.token_corruption_cfg = self._init_token_corruption_cfg(cfg)
        if self.token_corruption_cfg.get("enabled", False):
            print(
                "Token corruption enabled with probability "
                f"{self.token_corruption_cfg.get('p_end', 0.0)}"
            )

    def _timestep_corruption_schedule(
        self, n_steps: int, device: torch.device
    ) -> torch.Tensor:
        """
        Exponential interpolation between start/end probabilities across timesteps.
        """
        cfg = self.token_corruption_cfg
        start = float(max(0.0, cfg.get("p_start", 0.0)))
        end = float(max(0.0, cfg.get("p_end", 0.0)))

        if n_steps <= 0 or (start == 0.0 and end == 0.0):
            return torch.zeros(n_steps, device=device, dtype=torch.float32)

        if n_steps == 1:
            return torch.tensor(
                [max(start, end)], device=device, dtype=torch.float32
            ).clamp_(0.0, 1.0)

        # Log-space interpolation for smoother progression across many steps.
        start_safe = torch.clamp(
            torch.tensor(start, device=device, dtype=torch.float32), min=1e-12
        )
        end_safe = torch.clamp(
            torch.tensor(end, device=device, dtype=torch.float32), min=1e-12
        )
        steps = torch.linspace(
            0.0, 1.0, steps=n_steps, device=device, dtype=torch.float32
        )

        log_start, log_end = torch.log(start_safe), torch.log(end_safe)
        probs = torch.exp(log_start + (log_end - log_start) * steps)  # [n_steps]

        # Preserve exact zeros at boundaries if requested.
        if start == 0.0:
            probs[0] = 0.0
        if end == 0.0:
            probs[-1] = 0.0

        return probs.clamp_(0.0, 1.0)

    def _apply_token_corruption(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Token corruption for robustness:
        - builds a timestep-level corruption probability schedule,
        - samples *block starts* using that schedule,
        - expands starts into *contiguous blocks* of timesteps,
        - broadcasts the timestep mask to all tokens in the timestep (C*Q tokens),
        - replaces corrupted tokens with random other tokens.

        Assumes the flattened token stream is contiguous in time such that
        each reduced timestep
        corresponds to `tokens_per_step = C * Q` consecutive tokens.
        """
        cfg = self.token_corruption_cfg
        if not (self.training and cfg.get("enabled", False)):
            return codes

        # How many flattened tokens correspond to 1 reduced timestep.
        # For [T, C=8, Q=4], this is 32.
        tokens_per_step = int(
            cfg.get("tokens_per_step", getattr(self, "tokens_per_step", 32))
        )
        tokens_per_step = max(1, tokens_per_step)

        # Block length in *timesteps* (contiguous forward span).
        block_len_steps = int(cfg.get("block_len_steps", 8))
        block_len_steps = max(1, block_len_steps)

        # Flatten everything after batch into a single token sequence.
        # (This supports inputs shaped like [B, L] or [B, T, C, Q] etc.)
        orig_shape = codes.shape
        if codes.dim() == 0:
            return codes
        if codes.dim() == 1:
            codes_flat = codes.unsqueeze(0)  # [1, L]
            had_batch = False
        else:
            codes_flat = codes.reshape(orig_shape[0], -1)  # [B, L]
            had_batch = True

        B, L = codes_flat.shape
        if L == 0:
            return codes

        # Number of reduced timesteps (ceil in case L isn't divisible).
        if L % tokens_per_step != 0:
            raise ValueError(
                f"L (={L}) must be divisible by tokens_per_step (={tokens_per_step}), "
                f"got remainder {L % tokens_per_step}"
            )
        n_steps = L // tokens_per_step
        if n_steps <= 0:
            return codes

        # Desired *per-timestep* corruption probability schedule (coverage probability).
        p_cov = self._timestep_corruption_schedule(n_steps, codes.device)  # [n_steps]
        if torch.all(p_cov == 0):
            return codes

        # Convert desired coverage probability p_cov into block-start
        # probability such that,
        # under an OR-over-last-K-steps model, marginal corruption per step is ~p_cov:
        #
        #   p_cov = 1 - (1 - p_start) ** block_len_steps
        #   => p_start = 1 - (1 - p_cov) ** (1 / block_len_steps)
        if block_len_steps > 1:
            p_start = 1.0 - torch.pow(1.0 - p_cov, 1.0 / float(block_len_steps))
        else:
            p_start = p_cov
        p_start = p_start.clamp_(0.0, 1.0)

        # Sample block starts independently per batch element and timestep.
        start_mask = torch.rand(
            (B, n_steps), device=codes.device, dtype=torch.float32
        ) < p_start.view(1, -1)
        if not start_mask.any():
            return codes

        # Expand starts into contiguous forward blocks of length block_len_steps.
        # A timestep t is corrupted if there was any start in {t, t-1, ..., t-(K-1)}.
        # Implement via causal 1D convolution: pad on the left only, kernel of ones.
        start_f = start_mask.float().unsqueeze(1)  # [B, 1, n_steps]
        kernel = torch.ones(
            (1, 1, block_len_steps), device=codes.device, dtype=torch.float32
        )
        start_f = F.pad(start_f, (block_len_steps - 1, 0))  # left pad only
        step_counts = F.conv1d(start_f, kernel)  # [B, 1, n_steps]
        step_mask = step_counts[:, 0, :] > 0  # [B, n_steps] boolean coverage mask

        # Broadcast per-step mask to per-token mask.
        tok_mask = step_mask.repeat_interleave(tokens_per_step, dim=1)[:, :L]  # [B, L]
        if not tok_mask.any():
            return codes

        # Replace corrupted tokens uniformly with a different token id.
        # (You said your vocab is already split per RVQ level, so cross-level
        # replacement isn't a concern.)
        replacement = torch.randint(
            0,
            self.vocab_size - 1,
            (B, L),
            device=codes.device,
            dtype=codes_flat.dtype,
        )
        replacement = replacement + (replacement >= codes_flat)

        codes_flat = torch.where(tok_mask, replacement, codes_flat)

        # Reshape back to original shape.
        if had_batch:
            return codes_flat.reshape(orig_shape)
        return codes_flat.squeeze(0)

    def _forecast_tokens_per_step(
        self, encoded: torch.Tensor, raw_input: torch.Tensor
    ) -> int:
        """Determine how many tokens correspond to one reduced timestep.

        Delegates to the tokenizer when available.
        """
        if hasattr(self.tokenizer, "forecast_tokens_per_step"):
            return int(
                self.tokenizer.forecast_tokens_per_step(
                    encoded, raw_input, self.reduced_shape
                )
            )
        return np.prod(self.reduced_shape[1:]) / self.temporal_reduction

    def _tokens_per_embedding(self, *args, **kwargs) -> int:
        return self._forecast_tokens_per_step(*args, **kwargs)

    def _forecast_strip_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        """Strip tokenizer-specific padding/markers after generation."""
        if hasattr(self.tokenizer, "forecast_strip_tokens"):
            return self.tokenizer.forecast_strip_tokens(seq)
        return seq

    def _is_vlm_transformer(self) -> bool:
        """Return True when the transformer exposes 3D RoPE position ids."""
        return hasattr(self.transformer, "reduced_shape") and hasattr(
            self.transformer, "_build_position_ids"
        )

    def _expected_transformer_reduced_shape_for_rope(self) -> Tuple[int, int, int]:
        """Return the transformer reduced shape expected by the token sequence."""
        return tuple(int(x) for x in self.reduced_shape)

    def _validate_vlm_rope_setup(self) -> None:
        """Ensure 3D RoPE grids line up with the token sequence layout."""
        if not self._is_vlm_transformer():
            return

        expected = self._expected_transformer_reduced_shape_for_rope()
        actual = getattr(self.transformer, "reduced_shape", None)
        if actual is None:
            raise ValueError("VLM transformer missing reduced_shape for 3D RoPE.")
        actual = tuple(int(x) for x in actual)

        if expected != actual:
            raise ValueError(
                "VLM reduced_shape mismatch: expected "
                f"{expected} to match token layout, got {actual}."
            )

    def _refresh_max_context_tokens(self) -> None:
        """Update max_context_tokens after a shape change."""
        self.max_context_tokens = int(np.prod(self.reduced_shape))

    def _maybe_update_tokenizer_shape(self) -> None:
        """Best-effort update of tokenizer shape metadata."""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return

        if hasattr(tokenizer, "input_shape"):
            tokenizer.input_shape = tuple(self.input_shape)
        if hasattr(tokenizer, "spatial_reduction"):
            tokenizer.spatial_reduction = tuple(self.spatial_reduction)
        if hasattr(tokenizer, "temporal_reduction"):
            tokenizer.temporal_reduction = int(self.temporal_reduction)
        if hasattr(tokenizer, "reduced_shape"):
            tokenizer.reduced_shape = tuple(self.reduced_shape)

    def _maybe_update_embedding_corruptor(self) -> None:
        if isinstance(self.pre_embedding, EmbeddingCorruptor):
            self.pre_embedding.n_time = int(self.reduced_shape[0])
            self.pre_embedding.n_space = int(self.reduced_shape[1])
            self.pre_embedding.n_levels = int(self.reduced_shape[2])

    def _maybe_update_transformer_shape(self) -> None:
        expected = self._expected_transformer_reduced_shape_for_rope()
        self.transformer_reduced_shape = expected
        if hasattr(self.transformer, "set_reduced_shape"):
            self.transformer.set_reduced_shape(expected)
        elif hasattr(self.transformer, "reduced_shape"):
            self.transformer.reduced_shape = expected

    def _maybe_update_transformer_rope(
        self,
        rope_theta: float | None = None,
        max_position_embeddings: int | None = None,
    ) -> None:
        if rope_theta is None and max_position_embeddings is None:
            return
        if hasattr(self.transformer, "set_rope_theta"):
            self.transformer.set_rope_theta(
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
            )
            return
        raise ValueError("Transformer does not support rope theta updates.")

    def resize_context(
        self,
        *,
        input_shape: Tuple[int, int, int] | None = None,
        spatial_reduction: int | Tuple[int, int] | None = None,
        temporal_reduction: int | None = None,
        rope_theta: float | None = None,
        max_position_embeddings: int | None = None,
    ) -> None:
        """Resize temporal context while reusing checkpointed weights.

        Only temporal length changes are supported; spatial dims and reductions must
        remain unchanged to keep embedding/head shapes compatible.
        """
        input_shape = tuple(input_shape or self.input_shape)
        temporal_reduction = (
            int(self.temporal_reduction)
            if temporal_reduction is None
            else int(temporal_reduction)
        )
        spatial_reduction = (
            self.spatial_reduction if spatial_reduction is None else spatial_reduction
        )
        if isinstance(spatial_reduction, int):
            spatial_reduction = (spatial_reduction, spatial_reduction)
        spatial_reduction = tuple(int(x) for x in spatial_reduction)

        if tuple(input_shape[1:]) != tuple(self.input_shape[1:]):
            raise ValueError(
                "resize_context only supports changing the temporal length "
                f"(input_shape spatial dims must stay {self.input_shape[1:]})."
            )
        if spatial_reduction != tuple(self.spatial_reduction):
            raise ValueError(
                "resize_context does not support changing spatial_reduction."
            )
        if temporal_reduction != int(self.temporal_reduction):
            raise ValueError(
                "resize_context does not support changing temporal_reduction."
            )

        new_reduced = (
            input_shape[0] // temporal_reduction,
            input_shape[1] // spatial_reduction[0],
            input_shape[2] // spatial_reduction[1],
        )
        old_reduced = tuple(self.reduced_shape)
        if new_reduced[1:] != old_reduced[1:]:
            raise ValueError(
                "resize_context only supports temporal expansion; "
                f"got reduced spatial {new_reduced[1:]} vs {old_reduced[1:]}."
            )
        if new_reduced[0] < old_reduced[0]:
            raise ValueError(
                "resize_context cannot shrink the reduced temporal length."
            )

        if (
            new_reduced == old_reduced
            and rope_theta is None
            and max_position_embeddings is None
        ):
            return

        self.input_shape = input_shape
        self.spatial_reduction = spatial_reduction
        self.temporal_reduction = temporal_reduction
        self.reduced_shape = new_reduced
        self._refresh_max_context_tokens()
        self._maybe_update_tokenizer_shape()
        self._maybe_update_embedding_corruptor()
        self._maybe_update_transformer_shape()
        self._maybe_update_transformer_rope(
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
        self._validate_vlm_rope_setup()

    def _validate_positive_setting(self, value: float, name: str) -> float:
        """Validate positive settings used during forecasting."""
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number, got {value!r}.") from exc

        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
        return value

    def _validate_integer_setting(self, value: float, name: str) -> int:
        value = self._validate_positive_setting(value, name)
        rounded = int(round(value))
        if abs(value - rounded) > 1e-6:
            raise ValueError(f"{name} must be an integer, got {value}.")
        return rounded

    def _forecast_expected_codes_rank(self) -> int:
        """Expected rank of token codes during forecasting."""
        return 2

    def _forecast_extra_validation(self, codes: torch.Tensor) -> None:
        """Variant-specific forecast validation."""
        return None

    def _validate_forecast_setup(
        self,
        codes: torch.Tensor,
        tokens_per_step: float,
        tokens_per_embedding: float,
    ) -> Tuple[float, int]:
        """Validate forecast shapes and integer assumptions."""
        if codes.dim() != self._forecast_expected_codes_rank():
            raise ValueError(
                "Unexpected token rank for forecast: "
                f"got {codes.dim()}, expected {self._forecast_expected_codes_rank()}."
            )

        self._forecast_extra_validation(codes)

        tokens_per_step = self._validate_positive_setting(
            tokens_per_step, "tokens_per_step"
        )
        tokens_per_embedding_i = self._validate_integer_setting(
            tokens_per_embedding, "tokens_per_embedding"
        )
        return tokens_per_step, tokens_per_embedding_i

    @torch.inference_mode()
    def _call_transformer(
        self,
        token_batch: torch.Tensor,
        cache_in=None,
        cache_enabled: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, tuple | None]:
        kwargs = dict(kwargs)
        # Drop helper-only args that should not be forwarded to the transformer
        chid = kwargs.pop("chid", None)
        if cache_in is not None:
            kwargs["past_key_values"] = cache_in
            kwargs["use_cache"] = True
        elif cache_enabled:
            kwargs["use_cache"] = True

        if self.pre_embedding is not None:
            if chid is not None and self._pre_embedding_accepts_chid is not False:
                try:
                    embeds = self.pre_embedding(token_batch, chid=chid)
                    self._pre_embedding_accepts_chid = True
                except TypeError as exc:
                    if "unexpected keyword" not in str(exc) or "chid" not in str(exc):
                        raise
                    self._pre_embedding_accepts_chid = False
                    embeds = self.pre_embedding(token_batch)
            else:
                embeds = self.pre_embedding(token_batch)

            kwargs["inputs_embeds"] = embeds
            token_batch = None

        if self._transformer_accepts_cache is False:
            kwargs.pop("past_key_values", None)
            kwargs.pop("use_cache", None)

        try:
            out = self.transformer(token_batch, **kwargs)
            if self._transformer_accepts_cache is None and (
                "past_key_values" in kwargs or "use_cache" in kwargs
            ):
                self._transformer_accepts_cache = True
        except TypeError as exc:
            msg = str(exc)
            is_unexpected_kw = "unexpected keyword" in msg and (
                "past_key_values" in msg or "use_cache" in msg
            )
            if not is_unexpected_kw:
                raise
            self._transformer_accepts_cache = False
            kwargs.pop("past_key_values", None)
            kwargs.pop("use_cache", None)
            out = self.transformer(token_batch, **kwargs)

        cache_out = None
        hidden_out = out
        if isinstance(out, tuple):
            hidden_out, cache_out = out
        elif hasattr(out, "last_hidden_state"):
            hidden_out = out.last_hidden_state
            cache_out = getattr(out, "past_key_values", None)

        return hidden_out, cache_out

    def _call_head(self, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
        chid = kwargs.pop("chid", None)
        if chid is not None and self._head_accepts_chid is not False:
            try:
                out = self.head(hidden, chid=chid)
                self._head_accepts_chid = True
                return out
            except TypeError as exc:
                if "unexpected keyword" not in str(exc) or "chid" not in str(exc):
                    raise
                self._head_accepts_chid = False
        return self.head(hidden)

    @torch.inference_mode()
    def forecast(
        self,
        initial_input: torch.Tensor,
        rollout_steps: int,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        max_context_tokens: int | None = None,
        use_cache: bool | None = None,
        sliding_window_overlap: float = 0.5,
    ) -> torch.Tensor:
        """Recursive autoregressive forecast starting from `initial_input`.

        Args:     initial_input: Raw input sample (shaped like training data) or integer
        token ids of shape (B, L).     rollout_steps: Number of *reduced* timesteps to
        roll out.     sample_fn: Callable applied to logits for the next token. When
        generating blocks, logits are flattened across the block dimension before
        calling `sample_fn`, which is expected to         accept a tensor of shape (N,
        vocab_size) and return integer         token ids. max_context_tokens: Optional
        user-specified sliding-window cap (tokens). Use `None`/`-1` to keep the
        window equal to the initial context length.     use_cache: Force- enable/disable
        KV caching; defaults to using caching when the transformer
        supports it.     sliding_window_overlap: If float, interpreted as the fraction
        of the         current window length to SHIFT by once the cacheable horizon is
        reached (smaller values keep more overlap). If int, used         directly as the
        shift/stride in tokens.

        Returns:     Token ids containing the original context followed by the generated
        rollout. If using DelimitedTokenizer, delimiter tokens     are stripped from the
        returned tensor.
        """

        def _make_chid_block(start_token: int, block_len: int, modulus: int):
            """Construct channel ids for embedding-aware subclasses.

            Only emit a scalar id when generating a single token so heads that expect an
            integer index (e.g., ChannelHead) continue to work.
            """
            if block_len == 1:
                return int(start_token % modulus)
            return None

        if rollout_steps < 0:
            raise ValueError("rollout_steps must be non-negative.")

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        seq = (
            initial_input[0]
            if isinstance(initial_input, (tuple, list))
            else initial_input
        )

        tokens = self._encode_tokens(initial_input)
        tokens = tokens["codes"]

        tokens = tokens.to(device)
        tokens_per_step = self._forecast_tokens_per_step(tokens, seq)
        toks_per_emb = self._tokens_per_embedding(tokens, seq)
        tokens_per_step, toks_per_emb = self._validate_forecast_setup(
            tokens, tokens_per_step, toks_per_emb
        )
        if tokens_per_step <= 0:
            raise ValueError("Invalid tokens_per_step; computed non-positive value.")

        total_new_tokens = int(rollout_steps * tokens_per_step)

        supports_cache = (
            True
            if self._transformer_accepts_cache is None
            else self._transformer_accepts_cache
        )
        enable_cache = (
            supports_cache if use_cache is None else (use_cache and supports_cache)
        )

        context_seq = tokens.long()
        generated_tokens: list[torch.Tensor] = []
        context_len = int(context_seq.shape[1])
        # Accumulate generated tokens without concatenating every step when caching.
        pending_tokens: list[torch.Tensor] = []

        window_size_limit = (
            context_len if max_context_tokens in (-1, None) else max_context_tokens
        )
        window_size = max(1, int(window_size_limit))

        # Interpret overlap: int = stride tokens; float = stride as fraction of window
        if isinstance(sliding_window_overlap, int):
            stride_tokens_direct = max(1, sliding_window_overlap)
            overlap_ratio = None
        else:
            overlap_ratio = float(sliding_window_overlap)
            if overlap_ratio < 0.0:
                raise ValueError("sliding_window_overlap must be non-negative.")
            stride_tokens_direct = None

        total_generated = 0
        pbar = tqdm(total=total_new_tokens, desc="Forecast (windowed)")
        while total_generated < total_new_tokens:
            if context_len > window_size:
                if pending_tokens:
                    context_seq = torch.cat([context_seq] + pending_tokens, dim=1)
                    pending_tokens = []
                context_seq = context_seq[:, -window_size:]
                context_len = int(context_seq.shape[1])

            if stride_tokens_direct is not None:
                stride = min(window_size, stride_tokens_direct)
            else:
                stride = int(round(window_size * overlap_ratio))
                stride = max(1, min(stride, window_size))

            if pending_tokens:
                context_seq = torch.cat([context_seq] + pending_tokens, dim=1)
                pending_tokens = []
                context_len = int(context_seq.shape[1])

            # Prefill on the current window to seed cache and next-block logits.
            hidden, cache = self._call_transformer(
                context_seq, cache_enabled=enable_cache
            )
            if enable_cache and (
                self._transformer_accepts_cache is False or cache is None
            ):
                enable_cache = False
                cache = None
                pending_tokens = []

            pred_hidden = hidden[:, -self.block_size :, :]
            chid_block = _make_chid_block(
                total_generated, pred_hidden.shape[1], toks_per_emb
            )
            next_logits = self._call_head(pred_hidden, chid=chid_block)
            if next_logits.dim() == 2:
                next_logits = next_logits.unsqueeze(1)

            for _ in tqdm(
                range(total_new_tokens - total_generated),
                desc="Window fill",
                leave=False,
            ):
                if total_generated >= total_new_tokens:
                    break
                if context_len >= window_size:
                    break

                tokens_left = total_new_tokens - total_generated
                room_in_window = window_size - context_len
                available_preds = int(next_logits.shape[1])
                tokens_this_step = min(
                    self.block_size, tokens_left, room_in_window, available_preds
                )
                if tokens_this_step <= 0:
                    break

                logits_block = next_logits[:, :tokens_this_step, :]
                next_block = sample_fn(logits_block).to(device)

                generated_tokens.append(next_block)
                if enable_cache:
                    pending_tokens.append(next_block)
                    context_len += int(next_block.shape[1])
                else:
                    context_seq = torch.cat([context_seq, next_block], dim=1)
                    context_len = int(context_seq.shape[1])
                total_generated += int(next_block.shape[1])
                pbar.update(int(next_block.shape[1]))

                if total_generated >= total_new_tokens or context_len >= window_size:
                    break

                chid_block = _make_chid_block(
                    total_generated - next_block.shape[1],
                    next_block.shape[1],
                    toks_per_emb,
                )
                if enable_cache:
                    hidden, cache = self._call_transformer(
                        next_block,
                        cache_in=cache,
                        cache_enabled=enable_cache,
                        chid=chid_block,
                    )
                else:
                    hidden, cache = self._call_transformer(
                        context_seq, cache_enabled=enable_cache
                    )

                pred_hidden = hidden[:, -self.block_size :, :]
                head_chid = _make_chid_block(
                    total_generated, pred_hidden.shape[1], toks_per_emb
                )
                next_logits = self._call_head(pred_hidden, chid=head_chid)

            if total_generated >= total_new_tokens:
                break

            if pending_tokens:
                context_seq = torch.cat([context_seq] + pending_tokens, dim=1)
                pending_tokens = []
                context_len = int(context_seq.shape[1])

            if context_seq.shape[1] > stride:
                context_seq = context_seq[:, stride:]
                context_len = int(context_seq.shape[1])
            else:
                context_seq = context_seq[:, -1:]
                context_len = int(context_seq.shape[1])

        pbar.close()
        full_seq = torch.cat(generated_tokens, dim=1)
        full_seq = self._forecast_strip_tokens(full_seq)

        if was_training:
            self.train()

        return full_seq


class FlatGPTMix(FlatGPT):
    def __init__(self, *args, mix_method: str = "mix", **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_embedding = MixEmbedding(
            self.vocab_size,
            self.hidden_size,
            self.reduced_shape[1],
            mix_method=mix_method,
        )

        self.head = MixHead(
            self.hidden_size,
            self.vocab_size,
            self.reduced_shape[1],
            mix_method=mix_method,
            emb=self.pre_embedding.emb.quant_emb,
        )
        self._pre_embedding_accepts_chid = None
        self._head_accepts_chid = None

        self.max_context_tokens = self.reduced_shape[0]

    def _refresh_max_context_tokens(self) -> None:
        self.max_context_tokens = int(self.reduced_shape[0])

    def _expected_transformer_reduced_shape_for_rope(self) -> Tuple[int, int, int]:
        return (int(self.reduced_shape[0]), 1, 1)

    def _forecast_expected_codes_rank(self) -> int:
        return 2

    def _forecast_extra_validation(self, codes: torch.Tensor) -> None:
        if codes.shape[0] % int(self.reduced_shape[1]) != 0:
            raise ValueError(
                "FlatGPTMix forecast expects batch divisible by num_channels "
                f"({self.reduced_shape[1]}), got batch {codes.shape[0]}."
            )


class FlatGPTRVQ(FlatGPT):
    """FlatGPT variant that operates with per-quantizer vocabularies and concatenated
    quantizer embeddings."""

    def __init__(
        self,
        *args,
        quantizer_head: str = "joint",
        quantizer_embed_dim: int | None = None,
        quantizer_levels: int | None = None,
        **kwargs,
    ):
        args = list(args)
        trf_args = kwargs.pop("trf_args", None)
        trf_args_is_positional = False
        if trf_args is None and len(args) > 1:
            trf_args = args[1]
            trf_args_is_positional = True
        if trf_args is None:
            raise ValueError("trf_args must be provided for FlatGPTRVQ.")
        trf_args = dict(trf_args)

        reduced_shape, _ = _compute_reduced_shape_from_init(tuple(args), kwargs)
        trf_args["reduced_shape"] = (reduced_shape[0], reduced_shape[1], 1)

        if trf_args_is_positional:
            args[1] = trf_args
        else:
            kwargs["trf_args"] = trf_args

        super().__init__(*args, **kwargs)

        # Determine quantizer meta
        if quantizer_levels is None:
            quantizer_levels = getattr(
                getattr(getattr(self.tokenizer, "quantizer", None), "rvq", None),
                "num_quantizers",
                None,
            )
        if quantizer_levels is None:
            raise ValueError("quantizer_levels must be provided for FlatGPTRVQ.")
        self.levels = int(quantizer_levels)

        level_vocab = getattr(self.tokenizer, "codebook_size", None)
        if level_vocab is None:
            rvq = getattr(getattr(self.tokenizer, "quantizer", None), "rvq", None)
            level_vocab = getattr(rvq, "codebook_size", self.vocab_size)

        if quantizer_embed_dim is None:
            if self.hidden_size % self.levels != 0:
                raise ValueError(
                    "hidden_size must be divisible by the number of quantizers or "
                    "set quantizer_embed_dim."
                )
            level_dim = self.hidden_size // self.levels
        else:
            level_dim = int(quantizer_embed_dim)

        total_hidden = level_dim * self.levels

        # Embeddings and heads per quantizer
        self.pre_embedding = QuantizerEmbedding(
            [int(level_vocab) for _ in range(self.levels)], level_dim
        )
        head_type = quantizer_head.lower()
        if head_type == "joint":
            self.head = JointRVQHead(total_hidden, self.levels, int(level_vocab))
        elif head_type == "tied":
            self.head = TiedRVQHead(self.pre_embedding.embeddings)
        else:
            raise ValueError("quantizer_head must be 'joint' or 'tied'.")
        self._pre_embedding_accepts_chid = None
        self._head_accepts_chid = None

        self.max_context_tokens = int(np.prod(self.reduced_shape[:2]))

    def _refresh_max_context_tokens(self) -> None:
        self.max_context_tokens = int(np.prod(self.reduced_shape[:2]))

    def _expected_transformer_reduced_shape_for_rope(self) -> Tuple[int, int, int]:
        return (int(self.reduced_shape[0]), int(self.reduced_shape[1]), 1)

    def _forecast_expected_codes_rank(self) -> int:
        return 3

    def _forecast_extra_validation(self, codes: torch.Tensor) -> None:
        if codes.shape[-1] != self.levels:
            raise ValueError(
                "FlatGPTRVQ forecast expects last dimension equal to "
                f"num_levels ({self.levels}), got {codes.shape[-1]}."
            )

    def _forecast_tokens_per_step(
        self, encoded: torch.Tensor, raw_input: torch.Tensor
    ) -> int:
        return self.reduced_shape[1] / self.temporal_reduction

    def _tokens_per_embedding(self, *args, **kwargs) -> int:
        return int(self.reduced_shape[1])

    def _encode_tokens(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = super()._encode_tokens(x)

        B = inputs["codes"].shape[0]

        L = int(np.prod(self.reduced_shape[:2]))
        inputs["codes"] = inputs["codes"].reshape(B, L, -1)
        return inputs

    def _call_transformer(
        self, token_batch: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, tuple | None]:
        codes = token_batch.long()
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)

        return super()._call_transformer(codes, **kwargs)

    def _call_head(self, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
        squeeze = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
            squeeze = True
        logits = self.head(hidden)
        if squeeze:
            logits = logits[:, -1, ...]
        return logits

    def _forecast_strip_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.dim() == 3:
            seq = seq.reshape(seq.shape[0], -1)
        return super()._forecast_strip_tokens(seq)


class FlatGPTMixRVQ(FlatGPT):
    """RVQ variant that concatenates channel + quantizer embeddings per timestep."""

    def __init__(
        self,
        *args,
        quantizer_head: str = "joint",
        quantizer_embed_dim: int | None = None,
        quantizer_levels: int | None = None,
        mix_method: str = "mix",
        **kwargs,
    ):
        args = list(args)
        trf_args = kwargs.pop("trf_args", None)
        trf_args_is_positional = False
        if trf_args is None and len(args) > 1:
            trf_args = args[1]
            trf_args_is_positional = True
        if trf_args is None:
            raise ValueError("trf_args must be provided for FlatGPTMixRVQ.")
        trf_args = dict(trf_args)

        reduced_shape, _ = _compute_reduced_shape_from_init(tuple(args), kwargs)
        trf_args["reduced_shape"] = (reduced_shape[0], 1, 1)

        if trf_args_is_positional:
            args[1] = trf_args
        else:
            kwargs["trf_args"] = trf_args

        super().__init__(*args, **kwargs)

        if quantizer_levels is None:
            quantizer_levels = getattr(
                getattr(getattr(self.tokenizer, "quantizer", None), "rvq", None),
                "num_quantizers",
                None,
            )

        level_vocab = getattr(self.tokenizer, "codebook_size", None)
        if level_vocab is None:
            rvq = getattr(getattr(self.tokenizer, "quantizer", None), "rvq", None)
            level_vocab = getattr(rvq, "codebook_size", self.vocab_size)

        channel_factor = self.reduced_shape[1] if mix_method == "mix" else 1
        if quantizer_embed_dim is None:
            if self.hidden_size % (quantizer_levels * channel_factor) != 0:
                raise ValueError(
                    "hidden_size must be divisible by num_channels*num_quantizers "
                    "when mix_method='mix', or by num_quantizers otherwise."
                )
            level_dim = self.hidden_size // (quantizer_levels * channel_factor)
        else:
            level_dim = int(quantizer_embed_dim)

        total_hidden = level_dim * quantizer_levels

        self.pre_embedding = MixQuantizerEmbedding(
            [int(level_vocab) for _ in range(quantizer_levels)],
            level_dim,
            self.reduced_shape[1],
            mix_method=mix_method,
        )
        head_type = quantizer_head.lower()
        if head_type == "joint":
            base_head = JointRVQHead(total_hidden, quantizer_levels, int(level_vocab))
        elif head_type == "tied":
            base_head = TiedRVQHead(self.pre_embedding.embeddings)
        else:
            raise ValueError("quantizer_head must be 'joint' or 'tied'.")

        self.head = MixRVQHead(
            base_head,
            self.reduced_shape[1],
            quantizer_levels,
            level_dim,
            mix_method=mix_method,
        )
        self._pre_embedding_accepts_chid = None
        self._head_accepts_chid = None
        self.max_context_tokens = int(self.reduced_shape[0])
        self.levels = int(quantizer_levels)

    def _refresh_max_context_tokens(self) -> None:
        self.max_context_tokens = int(self.reduced_shape[0])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, targets = super().forward(x)  # (b * c, t * q, k)

        c, q = self.reduced_shape[1], self.levels

        # reshape to canonical (B, T* C* Q, K) shape
        logits = rearrange(
            logits,
            "(b c) (t q) k -> b (t c q) k",
            c=c,
            q=q,
        )
        targets = rearrange(targets, "(b c) (t q) -> b (t c q)", c=c, q=q)
        return logits, targets

    def _encode_tokens(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = super()._encode_tokens(x)

        codes = inputs["codes"]
        if codes.dim() == 4:
            codes = rearrange(codes, "b t c q -> (b c) t q")
        else:
            batch = codes.shape[0]
            flat_len = int(np.prod(self.reduced_shape[:2]))
            codes = codes.reshape(batch, flat_len, -1)
            codes = codes.reshape(
                batch, self.reduced_shape[0], self.reduced_shape[1], -1
            )
            codes = rearrange(codes, "b t c q -> (b c) t q")

        inputs["codes"] = codes
        return inputs

    def _forecast_tokens_per_step(
        self, encoded: torch.Tensor, raw_input: torch.Tensor
    ) -> int:
        return 1 / self.temporal_reduction

    def _tokens_per_embedding(self, *args, **kwargs) -> int:
        return 1

    def _expected_transformer_reduced_shape_for_rope(self) -> Tuple[int, int, int]:
        return (int(self.reduced_shape[0]), 1, 1)

    def _forecast_expected_codes_rank(self) -> int:
        return 3

    def _forecast_extra_validation(self, codes: torch.Tensor) -> None:
        if codes.shape[0] % int(self.reduced_shape[1]) != 0:
            raise ValueError(
                "FlatGPTMixRVQ forecast expects batch divisible by num_channels "
                f"({self.reduced_shape[1]}), got batch {codes.shape[0]}."
            )
        if codes.shape[-1] != self.levels:
            raise ValueError(
                "FlatGPTMixRVQ forecast expects last dimension equal to "
                f"num_levels ({self.levels}), got {codes.shape[-1]}."
            )

    def _call_transformer(
        self, token_batch: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, tuple | None]:
        codes = token_batch.long()
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)

        return super()._call_transformer(codes, **kwargs)

    def _call_head(self, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
        squeeze = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
            squeeze = True
        logits = self.head(hidden)
        if squeeze:
            logits = logits[:, -1, ...]
        return logits

    def _forecast_strip_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.dim() == 3:
            seq = rearrange(seq, "(b c) t q -> b (t c) q", c=self.reduced_shape[1])
        return super()._forecast_strip_tokens(seq)


class FlatGPTEmbeds(FlatGPT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pre_embedding = ListEmbedding(
            self.vocab_size, self.hidden_size, self.reduced_shape[1]
        )

        # Replace the shared head with a channel-aware tied head.
        self.head = ChannelHead(self.pre_embedding.emb)
        self._pre_embedding_accepts_chid = None
        self._head_accepts_chid = None

    def _forecast_expected_codes_rank(self) -> int:
        return 2

    def _forecast_extra_validation(self, codes: torch.Tensor) -> None:
        if codes.shape[1] % int(self.reduced_shape[1]) != 0:
            raise ValueError(
                "FlatGPTEmbeds forecast expects sequence length divisible by "
                f"num_channels ({self.reduced_shape[1]}), got {codes.shape[1]}."
            )


class FlatGPTEmbedsRVQ(FlatGPTEmbeds):
    def __init__(self, *args, corrupt: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if corrupt:
            self.pre_embedding = EmbeddingCorruptor(
                self.vocab_size, self.hidden_size, self.reduced_shape
            )
        else:
            self.pre_embedding = ListEmbedding(
                self.vocab_size, self.hidden_size, self.reduced_shape[2]
            )

        # Replace the shared head with a channel-aware tied head.
        self.head = ChannelHead(self.pre_embedding.emb)
        self._pre_embedding_accepts_chid = None
        self._head_accepts_chid = None

    def _tokens_per_embedding(self, *args, **kwargs) -> int:
        return self.reduced_shape[2]

    def _forecast_expected_codes_rank(self) -> int:
        return 2

    def _forecast_extra_validation(self, codes: torch.Tensor) -> None:
        if codes.shape[1] % int(self.reduced_shape[2]) != 0:
            raise ValueError(
                "FlatGPTEmbedsRVQ forecast expects sequence length divisible by "
                f"num_levels ({self.reduced_shape[2]}), got {codes.shape[1]}."
            )
