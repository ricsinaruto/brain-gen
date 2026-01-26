import pytest
import torch
import torch.nn as nn

from brain_gen.models import FlatGPT, FlatGPTMix, FlatGPTMixRVQ
from brain_gen.models import flatgpt as flatgpt_module
from tests.models.utils import assert_future_grad_zero


class _DummyTransformer(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, *args, **kwargs):
        super().__init__()
        self.block_size = kwargs.get("block_size", 1)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.last_seen = None

    def forward(
        self,
        x: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        if (x is None) == (inputs_embeds is None):
            raise ValueError(
                "DummyTransformer expects exactly one of token ids or embeddings."
            )
        if inputs_embeds is not None:
            self.last_seen = inputs_embeds
            return inputs_embeds
        self.last_seen = x
        return self.emb(x)

    def get_embed_layer(self) -> nn.Module:
        return self.emb


flatgpt_module._DummyTransformer = _DummyTransformer


class _DummyReducedShapeTransformer(_DummyTransformer):
    def __init__(self, hidden_size: int, vocab_size: int, *args, **kwargs):
        self.reduced_shape = kwargs.get("reduced_shape")
        super().__init__(hidden_size, vocab_size, *args, **kwargs)


flatgpt_module._DummyReducedShapeTransformer = _DummyReducedShapeTransformer


class _DummyVLM(_DummyTransformer):
    def __init__(self, hidden_size: int, vocab_size: int, *args, **kwargs):
        self.reduced_shape = kwargs.get("reduced_shape")
        super().__init__(hidden_size, vocab_size, *args, **kwargs)

    def _build_position_ids(self, *args, **kwargs):
        t, h, w = self.reduced_shape
        total = max(1, int(t * h * w))
        return torch.zeros((3, 1, total), dtype=torch.long)


flatgpt_module._DummyVLM = _DummyVLM


class _CacheDummyTransformer(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, *args, **kwargs):
        super().__init__()
        self.block_size = kwargs.get("block_size", 1)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.calls = []

    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: int | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        if (x is None) == (inputs_embeds is None):
            raise ValueError(
                "CacheDummyTransformer expects exactly one of token ids or embeddings."
            )
        if inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1]
            hidden = inputs_embeds
        else:
            seq_len = x.shape[1]
            hidden = self.emb(x)

        use_cache = bool(use_cache)
        cache_len = 0 if past_key_values is None else int(past_key_values)
        cache_out = cache_len + seq_len if use_cache else None

        self.calls.append(
            {
                "seq_len": seq_len,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )
        return hidden, cache_out

    def get_embed_layer(self) -> nn.Module:
        return self.emb


flatgpt_module._CacheDummyTransformer = _CacheDummyTransformer


class _DummyTokenizer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encode_called = False

    def encode(self, x):
        self.encode_called = True
        raise AssertionError("Tokenizer encode should not be called.")


flatgpt_module._DummyTokenizer = _DummyTokenizer


def test_qwen3_positions_match_qwen2_5_for_video_only():
    """
    Qwen3_Video uses the same 3D (T, H, W) position IDs as Qwen2_5_Video.

    Qwen3VLTextModel handles this directly: it uses position_ids[0] (T) for
    causal masking and the full (3, batch, seq) for 3D rotary embedding.
    """
    model = flatgpt_module.Qwen3_Video(
        hidden_size=16,
        vocab_size=32,
        reduced_shape=(2, 2, 1),
        max_position_embeddings=64,
        rope_scaling={"rope_type": "default"},
    )

    pos = model._build_position_ids(batch_size=1, device=torch.device("cpu"), seq_len=4)
    # Shape is (3, batch, seq) for T, H, W
    assert pos.shape == (3, 1, 4)
    assert torch.equal(pos[0, 0], torch.tensor([0, 0, 1, 1]))  # T
    assert torch.equal(pos[1, 0], torch.tensor([0, 1, 0, 1]))  # H
    assert torch.equal(pos[2, 0], torch.zeros(4, dtype=torch.long))  # W

    # With offset (for caching), positions shift within the flattened grid
    offset = model._build_position_ids(
        batch_size=1, device=torch.device("cpu"), seq_len=2, position_offset=1
    )
    assert torch.equal(offset[0, 0], torch.tensor([0, 1]))  # T
    assert torch.equal(offset[1, 0], torch.tensor([1, 0]))  # H
    assert torch.equal(offset[2, 0], torch.zeros(2, dtype=torch.long))  # W


def test_flatgpt_resize_context_updates_qwen2_5_rope():
    hidden_size = 16
    vocab_size = 32

    trf_args = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "rope_scaling": {"rope_type": "default", "mrope_section": 1},
        "max_position_embeddings": 32,
        "rope_theta": 1.0e4,
        "use_cache": False,
        "attention_dropout": 0.0,
        "use_sliding_window": False,
    }

    model = FlatGPT(
        trf_class="Qwen2_5_Video",
        trf_args=trf_args,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(4, 1, 1),
        tok_args={},
        train_tokenizer=True,
    )

    rotary = next(
        module
        for module in model.transformer.model.modules()
        if hasattr(module, "rope_init_fn") and hasattr(module, "inv_freq")
    )
    old_inv = rotary.inv_freq.clone()

    model.resize_context(
        input_shape=(8, 1, 1),
        rope_theta=2.0e5,
        max_position_embeddings=64,
    )

    assert model.reduced_shape == (8, 1, 1)
    assert model.transformer.reduced_shape == (8, 1, 1)
    assert model.transformer._total_positions == 8
    assert model.transformer.config.rope_theta == 2.0e5
    assert rotary.max_seq_len_cached == 64
    assert not torch.allclose(old_inv, rotary.inv_freq)


def test_grad_causality_flatgpt():
    B, T = 2, 6
    hidden_size = 24
    vocab_size = 32

    trf_args = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 96,
        "rope_scaling": {"rope_type": "default", "mrope_section": 1},
        "max_position_embeddings": 128,
        "use_cache": False,
        "attention_dropout": 0.0,
        "use_sliding_window": False,
    }

    model = FlatGPT(
        trf_class="Qwen2_5_Video",
        trf_args=trf_args,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(T, 1, 1),
        tok_args={},
        train_tokenizer=True,
    )
    model.eval()

    # Differentiable embeddings fed directly to the decoder
    emb = torch.randn(B, T, hidden_size, requires_grad=True)
    position_ids = model.transformer._build_position_ids(
        batch_size=B, device=emb.device, seq_len=T - 1
    )
    outputs = model.transformer.model(
        inputs_embeds=emb[:, :-1],
        position_ids=position_ids,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = model.head(outputs.last_hidden_state)
    loss = logits.sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


@pytest.mark.parametrize("mix_method", ["mix", "none"])
def test_grad_causality_flatgptmix(mix_method: str):
    B, T, C = 2, 6, 4
    hidden_size = 24
    vocab_size = 32

    trf_args = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 4 * hidden_size,
        "max_position_embeddings": 128,
        "use_cache": False,
        "attention_dropout": 0.0,
        "mix_method": mix_method,
        "pad_token_id": None,
    }

    model = FlatGPTMix(
        trf_class="SmoLLM3",
        trf_args=trf_args,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(T, C, 1),
    )
    model.eval()

    # Differentiable embeddings fed directly to the decoder
    if mix_method == "mix":
        emb = torch.randn(B, T, hidden_size, requires_grad=True)
    else:
        emb = torch.randn(B * C, T, hidden_size, requires_grad=True)

    outputs = model.transformer.model(
        inputs_embeds=emb[:, :-1],
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = model.head(outputs.last_hidden_state)
    loss = logits.sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def test_grad_causality_flatgptmixrvq():
    B, T, C, Q = 2, 6, 4, 3
    hidden_size = C * Q * 2
    vocab_size = 32

    trf_args = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 4 * hidden_size,
        "max_position_embeddings": 128,
        "use_cache": False,
        "attention_dropout": 0.0,
        "mix_method": "mix",
        "pad_token_id": None,
    }

    model = FlatGPTMixRVQ(
        trf_class="SmoLLM3",
        trf_args=trf_args,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(T, C, Q),
        quantizer_levels=Q,
        mix_method="mix",
    )
    model.eval()

    emb = torch.randn(B, T, hidden_size, requires_grad=True)

    outputs = model.transformer.model(
        inputs_embeds=emb[:, :-1],
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = model.head(outputs.last_hidden_state)
    loss = logits.sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def test_flatgptmixrvq_channel_embedding_mix():
    batch, timesteps, channels, levels = 2, 3, 4, 2
    hidden_size = channels * levels * 2
    vocab_size = 8

    model = FlatGPTMixRVQ(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(timesteps, channels, levels),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
        quantizer_levels=levels,
        mix_method="mix",
    )
    model.eval()

    codes = torch.zeros(batch, timesteps, channels, levels, dtype=torch.long).reshape(
        batch, -1
    )

    with torch.no_grad():
        for emb in model.pre_embedding.quantizer.embeddings:
            emb.weight.zero_()
        channel_vals = torch.arange(
            channels * levels * model.pre_embedding.embed_dim, dtype=torch.float32
        ).reshape(channels, -1)
        model.pre_embedding.channel_emb.weight.copy_(channel_vals)

    logits, targets = model({"codes": codes})

    embeds = model.transformer.last_seen
    assert embeds.shape == (batch, timesteps, hidden_size)
    assert logits.shape == (
        batch,
        (timesteps - 1) * channels * levels,
        vocab_size,
    )
    assert targets.shape == (batch, (timesteps - 1) * channels * levels)

    expected = model.pre_embedding.channel_emb.weight
    got = embeds[0, 0].view(channels, -1)
    assert torch.allclose(got, expected, atol=0, rtol=0)


def test_vlm_rope_flatgpt_variants():
    timesteps, channels, levels = 4, 3, 2

    flatgpt_module.FlatGPT(
        trf_class="_DummyVLM",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=16,
        input_shape=(timesteps, channels, 1),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )

    flatgpt_module.FlatGPTRVQ(
        trf_class="_DummyVLM",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=16,
        input_shape=(timesteps, channels, levels),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
        quantizer_levels=levels,
    )

    flatgpt_module.FlatGPTMixRVQ(
        trf_class="_DummyVLM",
        trf_args={"block_size": 1},
        hidden_size=channels * levels * 2,
        vocab_size=16,
        input_shape=(timesteps, channels, levels),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
        quantizer_levels=levels,
        mix_method="mix",
    )

    flatgpt_module.FlatGPTEmbeds(
        trf_class="_DummyVLM",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=16,
        input_shape=(timesteps, channels, 1),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )

    flatgpt_module.FlatGPTEmbedsRVQ(
        trf_class="_DummyVLM",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=16,
        input_shape=(timesteps, channels, levels),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )


def test_vlm_rope_flatgptmix_requires_override():
    timesteps, channels = 4, 3
    with pytest.raises(ValueError):
        flatgpt_module.FlatGPTMix(
            trf_class="_DummyVLM",
            trf_args={"block_size": 1},
            hidden_size=8,
            vocab_size=16,
            input_shape=(timesteps, channels, 1),
            tok_class="_DummyTokenizer",
            tok_args={},
            train_tokenizer=False,
        )

    flatgpt_module.FlatGPTMix(
        trf_class="_DummyVLM",
        trf_args={"block_size": 1, "reduced_shape": (timesteps, 1, 1)},
        hidden_size=8,
        vocab_size=16,
        input_shape=(timesteps, channels, 1),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )


def test_forecast_setup_validation():
    model = FlatGPT(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=16,
        input_shape=(5, 2, 1),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )
    model._validate_forecast_setup(
        torch.zeros(1, 5, dtype=torch.long), tokens_per_step=1, tokens_per_embedding=1
    )
    with pytest.raises(ValueError):
        model._validate_forecast_setup(
            torch.zeros(1, 5, 2, dtype=torch.long),
            tokens_per_step=1,
            tokens_per_embedding=1,
        )

    mix = flatgpt_module.FlatGPTMix(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=16,
        input_shape=(5, 2, 1),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )
    mix._validate_forecast_setup(
        torch.zeros(2, 5, dtype=torch.long), tokens_per_step=1, tokens_per_embedding=1
    )
    with pytest.raises(ValueError):
        mix._validate_forecast_setup(
            torch.zeros(3, 5, dtype=torch.long),
            tokens_per_step=1,
            tokens_per_embedding=1,
        )

    rvq = flatgpt_module.FlatGPTRVQ(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=12,
        vocab_size=16,
        input_shape=(5, 2, 3),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
        quantizer_levels=3,
    )
    rvq._validate_forecast_setup(
        torch.zeros(1, 5, 3, dtype=torch.long),
        tokens_per_step=1,
        tokens_per_embedding=1,
    )
    with pytest.raises(ValueError):
        rvq._validate_forecast_setup(
            torch.zeros(1, 5, 2, dtype=torch.long),
            tokens_per_step=1,
            tokens_per_embedding=1,
        )

    mix_rvq = flatgpt_module.FlatGPTMixRVQ(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=12,
        vocab_size=16,
        input_shape=(5, 2, 3),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
        quantizer_levels=3,
        mix_method="mix",
    )
    mix_rvq._validate_forecast_setup(
        torch.zeros(2, 5, 3, dtype=torch.long),
        tokens_per_step=1,
        tokens_per_embedding=1,
    )
    with pytest.raises(ValueError):
        mix_rvq._validate_forecast_setup(
            torch.zeros(3, 5, 3, dtype=torch.long),
            tokens_per_step=1,
            tokens_per_embedding=1,
        )

    embeds = flatgpt_module.FlatGPTEmbeds(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=16,
        input_shape=(5, 2, 1),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )
    embeds._validate_forecast_setup(
        torch.zeros(1, 6, dtype=torch.long), tokens_per_step=1, tokens_per_embedding=1
    )
    with pytest.raises(ValueError):
        embeds._validate_forecast_setup(
            torch.zeros(1, 5, dtype=torch.long),
            tokens_per_step=1,
            tokens_per_embedding=1,
        )

    embeds_rvq = flatgpt_module.FlatGPTEmbedsRVQ(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=16,
        input_shape=(5, 2, 3),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )
    embeds_rvq._validate_forecast_setup(
        torch.zeros(1, 6, dtype=torch.long), tokens_per_step=1, tokens_per_embedding=1
    )
    with pytest.raises(ValueError):
        embeds_rvq._validate_forecast_setup(
            torch.zeros(1, 5, dtype=torch.long),
            tokens_per_step=1,
            tokens_per_embedding=1,
        )

    model._validate_forecast_setup(
        torch.zeros(1, 5, dtype=torch.long),
        tokens_per_step=0.5,
        tokens_per_embedding=1,
    )
    with pytest.raises(ValueError):
        model._validate_forecast_setup(
            torch.zeros(1, 5, dtype=torch.long),
            tokens_per_step=0.0,
            tokens_per_embedding=1,
        )


def test_flatgpt_rvq_passes_reduced_shape_without_levels():
    timesteps, channels, levels = 5, 3, 4
    model = flatgpt_module.FlatGPTRVQ(
        trf_class="_DummyReducedShapeTransformer",
        trf_args={"block_size": 1},
        hidden_size=12,
        vocab_size=8,
        input_shape=(timesteps, channels, levels),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
        quantizer_levels=levels,
    )

    assert model.transformer.reduced_shape == (timesteps, channels, 1)


def test_flatgptmixrvq_passes_time_only_reduced_shape():
    timesteps, channels, levels = 6, 2, 3
    model = flatgpt_module.FlatGPTMixRVQ(
        trf_class="_DummyReducedShapeTransformer",
        trf_args={"block_size": 1},
        hidden_size=channels * levels * 2,
        vocab_size=8,
        input_shape=(timesteps, channels, levels),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
        quantizer_levels=levels,
        mix_method="mix",
    )

    assert model.transformer.reduced_shape == (timesteps, 1, 1)


def test_tokens_per_embedding_rvq_variants():
    timesteps, channels, levels = 8, 4, 3

    rvq = flatgpt_module.FlatGPTRVQ(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=12,
        vocab_size=8,
        input_shape=(timesteps, channels, levels),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
        quantizer_levels=levels,
        temporal_reduction=2,
    )
    assert rvq._tokens_per_embedding(None, None) == channels

    mix_rvq = flatgpt_module.FlatGPTMixRVQ(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=channels * levels * 2,
        vocab_size=8,
        input_shape=(timesteps, channels, levels),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
        quantizer_levels=levels,
        mix_method="mix",
        temporal_reduction=2,
    )
    assert mix_rvq._tokens_per_embedding(None, None) == 1


def test_grad_causality_flatgpt_block_causal():
    B, T = 2, 6
    block_size = 2
    hidden_size = 24
    vocab_size = 32

    trf_args = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 96,
        "rope_scaling": {"rope_type": "default", "mrope_section": 1},
        "max_position_embeddings": 128,
        "use_cache": False,
        "attention_dropout": 0.0,
        "use_sliding_window": False,
        "block_size": block_size,
    }

    model = FlatGPT(
        trf_class="Qwen2_5_Video",
        trf_args=trf_args,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(T, 1, 1),
        tok_args={},
        train_tokenizer=True,
    )
    model.eval()

    emb = torch.randn(B, T, hidden_size, requires_grad=True)
    hidden = model.transformer(inputs_embeds=emb, use_cache=False)
    logits = model.head(hidden)
    loss = logits[:, : T - block_size].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - block_size)


def test_token_corruption_schedule_exponential():
    seq_len = 5
    model = FlatGPT(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=16,
        input_shape=(seq_len, 1, 1),
        tok_args={},
        train_tokenizer=True,
        token_corruption_cfg={"enabled": True, "p_start": 0.01, "p_end": 0.05},
    )
    probs = model._timestep_corruption_schedule(seq_len, torch.device("cpu"))

    assert probs.shape == (seq_len,)
    assert float(probs[0]) == pytest.approx(0.01, rel=1e-2)
    assert float(probs[-1]) == pytest.approx(0.05, rel=1e-2)
    assert torch.all(probs[1:] >= probs[:-1])


def test_flatgpt_accepts_pretokenized_codes():
    vocab_size = 16
    seq_len = 5
    batch = 2

    model = FlatGPT(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=vocab_size,
        input_shape=(seq_len, 1, 1),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )

    codes = torch.randint(0, vocab_size, (batch, seq_len))
    logits, targets = model({"codes": codes})

    assert logits.shape[:2] == (batch, seq_len - model.block_size)
    assert torch.equal(targets, codes[:, model.block_size :])
    assert model.tokenizer.encode_called is False


def test_flatgpt_forecast_cache_then_stride():
    vocab_size = 8
    block_size = 2
    hidden_size = 4
    max_context_tokens = 10
    input_len = 16
    initial_len = 4
    rollout_steps = 8

    model = FlatGPT(
        trf_class="_CacheDummyTransformer",
        trf_args={"block_size": block_size},
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(input_len, 1, 1),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )
    model._encode_tokens = lambda x: {"codes": x}

    initial_tokens = torch.randint(0, vocab_size, (1, initial_len))
    sample_fn = lambda logits: torch.zeros(
        logits.shape[:2], dtype=torch.long, device=logits.device
    )

    _ = model.forecast(
        initial_tokens,
        rollout_steps,
        sample_fn,
        max_context_tokens=max_context_tokens,
        use_cache=True,
        sliding_window_overlap=0.5,
    )

    calls = model.transformer.calls
    cacheless_indices = [
        idx for idx, call in enumerate(calls) if call["past_key_values"] is None
    ]

    assert calls[0]["seq_len"] == initial_len
    assert calls[0]["use_cache"] is True
    assert any(call["past_key_values"] is not None for call in calls[1:])
    assert len(cacheless_indices) >= 2

    stride = int(round(max_context_tokens * 0.5))
    expected_len = max_context_tokens - stride
    assert calls[cacheless_indices[1]]["seq_len"] == expected_len


def test_flatgpt_forecast_default_max_context_uses_initial_length():
    vocab_size = 8
    block_size = 2
    hidden_size = 4
    input_len = 12
    initial_len = 4
    rollout_steps = 10

    model = FlatGPT(
        trf_class="_CacheDummyTransformer",
        trf_args={"block_size": block_size},
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(input_len, 1, 1),
        tok_class="_DummyTokenizer",
        tok_args={},
        train_tokenizer=False,
    )
    model._encode_tokens = lambda x: {"codes": x}

    initial_tokens = torch.randint(0, vocab_size, (1, initial_len))
    sample_fn = lambda logits: torch.zeros(
        logits.shape[:2], dtype=torch.long, device=logits.device
    )

    _ = model.forecast(
        initial_tokens,
        rollout_steps,
        sample_fn,
        use_cache=True,
        sliding_window_overlap=0.5,
    )

    calls = model.transformer.calls
    cacheless_indices = [
        idx for idx, call in enumerate(calls) if call["past_key_values"] is None
    ]

    stride = int(round(initial_len * 0.5))
    expected_len = initial_len - stride
    assert calls[cacheless_indices[1]]["seq_len"] == expected_len
