import torch
from einops import rearrange

from brain_gen.models.brainomni import (
    BrainOmniCausalForecast,
)

from brain_gen.models.tokenizers.brainomni import (
    BrainOmniCausalTokenizer,
    CausalTokenSequence,
)
from brain_gen.models.tokenizers.braintokmix import (
    BrainOmniCausalTokenizerSEANetChannelMix,
)

from tests.models.utils import assert_future_grad_zero


def _make_tokenizer():
    return BrainOmniCausalTokenizer(
        window_length=8,
        n_filters=2,
        ratios=[2, 2],
        kernel_size=3,
        last_kernel_size=3,
        n_dim=8,
        n_neuro=3,
        n_head=2,
        dropout=0.0,
        codebook_dim=8,
        codebook_size=16,
        num_quantizers=2,
        rotation_trick=False,
    )


def _make_seanet_channelmix_tokenizer():
    return BrainOmniCausalTokenizerSEANetChannelMix(
        window_length=8,
        n_filters=2,
        ratios=[2, 2],
        kernel_size=3,
        last_kernel_size=3,
        n_dim=8,
        n_neuro=2,
        n_head=2,
        dropout=0.0,
        codebook_dim=4,
        codebook_size=16,
        num_quantizers=2,
        rotation_trick=False,
        num_sensors=4,
    )


@torch.no_grad()
def test_causal_tokenizer_forward_and_decode():
    tokenizer = _make_tokenizer()
    B, C, T = 1, 3, 16
    x = torch.randn(B, C, T)
    pos = torch.randn(B, C, 6)
    sensor_type = torch.zeros(B, C, dtype=torch.long)

    tokens = tokenizer.tokenize((x, pos, sensor_type), overlap_ratio=0.0)
    assert tokens.embeddings.shape[0] == B
    assert tokens.indices.shape[-1] == tokenizer.quantizer.rvq.num_quantizers

    windows = tokenizer.decode_windows(
        tokens.indices, pos, sensor_type, tokens.tokens_per_window
    )
    stride = tokenizer._stride(tokens.overlap_ratio)
    recon = tokenizer.overlap_add(windows, stride=stride)
    expected_len = stride * (tokens.num_windows - 1) + tokenizer.window_length
    assert recon.shape == (B, C, expected_len)


@torch.no_grad()
def test_causal_tokenizer_forecast_strip_tokens_chunked():
    tokenizer = _make_tokenizer()
    tokenizer.eval()

    B, C, T = 1, 3, 32
    x = torch.randn(B, C, T)
    pos = torch.randn(B, C, 2)
    sensor_type = torch.zeros(B, C, dtype=torch.long)

    tokens = tokenizer.tokenize((x, pos, sensor_type), overlap_ratio=0.0)
    indices = tokens.indices[:, :, :-1]
    seq = rearrange(indices, "B C W Q -> B (W C) Q").reshape(B, -1)

    tokenizer.forecast_decode_window_chunk = None
    tokenizer.forecast_decode_max_tokens = 0
    decoded = tokenizer.forecast_strip_tokens(seq)

    tokenizer.forecast_decode_window_chunk = 1
    decoded_chunked = tokenizer.forecast_strip_tokens(seq)

    assert decoded.shape == decoded_chunked.shape
    torch.testing.assert_close(decoded_chunked, decoded, rtol=0.0, atol=1e-7)


@torch.no_grad()
def test_causal_tokenizer_short_window_no_padding():
    tokenizer = _make_tokenizer()
    B, C, T = 1, 3, 4
    x = torch.randn(B, C, T)
    pos = torch.randn(B, C, 2)
    sensor_type = torch.zeros(B, C, dtype=torch.long)

    tokens = tokenizer.tokenize(
        (x, pos, sensor_type), allow_shorter_windows=True, overlap_ratio=0.0
    )
    assert tokens.num_windows == 1
    assert tokens.tokens_per_window == 1

    windows = tokenizer.decode_windows(
        tokens.indices, pos, sensor_type, tokens.tokens_per_window
    )
    stride = tokenizer._stride(tokens.overlap_ratio)
    recon = tokenizer.overlap_add(windows, stride=stride)
    assert recon.shape == (B, C, T)


@torch.no_grad()
def test_seanet_channelmix_tokenizer_forward_and_decode():
    tokenizer = _make_seanet_channelmix_tokenizer()
    B, C, T = 1, 4, 16
    x = torch.randn(B, C, T)
    pos = torch.randn(B, C, 2)
    sensor_type = torch.zeros(B, C, dtype=torch.long)

    tokens = tokenizer.tokenize((x, pos, sensor_type), overlap_ratio=0.0)
    assert tokens.embeddings.shape[1] == tokenizer.n_neuro
    assert tokens.embeddings.shape[-1] == tokenizer.n_dim
    assert (
        tokens.embeddings.shape[1] * tokens.embeddings.shape[-1] == tokenizer.seanet_dim
    )

    windows = tokenizer.decode_windows(
        tokens.indices, pos, sensor_type, tokens.tokens_per_window
    )
    stride = tokenizer._stride(tokens.overlap_ratio)
    recon = tokenizer.overlap_add(windows, stride=stride)
    expected_len = stride * (tokens.num_windows - 1) + tokenizer.window_length
    assert recon.shape == (B, C, expected_len)


def test_causal_tokenizer_tokenize_decode_grad_causality():
    tokenizer = _make_tokenizer()
    tokenizer.train()

    B, C, T = 1, 32, 128
    x = torch.randn(B, C, T, requires_grad=True)
    pos = torch.randn(B, C, 2)
    sensor_type = torch.zeros(B, C, dtype=torch.long)

    tokens = tokenizer.tokenize((x, pos, sensor_type), overlap_ratio=0.0)
    token_emb = tokens.embeddings.reshape(
        B, tokens.embeddings.shape[1], tokens.num_windows, tokens.tokens_per_window, -1
    )
    windows = tokenizer.decode_windows(
        tokens.indices,
        pos,
        sensor_type,
        tokens.tokens_per_window,
        embeddings=token_emb,
    )
    recon = tokenizer.overlap_add(
        windows, stride=tokenizer._stride(tokens.overlap_ratio)
    )

    loss = recon[..., :-4].sum()
    loss.backward()

    assert x.grad is not None
    last_grad = x.grad[..., -1]
    assert torch.allclose(last_grad, torch.zeros_like(last_grad), atol=1e-5)


def test_causal_forecaster_grad_causality():
    tok_args = {
        "window_length": 8,
        "n_filters": 2,
        "ratios": [2],
        "kernel_size": 3,
        "last_kernel_size": 3,
        "n_dim": 8,
        "n_head": 2,
        "n_neuro": 2,
        "dropout": 0.0,
        "codebook_dim": 8,
        "codebook_size": 16,
        "num_quantizers": 2,
        "rotation_trick": False,
    }
    model = BrainOmniCausalForecast(
        tokenizer_kwargs=tok_args,
        overlap_ratio=0.25,
        lm_dim=8,
        lm_head=2,
        lm_depth=2,
        lm_dropout=0.0,
        num_quantizers_used=2,
        freeze_tokenizer=False,
    )

    B, C, W, D = 1, 2, 6, 8
    emb = torch.randn(B, C, W, D, requires_grad=True)
    idx = torch.randint(0, 16, (B, C, W, 2))
    seq = CausalTokenSequence(
        embeddings=emb,
        indices=idx,
        tokens_per_window=W,
        num_windows=1,
        overlap_ratio=0.25,
    )

    outputs = model.forward_token_sequence(seq)
    loss = outputs["logits"].sum()
    loss.backward()
    last_grad = emb.grad[:, :, -1, :]
    assert torch.allclose(last_grad, torch.zeros_like(last_grad), atol=1e-7)


def test_causal_forecaster_full_forward_grad_causality_with_tokenizer():
    tok_args = {
        "window_length": 64,  # paper uses 2 seconds, we use 0.64s
        "n_filters": 8,  # 32
        "ratios": [2, 2],  # 8,4,2
        "kernel_size": 5,  # 5
        "last_kernel_size": 5,  # 5
        "n_dim": 16,  # 256
        "n_neuro": 16,  # 16
        "n_head": 2,  # 4
        "codebook_size": 512,  # 512
        "codebook_dim": 16,  # 256
        "num_quantizers": 2,  # 4
        "rotation_trick": True,  # true
        "mask_ratio": 0.25,  # 0.25
        "dropout": 0.0,  # 0.0
        "noise_std": 0.1,  # 0.1
        "num_sensors": 68,
    }

    model = BrainOmniCausalForecast(
        tokenizer_kwargs=tok_args,
        overlap_ratio=0.0,
        lm_dim=16,
        lm_head=2,
        lm_depth=2,
        lm_dropout=0.0,
        num_quantizers_used=2,
        freeze_tokenizer=False,
    )

    B, C, T = 1, 68, 128
    x = torch.randn(B, C, T, requires_grad=True)
    pos = torch.randn(B, C, 2)
    sensor_type = torch.zeros(B, C, dtype=torch.long)

    token_seq = model.tokenizer.tokenize(
        (x, pos, sensor_type), overlap_ratio=model.overlap_ratio
    )

    outputs = model.forward_token_sequence(token_seq, compute_targets=True)
    logits = outputs["logits"]

    loss = logits[:, :, :].sum()
    loss.backward()

    assert_future_grad_zero(x, T - 4)
