import json

import numpy as np
import torch

from brain_gen.eval.eval_runner import TokenSummaryPlotter


class DummyTokenizer(torch.nn.Module):
    def __init__(self, timesteps: int, channels: int, quantizers: int):
        super().__init__()
        self.timesteps = timesteps
        self.channels = channels
        self.quantizers = quantizers

    def forecast_strip_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        reshaped = seq.view(
            seq.shape[0], self.timesteps, self.channels, self.quantizers
        ).float()
        return reshaped.mean(dim=-1).permute(0, 2, 1)


class TimeFirstTokenizer(torch.nn.Module):
    def __init__(self, timesteps: int, channels: int, quantizers: int):
        super().__init__()
        self.timesteps = timesteps
        self.channels = channels
        self.quantizers = quantizers

    def forecast_strip_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        reshaped = seq.view(
            seq.shape[0], self.timesteps, self.channels, self.quantizers
        ).float()
        return reshaped.mean(dim=-1)


class DummyLoss(torch.nn.Module):
    def forward(self, outputs, targets, reduction: str = "mean", **kwargs):
        loss = (outputs[0].float() - targets.float()).abs()
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        return loss.mean()


class FlatGPTStub(torch.nn.Module):
    def __init__(
        self,
        tokenizer: DummyTokenizer,
        vocab_size: int,
        *,
        input_shape: tuple[int, int, int] | None = None,
        temporal_reduction: float | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.reduced_shape = (
            tokenizer.timesteps,
            tokenizer.channels,
            tokenizer.quantizers,
        )
        self.vocab_size = vocab_size
        if input_shape is not None:
            self.input_shape = input_shape
        if temporal_reduction is not None:
            self.temporal_reduction = temporal_reduction


def test_token_summary_plotter_outputs(tmp_path):
    torch.manual_seed(0)
    timesteps = 6
    channels = 2
    quantizers = 3
    vocab = 16
    tokenizer = DummyTokenizer(timesteps, channels, quantizers)
    model = FlatGPTStub(tokenizer, vocab_size=vocab, temporal_reduction=1.0)
    plotter = TokenSummaryPlotter(
        {"tokens_per_second": 10.0, "enabled": True},
        model,
        loss_fn=DummyLoss(),
        sfreq=10.0,
    )

    batch = 1
    seq_len = timesteps * channels * quantizers
    target_tokens = torch.randint(0, vocab, (batch, seq_len))
    pred_tokens = torch.randint(0, vocab, (batch, seq_len))
    outputs = (pred_tokens, target_tokens)
    plotter.update(outputs)
    plotter.finalize(tmp_path)

    assert (tmp_path / "token_summary.png").exists()
    payload = json.loads((tmp_path / "token_summary.json").read_text())
    assert payload["context_unit"] == "seconds"
    assert "bits_per_token" in payload["metrics"]
    assert "perplexity" in payload["metrics"]
    assert "unigram_perplexity" in payload["metrics"]
    assert "decoded_mse" in payload["metrics"]
    assert len(payload["metrics"]["bits_per_token"]["mean"]) == timesteps
    assert len(payload["metrics"]["decoded_mse"]["mean"]) == timesteps


def test_token_summary_decoded_mse_time_first(tmp_path):
    torch.manual_seed(0)
    timesteps = 5
    channels = 3
    quantizers = 1
    vocab = 8
    tokenizer = TimeFirstTokenizer(timesteps, channels, quantizers)
    model = FlatGPTStub(tokenizer, vocab_size=vocab, temporal_reduction=1.0)
    plotter = TokenSummaryPlotter(
        {"tokens_per_second": 10.0, "enabled": True},
        model,
        loss_fn=DummyLoss(),
        sfreq=10.0,
    )

    batch = 1
    seq_len = timesteps * channels * quantizers
    target_tokens = torch.randint(0, vocab, (batch, seq_len))
    pred_tokens = torch.randint(0, vocab, (batch, seq_len))
    outputs = (pred_tokens, target_tokens)
    plotter.update(outputs)
    plotter.finalize(tmp_path)

    payload = json.loads((tmp_path / "token_summary.json").read_text())
    assert len(payload["metrics"]["decoded_mse"]["mean"]) == channels


def test_token_summary_time_axis_uses_temporal_reduction():
    timesteps = 6
    channels = 2
    quantizers = 3
    tokenizer = DummyTokenizer(timesteps, channels, quantizers)
    model = FlatGPTStub(
        tokenizer,
        vocab_size=16,
        input_shape=(24, channels, quantizers),
    )
    plotter = TokenSummaryPlotter(
        {"tokens_per_second": 10.0, "enabled": True},
        model,
        loss_fn=DummyLoss(),
        sfreq=10.0,
    )

    x, label = plotter._time_axis(3, "tokens")
    expected = torch.tensor([0.4, 0.8, 1.2]).numpy()
    assert label == "Time (s)"
    assert x.shape == expected.shape
    assert torch.allclose(torch.from_numpy(x), torch.from_numpy(expected))


def test_token_summary_smoothing_window_steps():
    timesteps = 10
    tokenizer = DummyTokenizer(timesteps, channels=1, quantizers=2)
    model = FlatGPTStub(
        tokenizer,
        vocab_size=8,
        input_shape=(20, 1, 2),
    )
    plotter = TokenSummaryPlotter(
        {"enabled": True, "tokens_per_second": 10.0},
        model,
        loss_fn=DummyLoss(),
        sfreq=10.0,
    )

    x, _ = plotter._time_axis(timesteps, "tokens")
    window = plotter._smoothing_window_steps(x)
    assert window == 5
    smoothed = plotter._window_mean(np.arange(10, dtype=np.float32), window)
    assert np.allclose(smoothed, np.array([2.0, 7.0], dtype=np.float32))
