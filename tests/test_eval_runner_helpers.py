import json
from pathlib import Path
import numpy as np
import pytest
import torch
from matplotlib.axes import Axes

import brain_gen.eval.generation as generation
import brain_gen.eval.session_sampler as session_sampler

from brain_gen.eval.generation import (
    GenerationResult,
    RolloutGenerator,
    _generation_input_key,
    _group_generation_samples,
    _stack_generation_inputs,
)
from brain_gen.eval.plotting import EvaluationPlotting
from brain_gen.eval.rollout_divergence import RolloutDivergenceAnalysis
from brain_gen.eval.rollout_sliding_windows import RolloutSlidingWindowAnalysis
from brain_gen.eval.session_sampler import SessionSample, SessionSampler
from brain_gen.eval.rollout_metrics import (
    MetricDistances,
    RolloutMetricsConfig,
    RolloutMetrics,
    resolve_frequency_bands,
)
from tests.models.utils import make_dummy_session


def _dummy_dataset(root, indices, *, sfreq=1.0, has_condition=False):
    class DummyDataset:
        def __init__(self, base):
            self.root_dirs = {"dataset0": str(base)}
            self.ch_names = ["c0", "c1"]
            self.fill_value = 0
            self.indices = indices
            self.sfreq = sfreq
            self.has_condition = has_condition

        def _get_session_indices(self, dataset_key, session_name, n_channels):
            return np.arange(n_channels)

    return DummyDataset(root)


def test_session_sampler_uses_first_chunk_only(tmp_path, monkeypatch):
    root = tmp_path / "omega"
    session = "rest_sub-001"
    data0 = np.arange(2 * 6, dtype=np.float32).reshape(2, 6)
    data1 = np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + 100
    make_dummy_session(str(root), session, data=data0, chunk_idx=0)
    make_dummy_session(str(root), session, data=data1, chunk_idx=1)

    indices = [("dataset0", session, "0.npy", 0)]
    dataset = _dummy_dataset(root, indices, sfreq=1.0)
    cfg = {
        "context_length_s": 2,
        "total_length_s": 6,
        "num_sessions": 1,
        "split": "val",
    }
    orig_load = session_sampler._load_chunk_cached

    def load_chunk(path):
        if Path(path).name == "1.npy":
            raise AssertionError("second chunk should not be loaded")
        return orig_load(path)

    monkeypatch.setattr(session_sampler, "_load_chunk_cached", load_chunk)
    sampler = SessionSampler(dataset, cfg)
    samples = sampler.sample_sessions()
    assert len(samples) == 1
    assert np.array_equal(samples[0].data, data0)


def test_session_sampler_skips_sessions_with_short_first_chunk(tmp_path):
    root = tmp_path / "omega"
    long_session = "rest_sub-002"
    short_session = "rest_sub-003"
    make_dummy_session(str(root), long_session, data=np.zeros((2, 8), dtype=np.float32))
    make_dummy_session(
        str(root), short_session, data=np.zeros((2, 4), dtype=np.float32)
    )
    make_dummy_session(
        str(root), short_session, data=np.zeros((2, 8), dtype=np.float32), chunk_idx=1
    )

    indices = [
        ("dataset0", long_session, "0.npy", 0),
        ("dataset0", short_session, "0.npy", 0),
    ]
    dataset = _dummy_dataset(root, indices, sfreq=1.0)
    cfg = {
        "context_length_s": 2,
        "total_length_s": 6,
        "num_sessions": 0,
        "task_type": "rest",
        "split": "val",
    }
    sampler = SessionSampler(dataset, cfg)
    samples = sampler.sample_sessions()
    sessions = {sample.session for sample in samples}
    assert sessions == {long_session}


def test_session_sampler_filters_by_task_and_length(tmp_path, capsys):
    root = tmp_path / "omega"
    long_session = "rest_sub-002"
    short_session = "auditory_sub-003"
    make_dummy_session(
        str(root), long_session, data=np.zeros((2, 12), dtype=np.float32)
    )
    make_dummy_session(
        str(root), short_session, data=np.zeros((2, 4), dtype=np.float32)
    )

    indices = [
        ("dataset0", long_session, "0.npy", 0),
        ("dataset0", short_session, "0.npy", 0),
    ]
    dataset = _dummy_dataset(root, indices, sfreq=1.0)
    cfg = {
        "context_length_s": 2,
        "total_length_s": 8,
        "num_sessions": 2,
        "task_type": "rest",
        "split": "val",
    }
    sampler = SessionSampler(dataset, cfg)
    samples = sampler.sample_sessions()
    captured = capsys.readouterr()
    assert f"dataset0/{long_session}" in captured.out
    assert f"dataset0/{short_session}" not in captured.out
    assert len(samples) == 1
    assert samples[0].session == long_session


def test_session_sampler_filters_by_dataset_key(tmp_path, capsys):
    root_a = tmp_path / "dataset_a"
    root_b = tmp_path / "dataset_b"
    data = np.zeros((2, 8), dtype=np.float32)
    make_dummy_session(str(root_a), "rest_a", data=data, chunk_idx=0)
    make_dummy_session(str(root_b), "rest_b", data=data, chunk_idx=0)

    indices = [
        ("dataset_a", "rest_a", "0.npy", 0),
        ("dataset_b", "rest_b", "0.npy", 0),
    ]

    class DummyDataset:
        def __init__(self):
            self.root_dirs = {
                "dataset_a": str(root_a),
                "dataset_b": str(root_b),
            }
            self.ch_names = ["c0", "c1"]
            self.fill_value = 0
            self.indices = indices
            self.sfreq = 1.0

        def _get_session_indices(self, dataset_key, session_name, n_channels):
            return np.arange(n_channels)

    cfg = {
        "context_length_s": 2,
        "total_length_s": 4,
        "num_sessions": 0,
        "split": "val",
        "dataset_key": "dataset_b",
    }
    sampler = SessionSampler(DummyDataset(), cfg)
    samples = sampler.sample_sessions()
    captured = capsys.readouterr()
    assert "dataset_b/rest_b" in captured.out
    assert "dataset_a/rest_a" not in captured.out
    assert len(samples) == 1
    assert samples[0].dataset_key == "dataset_b"


def test_session_sampler_orders_sessions_deterministically(tmp_path, monkeypatch):
    root_a = tmp_path / "dataset_a"
    root_b = tmp_path / "dataset_b"
    sessions = ["b_session", "a_session"]
    data = np.zeros((2, 8), dtype=np.float32)

    for root in (root_a, root_b):
        for session in sessions:
            make_dummy_session(str(root), session, data=data, chunk_idx=0)

    indices = []
    for dataset_key in ("dataset_b", "dataset_a"):
        for session in sessions:
            indices.append((dataset_key, session, "0.npy", 0))

    class DummyDataset:
        def __init__(self):
            self.root_dirs = {
                "dataset_b": str(root_b),
                "dataset_a": str(root_a),
            }
            self.ch_names = ["c0", "c1"]
            self.fill_value = 0
            self.indices = indices
            self.sfreq = 1.0

        def _get_session_indices(self, dataset_key, session_name, n_channels):
            return np.arange(n_channels)

    orig_iterdir = Path.iterdir

    def iterdir_reversed(self):
        items = list(orig_iterdir(self))
        if self in (root_a, root_b):
            return iter(reversed(items))
        return iter(items)

    monkeypatch.setattr(Path, "iterdir", iterdir_reversed)

    cfg = {
        "context_length_s": 2,
        "total_length_s": 4,
        "num_sessions": 0,
        "split": "val",
    }
    sampler = SessionSampler(DummyDataset(), cfg)
    samples = sampler.sample_sessions()
    ordered = [(sample.dataset_key, sample.session) for sample in samples]
    assert ordered == [
        ("dataset_a", "a_session"),
        ("dataset_a", "b_session"),
        ("dataset_b", "a_session"),
        ("dataset_b", "b_session"),
    ]


def test_generation_input_key_handles_nested_structures():
    inputs = (torch.zeros(1, 2), {"pos": torch.ones(1, 3)})
    key = _generation_input_key(inputs)
    assert key is not None
    assert key[0] == "tuple"


def test_stack_generation_inputs_stacks_tensors_and_dicts():
    inputs_a = (torch.ones(1, 2), {"pos": torch.zeros(1, 3)})
    inputs_b = (torch.full((1, 2), 2.0), {"pos": torch.ones(1, 3)})
    stacked = _stack_generation_inputs([inputs_a, inputs_b])
    assert isinstance(stacked, tuple)
    assert stacked[0].shape == (2, 2)
    assert torch.allclose(stacked[0], torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    assert isinstance(stacked[1], dict)
    assert stacked[1]["pos"].shape == (2, 3)


def test_group_generation_samples_batches_by_key():
    samples = [
        {"inputs": torch.zeros(1, 2), "rollout_horizon": 5, "run_idx": 0},
        {"inputs": torch.ones(1, 2), "rollout_horizon": 5, "run_idx": 1},
        {"inputs": torch.zeros(1, 3), "rollout_horizon": 5, "run_idx": 2},
        {"inputs": torch.zeros(1, 2), "rollout_horizon": 4, "run_idx": 3},
    ]
    batches = _group_generation_samples(samples, batch_size=2)
    assert batches[0][0]["run_idx"] == 0
    assert batches[0][1]["run_idx"] == 1
    remaining = [batch[0]["run_idx"] for batch in batches[1:]]
    assert set(remaining) == {2, 3}


def test_rollout_generator_plots_trim_context_and_caps_pair_plots(tmp_path):
    data = np.arange(1 * 5, dtype=np.float32).reshape(1, 5)
    sample = SessionSample(
        dataset_key="dataset0",
        session="session0",
        task_type=None,
        data=data,
        condition=None,
        context_steps=2,
        total_steps=5,
        pos=None,
        sensor_type=None,
    )

    class DummyModel(torch.nn.Module):
        def forecast(self, inputs, horizon, sample_fn, **kwargs):
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            if not torch.is_tensor(inputs):
                inputs = torch.as_tensor(inputs)
            batch = inputs.shape[0] if inputs.dim() >= 2 else 1
            channels = inputs.shape[1] if inputs.dim() >= 3 else 1
            return torch.zeros((batch, channels, horizon), dtype=torch.float32)

    class DummyPlotting:
        def __init__(self):
            self.psd_args = None
            self.timeseries_calls = 0
            self.stft_calls = 0
            self.stacked_calls = 0

        def plot_psd_cov_pair(self, generated, target, out_dir, prefix=""):
            self.psd_args = (generated, target)

        def plot_timeseries_pair(self, *args, **kwargs):
            self.timeseries_calls += 1

        def plot_stft_pair(self, *args, **kwargs):
            self.stft_calls += 1

        def plot_stacked_timeseries_pair(self, *args, **kwargs):
            self.stacked_calls += 1

    plotting = DummyPlotting()
    generator = RolloutGenerator(
        {"rollouts_per_context": 12, "rollout_batch_size": 1, "seed": 0},
        device=torch.device("cpu"),
        sfreq=1.0,
    )

    result = generator.generate(
        DummyModel(),
        [sample],
        out_dir=tmp_path,
        plotting=plotting,
    )

    assert result is not None
    assert plotting.psd_args is not None
    generated_no_context, target_no_context = plotting.psd_args
    assert len(generated_no_context) == 12
    assert generated_no_context[0].shape == (1, 3)
    assert target_no_context[0].shape == (1, 3)
    assert np.allclose(generated_no_context[0], 0.0)
    assert np.array_equal(target_no_context[0], data[:, 2:5])
    assert plotting.timeseries_calls == 10
    assert plotting.stft_calls == 10
    assert plotting.stacked_calls == 1


def test_rollout_generator_plot_channels_use_sampling_seed(tmp_path):
    data = np.arange(12 * 30, dtype=np.float32).reshape(12, 30)
    sample = SessionSample(
        dataset_key="dataset0",
        session="session0",
        task_type=None,
        data=data,
        condition=None,
        context_steps=10,
        total_steps=30,
        pos=None,
        sensor_type=None,
    )

    class DummyModel(torch.nn.Module):
        def forecast(self, inputs, horizon, sample_fn, **kwargs):
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            if not torch.is_tensor(inputs):
                inputs = torch.as_tensor(inputs)
            batch = inputs.shape[0] if inputs.dim() >= 2 else 1
            channels = inputs.shape[1] if inputs.dim() >= 3 else 1
            return torch.zeros((batch, channels, horizon), dtype=torch.float32)

    class DummyPlotting:
        def __init__(self):
            self.indices = []

        def plot_psd_cov_pair(self, generated, target, out_dir, prefix=""):
            _ = (generated, target, out_dir, prefix)

        def plot_timeseries_pair(
            self, generated, target, prefix, out_dir, channel_indices=None, **kwargs
        ):
            _ = (generated, target, prefix, out_dir, kwargs)
            self.indices.append(np.asarray(channel_indices))

        def plot_stft_pair(
            self, generated, target, prefix, out_dir, channel_indices=None, **kwargs
        ):
            _ = (generated, target, prefix, out_dir, kwargs)
            self.indices.append(np.asarray(channel_indices))

        def plot_stacked_timeseries_pair(self, *args, **kwargs):
            _ = (args, kwargs)

    cfg = {
        "rollouts_per_context": 1,
        "rollout_batch_size": 1,
        "sampling": {"seed": 123},
    }

    def run_once():
        plotting = DummyPlotting()
        generator = RolloutGenerator(cfg, device=torch.device("cpu"), sfreq=1.0)
        generator.generate(
            DummyModel(),
            [sample],
            out_dir=tmp_path,
            plotting=plotting,
        )
        return plotting.indices

    first = run_once()
    second = run_once()

    assert len(first) == 2
    assert len(second) == 2
    assert np.array_equal(first[0], first[1])
    assert np.array_equal(second[0], second[1])
    assert np.array_equal(first[0], second[0])


def test_rollout_generator_plots_cached_rollouts(tmp_path):
    generated = np.zeros((3, 2, 12), dtype=np.float32)
    target = np.ones((3, 2, 12), dtype=np.float32)
    result = GenerationResult(
        generated=generated,
        target=target,
        context_steps=4,
        total_steps=12,
        metadata=[],
    )

    class DummyPlotting:
        def __init__(self):
            self.timeseries_calls = 0
            self.stft_calls = 0
            self.stacked_calls = 0

        def plot_timeseries_pair(self, *args, **kwargs):
            self.timeseries_calls += 1

        def plot_stft_pair(self, *args, **kwargs):
            self.stft_calls += 1

        def plot_stacked_timeseries_pair(self, *args, **kwargs):
            self.stacked_calls += 1

    plotting = DummyPlotting()
    generator = RolloutGenerator(
        {"seed": 0},
        device=torch.device("cpu"),
        sfreq=1.0,
    )
    generator.plot_cached_rollouts(result, out_dir=tmp_path, plotting=plotting)

    assert plotting.timeseries_calls == 0
    assert plotting.stft_calls == 0
    assert plotting.stacked_calls == 1


def test_rollout_generator_build_inputs_includes_sensor_metadata():
    data = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    sample = SessionSample(
        dataset_key="dataset0",
        session="session0",
        task_type=None,
        data=data,
        condition=None,
        context_steps=2,
        total_steps=4,
        pos=np.zeros((2, 2), dtype=np.float32),
        sensor_type=np.array([0, 1], dtype=np.int64),
    )
    generator = RolloutGenerator(
        {"rollouts_per_context": 1, "rollout_batch_size": 1},
        device=torch.device("cpu"),
        sfreq=1.0,
    )
    inputs = generator._build_inputs(sample, sample.data[:, : sample.context_steps])
    assert isinstance(inputs, tuple)
    x, pos, sensor_type = inputs
    assert x.shape == (1, 2, 2)
    assert pos.shape == (1, 2, 2)
    assert sensor_type.shape == (1, 2)


def test_rollout_generator_build_inputs_multichannel_condition():
    data = np.arange(2 * 6, dtype=np.float32).reshape(2, 6)
    condition = np.arange(2 * 6, dtype=np.int64).reshape(2, 6)
    sample = SessionSample(
        dataset_key="dataset0",
        session="session0",
        task_type=None,
        data=data,
        condition=condition,
        context_steps=3,
        total_steps=6,
        pos=None,
        sensor_type=None,
    )
    generator = RolloutGenerator(
        {"rollouts_per_context": 1, "rollout_batch_size": 1},
        device=torch.device("cpu"),
        sfreq=1.0,
    )
    inputs = generator._build_inputs(sample, sample.data[:, : sample.context_steps])
    assert isinstance(inputs, tuple)
    x, cond = inputs
    assert x.shape == (1, 2, 3)
    assert cond.shape == (1, 2, 3)


def test_move_inputs_to_device_handles_nested_structures():
    generator = RolloutGenerator({}, device=torch.device("cpu"), sfreq=1.0)
    nested = (
        (torch.zeros(1, 2), {"pos": torch.ones(1, 3)}),
        torch.full((1, 4), 2.0),
    )
    moved = generator._move_inputs_to_device(nested)
    assert isinstance(moved, tuple)
    assert isinstance(moved[0], tuple)
    assert torch.is_tensor(moved[0][0])
    assert torch.is_tensor(moved[0][1]["pos"])
    assert moved[0][0].device.type == "cpu"
    assert moved[0][1]["pos"].device.type == "cpu"
    assert moved[1].device.type == "cpu"


def test_rollout_generator_drops_debug_timing_kwarg():
    generator = RolloutGenerator(
        {"debug_timing": True},
        device=torch.device("cpu"),
        sfreq=1.0,
    )
    calls = {"count": 0}

    def forecast(
        inputs,
        horizon,
        sample_fn,
        *,
        use_cache=None,
        sliding_window_overlap=0.5,
        max_context_tokens=-1,
    ):
        _ = (use_cache, sliding_window_overlap, max_context_tokens, sample_fn)
        calls["count"] += 1
        return torch.zeros((1, 1, horizon), dtype=torch.float32)

    def sample_fn(logits):
        return torch.zeros((1,), dtype=torch.long)

    inputs = torch.zeros((1, 2), dtype=torch.float32)
    out = generator._run_forecast(
        forecast, inputs, 2, sample_fn, generator._forecast_kwargs()
    )
    assert torch.is_tensor(out)
    assert calls["count"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_run_forecast_moves_outputs_to_cpu_on_cuda():
    generator = RolloutGenerator({}, device=torch.device("cuda"), sfreq=1.0)

    def forecast(
        inputs,
        horizon,
        sample_fn,
        **kwargs,
    ):
        _ = (inputs, sample_fn, kwargs)
        return {
            "output": torch.zeros((1, 1, horizon), device="cuda", dtype=torch.float16)
        }

    def sample_fn(logits):
        _ = logits
        return torch.zeros((1,), device="cuda", dtype=torch.long)

    inputs = torch.zeros((1, 2), dtype=torch.float32)
    out = generator._run_forecast(
        forecast, inputs, 2, sample_fn, generator._forecast_kwargs()
    )
    assert isinstance(out, dict)
    tensor = out["output"]
    assert torch.is_tensor(tensor)
    assert tensor.device.type == "cpu"


def test_session_sampler_extracts_condition_channel(tmp_path):
    root = tmp_path / "omega"
    session = "auditory_sub-004"
    data = np.arange(3 * 6, dtype=np.float32).reshape(3, 6)
    make_dummy_session(str(root), session, data=data, chunk_idx=0)

    indices = [("dataset0", session, "0.npy", 0)]
    dataset = _dummy_dataset(root, indices, sfreq=1.0, has_condition=True)
    cfg = {
        "context_length_s": 2,
        "total_length_s": 6,
        "num_sessions": 1,
        "split": "val",
    }
    sampler = SessionSampler(dataset, cfg)
    samples = sampler.sample_sessions()
    assert len(samples) == 1
    sample = samples[0]
    assert sample.condition is not None
    assert np.array_equal(sample.condition, data[-1])


def test_stack_rollout_runs_filters_metadata():
    plotting = EvaluationPlotting(sfreq=1.0)
    analysis = RolloutDivergenceAnalysis({}, sfreq=1.0, plotting=plotting)

    gen_runs = [
        np.zeros((2, 10), dtype=np.float32),
        np.ones((2, 10), dtype=np.float32),
        np.ones((3, 10), dtype=np.float32),
    ]
    tgt_runs = [
        np.zeros((2, 10), dtype=np.float32),
        np.ones((2, 10), dtype=np.float32),
        np.ones((3, 10), dtype=np.float32),
    ]
    metadata = [{"run": "a"}, {"run": "b"}, {"run": "c"}]
    analysis.set_rollout_info(metadata=metadata, context_steps=1, total_steps=10)

    stacked = analysis._stack_rollout_runs(gen_runs, tgt_runs)
    assert stacked is not None
    gen_stack, tgt_stack = stacked
    assert gen_stack.shape[0] == 2
    assert tgt_stack.shape[0] == 2
    assert analysis.metadata is not None
    assert [entry.get("run") for entry in analysis.metadata] == ["a", "b"]


def test_rollout_divergence_zero_when_sequences_match():
    plotting = EvaluationPlotting(sfreq=100.0)
    analysis = RolloutDivergenceAnalysis({}, sfreq=100.0, plotting=plotting)
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False, dtype=np.float64)
    gen = np.stack(
        [
            np.sin(t),
            np.cos(t),
        ]
    )
    curves = analysis._rollout_divergence_curve(
        gen[None, ...], gen[None, ...], window_steps=50
    )
    curves = curves["gen"]
    expected = {
        "correlation",
        "covariance",
        "psd_jsd",
        "band_jsd",
        "stft_magnitude",
        "stft_angle",
        "fft_magnitude",
        "fft_angle",
    }
    assert expected.issubset(curves.keys())
    for name in expected:
        curve = curves[name][0]
        assert curve.size > 0
        finite = curve[np.isfinite(curve)]
        assert finite.size > 0
        assert np.allclose(finite, 0.0, atol=1e-7), name


def test_rollout_divergence_includes_final_window():
    plotting = EvaluationPlotting(sfreq=None)
    analysis = RolloutDivergenceAnalysis({}, sfreq=None, plotting=plotting)
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False, dtype=np.float64)
    target = np.stack([np.sin(t), np.cos(t)])

    curves = analysis._rollout_divergence_curve(
        target[None, ...], target[None, ...], window_steps=50
    )
    curves = curves["gen"]
    expected_len = target.shape[1] // 50

    for name, curve in curves.items():
        assert curve.shape[1] == expected_len, name


def test_build_sample_fn_rvq_temperature_curriculum(monkeypatch):
    generator = RolloutGenerator(
        {},
        device=torch.device("cpu"),
        sfreq=None,
    )
    params = {
        "temperature": 1.0,
        "temperature_decay": 0.5,
        "temperature_curriculum": True,
        "top_p": 1.0,
    }

    temps_seen = []

    def fake_sample(logits, **kwargs):
        temps_seen.append(kwargs["temperature"].detach().cpu())
        return torch.zeros(logits.shape[:-1], dtype=torch.long)

    monkeypatch.setattr(generation, "sample_logits", fake_sample)

    sample_fn = generator._build_sample_fn(
        params, rvq_levels=3, default_curriculum=True
    )
    sample_fn(torch.randn(2, 4, 10))
    sample_fn(torch.randn(2, 2, 10))

    assert len(temps_seen) == 2
    first = temps_seen[0].squeeze()
    second = temps_seen[1].squeeze()
    assert torch.allclose(first, torch.tensor([1.0, 0.5, 0.25, 1.0]))
    assert torch.allclose(second, torch.tensor([0.5, 0.25]))


def test_build_sample_fn_rvq_temperature_curriculum_default_off(monkeypatch):
    generator = RolloutGenerator(
        {},
        device=torch.device("cpu"),
        sfreq=None,
    )
    params = {"temperature": 1.25, "top_p": 1.0}

    temps_seen = []

    def fake_sample(logits, **kwargs):
        temps_seen.append(kwargs["temperature"])
        return torch.zeros(logits.shape[:-1], dtype=torch.long)

    monkeypatch.setattr(generation, "sample_logits", fake_sample)

    sample_fn = generator._build_sample_fn(
        params, rvq_levels=4, default_curriculum=False
    )
    sample_fn(torch.randn(2, 3, 5))

    assert temps_seen == [1.25]


def test_build_sample_fn_rvq_temperature_levels_enable_curriculum(monkeypatch):
    generator = RolloutGenerator(
        {},
        device=torch.device("cpu"),
        sfreq=None,
    )
    params = {"temperature_levels": [1.0, 0.7, 0.5], "top_p": 1.0}

    temps_seen = []

    def fake_sample(logits, **kwargs):
        temps_seen.append(kwargs["temperature"].detach().cpu())
        return torch.zeros(logits.shape[:-1], dtype=torch.long)

    monkeypatch.setattr(generation, "sample_logits", fake_sample)

    sample_fn = generator._build_sample_fn(
        params, rvq_levels=3, default_curriculum=False
    )
    sample_fn(torch.randn(2, 4, 10))

    first = temps_seen[0].squeeze()
    assert torch.allclose(first, torch.tensor([1.0, 0.7, 0.5, 1.0]))


def test_rollout_divergence_respects_stride_steps():
    plotting = EvaluationPlotting(sfreq=None)
    analysis = RolloutDivergenceAnalysis({}, sfreq=None, plotting=plotting)
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False, dtype=np.float64)
    target = np.stack([np.sin(t), np.cos(t)])

    curves = analysis._rollout_divergence_curve(
        target[None, ...], target[None, ...], window_steps=50, stride_steps=20
    )
    curves = curves["gen"]
    expected_len = ((target.shape[1] - 50) // 20) + 1

    for name, curve in curves.items():
        assert curve.shape[1] == expected_len, name
        run_curve = curve[0]
        if np.isfinite(run_curve).all():
            assert np.allclose(run_curve, 0.0, atol=1e-7)


def test_rollout_divergence_includes_band_coherence_when_sfreq_set():
    plotting = EvaluationPlotting(sfreq=100.0)
    analysis = RolloutDivergenceAnalysis({}, sfreq=100.0, plotting=plotting)
    t = np.linspace(0, 2, 400, endpoint=False, dtype=np.float64)
    target = np.stack([np.sin(2 * np.pi * 10 * t), np.cos(2 * np.pi * 20 * t)])

    curves = analysis._rollout_divergence_curve(
        target[None, ...], target[None, ...], window_steps=100
    )
    curves = curves["gen"]
    coherence_keys = [k for k in curves.keys() if k.startswith("coherence_")]

    assert coherence_keys
    for key in coherence_keys:
        curve = curves[key][0]
        finite = curve[np.isfinite(curve)]
        assert finite.size > 0
        assert np.allclose(finite, 0.0, atol=1e-4)


def test_rollout_divergence_includes_psd_band_metrics_when_sfreq_set():
    plotting = EvaluationPlotting(sfreq=100.0)
    analysis = RolloutDivergenceAnalysis({}, sfreq=100.0, plotting=plotting)
    t = np.linspace(0, 2, 400, endpoint=False, dtype=np.float64)
    target = np.stack([np.sin(2 * np.pi * 10 * t), np.cos(2 * np.pi * 20 * t)])

    curves = analysis._rollout_divergence_curve(
        target[None, ...], target[None, ...], window_steps=100
    )
    curves = curves["gen"]
    bands = resolve_frequency_bands(None)
    jsd_keys = [f"psd_jsd_{band}" for band in bands]
    corr_keys = [f"psd_corr_{band}" for band in bands]
    assert all(key in curves for key in jsd_keys)
    assert all(key in curves for key in corr_keys)
    for key in jsd_keys:
        curve = curves[key][0]
        finite = curve[np.isfinite(curve)]
        assert finite.size > 0
        assert np.allclose(finite, 0.0, atol=1e-6)
    for key in corr_keys:
        curve = curves[key][0]
        finite = curve[np.isfinite(curve)]
        if finite.size > 0:
            assert np.allclose(finite, 1.0, atol=1e-6)


def test_rollout_divergence_includes_long_range_metrics():
    plotting = EvaluationPlotting(sfreq=100.0)
    analysis = RolloutDivergenceAnalysis({}, sfreq=100.0, plotting=plotting)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1200)).astype(np.float64)

    curves = analysis._rollout_divergence_curve(
        data[None, ...], data[None, ...], window_steps=200
    )
    curves = curves["gen"]
    expected_keys = [
        "amplitude_kurtosis",
        "amplitude_tail_fraction",
        "dfa_exponent",
        "hurst_exponent",
        "one_over_f_exponent",
        "spatial_connectivity",
    ]

    for key in expected_keys:
        assert key in curves
        curve = curves[key][0]
        assert curve.size > 0
        assert np.all(np.isfinite(curve))

    assert np.allclose(curves["amplitude_kurtosis"][0], 0.0, atol=1e-8)
    assert np.allclose(curves["amplitude_tail_fraction"][0], 0.0, atol=1e-8)
    assert np.allclose(curves["dfa_exponent"][0], 0.0, atol=1e-6)
    assert np.allclose(curves["hurst_exponent"][0], 0.0, atol=1e-6)
    assert np.allclose(curves["one_over_f_exponent"][0], 0.0, atol=1e-6)
    assert np.allclose(curves["spatial_connectivity"][0], 0.0, atol=1e-8)

    bandpower_keys = [
        key for key in curves.keys() if key.startswith("bandpower_ratio_")
    ]
    assert bandpower_keys
    for key in bandpower_keys:
        assert np.allclose(curves[key][0], 0.0, atol=1e-6)


def test_rollout_divergence_grows_with_noise():
    plotting = EvaluationPlotting(sfreq=None)
    analysis = RolloutDivergenceAnalysis({}, sfreq=None, plotting=plotting)
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 200, endpoint=False, dtype=np.float64)
    target = np.stack([np.sin(t), np.cos(t)])
    noise_scale = np.linspace(0.0, 1.5, t.size, dtype=np.float64)
    noisy = target + rng.standard_normal(target.shape) * noise_scale

    curves = analysis._rollout_divergence_curve(
        noisy[None, ...], target[None, ...], window_steps=50
    )
    curves = curves["gen"]
    corr = curves["correlation"][0]
    fft_mag = curves["fft_magnitude"][0]

    assert corr[-1] > corr[0]
    assert corr[-1] > 0.1
    assert fft_mag[-1] > fft_mag[0]
    assert np.all(fft_mag >= 0)


def test_divergence_identity_metrics_near_zero():
    plotting = EvaluationPlotting(sfreq=100.0)
    analysis = RolloutDivergenceAnalysis({}, sfreq=100.0, plotting=plotting)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 500)).astype(np.float64)

    curves = analysis._rollout_divergence_curve(
        data[None, ...], data[None, ...], window_steps=200
    )
    curves = curves["gen"]
    expected_metrics = [
        "correlation",
        "covariance",
        "fft_magnitude",
        "fft_angle",
        "stft_magnitude",
        "stft_angle",
        "psd_jsd",
        "band_jsd",
    ]
    for name in expected_metrics:
        curve = curves[name][0]
        finite = curve[np.isfinite(curve)]
        assert finite.size > 0
        assert np.allclose(finite, 0.0, atol=1e-7), name


def test_correlation_divergence_constant_identity():
    dist = MetricDistances(RolloutMetricsConfig())
    zeros = np.zeros((2, 500), dtype=np.float64)
    pair_ts = np.stack([zeros, zeros], axis=0)[None, ...]
    val = dist._correlation_divergence(pair_ts)[0]
    assert np.isclose(val, 0.0)


def test_correlation_divergence_scale_offset_invariant():
    dist = MetricDistances(RolloutMetricsConfig())
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 500)).astype(np.float64)
    y = 3.0 * x + 10.0
    pair_ts = np.stack([x, y], axis=0)[None, ...]
    val = dist._correlation_divergence(pair_ts)[0]
    assert val < 1e-8


def test_phase_shift_affects_phase_not_magnitude():
    plotting = EvaluationPlotting(sfreq=128.0)
    analysis = RolloutDivergenceAnalysis({}, sfreq=128.0, plotting=plotting)
    timesteps = 1024
    t = np.arange(timesteps, dtype=np.float64) / 128.0
    freq = 8.0
    target = np.sin(2 * np.pi * freq * t)
    shifted = np.sin(2 * np.pi * freq * t + np.pi / 2)

    curves = analysis._rollout_divergence_curve(
        shifted[None, None, :], target[None, None, :], window_steps=timesteps
    )
    curves = curves["gen"]
    fft_mag = float(curves["fft_magnitude"][0, -1])
    fft_angle = float(curves["fft_angle"][0, -1])
    stft_mag = float(curves["stft_magnitude"][0, -1])
    stft_angle = float(curves["stft_angle"][0, -1])

    assert fft_mag < 0.05
    assert fft_angle > fft_mag
    assert stft_angle > stft_mag


def test_jsd_distance_properties():
    dist = MetricDistances(RolloutMetricsConfig())
    p = np.array([1.0, 0.0], dtype=np.float64)
    q = np.array([0.0, 1.0], dtype=np.float64)
    d_pq = float(dist._jsd(p, q))
    d_qp = float(dist._jsd(q, p))

    assert np.isclose(float(dist._jsd(p, p)), 0.0)
    assert np.isclose(d_pq, d_qp)
    assert 0.83 < d_pq < 0.84


def test_rollout_metrics_from_params_matches_defaults():
    cfg_default = RolloutMetricsConfig()
    cfg_from = RolloutMetricsConfig.from_params({}, sfreq=123.0)

    assert cfg_from.fs == 123.0
    assert cfg_from.fmin == cfg_default.fmin
    assert cfg_from.stft_noverlap_frac == cfg_default.stft_noverlap_frac
    assert cfg_from.welch_noverlap_frac == cfg_default.welch_noverlap_frac


def test_rollout_metrics_from_params_respects_fs_without_sfreq():
    cfg_from = RolloutMetricsConfig.from_params(
        {"timeseries_divergence": {"fs": 200.0}}, sfreq=None
    )
    assert cfg_from.fs == 200.0


def test_token_entropy_vectorized_matches_naive():
    tokens = np.array(
        [
            [[1, 1, 2, 3], [1, 2, 2, 3], [1, 3, 2, 4]],
            [[0, 0, 1, 1], [0, 2, 1, 3], [0, 2, 1, 3]],
        ],
        dtype=np.int64,
    )
    metrics = RolloutMetrics(RolloutMetricsConfig())
    fast = metrics._token_entropy_per_step(tokens)

    batch, _, steps = tokens.shape
    entropies = np.full((batch, steps), np.nan, dtype=np.float32)
    for b in range(batch):
        for t in range(steps):
            vals = tokens[b, :, t].ravel()
            _, counts = np.unique(vals, return_counts=True)
            probs = counts.astype(np.float32)
            probs = probs / np.sum(probs)
            entropies[b, t] = -np.sum(probs * np.log2(probs + 1e-12))
    expected = np.nanmean(entropies, axis=1)

    assert np.allclose(fast, expected, atol=1e-6)


def test_relative_l2_handles_tiny_target():
    dist = MetricDistances(RolloutMetricsConfig())
    target = np.zeros((3, 10), dtype=np.float32)
    generated = np.ones((3, 10), dtype=np.float32)
    val = dist._relative_l2(generated[None, ...], target[None, ...])[0]
    assert np.isfinite(val)
    assert np.isclose(val, 1.0)


def test_evaluate_rollout_divergence_includes_spectral_metrics(tmp_path):
    plotting = EvaluationPlotting(sfreq=100.0)
    analysis = RolloutDivergenceAnalysis({}, sfreq=100.0, plotting=plotting)

    t = np.linspace(0, 2 * np.pi, 500, dtype=np.float32)
    target = np.stack([np.sin(t), np.cos(t)])
    generated = target * 0.9

    analysis.run(
        np.stack([generated], axis=0),
        np.stack([target], axis=0),
        tmp_path,
    )

    payload = json.loads((tmp_path / "rollout_divergence.json").read_text())
    metrics = payload["metrics"]
    for key in ("psd_jsd", "band_jsd", "stft_magnitude", "stft_angle"):
        assert key in metrics
        assert "median" in metrics[key]
        assert "q25" in metrics[key]
        assert "q75" in metrics[key]
        assert metrics[key]["median"]


def test_rollout_divergence_prompt_swap_stats(tmp_path):
    plotting = EvaluationPlotting(sfreq=10.0)
    params = {
        "prefix_times_s": [0.5, 1.0],
        "controls": {"prompt_swap": True, "real_real": True},
        "plot_rollout_divergence": False,
        "stats": {
            "unit_of_replication": "context",
            "horizons": [0.5],
            "comparisons": ["correct_vs_prompt_swap", "correct_vs_real_real"],
            "report": ["median_delta", "prob_improvement"],
            "inference": {
                "ci": {"method": "bootstrap", "n_boot": 50, "seed": 0},
                "test": {"type": "wilcoxon_signed_rank"},
            },
        },
    }
    analysis = RolloutDivergenceAnalysis(params, sfreq=10.0, plotting=plotting)

    t = np.linspace(0, 2 * np.pi, 20, endpoint=False, dtype=np.float32)
    runs = [
        np.stack([np.sin(t), np.cos(t)]),
        np.stack([np.sin(t + 0.5), np.cos(t + 0.5)]),
        np.stack([np.sin(t + 1.0), np.cos(t + 1.0)]),
    ]
    generated = np.stack(runs, axis=0)
    target = np.stack(runs, axis=0)

    analysis.run(generated, target, tmp_path)

    payload = json.loads((tmp_path / "rollout_divergence.json").read_text())
    metrics = payload["metrics"]
    assert "covariance" in metrics
    prompt = metrics["covariance"]["prompt_swap"]
    assert prompt is not None
    median = np.asarray(prompt["median"], dtype=np.float32)
    assert np.any(median > 0)

    stats_path = tmp_path / "rollout_divergence_stats.json"
    assert stats_path.exists()
    stats_payload = json.loads(stats_path.read_text())
    assert stats_payload["comparisons"]


def test_rollout_divergence_stats_similarity_delta(tmp_path):
    plotting = EvaluationPlotting(sfreq=None)
    params = {
        "stats": {
            "comparisons": ["correct_vs_prompt_swap"],
            "horizons": [0.0],
        }
    }
    analysis = RolloutDivergenceAnalysis(params, sfreq=None, plotting=plotting)

    metrics = {
        "psd_corr": {
            "runs": [np.array([0.9], dtype=np.float32)],
            "context_ids": [0],
            "prompt_swap": {
                "runs": [np.array([0.5], dtype=np.float32)],
                "context_ids": [0],
            },
        },
        "covariance": {
            "runs": [np.array([0.2], dtype=np.float32)],
            "context_ids": [0],
            "prompt_swap": {
                "runs": [np.array([0.8], dtype=np.float32)],
                "context_ids": [0],
            },
        },
    }

    analysis._compute_stats(
        params, metrics, np.array([0.0], dtype=np.float32), tmp_path
    )

    payload = json.loads((tmp_path / "rollout_divergence_stats.json").read_text())
    entries = payload["comparisons"]
    psd_entry = next(e for e in entries if e["metric"] == "psd_corr")
    cov_entry = next(e for e in entries if e["metric"] == "covariance")
    assert psd_entry["median_delta"] > 0
    assert cov_entry["median_delta"] > 0


def test_evaluate_rollout_window_metrics_outputs(tmp_path):
    plotting = EvaluationPlotting(sfreq=100.0)
    params = {
        "window_length_s": 1.0,
        "stride_s": 0.5,
        "out_of_envelope_rate": "mean",
        "plot_rollout_window_metrics": True,
    }
    analysis = RolloutSlidingWindowAnalysis(params, sfreq=100.0, plotting=plotting)

    t = np.linspace(0, 2 * np.pi, 500, dtype=np.float32)
    target = np.stack([np.sin(t), np.cos(t)])
    generated = target * 0.9

    analysis.run(
        np.stack([generated], axis=0),
        np.stack([target], axis=0),
        tmp_path,
    )

    json_path = tmp_path / "rollout_window_metrics.json"
    plot_path = tmp_path / "rollout_window_metrics.png"
    assert json_path.exists()
    assert plot_path.exists()

    payload = json.loads(json_path.read_text())
    metrics = payload["metrics"]
    assert "spatial_connectivity" in metrics
    assert "bandpower_ratio_delta" in metrics
    assert "cov_eig_entropy" in metrics
    assert "psd_entropy" in metrics
    assert "out_of_envelope_rate" in payload


def test_rollout_window_metrics_plot_metrics_filter(tmp_path, monkeypatch):
    plotting = EvaluationPlotting(sfreq=1.0)
    params = {
        "window_length_steps": 2,
        "stride_steps": 1,
        "out_of_envelope_rate": "mean",
        "plot_rollout_window_metrics": True,
        "plots": {"metrics": ["metric_a"]},
    }
    analysis = RolloutSlidingWindowAnalysis(params, sfreq=1.0, plotting=plotting)

    def _fake_rollout_window_metrics(
        generated,
        target,
        window_steps,
        stride_steps=None,
        params=None,
    ):
        return {
            "generated": {
                "metric_a": np.array([[0.0, 2.0]], dtype=np.float32),
                "metric_b": np.array([[1.0, 1.0]], dtype=np.float32),
            },
            "target": {
                "metric_a": np.array([[0.0, 0.0]], dtype=np.float32),
                "metric_b": np.array([[1.0, 1.0]], dtype=np.float32),
            },
            "window_starts": [0, 1],
        }

    monkeypatch.setattr(
        analysis, "_rollout_window_metrics", _fake_rollout_window_metrics
    )

    captured = {}

    def _capture_plot(metrics, out_dir, x, out_of_envelope=None, summary_curves=None):
        captured["metrics"] = list(metrics.keys())
        captured["summary_curves"] = summary_curves or []

    monkeypatch.setattr(plotting, "plot_rollout_window_metrics", _capture_plot)

    dummy = np.zeros((1, 2, 4), dtype=np.float32)
    analysis.run(dummy, dummy, tmp_path)

    assert captured["metrics"] == ["metric_a"]
    summary = next(
        entry
        for entry in captured["summary_curves"]
        if entry.get("key") == "out_of_envelope_rate"
    )
    assert np.allclose(summary["curve"], np.array([0.0, 1.0], dtype=np.float32))


def test_rollout_window_metrics_trims_context_and_diversity_proxy(tmp_path):
    plotting = EvaluationPlotting(sfreq=None)
    params = {
        "window_length_steps": 2,
        "stride_steps": 2,
        "plot_rollout_window_metrics": False,
    }
    analysis = RolloutSlidingWindowAnalysis(params, sfreq=None, plotting=plotting)

    t = np.linspace(0, 2 * np.pi, 6, endpoint=False, dtype=np.float32)
    target = np.stack([np.sin(t), np.cos(t)])
    target_runs = np.stack([target, target * 0.95], axis=0)
    generated_runs = np.stack([target * 0.9, target * 0.85], axis=0)

    analysis.set_rollout_info(context_steps=2, total_steps=6)
    analysis.run(
        generated_runs,
        target_runs,
        tmp_path,
    )

    payload = json.loads((tmp_path / "rollout_window_metrics.json").read_text())
    assert len(payload["x"]) == 2
    assert "diversity_proxy" in payload
    assert len(payload["diversity_proxy"]["mean"]) == 2

    x_len = len(payload["x"])
    metrics = payload["metrics"]
    first_metric = next(iter(metrics.values()))
    assert len(first_metric["generated"]["median"]) == x_len


def test_rollout_window_metrics_includes_token_stats(tmp_path):
    plotting = EvaluationPlotting(sfreq=None)
    params = {
        "window_length_steps": 10,
        "stride_steps": 5,
        "plot_rollout_window_metrics": True,
    }
    analysis = RolloutSlidingWindowAnalysis(params, sfreq=None, plotting=plotting)

    tokens = np.arange(60, dtype=np.int64).reshape(2, 30) % 7
    analysis.run(tokens[None, ...], tokens[None, ...], tmp_path)

    payload = json.loads((tmp_path / "rollout_window_metrics.json").read_text())
    metrics = payload["metrics"]
    assert "token_entropy" in metrics
    assert "token_unique_rate" in metrics
    assert "token_repetition_rate" in metrics


def test_generated_target_pair_plots(tmp_path):
    plotting = EvaluationPlotting(sfreq=10.0)

    t = np.linspace(0, 1, 256, endpoint=False, dtype=np.float32)
    generated = np.stack([np.sin(2 * np.pi * 2 * t), np.cos(2 * np.pi * 3 * t)])
    target = 0.9 * generated

    plotting.plot_timeseries_pair(generated, target, "pair", tmp_path, context_len=10)
    plotting.plot_stft_pair(generated, target, "pair", tmp_path, context_len=10)
    plotting.plot_stacked_timeseries_pair(
        generated, target, "pair", tmp_path, context_len=10
    )

    assert (tmp_path / "pair_timeseries.png").exists()
    assert (tmp_path / "pair_stft.png").exists()
    assert (tmp_path / "pair_stacked_timeseries.png").exists()


def test_plot_steps_respects_max_seconds():
    plotting = EvaluationPlotting(sfreq=10.0)
    assert plotting._resolve_plot_steps(200, None) == 200
    assert plotting._resolve_plot_steps(200, 5.0) == 50


def test_plot_examples_writes_summary_psd_and_cov(tmp_path):
    plotting = EvaluationPlotting(sfreq=50.0)

    t = np.linspace(0, 1, 256, endpoint=False, dtype=np.float32)
    base = np.stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 7 * t)])

    examples = []
    for scale in (1.0, 0.6):
        inp = torch.from_numpy(base * scale)
        out = torch.from_numpy(base * (scale * 0.9))
        tgt = torch.from_numpy(base * scale)
        examples.append((inp, out, tgt))

    figs = plotting.plot_examples(examples, tmp_path)

    assert len(figs) == 2
    assert (tmp_path / "examples_psd_summary.png").exists()
    assert (tmp_path / "examples_cov_summary.png").exists()


def test_plot_psd_cov_pair_uses_shared_color_scale(tmp_path, monkeypatch):
    plotting = EvaluationPlotting(sfreq=100.0)

    t = np.linspace(0, 1, 200, endpoint=False, dtype=np.float32)
    generated = np.stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 7 * t)])
    target = generated * 10.0

    captured = []
    original_imshow = Axes.imshow

    def _capture_imshow(self, *args, **kwargs):
        captured.append(kwargs)
        return original_imshow(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", _capture_imshow)

    plotting.plot_psd_cov_pair([generated], [target], tmp_path, prefix="pair")

    assert len(captured) == 2
    gen_cov = np.cov(generated)
    tgt_cov = np.cov(target)
    expected_vmin = min(float(np.nanmin(gen_cov)), float(np.nanmin(tgt_cov)))
    expected_vmax = max(float(np.nanmax(gen_cov)), float(np.nanmax(tgt_cov)))
    assert np.isclose(captured[0]["vmin"], expected_vmin)
    assert np.isclose(captured[1]["vmin"], expected_vmin)
    assert np.isclose(captured[0]["vmax"], expected_vmax)
    assert np.isclose(captured[1]["vmax"], expected_vmax)


def test_resolve_divergence_window_prefers_seconds_when_available():
    plotting = EvaluationPlotting(sfreq=100.0)
    analysis = RolloutDivergenceAnalysis({}, sfreq=100.0, plotting=plotting)

    params_steps = {"divergence_window_steps": 12}
    params_seconds = {"divergence_window_seconds": 0.25}

    assert analysis._resolve_divergence_window(params_steps) == 12
    assert analysis._resolve_divergence_window(params_seconds) == 25
