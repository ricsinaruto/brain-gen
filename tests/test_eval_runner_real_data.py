import json
import shutil
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import yaml

from brain_gen.eval.plotting import EvaluationPlotting
from brain_gen.eval.rollout_divergence import RolloutDivergenceAnalysis
from brain_gen.eval.rollout_sliding_windows import RolloutSlidingWindowAnalysis
from brain_gen.eval.rollout_metrics import resolve_frequency_bands


def _run_index(path: Path, prefix: str) -> int:
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected filename: {path}")
    return int(stem.replace(prefix, ""))


def _load_run_pairs(data_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    gen_files = sorted(
        data_dir.glob("generated_run*.npy"),
        key=lambda p: _run_index(p, "generated_run"),
    )
    tgt_files = sorted(
        data_dir.glob("target_run*.npy"),
        key=lambda p: _run_index(p, "target_run"),
    )
    if not gen_files or not tgt_files:
        raise FileNotFoundError(f"Missing run data under {data_dir}")

    gen_runs = [np.load(path).astype(np.float32) for path in gen_files]
    tgt_runs = [np.load(path).astype(np.float32) for path in tgt_files]
    return gen_runs, tgt_runs


def _filter_runs(
    gen_runs: list[np.ndarray], tgt_runs: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    shapes = [run.shape for run in gen_runs]
    shape_counts = {}
    for shape in shapes:
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    common_shape = max(shape_counts, key=shape_counts.get)
    filtered_pairs = [
        (gen, tgt)
        for gen, tgt in zip(gen_runs, tgt_runs)
        if gen.shape == common_shape and tgt.shape == common_shape
    ]

    gen_runs = [pair[0] for pair in filtered_pairs]
    tgt_runs = [pair[1] for pair in filtered_pairs]

    return gen_runs, tgt_runs, common_shape


def test_rollout_divergence_real_data_outputs(tmp_path):
    data_dir = Path(__file__).resolve().parent / "data" / "qwen2p5"
    cfg = yaml.safe_load((data_dir / "eval_runner.yaml").read_text())
    gen_cfg = cfg["eval_runner"]["generate"]

    gen_runs, tgt_runs = _load_run_pairs(data_dir)
    assert len(gen_runs) == len(tgt_runs)
    assert len(gen_runs) == int(gen_cfg["num_runs"])

    gen_runs, tgt_runs, common_shape = _filter_runs(gen_runs, tgt_runs)
    assert len(gen_runs) >= 2

    channels = {run.shape[0] for run in gen_runs}
    assert len(channels) == 1
    for gen, tgt in zip(gen_runs, tgt_runs):
        assert gen.shape == tgt.shape

    sfreq = 100
    plotting = EvaluationPlotting(sfreq=sfreq)
    analysis = RolloutDivergenceAnalysis({}, sfreq=sfreq, plotting=plotting)

    repo_root = Path(__file__).resolve().parents[1]
    params = yaml.safe_load(
        (repo_root / "configs" / "evals" / "test_rollout_divergence.yaml").read_text()
    )
    params.setdefault("timeseries_divergence", {})
    params["timeseries_divergence"]["fs"] = sfreq

    analysis.cfg.update(params)
    analysis.run(
        np.stack(gen_runs, axis=0),
        np.stack(tgt_runs, axis=0),
        tmp_path,
    )

    json_path = tmp_path / "rollout_divergence.json"
    plot_path = tmp_path / "rollout_divergence.png"
    pdf_path = tmp_path / "rollout_divergence.pdf"
    assert json_path.exists()
    assert plot_path.exists()
    assert pdf_path.exists()
    stats_json_path = tmp_path / "rollout_divergence_stats.json"
    stats_md_path = tmp_path / "rollout_divergence_stats.md"
    assert stats_json_path.exists()
    assert stats_md_path.exists()
    visible_dir = repo_root / "tmp"
    visible_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plot_path, visible_dir / plot_path.name)
    shutil.copy2(stats_json_path, visible_dir / stats_json_path.name)
    shutil.copy2(stats_md_path, visible_dir / stats_md_path.name)

    payload = json.loads(json_path.read_text())
    metrics = payload["metrics"]
    prefix_steps = payload.get("prefix_steps") or []
    expected_len = len(prefix_steps)
    assert expected_len > 0
    prefix_seconds = payload.get("prefix_seconds")
    if prefix_seconds is not None:
        assert len(prefix_seconds) == expected_len

    bands = resolve_frequency_bands(None)
    band_names = list(bands.keys())
    expected_metrics = {
        "correlation",
        "covariance",
        "psd_jsd",
        "band_jsd",
        "stft_magnitude",
        "stft_angle",
        "fft_magnitude",
        "fft_angle",
        "amplitude_kurtosis",
        "amplitude_tail_fraction",
        "dfa_exponent",
        "hurst_exponent",
        "one_over_f_exponent",
        "spatial_connectivity",
        "psd_corr",
    }
    for band in band_names:
        expected_metrics.add(f"coherence_{band}")
        expected_metrics.add(f"bandpower_ratio_{band}")
        expected_metrics.add(f"psd_corr_{band}")
        expected_metrics.add(f"psd_jsd_{band}")

    for key in expected_metrics:
        assert key in metrics
        entry = metrics[key]
        median = np.asarray(entry["median"], dtype=np.float32)
        assert median.shape[0] == expected_len
        assert np.isfinite(median).any()
        assert len(entry["per_run_lengths"]) == len(gen_runs)
        assert entry["target_swap"] is not None
        assert entry["prompt_swap"] is not None
        assert entry["baseline"] is not None
        target = entry["target_swap"]
        prompt = entry["prompt_swap"]
        baseline = entry["baseline"]
        assert len(target["per_run_lengths"]) == len(gen_runs)
        assert len(prompt["per_run_lengths"]) == len(gen_runs)
        assert len(baseline["per_run_lengths"]) == len(gen_runs)

    img = mpimg.imread(plot_path)
    assert img.size > 0

    stats_payload = json.loads(stats_json_path.read_text())
    assert stats_payload["comparisons"]


def test_rollout_window_metrics_real_data_outputs(tmp_path):
    data_dir = Path(__file__).resolve().parent / "data" / "qwen2p5"
    cfg = yaml.safe_load((data_dir / "eval_runner.yaml").read_text())
    gen_cfg = cfg["eval_runner"]["generate"]

    gen_runs, tgt_runs = _load_run_pairs(data_dir)
    assert len(gen_runs) == len(tgt_runs)
    assert len(gen_runs) == int(gen_cfg["num_runs"])

    gen_runs, tgt_runs, common_shape = _filter_runs(gen_runs, tgt_runs)
    assert len(gen_runs) >= 2

    channels = {run.shape[0] for run in gen_runs}
    assert len(channels) == 1
    for gen, tgt in zip(gen_runs, tgt_runs):
        assert gen.shape == tgt.shape

    sfreq = 100
    plotting = EvaluationPlotting(sfreq=sfreq)
    analysis = RolloutSlidingWindowAnalysis({}, sfreq=sfreq, plotting=plotting)
    repo_root = Path(__file__).resolve().parents[1]
    params = yaml.safe_load(
        (repo_root / "configs" / "evals" / "test_sliding_window.yaml").read_text()
    )
    params.setdefault("timeseries_divergence", {})
    params["timeseries_divergence"]["fs"] = sfreq

    steps = int(common_shape[1])

    analysis.cfg.update(params)
    analysis.run(
        np.stack(gen_runs, axis=0),
        np.stack(tgt_runs, axis=0),
        tmp_path,
    )

    json_path = tmp_path / "rollout_window_metrics.json"
    plot_path = tmp_path / "rollout_window_metrics.png"
    assert json_path.exists()
    assert plot_path.exists()

    # also copy to visible directory
    repo_root = Path(__file__).resolve().parents[1]
    visible_dir = repo_root / "tmp"
    visible_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plot_path, visible_dir / plot_path.name)

    payload = json.loads(json_path.read_text())
    metrics = payload["metrics"]
    assert "out_of_envelope_rate" in payload
    window_steps = int(payload["window_steps"])
    stride_steps = int(payload["stride_steps"])
    expected_len = ((steps - window_steps) // stride_steps) + 1
    x_vals = np.asarray(payload["x"], dtype=np.float32)
    assert x_vals.shape[0] == expected_len

    expected_metrics = {
        "spatial_connectivity",
        "amplitude_kurtosis",
        "amplitude_tail_fraction",
        "dfa_exponent",
        "hurst_exponent",
        "one_over_f_exponent",
        "cov_eig_entropy",
        "cov_eig_top_frac",
        "psd_entropy",
        "psd_centroid",
    }
    for band in resolve_frequency_bands(None):
        expected_metrics.add(f"bandpower_ratio_{band}")

    for key in expected_metrics:
        assert key in metrics
        entry = metrics[key]
        gen = entry["generated"]["median"]
        tgt = entry["target"]["q05"]
        assert len(gen) == expected_len
        assert len(tgt) == expected_len

    img = mpimg.imread(plot_path)
    assert img.size > 0
