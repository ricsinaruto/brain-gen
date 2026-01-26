import numpy as np

from brain_gen.eval.rollout_metrics import (
    MetricDistances,
    RolloutMetrics,
    RolloutMetricsConfig,
)


def test_bandpower_ratios_falls_back_without_trapezoid(monkeypatch):
    cfg = RolloutMetricsConfig(bands={"alpha": (1.0, 3.0)})
    metrics = RolloutMetrics(cfg)
    freqs = np.array([1.0, 2.0, 2.5], dtype=np.float32)
    psd = np.ones((1, 1, freqs.size), dtype=np.float32)
    trapz = np.trapz

    monkeypatch.delattr(np, "trapezoid", raising=False)
    ratios = metrics._bandpower_ratios(freqs, psd)

    mask = (freqs >= 1.0) & (freqs < 3.0)
    expected_total = trapz(psd, freqs, axis=2)
    expected_band = trapz(psd[:, :, mask], freqs[mask], axis=2)
    expected_ratio = expected_band / np.clip(expected_total, 1e-12, None)

    assert ratios.shape == (1, 1, 1)
    assert np.allclose(ratios[:, :, 0], expected_ratio)


def test_band_jsd_falls_back_without_trapezoid(monkeypatch):
    cfg = RolloutMetricsConfig(bands={"alpha": (1.0, 3.0)})
    distances = MetricDistances(cfg)
    freqs = np.array([1.0, 2.0, 2.5], dtype=np.float32)
    psd = np.ones((1, 2, 1, freqs.size), dtype=np.float32)

    monkeypatch.delattr(np, "trapezoid", raising=False)
    jsd = distances._band_jsd(freqs, psd)

    assert jsd.shape == (1,)
    assert np.allclose(jsd, 0.0)
