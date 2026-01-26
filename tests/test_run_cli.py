import yaml
import pytest

import run


def _write_cfg(tmp_path, extra: dict | None = None):
    cfg = {"save_dir": str(tmp_path / "runs")}
    if extra:
        cfg.update(extra)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_run_main_dispatches_train(monkeypatch, tmp_path):
    called = {}

    class DummyExperiment:
        def __init__(self, cfg):
            called["cfg"] = cfg

        def train(self):
            called["mode"] = "train"

    monkeypatch.setattr(run, "ExperimentDL", DummyExperiment)
    cfg_path = _write_cfg(tmp_path)

    run.main(["--args", str(cfg_path), "--mode", "train"])

    assert called["mode"] == "train"
    assert called["cfg"]["save_dir"].endswith("runs")


def test_run_main_dispatches_test(monkeypatch, tmp_path):
    called = {}

    class DummyExperiment:
        def __init__(self, cfg):
            called["cfg"] = cfg

        def test(self):
            called["mode"] = "test"

    monkeypatch.setattr(run, "ExperimentDL", DummyExperiment)
    cfg_path = _write_cfg(tmp_path)

    run.main(["--args", str(cfg_path), "--mode", "test"])

    assert called["mode"] == "test"


def test_run_main_rejects_invalid_mode(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    with pytest.raises(SystemExit):
        run.main(["--args", str(cfg_path), "--mode", "unknown"])
