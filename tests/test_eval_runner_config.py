import torch
import yaml

from brain_gen.eval.eval_runner import EvaluationRunner
import evals


def test_eval_runner_saves_config_yaml(tmp_path):
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.cfg = {"eval_runner": {"ckpt_path": "dummy.ckpt"}, "save_dir": "logs"}

    runner._save_config(tmp_path)

    saved = yaml.safe_load((tmp_path / "eval_config.yaml").read_text())
    assert saved == runner.cfg


def test_eval_runner_compile_flag_skips_compile(monkeypatch):
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.compile_model = False

    def _boom(_model):
        raise AssertionError("torch.compile should not be called")

    monkeypatch.setattr(torch, "compile", _boom)
    model = torch.nn.Linear(1, 1)
    assert runner._ensure_compiled(model) is model


def test_eval_runner_compile_flag_unwraps_compiled_model(monkeypatch):
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.compile_model = False

    def _boom(_model):
        raise AssertionError("torch.compile should not be called")

    monkeypatch.setattr(torch, "compile", _boom)
    orig = torch.nn.Linear(1, 1)
    compiled = torch.nn.Linear(1, 1)
    compiled._orig_mod = orig
    assert runner._ensure_compiled(compiled) is orig


def test_eval_runner_compile_flag_enables_compile(monkeypatch):
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.compile_model = True
    model = torch.nn.Linear(1, 1)

    def _fake_compile(module):
        return ("compiled", module)

    monkeypatch.setattr(torch, "compile", _fake_compile)
    assert runner._ensure_compiled(model) == ("compiled", model)


def test_eval_runner_output_dir_uses_run_name_without_ckpt(tmp_path):
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.save_dir = tmp_path
    runner.output_dir = None
    runner.run_version = 0
    runner.ckpt_path = None
    runner.eval_epoch = None
    runner.run_name = "osl_meg_gpt"

    out_dir = EvaluationRunner._prepare_output_dir(runner)

    assert out_dir == tmp_path / "logs" / "version_0" / "osl_meg_gpt"


def test_eval_runner_resolves_relative_output_dir(tmp_path):
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.save_dir = tmp_path

    resolved = runner._resolve_output_dir({"output_dir": "custom/outputs"})

    assert resolved == tmp_path / "custom" / "outputs"


def test_eval_runner_output_dir_override(tmp_path):
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.save_dir = tmp_path
    runner.output_dir = tmp_path / "custom_outputs"
    runner.run_version = 3
    runner.ckpt_path = "checkpoint.ckpt"
    runner.eval_epoch = 2
    runner.run_name = "unused"

    out_dir = EvaluationRunner._prepare_output_dir(runner)

    assert out_dir == tmp_path / "custom_outputs"


def test_eval_runner_run_name_timesfm_from_model_id():
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.use_timesfm = True
    runner.timesfm_cfg = {"model_id": "google/timesfm-2.5-200m-pytorch"}

    run_name = runner._resolve_run_name({})

    assert run_name == "timesfm-2.5-200m-pytorch"


def test_eval_runner_metrics_split_defaults_to_val():
    runner = EvaluationRunner.__new__(EvaluationRunner)

    assert runner._resolve_metrics_split({}) == "val"


def test_eval_runner_selects_test_loader_for_metrics():
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.metrics_split = "test"
    runner.val_loader = object()
    runner.test_loader = object()

    assert runner._select_metrics_loader() is runner.test_loader


def test_evals_main_passes_device_override(tmp_path, monkeypatch):
    cfg_path = tmp_path / "eval_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"eval_runner": {"device": "cpu"}}))

    called = {}

    class DummyRunner:
        def __init__(self, cfg, device=None):
            called["cfg"] = cfg
            called["device"] = device

        def run(self):
            called["ran"] = True

    monkeypatch.setattr(evals, "EvaluationRunner", DummyRunner)

    evals.main(["--args", str(cfg_path)])

    assert called["device"] == "cpu"
    assert called.get("ran", False)
