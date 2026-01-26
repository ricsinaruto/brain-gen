import torch

from brain_gen.eval import eval_runner as eval_runner_mod
from brain_gen.eval.eval_runner import EvaluationRunner


def test_eval_runner_retries_on_truncated_checkpoint(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "ckpt.ckpt"
    ckpt_path.write_text("partial")

    calls = {"count": 0}

    class DummyModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    class DummyLit:
        def __init__(self):
            self.model = DummyModel()

    def fake_load_from_checkpoint(*_args, **_kwargs):
        if calls["count"] == 0:
            calls["count"] += 1
            raise RuntimeError(
                "PytorchStreamReader failed reading zip archive: failed finding "
                "central directory"
            )
        calls["count"] += 1
        return DummyLit()

    monkeypatch.setattr(
        eval_runner_mod.LitModel, "load_from_checkpoint", fake_load_from_checkpoint
    )

    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.lit_module_name = None
    runner.postprocessor = None
    runner.device = torch.device("cpu")
    runner.compile_model = False
    runner.checkpoint_wait_timeout_s = 1
    runner.checkpoint_stable_seconds = 0.0
    runner.checkpoint_poll_seconds = 0.01
    runner.checkpoint_load_retries = 1
    runner.checkpoint_retry_wait_s = 0.01

    lit_model = EvaluationRunner._load_model(runner, str(ckpt_path))
    assert isinstance(lit_model, DummyLit)
    assert calls["count"] == 2
