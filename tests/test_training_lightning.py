import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace

from brain_gen.training.lightning import LitModel, LitModelFreerun


class DummyModel(nn.Module):
    def __init__(self, input_dim: int = 4, output_dim: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyResizableModel(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.resize_calls: list[dict] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def resize_context(self, **kwargs) -> None:
        self.resize_calls.append(kwargs)


class DummyTokenCorruptModel(nn.Module):
    def __init__(self, token_corruption_cfg: dict | None = None) -> None:
        super().__init__()
        self.token_corruption_cfg = {"enabled": False, "p_start": 0.0, "p_end": 0.0}
        if token_corruption_cfg is not None:
            self.update_token_corruption_cfg(token_corruption_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def update_token_corruption_cfg(self, cfg: dict | None) -> None:
        if cfg is None:
            return
        parsed = dict(self.token_corruption_cfg)
        parsed.update(cfg)
        parsed["p_start"] = max(0.0, float(parsed.get("p_start", 0.0)))
        parsed["p_end"] = max(0.0, float(parsed.get("p_end", 0.0)))
        parsed["enabled"] = bool(parsed.get("enabled", True)) and (
            parsed["p_start"] > 0.0 or parsed["p_end"] > 0.0
        )
        self.token_corruption_cfg = parsed


class DummyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.metrics: dict[str, object] = {}
        self._impl = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        return self._impl(outputs, targets)


def make_litmodel(trainer_cfg: dict) -> LitModel:
    return LitModel(
        model_class=DummyModel,
        loss_class=DummyLoss,
        model_cfg={"input_dim": 4, "output_dim": 3},
        loss_cfg={},
        trainer_cfg=trainer_cfg,
    )


def test_configure_optimizers_builds_scheduler_without_mutation():
    trainer_cfg = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "lr_scheduler": {
            "class_name": "ExponentialLR",
            "gamma": 0.9,
            "interval": "epoch",
            "frequency": 2,
            "monitor": "val_loss",
        },
    }
    lit = make_litmodel(trainer_cfg)

    opt_cfg = lit.configure_optimizers()
    scheduler = opt_cfg["lr_scheduler"]["scheduler"]

    assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
    # User config should remain intact (class name not popped)
    assert trainer_cfg["lr_scheduler"]["class_name"] == "ExponentialLR"
    assert opt_cfg["lr_scheduler"]["interval"] == "epoch"
    assert opt_cfg["lr_scheduler"]["frequency"] == 2
    assert opt_cfg["lr_scheduler"]["monitor"] == "val_loss"


def test_configure_optimizers_raises_on_unknown_scheduler():
    trainer_cfg = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "lr_scheduler": {"class_name": "NotAScheduler"},
    }
    lit = make_litmodel(trainer_cfg)

    with pytest.raises(ValueError):
        lit.configure_optimizers()


def test_configure_optimizers_adds_warmup_scheduler():
    trainer_cfg = {
        "lr": 0.1,
        "weight_decay": 0.0,
        "lr_warmup": {"steps": 3},
        "lr_scheduler": {"class_name": "CosineAnnealingLR", "T_max": 10},
    }
    lit = make_litmodel(trainer_cfg)

    opt_cfg = lit.configure_optimizers()
    scheduler = opt_cfg["lr_scheduler"]["scheduler"]

    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
    assert opt_cfg["lr_scheduler"]["interval"] == "epoch"

    optimizer = opt_cfg["optimizer"]
    lrs = []
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] < lrs[1]
    assert pytest.approx(trainer_cfg["lr"], rel=1e-6) == lrs[1]
    assert pytest.approx(lrs[1], rel=1e-6) == lrs[2]


def test_configure_optimizers_warmup_only_defaults_to_step_interval():
    trainer_cfg = {"lr": 1e-3, "weight_decay": 0.0, "lr_warmup": 2}
    lit = make_litmodel(trainer_cfg)

    opt_cfg = lit.configure_optimizers()
    scheduler = opt_cfg["lr_scheduler"]["scheduler"]

    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert opt_cfg["lr_scheduler"]["interval"] == "step"


def test_configure_optimizers_no_decay_verbose_lists_params(capsys):
    trainer_cfg = {"lr": 1e-3, "weight_decay": 0.0, "no_decay_verbose": True}
    lit = make_litmodel(trainer_cfg)

    lit.configure_optimizers()
    captured = capsys.readouterr()

    assert "no_decay params:" in captured.out
    assert "model.linear.bias" in captured.out


def test_resume_lr_override_updates_optimizer_and_scheduler():
    trainer_cfg = {
        "lr": 1e-2,
        "weight_decay": 0.0,
        "resume_lr": 1e-4,
        "lr_scheduler": {"class_name": "ExponentialLR", "gamma": 0.9},
    }
    lit = make_litmodel(trainer_cfg)

    opt_cfg = lit.configure_optimizers()
    optimizer = opt_cfg["optimizer"]
    scheduler = opt_cfg["lr_scheduler"]["scheduler"]

    class DummyTrainer:
        def __init__(self, optimizer, scheduler):
            self.optimizers = [optimizer]
            self.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]

    lit.trainer = DummyTrainer(optimizer, scheduler)
    lit._apply_resume_lr_override()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(trainer_cfg["resume_lr"])
    assert scheduler.base_lrs == [trainer_cfg["resume_lr"] for _ in scheduler.base_lrs]
    for group in optimizer.param_groups:
        if "initial_lr" in group:
            assert group["initial_lr"] == pytest.approx(trainer_cfg["resume_lr"])


def test_resume_lr_override_applies_when_optimizers_available():
    trainer_cfg = {
        "lr": 1e-2,
        "weight_decay": 0.0,
        "resume_lr": 2e-4,
        "lr_scheduler": {"class_name": "ExponentialLR", "gamma": 0.9},
    }
    lit = make_litmodel(trainer_cfg)

    class DummyTrainer:
        def __init__(self, optimizer, scheduler):
            self.optimizers = [optimizer]
            self.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]

    lit.trainer = SimpleNamespace(optimizers=[], lr_scheduler_configs=[])
    lit.on_fit_start()
    assert lit._resume_lr_applied is False

    opt_cfg = lit.configure_optimizers()
    lit.trainer = DummyTrainer(
        opt_cfg["optimizer"], opt_cfg["lr_scheduler"]["scheduler"]
    )
    lit.on_train_start()
    assert lit._resume_lr_applied is True


def test_resume_weight_decay_override_updates_optimizer_groups():
    trainer_cfg = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "resume_weight_decay": 0.2,
    }
    lit = make_litmodel(trainer_cfg)

    opt_cfg = lit.configure_optimizers()
    optimizer = opt_cfg["optimizer"] if isinstance(opt_cfg, dict) else opt_cfg

    class DummyTrainer:
        def __init__(self, optimizer):
            self.optimizers = [optimizer]

    lit.trainer = DummyTrainer(optimizer)
    lit._apply_resume_weight_decay_override()

    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        trainer_cfg["resume_weight_decay"]
    )
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_resume_context_override_applies_on_fit_start():
    trainer_cfg = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "resume_context": {
            "input_shape": [8, 1, 1],
            "rope_theta": 2.0e5,
            "max_position_embeddings": 64,
        },
    }
    lit = LitModel(
        model_class=DummyResizableModel,
        loss_class=DummyLoss,
        model_cfg={},
        loss_cfg={},
        trainer_cfg=trainer_cfg,
    )

    lit.trainer = SimpleNamespace(
        optimizers=[],
        lr_scheduler_configs=[],
        ckpt_path="dummy.ckpt",
    )
    lit.on_fit_start()

    assert len(lit.model.resize_calls) == 1
    call = lit.model.resize_calls[0]
    assert call["input_shape"] == (8, 1, 1)
    assert call["rope_theta"] == pytest.approx(2.0e5)
    assert call["max_position_embeddings"] == 64


def test_resume_token_corruption_override_uses_model_cfg():
    trainer_cfg = {"lr": 1e-3, "weight_decay": 0.0}
    model_cfg = {"token_corruption_cfg": {"enabled": True, "p_end": 0.25}}
    lit = LitModel(
        model_class=DummyTokenCorruptModel,
        loss_class=DummyLoss,
        model_cfg=model_cfg,
        loss_cfg={},
        trainer_cfg=trainer_cfg,
    )
    lit.model.token_corruption_cfg = {"enabled": False, "p_start": 0.0, "p_end": 0.0}
    lit.trainer = SimpleNamespace(
        optimizers=[],
        lr_scheduler_configs=[],
        ckpt_path="dummy.ckpt",
    )

    lit.on_fit_start()

    assert lit.model.token_corruption_cfg["enabled"] is True
    assert lit.model.token_corruption_cfg["p_end"] == pytest.approx(0.25)


def test_on_load_checkpoint_overrides_optimizer_state():
    trainer_cfg = {"lr": 1e-3, "weight_decay": 0.0, "resume_lr": 5e-5}
    lit = make_litmodel(trainer_cfg)
    checkpoint = {
        "optimizer_states": [{"param_groups": [{"lr": 1e-3, "initial_lr": 1e-3}]}],
        "lr_schedulers": [{"base_lrs": [1e-3], "last_epoch": 4}],
    }

    lit.on_load_checkpoint(checkpoint)

    assert checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(
        trainer_cfg["resume_lr"]
    )
    assert checkpoint["optimizer_states"][0]["param_groups"][0][
        "initial_lr"
    ] == pytest.approx(trainer_cfg["resume_lr"])
    assert checkpoint["lr_schedulers"][0]["base_lrs"] == [trainer_cfg["resume_lr"]]


def test_on_load_checkpoint_overrides_weight_decay():
    trainer_cfg = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "resume_weight_decay": 0.2,
    }
    lit = make_litmodel(trainer_cfg)
    checkpoint = {
        "optimizer_states": [
            {"param_groups": [{"weight_decay": 0.01}, {"weight_decay": 0.0}]}
        ]
    }

    lit.on_load_checkpoint(checkpoint)

    assert checkpoint["optimizer_states"][0]["param_groups"][0][
        "weight_decay"
    ] == pytest.approx(trainer_cfg["resume_weight_decay"])
    assert checkpoint["optimizer_states"][0]["param_groups"][1]["weight_decay"] == 0.0


def test_on_load_checkpoint_unwraps_compiled_checkpoint_when_compile_false():
    trainer_cfg = {"lr": 1e-3, "weight_decay": 0.0, "compile": False}
    lit = make_litmodel(trainer_cfg)
    orig_model = lit.model

    class DummyCompiled(nn.Module):
        def __init__(self, orig: nn.Module) -> None:
            super().__init__()
            self._orig_mod = orig

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._orig_mod(x)

    lit.model = DummyCompiled(orig_model)
    checkpoint = {
        "state_dict": {
            f"model._orig_mod.{key}": value
            for key, value in orig_model.state_dict().items()
        }
    }

    lit.on_load_checkpoint(checkpoint)

    assert lit.model is orig_model
    expected_keys = {f"model.{key}" for key in orig_model.state_dict().keys()}
    assert set(checkpoint["state_dict"].keys()) == expected_keys


def test_compute_grad_norm_matches_expected():
    trainer_cfg = {"lr": 1e-3, "weight_decay": 0.0}
    lit = make_litmodel(trainer_cfg)
    opt_cfg = lit.configure_optimizers()
    optimizer = opt_cfg["optimizer"] if isinstance(opt_cfg, dict) else opt_cfg

    for param in lit.parameters():
        param.grad = torch.ones_like(param)

    expected = 0.0
    for param in lit.parameters():
        expected += param.grad.detach().pow(2).sum().item()
    expected = expected**0.5

    grad_norm = lit._compute_grad_norm(optimizer)
    assert grad_norm == pytest.approx(expected)


def test_freerun_config_validation_and_normalisation():
    trainer_cfg = {"lr": 1e-3, "weight_decay": 0.0}
    lit = LitModelFreerun(
        model_class=DummyModel,
        loss_class=DummyLoss,
        model_cfg={"input_dim": 4, "output_dim": 3},
        loss_cfg={},
        trainer_cfg=trainer_cfg,
        free_run_cfg=None,
    )

    cfg = lit._prepare_free_run_cfg(
        {
            "enabled": True,
            "warmup_range": [2, 3],
            "rollout_range": 4,
            "sample_strategy": "sample",
            "temperature": 0.5,
            "log_lengths": True,
        }
    )
    assert cfg["enabled"] is True
    assert cfg["warmup_range"] == (2, 3)
    assert cfg["rollout_range"] == (4, 4)
    assert cfg["sample_strategy"] == "sample"
    assert cfg["temperature"] == 0.5
    assert cfg["log_lengths"] is True

    with pytest.raises(ValueError):
        lit._prepare_free_run_cfg(
            {"enabled": True, "warmup_range": 0, "rollout_range": 1}
        )
    with pytest.raises(ValueError):
        lit._prepare_free_run_cfg(
            {
                "enabled": True,
                "warmup_range": 1,
                "rollout_range": 1,
                "sample_strategy": "greedy",
            }
        )
    with pytest.raises(ValueError):
        lit._prepare_free_run_cfg(
            {"enabled": True, "warmup_range": 1, "rollout_range": 1, "temperature": 0}
        )


def test_test_step_collects_predictions_and_targets():
    trainer_cfg = {"lr": 1e-3, "weight_decay": 0.0}
    lit = make_litmodel(trainer_cfg)
    lit.eval()

    batch = (torch.zeros(2, 4), torch.tensor([0, 1]))
    lit.test_step(batch, batch_idx=0)

    assert len(lit.test_predictions) == 1
    assert len(lit.test_targets) == 1
    assert lit.test_predictions[0].shape[0] == 2
    assert torch.equal(lit.test_targets[0], torch.tensor([0, 1]))
