from types import SimpleNamespace
import yaml
import torch
import torch.nn as nn
import pytest
from torch.utils.data import Dataset, IterableDataset

from brain_gen.training import train as train_module


class ToyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        return torch.zeros(4), torch.tensor(idx % 2, dtype=torch.long)


class ToyIterable(IterableDataset):
    def __iter__(self):
        for idx in range(2):
            yield torch.zeros(4), torch.tensor(idx % 2, dtype=torch.long)


class DummyModel(nn.Module):
    def __init__(self, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(4, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


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


class RecordingLoader:
    instances: list["RecordingLoader"] = []

    def __init__(self, dataset, shuffle=None, **kwargs):
        self.dataset = dataset
        self.shuffle = shuffle
        self.kwargs = kwargs
        RecordingLoader.instances.append(self)

    def __iter__(self):
        yield from self.dataset

    def __len__(self):
        return len(self.dataset)


class DummyTrainer:
    last_instance: "DummyTrainer | None" = None

    def __init__(self, **kwargs) -> None:
        DummyTrainer.last_instance = self
        self.kwargs = kwargs
        self.callbacks = kwargs.get("callbacks", [])
        self.logger = kwargs.get("logger")
        self.loggers = []
        self.is_global_zero = True
        self.global_step = 0
        self.current_epoch = 0
        self.fit_called = False
        self.test_called = False
        self.validate_called = False
        self.calls = []

    def fit(
        self, model, train_loader=None, val_loader=None, datamodule=None, ckpt_path=None
    ):
        self.fit_called = True
        self.fit_args = (train_loader, val_loader, datamodule, ckpt_path)
        self.calls.append(("fit", ckpt_path, datamodule))

    def test(self, model, test_loader, ckpt_path=None):
        self.test_called = True
        # mimic PL test loop enough to trigger test_step
        for batch_idx, batch in enumerate(test_loader):
            model.test_step(batch, batch_idx)

    def validate(self, model=None, datamodule=None, ckpt_path=None):
        self.validate_called = True
        self.validate_args = (datamodule, ckpt_path)
        self.calls.append(("validate", ckpt_path, datamodule))


def _make_cfg(tmp_path, model_cfg_path):
    return {
        "trainer": {"max_epochs": 1, "callbacks": []},
        "dataloader": {"batch_size": 2, "num_workers": 0},
        "resume_from": None,
        "dataset_name": "omega",
        "save_dir": str(tmp_path / "runs"),
        "model_config": str(model_cfg_path),
        "model_name": "DummyModel",
        "loss_name": "DummyLoss",
        "lightning": {"lr": 1e-3, "weight_decay": 0.0},
        "datasplitter": {
            "dataset_root": "ignored",
            "example_seconds": 0.1,
            "overlap_seconds": 0.0,
        },
        "dataloader_class": "RecordingLoader",
    }


def _register_patches(monkeypatch, datasets):
    monkeypatch.setattr(train_module, "split_datasets", lambda **_: datasets)
    monkeypatch.setattr(train_module, "get_model_class", lambda name: DummyModel)
    monkeypatch.setattr(train_module, "get_loss_class", lambda name: DummyLoss)
    monkeypatch.setattr(train_module.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(train_module, "RecordingLoader", RecordingLoader, raising=False)


def test_build_early_stopping_normalises_inputs():
    exp = object.__new__(train_module.ExperimentDL)

    cb_bool = exp._build_early_stopping(True)
    cb_int = exp._build_early_stopping(3)
    cb_dict = exp._build_early_stopping({"patience": 2, "min_delta": 0.1})

    assert cb_bool.patience == 10
    assert cb_int.patience == 3
    assert cb_dict.patience == 2
    assert abs(cb_dict.min_delta) == 0.1

    with pytest.raises(ValueError):
        exp._build_early_stopping({"patience": -1})
    with pytest.raises(ValueError):
        exp._build_early_stopping("bad")  # type: ignore[arg-type]


def test_experimentdl_test_runs_test_step(monkeypatch, tmp_path):
    RecordingLoader.instances.clear()
    model_cfg_path = tmp_path / "model.yaml"
    model_cfg_path.write_text(yaml.safe_dump({"output_dim": 2}))

    datasets = SimpleNamespace(train=ToyDataset(), val=ToyDataset(), test=ToyDataset())
    _register_patches(monkeypatch, datasets)

    cfg = _make_cfg(tmp_path, model_cfg_path)
    exp = train_module.ExperimentDL(cfg)

    exp.test()

    assert DummyTrainer.last_instance is not None
    assert DummyTrainer.last_instance.test_called is True
    # test_step should have stored predictions and targets
    assert exp.lit_model.test_predictions
    assert exp.lit_model.test_targets


def test_experimentdl_validates_before_resuming(monkeypatch, tmp_path):
    RecordingLoader.instances.clear()
    model_cfg_path = tmp_path / "model.yaml"
    model_cfg_path.write_text(yaml.safe_dump({"output_dim": 2}))

    datasets = SimpleNamespace(train=ToyDataset(), val=ToyDataset(), test=ToyDataset())
    _register_patches(monkeypatch, datasets)

    cfg = _make_cfg(tmp_path, model_cfg_path)
    cfg["resume_from"] = str(tmp_path / "resume.ckpt")
    cfg["trainer"]["validate_before_resume"] = True

    exp = train_module.ExperimentDL(cfg)
    exp.train()

    trainer = DummyTrainer.last_instance
    assert trainer is not None
    assert trainer.validate_called is True
    assert trainer.fit_called is True
    assert trainer.calls[0][0] == "validate"
    assert trainer.calls[1][0] == "fit"
    assert trainer.calls[0][1] == cfg["resume_from"]
    assert trainer.calls[1][1] == cfg["resume_from"]
