import sys
import types

import numpy as np
import torch

from brain_gen.models.timesfm_wrapper import TimesFMWrapper


class DummyForecastConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyTimesFMModel:
    def __init__(self):
        self.compiled = None
        self.device = None
        self.eval_called = False
        self.last_inputs = None
        self.last_horizon = None
        self.last_kwargs = None

    def compile(self, config):
        self.compiled = config

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def forecast(self, inputs=None, horizon=None, **kwargs):
        self.last_inputs = inputs
        self.last_horizon = horizon
        self.last_kwargs = kwargs
        series = [np.asarray(x) for x in inputs]
        horizon = int(horizon)
        point = np.stack(
            [
                np.full(horizon, fill_value=float(idx), dtype=np.float32)
                for idx in range(len(series))
            ],
            axis=0,
        )
        quantiles = np.zeros((len(series), horizon, 1), dtype=np.float32)
        return point, quantiles


class DummyTimesFMClass:
    @classmethod
    def from_pretrained(cls, model_id):
        return DummyTimesFMModel()


def _install_dummy_timesfm(monkeypatch):
    module = types.ModuleType("timesfm")
    module.ForecastConfig = DummyForecastConfig
    module.TimesFM_2p5_200M_torch = DummyTimesFMClass
    monkeypatch.setitem(sys.modules, "timesfm", module)


def test_timesfm_wrapper_compiles_and_forecasts(monkeypatch):
    _install_dummy_timesfm(monkeypatch)
    wrapper = TimesFMWrapper(
        model_id="google/timesfm-2.5-200m-pytorch",
        forecast_config={"max_context": 16},
        device="cpu",
    )

    assert wrapper.model.compiled.kwargs["max_context"] == 16
    assert wrapper.model.eval_called is True

    inputs = torch.randn(2, 3, 5)
    output = wrapper.forecast(inputs, 4, None)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 3, 4)
    assert wrapper.model.last_horizon == 4
    assert len(wrapper.model.last_inputs) == 3


def test_timesfm_wrapper_ignores_aux_inputs(monkeypatch):
    _install_dummy_timesfm(monkeypatch)
    wrapper = TimesFMWrapper(
        model_id="google/timesfm-2.5-200m-pytorch",
        forecast_config=None,
    )

    inputs = (torch.randn(1, 2, 6), torch.ones(1))
    output = wrapper.forecast(inputs, 3, None)

    assert output.shape == (1, 2, 3)
