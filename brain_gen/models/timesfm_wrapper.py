"""TimesFM wrapper with a FlatGPT-style forecast interface."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class TimesFMWrapper:
    """Wrap a TimesFM model and expose a forecast() compatible with eval_runner.

    The wrapper treats each channel as an independent univariate series. It is
    intended for generation-only evaluation (no training losses).
    """

    def __init__(
        self,
        model_id: str,
        *,
        model_cls: str | None = None,
        forecast_config: dict[str, Any] | None = None,
        forecast_kwargs: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Load a pretrained TimesFM model.

        Args:
            model_id: HuggingFace-style identifier (e.g. "google/timesfm-2.5-200m-pytorch").
            model_cls: Name of the TimesFM class to load (defaults to TimesFM_2p5_200M_torch).
            forecast_config: Optional ForecastConfig kwargs for model.compile(...).
            forecast_kwargs: Optional kwargs passed to model.forecast(...).
            device: Optional torch device for the model.
        """
        try:
            import timesfm  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local install
            raise ImportError(
                "timesfm is required to use TimesFMWrapper. Install it first."
            ) from exc

        if not model_id:
            raise ValueError("TimesFMWrapper requires a non-empty model_id.")

        model_cls_name = model_cls or "TimesFM_2p5_200M_torch"
        if not hasattr(timesfm, model_cls_name):
            raise ValueError(
                f"timesfm.{model_cls_name} was not found. Check the model_cls name."
            )

        model_class = getattr(timesfm, model_cls_name)
        self.model = model_class.from_pretrained(str(model_id))
        self.model_id = str(model_id)
        self.forecast_kwargs = dict(forecast_kwargs or {})

        if forecast_config:
            config = timesfm.ForecastConfig(**forecast_config)
            self.model.compile(config)

        if device is not None and hasattr(self.model, "to"):
            self.model = self.model.to(device)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def _coerce_batch(self, inputs: Any) -> np.ndarray:
        """Coerce inputs to a (B, C, T) numpy array."""
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]
        if isinstance(inputs, dict):
            for key in ("inputs", "data", "codes"):
                if key in inputs:
                    inputs = inputs[key]
                    break

        if torch.is_tensor(inputs):
            arr = inputs.detach().cpu().numpy()
        else:
            arr = np.asarray(inputs)

        if arr.ndim == 1:
            arr = arr[None, None, :]
        elif arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim != 3:
            raise ValueError(
                "TimesFMWrapper expects inputs with shape (C, T) or (B, C, T)."
            )

        return arr.astype(np.float32, copy=False)

    def _extract_point_forecast(self, output: Any) -> np.ndarray:
        """Extract the point forecast array from TimesFM outputs."""
        if isinstance(output, (tuple, list)) and output:
            return np.asarray(output[0])
        if isinstance(output, dict):
            for key in ("point_forecast", "point", "forecast"):
                if key in output:
                    return np.asarray(output[key])
        if hasattr(output, "point_forecast"):
            return np.asarray(output.point_forecast)
        return np.asarray(output)

    @torch.inference_mode()
    def forecast(
        self,
        initial_input: Any,
        rollout_steps: int,
        _sample_fn: Any = None,
        **_: Any,
    ) -> torch.Tensor:
        """Forecast a continuation for each channel independently.

        Args:
            initial_input: Context array shaped (C, T) or (B, C, T).
            rollout_steps: Number of steps to forecast.
            _sample_fn: Ignored (TimesFM uses its own deterministic sampling).
        """
        horizon = int(rollout_steps)
        series = self._coerce_batch(initial_input)
        batch, channels, _ = series.shape

        outputs: list[np.ndarray] = []
        for b in range(batch):
            inputs = [series[b, ch] for ch in range(channels)]
            forecast_out = self.model.forecast(
                inputs=inputs, horizon=horizon, **self.forecast_kwargs
            )
            point = self._extract_point_forecast(forecast_out)
            point = np.asarray(point, dtype=np.float32)
            if point.ndim == 1:
                point = point[None, :]
            if point.shape[0] != channels and point.shape[1] == channels:
                point = point.T
            outputs.append(point[:, :horizon])

        stacked = np.stack(outputs, axis=0)
        return torch.from_numpy(stacked)
