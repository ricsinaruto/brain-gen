from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import inspect
import torch

from ..models.flatgpt import FlatGPTEmbedsRVQ
from ..utils.eval import sample as sample_logits

# Backwards-compatible alias used by tests/monkeypatching.
sample = sample_logits
from ..utils.quantizers import mulaw_inv_torch
from .session_sampler import SessionSample


@dataclass(frozen=True)
class GenerationResult:
    """Paired generated/target tensors returned by the rollout generator."""

    generated: np.ndarray
    target: np.ndarray
    context_steps: int
    total_steps: int
    metadata: list[dict[str, Any]]


def _generation_input_key(inputs: Any) -> tuple[Any, ...] | None:
    """Return a hashable signature for batching compatible inputs."""
    if inputs is None:
        return None
    if torch.is_tensor(inputs):
        return ("tensor", tuple(inputs.shape))
    if isinstance(inputs, (tuple, list)):
        child_keys: list[Any] = []
        for item in inputs:
            key = _generation_input_key(item)
            if key is None:
                return None
            child_keys.append(key)
        return (type(inputs).__name__, tuple(child_keys))
    if isinstance(inputs, dict):
        items: list[tuple[str, Any]] = []
        for key in sorted(inputs.keys()):
            child_key = _generation_input_key(inputs[key])
            if child_key is None:
                return None
            items.append((str(key), child_key))
        return ("dict", tuple(items))
    return None


def _stack_generation_inputs(inputs_list: list[Any]) -> Any | None:
    """Stack a list of inputs into a batch when shapes are compatible."""
    if not inputs_list:
        return None

    first = inputs_list[0]
    if torch.is_tensor(first):
        try:
            if first.ndim >= 2:
                return torch.cat(inputs_list, dim=0)
            return None
        except Exception:
            return None

    if isinstance(first, (tuple, list)):
        if not all(isinstance(item, type(first)) for item in inputs_list):
            return None
        length = len(first)
        if not all(len(item) == length for item in inputs_list):
            return None
        stacked: list[Any] = []
        for idx in range(length):
            child = _stack_generation_inputs([item[idx] for item in inputs_list])
            if child is None:
                return None
            stacked.append(child)
        return type(first)(stacked)

    if isinstance(first, dict):
        keys = set(first.keys())
        if not all(
            isinstance(item, dict) and set(item.keys()) == keys for item in inputs_list
        ):
            return None
        stacked_dict: dict[str, Any] = {}
        for key in first.keys():
            child = _stack_generation_inputs([item[key] for item in inputs_list])
            if child is None:
                return None
            stacked_dict[key] = child
        return stacked_dict

    return None


def _group_generation_samples(
    samples: list[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    """Group prepared samples into batches with matching input signatures."""
    if batch_size <= 1:
        return [[s] for s in samples]

    batches: list[list[dict[str, Any]]] = []
    pending: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    key_order: list[tuple[Any, ...]] = []

    for s in samples:
        input_key = _generation_input_key(s.get("inputs"))
        if input_key is None:
            batches.append([s])
            continue

        horizon = s.get("rollout_horizon")
        batch_key = (horizon, input_key)
        if batch_key not in pending:
            pending[batch_key] = []
            key_order.append(batch_key)

        pending[batch_key].append(s)
        if len(pending[batch_key]) >= batch_size:
            batches.append(pending[batch_key])
            pending[batch_key] = []

    for batch_key in key_order:
        leftover = pending.get(batch_key)
        if leftover:
            batches.append(leftover)

    return batches


def load_paired_rollouts(path: Path) -> GenerationResult:
    """Load cached paired rollouts from disk."""
    payload = np.load(path, allow_pickle=True).item()
    return GenerationResult(
        generated=np.asarray(payload["generated"]),
        target=np.asarray(payload["target"]),
        context_steps=int(payload["context_steps"]),
        total_steps=int(payload["total_steps"]),
        metadata=list(payload.get("metadata", [])),
    )


def save_paired_rollouts(path: Path, result: GenerationResult) -> None:
    """Persist paired rollouts to disk as a numpy payload."""
    payload = {
        "generated": result.generated,
        "target": result.target,
        "context_steps": int(result.context_steps),
        "total_steps": int(result.total_steps),
        "metadata": result.metadata,
    }
    np.save(path, payload)


class RolloutGenerator:
    """Generate session rollouts given sampled contexts."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        device: torch.device,
        sfreq: float | None,
    ) -> None:
        """Initialize the rollout generator."""
        self.cfg = cfg
        self.device = device
        self.sfreq = sfreq
        self.mu = cfg.get("mu", None)
        self.max_plot_seconds = self._resolve_max_plot_seconds()

    def _resolve_seed(self, params: dict[str, Any]) -> int | None:
        """Resolve the seed used for sampling and plotting."""
        seed = params.get("seed")
        if seed is None:
            seed = self.cfg.get("seed")
        if seed is None:
            return None
        return int(seed)

    def _sampling_params(self) -> dict[str, Any]:
        """Return sampling params dict for seeding and strategy choices."""
        params_source = self.cfg.get("sampling")
        if isinstance(params_source, dict):
            return dict(params_source)
        return dict(self.cfg)

    def _resolve_max_plot_seconds(self) -> float | None:
        """Resolve the optional cap on plotted rollout length."""
        value = self.cfg.get("max_plot_seconds", self.cfg.get("max_plot_length_s"))
        if value is None:
            return None
        return float(value)

    def generate(
        self,
        model: torch.nn.Module,
        samples: list[SessionSample],
        *,
        out_dir: Path,
        plotting: Any,
    ) -> GenerationResult | None:
        """Generate rollouts for sampled sessions and return paired tensors."""
        if not samples:
            return None

        forecast_fn = getattr(model, "forecast", None)
        if not callable(forecast_fn):
            print("[rollout_generator] Model has no forecast() method; skipping.")
            return None

        rollouts_per_context = int(
            self.cfg.get(
                "rollouts_per_context", self.cfg.get("num_rollouts_per_context", 1)
            )
        )
        rollouts_per_context = max(1, rollouts_per_context)
        batch_size = int(self.cfg.get("rollout_batch_size", 1))
        batch_size = max(1, batch_size)

        params = self._sampling_params()
        seed = self._resolve_seed(params)
        if seed is not None:
            torch.manual_seed(seed)
        plot_rng = np.random.default_rng(seed)

        rvq_levels = self._infer_rvq_levels(model)
        default_curriculum = False

        def make_sample_fn():
            return self._build_sample_fn(
                params, rvq_levels=rvq_levels, default_curriculum=default_curriculum
            )

        prepared: list[dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            context = sample.data[:, : sample.context_steps]
            inputs = self._build_inputs(sample, context)
            rollout_horizon = int(sample.total_steps - sample.context_steps)
            target_full = sample.data[:, : sample.total_steps]
            for rollout_idx in range(rollouts_per_context):
                run_idx = len(prepared)
                prepared.append(
                    {
                        "run_idx": run_idx,
                        "inputs": inputs,
                        "context_tensor": torch.from_numpy(context),
                        "context_steps": sample.context_steps,
                        "rollout_horizon": rollout_horizon,
                        "target_full": torch.from_numpy(target_full),
                        "metadata": {
                            "session_index": idx,
                            "rollout_index": rollout_idx,
                            "dataset_key": sample.dataset_key,
                            "session": sample.session,
                            "task_type": sample.task_type,
                        },
                    }
                )

        if not prepared:
            return None

        batch_queue = _group_generation_samples(prepared, batch_size)
        generated_runs: list[np.ndarray] = []
        target_runs: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []

        for batch in batch_queue:
            batch_inputs = _stack_generation_inputs(
                [sample["inputs"] for sample in batch]
            )
            if batch_inputs is None:
                for sample in batch:
                    self._generate_single(
                        sample,
                        forecast_fn,
                        make_sample_fn(),
                        generated_runs,
                        target_runs,
                        metadata,
                        plotting,
                        out_dir,
                        plot_rng,
                    )
                continue

            batch_horizon = int(batch[0]["rollout_horizon"])
            generated = self._run_forecast(
                forecast_fn,
                batch_inputs,
                batch_horizon,
                make_sample_fn(),
                self._forecast_kwargs(),
            )
            gen_tensor = self._extract_tensor(generated)
            if gen_tensor is None:
                print("[rollout_generator] Unable to extract tensor from forecast.")
                continue
            if gen_tensor.dim() == 1:
                gen_tensor = gen_tensor.unsqueeze(0)
            if gen_tensor.shape[0] != len(batch):
                print(
                    "[rollout_generator] Generated batch size mismatch "
                    f"(expected {len(batch)}, got {gen_tensor.shape[0]})."
                )
                continue

            for sample_idx, sample in enumerate(batch):
                self._record_generation(
                    sample,
                    gen_tensor[sample_idx],
                    generated_runs,
                    target_runs,
                    metadata,
                    plotting,
                    out_dir,
                    plot_rng,
                )

        if not generated_runs:
            return None

        context_steps = int(samples[0].context_steps)
        total_steps = int(samples[0].total_steps)
        if context_steps > 0:
            gen_no_context = [run[:, context_steps:] for run in generated_runs]
            tgt_no_context = [run[:, context_steps:] for run in target_runs]
        else:
            gen_no_context = generated_runs
            tgt_no_context = target_runs
        plotting.plot_psd_cov_pair(
            gen_no_context, tgt_no_context, out_dir, prefix="gen_vs_target"
        )
        if generated_runs and target_runs:
            plotting.plot_stacked_timeseries_pair(
                np.stack(generated_runs, axis=0),
                np.stack(target_runs, axis=0),
                "gen_vs_target",
                out_dir,
                context_len=context_steps,
                max_plot_seconds=self.max_plot_seconds,
            )
        return GenerationResult(
            generated=np.stack(generated_runs, axis=0),
            target=np.stack(target_runs, axis=0),
            context_steps=context_steps,
            total_steps=total_steps,
            metadata=metadata,
        )

    def plot_cached_rollouts(
        self,
        result: GenerationResult,
        *,
        out_dir: Path,
        plotting: Any,
    ) -> None:
        """Plot cached rollout pairs for quick visual inspection."""
        generated = np.asarray(result.generated)
        target = np.asarray(result.target)
        if generated.ndim != 3 or target.ndim != 3:
            return

        max_runs = min(generated.shape[0], target.shape[0])
        if max_runs <= 0:
            return

        plotting.plot_stacked_timeseries_pair(
            generated[:max_runs],
            target[:max_runs],
            "gen_vs_target",
            out_dir,
            context_len=result.context_steps,
            max_plot_seconds=self.max_plot_seconds,
        )

    def _build_inputs(self, sample: SessionSample, context: np.ndarray) -> Any:
        """Build model inputs from a session context."""
        inputs = torch.from_numpy(context)
        if np.issubdtype(context.dtype, np.integer):
            inputs = inputs.long()
        else:
            inputs = inputs.float()

        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)

        pos = getattr(sample, "pos", None)
        sensor_type = getattr(sample, "sensor_type", None)
        if pos is not None and sensor_type is not None:
            pos_tensor = torch.as_tensor(pos).float()
            sensor_tensor = torch.as_tensor(sensor_type).long()
            if pos_tensor.ndim == 2:
                pos_tensor = pos_tensor.unsqueeze(0)
            if sensor_tensor.ndim == 1:
                sensor_tensor = sensor_tensor.unsqueeze(0)
            inputs = (inputs, pos_tensor, sensor_tensor)

        if sample.condition is None:
            return inputs

        cond = sample.condition
        if cond.ndim == 1:
            cond = cond[: sample.context_steps]
        else:
            cond = cond[..., : sample.context_steps]
        cond_tensor = torch.from_numpy(cond).long()
        if cond_tensor.ndim == 1:
            cond_tensor = cond_tensor.unsqueeze(0)
        elif cond_tensor.ndim == 2:
            cond_tensor = cond_tensor.unsqueeze(0)
        return (inputs, cond_tensor)

    def _generate_single(
        self,
        sample: dict[str, Any],
        forecast_fn: Callable[..., Any],
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        gen_runs: list[np.ndarray],
        tgt_runs: list[np.ndarray],
        metadata: list[dict[str, Any]],
        plotting: Any,
        out_dir: Path,
        plot_rng: np.random.Generator,
    ) -> None:
        """Generate a single rollout and record outputs."""
        generated = self._run_forecast(
            forecast_fn,
            sample["inputs"],
            int(sample["rollout_horizon"]),
            sample_fn,
            self._forecast_kwargs(),
        )
        gen_tensor = self._extract_tensor(generated)
        if gen_tensor is None:
            return
        self._record_generation(
            sample,
            gen_tensor,
            gen_runs,
            tgt_runs,
            metadata,
            plotting,
            out_dir,
            plot_rng,
        )

    def _record_generation(
        self,
        sample: dict[str, Any],
        sample_gen: torch.Tensor,
        gen_runs: list[np.ndarray],
        tgt_runs: list[np.ndarray],
        metadata: list[dict[str, Any]],
        plotting: Any,
        out_dir: Path,
        plot_rng: np.random.Generator,
    ) -> None:
        """Normalize, record, and plot one rollout pair."""
        run_idx = int(sample["run_idx"])
        rollout_horizon = int(sample["rollout_horizon"])
        context_steps = int(sample["context_steps"])

        gen_tensor = self._normalise_timeseries(sample_gen)
        if gen_tensor is None:
            return
        effective_steps = min(rollout_horizon, gen_tensor.shape[1])
        if effective_steps <= 0:
            return
        gen_tensor = gen_tensor[:, :effective_steps]

        context_tensor = sample["context_tensor"]
        if self.mu is not None:
            gen_tensor = mulaw_inv_torch(gen_tensor, self.mu)
            context_tensor = mulaw_inv_torch(context_tensor, self.mu)

        gen_arr = gen_tensor.cpu().to(torch.float32).numpy()
        context_arr = context_tensor.cpu().to(torch.float32).numpy()

        full_generated = np.concatenate([context_arr, gen_arr], axis=1)
        gen_runs.append(full_generated)

        target_full = sample.get("target_full")
        if target_full is None:
            target_full = np.concatenate([context_arr, gen_arr], axis=1)
        if torch.is_tensor(target_full):
            target_tensor = target_full
        else:
            target_tensor = torch.as_tensor(target_full)
        expected_steps = context_steps + effective_steps
        if target_tensor.shape[-1] >= expected_steps:
            target_tensor = target_tensor[:, :expected_steps]
        if self.mu is not None:
            target_tensor = mulaw_inv_torch(target_tensor, self.mu)
        tgt_runs.append(target_tensor.cpu().to(torch.float32).numpy())
        metadata.append(sample.get("metadata", {}))

        if run_idx >= 10:
            return

        max_channels = min(full_generated.shape[0], tgt_runs[-1].shape[0])
        if max_channels <= 0:
            return

        num_plot_channels = min(10, max_channels)
        channel_indices = plot_rng.choice(
            max_channels, size=num_plot_channels, replace=False
        )
        prefix = f"gen_vs_target_run{run_idx}"
        plotting.plot_timeseries_pair(
            full_generated,
            tgt_runs[-1],
            prefix,
            out_dir,
            channel_indices,
            context_len=context_steps,
        )
        plotting.plot_stft_pair(
            full_generated,
            tgt_runs[-1],
            prefix,
            out_dir,
            channel_indices,
            context_len=context_steps,
        )

    def _infer_rvq_levels(self, model: torch.nn.Module | None) -> int | None:
        """Infer the number of RVQ levels for FlatGPTEmbedsRVQ models."""
        if model is None:
            return None
        model_ref = getattr(model, "_orig_mod", model)
        if not isinstance(model_ref, FlatGPTEmbedsRVQ):
            return None
        reduced_shape = getattr(model_ref, "reduced_shape", None)
        if not reduced_shape or len(reduced_shape) < 3:
            return None
        levels = int(reduced_shape[2])
        return levels if levels > 0 else None

    def _build_sample_fn(
        self,
        params: dict[str, Any],
        *,
        rvq_levels: int | None = None,
        default_curriculum: bool = False,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build a token sampling function from generation params."""
        strategy = str(params.get("strategy", params.get("sampling", "top_p"))).lower()
        sample_args = {
            "strategy": strategy,
            "top_k": int(params.get("top_k", 0)),
            "top_p": float(params.get("top_p", 0.8)),
        }

        temp_levels = _resolve_temperature_curriculum(
            params, rvq_levels, default_curriculum
        )
        if temp_levels is None:
            sample_args["temperature"] = float(params.get("temperature", 1.0))

            def _sample_fn(logits: torch.Tensor) -> torch.Tensor:
                return sample_logits(logits, **sample_args)

            return _sample_fn

        token_offset = 0
        rvq_levels = int(rvq_levels) if rvq_levels is not None else 1

        def _sample_fn(logits: torch.Tensor) -> torch.Tensor:
            nonlocal token_offset
            if logits.dim() == 2:
                token_count = 1
            else:
                token_count = int(logits.shape[1])

            level_ids = (
                torch.arange(token_count, device=logits.device) + token_offset
            ) % rvq_levels
            temperature = temp_levels.to(device=logits.device, dtype=logits.dtype)[
                level_ids
            ]
            token_offset += token_count
            if logits.dim() > 2:
                temperature = temperature.view(1, -1, 1)

            return sample_logits(logits, temperature=temperature, **sample_args)

        return _sample_fn

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        """Move nested inputs to the configured device."""
        if torch.is_tensor(inputs):
            return inputs.to(self.device)
        if isinstance(inputs, (tuple, list)):
            return type(inputs)(self._move_inputs_to_device(x) for x in inputs)
        if isinstance(inputs, dict):
            return {
                key: self._move_inputs_to_device(val) for key, val in inputs.items()
            }
        return inputs

    def _run_forecast(
        self,
        forecast_fn: Callable[..., Any],
        inputs: Any,
        horizon: int,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        forecast_kwargs: dict[str, Any],
    ) -> Any:
        """Run a model forecast with autocast when on CUDA."""
        inputs = self._move_inputs_to_device(inputs)
        if "debug_timing" in forecast_kwargs:
            try:
                params = inspect.signature(forecast_fn).parameters
                accepts_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                if not accepts_kwargs and "debug_timing" not in params:
                    forecast_kwargs = dict(forecast_kwargs)
                    forecast_kwargs.pop("debug_timing", None)
            except (TypeError, ValueError):
                pass
        if self.device.type == "cuda":
            print(f"Running forecast on {self.device.type}.")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = forecast_fn(inputs, horizon, sample_fn, **forecast_kwargs)
        else:
            generated = forecast_fn(inputs, horizon, sample_fn, **forecast_kwargs)
        generated = self._move_outputs_to_cpu(generated)
        return self._to_float32(generated)

    def _to_float32(self, value: Any) -> Any:
        """Cast tensors to float32 while leaving non-tensors untouched."""
        if torch.is_tensor(value):
            return value.to(torch.float32)
        return value

    def _move_outputs_to_cpu(self, value: Any) -> Any:
        """Move forecast outputs to CPU to release device memory early."""
        if torch.is_tensor(value):
            return value.detach().to("cpu")
        if isinstance(value, (tuple, list)):
            return type(value)(self._move_outputs_to_cpu(item) for item in value)
        if isinstance(value, dict):
            return {key: self._move_outputs_to_cpu(val) for key, val in value.items()}
        return value

    def _extract_tensor(self, output: Any) -> torch.Tensor | None:
        """Extract the first tensor-like output from a forecast result."""
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            for key in ("logits", "output", "outputs", "pred", "preds"):
                val = output.get(key)
                if torch.is_tensor(val):
                    return val
            for val in output.values():
                if torch.is_tensor(val):
                    return val
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        return None

    def _normalise_timeseries(self, tensor: torch.Tensor | Any) -> torch.Tensor | None:
        """Convert arbitrary tensors to a (C, T) float tensor without a batch dim."""
        if tensor is None:
            return None
        try:
            arr = (
                tensor.detach() if torch.is_tensor(tensor) else torch.as_tensor(tensor)
            )
        except Exception:
            return None
        if arr.ndim >= 3:
            arr = arr[0]
        if arr.ndim == 0:
            return None
        if arr.ndim == 1:
            arr = arr.unsqueeze(0)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[0] > arr.shape[1]:
            arr = arr.transpose(0, 1)
        return arr.to(torch.float32)

    def _forecast_kwargs(self) -> dict[str, Any]:
        """Return kwargs passed into the model forecast function."""
        kwargs = {
            "use_cache": True,
            "sliding_window_overlap": self.cfg.get("kv_overlap", 0.5),
            "max_context_tokens": self.cfg.get("max_context_tokens", -1),
        }
        if self.cfg.get("debug_timing", False):
            kwargs["debug_timing"] = True
        return kwargs


def _resolve_temperature_curriculum(
    params: dict[str, Any],
    rvq_levels: int | None,
    default_enabled: bool,
) -> torch.Tensor | None:
    """Resolve per-level temperatures for RVQ curricula."""
    if rvq_levels is None or rvq_levels <= 1:
        return None

    curriculum_cfg = params.get("temperature_curriculum", None)
    explicit_levels = None
    if isinstance(curriculum_cfg, (list, tuple)):
        explicit_levels = curriculum_cfg
        enabled = True
    else:
        explicit_levels = params.get("temperature_levels") or params.get(
            "temperature_per_level"
        )
        if curriculum_cfg is None:
            enabled = bool(explicit_levels) or default_enabled
        else:
            enabled = bool(curriculum_cfg)

    if not enabled:
        return None

    if explicit_levels:
        temps = [float(val) for val in explicit_levels]
        if not temps:
            return None
    else:
        base = float(params.get("temperature", 1.0))
        decay = float(params.get("temperature_decay", 0.85))
        temps = [base * (decay**idx) for idx in range(rvq_levels)]

    min_temp = params.get("temperature_min", None)
    if min_temp is not None:
        min_temp = float(min_temp)
        temps = [max(temp, min_temp) for temp in temps]

    if len(temps) < rvq_levels:
        temps = temps + [temps[-1]] * (rvq_levels - len(temps))
    elif len(temps) > rvq_levels:
        temps = temps[:rvq_levels]

    return torch.tensor(temps, dtype=torch.float32)
