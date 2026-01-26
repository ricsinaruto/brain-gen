from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List
import time

import numpy as np
import torch
import yaml
import logging
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, IterableDataset
from ..dataset import split_datasets
from .generation import (
    RolloutGenerator,
    load_paired_rollouts,
    save_paired_rollouts,
)
from .session_sampler import SessionSampler
from ..training.lightning import LitModel
from .rollout_divergence import RolloutDivergenceAnalysis
from .rollout_sliding_windows import RolloutSlidingWindowAnalysis
from .plotting import EvaluationPlotting
from .token_summary import TokenSummaryPlotter

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Lightweight evaluation loop used by automated checkpoint triggers."""

    def __init__(
        self,
        cfg: dict,
        device: str | None = "cuda",
    ) -> None:
        """Initialize the evaluation runner from a config dict."""
        eval_cfg = cfg.get("eval_runner", {})
        self.lit_module_name = eval_cfg.get("lit_module", None)
        self.compile_model = eval_cfg.get("compile", True)
        self.timesfm_cfg = eval_cfg.get("timesfm")
        self.use_timesfm = bool(self.timesfm_cfg)
        self.ckpt_path = None
        if not self.use_timesfm:
            self.ckpt_path = self._resolve_checkpoint(eval_cfg.get("ckpt_path"))
        self.eval_step = eval_cfg.get("step")
        self.eval_epoch = eval_cfg.get("epoch")
        self.run_version = eval_cfg.get("version")
        self.run_name = self._resolve_run_name(eval_cfg)
        self.max_batches = eval_cfg.get("max_batches", 2)
        self.num_examples = eval_cfg.get("num_examples", 2)
        self.checkpoint_wait_timeout_s = int(
            eval_cfg.get("checkpoint_wait_timeout_s", 300)
        )
        self.checkpoint_stable_seconds = float(
            eval_cfg.get("checkpoint_stable_seconds", 5)
        )
        self.checkpoint_poll_seconds = float(
            eval_cfg.get("checkpoint_poll_seconds", 1.0)
        )
        self.checkpoint_load_retries = int(eval_cfg.get("checkpoint_load_retries", 3))
        self.checkpoint_retry_wait_s = float(
            eval_cfg.get("checkpoint_retry_wait_s", 5.0)
        )
        self.metrics_split = self._resolve_metrics_split(eval_cfg)

        self.token_summary_cfg = eval_cfg.get("token_summary", {})
        self.sampler_cfg = self._resolve_sampler_cfg(eval_cfg)
        self.generator_cfg = self._resolve_generator_cfg(eval_cfg)
        self.analysis_cfgs = self._resolve_analysis_cfgs(eval_cfg)

        self.cfg = cfg
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                and cfg.get("trainer", {}).get("accelerator", "cpu") != "cpu"
                else "cpu"
            )
        )
        self.save_dir = Path(cfg.get("save_dir", "."))
        self.output_dir = self._resolve_output_dir(eval_cfg)
        self.sfreq: float | None = None
        self._prepare_data()
        self.plotting = EvaluationPlotting(
            sfreq=self.sfreq, val_dataset=self.val_dataset
        )

    def _resolve_sampler_cfg(self, eval_cfg: dict[str, Any]) -> dict[str, Any]:
        """Resolve sampler settings from config."""
        sampler_cfg = eval_cfg.get("example_sampler") or eval_cfg.get("sampler") or {}
        return dict(sampler_cfg) if sampler_cfg else {}

    def _resolve_generator_cfg(self, eval_cfg: dict[str, Any]) -> dict[str, Any]:
        """Resolve generator settings from config."""
        generator_cfg = eval_cfg.get("generator") or {}
        return dict(generator_cfg) if generator_cfg else {}

    def _resolve_analysis_cfgs(self, eval_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize analysis config entries into a list."""
        analyses = eval_cfg.get("analyses", [])
        if isinstance(analyses, dict):
            return [analyses]
        if isinstance(analyses, list):
            return analyses
        return []

    def _resolve_metrics_split(self, eval_cfg: dict[str, Any]) -> str:
        """Resolve which dataloader split to use for metric computation."""
        split = str(eval_cfg.get("metrics_split", "val")).lower()
        if split not in {"val", "test"}:
            raise ValueError("eval_runner.metrics_split must be 'val' or 'test'.")
        return split

    def _resolve_run_name(self, eval_cfg: dict[str, Any]) -> str | None:
        """Resolve a stable output name for eval runs without checkpoints."""
        run_name = eval_cfg.get("run_name")
        if run_name:
            return str(run_name)
        if self.use_timesfm and isinstance(self.timesfm_cfg, dict):
            model_id = self._resolve_timesfm_model_id(self.timesfm_cfg)
            if model_id:
                return Path(model_id).name
            model_cls = self.timesfm_cfg.get("model_cls") or self.timesfm_cfg.get(
                "model_class"
            )
            if model_cls:
                return f"timesfm_{model_cls}"
            return "timesfm"
        return None

    def _resolve_timesfm_model_id(self, cfg: dict[str, Any]) -> str | None:
        """Resolve the TimesFM model identifier from config aliases."""
        for key in ("model_id", "model", "pretrained", "model_name"):
            value = cfg.get(key)
            if value:
                return str(value)
        return None

    def _resolve_output_dir(self, eval_cfg: dict[str, Any]) -> Path | None:
        """Resolve an explicit output directory override, if provided."""
        output_dir = eval_cfg.get("output_dir") or eval_cfg.get("out_dir")
        if not output_dir:
            return None
        path = Path(output_dir)
        if not path.is_absolute():
            path = self.save_dir / path
        return path

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _prepare_data(self) -> None:
        """Build datasets and the validation/test dataloaders."""
        datasets = split_datasets(**self.cfg["datasplitter"])

        dataloader_cls_name = self.cfg.get("dataloader_class", "DataLoader")
        dataloader_cls = globals().get(dataloader_cls_name, DataLoader)

        args = self.cfg.get("dataloader", {})
        shuffle = None if isinstance(datasets.val, IterableDataset) else False

        self.val_loader = dataloader_cls(datasets.val, shuffle=shuffle, **args)
        self.test_loader = dataloader_cls(datasets.test, shuffle=shuffle, **args)
        self.datasets = datasets
        self.val_dataset = datasets.val
        self.test_dataset = datasets.test
        self.train_dataset = datasets.train
        self.postprocessor = getattr(datasets.train, "postprocessor", None)
        self.sfreq = getattr(datasets.val, "sfreq", None)
        self._spatial_distances = None
        self._spatial_weights = None

    def _resolve_checkpoint(self, ckpt_path: str | None) -> str:
        """Fallback to the newest matching checkpoint if the requested one is gone."""
        if ckpt_path is None:
            raise FileNotFoundError("No checkpoint path provided for evaluation.")

        path = Path(ckpt_path)
        if path.exists():
            return str(path)

        alt = self._wait_for_checkpoint(path, timeout=30)
        if alt:
            logger.warning(
                "Requested checkpoint %s missing; using latest available %s",
                path,
                alt,
            )
            return str(alt)

        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    def _find_latest_checkpoint(self, path: Path) -> Path | None:
        """Return the most recently modified checkpoint matching the prefix."""
        prefix = path.stem.split("-epoch")[0]
        pattern = f"{prefix}-epoch*{path.suffix}"
        candidates = sorted(path.parent.glob(pattern))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _load_model(self, ckpt_path: str) -> LitModel:
        """Load the Lightning model from a checkpoint path."""
        path = Path(ckpt_path)
        if not path.exists():
            alt = self._wait_for_checkpoint(path, timeout=60)
            if alt:
                logger.warning(
                    "Checkpoint %s disappeared; falling back to %s", path, alt
                )
                path = alt
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self._wait_for_checkpoint_ready(path)

        def _load() -> LitModel:
            if self.lit_module_name is not None:
                lit_module = globals().get(self.lit_module_name)
                # Pass postprocessor for modules that need it (e.g., VidtokLightning)
                return lit_module.load_from_checkpoint(
                    str(path), strict=False, postprocessor=self.postprocessor
                )
            return LitModel.load_from_checkpoint(str(path), strict=False)

        lit_model = self._load_with_retries(_load, path)
        lit_model.model.to(self.device)
        lit_model.model.eval()
        lit_model.model = self._ensure_compiled(lit_model.model)
        lit_model.model.eval()
        return lit_model

    def _load_timesfm(self) -> Any:
        """Load a TimesFM wrapper for generation-only evals."""
        if not isinstance(self.timesfm_cfg, dict):
            raise ValueError("eval_runner.timesfm must be a mapping.")

        model_id = self._resolve_timesfm_model_id(self.timesfm_cfg)
        if not model_id:
            raise ValueError(
                "eval_runner.timesfm.model_id must be set for TimesFM evaluation."
            )

        model_cls = self.timesfm_cfg.get("model_cls") or self.timesfm_cfg.get(
            "model_class"
        )
        forecast_config = self.timesfm_cfg.get(
            "forecast_config"
        ) or self.timesfm_cfg.get("compile_config")
        forecast_kwargs = self.timesfm_cfg.get("forecast_kwargs")
        device = self.timesfm_cfg.get("device") or self.device

        from ..models.timesfm_wrapper import TimesFMWrapper

        return TimesFMWrapper(
            model_id=model_id,
            model_cls=model_cls,
            forecast_config=forecast_config,
            forecast_kwargs=forecast_kwargs,
            device=device,
        )

    def _ensure_compiled(self, model: torch.nn.Module) -> torch.nn.Module:
        """Compile the model if possible; otherwise raise."""
        if not self.compile_model:
            if hasattr(model, "_orig_mod"):
                return model._orig_mod
            return model
        if hasattr(model, "_orig_mod"):
            return model

        try:
            return torch.compile(model)
        except Exception as exc:
            raise RuntimeError("Failed to compile model for evaluation.") from exc

    def _wait_for_checkpoint(self, path: Path, timeout: int = 30) -> Path | None:
        """Poll for a matching checkpoint for up to `timeout` seconds."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            alt = self._find_latest_checkpoint(path)
            if alt:
                return alt
            time.sleep(1)
        return None

    def _wait_for_checkpoint_ready(self, path: Path) -> None:
        """Wait for a checkpoint to exist and stop changing on disk."""
        deadline = time.time() + self.checkpoint_wait_timeout_s
        last_size: int | None = None
        last_mtime: float | None = None
        stable_since: float | None = None

        while time.time() < deadline:
            if not path.exists():
                time.sleep(self.checkpoint_poll_seconds)
                continue

            stat = path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
            now = time.time()

            # Treat the checkpoint as ready once size + mtime are stable for a window.
            if size > 0 and size == last_size and mtime == last_mtime:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= self.checkpoint_stable_seconds:
                    return
            else:
                stable_since = None
                last_size = size
                last_mtime = mtime

            time.sleep(self.checkpoint_poll_seconds)

        raise TimeoutError(
            "Checkpoint did not stabilize before timeout: "
            f"{path} (waited {self.checkpoint_wait_timeout_s}s)"
        )

    def _load_with_retries(
        self, loader: Callable[[], LitModel], path: Path
    ) -> LitModel:
        """Retry checkpoint loads that fail due to truncated zip files."""
        for attempt in range(self.checkpoint_load_retries + 1):
            try:
                return loader()
            except (RuntimeError, OSError, EOFError) as exc:
                if not self._is_checkpoint_load_error(exc):
                    raise
                if attempt >= self.checkpoint_load_retries:
                    raise
                logger.warning(
                    "Checkpoint %s not ready (%s); retrying in %.1fs",
                    path,
                    exc,
                    self.checkpoint_retry_wait_s,
                )
                time.sleep(self.checkpoint_retry_wait_s)
                self._wait_for_checkpoint_ready(path)
        raise RuntimeError(f"Failed to load checkpoint after retries: {path}")

    def _is_checkpoint_load_error(self, exc: Exception) -> bool:
        """Check if a checkpoint load error is likely due to partial writes."""
        msg = str(exc).lower()
        return any(
            snippet in msg
            for snippet in (
                "pytorchstreamreader failed reading zip archive",
                "failed finding central directory",
                "not a zip file",
                "unexpected eof",
                "end of file",
                "bad zipfile",
            )
        )

    def _move_to_device(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move inputs and targets to the configured device."""
        inputs, targets = batch

        if isinstance(inputs, (tuple, list)):
            inputs = tuple(x.to(self.device) for x in inputs)
        else:
            inputs = inputs.to(self.device)

        return inputs, targets.to(self.device)

    def _extract_logits(self, outputs: Any) -> torch.Tensor:
        """Resolve model logits from heterogeneous forward outputs."""
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs

    def _pack_outputs(self, logits, outputs: Any) -> tuple[torch.Tensor, ...]:
        """Ensure logits are the first element in the output tuple."""
        if isinstance(outputs, (tuple, list)):
            return [logits] + [out for out in outputs[1:]]

        return outputs

    def _select_generation_dataset(self) -> Any:
        """Select the dataset split used for generation."""
        split = str(self.sampler_cfg.get("split", "val")).lower()
        if split == "test":
            return self.test_dataset
        elif split == "train":
            return self.train_dataset
        return self.val_dataset

    def _select_metrics_loader(self) -> DataLoader:
        """Select the dataloader split used for loss/metric aggregation."""
        if self.metrics_split == "test":
            return self.test_loader
        return self.val_loader

    def _load_analysis_params(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Load analysis params from inline dict or a YAML file."""
        if "params" in entry and isinstance(entry["params"], dict):
            return dict(entry["params"])
        cfg_path = entry.get("config")
        if cfg_path:
            path = Path(cfg_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return dict(data)
        return {}

    def _build_analysis(self, entry: dict[str, Any]) -> Any | None:
        """Instantiate an analysis class from a config entry."""
        class_name = entry.get("class")
        if not class_name:
            return None
        classes = {
            "RolloutDivergenceAnalysis": RolloutDivergenceAnalysis,
            "RolloutSlidingWindowAnalysis": RolloutSlidingWindowAnalysis,
        }
        analysis_cls = classes.get(class_name)
        if analysis_cls is None:
            raise ValueError(f"Unknown analysis class: {class_name}")
        params = self._load_analysis_params(entry)
        if params.get("enabled", True) is False:
            return None
        return analysis_cls(
            params,
            sfreq=self.sfreq,
            plotting=self.plotting,
            spatial_weights=self._spatial_weights,
        )

    def _run_generation_and_analysis(
        self, model: torch.nn.Module, out_dir: Path
    ) -> None:
        """Run sampler/generator and analysis passes (with cache reuse)."""
        analyses = list(self.analysis_cfgs)
        if not analyses and not self.generator_cfg:
            return

        paired_name = self.generator_cfg.get(
            "paired_file", self.generator_cfg.get("paired_name", "paired_rollouts.npy")
        )
        paired_path = out_dir / paired_name
        result = None
        loaded_from_cache = False
        if paired_path.exists():
            result = load_paired_rollouts(paired_path)
            loaded_from_cache = True
        else:
            if not self.generator_cfg:
                print("[eval_runner] Missing generator config; cannot generate.")
                return
            if not self.generator_cfg.get("enabled", True):
                return
            if not self.sampler_cfg:
                print("[eval_runner] Missing sampler config; cannot generate.")
                return
            sampler = SessionSampler(
                self._select_generation_dataset(), self.sampler_cfg
            )
            samples = sampler.sample_sessions()
            if not samples:
                print("[eval_runner] No valid sessions for generation.")
                return
            generator = RolloutGenerator(
                self.generator_cfg,
                device=self.device,
                sfreq=self.sfreq,
            )
            result = generator.generate(
                model,
                samples,
                out_dir=out_dir,
                plotting=self.plotting,
            )
            if result is None:
                return
            save_paired_rollouts(paired_path, result)

        if result is None:
            return
        if loaded_from_cache and self.generator_cfg.get("enabled", True):
            generator = RolloutGenerator(
                self.generator_cfg,
                device=self.device,
                sfreq=self.sfreq,
            )
            generator.plot_cached_rollouts(
                result,
                out_dir=out_dir,
                plotting=self.plotting,
            )
        for entry in analyses:
            analysis = self._build_analysis(entry)
            if analysis is None:
                continue
            if hasattr(analysis, "set_rollout_info"):
                analysis.set_rollout_info(
                    metadata=result.metadata,
                    context_steps=result.context_steps,
                    total_steps=result.total_steps,
                )
            analysis.run(result.generated, result.target, out_dir)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def run(self) -> Dict[str, Any]:
        """Run evaluation and return the aggregated summary dict."""
        if self.use_timesfm:
            model = self._load_timesfm()
            out_dir = self._prepare_output_dir()
            self._save_config(out_dir)
            self._run_generation_and_analysis(model, out_dir)
            summary = {
                "loss_mean": float("nan"),
                "loss_std": float("nan"),
                "losses": [],
                "metrics": {},
                "mode": "generation_only",
            }
            self._persist_results(summary, out_dir)
            return summary

        lit_model = self._load_model(self.ckpt_path)
        out_dir = self._prepare_output_dir()
        self._save_config(out_dir)

        if hasattr(lit_model.model, "set_eval_mode"):
            lit_model.model.set_eval_mode()

        token_plotter = TokenSummaryPlotter(
            self.token_summary_cfg,
            lit_model.model,
            loss_fn=lit_model.loss,
            sfreq=self.sfreq,
        )

        self._run_generation_and_analysis(lit_model.model, out_dir)

        losses: List[float] = []
        metrics: dict[str, list[float]] = defaultdict(list)
        example_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        metric_loader = self._select_metrics_loader()
        for batch_idx, batch in enumerate(metric_loader):
            inputs, targets = self._move_to_device(batch)

            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = lit_model.model(inputs)
            else:
                outputs = lit_model.model(inputs)

            logits = self._extract_logits(outputs)
            logits = logits.to(torch.float32)

            outputs = self._pack_outputs(logits, outputs)

            loss = lit_model.loss(outputs, targets, model=lit_model.model)
            losses.append(float(loss.cpu()))

            if self.postprocessor is not None:
                inputs, logits, targets = self.postprocessor(inputs, logits, targets)

            outputs = self._pack_outputs(logits, outputs)

            for name, metric in lit_model.loss.metrics.items():
                metric_val = metric(outputs, targets)
                metrics[name].append(float(metric_val.cpu()))

            inputs = inputs[0] if isinstance(inputs, (tuple, list)) else inputs

            if len(example_batches) < self.num_examples:
                example_batches.append(
                    (
                        inputs.cpu(),
                        logits.cpu(),
                        targets.cpu(),
                    )
                )

            token_plotter.update(outputs)

            if self.max_batches and (batch_idx + 1) >= self.max_batches:
                break

        summary = self._summarise_results(losses, metrics)
        self._persist_results(summary, out_dir)
        self._log(summary, example_batches, out_dir)
        token_plotter.finalize(out_dir)
        return summary

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    def _summarise_results(
        self, losses: list[float], metrics: dict[str, list[float]]
    ) -> Dict[str, Any]:
        """Aggregate batch-level losses/metrics into summary statistics."""
        summary = {
            "loss_mean": float(np.mean(losses)) if losses else float("nan"),
            "loss_std": float(np.std(losses)) if losses else float("nan"),
            "losses": losses,
            "metrics": {},
        }
        for name, values in metrics.items():
            summary["metrics"][name] = {
                "mean": float(np.mean(values)) if values else float("nan"),
                "std": float(np.std(values)) if values else float("nan"),
                "values": values,
            }
        return summary

    def _prepare_output_dir(self) -> Path:
        """Create and return the output directory for this eval run.

        If eval_runner.output_dir is set, use it as-is.
        """
        output_dir = getattr(self, "output_dir", None)
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir
        base_dir = self.save_dir / "logs" / f"version_{self.run_version}"
        if self.eval_epoch is not None:
            subdir = f"epoch_{int(self.eval_epoch):03d}"
        elif self.ckpt_path is not None:
            subdir = Path(self.ckpt_path).stem
        elif self.run_name:
            subdir = self.run_name
        else:
            subdir = "eval"
        out_dir = base_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _save_config(self, out_dir: Path) -> None:
        """Write the eval config used for this run."""
        cfg_path = out_dir / "eval_config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)

    def _persist_results(
        self,
        summary: Dict[str, Any],
        out_dir: Path,
    ) -> None:
        """Write the summary payload to disk."""
        payload = dict(summary)
        payload["step"] = self.eval_step
        payload["epoch"] = self.eval_epoch
        with open(out_dir / "summary.json", "w") as f:
            json.dump(payload, f, indent=2)

    def _log(
        self,
        summary: Dict[str, Any],
        examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        out_dir: Path,
    ) -> None:
        """Save metric distribution and example plots."""
        metric_fig = self.plotting.plot_metric_grid(
            summary["losses"], {k: v["values"] for k, v in summary["metrics"].items()}
        )
        example_figs = self.plotting.plot_examples(examples, out_dir)

        metrics_path = out_dir / "metrics_distributions.png"
        metric_fig.savefig(metrics_path, bbox_inches="tight")
        plt.close(metric_fig)

        for idx, fig in enumerate(example_figs):
            fig_path = out_dir / f"example_{idx}.png"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for running an eval config."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", required=True, type=Path, help="Path to YAML config"
    )
    parser.add_argument("--ckpt", type=Path, help="Checkpoint to evaluate")
    parser.add_argument(
        "--step", type=int, default=None, help="Global step for logging"
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.setdefault("eval_runner", {})

    ckpt_path = args.ckpt or eval_cfg.get("ckpt_path")
    timesfm_cfg = eval_cfg.get("timesfm")
    timesfm_model_id = None
    if isinstance(timesfm_cfg, dict):
        for key in ("model_id", "model", "pretrained", "model_name"):
            value = timesfm_cfg.get(key)
            if value:
                timesfm_model_id = str(value)
                break
    if ckpt_path is None and not timesfm_model_id:
        raise ValueError(
            "Provide --ckpt or set eval_runner.ckpt_path (or eval_runner.timesfm.model_id) "
            "in the config."
        )
    if ckpt_path is not None:
        eval_cfg["ckpt_path"] = str(ckpt_path)
    if args.step is not None:
        eval_cfg["step"] = args.step

    runner = EvaluationRunner(cfg)
    runner.run()


if __name__ == "__main__":  # pragma: no cover
    main()
