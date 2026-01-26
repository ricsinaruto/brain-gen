from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)


def _count_tokens(batch: Any) -> Optional[int]:
    """Best-effort token count for throughput metrics."""
    if torch.is_tensor(batch):
        if batch.dim() >= 2:
            # interpret the last dimension as sequence length when possible
            return int(batch.shape[0] * batch.shape[-1])
        return int(batch.numel())
    if isinstance(batch, (list, tuple)) and batch:
        return _count_tokens(batch[0])
    if isinstance(batch, dict) and batch:
        return _count_tokens(next(iter(batch.values())))
    return None


def _gpu_stats(device: int = 0) -> Dict[str, float]:
    """Collect lightweight GPU metrics; returns empty dict when unavailable."""
    stats: Dict[str, float] = {}
    if not torch.cuda.is_available():
        return stats

    try:
        gb = 1024**3
        stats["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated(device) / gb
        stats["gpu/max_memory_allocated_gb"] = (
            torch.cuda.max_memory_allocated(device) / gb
        )
        stats["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved(device) / gb
    except Exception:
        pass

    # Utilization via pynvml if available, otherwise via nvidia-smi
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats["gpu/utilization"] = float(util.gpu)
    except Exception:
        try:
            res = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                    f"--id={device}",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            util_value = res.strip().split("\n")[0]
            stats["gpu/utilization"] = float(util_value)
        except Exception:
            pass
    return stats


class PerformanceMonitor(pl.Callback):
    """Logs throughput, epoch timing, and GPU health to the active Lightning logger."""

    def __init__(self, log_every_n_steps: int = 1) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, log_every_n_steps)
        self._batch_start: float | None = None
        self._epoch_start: float | None = None
        self._reset_accumulators()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._epoch_start = time.perf_counter()

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._batch_start = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._batch_start is None:
            return
        duration = time.perf_counter() - self._batch_start
        if duration <= 0:
            return

        global_step = int(getattr(trainer, "global_step", 0))
        self._accumulate(duration, batch)

        if global_step % self.log_every_n_steps == 0:
            self._flush(trainer, global_step)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._epoch_start is None:
            return
        elapsed = time.perf_counter() - self._epoch_start
        step = int(getattr(trainer, "global_step", 0))
        metrics = {"perf/epoch_time_sec": elapsed}
        self._log(trainer, metrics, step)
        # Also flush any pending batch accumulations at epoch end
        self._flush(trainer, step)

    # ------------------------------------------------------------------ #
    # Accumulation helpers
    # ------------------------------------------------------------------ #
    def _reset_accumulators(self) -> None:
        self._batch_time_total = 0.0
        self._batch_count = 0
        self._token_total: float | None = None
        self._gpu_stats_sum: Dict[str, float] = {}

    def _accumulate(self, duration: float, batch: Any) -> None:
        self._batch_time_total += duration
        self._batch_count += 1

        tokens = _count_tokens(batch)
        if tokens:
            self._token_total = (
                float(tokens)
                if self._token_total is None
                else self._token_total + float(tokens)
            )

        gpu_stats = _gpu_stats()
        for key, value in gpu_stats.items():
            self._gpu_stats_sum[key] = self._gpu_stats_sum.get(key, 0.0) + value

    def _flush(self, trainer: pl.Trainer, step: int) -> None:
        if self._batch_count == 0:
            return

        total_time = self._batch_time_total
        metrics: Dict[str, float] = {
            "perf/batch_time_sec": total_time / self._batch_count,
            "perf/batches_per_sec": (
                self._batch_count / total_time if total_time > 0 else 0.0
            ),
        }

        if self._token_total is not None and total_time > 0:
            metrics["perf/tokens_per_batch"] = self._token_total / self._batch_count
            metrics["perf/tokens_per_sec"] = self._token_total / total_time

        for key, value in self._gpu_stats_sum.items():
            metrics[key] = value / self._batch_count

        self._log(trainer, metrics, step)
        self._reset_accumulators()

    def _log(self, trainer: pl.Trainer, metrics: Dict[str, float], step: int) -> None:
        logger = getattr(trainer, "logger", None)
        if not logger:
            return

        try:
            logger.log_metrics(metrics, step=step)
        except Exception:
            # Fallback for loggers without log_metrics, e.g., direct SummaryWriter
            experiment = getattr(logger, "experiment", None)
            if experiment is None:
                return
            for key, value in metrics.items():
                try:
                    experiment.add_scalar(key, value, step)
                except Exception:
                    pass
