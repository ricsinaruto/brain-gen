from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from pathlib import Path
from typing import Callable
from weakref import proxy
import pytorch_lightning as pl
import yaml
import time
import os
from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)


class ThreadedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint variant that offloads serialization to a background thread."""

    def __init__(
        self,
        *args,
        after_save: Callable | None = None,
        epoch_cadence: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.after_save = after_save
        self.epoch_cadence = (
            max(1, int(epoch_cadence)) if epoch_cadence is not None else None
        )
        self._threads: list[threading.Thread] = []
        self._state_lock = threading.Lock()
        self._latest_epoch: int | None = None
        self._latest_ckpt_path: str | None = None

    def _format_with_epoch(self, filepath: str, epoch: int) -> str:
        path = Path(filepath)
        return str(path.with_name(f"{path.stem}-epoch{epoch:05d}{path.suffix}"))

    def _mark_latest_and_get_stale(self, ckpt_path: str, epoch: int) -> str | None:
        """Track the newest checkpoint and return any path that should be deleted."""
        stale_path: str | None = None
        with self._state_lock:
            if self._latest_epoch is None or epoch >= self._latest_epoch:
                if self._latest_ckpt_path and self._latest_ckpt_path != ckpt_path:
                    stale_path = self._latest_ckpt_path
                self._latest_epoch = epoch
                self._latest_ckpt_path = ckpt_path
                self.last_model_path = ckpt_path
                if self.monitor is not None:
                    self.best_model_path = ckpt_path
            else:
                stale_path = ckpt_path
        return stale_path

    def _delete_checkpoint(self, ckpt_path: str) -> None:
        try:
            Path(ckpt_path).unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.warning("Failed to delete old checkpoint %s: %s", ckpt_path, exc)

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        curr_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        if self.epoch_cadence is not None:
            # PL epochs are 0-based; add 1 for human-friendly cadence
            if curr_epoch % self.epoch_cadence != 0:
                return
        # mirror the state updates from the base implementation
        filepath = self._format_with_epoch(filepath, curr_epoch)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        def _target() -> None:
            save_succeeded = False
            try:
                trainer.save_checkpoint(filepath, self.save_weights_only)
                save_succeeded = True
                if trainer.is_global_zero:
                    for log in trainer.loggers:
                        log.after_save_checkpoint(proxy(self))
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to save checkpoint %s: %s", filepath, exc)
            finally:
                if save_succeeded:
                    stale_path = self._mark_latest_and_get_stale(filepath, curr_epoch)
                    if stale_path:
                        self._delete_checkpoint(stale_path)
                if self.after_save and trainer.is_global_zero:
                    try:
                        self.after_save(filepath, trainer, trainer.lightning_module)
                    except Exception as exc:  # pragma: no cover
                        logger.exception(
                            "Post-save hook failed for %s: %s", filepath, exc
                        )

        thread = threading.Thread(target=_target, daemon=True)
        self._threads.append(thread)
        thread.start()

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: pl.LightningModule
    ) -> None:
        for thread in list(self._threads):
            thread.join(timeout=10)


class EvaluationLauncher:
    """Triggers evaluation runs when checkpoints are written."""

    def __init__(self, cfg: dict, save_dir: Path) -> None:
        self.cfg = deepcopy(cfg)
        self.save_dir = Path(save_dir)

        eval_cfg = cfg.get("eval_runner", {})
        self.enabled = bool(eval_cfg.get("enabled", True))
        self.use_modal = bool(eval_cfg.get("use_modal", True))
        self.modal_app = eval_cfg.get("modal_app", "brain-gen")
        self.modal_function = eval_cfg.get("modal_function", "runevals")
        self.modal_env = eval_cfg.get("modal_env", "main")
        self.max_batches = eval_cfg.get("max_batches", 8)
        self.num_examples = eval_cfg.get("num_examples", 3)
        self.checkpoint_wait_timeout_s = int(
            eval_cfg.get("checkpoint_wait_timeout_s", 300)
        )
        self.checkpoint_stable_seconds = float(
            eval_cfg.get("checkpoint_stable_seconds", 5)
        )
        self.checkpoint_poll_seconds = float(
            eval_cfg.get("checkpoint_poll_seconds", 1.0)
        )
        self._seen_paths: set[str] = set()
        self._pending_threads: list[threading.Thread] = []

    def __call__(
        self, ckpt_path: str, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self.enabled:
            return
        # if ckpt_path in self._seen_paths:
        #    return
        # self._seen_paths.add(ckpt_path)

        step = int(getattr(trainer, "global_step", 0))
        epoch = int(getattr(trainer, "current_epoch", -1)) + 1
        vnum = int(getattr(trainer.logger, "version", "test"))
        if self.use_modal and self._launch_modal(ckpt_path, vnum, step, epoch):
            return

    # ------------------------------------------------------------------ #
    # Modal evaluation
    # ------------------------------------------------------------------ #
    def _launch_modal(
        self,
        ckpt_path: str,
        vnum: int,
        step: int | None,
        epoch: int | None,
    ) -> bool:
        cfg = self._write_eval_config(ckpt_path, vnum, step, epoch)
        self._wait_for_checkpoint_ready(ckpt_path)

        time.sleep(30)

        try:
            import modal  # type: ignore
        except Exception:
            logger.warning("Modal SDK not available; skipping modal eval dispatch.")
            return False

        try:
            fn = modal.Function.from_name(
                self.modal_app,
                self.modal_function,
                environment_name=self.modal_env,
            )
            fn.spawn(args=json.dumps(cfg, indent=4), dict_mode=True)
            logger.info(
                "Launched modal eval for %s via %s.%s",
                ckpt_path,
                self.modal_app,
                self.modal_function,
            )
            return True
        except Exception as exc:  # pragma: no cover - network/Modal errors
            logger.warning("Modal eval dispatch failed: %s", exc)
        return False

    def _wait_for_checkpoint_ready(self, ckpt_path: str) -> None:
        """Wait for a checkpoint to appear and stop changing on disk."""
        deadline = time.time() + self.checkpoint_wait_timeout_s
        last_size: int | None = None
        last_mtime: float | None = None
        stable_since: float | None = None

        while time.time() < deadline:
            if not os.path.exists(ckpt_path):
                time.sleep(self.checkpoint_poll_seconds)
                continue

            stat = os.stat(ckpt_path)
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
            f"{ckpt_path} (waited {self.checkpoint_wait_timeout_s}s)"
        )

    def _write_eval_config(
        self, ckpt_path: str, vnum: int, step: int | None, epoch: int | None
    ) -> Path:
        cfg = deepcopy(self.cfg)
        cfg.setdefault("eval_runner", {})
        cfg["eval_runner"]["ckpt_path"] = str(ckpt_path)
        cfg["eval_runner"]["max_batches"] = self.max_batches
        cfg["eval_runner"]["num_examples"] = self.num_examples
        cfg["eval_runner"]["epoch"] = int(epoch) if epoch is not None else None
        cfg["eval_runner"]["version"] = int(vnum)
        if step is not None:
            cfg["eval_runner"]["step"] = int(step)

        out_dir = self.save_dir / "logs" / f"version_{vnum}" / f"epoch_{int(epoch):03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = Path(ckpt_path).stem
        cfg_path = out_dir / f"{ckpt_name}.yaml"

        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        return cfg
