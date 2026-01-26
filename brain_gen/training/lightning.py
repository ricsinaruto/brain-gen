from functools import wraps
from time import perf_counter

import torch.nn as nn
import pytorch_lightning as pl
import torch
from typing import Any, Sequence


def benchmark_train_step(log_key: str = "perf/train_step_ms"):
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            cfg = getattr(self, "trainer_cfg", {})
            if not cfg.get("benchmark_train_step", False):
                return fn(self, *args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = perf_counter()
            out = fn(self, *args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (perf_counter() - start) * 1000.0
            try:
                self.log(
                    log_key,
                    elapsed_ms,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )
            except Exception:
                pass
            return out

        return wrapper

    return decorator


class LitDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, dataloader_cls, dataloader_args):
        super().__init__()
        self.train_ds, self.val_ds = train_ds, val_ds
        self.dataloader_cls = dataloader_cls

        self.batch_size = dataloader_args.pop("batch_size", 1)
        self.dataloader_args = dataloader_args

    def train_dataloader(self):
        return self.dataloader_cls(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            **self.dataloader_args,
        )

    def val_dataloader(self):
        return self.dataloader_cls(
            self.val_ds,
            shuffle=False,
            batch_size=self.batch_size,
            **self.dataloader_args,
        )


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model_class: nn.Module,
        loss_class: nn.Module,
        model_cfg: dict,
        loss_cfg: dict,
        trainer_cfg: dict,
        postprocessor=None,
        free_run_cfg: dict | None = None,
    ) -> None:
        """Args:

        model_class: Name of the model to use loss_class: Name of the loss function to
        use datasets: Dataset configuration dictionary model_cfg: Model configuration
        dictionary loss_cfg: Loss configuration dictionary trainer_cfg: Trainer
        configuration dictionary
        """
        super().__init__()
        try:
            self.save_hyperparameters()
        except TypeError:
            print("Could not save hyperparameters")
            pass

        # keep a reference to the trainer config (for scheduler, etc.)
        self.trainer_cfg = trainer_cfg
        self.postprocessor = postprocessor
        self.free_run_cfg = free_run_cfg
        # Collect test-time outputs to enable downstream metrics/plots
        self.test_predictions: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []
        self._resume_lr_applied = False
        self._resume_weight_decay_applied = False
        self._resume_context_applied = False
        self._resume_token_corruption_applied = False
        self._grad_clip_total_steps = 0
        self._grad_clip_count = 0
        self._last_grad_norm_step = None
        self.model_cfg = dict(model_cfg)

        # Create model and loss instances
        self.model = model_class(**model_cfg)

        # compile the model if requested
        if trainer_cfg.get("compile", False):
            self.model = torch.compile(self.model)

        self.loss = loss_class(**loss_cfg)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._apply_resume_lr_override()
        self._apply_resume_weight_decay_override()
        self._apply_resume_context_override()
        self._apply_resume_token_corruption_override()

    def on_train_start(self) -> None:
        super().on_train_start()
        self._apply_resume_lr_override()
        self._apply_resume_weight_decay_override()
        self._apply_resume_context_override()
        self._apply_resume_token_corruption_override()

    def on_before_optimizer_step(self, optimizer) -> None:
        super().on_before_optimizer_step(optimizer)
        self._log_grad_norm(optimizer)

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ) -> None:
        self._log_grad_norm(optimizer)
        super().configure_gradient_clipping(
            optimizer, gradient_clip_val, gradient_clip_algorithm
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        self._unwrap_compiled_checkpoint(checkpoint)
        self._apply_resume_token_corruption_override(allow_without_trainer=True)
        resume_lr = self.trainer_cfg.get("resume_lr", None)
        resume_weight_decay = self.trainer_cfg.get("resume_weight_decay", None)
        if resume_lr is None and resume_weight_decay is None:
            return
        resume_lr = float(resume_lr) if resume_lr is not None else None
        resume_weight_decay = (
            float(resume_weight_decay) if resume_weight_decay is not None else None
        )
        opt_states = checkpoint.get("optimizer_states", []) or []
        for opt_state in opt_states:
            param_groups = opt_state.get("param_groups", []) or []
            for idx, group in enumerate(param_groups):
                if resume_lr is not None:
                    group["lr"] = resume_lr
                    if "initial_lr" in group:
                        group["initial_lr"] = resume_lr
                if resume_weight_decay is not None:
                    # Configure decay group first, then no-decay groups.
                    if idx == 0:
                        group["weight_decay"] = resume_weight_decay
                    else:
                        group["weight_decay"] = 0.0
        if resume_lr is not None:
            for sched_key in ("lr_schedulers", "lr_scheduler_states"):
                sched_states = checkpoint.get(sched_key, []) or []
                for sched_state in sched_states:
                    if "base_lrs" in sched_state:
                        sched_state["base_lrs"] = [
                            resume_lr for _ in sched_state["base_lrs"]
                        ]

    def _unwrap_compiled_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Unwrap compiled checkpoints when compile is disabled."""
        if self.trainer_cfg.get("compile", False):
            return
        if hasattr(self.model, "_orig_mod"):
            self.model = self.model._orig_mod
        state_dict = checkpoint.get("state_dict", None)
        if not state_dict:
            return
        prefix = "model._orig_mod."
        if not any(key.startswith(prefix) for key in state_dict):
            return
        checkpoint["state_dict"] = {
            (f"model.{key[len(prefix):]}" if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }

    def _compute_grad_norm(self, optimizer) -> float | None:
        params = []
        for group in optimizer.param_groups:
            params.extend([p for p in group["params"] if p.grad is not None])
        if not params:
            return None
        norms = []
        for param in params:
            grad = param.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            norms.append(torch.linalg.vector_norm(grad, ord=2))
        if not norms:
            return None
        total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
        return float(total_norm)

    def _log_grad_norm(self, optimizer) -> None:
        if not self.training:
            return
        if self._last_grad_norm_step == self.global_step:
            return
        grad_norm = self._compute_grad_norm(optimizer)
        if grad_norm is None:
            return
        self._grad_clip_total_steps += 1
        if grad_norm > self.trainer_cfg.get("grad_clip_threshold", 1.0):
            self._grad_clip_count += 1
        clip_pct = 100.0 * self._grad_clip_count / self._grad_clip_total_steps
        self.log("grad_norm", grad_norm, on_step=True, on_epoch=False)
        self.log("grad_clip_pct", clip_pct, on_step=True, on_epoch=False)
        self._last_grad_norm_step = self.global_step

    def _apply_resume_lr_override(self) -> None:
        """Override optimizer + scheduler LRs after checkpoint restore."""
        if self._resume_lr_applied:
            return
        resume_lr = self.trainer_cfg.get("resume_lr", None)
        if resume_lr is None:
            return
        resume_lr = float(resume_lr)
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        optimizers = getattr(trainer, "optimizers", None)
        if not optimizers:
            return
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = resume_lr
                if "initial_lr" in group:
                    group["initial_lr"] = resume_lr
        sched_cfgs = getattr(trainer, "lr_scheduler_configs", None) or []
        for cfg in sched_cfgs:
            scheduler = None
            if hasattr(cfg, "scheduler"):
                scheduler = cfg.scheduler
            elif isinstance(cfg, dict):
                scheduler = cfg.get("scheduler")
            if scheduler is None:
                continue
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [resume_lr for _ in scheduler.base_lrs]
        self._resume_lr_applied = True

    def _apply_resume_weight_decay_override(self) -> None:
        """Override optimizer weight decay after checkpoint restore."""
        if self._resume_weight_decay_applied:
            return
        resume_weight_decay = self.trainer_cfg.get("resume_weight_decay", None)
        if resume_weight_decay is None:
            return
        resume_weight_decay = float(resume_weight_decay)
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        optimizers = getattr(trainer, "optimizers", None)
        if not optimizers:
            return

        _, no_decay = self._setup_params()
        no_decay_ids = {id(p) for p in no_decay}

        for optimizer in optimizers:
            for group in optimizer.param_groups:
                params = group.get("params", [])
                if not params:
                    group["weight_decay"] = 0.0
                    continue
                if all(id(p) in no_decay_ids for p in params):
                    group["weight_decay"] = 0.0
                else:
                    group["weight_decay"] = resume_weight_decay

        self._resume_weight_decay_applied = True

    def _apply_resume_context_override(self) -> None:
        """Resize model context/rope when resuming from a checkpoint."""
        if self._resume_context_applied:
            return
        cfg = self.trainer_cfg.get("resume_context", None)
        if not cfg:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        ckpt_path = getattr(trainer, "ckpt_path", None)
        if ckpt_path is None:
            return
        if not hasattr(self.model, "resize_context"):
            raise ValueError(
                "resume_context configured but model does not support resize_context."
            )

        ctx = dict(cfg)
        input_shape = ctx.get("input_shape", None)
        if input_shape is not None:
            input_shape = tuple(int(x) for x in input_shape)
        spatial_reduction = ctx.get("spatial_reduction", None)
        if spatial_reduction is not None:
            if isinstance(spatial_reduction, int):
                spatial_reduction = int(spatial_reduction)
            else:
                spatial_reduction = tuple(int(x) for x in spatial_reduction)
        temporal_reduction = ctx.get("temporal_reduction", None)
        if temporal_reduction is not None:
            temporal_reduction = int(temporal_reduction)
        rope_theta = ctx.get("rope_theta", None)
        if rope_theta is not None:
            rope_theta = float(rope_theta)
        max_position_embeddings = ctx.get("max_position_embeddings", None)
        if max_position_embeddings is not None:
            max_position_embeddings = int(max_position_embeddings)

        self.model.resize_context(
            input_shape=input_shape,
            spatial_reduction=spatial_reduction,
            temporal_reduction=temporal_reduction,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
        self._resume_context_applied = True

    def _apply_resume_token_corruption_override(
        self, *, allow_without_trainer: bool = False
    ) -> None:
        """Re-apply token corruption settings when resuming from a checkpoint."""
        if self._resume_token_corruption_applied:
            return
        if not allow_without_trainer:
            trainer = getattr(self, "trainer", None)
            if trainer is None:
                return
            if getattr(trainer, "ckpt_path", None) is None:
                return

        cfg = self._resolve_token_corruption_cfg()
        if not cfg:
            return

        model = self._unwrap_model_for_corruption_cfg()
        if model is None:
            return
        if hasattr(model, "update_token_corruption_cfg"):
            model.update_token_corruption_cfg(cfg)
        elif hasattr(model, "_init_token_corruption_cfg"):
            model.token_corruption_cfg = model._init_token_corruption_cfg(cfg)
        else:
            return

        self._resume_token_corruption_applied = True

    def _resolve_token_corruption_cfg(self) -> dict | None:
        """Select token corruption config from lightning or model config."""
        cfg = self.trainer_cfg.get("resume_token_corruption", None)
        if cfg is True:
            cfg = self.model_cfg.get("token_corruption_cfg", None)
        if cfg is None:
            cfg = self.model_cfg.get("token_corruption_cfg", None)
        if isinstance(cfg, dict):
            return cfg
        return None

    def _unwrap_model_for_corruption_cfg(self) -> nn.Module | None:
        """Return underlying model for attribute updates when compiled."""
        model = getattr(self, "model", None)
        if model is None:
            return None
        if hasattr(model, "_orig_mod"):
            return model._orig_mod
        return model

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ):
        inputs, targets = batch
        outputs = self.model(inputs)

        logits = outputs
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]

        if self.postprocessor is not None:
            inputs, logits, targets = self.postprocessor(inputs, logits, targets)

        if isinstance(outputs, (tuple, list)):
            outputs = [logits] + [out for out in outputs[1:]]

        loss = self.loss(outputs, targets, model=self.model)

        targets = targets.detach()
        if isinstance(outputs, (tuple, list)):
            logits = logits.detach()
            outputs = [logits] + [out for out in outputs[1:]]

        metrics_for_stage: dict[str, torch.Tensor] = {}
        self.log(f"{stage}/loss", loss.detach(), prog_bar=True)
        for name, metric in self.loss.metrics.items():
            metric_val = metric(outputs, targets)
            metrics_for_stage[name] = metric_val
            self.log(f"{stage}/{name}", metric_val, prog_bar=True)

        # log learning rate
        if stage == "train":
            optimizer = self.optimizers()
            lr = optimizer.param_groups[0]["lr"]
            self.log("lr", lr)
            weight_decay = optimizer.param_groups[0].get("weight_decay", None)
            if weight_decay is not None:
                self.log("weight_decay", weight_decay)

        return loss

    @benchmark_train_step()
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Args: batch: Batch of data batch_idx: Batch index.

        Returns:     Loss value
        """
        return self._step(batch, batch_idx, stage="train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Args: batch: Batch of data batch_idx: Batch index.

        Returns:     Loss value
        """
        _ = self._step(batch, batch_idx, stage="val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        targets = None
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch, targets = batch

        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]

        if isinstance(targets, tuple) or isinstance(targets, list):
            targets = targets[0]

        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)

        self.test_predictions.append(probs.cpu())

        if targets is not None:
            self.test_targets.append(targets.cpu())

    @staticmethod
    def _parse_lr_warmup_cfg(cfg: Any) -> tuple[int, str | None]:
        if cfg is None:
            return 0, None
        if isinstance(cfg, int):
            return int(cfg), None
        if isinstance(cfg, dict):
            steps = int(cfg.get("steps", 0))
            interval = cfg.get("interval", None)
            return steps, interval
        raise ValueError("lr_warmup must be an int or dict with 'steps'.")

    @staticmethod
    def _is_no_decay_param(name: str, param: torch.nn.Parameter) -> bool:
        """Return True for params that should skip weight decay."""
        lname = name.lower()
        return (
            lname.endswith("bias")
            or "layernorm" in lname
            or "rmsnorm" in lname
            or ".norm." in lname
            or lname.endswith(".norm.weight")
            or "_norm" in lname
            or param.ndim == 1  # robust heuristic: norms + biases are usually 1D
        )

    def _setup_params(self) -> torch.nn.ParameterList:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if self._is_no_decay_param(name, p):
                no_decay.append(p)
            else:
                decay.append(p)

        return decay, no_decay

    def configure_optimizers(self) -> torch.optim.Optimizer:
        decay, no_decay = self._setup_params()
        param = [
            {"params": decay, "weight_decay": self.trainer_cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            param,
            lr=self.trainer_cfg["lr"],
            betas=(0.9, 0.95),  # recommended by the Qwen team
            eps=1e-8,
        )

        # verify!
        wd0 = set(map(id, optimizer.param_groups[1]["params"]))
        mismatches = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            should_no_decay = self._is_no_decay_param(n, p)
            in_no_decay = id(p) in wd0
            if should_no_decay != in_no_decay:
                mismatches.append((n, should_no_decay, in_no_decay))

        if self.trainer_cfg.get("no_decay_verbose", False):
            name_by_id = {
                id(p): n for n, p in self.named_parameters() if p.requires_grad
            }
            no_decay_names = sorted(
                {
                    name_by_id[id(p)]
                    for p in optimizer.param_groups[1]["params"]
                    if id(p) in name_by_id
                }
            )
            print("no_decay params:")
            for name in no_decay_names:
                print(f"  {name}")
            print(f"no_decay params count: {len(no_decay_names)}")

        if mismatches:
            for n, should_no_decay, in_no_decay in mismatches:
                expected = "no_decay" if should_no_decay else "decay"
                actual = "no_decay" if in_no_decay else "decay"
                print(f"{n}: expected {expected}, got {actual}")
        else:
            print("no_decay check: all params match heuristic")

        warmup_cfg = self.trainer_cfg.get("lr_warmup", None)
        warmup_steps, warmup_interval = self._parse_lr_warmup_cfg(warmup_cfg)

        raw_sched_cfg = self.trainer_cfg.get("lr_scheduler", None)
        # Work on a copy to avoid mutating hyperparameters
        sched_cfg = dict(raw_sched_cfg) if raw_sched_cfg else None
        scheduler = None
        interval = None
        frequency = 1
        monitor = None

        if sched_cfg:
            # Extract PL-specific options
            interval = sched_cfg.pop("interval", "epoch")
            frequency = sched_cfg.pop("frequency", 1)
            monitor = sched_cfg.pop("monitor", None)
            class_name = sched_cfg.pop("class_name")

            # Build the scheduler from torch.optim.lr_scheduler
            try:
                scheduler_cls = getattr(torch.optim.lr_scheduler, class_name)
            except AttributeError as e:
                raise ValueError(
                    f"Unknown lr_scheduler '{class_name}' in torch.optim.lr_scheduler"
                ) from e

            # Remaining entries in sched_cfg are passed to the constructor
            scheduler = scheduler_cls(optimizer, **sched_cfg)

        if warmup_steps > 0:
            if warmup_interval is not None and warmup_interval not in {"step", "epoch"}:
                raise ValueError("lr_warmup.interval must be 'step' or 'epoch'.")
            if scheduler is None:
                interval = warmup_interval or "step"
                frequency = 1
            else:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    raise ValueError(
                        "lr_warmup does not support ReduceLROnPlateau schedulers."
                    )
                if warmup_interval is not None and warmup_interval != interval:
                    raise ValueError(
                        "lr_warmup.interval must match lr_scheduler.interval."
                    )

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)),
            )
            if scheduler is None:
                scheduler = warmup_scheduler
            else:
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[warmup_steps],
                )

        if scheduler is None:
            return optimizer

        # Compose the PL optimizer/scheduler config
        lr_sched_dict = {
            "scheduler": scheduler,
            "interval": interval,  # unit of the scheduler's step
            "frequency": frequency,  # how often to call
        }
        if monitor is not None:
            lr_sched_dict["monitor"] = monitor

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_sched_dict,
        }


class LitModelFreerun(LitModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.free_run_cfg = self._prepare_free_run_cfg(self.free_run_cfg)

    def _prepare_free_run_cfg(self, cfg: dict | None) -> dict[str, Any]:
        """Normalise optional K-step free-run settings."""
        base_cfg = {
            "enabled": False,
            "warmup_range": (0, 0),
            "rollout_range": (0, 0),
            "sample_strategy": "argmax",
            "temperature": 1.0,
            "log_lengths": False,
        }

        if not cfg or not cfg.get("enabled", False):
            return base_cfg

        warmup_range = self._parse_length_range(cfg.get("warmup_range"), "warmup_range")
        rollout_range = self._parse_length_range(
            cfg.get("rollout_range"), "rollout_range"
        )

        strategy = cfg.get("sample_strategy", "argmax").lower()
        if strategy not in {"argmax", "sample"}:
            raise ValueError(
                "sample_strategy must be either 'argmax' or 'sample', "
                f"got '{strategy}'"
            )

        temperature = float(cfg.get("temperature", 1.0))
        if temperature <= 0:
            raise ValueError("temperature must be strictly positive.")

        base_cfg.update(
            {
                "enabled": True,
                "warmup_range": warmup_range,
                "rollout_range": rollout_range,
                "sample_strategy": strategy,
                "temperature": temperature,
                "log_lengths": bool(cfg.get("log_lengths", False)),
            }
        )
        return base_cfg

    @staticmethod
    def _parse_length_range(
        value: int | Sequence[int] | None, name: str
    ) -> tuple[int, int]:
        """Parse integer or [min, max] length spec."""
        if value is None:
            raise ValueError(f"{name} must be provided when enabling free-run.")

        if isinstance(value, int):
            low = high = int(value)
        elif isinstance(value, Sequence) and len(value) == 2:
            low, high = int(value[0]), int(value[1])
        else:
            raise ValueError(
                f"{name} must be an int or a [min, max] sequence; got {value}"
            )

        if low <= 0 or high <= 0:
            raise ValueError(f"{name} elements must be > 0, got {value}")
        if high < low:
            raise ValueError(f"{name} upper bound must be >= lower bound, got {value}")
        return low, high

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ):
        """K-step free-run training pass.

        Returns None to fall back to teacher forcing.
        """
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            return None

        inputs, targets = batch
        input_tensor, input_path = self._extract_primary_tensor(inputs)
        target_tensor, target_path = self._extract_primary_tensor(targets)

        if input_tensor is None or target_tensor is None:
            return None

        if input_tensor.shape[-1] != target_tensor.shape[-1]:
            return None

        seq_len = int(input_tensor.shape[-1])
        lengths = self._determine_rollout_lengths(seq_len)
        if lengths is None:
            return None
        warmup, rollout = lengths

        start_idx = self._sample_start_index(seq_len, warmup, rollout)
        if start_idx is None:
            return None

        if self.free_run_cfg.get("log_lengths", False):
            self.log(
                "free_run_warmup",
                warmup,
                on_step=True,
                prog_bar=False,
                batch_size=input_tensor.shape[0],
            )
            self.log(
                "free_run_rollout",
                rollout,
                on_step=True,
                prog_bar=False,
                batch_size=input_tensor.shape[0],
            )

        device = input_tensor.device
        context_tokens = input_tensor[..., start_idx : start_idx + warmup].clone()

        logits_steps: list[torch.Tensor] = []
        target_steps: list[torch.Tensor] = []

        curr_start = start_idx
        logits_time_dim = None
        target_time_dim = None
        input_seq_len = seq_len
        target_seq_len = int(target_tensor.shape[-1])

        for _ in range(rollout):
            ctx_inputs = self._slice_time_axis(
                inputs, curr_start, curr_start + warmup, input_seq_len
            )
            ctx_inputs = self._set_by_path(ctx_inputs, input_path, context_tokens)

            ctx_targets = self._slice_time_axis(
                targets, curr_start, curr_start + warmup, target_seq_len
            )

            logits = self.model(ctx_inputs)
            if not torch.is_tensor(logits):
                return None

            ctx_targets_tensor = self._get_from_path(ctx_targets, target_path)
            if self.postprocessor is not None and torch.is_tensor(ctx_targets_tensor):
                ctx_inputs_tensor = self._get_from_path(ctx_inputs, input_path)
                (
                    ctx_inputs_tensor,
                    logits,
                    ctx_targets_tensor,
                ) = self.postprocessor(ctx_inputs_tensor, logits, ctx_targets_tensor)
                ctx_inputs = ctx_inputs_tensor

            if logits_time_dim is None:
                logits_time_dim = logits.dim() - 2
            if target_time_dim is None:
                target_time_dim = min(logits_time_dim, ctx_targets_tensor.dim() - 1)

            last_idx = logits.shape[logits_time_dim] - 1
            step_logits = logits.select(logits_time_dim, last_idx)
            logits_steps.append(step_logits.unsqueeze(logits_time_dim))

            last_target = ctx_targets_tensor.select(
                target_time_dim, ctx_targets_tensor.shape[target_time_dim] - 1
            )
            target_steps.append(last_target.unsqueeze(target_time_dim))

            next_token = self._sample_next_token(
                step_logits, dtype=context_tokens.dtype
            ).to(device)
            next_token = next_token.unsqueeze(-1)

            if context_tokens.shape[-1] == 1:
                context_tokens = next_token
            else:
                context_tokens = torch.cat(
                    [context_tokens[..., 1:], next_token], dim=-1
                )
            curr_start += 1

        if not logits_steps or logits_time_dim is None or target_time_dim is None:
            return None

        rollout_logits = torch.cat(logits_steps, dim=logits_time_dim)
        rollout_targets_tensor = torch.cat(target_steps, dim=target_time_dim)
        rollout_targets = self._set_by_path(
            targets, target_path, rollout_targets_tensor
        )

        loss = self.loss(rollout_logits, rollout_targets, model=self.model)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        metrics_for_stage: dict[str, torch.Tensor] = {}
        for name, metric in self.loss.metrics.items():
            metric_val = metric(rollout_logits, rollout_targets)
            metrics_for_stage[name] = metric_val
            self.log(
                f"{stage}_{name}",
                metric_val,
                prog_bar=True,
            )

        lr = None
        if stage == "train":
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr)
        return loss

    def _extract_primary_tensor(
        self, data: Any
    ) -> tuple[torch.Tensor | None, tuple[Any, ...]]:
        if torch.is_tensor(data):
            return data, tuple()
        path = self._find_tensor_path(data)
        if path is None:
            return None, tuple()
        tensor = self._get_from_path(data, path)
        if torch.is_tensor(tensor):
            return tensor, path
        return None, tuple()

    def _find_tensor_path(
        self, data: Any, prefix: tuple[Any, ...] = tuple()
    ) -> tuple[Any, ...] | None:
        if torch.is_tensor(data):
            return prefix
        if isinstance(data, (list, tuple)):
            for idx, item in enumerate(data):
                path = self._find_tensor_path(item, prefix + (idx,))
                if path is not None:
                    return path
        elif isinstance(data, dict):
            for key in data:
                path = self._find_tensor_path(data[key], prefix + (key,))
                if path is not None:
                    return path
        return None

    def _get_from_path(self, data: Any, path: tuple[Any, ...]) -> Any:
        if not path:
            return data
        key = path[0]
        rest = path[1:]
        if isinstance(data, (list, tuple)):
            return self._get_from_path(data[key], rest)
        if isinstance(data, dict):
            return self._get_from_path(data[key], rest)
        raise TypeError(f"Unsupported container type {type(data)} for key {key}")

    def _set_by_path(self, data: Any, path: tuple[Any, ...], value: Any) -> Any:
        if not path:
            return value
        key = path[0]
        rest = path[1:]
        if isinstance(data, tuple):
            data_list = list(data)
            data_list[key] = self._set_by_path(data_list[key], rest, value)
            return type(data)(data_list)
        if isinstance(data, list):
            data_list = list(data)
            data_list[key] = self._set_by_path(data_list[key], rest, value)
            return data_list
        if isinstance(data, dict):
            new_data = dict(data)
            new_data[key] = self._set_by_path(new_data[key], rest, value)
            return new_data
        raise TypeError(f"Unsupported container type {type(data)} for key {key}")

    def _slice_time_axis(self, data: Any, start: int, end: int, seq_len: int) -> Any:
        if torch.is_tensor(data):
            if data.shape[-1] == seq_len:
                return data[..., start:end]
            return data
        if isinstance(data, tuple):
            return tuple(
                self._slice_time_axis(item, start, end, seq_len) for item in data
            )
        if isinstance(data, list):
            return [self._slice_time_axis(item, start, end, seq_len) for item in data]
        if isinstance(data, dict):
            return {
                key: self._slice_time_axis(val, start, end, seq_len)
                for key, val in data.items()
            }
        return data

    def _determine_rollout_lengths(self, seq_len: int) -> tuple[int, int] | None:
        if seq_len < 2:
            return None
        full_len = seq_len + 1
        warmup = min(
            self._sample_length(self.free_run_cfg["warmup_range"]), full_len - 1
        )
        warmup = max(1, warmup)
        future_capacity = full_len - warmup
        if future_capacity <= 0:
            return None
        rollout = min(
            self._sample_length(self.free_run_cfg["rollout_range"]),
            future_capacity,
        )
        rollout = max(1, rollout)
        return warmup, rollout

    def _sample_length(self, bounds: tuple[int, int]) -> int:
        low, high = bounds
        if low == high:
            return low
        device = getattr(self, "device", torch.device("cpu"))
        return int(torch.randint(low, high + 1, (1,), device=device).item())

    def _sample_start_index(
        self, seq_len: int, warmup: int, rollout: int
    ) -> int | None:
        full_len = seq_len + 1
        max_start = full_len - (warmup + rollout)
        if max_start < 0:
            return None
        if max_start == 0:
            return 0
        device = getattr(self, "device", torch.device("cpu"))
        return int(torch.randint(0, max_start + 1, (1,), device=device).item())

    def _sample_next_token(
        self, logits: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        strategy = self.free_run_cfg.get("sample_strategy", "argmax")
        temperature = float(self.free_run_cfg.get("temperature", 1.0))

        with torch.no_grad():
            if strategy == "argmax":
                next_token = torch.argmax(logits, dim=-1)
            else:
                scaled = logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                flat = probs.reshape(-1, probs.shape[-1])
                sampled = torch.multinomial(flat, num_samples=1).view(probs.shape[:-1])
                next_token = sampled
        return next_token.to(dtype=dtype)

    def _log_teacher_forced_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> None:
        inputs, targets = batch
        with torch.no_grad():
            logits = self.model(inputs)
            if not torch.is_tensor(logits):
                return
        if self.postprocessor is not None:
            inputs, logits, targets = self.postprocessor(inputs, logits, targets)
        self._log_random_samples(inputs, logits, targets, stage)


class DatasetEpochCallback(pl.Callback):
    """Calls a dataset-provided epoch hook at train epoch boundaries.

    Supports datasets that expose either `set_epoch(int)` for cross-worker coordination
    via shared state, or a simpler `on_epoch_start(epoch)` method.
    """

    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # 0-based current epoch index in PL
        epoch = int(getattr(trainer, "current_epoch", 0))
        hook = getattr(self.dataset, "set_epoch", None)
        if callable(hook):
            hook(epoch)
        hook2 = getattr(self.dataset, "on_epoch_start", None)
        if callable(hook2):
            hook2(epoch)

        # Optional: print a fingerprint for debugging multi-worker consistency
        fp = getattr(self.dataset, "epoch_fingerprint", None)
        if callable(fp):
            try:
                print(f"[Dataset] epoch={epoch} fingerprint={fp()}")
            except Exception:
                pass
