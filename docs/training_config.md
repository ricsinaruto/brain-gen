# Training configuration (dataloader, lightning, trainer)

This page summarizes the `dataloader`, `lightning`, and `trainer` config
sections used by `brain_gen.training.train.ExperimentDL` and
`brain_gen.training.lightning.LitModel`.

## Top‑level context
A typical training config includes:
- `save_dir` and `resume_from`
- `model_name`, `loss_name`, `model_config`
- `datasplitter` and `dataloader` sections
- `lightning` and `trainer` sections
- Optional `eval_runner` block (for automated checkpoint evals)

## Dataloader class (`dataloader_class`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `dataloader_class` | str | `"DataLoader"` | Class name resolved from `brain_gen.training.train` globals. Common values: `DataLoader`, `MixupDataLoader`, `TextDataLoader`. |

## Dataloader (`dataloader`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `batch_size` | int | `1` | Batch size for train/val dataloaders. |
| `num_workers` | int | `0` | Worker processes for data loading. |
| `prefetch_factor` | int | `2` | Prefetch batches per worker (when `num_workers > 0`). |
| `pin_memory` | bool | `false` | Pin CPU memory (recommended for GPU training). |
| `persistent_workers` | bool | `false` | Keep workers alive between epochs (requires `num_workers > 0`). |
| `drop_last` | bool | `false` | Drop the final incomplete batch. |

### `MixupDataLoader` extras
When `dataloader_class: MixupDataLoader` is used:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `mixup_alpha` | float | `0.0` | MixUp Beta distribution parameter. |
| `mixup_prob` | float | `1.0` | Probability of applying MixUp per batch. |
| `num_classes` | int \| null | `null` | Required for integer targets (one‑hot mixing). |
| `quantize` | bool | `false` | Apply mu‑law quantization to mixed inputs. |
| `mu` | int | `255` | Mu‑law parameter (when `quantize` is true). |
| `max_val` | float | `10.0` | Max value for quantization scaling. |

### `TextDataLoader` extras
When `dataloader_class: TextDataLoader` is used:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `tokenizer_path` | str | required | Path to a trained tokenizer directory containing `tokenizer/`. |
| `max_length` | int \| null | `null` | Max sequence length before truncation. |
| `use_separator` | bool | `true` | Insert separator tokens between channel groups. |

## Lightning (`lightning`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `lr` | float | required | Base learning rate for AdamW. |
| `weight_decay` | float | required | AdamW weight decay. |
| `resume_lr` | float \| null | `null` | Override optimizer LR after resuming; resets scheduler base LRs. |
| `resume_weight_decay` | float \| null | `null` | Override weight decay after resuming. |
| `resume_context` | dict \| null | `null` | Resize context/rope on resume (requires model `resize_context`). Keys: `input_shape`, `spatial_reduction`, `temporal_reduction`, `rope_theta`, `max_position_embeddings`. |
| `resume_token_corruption` | dict \| bool \| null | `null` | Override token corruption on resume. Set `true` to reuse model config. |
| `compile` | bool | `false` | Use `torch.compile` on the model. |
| `benchmark_train_step` | bool | `false` | Log `perf/train_step_ms` per step (adds GPU sync). |
| `no_decay_verbose` | bool | `false` | Print params assigned to the no‑decay optimizer group. |
| `grad_clip_threshold` | float | `1.0` | Threshold used for `grad_clip_pct` logging. |
| `lr_scheduler` | dict \| null | `null` | Optional scheduler config (see below). |
| `lr_warmup` | int \| dict \| null | `null` | Optional linear LR warmup (see below). |

### `lr_scheduler` sub‑keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `class_name` | str | required | Scheduler class name from `torch.optim.lr_scheduler`. |
| `interval` | str | `"epoch"` | Lightning scheduler interval (`"epoch"` or `"step"`). |
| `frequency` | int | `1` | How often to step the scheduler. |
| `monitor` | str \| null | `null` | Metric name for schedulers that need it. |
| *(other keys)* | any | — | Passed directly to the scheduler constructor. |

### `lr_warmup` options
`lr_warmup` can be an integer (number of warmup steps) or a dict:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `steps` | int | required | Number of scheduler steps/epochs to linearly warm the LR. |
| `interval` | str | inherited | Defaults to scheduler interval if set; otherwise `"step"`. |

Warmup uses `SequentialLR` and does not support `ReduceLROnPlateau`.

## Trainer (`trainer`)

`trainer` is passed into `pytorch_lightning.Trainer`, with a few custom keys
handled by `ExperimentDL` before construction.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `tune_batch_size` | bool | `true` | Adds a `BatchSizeFinder` callback. |
| `early_stopping` | bool \| int \| dict \| null | `null` | Enable/configure early stopping (see below). |
| `checkpoint_cadence_epochs` | int \| null | `null` | Save `last-checkpoint` every N epochs (1‑based). |
| `validate_before_resume` | bool | `false` | Run a validation pass before resuming. |
| `callbacks` | list | `[]` | Extra Lightning callbacks to append. |
| *(PL Trainer args)* | any | PL defaults | `max_epochs`, `accelerator`, `devices`, `precision`, `gradient_clip_val`, etc. |

### `early_stopping` sub‑keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `monitor` | str | `"val/loss"` | Metric to monitor. |
| `mode` | str | `"min"` | `"min"` or `"max"`. |
| `patience` | int | `10` | Epochs to wait before stopping. |
| `min_delta` | float | `0.0` | Minimum change to qualify as improvement. |
| `check_on_train_epoch_end` | bool | `false` | Evaluate early stopping on train epoch end. |
| `stopping_threshold` | float | — | Optional Lightning EarlyStopping arg. |
| `divergence_threshold` | float | — | Optional Lightning EarlyStopping arg. |
| `strict` | bool | — | Optional Lightning EarlyStopping arg. |
| `verbose` | bool | — | Optional Lightning EarlyStopping arg. |
| `check_finite` | bool | — | Optional Lightning EarlyStopping arg. |
| `log_rank_zero_only` | bool | — | Optional Lightning EarlyStopping arg. |

## Checkpointing and logging behavior
- **Checkpoints**: `best-checkpoint` (monitors `val/loss`) and `last-checkpoint`
  are written via `ThreadedModelCheckpoint`. `checkpoint_cadence_epochs` throttles
  the `last-checkpoint` writes.
- **Logging**: a `TensorBoardLogger` is always attached under `save_dir/logs/`.
- **Optimizer scalars**: `lr` and `weight_decay` are logged every train step.
- **Performance metrics**: `PerformanceMonitor` logs `perf/*` throughput metrics.
- **Grad‑norm metrics**: `grad_norm` and `grad_clip_pct` are logged per step.

## Minimal example

```yaml
dataloader:
  batch_size: 4
  num_workers: 16
  prefetch_factor: 4
  pin_memory: true
  persistent_workers: true

lightning:
  lr: 2.0e-4
  weight_decay: 0.1
  compile: false
  lr_warmup:
    steps: 1000
    interval: step
  lr_scheduler:
    class_name: CosineAnnealingLR
    eta_min: 1.0e-5
    T_max: 100

trainer:
  max_epochs: 100
  accelerator: cuda
  precision: bf16-mixed
  gradient_clip_val: 1.0
  log_every_n_steps: 100
  early_stopping:
    monitor: val/loss
    patience: 5
```
