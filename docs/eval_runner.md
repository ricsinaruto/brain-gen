# Eval runner configuration

`brain_gen.eval.eval_runner.EvaluationRunner` evaluates a checkpoint (or a
TimesFM model) and produces generation‑based analyses and optional metric
summaries. It is used by:

- `evals.py` for checkpoint sweeps
- `python -m brain_gen.eval.eval_runner` for single‑checkpoint evaluation

The runner always executes generation/analysis first (if `forecast()` is
available), then computes losses/metrics over a limited number of batches.

## Quick start

```yaml
save_dir: /path/to/trainings/run
model_name: FlatGPTEmbedsRVQ
loss_name: CrossEntropyWithCodes
model_config: configs/flatgpt/flat/model_chmix.yaml

# ... datasplitter/dataloader/trainer ...

eval_runner:
  ckpt_path: /path/to/checkpoints/last-checkpoint-epoch00010.ckpt
  metrics_split: test
  max_batches: 10

  example_sampler:
    split: test
    num_sessions: 8
    context_length_s: 10.24
    total_length_s: 20.48

  generator:
    rollouts_per_context: 1
    rollout_batch_size: 4
    kv_overlap: 0.5
    sampling:
      strategy: top_p
      top_p: 0.9
      temperature: 1.0

  analyses:
    - class: RolloutSlidingWindowAnalysis
      config: configs/flatgpt/analyses/sliding_window_full.yaml
    - class: RolloutDivergenceAnalysis
      config: configs/flatgpt/analyses/rollout_divergence_full.yaml
```

## Output locations
If `eval_runner.output_dir` is set, outputs are written there.
Otherwise, the runner writes to:

```
<save_dir>/logs/version_<eval_runner.version>/<ckpt_stem or epoch_XXX>
```

If `eval_runner.version` is omitted, the default directory becomes
`version_None`. For clean runs, set `version` or `output_dir` explicitly.

Outputs include:
- `eval_config.yaml`
- `summary.json`
- `metrics_distributions.png`
- `example_<idx>.png`
- `examples_psd_summary.png`
- `examples_cov_summary.png`
- `gen_vs_target_*` time‑series / STFT panels (if generation ran)
- `rollout_divergence.*` and `rollout_window_metrics.*` (if analyses configured)
- `token_summary.*` (if token summary enabled)

## Core settings (`eval_runner`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `ckpt_path` | str | required | Checkpoint path. Required unless `timesfm` is set. |
| `lit_module` | str \| null | `null` | Optional Lightning module class name to load instead of `LitModel`. |
| `compile` | bool | `true` | Compile model with `torch.compile` before eval (when supported). |
| `device` | str \| null | `null` | Override device used by the eval runner. |
| `metrics_split` | str | `"val"` | Which split to use for loss/metric aggregation (`val` or `test`). |
| `max_batches` | int | `2` | Max number of metric batches to evaluate. |
| `num_examples` | int | `2` | Number of example batches to plot. |
| `version` | int \| null | `null` | Used in the output directory path. |
| `step` / `epoch` | int \| null | `null` | Stored in `summary.json` and affects output naming. |
| `output_dir` | str \| null | `null` | Optional override for output directory (alias: `out_dir`). |
| `timesfm` | dict \| null | `null` | Enable TimesFM generation‑only evaluation (see below). |
| `checkpoint_wait_timeout_s` | int | `300` | Max wait for a checkpoint to appear. |
| `checkpoint_stable_seconds` | float | `5` | Time size/mtime must stay fixed before reading. |
| `checkpoint_poll_seconds` | float | `1.0` | Poll interval when waiting for a checkpoint. |
| `checkpoint_load_retries` | int | `3` | Retry count if checkpoint load fails. |
| `checkpoint_retry_wait_s` | float | `5.0` | Wait between load retries. |

Note: `eval_runner.enabled` is respected by the training‑time eval launcher; the
standalone eval runner will execute regardless of this flag.

## Example sampler (`eval_runner.example_sampler`)
The sampler **reads only the first chunk** of each session. It slices a fixed
context + continuation window from that chunk.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `seed` | int \| null | `null` | RNG seed for session selection. |
| `split` | str | `"val"` | Which split to sample (`val` or `test`). |
| `dataset_key` | str \| list[str] \| null | `null` | Filter by dataset key (matches datasplitter mapping). Aliases: `dataset_keys`, `dataset`. |
| `task_type` | str \| list[str] \| null | `null` | Filter sessions by substring match in folder name (e.g., `rest`, `auditory`). |
| `num_sessions` | int | `1` | Number of sessions to sample. |
| `context_length_s` / `context_length_steps` | float \| int | required | Context length. |
| `total_length_s` / `total_length_steps` | float \| int | required | Total length (context + rollout). Must exceed context length. |

## Generator (`eval_runner.generator`)
Generation is skipped if the model does not implement `forecast()`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Turn generation on/off. |
| `paired_file` | str | `paired_rollouts.npy` | Cache filename written under the eval output dir. |
| `rollouts_per_context` | int | `1` | Number of rollouts per context. |
| `rollout_batch_size` | int | `1` | Batch size for generation (grouped by input signature). |
| `seed` | int \| null | `null` | RNG seed for token sampling and plot selection. |
| `max_plot_seconds` | float \| null | `null` | Optional cap on plotted rollout duration (seconds). |
| `kv_overlap` | float \| int | `0.5` | Passed to `forecast(..., sliding_window_overlap=...)`. Float = fraction of window; int = token stride. |
| `max_context_tokens` | int | `-1` | Passed to `forecast(..., max_context_tokens=...)`. Use `-1`/`null` for full context. |
| `debug_timing` | bool | `false` | Forwarded to `forecast()` if the model accepts it. |
| `mu` | int \| null | `null` | If set, inverse‑mu‑law decode generated/target signals before plotting. |

### Sampling parameters
Sampling can be specified either at the generator level or under a `sampling`
sub‑block. The following keys are supported:

| Key | Description |
| --- | --- |
| `strategy` | `argmax`, `roulette`, `top_k`, or `top_p` |
| `temperature` | Sampling temperature (float > 0) |
| `top_k` | Top‑k cutoff for `strategy: top_k` |
| `top_p` | Top‑p cutoff for `strategy: top_p` |

### Per‑RVQ temperature curricula
For RVQ‑based models (e.g., `FlatGPTEmbedsRVQ`), you can enable per‑level
temperatures:
- `temperature_curriculum: true` to enable default decay
- `temperature_levels` / `temperature_per_level`: explicit per‑level list
- `temperature_decay`: multiplicative decay across RVQ levels
- `temperature_min`: clamp minimum temperature

## Analyses (`eval_runner.analyses`)
Analyses are optional and independent of generation. Each entry specifies a class
and a config file passed verbatim to the analysis class.

```yaml
eval_runner:
  analyses:
    - class: RolloutDivergenceAnalysis
      config: configs/flatgpt/analyses/rollout_divergence_full.yaml
    - class: RolloutSlidingWindowAnalysis
      config: configs/flatgpt/analyses/sliding_window_full.yaml
```

Built‑in classes:
- `RolloutDivergenceAnalysis`
- `RolloutSlidingWindowAnalysis`

## Token summary (`eval_runner.token_summary`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Turn token summary plotting on/off. |
| `tokens_per_second` / `tokens_per_sec` | float \| null | `null` | Enables `bits_per_second` if set. |
| `tokens_per_step` | int \| null | inferred | Explicit tokens‑per‑timestep grouping. |
| `max_context_tokens` | int \| null | `null` | Cap plotted context length. |

Metrics include `bits_per_token`, `perplexity`, `bits_per_second` (if enabled),
and `decoded_mse` when decoding is possible.

