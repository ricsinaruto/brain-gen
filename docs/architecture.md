# Architecture overview

This repository supports an end‑to‑end MEG/ephys modeling workflow: preprocessing
raw recordings into chunked tensors, splitting them into datasets with a
canonical sensor layout, training tokenizers and sequence models, and evaluating
rollouts with specialized analyses.

## Entrypoints
- `preprocess.py` – dataset‑specific preprocessing (Omega, MOUS, CamCAN; conditioned
  variants supported). Stage‑1 runs OSL‑ephys; stages 2/3 handle cleaning,
  chunking, normalization/quantization, and optional text/conditioning outputs.
- `run.py` – training and testing driver. Supported modes: `train`, `test`,
  `tokenizer`, `tokenizer-text`, `vidtok`.
- `evals.py` – runs `brain_gen.eval.eval_runner.EvaluationRunner` for checkpoint
  sweeps.
- `python -m brain_gen.eval.eval_runner` – evaluate a single config + checkpoint.

## Data flow
1) **Preprocessing** writes per‑session `.npy` chunks containing `data`, `ch_names`,
   `pos_2d`, `sfreq` (optional `pos_3d`, `ori_3d`).
2) **Datasplitter** (`brain_gen.dataset.datasplitter`) scans one or more dataset
   roots, merges channel layouts into a canonical ordering, and produces window
   indices for train/val/test splits.
3) **Datasets** (`brain_gen.dataset.datasets`) map each window to a tensor in the
   canonical layout. Reconstruction datasets return tuples like `(x, pos, ch_type)`
   for tokenizer‑aware models; image datasets return sparse H×W grids.
4) **Training** (`brain_gen.training`) builds dataloaders and Lightning modules;
   optional eval hooks can launch checkpoint evaluations.
5) **Evaluation** (`brain_gen.eval`) samples session segments, generates rollouts
   with `model.forecast(...)` (when available), and runs divergence + sliding‑window
   analyses to produce publication‑ready plots.

## Module map
### `brain_gen.preprocessing`
Dataset‑specific pipelines (Omega, MOUS, CamCAN, text). Stage‑1 uses OSL‑ephys and
Maxwell filtering; stage‑2 performs chunking, normalization, and quantization;
stage‑3 optionally produces text/conditioned outputs.

### `brain_gen.dataset`
- `datasplitter.py`: builds train/val/test splits and canonical channel layouts;
  supports session/subject/dataset split strategies and caching.
- `datasets.py`: core datasets (`ChunkDataset`, reconstruction/forecast/image
  variants, `ChunkDatasetSensor3D`, `BPEDataset`).
- `dataloaders.py`: `MixupDataLoader` and `TextDataLoader` plus collate helpers.

### `brain_gen.models`
Model zoo covering autoregressive transformers, convolutional baselines,
spatiotemporal attention, diffusion, and tokenizers. HF adapters for Qwen video
backbones live under `models/hf_adapters`.

### `brain_gen.training`
- `train.py`: `ExperimentDL`, `ExperimentTokenizer`, `ExperimentTokenizerText`, and
  `ExperimentVidtok` wrappers.
- `lightning.py`: `LitModel` and helpers (resume‑LR/weight‑decay overrides, optional
  context resize, grad‑norm logging).
- `checkpointing.py`: threaded checkpointing + optional eval dispatch.

### `brain_gen.eval`
- `session_sampler.py`: samples the **first chunk** of each session for evaluation.
- `generation.py`: rollout generation and plot helpers.
- `rollout_divergence.py` / `rollout_sliding_windows.py`: publication‑ready analyses.
- `token_summary.py`: token‑level perplexity/bitrate/MSE summaries.

### `brain_gen.utils`
Quantizers (mu‑law, RVQ helpers), plotting utilities, session cleaning, and
sampling helpers.

## Optional utilities
- `scripts/pretokenize_flatgpt_dataset.py`: precompute tokenizer codes for
  FlatGPT training to avoid per‑batch tokenization.
