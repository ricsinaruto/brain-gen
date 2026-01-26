# brain-gen

Neural signal modeling toolkit for MEG/ephys research. The repo bundles
preprocessing, dataset utilities, tokenizers, autoregressive/diffusion/conv
models, and PyTorch Lightning training + evaluation workflows.

## What’s inside
- End-to-end pipeline: raw MEG → cleaned chunks → tokenized codes → model training →
  rollout evaluation.
- Tokenizers: BrainOmni causal tokenizer, BrainTokMix channel-mixing tokenizer,
  VidTok (FSQ/RVQ), plus lightweight flat tokenizers.
- Models: FlatGPT (Qwen/SmoLLM adapters), STGPT2MEG, Wavenet, BENDR, NTD diffusion,
  TASA3D, baselines/classifiers.
- Evaluation stack: rollout generation, divergence + sliding-window analyses,
  token summaries, and publication-ready plots.

## Install
- Python 3.13 (see `setup.py`).
- Create an environment, then install editable:

```bash
conda create -n ephys-gpt python=3.13
conda run -n ephys-gpt pip install -e .
```

Preprocessing depends on `osl-ephys` + MNE (both listed in
`requirements.txt`). Note that `osl-ephys` needs to be installed from source: https://github.com/OHBA-analysis/osl-ephys/

## Entry points
- `preprocess.py`: dataset-specific preprocessing pipelines (Omega, MOUS, CamCAN).
- `run.py`: training/testing/tokenizer entry point.
- `evals.py`: evaluation runner for checkpoint sweeps.

## Configuration quickstart
Training/eval configs are YAML. A minimal skeleton looks like:

```yaml
save_dir: /path/to/trainings/run
resume_from: null

model_name: FlatGPTEmbedsRVQ
loss_name: CrossEntropyWithCodes
model_config: configs/flatgpt/flat/model_chmix.yaml

# optional:
loss: {}
experiment_name: logs

datasplitter:
  dataset_class: ChunkDatasetReconstruction
  dataset_root:
    omega: /path/to/datasets/omega/cleaned
  example_seconds: 10.24
  overlap_seconds: 0.0

# Data loading
# Optional: set dataloader_class to MixupDataLoader or TextDataLoader.
dataloader:
  batch_size: 4
  num_workers: 8

# Optim + Lightning settings
lightning:
  lr: 2.0e-4
  weight_decay: 0.1

trainer:
  accelerator: cuda
  max_epochs: 100

# Optional: evaluation runner (see docs/eval_runner.md)
# eval_runner:
#   enabled: true
```

Model/loss names must match keys in `brain_gen/training/utils.py`. Current
`model_name` options include: `STGPT2MEG`, `FlatGPT*`, `BrainOmniCausalTokenizer`,
`BrainOmniCausalTokenizerSEANetChannelMix` (BrainTokMix),
`BrainOmniCausalForecast`, `Vidtok`, `VidtokRVQ`, `NTD`, `WavenetFullChannel`,
`Wavenet3D`, `TASA3D`, and classifier variants. Loss names include
`CrossEntropyWithCodes`, `BrainOmniCausalTokenizerLoss`, `VidtokLoss`, `NTDLoss`,
and reconstruction/classification losses.

## Pipelines
### Preprocessing (`preprocess.py`)
Stage-1 uses OSL-ephys; stages 2/3 handle cleaning, chunking, normalization,
quantization, and optional text/conditioning outputs.

```bash
conda run -n ephys-gpt python preprocess.py \
  --dataset <omega|mous|mous_conditioned|camcan|camcan_conditioned> \
  --stage <stage_1|stage_2|stage_3|both|all> \
  --args configs/preproc/<dataset>/osl.yaml
```

Outputs are written to the `save_path` configured in the YAML, with stage‑1 `.fif`
outputs going to `stage1_path`.

### Training / testing / tokenizers (`run.py`)
`run.py` dispatches to Lightning experiment wrappers:

```bash
conda run -n ephys-gpt python run.py --mode train --args configs/stgpt2meg/train_omega.yaml
conda run -n ephys-gpt python run.py --mode test  --args configs/stgpt2meg/train_omega.yaml
conda run -n ephys-gpt python run.py --mode train --args configs/brainomni/all_dataset/train_ds.yaml
conda run -n ephys-gpt python run.py --mode vidtok --args configs/vidtok/train.yaml
```

Notes:
- BrainOmni/BrainTokMix tokenizer training uses `run.py --mode train` with
  `model_name: BrainOmniCausalTokenizer` or
  `BrainOmniCausalTokenizerSEANetChannelMix`.
- `run.py --mode tokenizer-text` trains a BPE tokenizer on stage‑3 text chunks.
  Provide a `tokenizer` block (see `brain_gen/training/train_bpe.py`).

### Evaluation (`evals.py`)
`evals.py` runs `brain_gen.eval.eval_runner.EvaluationRunner` on a config:

```bash
conda run -n ephys-gpt python evals.py --args configs/flatgpt/flat/test_rest.yaml
```


## Examples and paper reproduction
The `examples/flatgpt/` tree mirrors the paper pipeline.
Small‑scale configs run on CPU.
Full‑scale configs under `examples/flatgpt/05_full_paper/` mirror the paper runs.

### Example → paper mapping

| Example | Purpose | Paper outputs it reproduces |
| --- | --- | --- |
| `examples/flatgpt/01_preprocess` | Stage‑1/2 preprocessing | Data cleaning + chunking pipeline used in all experiments |
| `examples/flatgpt/02_tokenizer` | BrainTokMix tokenizer training | Tokenizer/codebook used by FlatGPT models |
| `examples/flatgpt/03_training` | FlatGPT training | Main model training setup |
| `examples/flatgpt/04_eval` | Rollout generation + analyses | Evaluation figures: rollout divergence curves, sliding‑window metrics, token summary, PSD/covariance summaries, and qualitative time‑series panels |
| `examples/flatgpt/05_full_paper` | Full‑scale configs and scripts | Full‑paper training/eval settings (rest/auditory/visual); produces the same families of plots as the paper |

The evaluation pipeline writes:
- `rollout_divergence.png` + `rollout_divergence.json`
- `rollout_window_metrics.png` + `rollout_window_metrics.json`
- `token_summary.png` + `token_summary.json`
- `examples_psd_summary.png`, `examples_cov_summary.png`
- `gen_vs_target_*` time‑series / STFT panels

These are the exact plot families used in the paper’s evaluation section. Use the
`examples/flatgpt/05_full_paper/configs/flatgpt/flat/test_{rest,aud,vis}.yaml` configs
to reproduce the three evaluation conditions.

## Repository layout
- `preprocess.py` – dataset preprocessing CLI.
- `run.py` – training/testing/tokenizer/vidtok driver.
- `evals.py` – evaluation runner.
- `brain_gen/preprocessing` – dataset‑specific pipelines + Maxwell helpers.
- `brain_gen/dataset` – datasplitter, datasets, dataloaders, tokenized dataset helpers.
- `brain_gen/models` – model zoo + HF adapters (FlatGPT, BrainOmni, NTD, etc.).
- `brain_gen/eval` – rollout generator, analyses, plotting, token summary.
- `configs/` – canonical configs for preprocessing, training, and evals.
- `examples/` – small‑scale and full‑paper pipelines.
- `scripts/` – utility scripts (e.g., FlatGPT pretokenization).
- `tests/` – CPU‑friendly tests for datasets, models, and eval stack.

## Tests
Run only the relevant tests for a change:

```bash
conda run -n ephys-gpt pytest -q tests/<test_file_relevant_to_change>
```

The full suite is also CPU‑friendly:

```bash
conda run -n ephys-gpt pytest -q
```
