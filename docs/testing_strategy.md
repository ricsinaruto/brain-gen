# Testing strategy

The repository ships with a lightweight CPU‑friendly test suite under `tests/`.
It focuses on shape/causality checks, dataset splitting, tokenizer integrity,
and evaluation utilities.

## Goals
- Validate dataset splitting and channel alignment logic with synthetic `.npy`
  chunks.
- Exercise forward passes of key models (STGPT2MEG, FlatGPT variants, BrainOmni
  tokenizer/forecaster, NTD, Wavenet, TASA3D, Vidtok, classifiers) to ensure
  tensor shapes and causal masking are respected.
- Guard against information leakage with gradient‑causality checks
  (`tests.models.utils.assert_future_grad_zero`).
- Verify evaluation‑stack utilities (rollout metrics, eval runner helpers,
  TimesFM wrapper) on small synthetic inputs.

## How to run
```bash
conda run -n ephys-gpt pytest -q
```

To target a specific test file:
```bash
conda run -n ephys-gpt pytest -q tests/<test_file>
```

## What to watch for
- Causal attention/convolution layers should zero gradients for future
  timesteps.
- Dataset tests expect small synthetic shapes; new datasets should keep fixtures
  CPU‑friendly.
- Tokenizer tests verify encode/decode consistency and causal constraints.
- Eval runner tests assume `forecast()` exists on models used for generation.
