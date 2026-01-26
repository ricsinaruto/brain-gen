# FlatGPT full-paper configs

This folder mirrors the full-scale preprocessing, tokenizer, training, and eval
configs used in the paper. The YAMLs are copied from `configs/` and only
rewired so internal `model_config` / analysis references point back into this
folder. All dataset paths, cache paths, and checkpoint paths are placeholders
and must be edited for your environment.

## Structure

- `configs/preproc/<dataset>/`: full preprocessing configs (cloud + OSL)
- `configs/braintokmix/`: BrainTokMix tokenizer model + training config
- `configs/flatgpt/flat/`: FlatGPT model + training + eval configs
- `configs/analyses/`: rollout analysis configs
- `scripts/`: convenience launchers

## Before you run

Update these fields to your paths:
- `data_path`, `stage1_path`, `save_path`, `log_dir` (preproc)
- `dataset_root`, `cache_dir` (tokenizer/train/eval)
- `save_dir`, `resume_from`, `ckpt_path` (training/eval)

## Quick start (run from repo root)

1) Preprocess full datasets:

```
conda run -n brain-gen bash examples/flatgpt/05_full_paper/scripts/run_preprocess_omega.sh
conda run -n brain-gen bash examples/flatgpt/05_full_paper/scripts/run_preprocess_camcan.sh
conda run -n brain-gen bash examples/flatgpt/05_full_paper/scripts/run_preprocess_mous.sh
```

2) Train the BrainTokMix tokenizer:

```
conda run -n brain-gen bash examples/flatgpt/05_full_paper/scripts/run_braintokmix_full.sh
```

3) Train FlatGPT:

```
conda run -n brain-gen bash examples/flatgpt/05_full_paper/scripts/run_flatgpt_full.sh
```

4) Run evals (auditory / visual / rest):

```
conda run -n brain-gen bash examples/flatgpt/05_full_paper/scripts/run_eval_aud.sh
conda run -n brain-gen bash examples/flatgpt/05_full_paper/scripts/run_eval_vis.sh
conda run -n brain-gen bash examples/flatgpt/05_full_paper/scripts/run_eval_rest.sh
```

If you prefer to run directly without scripts, use `preprocess.py`, `run.py`,
and `evals.py` with the configs under `examples/flatgpt/05_full_paper/configs/`.
