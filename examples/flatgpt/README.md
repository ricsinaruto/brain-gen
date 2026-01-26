# FlatGPT end-to-end examples

These examples mirror the main pipeline in the paper (Sections 4.1-4.6):
preprocessing, BrainTokMix tokenization, FlatGPT training, and evaluation. The
configs here are scaled down to run quickly on CPU with the bundled small data
under `data/<dataset>/small`.

## Quick start (run from repo root)

1) Preprocess Omega, CamCAN, and MOUS (stage 1 + stage 2):

```
conda run -n brain-gen bash examples/flatgpt/01_preprocess/scripts/run_omega_small.sh
conda run -n brain-gen bash examples/flatgpt/01_preprocess/scripts/run_camcan_small.sh
conda run -n brain-gen bash examples/flatgpt/01_preprocess/scripts/run_mous_small.sh
```

2) Train the BrainTokMix tokenizer:

```
conda run -n brain-gen bash examples/flatgpt/02_tokenizer/scripts/run_braintokmix_small.sh
```

3) Train FlatGPT on the tokenizer codes:

```
conda run -n brain-gen bash examples/flatgpt/03_training/scripts/run_flatgpt_small.sh
```

4) Run evaluation (rollout generation + analyses):

```
conda run -n brain-gen bash examples/flatgpt/04_eval/scripts/run_eval_small.sh
```

## Notes

- Outputs are written under `tmp/examples/flatgpt/` by default.
- All paths in the small configs are relative to the repo root. If you keep data in `data/<dataset>/small`, no edits are required.
- To point at different locations, update:
  - Preprocess configs (`examples/flatgpt/01_preprocess/configs/*_small.yaml`):
    `data_path` (raw data), `stage1_path` (stage-1 outputs), `save_path`
    (stage-2 outputs), and `log_dir` (logs). Keep `stage1_path` and `save_path`
    consistent across stages.
  - Tokenizer config (`examples/flatgpt/02_tokenizer/configs/braintokmix_train_small.yaml`):
    `datasplitter.dataset_root` should match the stage-2 `save_path` above;
    `save_dir` and `cache_dir` can be anywhere writable.
  - FlatGPT train config (`examples/flatgpt/03_training/configs/flatgpt_train_small.yaml`):
    update `datasplitter.dataset_root`, `save_dir`, `cache_dir`, and `resume_from`
    as needed.
  - Eval config (`examples/flatgpt/04_eval/configs/flatgpt_eval_small.yaml`):
    set `save_dir` to the training `save_dir`, update `ckpt_path` to the checkpoint
    you want, and optionally adjust `output_dir` (relative to `save_dir` unless you
    use an absolute path).
- The stage-2 preprocess configs point at the bundled small outputs in
  `data/<dataset>/small/1to50hz_ss_cont` and set `skip_done: true`. This avoids
  re-running source projection (which requires an fsaverage directory). To
  regenerate stage-2 outputs, set `save_path` to a new location and supply a
  valid `fsaverage_dir` (or enable `get_fsaverage_data`).
- The tokenizer and FlatGPT configs point to specific checkpoint paths:
  - `tmp/examples/flatgpt/tokenizer/logs/version_0/checkpoints/best-checkpoint-epoch00001.ckpt`
  - `tmp/examples/flatgpt/flatgpt/logs/version_0/checkpoints/last-checkpoint-epoch00001.ckpt`
  If you change trainer settings or rerun multiple times, update those paths.
- For paper-scale runs, use the canonical configs under `configs/` and update
  dataset paths to your full data locations.

## Full paper configs

See `examples/flatgpt/05_full_paper/` for the full-size tokenizer/training/eval
configs copied from `configs/` (with paths relinked to stay self-contained).
