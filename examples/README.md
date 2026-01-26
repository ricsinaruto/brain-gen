# Examples

This folder contains reproducible, small-scale examples for the FlatGPT pipeline
from `flatgpt.pdf`. Each example uses the bundled small datasets under
`data/<dataset>/small` and is configured to run on CPU.

Structure:
- `flatgpt/01_preprocess`: Stage-1 OSL preprocessing (skips if outputs already
  exist) plus stage-2 cleaning.
- `flatgpt/02_tokenizer`: BrainTokMix tokenizer training.
- `flatgpt/03_training`: FlatGPT training on tokenizer codes.
- `flatgpt/04_eval`: Rollout generation and evaluation analyses.
- `flatgpt/05_full_paper`: Full-size paper configs copied from `configs/`.

The example configs are intentionally scaled down for quick local runs. For the
full paper-sized configs, start from the canonical files under `configs/`.

Path notes: all small-scale configs use repo-root relative paths by default.
If your data lives elsewhere, update the `data_path` / `dataset_root` fields in
the example YAMLs. See `examples/flatgpt/README.md` for the exact fields per
stage.
