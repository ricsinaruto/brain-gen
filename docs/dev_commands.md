# Developer commands

Common workflows use the Python entrypoints and YAML configs in `configs/`.
Run from the repo root.

## Install
```bash
conda run -n ephys-gpt pip install -e .
```

## Preprocessing
```bash
conda run -n ephys-gpt python preprocess.py \
  --dataset omega \
  --stage stage_2 \
  --args configs/preproc/omega/osl.yaml
```
Produces chunked `.npy` files under the configured `save_path` with metadata
(channel names, sampling frequency, 2‑D positions).

## Training
```bash
conda run -n ephys-gpt python run.py --mode train --args configs/stgpt2meg/train_omega.yaml
conda run -n ephys-gpt python run.py --mode train --args configs/flatgpt/flat/train_ds.yaml
```

## Testing
```bash
conda run -n ephys-gpt python run.py --mode test --args configs/stgpt2meg/train_omega.yaml
```

## Tokenizer training
BrainOmni/BrainTokMix tokenizers are trained with `run.py --mode train` and a
Tokenizer model config:

```bash
conda run -n ephys-gpt python run.py --mode train --args configs/braintokmix/train10s_ds.yaml
```

Text BPE tokenizers use `run.py --mode tokenizer-text`:

```bash
conda run -n ephys-gpt python run.py --mode tokenizer-text --args /path/to/text_tokenizer.yaml
```

## Vidtok training
```bash
conda run -n ephys-gpt python run.py --mode vidtok --args configs/vidtok/train.yaml
```

## Evaluation
```bash
conda run -n ephys-gpt python evals.py --args configs/flatgpt/flat/test_rest.yaml
```

Single‑checkpoint eval without the sweep wrapper:

```bash
conda run -n ephys-gpt python -m brain_gen.eval.eval_runner \
  --config configs/flatgpt/flat/test_rest.yaml \
  --ckpt /path/to/checkpoint.ckpt
```

## Pretokenize FlatGPT datasets
```bash
conda run -n ephys-gpt python scripts/pretokenize_flatgpt_dataset.py \
  --config configs/flatgpt/flat/train_ds.yaml \
  --output-root /path/to/tokenized_root
```
Then set `use_tokenized: true` + `tokenized_root` in the datasplitter.

## Formatting + tests
```bash
conda run -n ephys-gpt black --diff .
conda run -n ephys-gpt pytest -q tests/<test_file_relevant_to_change>
```
