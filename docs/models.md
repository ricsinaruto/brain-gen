# Models and tokenizers

This repository bundles autoregressive transformers, convolutional forecasters,
diffusion models, classifiers, and multiple tokenizer families. This document
summarizes each model/tokenizer file, its interface, and the corresponding
`model_name` identifiers used in configs.

## Supported `model_name` values
These names map to classes in `brain_gen/training/utils.py`:

- `STGPT2MEG`
- `FlatGPT`, `FlatGPTEmbeds`, `FlatGPTRVQ`, `FlatGPTMix`, `FlatGPTMixRVQ`,
  `FlatGPTEmbedsRVQ`
- `BrainOmniCausalTokenizer`, `BrainOmniCausalTokenizerSEANetChannelMix`,
  `BrainOmniCausalForecast`
- `Vidtok`, `VidtokRVQ`
- `WavenetFullChannel`, `Wavenet3D`
- `BENDRForecast`
- `NTD`
- `TASA3D`
- `ClassifierContinuous`, `ClassifierQuantized`, `ClassifierQuantizedImage`

Loss names are defined in `brain_gen/training/utils.py` (e.g.
`CrossEntropyWithCodes`, `BrainOmniCausalTokenizerLoss`, `VidtokLoss`, `NTDLoss`).

## Autoregressive transformers
### STGPT2MEG (`stgpt2meg.py`)
Quantized autoregressive transformer using spatiotemporal blocks.
- **Input**: `(B, C, T)` token IDs with optional conditioning.
- **Output**: logits `(B, C, T, V)`.

### FlatGPT family (`flatgpt.py`)
FlatGPT flattens spatiotemporal token grids into a single sequence and feeds them
through a transformer (Qwen2.5/3 Video, SmoLLM, MiniMax). All variants use
`input_shape`, `temporal_reduction`, and `spatial_reduction` to construct 3D
positional IDs and validate RoPE shapes for video backbones.

- **FlatGPT**: base variant. Uses `tok_class` (e.g., `AmplitudeTokenizer`,
  `DelimitedTokenizer`, `BPETokenizer`) or a pretrained tokenizer via
  `tokenizer_path`. Output logits `(B, L, V)`.
- **FlatGPTMix**: mixes per‑channel streams via `MixEmbedding`/`MixHead`.
- **FlatGPTEmbeds**: per‑channel embeddings and channel‑aware heads.
- **FlatGPTRVQ**: RVQ‑aware variant. Quantizer levels are not part of the token
  sequence; transformer RoPE uses `(T, C, 1)` reduced shape.
- **FlatGPTMixRVQ**: RVQ + channel‑mix variant. Sequence is per‑timestep; RoPE
  uses `(T, 1, 1)` reduced shape.
- **FlatGPTEmbedsRVQ**: RVQ‑aware with per‑level embeddings and optional embedding
  corruption.

Key config knobs (FlatGPT):
- `trf_class` / `trf_args`: transformer backend and HF adapter settings.
- `tok_class` / `tok_args`: tokenizer implementation and arguments.
- `tokenizer_path`: path to a trained tokenizer checkpoint (e.g. BrainOmni).
- `input_shape`: `(T, H, W)` grid describing the flattened token layout.
- `temporal_reduction` / `spatial_reduction`: define reduced grid sizes.
- `token_corruption_cfg`: optional corruption schedule for robustness.

**Resizing context**: FlatGPT supports `resize_context(...)` for extending the
context length after loading a checkpoint (update `input_shape`, `rope_theta`,
`max_position_embeddings` consistently).

## Convolutional and hybrid forecasters
- **WavenetFullChannel** (`wavenet.py`): causal dilated 1D conv stack. Input
  `(B, C, T)`; output logits `(B, Q, T)`.
- **Wavenet3D** (`wavenet.py`): causal 3D convs over spatiotemporal tokens. Input
  `(B, H, W, T)`; output `(B, H, W, T, V)`.
- **BENDRForecast** (`bendr.py`): convolutional encoder + transformer
  contextualizer; predicts continuous samples `(B, C, T_out)`.

## Diffusion
- **NTD** (`ntd.py`): noise‑to‑data diffusion model for continuous
  reconstruction. Input `(B, C, T)`; output `(B, C, T)`.

## Spatiotemporal attention
- **TASA3D** (`tasa3d.py`): 3D attention stack for spatiotemporal tokens.
  Input `(B, H, W, T)`; output `(B, H, W, T, V)`.

## Tokenizers
### BrainOmni tokenizer + forecaster (`brainomni.py`)
- **BrainOmniCausalTokenizer**: causal SEANet encoder/decoder with residual VQ.
  Inputs are continuous signals `(B, C, T)` plus sensor metadata
  `(pos, sensor_type)`. `tokenize()` returns a `CausalTokenSequence` with
  embeddings `(B, C, W, D)` and indices `(B, C, W, Q)`.
- **BrainOmniCausalForecast**: autoregressive forecaster over BrainOmni tokens.
  Predicts per‑quantizer logits `(B, C, W, Q, K)` and can decode forecasts back
  to MEG space.

### BrainTokMix (`models/tokenizers/braintokmix.py`)
- **BrainOmniCausalTokenizerSEANetChannelMix**: channel‑mixing tokenizer that
  splits latent dimensions into `n_neuro` spatial tokens for mixing across
  sensors.

### Vidtok (`models/tokenizers/vidtok.py`)
- **Vidtok**: causal 3D encoder/decoder with FSQ regularization. Inputs
  `(B, C, T, H, W)`.
- **VidtokRVQ**: residual‑VQ variant with optional temporal covariance loss.

### Flat tokenizers (`models/tokenizers/flat_tokenizers.py`)
Lightweight amplitude/BPE/delimited tokenizers used by FlatGPT.

### Factorized autoencoder (`models/tokenizers/factorized.py`)
Factorized 1D encoder/decoder utilities used for research; not wired into the
standard `run.py` model registry.

## Input/output conventions
- **Continuous signals**: `(B, C, T)` for channel‑first time series.
- **Quantized tokens**: `(B, C, T)` for channelized token IDs; logits typically
  `(B, C, T, V)`.
- **Spatiotemporal tokens**: `(B, H, W, T)` or `(B, C, T, H, W)` depending on
  model.
- **FlatGPT**: flattened token sequences `(B, L)` with 3D positional structure
  inferred from `input_shape` / `reduced_shape`.

## BrainOmni + FlatGPT (VQFlatGPT) workflow

### 1) Train a causal BrainOmni/BrainTokMix tokenizer
```yaml
model_name: BrainOmniCausalTokenizerSEANetChannelMix
loss_name: BrainOmniCausalTokenizerLoss
model_config: configs/braintokmix/tokenizer10s_ds.yaml
```

Key tokenizer config fields:
- `window_length`: window size in samples.
- `ratios`: temporal downsampling; tokens per window =
  `window_length / prod(ratios)`.
- `codebook_size`, `num_quantizers`: RVQ settings.
- `num_sensors`, `sensor_space`: sensor embedding settings.

### 2) Train FlatGPT on tokenizer codes
```yaml
model_name: FlatGPTEmbedsRVQ
loss_name: CrossEntropyWithCodes
model_config: configs/flatgpt/flat/model_chmix.yaml
```

`model_chmix.yaml` should point to the tokenizer checkpoint:
```yaml
tokenizer_path: /path/to/braintokmix/checkpoint.ckpt
vocab_size: 16384
input_shape: [1024, 68, 4]  # (T, H, W)
```

Best practices:
- **Freeze tokenizer** for LM training via `train_tokenizer: false` (when the
  model supports it).
- **Align vocab size**: set FlatGPT `vocab_size` equal to the tokenizer
  `codebook_size`.
- **Match token grid**: choose `input_shape` and reductions so that
  `reduced_shape[2]` equals the number of RVQ levels for `FlatGPTEmbedsRVQ`.
- **Pretokenize for speed**: use `scripts/pretokenize_flatgpt_dataset.py` and set
  `use_tokenized: true` + `tokenized_root` in the datasplitter.

