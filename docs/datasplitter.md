# Datasplitter configuration

This page documents the `datasplitter` config used by
`brain_gen.dataset.datasplitter.split_datasets`.

## Core keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_root` | str \| list[str] \| dict[str, str] | required | Dataset root(s) containing per‑session subfolders of chunked `.npy` files. Mapping keys become dataset IDs used in logs and `split_strategy: dataset`. |
| `example_seconds` | float | required | Window length in seconds. Converted to samples via dataset `sfreq`. Must be > 0. |
| `overlap_seconds` | float | required | Overlap in seconds between windows. Must be `< example_seconds`. |
| `val_ratio` | float | `0.1` | Fraction of sessions/subjects to allocate to validation (per dataset). Ignored when `split_strategy=dataset`. |
| `test_ratio` | float | `0.1` | Fraction of sessions/subjects to allocate to test (per dataset). Ignored when `split_strategy=dataset`. |
| `seed` | int | `42` | RNG seed for shuffling sessions/subjects. |
| `dataset_class` | str | `"ChunkDataset"` | Dataset class name (see `DATASET_CLASSES` in `datasplitter.py`). |
| `dataset_kwargs` | dict | `{}` | Extra kwargs passed to the dataset constructor. |
| `cache_dir` | str \| null | `null` | Optional directory for cached session metadata. Speeds up repeated startups. |
| `refresh_cache` | bool | `false` | Rebuild cache even if present. Set `true` when data or windowing args change. |
| `split_strategy` | str | `"session"` | One of `session`, `subject`, or `dataset`. |
| `heldout_dataset` | str \| null | `null` | Required when `split_strategy=dataset`. The held‑out dataset is split 50/50 into val/test at the subject level. |

## Dataset root formats
- **String**: single dataset root, keyed internally as `dataset0`.
- **List**: multiple roots; keys are derived from folder names (deduplicated as
  needed).
- **Mapping**: explicit `{name: path}` mapping; use this for
  `split_strategy: dataset`.

## Dataset class options
Current built‑ins (from `DATASET_CLASSES`):

- `ChunkDataset`, `ChunkDatasetForecastCont`, `ChunkDatasetReconstruction`
- `ChunkDatasetImageReconstruction`, `ChunkDatasetImageQuantized`,
  `ChunkDatasetImage01`, `ChunkDatasetInterpolatedImage`
- `ChunkDataset3D`, `ChunkDatasetSensor3D`, `ChunkDatasetJIT`, `ChunkDatasetMasked`,
  `ChunkDatasetSubset`
- `BPEDataset`

## Common `dataset_kwargs`
These are passed directly to the dataset class and vary by class. Common keys:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `image_size` | int | `32` | Image height/width for image‑style datasets. |
| `tmp_dir` | str | `"tmp"` | Temp directory for cached layout indices. |
| `fill_value` | int | `0` | Fill value for missing channels/pixels. |
| `layout_path` | str \| null | `null` | Optional `.npy` path to reuse a 2‑D layout. |
| `has_condition` | bool | `false` | Return conditioning channels if available. |
| `transpose` | bool | `false` | Transpose time/channel axes in reconstruction/image datasets. |
| `use_tokenized` | bool | `false` | Load pretokenized codes instead of raw data. |
| `tokenized_root` | str \| dict | required if `use_tokenized` | Root(s) containing pretokenized chunk files. |
| `return_mask` | bool | `false` | For image reconstruction datasets: return row/col mask info. |
| `group_size` | int | `50` | For `BPEDataset`: tokens per group used when constructing text groups. |
| `escape_value` | int | `63` | For `BPEDataset`: escape value replaced with 0. |

## Constraints and invariants
- All dataset roots must share the same sampling frequency (`sfreq`), window
  length, and overlap in samples. The splitter raises if these disagree.
- `overlap_seconds` must be strictly smaller than `example_seconds`.
- Chunk files must include `data`, `ch_names`, `pos_2d`, and `sfreq`.
  Optional `pos_3d` and `ori_3d` must match channel count if present.
- Chunk lengths may vary within a session; cached metadata stores per‑chunk
  lengths.
- For `split_strategy=subject`, subject IDs are inferred from BIDS‑style
  `sub-` tokens in session names.

## Best practices
- **Cache metadata**: set `cache_dir` for large datasets and flip
  `refresh_cache: true` whenever preprocessing outputs or windowing args change.
- **Avoid leakage**: use `split_strategy: subject` to keep subject sessions
  together, or `split_strategy: dataset` for strict cross‑dataset generalization.
- **Align window sizes**: pick `example_seconds` so that
  `example_seconds * sfreq` is divisible by tokenizer window lengths when
  possible.
- **Check chunk sizes**: ensure chunks are longer than `example_seconds` or they
  will be skipped entirely.
- **Pretokenize for speed**: precompute tokens and set `use_tokenized: true` +
  `tokenized_root` for FlatGPT‑style runs.

## Minimal example

```yaml
datasplitter:
  dataset_class: ChunkDatasetReconstruction
  dataset_root:
    omega: /path/to/datasets/omega/cleaned
    mous: /path/to/datasets/mous/cleaned
  example_seconds: 10.24
  overlap_seconds: 0.0
  split_strategy: subject
  val_ratio: 0.1
  test_ratio: 0.1
  cache_dir: /path/to/cache/cleaned_10p24s
  refresh_cache: false
  dataset_kwargs:
    use_tokenized: true
    tokenized_root: /path/to/tokenized/brainomni
```
