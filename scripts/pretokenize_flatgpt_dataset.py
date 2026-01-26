import argparse
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

from brain_gen.dataset import split_datasets
from brain_gen.training.lightning import LitModel


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_tokenizer(model_cfg: dict) -> torch.nn.Module:
    tokenizer_path = model_cfg.get("tokenizer_path")
    if tokenizer_path is None:
        raise ValueError(
            "model config must define tokenizer_path for pre-tokenization."
        )

    lit = LitModel.load_from_checkpoint(tokenizer_path, strict=False)
    tokenizer = lit.model

    if hasattr(tokenizer, "_orig_mod"):
        tokenizer = tokenizer._orig_mod

    if hasattr(tokenizer, "set_eval_mode"):
        tokenizer.set_eval_mode()

    tokenizer.eval()
    print(f"INFO: Loaded tokenizer {tokenizer.__class__.__name__} from checkpoint.")
    return tokenizer


def _infer_codebook_size(tokenizer: torch.nn.Module, model_cfg: dict) -> int | None:
    for attr in ("codebook_size", "vocab_size"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)

    quantizer = getattr(tokenizer, "quantizer", None)
    rvq = getattr(quantizer, "rvq", None)
    value = getattr(rvq, "codebook_size", None)
    if value is not None:
        return int(value)

    value = model_cfg.get("vocab_size")
    if value is not None:
        return int(value)

    return None


def _infer_storage_dtype(codebook_size: int | None) -> np.dtype:
    if codebook_size is None:
        return np.uint32
    if codebook_size <= np.iinfo(np.uint16).max:
        return np.uint16
    if codebook_size <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_device(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: _to_device(val, device) for key, val in obj.items()}
    return obj


def _iter_chunks(
    indexed: List[Tuple[int, Tuple[str, str, str, int]]],
) -> Iterable[Tuple[Tuple[str, str, str], List[Tuple[int, Tuple[str, str, str, int]]]]]:
    indexed = sorted(
        indexed, key=lambda item: (item[1][0], item[1][1], item[1][2], item[1][3])
    )
    current_key = None
    current_entries: List[Tuple[int, Tuple[str, str, str, int]]] = []
    for entry in indexed:
        dataset_key, session, chunk, _ = entry[1]
        key = (dataset_key, session, chunk)
        if current_key is None:
            current_key = key
        if key != current_key:
            yield current_key, current_entries
            current_entries = []
            current_key = key
        current_entries.append(entry)
    if current_entries:
        yield current_key, current_entries


def _build_output_roots(root_dirs: Dict[str, str], output_root: str) -> Dict[str, str]:
    if len(root_dirs) == 1:
        key = next(iter(root_dirs))
        return {key: output_root}
    return {key: os.path.join(output_root, key) for key in root_dirs}


def _process_chunk(
    dataset,
    tokenizer: torch.nn.Module,
    entries: List[Tuple[int, Tuple[str, str, str, int]]],
    output_path: str,
    batch_size: int,
    device: torch.device,
    storage_dtype: np.dtype,
    codebook_size: int | None,
    overwrite: bool,
) -> None:
    if os.path.exists(output_path) and not overwrite:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    starts = np.array([entry[1][3] for entry in entries], dtype=np.int32)
    order = np.argsort(starts)
    starts = starts[order]
    entries = [entries[i] for i in order.tolist()]

    num_windows = len(entries)
    codes_store = None
    rvq_store = None
    num_quantizers = None

    write_idx = 0
    with torch.inference_mode():
        for batch_start in range(0, num_windows, batch_size):
            batch_entries = entries[batch_start : batch_start + batch_size]
            batch_inputs = []
            for idx, _ in batch_entries:
                inputs, _ = dataset[idx]
                batch_inputs.append(inputs)

            batch_inputs = default_collate(batch_inputs)
            batch_inputs = _to_device(batch_inputs, device)

            outputs = tokenizer.encode(batch_inputs)
            rvq_codes = outputs.get("rvq_codes")
            codes = outputs.get("codes")

            if rvq_codes is not None:
                rvq_np = rvq_codes.cpu().numpy().astype(storage_dtype, copy=False)
                if rvq_store is None:
                    rvq_store = np.empty(
                        (num_windows,) + rvq_np.shape[1:], dtype=storage_dtype
                    )
                    num_quantizers = rvq_np.shape[-1]
                rvq_store[write_idx : write_idx + rvq_np.shape[0]] = rvq_np
            elif codes is not None:
                codes_np = codes.cpu().numpy().astype(storage_dtype, copy=False)
                if codes_store is None:
                    codes_store = np.empty(
                        (num_windows,) + codes_np.shape[1:], dtype=storage_dtype
                    )
                codes_store[write_idx : write_idx + codes_np.shape[0]] = codes_np
            else:
                raise RuntimeError("Tokenizer encode returned neither codes nor rvq.")

            write_idx += int(
                rvq_codes.shape[0] if rvq_codes is not None else codes.shape[0]
            )

    payload: Dict[str, object] = {"starts": starts}
    if rvq_store is not None:
        payload["rvq_codes"] = rvq_store
    if codes_store is not None:
        payload["codes"] = codes_store
    if codebook_size is not None:
        payload["codebook_size"] = int(codebook_size)
    if num_quantizers is not None:
        payload["num_quantizers"] = int(num_quantizers)

    np.save(output_path, payload)


def _pretokenize_split(
    dataset,
    tokenizer: torch.nn.Module,
    output_roots: Dict[str, str],
    batch_size: int,
    device: torch.device,
    storage_dtype: np.dtype,
    codebook_size: int | None,
    overwrite: bool,
    split_name: str,
) -> None:
    indexed = list(enumerate(dataset.indices))
    total_chunks = sum(1 for _ in _iter_chunks(indexed))
    print(
        "INFO: Pretokenizing split "
        f"{split_name} with {len(indexed)} windows across {total_chunks} chunks."
    )
    pbar = tqdm(total=total_chunks, desc=f"Pretokenize {split_name}")
    for (dataset_key, session, chunk), entries in _iter_chunks(indexed):
        output_root = output_roots[dataset_key]
        output_path = os.path.join(output_root, session, chunk)
        _process_chunk(
            dataset,
            tokenizer,
            entries,
            output_path,
            batch_size,
            device,
            storage_dtype,
            codebook_size,
            overwrite,
        )
        pbar.update(1)
    pbar.close()
    print(f"INFO: Finished pretokenizing split {split_name}.")


def main(cli_args: Sequence[object] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset windows for FlatGPT training."
    )
    parser.add_argument("-c", "--config", required=True, help="Training config YAML.")
    parser.add_argument(
        "-o",
        "--output-root",
        required=True,
        help="Directory to store tokenized chunks.",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument(
        "-d",
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "-s",
        "--splits",
        default="all",
        choices=("train", "val", "test", "all"),
    )
    parser.add_argument("-ov", "--overwrite", action="store_true")

    args = parser.parse_args(cli_args)
    print(
        "INFO: Starting pretokenization with "
        f"config={args.config}, output_root={args.output_root}, "
        f"device={args.device}, batch_size={args.batch_size}, "
        f"splits={args.splits}, overwrite={args.overwrite}."
    )

    cfg = _load_yaml(args.config)
    model_cfg = _load_yaml(cfg["model_config"])
    print(f"INFO: Loaded model config from {cfg['model_config']}.")
    print(f"INFO: Tokenizer checkpoint: {model_cfg.get('tokenizer_path')}.")

    tokenizer = _load_tokenizer(model_cfg)
    device = torch.device(args.device)
    tokenizer = tokenizer.to(device)

    codebook_size = _infer_codebook_size(tokenizer, model_cfg)
    storage_dtype = _infer_storage_dtype(codebook_size)
    print(
        "INFO: Using storage dtype " f"{storage_dtype} (codebook_size={codebook_size})."
    )

    datasets = split_datasets(**cfg["datasplitter"])
    output_roots = _build_output_roots(datasets.train.root_dirs, args.output_root)
    for path in output_roots.values():
        os.makedirs(path, exist_ok=True)
    print(f"INFO: Output roots: {output_roots}.")

    splits = {
        "train": datasets.train,
        "val": datasets.val,
        "test": datasets.test,
    }
    selected = list(splits.keys()) if args.splits == "all" else [args.splits]

    for split_name in selected:
        _pretokenize_split(
            splits[split_name],
            tokenizer,
            output_roots,
            args.batch_size,
            device,
            storage_dtype,
            codebook_size,
            args.overwrite,
            split_name,
        )


if __name__ == "__main__":
    main()
