"""Helpers for running Maxwell filtering outside the main preprocessing pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable

import mne


def maxwell_output_path(save_folder: str | Path, subject: str) -> Path:
    """Return the Maxwell-filtered output path for a subject."""
    subject_name = Path(subject).name
    return Path(save_folder) / f"{subject_name}_maxwell-raw.fif"


def _is_fif(path: Path) -> bool:
    return ".fif" in "".join(path.suffixes)


def filter_maxwell_pairs(
    pairs: Iterable[tuple[str, str]],
    save_folder: str | Path,
    skip_done: bool = True,
) -> list[tuple[str, str]]:
    """Filter file/subject pairs to those needing Maxwell filtering."""
    save_folder = Path(save_folder)
    filtered: list[tuple[str, str]] = []
    for file_path, subject in pairs:
        input_path = Path(file_path)
        subject_name = Path(subject).name

        if not _is_fif(input_path):
            print(
                "WARNING: Maxwell filtering requires FIF input; "
                f"skipping {subject_name}"
            )
            continue

        out_path = maxwell_output_path(save_folder, subject_name)
        if skip_done and out_path.exists():
            print(
                f"INFO: Using existing Maxwell-filtered file for {subject_name} "
                f"at {out_path}"
            )
            continue

        filtered.append((file_path, subject))

    return filtered


def swap_maxwell_pairs(
    pairs: Iterable[tuple[str, str]],
    save_folder: str | Path,
    *,
    require_existing: bool = True,
    log: bool = True,
) -> list[tuple[str, str]]:
    """Replace input paths with Maxwell-filtered outputs when available."""
    save_folder = Path(save_folder)
    swapped: list[tuple[str, str]] = []
    for file_path, subject in pairs:
        input_path = Path(file_path)
        subject_name = Path(subject).name

        if not _is_fif(input_path):
            if log:
                print(
                    "WARNING: Maxwell filtering requires FIF input; "
                    f"skipping {subject_name}"
                )
            continue

        out_path = maxwell_output_path(save_folder, subject_name)
        if out_path.exists():
            swapped.append((str(out_path), subject))
            continue

        if require_existing:
            if log:
                print(
                    f"WARNING: Missing Maxwell-filtered file for {subject_name} "
                    f"at {out_path}"
                )
            continue

        swapped.append((file_path, subject))

    return swapped


def chunk_pairs(
    pairs: Iterable[tuple[str, str]],
    *,
    num_chunks: int | None = None,
    chunk_size: int | None = None,
) -> list[list[tuple[str, str]]]:
    """Split input pairs into batches for Modal map calls."""
    pairs = list(pairs)
    if not pairs:
        return []

    if (num_chunks is None) == (chunk_size is None):
        raise ValueError("Provide exactly one of num_chunks or chunk_size.")

    if chunk_size is None:
        num_chunks = max(1, min(int(num_chunks), len(pairs)))
        chunk_size = int(math.ceil(len(pairs) / num_chunks))

    chunk_size = max(1, int(chunk_size))
    return [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]


def apply_maxwell_filter(
    raw: mne.io.BaseRaw, save_folder: str | Path
) -> mne.io.BaseRaw:
    """Apply bad channel detection, head position, and Maxwell filter."""
    raw.del_proj()

    cal = None
    ctc = None
    save_folder = Path(save_folder)
    if (save_folder.parent / "cal.dat").exists():
        cal = save_folder.parent / "cal.dat"
    if (save_folder.parent / "ctc.fif").exists():
        ctc = save_folder.parent / "ctc.fif"

    if cal and ctc:
        print("INFO: Using calibration and cross talk files.")
    else:
        print(
            "INFO: No calibration and cross talk files found in "
            f"{save_folder.parent}, using default values"
        )

    noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(
        raw.copy().pick("meg"),
        calibration=cal,
        cross_talk=ctc,
    )

    raw.info["bads"] = sorted(
        set(raw.info.get("bads", [])) | set(noisy_chs) | set(flat_chs)
    )

    try:
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
        head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)
    except Exception as exc:
        print(
            f"WARNING: Failed to compute head position: {exc}, "
            "running without head position"
        )
        head_pos = None
    return mne.preprocessing.maxwell_filter(
        raw,
        head_pos=head_pos,
        calibration=cal,
        cross_talk=ctc,
        destination=raw.info.get("dev_head_t"),
        st_duration=10.0,
        mag_scale="auto",
    )


def process_maxwell_file(
    file_path: str,
    subject: str,
    save_folder: str | Path,
    skip_done: bool,
    apply_fn: Callable[[mne.io.BaseRaw], mne.io.BaseRaw],
) -> tuple[str, str] | None:
    """Run Maxwell filtering for a single file."""
    input_path = Path(file_path)
    subject_name = Path(subject).name

    if not _is_fif(input_path):
        print(
            "WARNING: Maxwell filtering requires FIF input; " f"skipping {subject_name}"
        )
        return None

    out_path = maxwell_output_path(save_folder, subject_name)
    if skip_done and out_path.exists():
        print(
            f"INFO: Using existing Maxwell-filtered file for {subject_name} "
            f"at {out_path}"
        )
        return str(out_path), subject

    try:
        raw = mne.io.read_raw_fif(input_path, allow_maxshield=True, preload=True)
    except Exception as exc:
        print(
            f"WARNING: Failed to load {input_path} for Maxwell filtering "
            f"(skipping {subject_name}): {exc}"
        )
        return None

    try:
        raw_sss = apply_fn(raw)
    except Exception as exc:
        print(
            f"WARNING: Maxwell filtering failed for {subject_name} "
            f"(skipping): {exc}"
        )
        return None

    try:
        raw_sss.save(out_path, overwrite=True)
    except Exception as exc:
        print(
            "WARNING: Failed to save Maxwell-filtered file for "
            f"{subject_name} (skipping): {exc}"
        )
        return None

    return str(out_path), subject
