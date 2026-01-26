import json
import os
from dataclasses import asdict
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import dask
import numpy as np
from dask.distributed import Client, default_client

from osl_ephys.preprocessing import run_proc_batch

from .text import (  # noqa: F401
    TextProcessor,
    GroupedTextProcessor,
    TemporalTextProcessor,
)
from ..utils.quantizers import mulaw


class Preprocessing(ABC):
    """Base class for preprocessing ephys data.

    Attributes:     data_path: Path to the data directory.     osl_config: OSL config
    string.     preproc_config: Dictionary containing the preprocessing configuration.
    n_workers: Number of workers for parallel processing.     chunk_seconds: Length of
    each chunk in seconds.     save_folder: Path to the save directory.     outdir_path:
    Path to the output directory.     stage3_save_path: Path to the text output
    directory for stage 3.     batch_args: Dictionary containing the batch arguments.
    extra_funcs: List of extra functions to apply to the data.     delete_fif: Whether
    to delete the .fif file after processing.     text_processor: Text processor used
    for building lookups and text chunks.
    """

    def __init__(
        self,
        data_path: str,
        log_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        stage1_path: Optional[str] = None,
        stage3_save_path: Optional[str] = None,
        osl_config: Optional[str] = None,
        preproc_config: Optional[dict[str, Any]] = None,
        n_workers: int = 1,
        gen_report: bool = True,
        use_dask: bool = True,
        delete_fif: bool = True,
        chunk_seconds: int = 60,
        skip_done: bool = False,
        text_num_bins: int = 256,
        stage3_quantize: bool = False,
        text_processor: Optional[TextProcessor] = None,
        text_processor_cls: str | None = None,
        text_processor_kwargs: Optional[dict[str, Any]] = None,
        session_cleaning: Optional[dict[str, Any]] = None,
    ) -> None:
        """Base class for preprocessing ephys data.

        Args:     data_path: Path to the data directory.     config: OSL config string.
        n_workers: Number of workers for parallel processing.     chunk_seconds: Length
        of each chunk in seconds.     stage3_save_path: Destination for text documents
        produced in stage 3.     text_num_bins: Number of quantization bins used when
        converting to text.     text_processor: Optional text processor instance to
        override the default.     text_processor_cls: Class used to create a text
        processor if not supplied.     text_processor_kwargs: Extra kwargs forwarded to
        the text processor class.
        """
        self.data_path = data_path
        self.preproc_config = preproc_config
        self.delete_fif = delete_fif
        self.n_workers = n_workers
        self.gen_report = gen_report
        self.use_dask = use_dask
        self.batch_args = {}
        self.chunk_seconds = chunk_seconds
        self.skip_done = skip_done
        self.stage3_quantize = stage3_quantize
        self.session_cleaning = session_cleaning or None

        # load osl config from file if provided
        self.osl_config = osl_config

        # Set save folder path
        default_save_path = os.path.join(os.path.dirname(data_path), "preprocessed")
        self.save_folder = save_path if save_path is not None else default_save_path
        self.log_dir = log_dir if log_dir is not None else self.save_folder
        self.stage1_path = stage1_path if stage1_path is not None else self.save_folder
        default_stage3_path = os.path.join(
            os.path.dirname(self.save_folder), "preprocessed_text"
        )
        self.stage3_save_path = (
            stage3_save_path if stage3_save_path is not None else default_stage3_path
        )

        # Set extra_funcs for this class based on preproc_config
        self.extra_funcs = []
        if self.preproc_config is not None:
            self.extra_funcs = [
                getattr(self, name) for name in self.preproc_config.keys()
            ]

        self.text_processor = None
        if text_processor is not None:
            self.text_processor = text_processor
        elif text_processor_cls is not None:
            processor_kwargs = dict(text_processor_kwargs or {})
            if "num_bins" not in processor_kwargs:
                processor_kwargs["num_bins"] = text_num_bins

            text_processor_cls = globals()[text_processor_cls]
            processor_cls = text_processor_cls or TextProcessor
            self.text_processor = processor_cls(**processor_kwargs)

        if self.text_processor is not None:
            self.text_num_bins = self.text_processor.num_bins

        self.load()

    def _apply_session_cleaning(self, session_dir: str, sfreq: float) -> None:
        cfg = dict(self.session_cleaning or {})
        if not cfg or not cfg.get("enabled", True):
            return

        from ..utils.session_cleaning import (
            clean_session_array,
            load_session_data,
            save_session_segments,
        )

        window_seconds = float(cfg.get("window_seconds", 5.0))
        std_threshold = float(cfg.get("std_threshold", 1.5))
        max_bad_pct = float(cfg.get("max_bad_pct", 20.0))
        clip_range = cfg.get("clip_range", (-10.0, 10.0))
        if isinstance(clip_range, (list, tuple)) and len(clip_range) == 2:
            clip_range = (float(clip_range[0]), float(clip_range[1]))
        else:
            clip_range = (-10.0, 10.0)

        max_segment_seconds = float(cfg.get("max_segment_seconds", 600.0))
        min_segment_seconds = float(cfg.get("min_segment_seconds", 60.0))
        min_first_segment_seconds = float(cfg.get("min_first_segment_seconds", 0.0))
        overwrite = bool(cfg.get("overwrite", True))

        session_path = Path(session_dir)
        data, metadata = load_session_data(session_path)
        segments, summary = clean_session_array(
            data,
            sfreq,
            window_seconds,
            std_threshold,
            max_bad_pct,
            clip_range,
            max_segment_seconds=max_segment_seconds,
            min_segment_seconds=min_segment_seconds,
            min_first_segment_seconds=min_first_segment_seconds,
        )
        save_session_segments(session_path, segments, metadata, overwrite=overwrite)

        summary_path = session_path / "cleaning_summary.json"
        summary_path.write_text(json.dumps(asdict(summary), indent=2))
        status = "discarded" if summary.discarded else "kept"
        print(
            f"INFO: Session cleaning {status} {session_path.name} "
            f"(bad={summary.bad_window_pct:.1f}%, segments={summary.segments_kept})"
        )

    @contextmanager
    def _dask_client(self, n_workers: Optional[int] = None):
        client = None
        created = False
        if self.use_dask:
            try:
                client = default_client()
            except ValueError:
                workers = n_workers or self.n_workers
                client = Client(threads_per_worker=1, n_workers=workers)
                created = True
        try:
            yield client
        finally:
            if created and client is not None:
                client.close()

    @abstractmethod
    def extract_raw(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract raw data from the data dictionary."""
        return NotImplementedError

    @abstractmethod
    def load(self) -> dict[str, np.ndarray]:
        """Should populate self.batch_args with the following keys: - files (str or list
        or mne.Raw) - Can be a list of Raw objects or a list of filenames (or .ds dir
        names if CTF data) or a path to a textfile list of.

        filenames (or .ds dir names if CTF data). - subjects (list of str) - Subject
        directory names. These are sub-directories in outdir.
        """
        return NotImplementedError

    def preprocess_stage_1(self) -> None:
        """Stage 1: Run OSL preprocessing pipeline."""
        print(f"INFO: Preprocessing {self.data_path}")
        # paths for logs and reports
        print(f"INFO: Will save preprocessed data to {self.save_folder}")
        logsdir = os.path.join(self.log_dir, "logs")
        reportdir = os.path.join(self.save_folder, "reports")

        # make these dirs
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(logsdir, exist_ok=True)
        os.makedirs(reportdir, exist_ok=True)

        # if skip_done is true, filter files and only keep those that
        # don't already exist in self.save_folder
        # apply same filter to subjects
        if self.skip_done:
            total_files = len(self.batch_args["subjects"])
            save_folder = Path(self.save_folder)
            files, subjects = [], []

            num_kept = 0
            for f, s in zip(self.batch_args["files"], self.batch_args["subjects"]):
                if not (save_folder / Path(s).name).exists():
                    files.append(f)
                    subjects.append(s)
                    num_kept += 1
                elif not (save_folder / Path(s).name / f"{s}_preproc-raw.fif").exists():
                    files.append(f)
                    subjects.append(s)
                    num_kept += 1
                else:
                    print(f"INFO: Skipping {s} because it already has .fif file")

            self.batch_args["files"] = files
            self.batch_args["subjects"] = subjects

            print(f"INFO: Kept {num_kept} files out of {total_files}")

        worker_count = max(1, len(self.batch_args.get("files", [])))
        worker_count = min(worker_count, self.n_workers)

        with self._dask_client(n_workers=worker_count) as client:
            # Stage 1: Run OSL preprocessing pipeline
            print("Stage 1: Running OSL preprocessing pipeline...")
            run_proc_batch(
                self.osl_config,
                files=self.batch_args["files"],
                subjects=self.batch_args["subjects"],
                outdir=self.save_folder,
                logsdir=logsdir,
                reportdir=reportdir,
                gen_report=self.gen_report,
                dask_client=bool(client),
                overwrite=True,
            )

    def _process_stage_2_subject(self, subject: str) -> None:
        subject_dir = os.path.join(self.stage1_path, subject)
        fif_file = os.path.join(subject_dir, f"{subject}_preproc-raw.fif")

        if not os.path.exists(fif_file):
            print(f"WARNING: No .fif file found for {subject}")
            return

        stage2_dir = os.path.join(self.save_folder, subject)
        npy_files = []
        if os.path.exists(stage2_dir):
            npy_files = [
                f
                for f in os.listdir(stage2_dir)
                if f.endswith(".npy") and "events" not in f
            ]

        if len(npy_files) > 0 and self.skip_done:
            print(f"INFO: Skipping {subject} because it already has .npy files")
            return

        try:
            print(f"Processing {subject}...")

            data = self.process_custom(fif_file, subject)
            if data is None:
                print(f"INFO: Skipping {subject} because data is None")
                return

            self.chunk_and_save(data, subject)

            print(f"INFO: Successfully processed {subject}")
        except Exception as e:
            print(f"ERROR: Failed to process {subject}: {str(e)}")

    def preprocess_stage_2(self) -> None:
        """Stage 2: Load and apply custom processing."""
        print("\nStage 2: Applying custom processing pipeline...")
        subjects = self.batch_args["subjects"]
        if self.use_dask:
            tasks = [
                dask.delayed(self._process_stage_2_subject)(subject)
                for subject in subjects
            ]
            if tasks:
                dask.compute(
                    *tasks,
                    scheduler="processes",
                    num_workers=self.n_workers,
                )
        else:
            for subject in subjects:
                self._process_stage_2_subject(subject)

        # Clean up intermediate files
        for subject in self.batch_args["subjects"]:
            fif_file = os.path.join(
                self.save_folder, subject, f"{subject}_preproc-raw.fif"
            )
            if os.path.exists(fif_file) and self.delete_fif:
                try:
                    os.remove(fif_file)
                except Exception as e:
                    print(f"Warning: Could not remove {fif_file}: {str(e)}")

    def process_custom(self, fif_file: str, subject: str) -> dict[str, Any]:
        """Process a single raw file with custom pipeline steps.

        Args:     raw: MNE Raw object after OSL preprocessing

        Returns:     dict: Processed data and metadata
        """
        data = self.extract_raw(fif_file, subject)
        if data is None:
            return None

        for func, name in zip(self.extra_funcs, self.preproc_config):
            print(f"INFO: Applying {name}")
            data = func(data, **self.preproc_config[name])
        return data

    def chunk_and_save(self, data: dict[str, Any], session: str) -> None:
        """Chunk session data into fixed length segments and save to disk.

        Args:     data: Processed data and metadata.     session: Session name.
        """
        bad_channels = {
            name for name in data.get("bad_channels", []) if name in data["ch_names"]
        }
        array = data["raw_array"]
        if bad_channels:
            bad_idx = [
                idx for idx, name in enumerate(data["ch_names"]) if name in bad_channels
            ]
            if bad_idx:
                array = np.array(array, copy=True)
                array[bad_idx, :] = 0.0

        extra_data = {
            "ch_names": data["ch_names"],
            "ch_types": data["ch_types"],
            "pos_2d": data["pos_2d"],
            "session": data["session"],
            "sfreq": data["sfreq"],
        }
        if "pos_3d" in data:
            extra_data["pos_3d"] = data["pos_3d"]
        if "ori_3d" in data:
            extra_data["ori_3d"] = data["ori_3d"]
        if bad_channels:
            extra_data["bad_channels"] = list(bad_channels)

        optional_metadata = (
            "predictor_kind",
            "residual_scale",
            "mulaw_mu",
            "mulaw_mse",
        )
        for key in optional_metadata:
            if key in data:
                extra_data[key] = data[key]

        # if event_array is in data, add it to raw_array as a new channel
        if "event_array" in data:
            event_array = data["event_array"].reshape(1, -1)
            array = np.concatenate([array, event_array], axis=0)

        chunk_len = int(data["sfreq"] * self.chunk_seconds)
        n_samples = array.shape[1]
        n_chunks = n_samples // chunk_len
        session_dir = os.path.join(self.save_folder, session)
        os.makedirs(session_dir, exist_ok=True)
        for i in range(n_chunks):
            start = i * chunk_len
            chunk = {
                "data": array[:, start : start + chunk_len],
                **extra_data,
            }
            np.save(os.path.join(session_dir, f"{i}.npy"), chunk)

        # Save the last chunk if there are leftover samples
        remainder = n_samples % chunk_len
        if remainder != 0:
            start = n_chunks * chunk_len
            chunk = {
                "data": array[:, start:],
                **extra_data,
            }
            np.save(os.path.join(session_dir, f"{n_chunks}.npy"), chunk)

        if self.session_cleaning:
            self._apply_session_cleaning(session_dir, data["sfreq"])

    def quantize_chunks(self, src_dir: str, dst_dir: str, chunk_files: list[str]):
        """Quantize chunks using mu-law compression.

        Args:     src_dir: Path to the source directory.     chunk_files: List of chunk
        files.
        """
        chunks = []
        full_chunks = []

        # iterate over chunks to compute maxval
        for chunk_name in chunk_files:
            chunk_path = os.path.join(src_dir, chunk_name)
            chunk = np.load(chunk_path, allow_pickle=True).item()
            chunks.append(chunk["data"])
            full_chunks.append(chunk)

        chunks = np.concatenate(chunks, axis=-1)
        maxval = np.max(np.abs(chunks))

        # compute lag-1 correlation
        corr = np.corrcoef(chunks[30, :-1], chunks[30, 1:])
        print(f"INFO: Lag-1 correlation: {corr[0, 1]}")

        for chunk_name, chunk in zip(chunk_files, full_chunks):
            data = chunk["data"] / maxval
            data, _ = mulaw(data, self.text_num_bins - 1)
            chunk["data"] = data
            np.save(os.path.join(dst_dir, chunk_name), chunk)

    def preprocess_stage_3(self) -> None:
        """Convert saved numpy chunks into UTF-8 text documents using a dataset-level
        mapping."""
        print("\nStage 3: Converting chunked data to text documents...")
        os.makedirs(self.stage3_save_path, exist_ok=True)

        processor = self.text_processor
        dataset_counts = np.zeros(processor.num_bins, dtype=np.int64)
        session_records: list[dict[str, Any]] = []

        # Pass 1: collect counts across all sessions (reserved bin collapsed into 0)
        for session in self.batch_args["subjects"]:
            src_dir = os.path.join(self.save_folder, session)
            if not os.path.isdir(src_dir):
                print(f"WARNING: No preprocessed chunks found for {session}")
                continue

            dst_dir = os.path.join(self.stage3_save_path, session)
            chunk_files = [f for f in os.listdir(src_dir) if f.endswith(".npy")]
            chunk_files.sort(key=processor.chunk_sort_key)

            os.makedirs(dst_dir, exist_ok=True)
            if not chunk_files:
                print(f"INFO: No .npy chunks found for {session}, skipping")
                continue

            if self.stage3_quantize:
                self.quantize_chunks(src_dir, dst_dir, chunk_files)
                src_dir = dst_dir

            session_counts = np.zeros(processor.num_bins, dtype=np.int64)
            for chunk_name in chunk_files:
                chunk_path = os.path.join(src_dir, chunk_name)
                try:
                    chunk = np.load(chunk_path, allow_pickle=True).item()
                except Exception as e:
                    print(f"WARNING: Failed to load {chunk_path}: {str(e)}")
                    continue

                if not isinstance(chunk, dict) or "data" not in chunk:
                    print(f"WARNING: {chunk_path} missing 'data'; skipping")
                    continue

                try:
                    session_counts += processor.count_bins(chunk["data"])
                except Exception as e:
                    print(f"WARNING: Failed to count {chunk_path}: {str(e)}")
                    continue

            dataset_counts += session_counts
            session_records.append(
                {
                    "session": session,
                    "src_dir": src_dir,
                    "dst_dir": dst_dir,
                    "chunk_files": chunk_files,
                    "counts": session_counts,
                }
            )

        if dataset_counts.sum() == 0:
            print("INFO: No valid samples found across sessions; nothing to convert")
            return

        # Save dataset-level counts
        counts_path = os.path.join(self.stage3_save_path, "dataset_counts.npy")
        np.save(counts_path, dataset_counts)
        counts_json_path = os.path.join(self.stage3_save_path, "dataset_counts.json")
        try:
            with open(counts_json_path, "w", encoding="utf-8") as handle:
                json.dump(dataset_counts.tolist(), handle)
        except Exception as e:
            print(f"WARNING: Failed to save dataset counts JSON: {e}")

        try:
            char_lookup = processor.build_char_lookup(dataset_counts)
        except Exception as e:
            print(f"WARNING: Failed to build dataset-level char lookup: {e}")
            return

        # Persist lookup as codepoints to avoid encoding issues
        lookup_path = os.path.join(self.stage3_save_path, "char_lookup_codepoints.json")
        try:
            with open(lookup_path, "w", encoding="utf-8") as handle:
                json.dump([ord(ch) for ch in char_lookup], handle)
        except Exception as e:
            print(f"WARNING: Failed to save char lookup: {e}")

        # Pass 2: convert each session using the shared lookup
        for record in session_records:
            session = record["session"]
            dst_dir = record["dst_dir"]

            if self.skip_done and os.path.isdir(dst_dir):
                existing_txt = [f for f in os.listdir(dst_dir) if f.endswith(".txt")]
                if existing_txt:
                    print(f"INFO: Skipping {session} (text chunks already exist)")
                    continue

            os.makedirs(dst_dir, exist_ok=True)
            chunk_files = record["chunk_files"]
            if not chunk_files:
                continue

            for chunk_name in chunk_files:
                chunk_path = os.path.join(record["src_dir"], chunk_name)
                try:
                    chunk = np.load(chunk_path, allow_pickle=True).item()
                except Exception as e:
                    print(f"WARNING: Failed to load {chunk_path}: {str(e)}")
                    continue

                if not isinstance(chunk, dict) or "data" not in chunk:
                    print(f"WARNING: {chunk_path} missing 'data'; skipping")
                    continue

                try:
                    text = processor.array_to_text(chunk["data"], char_lookup)
                except Exception as e:
                    print(f"WARNING: Failed to convert {chunk_path}: {str(e)}")
                    continue

                out_path = os.path.join(dst_dir, f"{Path(chunk_name).stem}.txt")
                with open(out_path, "w", encoding="utf-8") as handle:
                    handle.write(text)

            plot_counts = processor.normalize_counts(record["counts"])
            if plot_counts.sum() > 0:
                try:
                    import matplotlib.pyplot as plt  # type: ignore

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(np.arange(processor.num_bins), plot_counts, width=1.0)
                    ax.set_xlim(-0.5, processor.num_bins - 0.5)
                    ax.set_xlabel("Bin")
                    ax.set_ylabel("Count")
                    collapse_note = (
                        f" (bin {processor.collapse_bin}→0 merged)"
                        if processor.collapse_bin is not None
                        else ""
                    )
                    ax.set_title(f"{session} value distribution{collapse_note}")
                    fig.tight_layout()
                    plot_path = os.path.join(dst_dir, "distribution.png")
                    fig.savefig(plot_path, dpi=150)
                    plt.close(fig)
                except Exception as e:
                    print(
                        f"WARNING: Failed to save distribution plot for {session}: {e}"
                    )
