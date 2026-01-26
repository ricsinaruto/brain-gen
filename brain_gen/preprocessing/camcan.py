from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from .ephys import Ephys


class CamCAN(Ephys):
    """Loader for CamCAN MEG datasets (cc280/cc700)."""

    def _infer_cohort(self, path: Path) -> str:
        """Infer cohort label (cc280/cc700) from the directory path."""
        for part in path.parts:
            name = part.lower()
            if "cc700" in name:
                return "cc700"
            if "cc280" in name:
                return "cc280"
        return "camcan"

    def _extract_tag(self, tag: str, name: str) -> str | None:
        """Extract a BIDS-style tag (task/run/ses) from a filename."""
        match = re.search(rf"{tag}-([a-zA-Z0-9]+)", name)
        return match.group(1) if match else None

    def _discover_subject_roots(self, base: Path, max_depth: int = 7) -> list[Path]:
        """Find directories containing BIDS-style subject folders."""
        queue: list[tuple[Path, int]] = [(base, 0)]
        roots: list[Path] = []
        seen: set[Path] = set()

        while queue:
            current, depth = queue.pop(0)
            if current in seen or depth > max_depth:
                continue
            seen.add(current)

            try:
                children = [p for p in current.iterdir() if p.is_dir()]
            except (FileNotFoundError, PermissionError):
                continue

            subject_dirs = [p for p in children if p.name.startswith("sub-")]
            if subject_dirs:
                roots.append(current)
                continue

            for child in children:
                if not child.name.startswith("sub-"):
                    queue.append((child, depth + 1))

        return roots

    def load(self) -> dict[str, Any]:
        """Locate all CamCAN MEG sessions under cc280/cc700 roots."""

        files: list[str] = []
        subjects: list[str] = []
        subject_counts: dict[str, int] = {}

        base = Path(self.data_path)
        subject_roots = self._discover_subject_roots(base)
        if not subject_roots:
            raise FileNotFoundError(
                f"Could not find any CamCAN subject folders under {self.data_path}"
            )

        for root in sorted(subject_roots):
            cohort = self._infer_cohort(root)
            for subj_dir in sorted(p for p in root.glob("sub-*") if p.is_dir()):
                subj_id = subj_dir.name.replace("sub-", "")
                session_dirs = [p for p in subj_dir.glob("ses-*") if p.is_dir()]
                if not session_dirs:
                    session_dirs = [subj_dir]

                for ses_dir in sorted(session_dirs):
                    meg_dir = ses_dir / "meg"
                    if not meg_dir.exists():
                        continue

                    fif_files = [
                        f
                        for f in sorted(meg_dir.glob("*.fif"))
                        if (
                            f.name.endswith("_meg.fif")
                            and "crosstalk" not in f.name.lower()
                        )
                    ]
                    if not fif_files:
                        continue

                    for fif_file in fif_files:
                        task = self._extract_tag("task", fif_file.name)
                        run = self._extract_tag("run", fif_file.name)

                        name_parts = [cohort, subj_dir.name]
                        if ses_dir != subj_dir:
                            name_parts.append(ses_dir.name)
                        if task:
                            name_parts.append(f"task-{task}")
                        if run:
                            name_parts.append(f"run-{run}")

                        session_name = "_".join([p for p in name_parts if p])
                        if session_name in subject_counts:
                            subject_counts[session_name] += 1
                            session_name = (
                                f"{session_name}-{subject_counts[session_name]:02d}"
                            )
                        else:
                            subject_counts[session_name] = 0

                        print(
                            f"INFO: Found subject {subj_id} "
                            f" session {session_name} at {fif_file}"
                        )
                        files.append(str(fif_file))
                        subjects.append(session_name)

        self.batch_args = {"files": files, "subjects": subjects}


class CamCANConditioned(CamCAN):
    """CamCAN loader that also exposes task events via event_array."""

    @staticmethod
    def _clamp_sample(idx: int, n_samples: int) -> int:
        return max(0, min(int(idx), n_samples))

    def _events_path_for_fif(self, fif_path: Path) -> Path | None:
        """Return the paired BIDS events.tsv path if present."""
        if fif_path.name.endswith("_meg.fif"):
            candidate = fif_path.with_name(
                fif_path.name.replace("_meg.fif", "_events.tsv")
            )
        else:
            candidate = fif_path.with_name(f"{fif_path.stem}_events.tsv")
        return candidate if candidate.exists() else None

    def _decode_event_codes(self, df: pd.DataFrame) -> list[int]:
        """Prefer numeric `value` codes; fallback to trial_type labels."""
        if "value" in df.columns:
            values = pd.to_numeric(df["value"], errors="coerce")
            if values.notnull().any():
                filled = values.fillna(method="ffill").fillna(method="bfill")
                return [int(v) for v in filled]

        if "trial_type" in df.columns:
            labels = df["trial_type"].fillna("n/a").astype(str)
            mapping = {name: idx + 1 for idx, name in enumerate(sorted(set(labels)))}
            return [mapping[label] for label in labels]

        return list(range(1, len(df) + 1))

    def _event_array_from_tsv(
        self, fif_path: Path, sfreq: float, n_samples: int
    ) -> np.ndarray | None:
        events_path = self._events_path_for_fif(fif_path)
        if events_path is None:
            return None

        try:
            df = pd.read_csv(events_path, sep="\t")
        except Exception as exc:  # pragma: no cover - file read errors are non-fatal
            print(f"WARNING: Failed to read events tsv {events_path}: {exc}")
            return None

        if df.empty or "onset" not in df.columns:
            return None

        onsets = pd.to_numeric(df["onset"], errors="coerce")
        durations = (
            pd.to_numeric(df["duration"], errors="coerce") if "duration" in df else None
        )
        codes = self._decode_event_codes(df)

        event_array = np.zeros(n_samples, dtype=np.int16)

        for idx, onset in enumerate(onsets):
            if not np.isfinite(onset):
                continue

            start = self._clamp_sample(round(float(onset) * sfreq), n_samples)
            duration = (
                float(durations.iloc[idx])
                if durations is not None and idx < len(durations)
                else None
            )

            if duration is not None and np.isfinite(duration) and duration > 0:
                end = start + int(round(duration * sfreq))
            else:
                next_valid = next(
                    (float(val) for val in onsets.iloc[idx + 1 :] if np.isfinite(val)),
                    None,
                )
                end = (
                    int(round(next_valid * sfreq))
                    if next_valid is not None
                    else start + 1
                )

            end = self._clamp_sample(end, n_samples)
            if end <= start:
                end = min(start + 1, n_samples)

            event_array[start:end] = int(codes[idx] if idx < len(codes) else 1)

        if event_array.any():
            print(f"INFO: Loaded {len(onsets.dropna())} events from {events_path.name}")
            return event_array

        return None

    def extract_events_from_raw(self, raw, stim_channel: str = "STI101"):
        """Return events using stim channels or annotations."""
        stim_picks = mne.pick_types(raw.info, meg=False, stim=True)
        if len(stim_picks) > 0:
            stim_chs = [raw.ch_names[idx] for idx in stim_picks]
            preferred = next(
                (ch for ch in ("STI101", "STI 014", "UPPT001") if ch in stim_chs), None
            )
            stim_sel = [preferred] if preferred else stim_chs
            events = mne.find_events(
                raw,
                stim_channel=stim_sel,
                min_duration=0.002,
                output="onset",
                shortest_event=1,
            )
            if len(events) > 0:
                print(f"INFO: Detected {len(events)} events from stim chns: {stim_sel}")
                return events

        try:
            events, _ = mne.events_from_annotations(raw)
            if len(events) > 0:
                print(f"INFO: Detected {len(events)} events from annotations.")
                return events
        except Exception as exc:  # pragma: no cover - annotation parsing rarely fails
            print(f"WARNING: Failed to extract events from annotations: {exc}")

        return None

    def _event_array_from_events(
        self, events: np.ndarray, n_samples: int, decimate: int = 1
    ) -> np.ndarray | None:
        if events is None or len(events) == 0:
            return None

        if decimate > 1:
            events = events.copy()
            events[:, 0] = events[:, 0] // decimate

        event_array = np.zeros(n_samples, dtype=np.int16)
        for idx, event in enumerate(events):
            start = self._clamp_sample(event[0], n_samples)
            end = (
                self._clamp_sample(events[idx + 1, 0], n_samples)
                if idx + 1 < len(events)
                else start + 1
            )
            if end <= start:
                end = min(start + 1, n_samples)
            event_array[start:end] = int(event[2])

        return event_array if event_array.any() else None

    def extract_raw(self, fif_file: str, subject: str) -> dict[str, Any]:
        """Extract raw data and attach a dense event_array channel."""
        data = super().extract_raw(fif_file, subject)
        if data is None:
            return None

        n_samples = data["raw_array"].shape[1]
        sfreq = data["sfreq"]
        fif_path = Path(fif_file)

        event_array = self._event_array_from_tsv(fif_path, sfreq, n_samples)

        if event_array is None:
            raw = mne.io.read_raw_fif(fif_file, preload=True)
            decimate = data.get("decimate", 1)
            if raw.info.get("sfreq", 0) and data.get("sfreq"):
                ratio = raw.info["sfreq"] / data["sfreq"]
                if ratio > 0:
                    decimate = max(1, int(round(ratio)))

            events = self.extract_events_from_raw(raw)
            event_array = self._event_array_from_events(events, n_samples, decimate)

        if event_array is not None:
            data["event_array"] = event_array

        return data
