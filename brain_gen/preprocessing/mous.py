from typing import Any
from pathlib import Path
import mne
import numpy as np

from .ephys import Ephys


class MOUS(Ephys):
    def load(self) -> dict[str, Any]:
        """Load Omega dataset and generate file and subject names.

        Returns:     dict[str, Any]: Dictionary containing the file and subject names.
        """

        files = []
        subjects = []
        subject_counts = {}  # Track counts of each subject name

        # List subject folders
        subject_dirs = [d for d in Path(self.data_path).iterdir() if d.is_dir()]
        subject_dirs = [d for d in subject_dirs if "sub" in d.name]
        subject_dirs.sort()

        # keep just two subjects
        # subject_dirs = subject_dirs[:1]

        # Iterate through each subject directory
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name.split("-")[1]
            print(f"INFO: Found subject {subject_id}")

            # List session folders within subject
            sub_dir = subject_dir / "meg"
            session_dirs = [d for d in sub_dir.iterdir() if d.is_dir()]

            # Iterate through each session
            for session_dir in session_dirs:
                task_type = session_dir.name.split("-")[2:]
                task_type = "-".join(task_type).split(".")[0]
                print(f"INFO: Found session {task_type}")

                if task_type is None:
                    print(f"Warning: Unknown task type in folder {session_dir}")
                    continue

                # Create base subject name
                subject_name = f"sub-{subject_id}_ses-{task_type}"

                # Add index if this subject_name already exists
                if subject_name in subject_counts:
                    subject_counts[subject_name] += 1
                    subject_name = f"{subject_name}-{subject_counts[subject_name]:02d}"
                else:
                    subject_counts[subject_name] = 0

                subjects.append(subject_name)
                files.append(str(session_dir))

        self.batch_args = {"files": files, "subjects": subjects}


class MOUSConditioned(MOUS):
    def extract_events_from_raw(self, raw, stim_channel: str = "UPPT001"):
        """Return every event using stim channels or annotations."""
        stim_picks = mne.pick_types(raw.info, stim=True)
        if len(stim_picks) > 0:
            stim_chs = [raw.ch_names[idx] for idx in stim_picks]
            if stim_channel in stim_chs:
                stim_chs = [stim_channel]
            print(f"Using stim channels: {stim_chs}")
            events = mne.find_events(
                raw,
                stim_channel=stim_chs,
                min_duration=0.005,
                output="onset",
                shortest_event=1,
            )
            print(f"Detected {len(events)} events from stim channels.")
            return events

        raise RuntimeError(
            "Unable to extract events because no stim channels are present."
        )

    def extract_raw(self, fif_file: str, subject: str) -> dict[str, Any]:
        """Extract raw data and metadata from MNE Raw object with memory efficiency.

        Args:
        fif_file: Path to the fif file
        subject: Subject name

        Returns:
        Dictionary containing raw data and metadata
        """
        data = super().extract_raw(fif_file, subject)
        if "rest" in fif_file:
            return data

        raw = mne.io.read_raw_fif(fif_file, preload=True)
        events = self.extract_events_from_raw(raw)

        for i in range(len(events)):
            events[i, 0] = events[i, 0] // data.get("decimate", 1)

        # create an array with length of data["raw_array"]
        n_samples = data["raw_array"].shape[1]
        event_array = np.zeros(n_samples, dtype=np.int16)

        # loop through events and set with the following logic:
        # if event code is 1, 3, 5, 7, set the event_array to 1
        # with the duration of the event until event code 15 is reached
        # event code 10 maps to 2, until next event
        # event code 20 maps to 3, until next event
        # event code 40 maps to 4, until next event
        # event codes are in events[:, 2], timestamps are in events[:, 0]
        start_codes = {1, 3, 5, 7}
        category_codes = {10: 2, 20: 3, 40: 4}
        fill_val = 1

        if "visual" in fif_file:
            start_codes = {1, 2, 3, 4, 5, 6, 7, 8}
            fill_val = 5

        def clamp_idx(idx: int) -> int:
            """Clamp indices to valid sample range."""
            return max(0, min(n_samples, int(idx)))

        def fill_range(start_idx: int, end_idx: int, value: int) -> None:
            """Fill event_array[start:end] with value, guarding bounds."""
            start, end = clamp_idx(start_idx), clamp_idx(end_idx)
            if end > start and end - start > 5:
                event_array[start:end] = value

        for i, event in enumerate(events):
            sample_idx, _, code = event
            if code in start_codes:
                end_sample = n_samples
                for next_event in events[i + 1 :]:
                    if next_event[2] == 15:
                        end_sample = next_event[0]
                        break
                fill_range(sample_idx, end_sample, fill_val)
            elif code in category_codes:
                next_sample = events[i + 1, 0] if i + 1 < len(events) else n_samples
                fill_range(sample_idx, next_sample, category_codes[code])

        data["event_array"] = event_array
        return data
