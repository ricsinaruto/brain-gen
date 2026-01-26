from typing import Any
from .ephys import Ephys
from pathlib import Path


class Omega(Ephys):
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
            subject_id = subject_dir.name.split("-")[1][1:]
            print(f"INFO: Found subject {subject_id}")

            # List session folders within subject
            session_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]

            # Iterate through each session
            for session_dir in session_dirs:
                session_id = session_dir.name.split("-")[1]
                print(f"INFO: Found session {session_id}")

                # Find .ds folders and categorize by task type
                meg_dir = session_dir / "meg"
                ds_folders = list(meg_dir.glob("*.ds"))

                for ds_folder in ds_folders:
                    # Extract task type from folder name
                    folder_name = ds_folder.name.lower()
                    task_map = {
                        "noise": "noise",
                        "restafter": "restaftertask",
                        "rest": "rest",
                    }

                    task_type = next(
                        (v for k, v in task_map.items() if k in folder_name), None
                    )
                    if task_type is None or task_type == "noise":
                        print(f"Warning: Unknown task type in folder {ds_folder}")
                        continue

                    # Create base subject name
                    subject_name = f"sub-{subject_id}_ses-{session_id}_{task_type}"

                    # Add index if this subject_name already exists
                    if subject_name in subject_counts:
                        subject_counts[subject_name] += 1
                        subject_name = (
                            f"{subject_name}-{subject_counts[subject_name]:02d}"
                        )
                    else:
                        subject_counts[subject_name] = 0

                    subjects.append(subject_name)
                    files.append(str(ds_folder))

        self.batch_args = {"files": files, "subjects": subjects}
