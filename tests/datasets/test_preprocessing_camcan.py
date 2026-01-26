from pathlib import Path

import dask
import numpy as np
import mne

from brain_gen.preprocessing import base as base_module
from brain_gen.preprocessing import ephys as ephys_module
from brain_gen.preprocessing.camcan import CamCAN, CamCANConditioned
from brain_gen.preprocessing.ephys import Ephys


def _make_raw(fif_path, sfreq: float = 100.0, n_times: int = 200) -> None:
    info = mne.create_info(
        ["MEG0111", "MEG0122"],
        sfreq,
        ch_types=["mag", "grad"],
    )
    raw = mne.io.RawArray(np.zeros((2, n_times)), info)
    raw.save(fif_path, overwrite=True)


def test_camcan_load_discovers_cc280_and_cc700(tmp_path):
    root = tmp_path / "camcan"

    cc700_meg = root / "cc700" / "syntask" / "sub-CC7001" / "ses-001" / "meg"
    cc700_meg.mkdir(parents=True)
    (cc700_meg / "sub-CC7001_ses-001_task-rest_run-01_meg.fif").touch()

    cc280_meg = root / "cc280" / "rest" / "sub-CC2801" / "meg"
    cc280_meg.mkdir(parents=True)
    (cc280_meg / "sub-CC2801_task-rest_meg.fif").touch()

    loader = CamCAN(
        str(root),
        use_dask=False,
        gen_report=False,
    )

    assert len(loader.batch_args["files"]) == 2
    assert any("cc700" in subj for subj in loader.batch_args["subjects"])
    assert any("cc280" in subj for subj in loader.batch_args["subjects"])


def test_camcan_conditioned_builds_event_array_from_tsv(tmp_path):
    base = tmp_path / "cc700" / "syntask" / "sub-CC0001" / "meg"
    base.mkdir(parents=True)

    fif_path = base / "sub-CC0001_task-smt_run-01_meg.fif"
    _make_raw(fif_path)

    events_tsv = base / "sub-CC0001_task-smt_run-01_events.tsv"
    events_tsv.write_text(
        "onset\tduration\ttrial_type\n0.1\t0.2\tauditory\n0.6\t\tvisual\n"
    )

    loader = CamCANConditioned(
        str(tmp_path / "cc700"),
        use_dask=False,
        gen_report=False,
    )
    data = loader.extract_raw(str(fif_path), "sub-CC0001")
    event_array = data["event_array"]

    assert event_array.shape[0] == data["raw_array"].shape[1]
    assert np.all(event_array[10:30] == 1)  # 0.1s onset, 0.2s duration @ 100 Hz
    assert event_array[60] == 2


def test_preprocess_stage_runs_maxwell_and_updates_batch_args(tmp_path, monkeypatch):
    root = tmp_path / "camcan"
    meg_dir = root / "cc700" / "syntask" / "sub-CC7001" / "meg"
    meg_dir.mkdir(parents=True)
    fif_path = meg_dir / "sub-CC7001_task-rest_meg.fif"
    fif_path.touch()

    saved: dict[str, object] = {}

    class DummySSS:
        def save(self, path, overwrite=True):
            saved["saved_path"] = Path(path)

    dummy_sss = DummySSS()

    def fake_read_raw_fif(path, allow_maxshield=True, preload=True):
        saved["read_path"] = Path(path)
        return "RAW"

    def fake_apply(self, raw):
        saved["applied_raw"] = raw
        return dummy_sss

    def fake_run_proc_batch(osl_config, files, subjects, outdir, **kwargs):
        saved["run_proc"] = {"files": files, "subjects": subjects, "outdir": outdir}

    monkeypatch.setattr(mne.io, "read_raw_fif", fake_read_raw_fif)
    monkeypatch.setattr(Ephys, "_apply_maxwell_filter", fake_apply, raising=False)
    monkeypatch.setattr(base_module, "run_proc_batch", fake_run_proc_batch)

    loader = CamCAN(
        str(root),
        use_dask=False,
        gen_report=False,
        maxwell=True,
    )
    subject = loader.batch_args["subjects"][0]
    expected_out = Path(loader.save_folder) / f"{Path(subject).name}_maxwell-raw.fif"

    loader.preprocess_stage_1()

    assert saved["read_path"] == fif_path
    assert saved["applied_raw"] == "RAW"
    assert saved["saved_path"] == expected_out
    assert saved["run_proc"]["files"] == [str(expected_out)]
    assert saved["run_proc"]["subjects"] == [subject]
    assert saved["run_proc"]["outdir"] == loader.save_folder


def test_preprocess_stage_2_uses_dask_compute(tmp_path, monkeypatch):
    calls = {}
    orig_compute = dask.compute

    def wrapped_compute(*args, **kwargs):
        calls["kwargs"] = kwargs
        return orig_compute(*args, **kwargs)

    monkeypatch.setattr(base_module.dask, "compute", wrapped_compute)

    class DummyPreprocessing(base_module.Preprocessing):
        def load(self):
            self.batch_args = {"files": [], "subjects": ["sub-01", "sub-02"]}

        def extract_raw(self, fif_file: str, subject: str):
            return {
                "raw_array": np.zeros((2, 10)),
                "sfreq": 1.0,
                "ch_names": ["A", "B"],
                "ch_types": ["mag", "grad"],
                "pos_2d": np.zeros((2, 2)),
                "session": subject,
            }

    data_path = tmp_path / "data"
    data_path.mkdir()
    loader = DummyPreprocessing(
        str(data_path),
        preproc_config={},
        use_dask=True,
        gen_report=False,
        delete_fif=False,
    )

    for subject in loader.batch_args["subjects"]:
        subject_dir = Path(loader.stage1_path) / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        (subject_dir / f"{subject}_preproc-raw.fif").touch()

    loader.preprocess_stage_2()

    assert calls["kwargs"]["scheduler"] == "processes"
    assert calls["kwargs"]["num_workers"] == loader.n_workers
    for subject in loader.batch_args["subjects"]:
        session_dir = Path(loader.save_folder) / subject
        assert any(path.suffix == ".npy" for path in session_dir.iterdir())


def test_dask_client_context_reuses_default(tmp_path, monkeypatch):
    created = {}

    class DummyClient:
        def close(self):
            created["closed"] = True

    def fake_default_client():
        created["default"] = True
        return DummyClient()

    def fake_client(*args, **kwargs):
        created["new"] = True
        return DummyClient()

    monkeypatch.setattr(base_module, "default_client", fake_default_client)
    monkeypatch.setattr(base_module, "Client", fake_client)

    class DummyPreprocessing(base_module.Preprocessing):
        def load(self):
            self.batch_args = {"files": [], "subjects": []}

        def extract_raw(self, fif_file: str, subject: str):
            return {}

    loader = DummyPreprocessing(
        str(tmp_path),
        preproc_config={},
        use_dask=True,
        gen_report=False,
    )

    with loader._dask_client() as client:
        assert isinstance(client, DummyClient)

    assert "new" not in created
    assert "closed" not in created


def test_dask_client_context_closes_created(tmp_path, monkeypatch):
    created = {}

    class DummyClient:
        def close(self):
            created["closed"] = True

    def fake_default_client():
        raise ValueError("no client")

    def fake_client(*args, **kwargs):
        created["new"] = True
        return DummyClient()

    monkeypatch.setattr(base_module, "default_client", fake_default_client)
    monkeypatch.setattr(base_module, "Client", fake_client)

    class DummyPreprocessing(base_module.Preprocessing):
        def load(self):
            self.batch_args = {"files": [], "subjects": []}

        def extract_raw(self, fif_file: str, subject: str):
            return {}

    loader = DummyPreprocessing(
        str(tmp_path),
        preproc_config={},
        use_dask=True,
        gen_report=False,
    )

    with loader._dask_client() as client:
        assert isinstance(client, DummyClient)

    assert created["new"] is True
    assert created["closed"] is True


def test_preprocess_stage_1_caps_dask_workers(tmp_path, monkeypatch):
    created = {}

    class DummyClient:
        def close(self):
            created["closed"] = True

    def fake_default_client():
        raise ValueError("no client")

    def fake_client(*args, **kwargs):
        created["kwargs"] = kwargs
        return DummyClient()

    def fake_run_proc_batch(*args, **kwargs):
        created["run_proc"] = True

    monkeypatch.setattr(base_module, "default_client", fake_default_client)
    monkeypatch.setattr(base_module, "Client", fake_client)
    monkeypatch.setattr(base_module, "run_proc_batch", fake_run_proc_batch)

    class DummyPreprocessing(base_module.Preprocessing):
        def load(self):
            self.batch_args = {
                "files": ["f1.fif", "f2.fif"],
                "subjects": ["s1", "s2"],
            }

        def extract_raw(self, fif_file: str, subject: str):
            return {}

    loader = DummyPreprocessing(
        str(tmp_path),
        preproc_config={},
        use_dask=True,
        n_workers=8,
        gen_report=False,
    )

    loader.preprocess_stage_1()

    assert created["kwargs"]["n_workers"] == 2
    assert created["run_proc"] is True
