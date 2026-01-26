"""Base classes for ephys recordings."""

import warnings
from pathlib import Path
from typing import Any
from scipy import signal
from functools import lru_cache
import mne
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from mne.io.constants import FIFF
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from osl_ephys import preprocessing
import os

from ..utils.quantizers import (
    mulaw,
    predictive_mulaw_decode,
    predictive_residual_encode,
    quantize_deadzone_linear,
)
from ..utils.utils import compute_roi_layout_2d

from .base import Preprocessing
from .maxwell import apply_maxwell_filter, process_maxwell_file

# --- Coreg params (tune once, keep fixed) ---
HSP_MIN = 8

GROW_HAIR_MM = 5.0  # IMPORTANT: set_grow_hair() takes mm (not meters)
OMIT_COARSE_M = 0.03  # meters
OMIT_FINE_M = 0.02  # meters
ICP_COARSE_ITERS = 10
ICP_FINE_ITERS = 15

# Robust QC thresholds (template-MRI coreg is not subject-MRI coreg)
QC_MED_MAX_M = 0.012  # 12 mm
QC_P95_MAX_M = 0.025  # 25 mm
QC_TRIM_DIST_M = 0.02  # 20 mm (match omit threshold)
QC_MAX_TRIM_FRACTION = 0.50  # allow up to 30% points trimmed before rejecting


def _coreg_rigid(raw, hsp, subject: str, subjects_dir: str) -> mne.coreg.Coregistration:
    """Rigid (no scaling) fsaverage coreg with hair + robust ICP."""
    coreg = mne.coreg.Coregistration(
        info=raw.info, subject=subject, subjects_dir=subjects_dir
    )

    coreg.fit_fiducials()

    if hsp is not None and len(hsp) >= HSP_MIN:
        # Coarse -> fine: helps avoid a few far points dominating the fit
        coreg.omit_head_shape_points(distance=OMIT_COARSE_M)
        # When using template fiducials, downweight nasion a bit initially
        coreg.fit_icp(n_iterations=ICP_COARSE_ITERS, nasion_weight=2.0)

        coreg.omit_head_shape_points(distance=OMIT_FINE_M)
        coreg.fit_icp(n_iterations=ICP_FINE_ITERS, nasion_weight=10.0)

    return coreg


def _qc_coreg(raw, trans, subject: str, subjects_dir: str):
    """Robust QC:

    - exclude_frontal=True to reduce nose/face mismatch effects - trimmed stats to avoid
    mean being dominated by a few bad points
    """
    try:
        d = np.asarray(
            mne.dig_mri_distances(
                raw.info,
                trans,
                subject,
                subjects_dir=subjects_dir,
                exclude_frontal=True,
            )
        )
    except ValueError as exc:
        print(f"WARNING: Failed to compute dig_mri_distances: {exc}")
        return dict(ok=True)

    if d.size == 0:
        # No points to evaluate: treat as failure
        return dict(ok=False, reason="no_dig_points", d=d)

    keep = d < QC_TRIM_DIST_M
    d_trim = d[keep]
    trim_fraction = 1.0 - float(np.mean(keep))

    if d_trim.size == 0:
        return dict(ok=False, reason="all_points_trimmed", d=d, d_trim=d_trim)

    med = float(np.median(d_trim))
    p95 = float(np.percentile(d_trim, 95))
    mean_trim = float(np.mean(d_trim))

    ok = (
        (med <= QC_MED_MAX_M)
        and (p95 <= QC_P95_MAX_M)
        and (trim_fraction <= QC_MAX_TRIM_FRACTION)
    )

    return dict(
        ok=ok,
        med=med,
        p95=p95,
        mean_trim=mean_trim,
        trim_fraction=trim_fraction,
        d=d,
        d_trim=d_trim,
    )


class Ephys(Preprocessing):
    # OSL pipeline config for basic preprocessing after optional Maxwell filtering
    def __init__(
        self,
        *args,
        maxwell: bool = False,
        source_space: bool = False,
        sfreq: int = None,
        get_fsaverage_data: bool = False,
        residual_scale: float = 1.0,
        manual_bad_chn: str = "",
        run_bad_chn: bool = False,
        source_chunk_seconds: float = None,
        fsaverage_dir: str = "/vol/data/datasets/mne_data",
        label_mode: str = "mean_flip",  # "pca_flip" or "mean_flip"
        **kwargs,
    ) -> None:
        """Args:

        maxwell: Whether to apply a pre-stage Maxwell filter
        """
        super().__init__(*args, **kwargs)

        self.maxwell = maxwell
        self.residual_scale = residual_scale
        self.source_space = source_space
        self.sfreq = sfreq
        self.manual_bad_chn = manual_bad_chn
        self.run_bad_chn = run_bad_chn
        self.source_chunk_seconds = source_chunk_seconds
        self.label_mode = label_mode

        if get_fsaverage_data:
            fetch_fsaverage(Path(fsaverage_dir))

        self.subjects_dir = Path(fsaverage_dir)

    def _apply_maxwell_filter(self, raw):
        """Apply bad channel detection, head position, and Maxwell filter."""
        return apply_maxwell_filter(raw, self.save_folder)

    def _process_maxwell_filter_subject(
        self, file_path: str, subject: str
    ) -> tuple[str, str] | None:
        return process_maxwell_file(
            file_path,
            subject,
            self.save_folder,
            self.skip_done,
            apply_fn=self._apply_maxwell_filter,
        )

    def _run_maxwell_filter(self) -> None:
        """Run Maxwell filtering for all inputs and swap in the filtered files."""
        files = self.batch_args.get("files", [])
        subjects = self.batch_args.get("subjects", [])
        if not files or not subjects:
            print("WARNING: No files/subjects available for Maxwell filtering.")
            return

        out_dir = Path(self.save_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = [
            self._process_maxwell_filter_subject(file_path, subject)
            for file_path, subject in zip(files, subjects)
        ]

        filtered_files: list[str] = []
        filtered_subjects: list[str] = []
        for result in results:
            if result is None:
                continue
            out_path, subject = result
            filtered_files.append(out_path)
            filtered_subjects.append(subject)

        self.batch_args["files"] = filtered_files
        self.batch_args["subjects"] = filtered_subjects

    def preprocess_stage_1(self) -> None:
        if self.maxwell:
            self._run_maxwell_filter()
            if not self.batch_args.get("files"):
                print("INFO: No files to preprocess after Maxwell filtering.")
                return
        super().preprocess_stage_1()

    def find_events_safe(
        self, data: dict[str, Any], min_duration: float = 0.005
    ) -> dict[str, Any]:
        """Find events if stim channels are present, otherwise continue without events.

        Args:     data: Dictionary containing raw MNE object     min_duration: Minimum
        duration between events in seconds
        """
        raw = data["raw"]

        # Try to find stim channels
        stim_picks = mne.pick_types(raw.info, stim=True)

        if len(stim_picks) > 0:
            try:
                # Attempt to find events
                events = mne.find_events(raw, min_duration=min_duration)
                data["events"] = events
                data["has_events"] = True
                print(f"INFO: Found {len(events)} events")
            except Exception as e:
                print(f"WARNING: Failed to find events: {str(e)}")
                data["has_events"] = False
        else:
            print("INFO: No stim channels found, continuing without events")
            data["has_events"] = False

        return data

    @lru_cache(maxsize=64)
    def _fsaverage_labels(self, parc: str = "aparc"):
        labels = mne.read_labels_from_annot(
            "fsaverage", parc=parc, subjects_dir=self.subjects_dir
        )
        labels = [
            lab
            for lab in labels
            if "unknown" not in lab.name.lower()
            and "corpuscallosum" not in lab.name.lower()
        ]
        labels = sorted(labels, key=lambda lab: lab.name)
        return labels

    def _extract_fids_and_hsp(self, raw: mne.io.BaseRaw):
        digs = raw.info.get("dig", None)
        if not digs:
            raise RuntimeError("No digitization points found in raw.info['dig'].")

        nas = lpa = rpa = None
        hsp = []
        hsp_eeg = []

        # has_eeg_chs = len(mne.pick_types(raw.info, eeg=True)) > 0

        for d in digs:
            if d["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
                continue

            if d["kind"] == FIFF.FIFFV_POINT_CARDINAL:
                if d["ident"] == FIFF.FIFFV_POINT_NASION:
                    nas = d["r"]
                elif d["ident"] == FIFF.FIFFV_POINT_LPA:
                    lpa = d["r"]
                elif d["ident"] == FIFF.FIFFV_POINT_RPA:
                    rpa = d["r"]

            # Prefer true headshape points if present
            elif d["kind"] == FIFF.FIFFV_POINT_EXTRA:
                hsp.append(d["r"])

            # Only use EEG points as headshape if you don't actually have EEG sensors
            elif d["kind"] == FIFF.FIFFV_POINT_EEG:
                hsp_eeg.append(d["r"])

        # if no extra points, use EEG points as headshape
        if len(hsp) == 0:
            hsp = hsp_eeg

        if nas is None or lpa is None or rpa is None:
            raise RuntimeError("Missing fiducials (NAS/LPA/RPA) in digitization.")

        hsp = np.array(hsp) if len(hsp) else None
        return nas, lpa, rpa, hsp

    @lru_cache(maxsize=64)
    def _fsaverage_fwd_assets(
        self, subject: str = "fsaverage", spacing: str = "ico5", bem_ico: int = 4
    ):
        src = mne.setup_source_space(
            subject, spacing=spacing, subjects_dir=self.subjects_dir, add_dist=False
        )
        bem_model = mne.make_bem_model(
            subject, ico=bem_ico, conductivity=[0.3], subjects_dir=self.subjects_dir
        )
        bem = mne.make_bem_solution(bem_model)
        return src, bem

    def _robust_coreg(self, raw, hsp):
        # ----------------------------
        # 1) First attempt: rigid fsaverage
        # ----------------------------
        template_subject = "fsaverage"
        coreg = _coreg_rigid(raw, hsp, template_subject, self.subjects_dir)
        trans = coreg.trans

        qc = _qc_coreg(raw, trans, template_subject, self.subjects_dir)

        # ----------------------------
        # 2) If QC fails and we have enough headshape, try uniform scaling
        #    (estimate scale on fsaverage, create scaled surrogate, redo rigid coreg)
        # ----------------------------
        if (not qc["ok"]) and (hsp is not None) and (len(hsp) >= HSP_MIN):
            # Estimate scale factors using Coregistration's scale mode
            coreg_scale = mne.coreg.Coregistration(
                info=raw.info, subject="fsaverage", subjects_dir=self.subjects_dir
            )
            coreg_scale.set_grow_hair(GROW_HAIR_MM)
            coreg_scale.set_scale_mode("uniform")  # estimate one scale factor
            coreg_scale.fit_fiducials()

            coreg_scale.omit_head_shape_points(distance=OMIT_COARSE_M)
            coreg_scale.fit_icp(n_iterations=20, nasion_weight=2.0)
            coreg_scale.omit_head_shape_points(distance=OMIT_FINE_M)
            coreg_scale.fit_icp(n_iterations=20, nasion_weight=10.0)

            scale = float(np.mean(coreg_scale.scale))  # scale is dimensionless
            scale_bin = round(scale, 2)  # bin to avoid generating tons of subjects

            # Only bother if scale differs meaningfully from 1
            if abs(scale_bin - 1.0) > 1e-3:
                scaled_subject = f"fsaverage_scale_{scale_bin:.2f}"
                scaled_path = os.path.join(self.subjects_dir, scaled_subject)

                # Create the scaled subject once (safe if already exists)
                if not os.path.isdir(scaled_path):
                    try:
                        mne.scale_mri(
                            subject_from="fsaverage",
                            subject_to=scaled_subject,
                            scale=scale_bin,
                            subjects_dir=self.subjects_dir,
                            annot=True,
                            overwrite=False,
                        )
                    except Exception:
                        # In parallel runs, another worker may create it first
                        if not os.path.isdir(scaled_path):
                            raise

                # Now redo a *rigid* coreg to the scaled surrogate
                template_subject = scaled_subject
                coreg = _coreg_rigid(raw, hsp, template_subject, self.subjects_dir)
                trans = coreg.trans
                qc = _qc_coreg(raw, trans, template_subject, self.subjects_dir)

        # ----------------------------
        # 3) Final QC gate (robust, template-aware)
        # ----------------------------
        if not qc.get("ok", False):
            # Report in mm for readability
            msg = (
                f"Coreg is poor for template={template_subject}. "
                f"median={qc.get('med', np.nan) * 1e3:.2f} mm, "
                f"p95={qc.get('p95', np.nan) * 1e3:.2f} mm, "
                f"trimmed_mean={qc.get('mean_trim', np.nan) * 1e3:.2f} mm, "
                f"trim_fraction={qc.get('trim_fraction', np.nan) * 100:.1f}%"
            )
            raise RuntimeError(msg)

        # NOTE: don't overwrite this for every subject/session
        # mne.write_trans("to_fsaverage-trans.fif", trans, overwrite=True)

        return template_subject, trans

    def source_space_proj(
        self,
        raw,
        subject: str,
        parc: str = "aparc",  # Desikan-Killiany (68)
        spacing: str = "ico5",  # ~10k verts/hemis
        snr: float = 3.0,  # inverse regularization
    ):
        label_mode = self.label_mode
        sfreq = raw.info["sfreq"]

        nas, lpa, rpa, hsp = self._extract_fids_and_hsp(raw)
        montage = mne.channels.make_dig_montage(
            nasion=nas, lpa=lpa, rpa=rpa, hsp=hsp, coord_frame="head"
        )
        raw.set_montage(montage, on_missing="ignore")

        # --- coregister to fsaverage (MRI-less), conservative ICP ---
        template_subject, trans = self._robust_coreg(raw, hsp)

        # --- forward & inverse on chosen template subject
        # You MUST use the same template_subject here that you used for QC.
        src, bem = self._fsaverage_fwd_assets(subject=template_subject, spacing=spacing)

        fwd = mne.make_forward_solution(
            raw.info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=3.0
        )

        noise_cov = mne.make_ad_hoc_cov(raw.info)
        inv = make_inverse_operator(
            raw.info, fwd, noise_cov, loose=0.2, depth=0.8, rank="info"
        )
        lambda2 = 1.0 / (snr**2)

        # --- parcel time series (stable, low-dimensional) ---
        labels = self._fsaverage_labels(parc=parc)

        chunk_samples = None
        if self.source_chunk_seconds is not None:
            chunk_samples = int(round(self.source_chunk_seconds * sfreq))
            print(f"INFO: Using chunk size of {chunk_samples} samples")

        stc = apply_inverse_raw(
            raw,
            inv,
            lambda2=lambda2,
            method="dSPM",
            pick_ori="normal",
            buffer_size=chunk_samples,
        )
        ts = mne.extract_label_time_course(stc, labels, src, mode=label_mode)

        # Post-processing
        # 1) detrend
        ts = signal.detrend(ts, axis=1, type="linear")

        # 2) resample
        if self.sfreq is not None:
            gcd = int(np.gcd(int(round(sfreq)), int(round(self.sfreq))))
            up = int(round(self.sfreq / gcd))
            down = int(round(sfreq / gcd))
            ts = signal.resample_poly(ts, up, down, axis=1, padtype="line")
        else:
            self.sfreq = sfreq

        pos2d = compute_roi_layout_2d(labels, src)

        # assemble dict
        data = {
            "raw_array": ts,
            "sfreq": self.sfreq,
            "ch_names": [f"src{i}" for i in range(ts.shape[0])],
            "ch_types": ["parcel"] * ts.shape[0],
            "pos_2d": pos2d,
            "session": subject,
            "decimate": int(sfreq / self.sfreq),
        }
        return data

    def _detect_bad_channels(self, raw):
        """Run bad channel detection while excluding reference channels."""
        bads: set[str] = set(raw.info.get("bads", []))
        for picks in ("mag", "grad"):
            try:
                tmp = preprocessing.osl_wrappers.bad_channels(
                    raw.copy(), picks=picks, ref_meg=False
                )
                bads.update(tmp.info.get("bads", []))
            except Exception as exc:  # pragma: no cover - detection failures are logged
                print(f"WARNING: bad channel detection failed for picks={picks}: {exc}")
                continue
        raw.info["bads"] = list(bads)

        return raw

    def extract_raw(self, fif_file: str, subject: str) -> dict[str, Any]:
        """Extract raw data and metadata from MNE Raw object with memory efficiency.

        Args:
        fif_file: Path to the fif file
        subject: Subject name

        Returns:
        Dictionary containing raw data and metadata
        """
        data = {}
        raw = mne.io.read_raw_fif(fif_file, preload=True)

        # do bad channel detection with osl-ephys
        if self.run_bad_chn:
            raw = self._detect_bad_channels(raw)
        if self.manual_bad_chn == "interpolate":
            raw = raw.copy().interpolate_bads(reset_bads=False)

        if self.source_space:
            return self.source_space_proj(raw, subject)

        # keep only the MEG channels (drop reference channels early)
        keep_chn = [
            raw.ch_names[idx]
            for idx in mne.pick_types(raw.info, meg=True, ref_meg=False)
        ]
        raw.pick(picks=keep_chn)

        # exclude session if more than 10% of channels are bad
        num_bads = len(raw.info["bads"])
        num_channels = len(raw.ch_names)
        if num_bads > 0.1 * num_channels:
            print(
                f"INFO: Excluding session {subject} because more than "
                f"{num_bads} of {num_channels} channels are bad"
            )
            return None

        # Use memory-efficient data loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data["raw_array"] = raw.get_data()

        # Extract metadata
        data["sfreq"] = raw.info["sfreq"]
        data["ch_names"] = raw.ch_names
        data["ch_types"] = [
            raw.info["chs"][idx]["kind"] for idx in range(len(raw.ch_names))
        ]
        if self.manual_bad_chn == "zero":
            data["bad_channels"] = raw.info["bads"]

        # Get 2D sensor positions
        layout = mne.channels.find_layout(raw.info)

        # Filter layout to only keep position of channels in ch_names
        positions = []
        positions_3d = []
        orientations = []
        for ch_name in data["ch_names"]:
            pos = layout.pos[layout.names.index(ch_name.split("-")[0])]
            positions.append(pos[:2])
            ch_info = raw.info["chs"][raw.ch_names.index(ch_name)]
            loc = ch_info.get("loc", np.zeros(12, dtype=float))
            positions_3d.append(loc[:3])

            # MNE stores a 3x3 rotation matrix in loc[3:], columns (ex, ey, ez).
            # The coil "normal" aligns with the ez column for MEG sensors.
            ori_vec = np.array(loc[9:12], dtype=float)
            norm = np.linalg.norm(ori_vec)
            orientations.append(ori_vec / norm if norm > 0 else ori_vec)
        data["pos_2d"] = np.array(positions)
        data["pos_3d"] = np.array(positions_3d)
        data["ori_3d"] = np.array(orientations)

        # Get bad channels
        # data["bad_chs"] = raw.info["bads"]

        data["session"] = subject

        return data

    def normalize(self, data, method: str = "robust") -> dict[str, Any]:
        """Memory-efficient normalization using batches.

        Args:     data: Dictionary containing raw data and metadata     method: Method
        to use for normalization

        Returns:     Dictionary containing normalized data
        """
        cont_data = data["raw_array"]

        if method == "robust":
            scaler = RobustScaler(with_centering=True, with_scaling=True)
        else:
            scaler = StandardScaler()

        # Fit on transposed data for channel-wise normalization
        cont_data = scaler.fit_transform(cont_data.T).T

        # Store normalization parameters
        if method == "robust":
            data["scaler_centers"] = scaler.center_
            data["scaler_scales"] = scaler.scale_
        data["raw_array"] = cont_data

        return data

    def clip(self, data: dict[str, Any], std_factor: float = 3) -> dict[str, Any]:
        """Outlier clipping over the whole data array at once.

        Args:     data: Dictionary containing raw data and metadata     std_factor:
        Factor to multiply the standard deviation by

        Returns:     Dictionary containing clipped data
        """
        arr = data["raw_array"]
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True)
        lower_bound = mean - std * std_factor
        upper_bound = mean + std * std_factor

        n_clipped = np.sum((arr < lower_bound) | (arr > upper_bound))
        np.clip(arr, lower_bound, upper_bound, out=arr)
        percent_clipped = (n_clipped / arr.size) * 100
        print(f"INFO: {percent_clipped:.2f}% of data clipped")
        data["clipped_percent"] = percent_clipped
        data["raw_array"] = arr
        return data

    def mulaw_quantize(self, data: dict[str, Any], n_bits: int = 8) -> dict[str, Any]:
        """Args: data: Dictionary containing raw data and metadata n_bits: Number of
        bits to use for quantization.

        Returns:     Dictionary containing quantized data
        """
        # first do max scaling for the data to be in -1, 1 range
        max_val = np.max(np.abs(data["raw_array"]))
        data["raw_array"] = data["raw_array"] / max_val

        mu = 2**n_bits - 1
        quant, recon = mulaw(data["raw_array"], mu)

        # compute the mean squared error
        mse = np.mean((data["raw_array"] - recon) ** 2)
        print(f"INFO: Mean squared error of quantization: {mse}")
        data["mse"] = mse

        data["raw_array"] = quant
        return data

    @staticmethod
    def decode_predictive_tokens(
        tokens: np.ndarray,
        *,
        scale: float | np.ndarray = 1.0,
        mu: int = 255,
    ) -> np.ndarray:
        """Inverse µ-law tokens and integrate predictor residuals."""
        return predictive_mulaw_decode(tokens, scale=scale, mu=int(mu))

    # Override stage-3 chunk quantization to use predictor residuals + µ-law
    def _quantize_chunks(self, src_dir: str, dst_dir: str, chunk_files: list[str]):
        """Quantize chunks using predictive residuals + µ-law compression."""
        residuals: list[np.ndarray] = []
        chunk_records: list[tuple[str, dict[str, Any], np.ndarray]] = []

        for chunk_name in chunk_files:
            chunk_path = Path(src_dir) / chunk_name
            try:
                chunk = np.load(chunk_path, allow_pickle=True).item()
            except Exception as exc:  # pragma: no cover - IO errors logged
                print(f"WARNING: Failed to load {chunk_path}: {exc}")
                continue

            if not isinstance(chunk, dict) or "data" not in chunk:
                print(f"WARNING: {chunk_path} missing 'data'; skipping")
                continue

            data = np.asarray(chunk["data"], dtype=np.float32)
            residual = predictive_residual_encode(data)
            residuals.append(residual)
            chunk_records.append((chunk_name, chunk, residual))

        if not residuals:
            print("INFO: No valid chunks to quantize.")
            return

        residual_cat = np.concatenate(residuals, axis=-1)

        # print percentage of residuals clipped
        n_clipped = np.sum(np.abs(residual_cat) > self.residual_scale)
        print(f"INFO: {n_clipped / residual_cat.size * 100:.2f}% of residuals clipped")

        for chunk_name, chunk, residual in chunk_records:
            res_norm = residual / self.residual_scale
            res_norm = np.clip(res_norm, -0.99, 0.99)
            # quant, recon = mulaw(res_norm, mu)
            quant = quantize_deadzone_linear(res_norm, 0.01, bins=self.text_num_bins)

            chunk["data"] = quant.astype(np.uint8, copy=False)
            chunk["residual_scale"] = self.residual_scale
            # chunk["mulaw_mu"] = int(mu)
            # chunk["mulaw_mse"] = float(np.mean((res_norm - recon) ** 2))

            out_path = Path(dst_dir) / chunk_name
            np.save(out_path, chunk)
