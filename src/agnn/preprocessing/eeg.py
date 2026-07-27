"""
This is a function that runs the entire EEG preprocessing pipeline on a single subject.

It implememnts the 13 step preprocessing pipeline:
    1. Notch filter (50 Hz)
    2. Bandpass filter (1-45 Hz)
    3. Resample to target sample rate (1000 -> 250 Hz)
    4. Bad-channel detection (LOF, threshold 1.5)
    5. Bad-channel interpolation (spherical spline)
    6. Common Average Reference (CAR)
    7. Fit ICA (Picard with Extended Infomax settings)
    8. Classify components via ICLabel
    9. Remove artefact components and reconstruct signal
    10. Extract eyes-closed segment via S 4 -> S 11 markers
    11. Epoch into 4-second non-overlapping segments
    12. Apply autoreject in "interpolate" mode
    13. Save cleaned epochs to disk

The reasoning for all of the above decisions can be found in the `docs/decisions/eeg_preprocessing` folder.
"""

from __future__ import annotations
import io
import logging
import time
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
from mne_icalabel import label_components

# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
def _rest_vhdr(sub: str, data_root: Path) -> Path:
    """
    Path to a subject's resting-state BrainVision header file.
    """
    return data_root/sub/"eeg"/f"{sub}_task-rest_eeg.vhdr"
 
 
def _rest_events_tsv(sub: str, data_root: Path) -> Path:
    """
    Path to a subject's resting-state BIDS events sidecar.
    """
    return data_root/sub/"eeg"/f"{sub}_task-rest_events.tsv"
 
 
def _eyes_closed_onsets(events_path: Path, start_marker: str, end_marker: str, fallback_duration_s: float = 360.0, recording_end_s: float | None = None) -> tuple[float, float]:
    """
    Return (start, end) times of the eyes-closed condition, in seconds.

    Had to add an extra condition due to some end_marker being missing due to truncated events files.
    """
    events = pd.read_csv(events_path, sep="\t").sort_values("onset").reset_index(drop=True)
    codes = events["event_type"].astype(str).str.strip()

    start_matches = events[codes == start_marker.strip()]
    if len(start_matches) == 0:
        raise ValueError(f"Eyes-closed start marker {start_marker!r} is missing")

    start = float(start_matches["onset"].iloc[0])

    end_matches = events[codes == end_marker.strip()]
    if len(end_matches) > 0:
        end = float(end_matches["onset"].iloc[0])
    else:
        # If S 11 missing then fall back to fixed protocol duration.
        end = start + fallback_duration_s
        if recording_end_s is not None and end > recording_end_s:
            raise ValueError(
                f"Fallback end ({end:.1f}s) longer than recording length "
                f"({recording_end_s:.1f}s). Could be a very short recording."
            )

    if end <= start:
        raise ValueError(f"Eyes-closed end ({end:.1f}s) is not after start ({start:.1f}s)")
    return start, end
 
# ──────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────
def preprocess_subject(subject_id: str, config: dict, data_root: Path | str, output_root: Path | str) -> dict:
    """
    Preprocess one subject's resting-state EEG. The entire pipeline.
 
    Parameters
    ----------
    - subject_id: str
          BIDS subject identifier (e.g. "sub-01")
    - config: dict
          Loaded project config. Needs a "preprocessing" section matching the schema in `config/default.yaml`
    - data_root: Path or str
          Root of the PEARL-Neuro dataset (e.g. `C:/data/pearl-neuro`)
    - output_root: Path or str
          Where to write preprocessed outputs, e.g. `C:/data/pearl-neuro/derivatives/eeg_preprocessed`
 
    Returns
    -------
    - status : dict
          Per-subject summary with counts, ratios, and status. See the module docstring for schema.
    """
    # Setup
    data_root = Path(data_root)
    output_root = Path(output_root)
    subject_out = output_root/subject_id
    subject_out.mkdir(parents=True, exist_ok=True)
 
    output_path = subject_out/f"{subject_id}_task-rest_desc-preprocessed_epo.fif"
    log_path = subject_out/"preprocessing_log.txt"
 
    # Configure a per-subject file logger so MNE/autoreject's verbose output is captured to disk without polluting the batch runner's console output.
    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    mne_logger = mne.utils.logger
    mne_logger.addHandler(log_handler)
 
    # Initialise return dictionary now so partial info is preserved in case of an error
    status: dict = {
        "subject": subject_id,
        "status": "error",
        "output_path": None,
        "eyes_closed_duration_s": None,
        "n_epochs_before_ar": None,
        "n_epochs_after_ar": None,
        "n_epochs_rejected": None,
        "n_bad_channels": None,
        "bad_channel_names": None,
        "bad_channel_ratio": None,
        "n_ica_components_fit": None,
        "n_ica_components_removed": None,
        "ica_categories_removed": None,
        "runtime_seconds": None,
        "error_message": None,
    }
 
    start_time = time.time()
    cfg = config["preprocessing"]
 
    try:
        # Load raw data
        vhdr = _rest_vhdr(subject_id, data_root)
        if not vhdr.exists():
            raise FileNotFoundError(f"Raw file not found: {vhdr}")
        
        events_path = _rest_events_tsv(subject_id, data_root)
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")
 
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
 
        # Apply the standard_1005 montage. match_case=False should handle Brain Products' occasional non-standard capitalisation.
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose="ERROR")

        CHANNELS_WITHOUT_POSITIONS = ["O9", "O10"]
        to_drop = [ch for ch in CHANNELS_WITHOUT_POSITIONS if ch in raw.info["ch_names"]]
        if to_drop:
            raw.drop_channels(to_drop)
 
        # Notch filter (50 Hz)
        raw.notch_filter(freqs=cfg["filter"]["notch_hz"], verbose="ERROR")
 
        # Bandpass filter (1-45 Hz)
        raw.filter(l_freq=cfg["filter"]["highpass_hz"], h_freq=cfg["filter"]["lowpass_hz"], method="fir", verbose="ERROR")
 
        # Resample to target sample rate
        if "resample" in cfg:
            raw.resample(sfreq=cfg["resample"]["target_sfreq"], verbose="ERROR")
 
        # Bad-channel detection (LOF)
        bad_channels = mne.preprocessing.find_bad_channels_lof(raw, threshold=cfg["bad_channels"]["lof_threshold"], verbose="ERROR")
        n_channels = len(raw.ch_names)
        n_bad = len(bad_channels)
        bad_ratio = n_bad/n_channels
 
        status["n_bad_channels"] = n_bad
        status["bad_channel_names"] = list(bad_channels)
        status["bad_channel_ratio"] = float(bad_ratio)
 
        # Subject exclusion check usingHAPPE convention (more than 20% bad -> exclude)
        max_bad_ratio = cfg["bad_channels"]["max_bad_ratio"]

        if bad_ratio > max_bad_ratio:
            status["status"] = "excluded_bad_channels"
            status["error_message"] = f"Bad-channel ratio {bad_ratio:.1%} exceeds threshold {max_bad_ratio:.0%} ({n_bad}/{n_channels} channels). Subject excluded from cohort."
            status["runtime_seconds"] = time.time()-start_time
            log_path.write_text(log_stream.getvalue(), encoding="utf-8")
            return status
 
        raw.info["bads"] = bad_channels
 
        # Bad-channel interpolation (spherical spline)
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
 
        # Common Average Reference
        raw.set_eeg_reference(ref_channels=cfg["reference"], projection=False, verbose="ERROR")
 
        # Fit ICA
        # n_components set to effective rank of the data (channels minus 1 for CAR minus 0 for interpolated channels since interpolation is linear-combination).
        #effective_rank = mne.compute_rank(raw, rank="info", verbose="ERROR")["eeg"]
        effective_rank = int(np.linalg.matrix_rank(raw.get_data()))
 
        ica = mne.preprocessing.ICA(n_components=effective_rank, method=cfg["ica"]["method"], fit_params=cfg["ica"]["fit_params"], random_state=cfg["ica"]["random_state"], max_iter=cfg["ica"]["max_iter"], verbose="ERROR")
        ica.fit(raw, verbose="ERROR")
 
        status["n_ica_components_fit"] = ica.n_components_
 
        # Classify components via ICLabel
        ic_labels = label_components(raw, ica, method="iclabel")
        labels = ic_labels["labels"]  # List of category strings per component
        probs = ic_labels["y_pred_proba"]  # Array of max probabilities
 
        # Identify and remove artefact components
        reject_categories = set(cfg["iclabel"]["reject_categories"])
        threshold = cfg["iclabel"]["reject_threshold"]
 
        components_to_exclude = []
        categories_removed = {cat: 0 for cat in reject_categories}
        for ic_idx, (label, prob) in enumerate(zip(labels, probs)):
            if label in reject_categories and prob > threshold:
                components_to_exclude.append(ic_idx)
                categories_removed[label] += 1
 
        ica.exclude = components_to_exclude
        raw = ica.apply(raw.copy(), verbose="ERROR")
 
        status["n_ica_components_removed"] = len(components_to_exclude)
        status["ica_categories_removed"] = categories_removed
 
        # Extract eyes-closed segment (S 4 -> S 11)
        #events_path = _rest_events_tsv(subject_id, data_root)
        #if not events_path.exists():
            #raise FileNotFoundError(f"Events file not found: {events_path}")
 
        #ec_start, ec_end = _eyes_closed_onsets(events_path, start_marker=cfg["events"]["eyes_closed_start_marker"], end_marker=cfg["events"]["eyes_closed_end_marker"])
        ec_start, ec_end = _eyes_closed_onsets(events_path, start_marker=cfg["events"]["eyes_closed_start_marker"], end_marker=cfg["events"]["eyes_closed_end_marker"], fallback_duration_s=280.0, recording_end_s=raw.times[-1])
        raw_ec = raw.copy().crop(tmin=ec_start, tmax=ec_end)
        status["eyes_closed_duration_s"] = float(raw_ec.times[-1])
 
        # Epoch into 4-second non-overlapping segments
        epochs = mne.make_fixed_length_epochs(raw_ec, duration=cfg["epoching"]["duration_s"], overlap=cfg["epoching"]["overlap"], preload=True, verbose="ERROR")
        status["n_epochs_before_ar"] = len(epochs)
 
        # Apply autoreject in interpolate mode
        ar = AutoReject(n_interpolate=cfg["autoreject"]["n_interpolate"], consensus=cfg["autoreject"]["consensus"], thresh_method="bayesian_optimization", random_state=cfg["autoreject"]["random_state"], n_jobs=1, verbose=False)
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
 
        status["n_epochs_after_ar"] = len(epochs_clean)
        status["n_epochs_rejected"] = int(np.sum(reject_log.bad_epochs))
 
        # Save cleaned epochs
        epochs_clean.save(output_path, overwrite=True, verbose="ERROR")
 
        status["output_path"] = str(output_path)
        status["status"] = "success"
 
    except Exception as e:
        # Fail gracefully: return partial status with error message so the batch runner can continue with remaining subjects.
        status["status"] = "error"
        status["error_message"] = f"{type(e).__name__}: {e}"
    finally:
        status["runtime_seconds"] = time.time() - start_time
 
        # Persist the captured log regardless of outcome.
        log_stream.write(f"\n\n=== Preprocessing status: {status['status']} ===\nRuntime: {status['runtime_seconds']:.1f} s\n")
        if status["error_message"]:
            log_stream.write(f"Error: {status['error_message']}\n")

        log_path.write_text(log_stream.getvalue(), encoding="utf-8")
 
        # Detach the log handler so it doesn't leak between subjects.
        mne_logger.removeHandler(log_handler)
        log_handler.close()
 
    return status