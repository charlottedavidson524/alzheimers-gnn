"""
Batch runner for cohort-level EEG preprocessing.

Applies `preprocess_subject` function to a list of subjects. Collects per-subject status dictionaries into a summary 
DataFrame.
"""

from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
from agnn.preprocessing.eeg import preprocess_subject


def preprocess_cohort(subjects: list[str], config: dict, data_root: Path | str, output_root: Path | str, skip_if_exists: bool = True) -> pd.DataFrame:
    """
    Preprocess a list of subjects and return a summary DataFrame.

    Parameters
    ----------
    - subjects : list of str
          BIDS subject identifiers (e.g. ["sub-01", "sub-02", ...])
    - config : dict
          Loaded project config with a preprocessing" section
    - data_root : Path or str
          Root of the PEARL-Neuro dataset
    - output_root : Path or str
          Where preprocessed outputs are written, e.g. `.../derivatives/eeg_preprocessed`
    skip_if_exists : bool, default True
        If True, subjects whose output file already exists are skipped. If False, all subjects are reprocessed from scratch.

    Returns
    -------
    - summary : pd.DataFrame
          One row per subject with the fields from the status dicts, as well as an added "skipped" column for any pre-existing outputs.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    n_total = len(subjects)
    cohort_start_time = time.time()

    for i, sub in enumerate(subjects, start=1):
        # Check for existing output. Also used this pattern in the download script.
        expected_output = (output_root/sub/f"{sub}_task-rest_desc-preprocessed_epo.fif")

        if skip_if_exists and expected_output.exists():
            print(f"[{i}/{n_total}] {sub}: skipped because output already exists)")
            results.append({"subject": sub, "status": "skipped", "output_path": str(expected_output), "skipped": True, "error_message": None})
            continue

        # Process the subject. 
        print(f"[{i}/{n_total}] {sub}: processing")
        subject_start = time.time()
        status = preprocess_subject(subject_id=sub, config=config, data_root=data_root, output_root=output_root)
        status["skipped"] = False
        results.append(status)

        elapsed = time.time()-subject_start
        print(f"[{i}/{n_total}] {sub}: {status['status']} ({elapsed:.0f} s)")

    cohort_elapsed = time.time()-cohort_start_time
    print(f"\nCohort complete: {n_total} subjects in {cohort_elapsed / 60:.1f} min")

    return pd.DataFrame(results)