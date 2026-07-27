"""
Batch runner for cohort level graph construction.

Applies build_subject_graphs to a list of subjects, catches per subject errors and returns a summary DataFrame.
"""

from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
from agnn.connectivity.build_graphs import build_subject_graphs


def build_cohort_graphs(subjects: list[str], config: dict, preprocessed_root: Path | str, output_root: Path | str, apoe_labels: dict[str, int] | None = None, skip_if_exists: bool = True) -> pd.DataFrame:
    """
    Build graphs for a list of subjects. Return a summary DataFrame.

    Parameters
    ----------
    - subjects: list of str
          BIDS subject identifiers
    - config: dict
          Loaded config with connectivity section
    - preprocessed_root: Path or str
          Root of preprocessed epochs
    - output_root: Path or str
          Where graph .pt files are written
    - apoe_labels: dict or None
          {subject_id: 0/1}. If None then the graphs are unlabelled. Shouldn't be an issue.
    - skip_if_exists: bool
          If True, skip subjects whose graph .pt file already exists
    """
    # Set up paths for reading preprocessed epochs and writing graph outputs
    preprocessed_root = Path(preprocessed_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Prepare a list to collect per subject status dictionaries and start the cohort timer
    results: list[dict] = []
    n_total = len(subjects)
    cohort_start = time.time()

    # Iterate through subjects. List from 1 for easy to read progress logs
    for i, sub in enumerate(subjects, start=1):
        expected_output = output_root/sub/f"{sub}_task-rest_graphs.pt"
        # Skip subjects whose output already exists
        if skip_if_exists and expected_output.exists():
            print(f"[{i}/{n_total}] {sub}: skipped (output already exists)")
            results.append({"subject": sub, "status": "skipped", "output_path": str(expected_output), "skipped": True, "error_message": None})
            continue

        # Declare the start of processing and record  start timefor each subject
        print(f"[{i}/{n_total}] {sub}: processing")
        sub_start = time.time()

        # Look up the subject's APOE label then run the per subject pipeline
        apoe = apoe_labels.get(sub) if apoe_labels else None
        status = build_subject_graphs(subject_id=sub, config=config, preprocessed_root=preprocessed_root, output_root=output_root, apoe_label=apoe)
        status["skipped"] = False
        results.append(status)

        # Report subject level outcome and time taken
        elapsed = time.time()-sub_start
        print(f"[{i}/{n_total}] {sub}: {status['status']} ({elapsed:.0f} s)")

    # Cohort wide summary once all the subjects are done
    cohort_elapsed = time.time()-cohort_start
    print(f"\nCohort complete: {n_total} subjects in {cohort_elapsed/60:.1f} min")

    return pd.DataFrame(results)