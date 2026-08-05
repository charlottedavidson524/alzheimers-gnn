"""
This is a batch runner for cohort level subject-level aggregated graph construction. It applies build_aggregated_graphs to a 
list of subjects, catches per subject errors, and returns a summary DataFrame. This is different from batch.py which builds 
graphs per epoch.
"""

from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
from agnn.connectivity.build_graphs_subject import build_aggregated_graphs


def build_cohort_aggregated_graphs(subjects: list[str], config: dict, preprocessed_root: Path | str, output_root: Path | str,
    apoe_labels: dict[str, int] | None = None, skip_if_exists: bool = True) -> pd.DataFrame:
    """
    This function builds subject-level graphs for a list of subjects.

    Parameters
    ----------
    - subjects: list of str
          BIDS subject identifiers
    - config: dict
          Loaded config with "connectivity" section
    - preprocessed_root: Path or str
          Root of the preprocessed epochs
    - output_root: Path or str
          Where the graph .pt files are written
    - apoe_labels: dict or None
          {subject_id: 0/1}. If None then the graphs are unlabelled
    - skip_if_exists: bool
          If True, skip subjects whose graph .pt file already exists

    Returns
    -------
    - summary: pd.DataFrame
          One row per subject with counts, timings, and status
    """

    # Normalise paths and make sure the output root exists
    preprocessed_root = Path(preprocessed_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Collect per-subjects status dictionaries andf track the total runtime
    results: list[dict] = []
    n_total = len(subjects)
    cohort_start = time.time()

    # Main loop over the cohort
    for i, sub in enumerate(subjects, start=1):
        expected_output = output_root/sub/f"{sub}_task-rest_graphs.pt"
        if skip_if_exists and expected_output.exists():
            print(f"[{i}/{n_total}] {sub}: skipped (output exists)")
            results.append({"subject": sub, "status": "skipped", "output_path": str(expected_output), "skipped": True, "error_message": None})
            continue

        # Start timing this subject
        print(f"[{i}/{n_total}] {sub}: processing...")
        sub_start = time.time()

        # Look up the APOE label if possible
        apoe = apoe_labels.get(sub) if apoe_labels else None

        # Delegate heavy lifting to build_aggregate_graphs()
        status = build_aggregated_graphs(subject_id=sub, config=config, preprocessed_root=preprocessed_root, output_root=output_root, apoe_label=apoe)
        status["skipped"] = False
        results.append(status)

        # Report run time per subject as well as the subject's final status
        elapsed = time.time()-sub_start
        print(f"[{i}/{n_total}] {sub}: {status['status']} ({elapsed:.0f} s)")

    # Total time to build the graphs for the cohort once the loop is done
    cohort_elapsed = time.time()-cohort_start
    print(f"\nCohort complete: {n_total} subjects in {cohort_elapsed / 60:.1f} min")

    return pd.DataFrame(results)