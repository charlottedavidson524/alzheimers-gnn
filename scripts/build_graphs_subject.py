"""
This the the driver script for cohort level subject-level aggregated graph construction.

It lodas the project config and APOE labels, chooses a cohort based on `--stage skeleton|full`, invokes the batch runner, and 
saves a summary CSV.

Run from the project root:
    - python scripts/build_graphs_subject.py --stage skeleton
    - python scripts/build_graphs_subject.py --stage full

This is in contrast with scripts/build_graphs.py which produces per epoch graphs. It excludes sub-55 (missing rest data per 
paper) and sub-69 (no consent, as per paper)
"""

import argparse
from pathlib import Path
import pandas as pd
from agnn.config import load_config
from agnn.connectivity.batch_subject import build_cohort_aggregated_graphs

# The skeleton subjects
SKELETON_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 6)]

# Effective cohort needs to exclude sub-55 and sub-69
EXCLUDED_SUBJECTS = {"sub-55", "sub-69"}
FULL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 80) if f"sub-{i:02d}" not in EXCLUDED_SUBJECTS]


def load_apoe_labels(participants_tsv: Path) -> dict[str, int]:
    """
    This fucntion loads APOE e4 carrier labels from participants.tsv.

    it uses the APOE_haplotype column. A subject is a carrier (label=1) if their haplotype contains e4 (e.g. e3/e4, e4/e4, 
    e2/e4). Non-carriers (label=0) have no e4 allele (e.g. e3/e3, e2/e3).

    Subjects with missing  APOE data or APOE data that can't be parsed are left out of the output dictionary instead of being 
    assigned a default label.
    """
    # Load participants.tsv and keep only rows with valid BIDS subject IDs
    df = pd.read_csv(participants_tsv, sep="\t")
    df = df[df["participant_id"].str.startswith("sub-")]

    # Build the output dictionary, one subject at a time
    labels: dict[str, int] = {}
    for _, row in df.iterrows():
        sub = row["participant_id"]
        haplotype = str(row.get("APOE_haplotype", "")).strip().lower()

        # Skip subjects with missing or unparseable haplotype
        if not haplotype or haplotype == "nan":
            continue

        # e4 carrier if "4" appears anywhere in the haplotype string
        labels[sub] = int("4" in haplotype)

    return labels


def main() -> None:

    # Parse command line arguments for cohort stage and skipping behaviour
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["skeleton", "full"], required=True, help="Which cohort to build graphs for.")
    parser.add_argument("--no-skip", action="store_true", help="Rebuild graphs even if output file already exists.")
    args = parser.parse_args()

    # Load the config and resolve the paths used throughout the run
    cfg = load_config()
    data_root = Path(cfg["paths"]["data_root"])
    preprocessed_root = data_root/"derivatives"/"eeg_preprocessed"

    # Separate output directory. This keeps subject level graphs distinct from the per epoch graphs in derivatives/graphs
    output_root = data_root/"derivatives"/"graphs_subject"

    # Pick appropriate subject list based on the requested stage
    subjects = SKELETON_SUBJECTS if args.stage == "skeleton" else FULL_SUBJECTS

    # Load APOE labels. Missing participants.tsv results in unlabelled graphs
    participants_tsv = data_root/"participants.tsv"
    apoe_labels = None

    # Load labels if the file exists. If it doesn't then give a warning and continue unlabelled.
    if participants_tsv.exists():
        apoe_labels = load_apoe_labels(participants_tsv)
        print(f"Loaded APOE labels for {len(apoe_labels)} subjects.")
    else:
        print(f"Warning: {participants_tsv} not found. Graphs will be unlabelled.")

    print(f"Building subject-level graphs: stage={args.stage} ({len(subjects)} subjects)")
    print(f"Preprocessed root: {preprocessed_root}")
    print(f"Output root: {output_root}")
    print()

    # Delegate the cohort loop to the batch runner
    summary = build_cohort_aggregated_graphs(subjects=subjects, config=cfg, preprocessed_root=preprocessed_root, output_root=output_root, 
                                             apoe_labels=apoe_labels, skip_if_exists=not args.no_skip)

    # Save summary CSV. Filenames are distinct from per epoch summaries
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir/f"graph_construction_summary_subject_{args.stage}.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("Status counts:")
    print(summary["status"].value_counts().to_string())
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()