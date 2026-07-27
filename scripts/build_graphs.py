"""
Driver script for cohort level graph construction.

Loads project config and APOE labels, selects a cohort based on ``--stage skeleton|full``, invokes the batch runner and saves a summary CSV.

Run:
    - python scripts/build_graphs.py --stage skeleton
    - python scripts/build_graphs.py --stage full

Excludes sub-55 and sub-69 because they're both missing (recorded in PEARL_Neuro paper)
"""

import argparse
from pathlib import Path
import pandas as pd
from agnn.config import load_config
from agnn.connectivity.batch import build_cohort_graphs

# Skeleton subjects to try for first run through
SKELETON_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 6)]
# Exclude the subjects with missing data
EXCLUDED_SUBJECTS = {"sub-55", "sub-69"}
# Full range of subjects
FULL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 80) if f"sub-{i:02d}" not in EXCLUDED_SUBJECTS]


def load_apoe_labels(participants_tsv: Path) -> dict[str, int]:
    """
    Load APOE e4 carrier labels from participants.tsv.

    Uses the APOE_haplotype column. A subject is a carrier (label=1) if their haplotype contains e4 (e.g. e3/e4, e4/e4, e2/e4).
    Non-carriers (label=0) have no e4 allele (e.g. e3/e3 ot e2/e3)

    Subjects with missing or unparseable APOE data are left out of the output dictionary wont be assigning a default label.
    """
    # Load participants file and only keep rows with a BIDS subject ID
    df = pd.read_csv(participants_tsv, sep="\t")
    df = df[df["participant_id"].str.startswith("sub-")]

    # Walk through each subject and assign a binary carrier label
    labels: dict[str, int] = {}
    for _, row in df.iterrows():
        sub = row["participant_id"]
        haplotype = str(row.get("APOE_haplotype", "")).strip().lower()

        # Skip subjects with missing haplotype 
        if not haplotype or haplotype == "nan":
            continue

        # e4 carrier if 4 appears anywhere in the haplotype string
        labels[sub] = int("4" in haplotype)

    return labels


def main() -> None:
    # Parse command line arguments for cohort selection and skip behaviour
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["skeleton", "full"], required=True, help="Which cohort to build graphs for.")
    parser.add_argument("--no-skip", action="store_true", help="Rebuild graphs even if output file already exists.")
    args = parser.parse_args()

    # Load the project config and derive input/output paths.
    cfg = load_config()
    data_root = Path(cfg["paths"]["data_root"])
    preprocessed_root = data_root/"derivatives"/"eeg_preprocessed"
    output_root = data_root/"derivatives"/"graphs"

    # Pick appropriate subject list for this specific run
    subjects = SKELETON_SUBJECTS if args.stage == "skeleton" else FULL_SUBJECTS

    # Load APOE labels. Missing participants.tsv results in unlabelled graphs.
    participants_tsv = data_root/"participants.tsv"
    apoe_labels = None
    if participants_tsv.exists():
        apoe_labels = load_apoe_labels(participants_tsv)
        print(f"Loaded APOE labels for {len(apoe_labels)} subjects.")
    else:
        print(f"problem: {participants_tsv} not found. Graphs will be unlabelled.")

    print(f"Building graphs: stage={args.stage} ({len(subjects)} subjects)")
    print(f"Preprocessed root: {preprocessed_root}")
    print(f"Output root: {output_root}")
    print()

    # Run the batch pipeline. Returns a summary dataframe with one row per subject
    summary = build_cohort_graphs(subjects=subjects, config=cfg, preprocessed_root=preprocessed_root, output_root=output_root, apoe_labels=apoe_labels, skip_if_exists=not args.no_skip)

    # Save summary CSV
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir/f"graph_construction_summary_{args.stage}.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("Status counts:")
    print(summary["status"].value_counts().to_string())
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()