"""
Driver script for cohort level EEG preprocessing.

Loads project config, chooses a cohort based on a command-line flag (--stage skeleton for sub-01..sub-05, 
--stage full for sub-01..sub-79). Designed this way because of how the data download was done. Calls the batch 
runner and saves a summary CSV.

Run from the project root:
    python scripts/preprocess_eeg.py --stage skeleton
    python scripts/preprocess_eeg.py --stage full

Outputs:
    - Preprocessed epochs under C:/data/pearl-neuro/derivatives/eeg_preprocessed/sub-XX/ (one .fif file per subject, plus a preprocessing_log.txt)
    - Cohort-level summary CSV under `results/eeg_preprocessing_summary_{stage}.csv`
"""

import argparse
from pathlib import Path
from agnn.config import load_config
from agnn.preprocessing.batch_eeg import preprocess_cohort


# define the two cohorts
SKELETON_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 6)]  # sub-01 to sub-05
FULL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 80)] # sub-01 to sub-79


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["skeleton", "full"], required=True, help="Which cohort to preprocess.")
    parser.add_argument("--no-skip", action="store_true",
        help=(
            "Reprocess subjects even if their output file already exists. "
            "Default is to skip existing outputs."
        ),
    )
    args = parser.parse_args()

    # Load config and set up paths
    cfg = load_config()
    data_root = Path(cfg["paths"]["data_root"])
    output_root = data_root/"derivatives"/"eeg_preprocessed"

    # Select cohort
    subjects = SKELETON_SUBJECTS if args.stage == "skeleton" else FULL_SUBJECTS

    print(f"Preprocessing stage: {args.stage} ({len(subjects)} subjects)")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print()

    # Run
    summary = preprocess_cohort(subjects=subjects, config=cfg, data_root=data_root, output_root=output_root, skip_if_exists=not args.no_skip)

    # Save summary CSV to project's results directory
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir/f"eeg_preprocessing_summary_{args.stage}.csv"
    summary.to_csv(summary_path, index=False)

    # Report outcomes
    print()
    print("Status counts:")
    print(summary["status"].value_counts().to_string())
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()