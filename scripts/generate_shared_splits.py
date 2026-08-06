"""
The prupose of this file is to create shared cross-validation splits for GNN and tabular fusion analysis.

This code finds one canonical set of CV splits that both the GNN training (done in Colab) and the tabular baseline training 
(donw in VSCode) will both use. There are the same subjects and the same fold assignments. This is a requirement for late 
fusion to be done properly.

Need to find an intersection cohort. Currently have:
    - GNN cohort (N=77): missing sub-55 and sub-69, as per PEARL-Neuro paper and own analysis
    - logistic_compact_full (N=76): 3 subjects (sub-53, sub-68 and sub-79) are missing because of issues with the blood panel

The composition of the intesection is calculated at runtime by reading the subject lists of bot of the cohorts.

The code uses StratifiedGroupKFold with subject IDs as groups. This is because the same subject can't appear in both train and 
validation sets for any fold. Fold class balance is stratified by APOE e4 carrier status.

Run this code once from the project root:
    - python scripts/generate_shared_splits.py

The output will be data/shared_cv_splits.json . This will be committed to the repo, this way both Colab and VSCode load from 
the same source (as Colab will be needed to retain the GNN due to it's T4 GPU access).
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from agnn.config import load_config


# This is the GNN cohort. Excludes sub-55 and sub-69
GNN_EXCLUDED = {"sub-55", "sub-69"}
GNN_COHORT = [f"sub-{i:02d}" for i in range(1, 80) if f"sub-{i:02d}" not in GNN_EXCLUDED]

# For the tabular cohort, subjects were excluded from logistic_compact_full because of missing blood panel data. Also need to
# include GNN's exclusions.
TABULAR_EXCLUDED = {"sub-55", "sub-69"} 

# Also add the 3 subjects excluded from tabular baseline due to blood panel issues
TABULAR_EXCLUDED |= {"sub-53", "sub-68", "sub-79"} 

# Random state and fold count
RANDOM_STATE = 42
N_FOLDS = 5


def load_apoe_labels(participants_tsv: Path) -> dict[str, int]:
    """
    This function loads APOE e4 carrier labels. It uses the same logic as scripts/build_graphs_subject.py
    """
    # Load participants.tsv and keep only rows that have valid BIDS subject IDs
    df = pd.read_csv(participants_tsv, sep="\t")
    df = df[df["participant_id"].str.startswith("sub-")]

    # Build the output dictionary, one subject at a time
    labels: dict[str, int] = {}
    for _, row in df.iterrows():
        sub = row["participant_id"]
        haplotype = str(row.get("APOE_haplotype", "")).strip().lower()
        if not haplotype or haplotype == "nan":
            continue
        labels[sub] = int("4" in haplotype)

    return labels


def compute_intersection_cohort(gnn_cohort: list[str], tabular_excluded: set[str], apoe_labels: dict[str, int]) -> list[str]:
    """
    This function finds the cohort intersection, or subjects that are in the GNN cohort as well as not excluded from the 
    tabular cohort and are also labelled.
    """
    # Intersection subject list
    intersection = []
    for sub in gnn_cohort:
        if sub in tabular_excluded:
            continue
        # Check if the subject has no APOE label and therefore can't be used for either model
        if sub not in apoe_labels:
            continue
        # Append intersection suvjects to the list
        intersection.append(sub)

    return sorted(intersection)


def generate_splits(subjects: list[str], labels: list[int], n_folds: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    This function generates stratified group k-fold splits with subject IDs as groups.
    """
    # Convert inputs to numpy arrays for the splitter
    subjects_arr = np.array(subjects)
    labels_arr = np.array(labels)

    # StratifiedGroupKFold stratifies by label and groups by subject
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Groups are subject IDs. Each subject is one row in the split
    return list(splitter.split(subjects_arr, labels_arr, groups=subjects_arr))


def verify_splits(subjects: list[str], labels: list[int], splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """
    This function makes sure there is no subject leakage and checks the fold balance.
    """
    # Iterate over folds. Check for subject leakage and report the class balance
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_subs = set(subjects[i] for i in train_idx)
        val_subs = set(subjects[i] for i in val_idx)
        overlap = train_subs & val_subs
        assert not overlap, f"Fold {fold_idx} leakage: {overlap}"

        # Compute this specific fold's positive class rate for logging
        val_labels = [labels[i] for i in val_idx]
        n_pos = sum(val_labels)
        n_val = len(val_labels)
        pos_rate = n_pos/n_val if n_val else 0.0
        print(f"Fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val, {n_pos}/{n_val} positive ({pos_rate:.1%})")


def save_splits_json(output_path: Path, subjects: list[str], labels: list[int], splits: list[tuple[np.ndarray, np.ndarray]], 
                     n_folds: int, random_state: int) -> None:
    """
    This function saves splits as JSON with subject IDs (not just indices) for portability. This is because I also work in
    Colab
    """
    # Convert index-based splits to subject-ID-based splits. This means both Colab and VSCode can use them without needing 
    # the same subject ordering

    # One entry per fold. Holds both train and validation subject IDs and labels
    folds_data = []

    # Fill in each fold's training/validation subject IDs and labe;s
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        folds_data.append({
            "fold_idx": fold_idx,
            "train_subjects": [subjects[i] for i in train_idx],
            "val_subjects": [subjects[i] for i in val_idx],
            "train_labels": [labels[i] for i in train_idx],
            "val_labels": [labels[i] for i in val_idx],
        })

    # Wrap in a toplevel dictionary with cohort metadata for context
    output = {
        "description": "Shared CV splits for GNN and tabular fusion analysis",
        "cohort_size": len(subjects),
        "n_folds": n_folds,
        "random_state": random_state,
        "splitter": "StratifiedGroupKFold",
        "subjects": subjects,
        "labels": labels,
        "folds": folds_data,
    }

    # Persist to disk as an indented JSON. Makes it human readable
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main() -> None:
    # Load config and resolve the path to participants.tsv
    cfg = load_config()
    data_root = Path(cfg["paths"]["data_root"])
    participants_tsv = data_root/"participants.tsv"

    # Fail fast if participants.tsv is mising as it's needed for APOE labels (shouldn't ever be the case but for safety)
    if not participants_tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found: {participants_tsv}")

    # Load APOE labels
    apoe_labels = load_apoe_labels(participants_tsv)
    print(f"Loaded APOE labels for {len(apoe_labels)} subjects.")

    # Compute intersection cohort.
    intersection = compute_intersection_cohort(GNN_COHORT, TABULAR_EXCLUDED, apoe_labels)
    labels = [apoe_labels[sub] for sub in intersection]

    # Report cohort composition
    n_pos = sum(labels)
    n_total = len(intersection)

    print(f"\nIntersection cohort: {n_total} subjects")
    print(f"Carriers: {n_pos} ({n_pos/n_total:.1%})")
    print(f"Non-carriers: {n_total-n_pos} ({(n_total-n_pos)/n_total:.1%})")

    # Generate splits
    splits = generate_splits(intersection, labels, N_FOLDS, RANDOM_STATE)

    print(f"\nGenerated {len(splits)} folds:")

    # Sanity check the splits. Make sure there's no subject leakage and report the class balance
    verify_splits(intersection, labels, splits)

    # Save to project's data directory. Should be small enough to commit
    output_dir = Path(cfg["paths"]["project_root"])/"data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir/"shared_cv_splits.json"

    # Write the splits to the JSON file with all the fold metadat
    save_splits_json(output_path=output_path, subjects=intersection, labels=labels, splits=splits, n_folds=N_FOLDS, random_state=RANDOM_STATE)

    print(f"\nSaved: {output_path}")
    print(f"File size: {output_path.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()