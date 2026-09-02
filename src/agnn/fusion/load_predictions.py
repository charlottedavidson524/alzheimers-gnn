"""
This file loads and aligns per-fold predictions from the GNN and tabular models.

Both models produced predictions on the same intersection cohort (n=74) using the same shared CV splits, but their predictions 
are in different directory structures at the moment

- Tabular (from scripts/rerun_tabular_for_fusion.py):
    results/fusion/tabular_logistic_compact_full/fold_N_predictions.npz

- GNN (downloaded from Colab):
    results/fusion/gnn_subject_shared_splits/fold_N/predictions.npz

This module loads both, checks that each fold has the same subjects, aligns them by subject ID (sorting), and returns a 
per-fold structure  which bis ready for fusion.

Alignment is important because even though the CV splits are shared, the DataLoader in Colab might emit subjects in a 
different order than sklearn does. Sorting by subject ID guarantees position i in gnn_prob and position i in tabular_prob
arethe same subject.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np


def load_gnn_predictions(gnn_dir: Path | str, n_folds: int = 5) -> list[dict[str, Any]]:
    """
    Load GNN per-fold predictions from the nested Colab output structure.

    Parameters
    ----------
    - gnn_dir: Path or str
          Directory containing fold_0/, fold_1/, ... subfolders
    - n_folds: int
          Number of folds to load. Default 5

    Returns
    -------
    - predictions: list of dict
          One dict per fold with keys: fold, y_true, y_prob, subject_ids
    """
    # Normalise to a Path and prepare the output list
    gnn_dir = Path(gnn_dir)
    predictions: list[dict[str, Any]] = []

    # Load each fold's predictions file in turn
    for fold_idx in range(n_folds):
        path = gnn_dir/f"fold_{fold_idx}"/"predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"GNN predictions not found: {path}")

        # Read the saved arrays for this fold
        data = np.load(path, allow_pickle=True)

        # Store this fold's predictions in the same dictionary shape as the tabular loader
        predictions.append({
            "fold": fold_idx,
            "y_true": np.asarray(data["y_true"]),
            "y_prob": np.asarray(data["y_prob"]),
            "subject_ids": [str(s) for s in data["subject_ids"]],
        })

    return predictions


def load_tabular_predictions(tabular_dir: Path | str, n_folds: int = 5) -> list[dict[str, Any]]:
    """
    Load tabular per-fold predictions from the flat output structure.

    Parameters
    ----------
    - tabular_dir: Path or str
          Directory containing fold_0_predictions.npz, fold_1_predictions.npz, etc
    - n_folds: int
          Number of folds to load. Default 5

    Returns
    -------
    - predictions: list of dict
          One dict per fold with keys: fold, y_true, y_prob, subject_ids
    """
    # Normalise to a Path and prepare the output list
    tabular_dir = Path(tabular_dir)
    predictions: list[dict[str, Any]] = []

    # Load each fold's predictions file in turn
    for fold_idx in range(n_folds):
        path = tabular_dir/f"fold_{fold_idx}_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"Tabular predictions not found: {path}")

        # Read the saved arrays for this fold
        data = np.load(path, allow_pickle=True)

        # Store this fold's predictions in the same dictionary shape as the GNN loader
        predictions.append({
            "fold": fold_idx,
            "y_true": np.asarray(data["y_true"]),
            "y_prob": np.asarray(data["y_prob"]),
            "subject_ids": [str(s) for s in data["subject_ids"]],
        })

    return predictions


def _align_one_fold(gnn_fold: dict[str, Any], tabular_fold: dict[str, Any]) -> dict[str, Any]:
    """
    Align one fold's GNN and tabular predictions by the subject ID.

    This makes sure that both models predicted on the same subjects and sorts both prediction arrays by subject ID so that 
    position i refers to the same subject in both arrays
    """
    # Jsut for use in error messages below
    fold_idx = gnn_fold["fold"]

    # Verify identical subject sets between the two models.
    gnn_subjects = set(gnn_fold["subject_ids"])
    tab_subjects = set(tabular_fold["subject_ids"])

    # Fail loudly if two models disagree on which subjects are in the fold
    if gnn_subjects != tab_subjects:
        only_gnn = gnn_subjects - tab_subjects
        only_tab = tab_subjects - gnn_subjects
        raise ValueError(f"Fold {fold_idx} subject mismatch. Only in GNN: {only_gnn}. Only in tabular: {only_tab}.")

    # Build lookup dicts so it is possible to reorder both models' predictions by a canonical subject order
    gnn_lookup = dict(zip(gnn_fold["subject_ids"], zip(gnn_fold["y_true"], gnn_fold["y_prob"])))
    tab_lookup = dict(zip(tabular_fold["subject_ids"], zip(tabular_fold["y_true"], tabular_fold["y_prob"])))

    # Canonical order os the sorted subject IDs. Consistent across folds and models
    subject_ids_sorted = sorted(gnn_subjects)

    # reorder each model's true labels and probabilities into the canonical subject order
    y_true = np.array([gnn_lookup[s][0] for s in subject_ids_sorted])
    gnn_prob = np.array([gnn_lookup[s][1] for s in subject_ids_sorted])
    tabular_prob = np.array([tab_lookup[s][1] for s in subject_ids_sorted])

    # Quick sanity check. Labels should be identical between the two models (same subjects have the same APOE labels)
    tab_y_true = np.array([tab_lookup[s][0] for s in subject_ids_sorted])
    if not np.array_equal(y_true, tab_y_true):
        raise ValueError(
            f"Fold {fold_idx} label mismatch between models — "
            f"data alignment problem upstream."
        )

    # Return aligned, index-matched arrays for the fold
    return {
        "fold": fold_idx,
        "subject_ids": subject_ids_sorted,
        "y_true": y_true,
        "gnn_prob": gnn_prob,
        "tabular_prob": tabular_prob,
    }


def align_predictions_per_fold(gnn_predictions: list[dict[str, Any]], tabular_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Align GNN and tabular predictions fold-by-fold.

    Needs the two prediction lists to be in the same fold order (fold 0 first, fold 1 second, etc.). Each fold's predictions 
    are aligned by subject ID so that position i in the returned arrays refers to the same subject in both models.

    Returns
    -------
    - aligned: list of dict
          One dict per fold with keys: fold, subject_ids (sorted), y_true, gnn_prob, tabular_prob. All arrays are the same 
          length and index-aligned.
    """
    # Both loaders should return one entry per fold. Mismatched counts mean something went wrong further upstream
    if len(gnn_predictions) != len(tabular_predictions):
        raise ValueError(f"Fold count mismatch: {len(gnn_predictions)} GNN vs {len(tabular_predictions)} tabular.")

    # Align each corresponding pair of folds
    return [
        _align_one_fold(gnn_fold, tab_fold)
        for gnn_fold, tab_fold in zip(gnn_predictions, tabular_predictions)
    ]


def load_aligned_predictions(gnn_dir: Path | str, tabular_dir: Path | str, n_folds: int = 5) -> list[dict[str, Any]]:
    """
    This is a wrapper for convenience. It loads both models and aligns in one call.

    Parameters
    ----------
    - gnn_dir: Path or str
          Directory containing GNN per-fold predictions (nested structure)
    - tabular_dir: Path or str
          Directory containing tabular per-fold predictions (flat structure).
    - n_folds: int
          Number of folds to load. Default 5

    Returns
    -------
    - aligned: list of dict
          See align_predictions_per_fold.
    """
    gnn_predictions = load_gnn_predictions(gnn_dir, n_folds)
    tabular_predictions = load_tabular_predictions(tabular_dir, n_folds)
    return align_predictions_per_fold(gnn_predictions, tabular_predictions)