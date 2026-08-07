"""
This is cross-validation with shared splits and preprocessing that is fold aware.

It is similar to cross_validate.py but has three important differences for fusion:

- It loads pre-computed CV splits from a JSON file (data/shared_cv_splits.json) rather than computing splits internally. This 
  makes sure that the tabular baseline and the GNN evaluate on the exact same subjects in the exact same folds.

- It applies imputation and standardisation inside each fold, fit on training data only. This addresses the issue flagged in 
  where preprocessing was computed on the full cohort, which leaks validation information into training

- It returns per-fold predictions (y_true, y_prob) along with subject IDs. These predictions are the input to the fusion 
  analysis that will be done later.

Reuses summarise_folds from cross_validate.py for the bootstrap CI logic.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from agnn.baselines.data_processing_shared import apply_imputation, apply_standardisation, fit_imputation, fit_standardisation
from agnn.evaluation.cross_validate import summarise_folds


def load_shared_splits(splits_path: Path | str) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    """
    This function loads shared CV splits from the JSONfile.

    Parameters
    ----------
    - splits_path: Path or str
          Path to shared_cv_splits.json (this was produced by scripts/generate_shared_splits.py)

    Returns
    -------
    - subjects: list of str
          The intersection cohort's subject IDs (in order)
    - labels: list of int
          APOE e4 labels aligned with the relevant subjects
    - folds: list of dict
          One dictionary per fold with train_subjects, val_subjects, train_labels and val_labels
    """
    # Read the JSON files and unpack the three fields expected by the caller
    with open(splits_path) as f:
        splits_data = json.load(f)

    return splits_data["subjects"], splits_data["labels"], splits_data["folds"]


def _run_one_fold_shared(model_factory, X: pd.DataFrame, y: np.ndarray, subject_ids: list[str], train_subjects: list[str], 
                         val_subjects: list[str], fold_number: int) -> dict[str, Any]:
    """
    This function trains and evaluates a single fold with  preprocessing that is fold aware.

    It fits imputation and standardisation on training data only, then applies them to both train and validation. This makes sure 
    there are no information leaks from validation to train.

    This function returns a dictionary with per-fold metrics plus predictions (y_true, y_prob, subject_ids) which will be used
    for fusion analysis.
    """
    # Map the subject IDs to row positions in X and y
    subject_to_idx = {sub: i for i, sub in enumerate(subject_ids)}

    # Check that all subjects in the fold are present in the data and raise an error if not
    missing = [s for s in train_subjects + val_subjects if s not in subject_to_idx]
    if missing:
        raise ValueError(f"Fold {fold_number} references unknown subjects: {missing}")

    # Convert the subject IDs to row indices for slicing X and y
    train_idx = [subject_to_idx[s] for s in train_subjects]
    val_idx = [subject_to_idx[s] for s in val_subjects]

    # Split X and y into train and val slices for this fold
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Fit imputation on training data only, then apply to both train and validation
    fill_values = fit_imputation(X_train)
    X_train = apply_imputation(X_train, fill_values)
    X_val = apply_imputation(X_val, fill_values)

    # Fit standardisation on the imputed training data only, then apply to both train and validation same as above
    means, stds = fit_standardisation(X_train)
    X_train = apply_standardisation(X_train, means, stds)
    X_val = apply_standardisation(X_val, means, stds)

    # Create a fresh untrained model for this fold
    model = model_factory()
    model.fit(X_train.to_numpy(), y_train)

    # sklearn models expose predict_proba or decision_function depending on the algorithm, so make sure this can handle both. 
    # Same pattern as in the existing cross_validate.py script.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val.to_numpy())[:, 1]
    else:
        y_score = model.decision_function(X_val.to_numpy())

    # Threshold at 0.5 to get binary predictions for F1 and balanced accuracy
    y_pred = (y_score >= 0.5).astype(int)

    # Bundle metrics and predictions into one dictionary for the caller to unpack
    return {
        "fold": fold_number,
        "n_test": len(y_val),
        "n_test_positive": int(y_val.sum()),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_score),
        # Predictions carried through for downstream fusion analysis
        "y_true": y_val.tolist(),
        "y_prob": y_score.tolist(),
        "subject_ids": val_subjects,
    }


def cross_validate_shared(model_factory, X: pd.DataFrame, y: np.ndarray, subject_ids: list[str], folds: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    """
    This function runs cross-validation using pre computed splits and fold aware preprocessing.

    Parameters
    ----------
    - model_factory: callable
          This is a no argument function returning a fresh untrained sklearn model
    - X: DataFrame
          A feature matrix. Rows aligned with subject_ids. Hasn't been imputed or standardised yet.
    - y: ndarray
          Binary labels. Are aligned with subject_ids
    - subject_ids: list of str
          Subject IDs for each row of X (and each entry of y)
    - folds: list of dict
          These are pre-computed fold definitions from shared_cv_splits.json. Each dictionary has train_subjects and val_subjects.
    - seed: int
          This is for bootstrap confidence interval computation in summarise_folds

    Returns
    -------
    - dict with keys:
        - per_fold: DataFrame
            One row per fold with fold_number, n_test, n_test_positive and metrics
        - summary: DataFrame
            Mean +/- std and 95% bootstrap confidence intervals per metric across folds
        - predictions: list of dict
            Per-fold predictions with y_true, y_prob, subject_ids. Used in fusion
    """
    # Accumulate per-fold metric records and prediction dictionaries as the run goes
    fold_records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    # Iterate over the pre-computed folds, training and evaluating each one
    for fold_dict in folds:
        record = _run_one_fold_shared(model_factory=model_factory, X=X, y=y, subject_ids=subject_ids, train_subjects=fold_dict["train_subjects"], 
                                      val_subjects=fold_dict["val_subjects"], fold_number=fold_dict["fold_idx"])

        # Separate metrics from predictions so per_fold DataFrame stays nice and clean
        predictions.append({"fold": record["fold"], "y_true": np.array(record.pop("y_true")), "y_prob": np.array(record.pop("y_prob")), 
                            "subject_ids": record.pop("subject_ids")})
        
        fold_records.append(record)

    # Aggregate into the clean per-fold dataframe and the bootstrap confidence interval summary
    per_fold = pd.DataFrame(fold_records)
    summary = summarise_folds(per_fold, seed=seed)

    return {"per_fold": per_fold, "summary": summary, "predictions": predictions}