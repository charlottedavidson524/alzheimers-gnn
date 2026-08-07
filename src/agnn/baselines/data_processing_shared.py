"""
This is data loading for the tabular fusion model. 

This code is mostly the same as the data flow in data_processing.py but with two important differences for fusion analysis

- It filters down to a specific subject list (the intersection cohort), not the full second phase cohort like last time. This 
  makes sure that the tabular baseline predicts on exactly the same subjects the GNN will predict on.

- There isn't imputation or standardisation at load time. These are provided as separate fit/transform functions . This way they
  can be called inside CV folds. This fixes the issue where imputation and standardisation were previously computed on the 
  full cohort which leaked information from validation into train

The feature engineering is the same as data_processing.py so both pipelines produce identical feature matrices.

This file returns X, y, feature_names and subject_ids as arrays ready for cross-validation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from agnn.baselines.data_processing import NEVER_IMPUTE, add_derived_features, build_feature_matrix, derive_apoe_e4_carrier, load_participants


def filter_to_subject_list(df: pd.DataFrame, subject_list: list[str]) -> pd.DataFrame:
    """
    This function restricts the dataframe to a specific set of subject IDs.

    It replaces filter_to_modelling_cohort() from data_processing.py. Instead of filtering by the `second_phase == 1` flag, it
    filters to a list (the intersection cohort from shared_cv_splits.json).
    """
    filtered = df[df["participant_id"].isin(subject_list)].copy()

    # Preserve the order of subject_list rather than the order in the original DataFrame. This makes indexing predictable in the 
    # CV loop
    order_map = {sub: i for i, sub in enumerate(subject_list)}
    filtered["_sort_order"] = filtered["participant_id"].map(order_map)
    filtered = filtered.sort_values("_sort_order").drop(columns="_sort_order")

    return filtered.reset_index(drop=True)


def load_baseline_data_for_fusion(variant: str, subject_list: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """
    This function loads data for a baseline variant, filtered to the intersection cohort.

    Parameters
    ----------
    - variant: str
          One of 'compact-light', 'compact-full', 'all-features-light', 'all-features-full'. It determines the feature set.
    - subject_list: list of str
          This is the list of subjects to include. This should be the intersection cohort from shared_cv_splits.json.

    Returns
    -------
    - X: DataFrame
          Feature matrix. It's not imputed or standardised. Shape (n_subjects, n_features)
    - y: ndarray
          Binary APOE e4 carrier labels. Shape (n_subjects,)
    - feature_names: list of str
          Column names of X, in order
    - subject_ids: list of str
          Subject IDs in the same order as rows in X and y
    """
    df = load_participants()
    df = filter_to_subject_list(df, subject_list)
    df = add_derived_features(df)

    y = derive_apoe_e4_carrier(df).to_numpy()
    X = build_feature_matrix(df, variant)
    subject_ids = df["participant_id"].tolist()

    # This is a defensive check. The intersection cohort should already be excluding subjects that have missing blood panels.
    # If any blood panel columns still have NaNs there is an inconsistentcy between the splits file and the data.
    blood_cols_in_X = [c for c in X.columns if c in NEVER_IMPUTE]
    if blood_cols_in_X:
        na_by_subject = X[blood_cols_in_X].isna().any(axis=1)
        if na_by_subject.any():
            bad = [subject_ids[i] for i in np.where(na_by_subject)[0]]
            raise ValueError(f"Subjects with missing blood panel data in intersection cohort: {bad}. Update the shared splits to exclude these subjects")

    return X, y, list(X.columns), subject_ids


def fit_imputation(X_train: pd.DataFrame) -> dict[str, float]:
    """
    This function computes per column fill values from training data only.

    Fill values are computed for all non-blood columns, not just ones with NaN in training. This makes sure validation subjects 
    with NaN in any column can still be imputed. This is even if training happened not to have NaN in that column.
    """
    # Records per-column fill values
    fill_values: dict[str, float] = {}

    for col in X_train.columns:
        # Blood-panel missingness is structural (established in other modelling code). Skip this, don't fabricate.
        if col in NEVER_IMPUTE:
            continue

        # Categorical (few unique values) -> mode and continuous -> median. This is computed from training data even for 
        # columns without training NaN, so validation NaN in those columns can still be filled
        if X_train[col].nunique() <= 10:
            fill_values[col] = X_train[col].mode().iloc[0]
        else:
            fill_values[col] = X_train[col].median()

    return fill_values


def apply_imputation(X: pd.DataFrame, fill_values: dict[str, float]) -> pd.DataFrame:
    """
    This fucntion applies pre-computed fill values to fill missing entries in X.

    It's called separately on training and validation data with the same fill_values (from fit_imputation on training). 
    This makes sure that no information leaks from validation into the imputation choices
    """
    X = X.copy()
    for col, value in fill_values.items():
        if col in X.columns:
            X[col] = X[col].fillna(value)
    return X


def fit_standardisation(X_train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    This function computes per column means and standard deviations from training data only

    Returns
    -------
    - means: Series
          Per-column mean from training data
    - stds: Series
          Per-column standard deviation from training data. Any zero-std columns are replaced with 1.0 to avoid divide-by-zero 
          in apply_standardisation function seen below.
    """
    means = X_train.mean()
    stds = X_train.std()

    # Guard against zero-std columns (constant features). Dividing by zero would produce NaN. Replacing std=0 with std=1 leaves the 
    # column at (value - mean) = 0 it's standardised, which basically neutralises it
    stds = stds.where(stds>0, 1.0)

    return means, stds


def apply_standardisation(X: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    """
    This function applies pre-computed means and stds to standardise X.

    It's called separately on training and validation data with the same means/stds (from fit_standardisation on training)
    """
    return (X-means)/stds