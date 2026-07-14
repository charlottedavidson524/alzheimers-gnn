"""
Cross-validation and metrics for comparing the models.

Runs stratified k-fold cross-validation on any model that is compatible with sklearn and will return comparable
metrics. Should be used by tabular baselines, the GNN, and fusion models so every result comes from the same 
folds and metric computations. 

Metrics:
    - F1
    - Balanced accuracy 
    - ROC-AUC

Confidence intervals: 
    - 95th percentile bootstrap over fold results
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from agnn.config import load_config

# ──────────────────────────────────────────────────────────────────────
# Core cross-validation function
# ──────────────────────────────────────────────────────────────────────

def cross_validate(model_factory, X, y, n_splits=None, seed=None):
    """
    Run stratified k-fold CV and return per-fold and summary metrics.

    Parameters
    ----------
        - model_factory: callable. A no-argument fucntion that returns a fresh untrained model, Passing a factory
        rather than an instance makes sure each fold gets a new model with no state leaking between folds. 
        - X: array of shape (n_samples, n_features)
        - y: array of shape (n_samples,). Binary target
        - n_splits : int. Optional (default: from config)
        - seed : int. Optional (default: from config)

    Returns
    -------
        - dict with keys 'per_fold' and 'summary'. Both of these are DataFrames.
    """
    # Take defaults from the config. This makes sure every model uses the same folds unless they are overridden
    # on purpose
    cfg = load_config()

    if n_splits is None:
        n_splits = cfg["cv"]["n_splits"]

    if seed is None:
        seed = cfg["seed"]

    # Set X and y
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
 
    # Set splitter as stratified k fold
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Loop over the folds, collecting a record per fold.
    fold_records = []

    # Find run_one_fold() below
    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        record = run_one_fold(model_factory, X, y, train_idx, test_idx, fold_number)
        fold_records.append(record)

    # Set to dataframe
    per_fold = pd.DataFrame(fold_records)

    # Find summarise_folds() below.
    summary = summarise_folds(per_fold, seed)
 
    return {"per_fold": per_fold, "summary": summary}

# ──────────────────────────────────────────────────────────────────────
# Train and evaluate a single fold
# ──────────────────────────────────────────────────────────────────────

def run_one_fold(model_factory, X, y, train_idx, test_idx, fold_number):
    """
    Train a fresh model using the train indices. Evaluate using the test indices.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
 
    model = model_factory()
    model.fit(X_train, y_train)

    # Get the predicted probabilities for the positive class. sklearn models expose predict_proba or decision_function
    # depending on the algorithm so need to write code to handle both.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
 
    y_pred = (y_score >= 0.5).astype(int)
 
    return {
        "fold": fold_number,
        "n_test": len(y_test),
        "n_test_positive": int(y_test.sum()),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
    }

# ──────────────────────────────────────────────────────────────────────
# Aggregate fold results into a summary.
# Includes confidence intervals
# ──────────────────────────────────────────────────────────────────────

def summarise_folds(per_fold, seed, bootstrap_n=1000):
    """
    Compute mean, std, and 95% bootstrap CI for each metric.
 
    Bootstrap resamples the k folds with replacement. Wide confidence intervalks at k=5 honestly reflect the 
    uncertainty in a small-sample study.
    """
    rng = np.random.default_rng(seed)
    metric_names = ["f1", "balanced_accuracy", "roc_auc"]
 
    summary_rows = []
    for metric in metric_names:
        values = per_fold[metric].to_numpy()
 
        # Percentile bootstrap over fold-level values.
        boot_means = np.empty(bootstrap_n)
        for i in range(bootstrap_n):
            resample = rng.choice(values, size=len(values), replace=True)
            boot_means[i] = resample.mean()
 
        summary_rows.append({
            "metric": metric,
            "mean": values.mean(),
            "std": values.std(ddof=1) if len(values) > 1 else 0.0,
            "ci_low": float(np.percentile(boot_means, 2.5)),
            "ci_high": float(np.percentile(boot_means, 97.5)),
        })
 
    return pd.DataFrame(summary_rows)

# ──────────────────────────────────────────────────────────────────────
# Create summary for displaying in the terminal
# ──────────────────────────────────────────────────────────────────────

def format_summary(summary_df):
    """
    Return a readable string with a line per metric.
    """
    lines = []

    for _, row in summary_df.iterrows():
        line = (
            f"{row['metric']:20s}"
            f"{row['mean']:.3f} +/- {row['std']:.3f}"
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )
        lines.append(line)

    return "\n".join(lines)