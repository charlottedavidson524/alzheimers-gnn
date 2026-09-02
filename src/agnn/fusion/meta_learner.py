"""
Meta-learner fusion using logistic regression on GNN and tabular probabilities.

Cross-fold meta-learner (nested CV).

For each of the 5 outer folds:
  - Train the meta-learner on out-of-fold predictions from the other 4 folds (~59 subjects with both models' probabilities and the true labels).
  - Apply the meta-learner to this fold's predictions (~15 subjects)

Concatenating all 5 folds' meta-learner predictions gives a complete out-of-sample prediction for the whole cohort. Use to
compute fusion AUC.

Learned coefficients are saved per fold to check if the two models' relative contributions are stable or vary between folds. 
Stable coefficients indicate a robust finding. unstable ones suggest the fusion result is data-dependent.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_meta_learner(train_gnn_probs: np.ndarray, train_tabular_probs: np.ndarray, train_y: np.ndarray, C: float = 1.0) -> LogisticRegression:
    """
    Fit a logistic regression meta-learner on both models' probabilities.

    Parameters
    ----------
    - train_gnn_probs, train_tabular_probs: ndarray
          Per-subject probability predictions from each model. Same length
      train_y: ndarray
          Binary labels aligned with the probability arrays
    - C: float
          Inverse regularisation strength. Default 1.0 (sklearn default)

    Returns
    -------
    - fitted logistic regression model. Its coef_ has shape (1, 2) [gnn_coef, tabular_coef]
    """
    # Stack the two probability streams as columns of the design matrix.
    X = np.column_stack([train_gnn_probs, train_tabular_probs])
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X, train_y)
    return model


def evaluate_meta_learner(model: LogisticRegression, val_gnn_probs: np.ndarray, val_tabular_probs: np.ndarray, val_y: np.ndarray) -> dict[str, Any]:
    """
    Apply a fitted meta-learner to validation predictions.

    Returns
    -------
    - dict with keys:
          y_true: ndarray
          y_prob: ndarray
              Meta-learner's predicted probabilities.
          auc: float
    """
    # Build the validation design matrix in the same column order as training
    X = np.column_stack([val_gnn_probs, val_tabular_probs])

    # Predict probabilities and score against true labels
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(val_y, y_prob)

    return {"y_true": val_y, "y_prob": y_prob, "auc": auc}


def cross_fold_meta_learner(aligned_predictions: list[dict[str, Any]], C: float = 1.0) -> dict[str, Any]:
    """
    Run cross-fold meta-learner training and evaluation

    For each fold, trains the meta-learner on the other folds predictions (out-of-fold training data) and evaluates it on 
    this fold's predictions.

    Parameters
    ----------
    - aligned_predictions: list of dict
          Per-fold aligned predictions from load_aligned_predictions. Each dict has gnn_prob, tabular_prob, y_true, subject_ids
    - C: float
          Regularisation strength for logistic regression. Default 1.0.

    Returns
    -------
    - dict with keys:
          per_fold: list of dict
              One entry per fold with meta-learner predictions and coefficients.
          mean_auc, std_auc: float
              Cross-fold AUC statistics.
          mean_gnn_coef, mean_tabular_coef: float
              Average learned coefficients across folds.
          std_gnn_coef, std_tabular_coef: float
              Standard deviation of coefficients (stability check).
          mean_intercept: float
    """
    # Track per-fold results
    n_folds = len(aligned_predictions)
    per_fold: list[dict[str, Any]] = []

    # For each fold, train on the other folds and evaluate on this fold
    for held_out_idx in range(n_folds):

        # Collect training data from all other folds
        train_gnn_parts = []
        train_tabular_parts = []
        train_y_parts = []

        # For each fold, train on the other folds and evaluate on this fold
        for other_idx in range(n_folds):
            if other_idx == held_out_idx:
                continue

            # Collect training data from all other folds
            train_gnn_parts.append(aligned_predictions[other_idx]["gnn_prob"])
            train_tabular_parts.append(aligned_predictions[other_idx]["tabular_prob"])
            train_y_parts.append(aligned_predictions[other_idx]["y_true"])

        # Skip the held-out fold. Gather every other fold's predictions as training data
        train_gnn = np.concatenate(train_gnn_parts)
        train_tabular = np.concatenate(train_tabular_parts)
        train_y = np.concatenate(train_y_parts)

        # Fit the meta-learner on out-of-fold data
        model = train_meta_learner(train_gnn, train_tabular, train_y, C=C)

        # Apply to this fold's predictions
        held_out = aligned_predictions[held_out_idx]
        eval_result = evaluate_meta_learner(model, held_out["gnn_prob"], held_out["tabular_prob"], held_out["y_true"])

        # Extract the learned coefficients for interpretation. coef_[0] is [gnn_coef, tabular_coef]; intercept_ is a scalar
        gnn_coef = float(model.coef_[0, 0])
        tabular_coef = float(model.coef_[0, 1])
        intercept = float(model.intercept_[0])

        # Store this fold's predictions, score, coefficients and metadata
        per_fold.append({
            "fold": held_out["fold"],
            "y_true": eval_result["y_true"],
            "y_prob": eval_result["y_prob"],
            "auc": eval_result["auc"],
            "gnn_coef": gnn_coef,
            "tabular_coef": tabular_coef,
            "intercept": intercept,
            "subject_ids": held_out["subject_ids"],
            "n_train": len(train_y),
            "n_val": len(held_out["y_true"]),
        })

    # Aggregate across folds
    aucs = np.array([f["auc"] for f in per_fold])
    gnn_coefs = np.array([f["gnn_coef"] for f in per_fold])
    tabular_coefs = np.array([f["tabular_coef"] for f in per_fold])
    intercepts = np.array([f["intercept"] for f in per_fold])

    # return per-fold results and cross-fold summary statistics
    return {
        "per_fold": per_fold,
        "mean_auc": float(aucs.mean()),
        "std_auc": float(aucs.std(ddof=1)),
        "mean_gnn_coef": float(gnn_coefs.mean()),
        "mean_tabular_coef": float(tabular_coefs.mean()),
        "std_gnn_coef": float(gnn_coefs.std(ddof=1)),
        "std_tabular_coef": float(tabular_coefs.std(ddof=1)),
        "mean_intercept": float(intercepts.mean()),
        "regularisation_C": C,
    }


def summarise_meta_learner(result: dict[str, Any], baseline_aucs: dict[str, Any]) -> str:
    """
    Format the meta-learner results as a human-readable string.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Meta-learner fusion: cross-fold logistic regression")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Regularisation (C): {result['regularisation_C']}")
    lines.append(f"Standalone GNN AUC: {baseline_aucs['gnn_mean_auc']:.4f}")
    lines.append(f"Standalone Tabular AUC: {baseline_aucs['tabular_mean_auc']:.4f}")
    lines.append(f"Meta-learner AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    lines.append("")

    lines.append(f"Improvement over tabular alone: {result['mean_auc'] - baseline_aucs['tabular_mean_auc']:+.4f}")
    lines.append(f"Improvement over GNN alone: {result['mean_auc'] - baseline_aucs['gnn_mean_auc']:+.4f}")
    
    lines.append("")

    lines.append("Learned coefficients (mean +/- std across folds):")
    lines.append(f"GNN coefficient: {result['mean_gnn_coef']:+.4f} +/- {result['std_gnn_coef']:.4f}")
    lines.append(f"Tabular coefficient: {result['mean_tabular_coef']:+.4f} +/- {result['std_tabular_coef']:.4f}")
    lines.append(f"Intercept: {result['mean_intercept']:+.4f}")

    lines.append("")

    lines.append("Per-fold coefficients:")

    lines.append(f" {'Fold':<6}{'AUC':<10}{'GNN coef':<14}{'Tab coef':<14}{'Intercept':<12}")

    # Print one row per fold, columns aligned to header 
    for fold in result["per_fold"]:
        lines.append(
            f"{fold['fold']:<6}"
            f"{fold['auc']:<10.4f}"
            f"{fold['gnn_coef']:<14.4f}"
            f"{fold['tabular_coef']:<14.4f}"
            f"{fold['intercept']:<12.4f}"
        )
    lines.append("")

    # Interpret the coefficient stability
    gnn_range = max(f["gnn_coef"] for f in result["per_fold"]) - min(f["gnn_coef"] for f in result["per_fold"])
    tab_range = max(f["tabular_coef"] for f in result["per_fold"]) - min(f["tabular_coef"] for f in result["per_fold"])

    # Contribution of each model measured by mean absolute coefficient. Larger magnitude = more influence on the final prediction
    total = abs(result["mean_gnn_coef"]) + abs(result["mean_tabular_coef"])

    # Only report relative contribution if theyre not both zero
    if total > 0:
        gnn_share = abs(result["mean_gnn_coef"])/total
        tab_share = abs(result["mean_tabular_coef"])/total
        lines.append(f"Relative contribution (by |coefficient|):")
        lines.append(f"GNN: {gnn_share:.1%}")
        lines.append(f"Tabular: {tab_share:.1%}")
        lines.append("")

    # Report raw coefficient ranges alongside stability verdict
    lines.append("Coefficient stability (range across folds):")
    lines.append(f"GNN coefficient range: {gnn_range:.4f}")
    lines.append(f"Tabular coefficient range: {tab_range:.4f}")

    # Classify stability by how widely both coefficients spread across folds
    if gnn_range < 1.0 and tab_range < 1.0:
        lines.append("(Coefficients cluster tightly so it's a stable finding.)")
    elif gnn_range < 3.0 and tab_range < 3.0:
        lines.append("(Coefficients vary moderately across folds.)")
    else:
        lines.append("(Coefficients vary widely so fusion is not stable.)")

    return "\n".join(lines)