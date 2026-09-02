"""
Weighted averaging fusion of GNN and tabular probability predictions.

For each fold, it combines GNN and tabular probabilities as:

    fused_prob = w_gnn * gnn_prob + (1 - w_gnn) * tabular_prob

A grid search over w_gnn in [0, 1] finds the optimal weight that maximises mean cross-fold AUC.

This code will report:
    - Best fusion weight and combined AUC
    - Per-fold optimal weights (stability check across folds)
    - Per-weight fold AUCs (for plotting AUC vs weight)
    - Comparison to standalone GNN and tabular baselines
"""

from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import roc_auc_score


def fuse_predictions(gnn_prob: np.ndarray, tabular_prob: np.ndarray, gnn_weight: float) -> np.ndarray:
    """
    Combine GNN and tabular probabilities with weighted averaging. Uses the formula above.

    Parameters
    ----------
    - gnn_prob, tabular_prob: ndarray
          Per-subject probability predictions from each model, index-aligned
    - gnn_weight: float
          Weight for the GNN prediction. Tabular gets weight (1 - gnn_weight)

    Returns
    -------
    - fused: ndarray
          Combined probability. Same shape as inputs
    """
    return gnn_weight*gnn_prob+(1-gnn_weight)*tabular_prob


def grid_search_weights(aligned_predictions: list[dict[str, Any]], weight_grid: np.ndarray | None = None) -> dict[str, Any]:
    """
    Grid search for the fusion weight that maximises mean cross-fold AUC.

    Parameters
    ----------
    - aligned_predictions: list of dict
          The per-fold aligned predictions from load_aligned_predictions
    - weight_grid: ndarray or None
          GNN weights to try. Defaults to np.arange(0.0, 1.01, 0.05)

    Returns
    -------
    - dict with keys:
          weight_grid : ndarray
              The GNN weights that were tried.
          mean_aucs: ndarray
              Mean cross-fold AUC for each weight
          per_weight_fold_aucs: ndarray, shape (n_weights, n_folds)
              Per-fold AUC at each weight
          best_weight: float
              GNN weight that maximised mean AUC
          best_auc: float
              Mean cross-fold AUC at best_weight.
          per_fold_best_weights: list of float
              Optimal GNN weight for each fold considered individually (mostly a diagnostic for stability the best_weight is 
              not stable).
    """
    # Default to a 5% step-sweep across the whole [0, 1] weight range
    if weight_grid is None:
        weight_grid = np.arange(0.0, 1.01, 0.05)

    # Pre-allocate a grid to store every combination's AUC
    n_folds = len(aligned_predictions)
    per_weight_fold_aucs = np.zeros((len(weight_grid), n_folds))

    # Grid over GNN weights, compute per-fold AUC at each weight
    for w_idx, w in enumerate(weight_grid):
        for fold_idx, fold in enumerate(aligned_predictions):
            fused = fuse_predictions(fold["gnn_prob"], fold["tabular_prob"], w)
            per_weight_fold_aucs[w_idx, fold_idx] = roc_auc_score(fold["y_true"], fused)

    # Mean across folds for each weight, pick the max
    mean_aucs = per_weight_fold_aucs.mean(axis=1)
    best_idx = int(np.argmax(mean_aucs))
    best_weight = float(weight_grid[best_idx])
    best_auc = float(mean_aucs[best_idx])

    # Per-fold optimal weights. If different folds prefer very different weights, the grand-optimum weight isn't a stable finding.
    per_fold_best_weights: list[float] = []
    for fold_idx in range(n_folds):
        fold_aucs = per_weight_fold_aucs[:, fold_idx]
        per_fold_best_weights.append(float(weight_grid[int(np.argmax(fold_aucs))]))

    return {
        "weight_grid": weight_grid,
        "mean_aucs": mean_aucs,
        "per_weight_fold_aucs": per_weight_fold_aucs,
        "best_weight": best_weight,
        "best_auc": best_auc,
        "per_fold_best_weights": per_fold_best_weights,
    }


def compute_baseline_aucs(aligned_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute standalone GNN and tabular AUCs for comparison.

    These are the AUCs the two models achieve on their own on the aligned per-fold predictions, without any fusion. Useful as 
    reference for whether fusion improves over either component alone.

    Returns
    -------
    - dict with keys:
          gnn_per_fold_aucs: ndarray
          gnn_mean_auc: float
          tabular_per_fold_aucs: ndarray
          tabular_mean_auc: float
    """
    # Pre-allocate per-fold AUC arrays for each standalone model
    n_folds = len(aligned_predictions)
    gnn_aucs = np.zeros(n_folds)
    tabular_aucs = np.zeros(n_folds)

    # Score each model's own predictions against true labels, fold by fold
    for fold_idx, fold in enumerate(aligned_predictions):
        gnn_aucs[fold_idx] = roc_auc_score(fold["y_true"], fold["gnn_prob"])
        tabular_aucs[fold_idx] = roc_auc_score(fold["y_true"], fold["tabular_prob"])

    return {
        "gnn_per_fold_aucs": gnn_aucs,
        "gnn_mean_auc": float(gnn_aucs.mean()),
        "tabular_per_fold_aucs": tabular_aucs,
        "tabular_mean_auc": float(tabular_aucs.mean()),
    }


def summarise_grid_search(grid_result: dict[str, Any], baseline_aucs: dict[str, Any]) -> str:
    """
    Format the grid-search results as a human-readable string.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Weighted averaging fusion: grid search results")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Standalone GNN AUC: {baseline_aucs['gnn_mean_auc']:.4f}")
    lines.append(f"Standalone Tabular AUC: {baseline_aucs['tabular_mean_auc']:.4f}")
    lines.append(f"Best fused AUC: {grid_result['best_auc']:.4f}")
    lines.append(f"Best GNN weight: {grid_result['best_weight']:.2f}")
    lines.append(f"Tabular weight: {1-grid_result['best_weight']:.2f}")
    lines.append("")
    lines.append(f"Improvement over tabular alone: {grid_result['best_auc'] - baseline_aucs['tabular_mean_auc']:+.4f}")
    lines.append(f"Improvement over GNN alone: {grid_result['best_auc'] - baseline_aucs['gnn_mean_auc']:+.4f}")
    lines.append("")
    lines.append("Per-fold optimal GNN weights:")

    # List each fold's individually preferred weight for the stability check
    for fold_idx, w in enumerate(grid_result["per_fold_best_weights"]):
        lines.append(f"  Fold {fold_idx}: {w:.2f}")
    lines.append("")

    # Classify stability by how widely per-fold optimal weights spread
    if len(set(grid_result["per_fold_best_weights"]))==1:
        lines.append("(All folds prefer the same weight. Very stable)")
    else:
        weight_range = (
            max(grid_result["per_fold_best_weights"])
            - min(grid_result["per_fold_best_weights"])
        )

        # Delegate thresholds to help interpretation
        if weight_range < 0.2:
            lines.append("(Per-fold weights cluster tightly, meaningstable finding)")
        elif weight_range < 0.5:
            lines.append("(Per-fold weights vary moderately, so the best weight is a compromise)")
        else:
            lines.append("(Per-fold weights vary widely, implying the best weight is not stable)")

    return "\n".join(lines)