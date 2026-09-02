"""Smoke test for weighted averaging fusion."""

from pathlib import Path

from agnn.fusion.load_predictions import load_aligned_predictions
from agnn.fusion.weighted_averaging import (
    compute_baseline_aucs,
    grid_search_weights,
    summarise_grid_search,
)


aligned = load_aligned_predictions(
    gnn_dir=Path("results/fusion/gnn_subject_shared_splits"),
    tabular_dir=Path("results/fusion/tabular_logistic_compact_full"),
    n_folds=5,
)

baseline_aucs = compute_baseline_aucs(aligned)
grid_result = grid_search_weights(aligned)

print(summarise_grid_search(grid_result, baseline_aucs))