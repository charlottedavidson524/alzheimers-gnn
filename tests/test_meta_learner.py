"""
Smoke test for meta-learner fusion.
"""

from pathlib import Path
from agnn.fusion.load_predictions import load_aligned_predictions
from agnn.fusion.weighted_averaging import compute_baseline_aucs
from agnn.fusion.meta_learner import cross_fold_meta_learner, summarise_meta_learner

# Load and align both model predictions using subject ID
aligned = load_aligned_predictions(gnn_dir=Path("results/fusion/gnn_subject_shared_splits"), tabular_dir=Path("results/fusion/tabular_logistic_compact_full"),
                                   n_folds=5)

# Calculate standalone AUC for reference
baseline_aucs = compute_baseline_aucs(aligned)

# Run cross-fold meta learner
result = cross_fold_meta_learner(aligned)

# Print summary. Compares meta-learner AUC to dstandalone baselines
print(summarise_meta_learner(result, baseline_aucs))