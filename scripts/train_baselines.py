"""
Train the feature-selected logistic regression baseline (tabular-light).

This runs the LR variant on the n=79 modelling cohort using the compact-light feature set. Saves per-fold and
summary metrics.

Run it from the project root:
    - python scripts/train_baselines.py

Future extensions:
    - LASSO logistic regression (L1-penalised)
    - Random forest
    - Compact full variants (with blood panel, n=76)
"""

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from agnn.baselines.data_processing import load_baseline_data
from agnn.config import load_config
from agnn.evaluation.cross_validate import cross_validate, format_summary

# ──────────────────────────────────────────────────────────────────────
# Setup the results directory
# ──────────────────────────────────────────────────────────────────────

cfg = load_config()
results_dir = Path(cfg["paths"]["results"])/"baselines"
results_dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Load the data
# ──────────────────────────────────────────────────────────────────────

X, y, feature_names = load_baseline_data("compact-light")
 
print("=" * 70)
print("Logistic regression: compact-light (tabular-light)")
print("=" * 70)
print(f"Participants: {len(y)}")
print(f"Carriers: {int(y.sum())} ({100*y.mean():.1f}%)")
print(f"Non-carriers: {int((1-y).sum())}")
print(f"Features: {len(feature_names)}")
print()

# ──────────────────────────────────────────────────────────────────────
# Define the model factory
#
# Set max_iter high so convergence isn't a limiting factor. No 
# regularisation tuning (for LASSO only)
# ──────────────────────────────────────────────────────────────────────
def make_model():
    return LogisticRegression(max_iter=5000)
 
 
# ──────────────────────────────────────────────────────────────────────
# Run cross-validation
# ──────────────────────────────────────────────────────────────────────
result = cross_validate(make_model, X, y)
 
 
# ──────────────────────────────────────────────────────────────────────
# Report to terminal
# ──────────────────────────────────────────────────────────────────────
print("Per-fold results:")
print(result["per_fold"].to_string(index=False))
print()
print("Summary (mean +/- std, [95% CI]):")
print(format_summary(result["summary"]))
print()

# ──────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────
per_fold_path = results_dir/"logistic_compact_light_per_fold.csv"
summary_path = results_dir/"logistic_compact_light_summary.csv"
 
result["per_fold"].to_csv(per_fold_path, index=False)
result["summary"].to_csv(summary_path, index=False)
 
print(f"Saved per-fold results: {per_fold_path}")
print(f"Saved summary: {summary_path}")