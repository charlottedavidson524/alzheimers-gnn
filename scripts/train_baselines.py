"""
Train the six tabular baseline variants. There are three types of models and two featutre-set variants.

 ---------------------------- ----------------------------------------------------------------- ----------------------------------------------------
|                            | tabular-light (n=79)                                            | tabular-full (n=76)                                |
|----------------------------|-----------------------------------------------------------------|----------------------------------------------------|
| Logistic (feature selected)| Selected features, no regularisation tuning                     | Selected features and 14-marker blood panels       |
|----------------------------|-----------------------------------------------------------------|----------------------------------------------------|
| Logistic (LASSO)           | All features, L1 penalty (feature selection via regularisation) | All features and full 28 blood markers, L1 penalty |
|----------------------------|-----------------------------------------------------------------|----------------------------------------------------|
| Random Forest              | All features, tree based, handles collinearity                  | All features and full 28-marker blood panel        |
 ---------------------------- ----------------------------------------------------------------- ----------------------------------------------------

Each combination is run through the same cross_validate() function so the metrics can be compared. Results are saved per
variant and a comparison table is printed and saved.

Run from project root:
    - - python scripts/train_baselines.py
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