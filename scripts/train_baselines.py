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
    - python scripts/train_baselines.py
"""

from pathlib import Path 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# Logging
log = []
def show(msg=""):
    print(msg)
    log.append(msg)

# ──────────────────────────────────────────────────────────────────────
# Define the baseline variants

# Pairs a short name with a data loading variant and a model factory.
# Model factory pattern is required by the cross_validate function
# ──────────────────────────────────────────────────────────────────────

BASELINES = [
    # Feature selected logistic regression (both feature sets)
    {
        "name": "logistic_compact_light",
        "data_variant": "compact-light",
        "model_factory": lambda: LogisticRegression(max_iter=5000),
    },
    {
        "name": "logistic_compact_full",
        "data_variant": "compact-full",
        "model_factory": lambda: LogisticRegression(max_iter=5000),
    },
 
    # LASSO (L1-penalised logistic regression) with all features C=1.0 (default). Adjust later if regularisation
    # strength needs tuning.
    {
        "name": "lasso_all_light",
        "data_variant": "all-features-light",
        "model_factory": lambda: LogisticRegression(solver="saga", max_iter=10000, C=1.0, l1_ratio=1.0)
    },
    {
        "name": "lasso_all_full",
        "data_variant": "all-features-full",
        "model_factory": lambda: LogisticRegression(solver="saga", max_iter=10000, C=1.0, l1_ratio=1.0)
    },
 
    # Random forest with all features. 500 trees because its a reasonable  default (trades a bit of extra compute 
    # for lower variance in feature-importance estimates.
    {
        "name": "random_forest_all_light",
        "data_variant": "all-features-light",
        "model_factory": lambda: RandomForestClassifier(n_estimators=500, random_state=cfg["seed"])
    },
    {
        "name": "random_forest_all_full",
        "data_variant": "all-features-full",
        "model_factory": lambda: RandomForestClassifier(n_estimators=500, random_state=cfg["seed"])
    },
]

# ──────────────────────────────────────────────────────────────────────
# Run all variants
# ──────────────────────────────────────────────────────────────────────

show("="*70)
show("Tabular baselines")
show("="*70)

# Collect for final comparison table
all_summaries = []  

for baseline in BASELINES:
    name = baseline["name"]
    show("")
    show("="*70)
    show(f"Baseline: {name}")
    show("="*70)
 
    # Load the data for the variant.
    X, y, feature_names = load_baseline_data(baseline["data_variant"])
    show(f"Participants: {len(y)}")
    show(f"Carriers: {int(y.sum())} ({100*y.mean():.1f}%)")
    show(f"Non-carriers: {int((1-y).sum())}")
    show(f"Features: {len(feature_names)}")
 
    # Run cross-validation.
    result = cross_validate(baseline["model_factory"], X, y)
 
    # Report and save per-baseline results.
    show("")
    show("Per-fold results:")
    show(result["per_fold"].to_string(index=False))
    show("")
    show("Summary (mean +/- std, [95% CI]):")
    show(format_summary(result["summary"]))
 
    # Set results to csv
    result["per_fold"].to_csv(results_dir/f"{name}_per_fold.csv", index=False)
    result["summary"].to_csv(results_dir/f"{name}_summary.csv", index=False)
 
    # Collect the summary rows for the final comparison table.
    summary_with_name = result["summary"].copy()
    summary_with_name.insert(0, "baseline", name)
    all_summaries.append(summary_with_name)

# ──────────────────────────────────────────────────────────────────────
# Create combined comparison table
# ──────────────────────────────────────────────────────────────────────

show("")
show("="*70)
show("Combined comparison across all six baselines")
show("="*70)
 
combined = pd.concat(all_summaries, ignore_index=True)
combined.to_csv(results_dir/"combined_summary.csv", index=False)
 
# Reshaping for readability in the terminal
comparison_wide = combined.pivot(index="baseline", columns="metric", values="mean")

show("")
show("Mean metrics per baseline:")
show(comparison_wide.round(3).to_string())

# ──────────────────────────────────────────────────────────────────────
# Save full terminal log
# ──────────────────────────────────────────────────────────────────────
log_path = results_dir/"all_baselines_log.txt"

show("")
show(f"Saved combined summary CSV: {results_dir/'combined_summary.csv'}")
show(f"Saved terminal log: {log_path}")

log_path.write_text("\n".join(log), encoding="utf-8")