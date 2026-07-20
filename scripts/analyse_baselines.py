"""
Feature importance for the tabular baselines.

There are three outputs:
    - Logistic regression coefficients from the best baseline (feature selected, full)
    - Random forst feature importances
    - LASSO feature selection stability across CV folds

Outputs wll go to results/baselines/feature_importance/
    - lr_compact_full_coefficients.csv
    - rf_compact_full_importances.csv
    - rf_all_full_importances.csv
    - lasso_stability.csv
    - analyse_baselines_log.txt

Run from 
    - python scripts/analyse_baselines.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from agnn.baselines.data_processing import load_baseline_data
from agnn.config import load_config

# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────
cfg = load_config()
results_dir = Path(cfg["paths"]["results"])/"baselines"/"feature_importance"
results_dir.mkdir(parents=True, exist_ok=True)
 
log = []
def show(msg=""):
    print(msg)
    log.append(msg)

# ──────────────────────────────────────────────────────────────────────
# Logistic regression coefficients on the feature selected, full
# ──────────────────────────────────────────────────────────────────────

show("="*70)
show("Logistic regression coefficients")
show("="*70)
 
X, y, feature_names = load_baseline_data("compact-full")
 
# Quickly fit model on whole cohort
lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X, y)
 
# Features are standardised -> coefficient size is feature importance
lr_coefficients = lr_model.coef_[0]

# Create dataframe
lr_table = pd.DataFrame({"feature": feature_names, "coefficient": lr_coefficients})
 
# Order by largest coefficient first
lr_table = lr_table.reindex(lr_table["coefficient"].abs().sort_values(ascending=False).index)
 
show("Top 15 features by coefficient magnitude:")
show(lr_table.head(15).to_string(index=False))
show("")

# Set to csv 
lr_table.to_csv(results_dir/"lr_compact_full_coefficients.csv", index=False)
show(f"Saved: {results_dir/'lr_compact_full_coefficients.csv'}")
show("")

# ──────────────────────────────────────────────────────────────────────
# Random forest feature importances on all features, full
# ──────────────────────────────────────────────────────────────────────

show("="*70)
show("Random forest feature importances")
show("="*70)
 
X_all, y_all, feature_names_all = load_baseline_data("all-features-full")

# Fit the model
rf_all = RandomForestClassifier(n_estimators=500, random_state=cfg["seed"])
rf_all.fit(X_all, y_all)

# Create dataframe. RF has feature importances inbuilt so use that
rf_all_table = pd.DataFrame({"feature": feature_names_all, "importance": rf_all.feature_importances_})
rf_all_table = rf_all_table.sort_values("importance", ascending=False)

# List by feature importance in acsneding order
show("Top 15 features by RF importance:")
show(rf_all_table.head(15).to_string(index=False))
show("")

# Set to csv
rf_all_table.to_csv(results_dir / "rf_all_full_importances.csv", index=False)
show(f"Saved: {results_dir / 'rf_all_full_importances.csv'}")
show("")

# ──────────────────────────────────────────────────────────────────────
# LASSO feature-selection stability across CV folds
# ──────────────────────────────────────────────────────────────────────
show("="*70)
show("LASSO feature-selection stability across CV folds")
show("="*70)
 
# Make sure to use same fold assignments as train_baselines.py.
n_splits = cfg["cv"]["n_splits"]
splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
 
# Fit LASSO on each training fold and record the coefficinets
fold_coefs = []
for train_idx, _ in splitter.split(X_all, y_all):
    lasso = LogisticRegression(penalty="l1", solver="saga", max_iter=10000, C=1.0)
    lasso.fit(X_all[train_idx], y_all[train_idx])
    fold_coefs.append(lasso.coef_[0])

# Shape: (n_folds, n_features)
fold_coefs = np.array(fold_coefs)  
 
# Choose feature in a fold if coefficient is non-zero.
selection_flags = np.abs(fold_coefs) > 1e-8
 
# How many folds each feature was selected in 
selection_counts = selection_flags.sum(axis=0)

# Mean coefficients across folds
mean_coefficients = fold_coefs.mean(axis=0)

# Create dataframe
stability_table = pd.DataFrame({"feature": feature_names_all, "folds_selected": selection_counts, "mean_coefficient": mean_coefficients})
 
# Sort by folds_selected first (descending), then by absolute
# mean coefficient (descending).
stability_table["abs_mean_coefficient"] = stability_table["mean_coefficient"].abs()
stability_table = stability_table.sort_values(["folds_selected", "abs_mean_coefficient"], ascending=[False, False])
stability_table = stability_table.drop(columns=["abs_mean_coefficient"])
 
show("Top 20 by stability and coefficient magnitude:")
show(stability_table.head(20).to_string(index=False))
show("")
 
# Print a summary of how many features fall into each stability bucket.
show("Stability summary:")
for count in range(n_splits, -1, -1):
    n_features_in_bucket = int((selection_counts == count).sum())
    show(f" Selected in {count}/{n_splits} folds: {n_features_in_bucket} features")
show("")
 
stability_table.to_csv(results_dir/"lasso_stability.csv", index=False)
show(f"Saved: {results_dir/'lasso_stability.csv'}")
show("")