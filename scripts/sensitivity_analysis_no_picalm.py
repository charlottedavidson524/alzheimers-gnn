"""
This is for the pruposes of sensitivity analysis. It's a LR, feature-selected, full model but without 
PICALM_G_count due to it's correlation with APOE e4 status in this cohort (Spearman r=0.477, p<0.001).
See `docs/findings/tabular-baselines.md` and `tests/test_picalm_apoe.py` for more reasoning.

The ROC-AUC produced here should provide an unconfounded tabular performance estimate which doesn't depend on the 
PICALM-APOE correlation that is specifically found in this sample.
 
Compare these results with the confounded result (ROC-AUC 0.830 with PICALM included).
Find these in `docs/findings/tabular-baselines.md`.
 
Outputs:
    - results/baselines/logistic_compact_full_no_picalm_per_fold.csv
    - results/baselines/logistic_compact_full_no_picalm_summary.csv
    - results/baselines/logistic_compact_full_no_picalm_log.txt
 
Run:
    - python scripts/sensitivity_analysis_no_picalm.py
"""

from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from agnn.baselines.data_processing import load_baseline_data
from agnn.config import load_config
from agnn.evaluation.cross_validate import cross_validate, format_summary

# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────
cfg = load_config()
results_dir = Path(cfg["paths"]["results"]) / "baselines"
results_dir.mkdir(parents=True, exist_ok=True)
 
log = []
def show(msg=""):
    print(msg)
    log.append(msg)

# ──────────────────────────────────────────────────────────────────────
# Load feature selected full  data and drop the PICALM column
# ──────────────────────────────────────────────────────────────────────
show("="*70)
show("Sensitivity analysis: LR feature-selected-full minus PICALM_G_count")
show("="*70)
 
X, y, feature_names = load_baseline_data("compact-full")
 
# Find PICALM's column index and drop it.
picalm_idx = feature_names.index("PICALM_G_count")
X = np.delete(X, picalm_idx, axis=1)
feature_names = [f for f in feature_names if f != "PICALM_G_count"]
 
show(f"After dropping PICALM_G_count: {len(feature_names)} features")
show(f"Participants: {len(y)}")
show(f"Carriers: {int(y.sum())} ({100*y.mean():.1f}%)")
show(f"Non-carriers: {int((1-y).sum())}")
show("")

# ──────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────
def make_model():
    return LogisticRegression(max_iter=5000)
 
# ──────────────────────────────────────────────────────────────────────
# Run cross-validation
# ──────────────────────────────────────────────────────────────────────
result = cross_validate(make_model, X, y)

# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────
show("Per-fold results:")
show(result["per_fold"].to_string(index=False))
show("")
show("Summary (mean +/- std, [95% CI]):")
show(format_summary(result["summary"]))
show("")
 
# ──────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────
per_fold_path = results_dir/"logistic_compact_full_no_picalm_per_fold.csv"
summary_path = results_dir/"logistic_compact_full_no_picalm_summary.csv"
log_path = results_dir/"logistic_compact_full_no_picalm_log.txt"
 
result["per_fold"].to_csv(per_fold_path, index=False)
result["summary"].to_csv(summary_path, index=False)
 
show(f"Saved per-fold results: {per_fold_path}")
show(f"Saved summary CSV: {summary_path}")
show(f"Saved terminal log: {log_path}")
 
log_path.write_text("\n".join(log), encoding="utf-8")