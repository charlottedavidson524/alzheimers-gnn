"""
Theis file reruns logistic_compact_full with shared CV splits and fold aware preprocessing. It produces per-fold predictions 
saved as .npz files. These are ready for late fusion with the GNN predictions that will be created as the next step.

There are a few differences from the original run in scripts/train_baselines.py. These are:

- It uses the intersection cohort (n=74) rather than the full compact-full cohort (n=76), because fusion needs the same subjects as the GNN.

- It loads pre-computed CV splits from data/shared_cv_splits.json. This way the tabular and GNN models predict on the exact same subjects in the
  exact same folds.

- It fits imputation and standardisation inside each CV fold, rather than on the full cohort. This was an issue previouslt. 

- It saves per-fold predictions (y_true, y_prob, subject_ids) so they can be combined with GNN predictions during fusion

Original results in results/baselines/ aren't changed. These are just additional results.

Run:
    - python scripts/rerun_tabular_for_fusion.py
"""

from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from agnn.baselines.data_processing_shared import load_baseline_data_for_fusion
from agnn.config import load_config
from agnn.evaluation.cross_validate import format_summary
from agnn.evaluation.cross_validate_shared import cross_validate_shared, load_shared_splits


def main() -> None:
    # Load the config and resolve the path to the shared CV splits
    cfg = load_config()
    project_root = Path(cfg["paths"]["project_root"])
    splits_path = project_root/"data"/"shared_cv_splits.json"

    # This creates the output directory for the fusion-ready results. They're kept separate from the original baseline results 
    # because this lets me keep both experimental versions. Good for keeping track of things
    output_dir = Path(cfg["paths"]["results"])/"fusion"/"tabular_logistic_compact_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Rerunning logistic_compact_full for fusion")
    print("="*70)
    print(f"Splits file: {splits_path}")
    print(f"Output dir: {output_dir}")

    # Load the shared CV splits
    subjects, labels, folds = load_shared_splits(splits_path)
    print(f"\nIntersection cohort: {len(subjects)} subjects, {len(folds)} folds.")

    # Load the compact-full feature set which has been filtered down to the intersection cohort
    X, y, feature_names, subject_ids = load_baseline_data_for_fusion(variant="compact-full", subject_list=subjects)

    print(f"Features: {len(feature_names)}")
    print(f"Carriers: {int(y.sum())} ({100*y.mean():.1f}%)")
    print(f"Non-carriers: {int((1-y).sum())}")

    # Run cross-validation with fold aware preprocessing and shared splits
    print("\nRunning 5-fold CV with fold-aware imputation and standardisation")

    result = cross_validate_shared(model_factory=lambda: LogisticRegression(max_iter=5000), X=X, y=y, subject_ids=subject_ids, 
                                   folds=folds, seed=cfg["seed"])

    # Report the metrics
    print("\nPer fold metrics:")
    print(result["per_fold"].to_string(index=False))
    print("\nSummary (mean +/- std, [95% CI]):")
    print(format_summary(result["summary"]))

    # Save per-fold metrics and summary CSVs
    result["per_fold"].to_csv(output_dir/"per_fold_metrics.csv", index=False)
    result["summary"].to_csv(output_dir/"summary.csv", index=False)

    # Save per-fold predictions as .npz files. These are the inputs to fusion so this is key.
    for pred in result["predictions"]:
        fold_idx = pred["fold"]
        np.savez(output_dir/f"fold_{fold_idx}_predictions.npz", y_true=pred["y_true"], y_prob=pred["y_prob"], subject_ids=np.array(pred["subject_ids"]))

    print(f"\nSaved per fold predictions to: {output_dir}")

    # Show new mean AUC
    mean_auc = result["per_fold"]["roc_auc"].mean()
    print(f"\nNew mean AUC: {mean_auc:.4f}")

if __name__ == "__main__":
    main()