"""
This is a smoke test to check that fusion prediction loading works and predictions align.

Run from the project root:
    - python tests/test_fusion_load.py
"""

from pathlib import Path
from agnn.fusion.load_predictions import load_aligned_predictions

# Point to the prediction directories that fusion is going to combine
gnn_dir = Path("results/fusion/gnn_subject_shared_splits")
tabular_dir = Path("results/fusion/tabular_logistic_compact_full")

# Load and align both model's per-fold predictions by subject ID
aligned = load_aligned_predictions(gnn_dir, tabular_dir, n_folds=5)

print(f"Loaded {len(aligned)} folds.")

# Sanity check first few entries per fold
for fold in aligned:
    print(f"\nFold {fold['fold']}:")
    print(f"n subjects: {len(fold['subject_ids'])}")
    print(f"First 3 subjects: {fold['subject_ids'][:3]}")
    print(f"y_true: {fold['y_true'][:5]}...")
    print(f"gnn_prob: {[f'{p:.3f}' for p in fold['gnn_prob'][:5]]}")
    print(f"tabular_prob: {[f'{p:.3f}' for p in fold['tabular_prob'][:5]]}")

# Cross-fold total subjects should equal the cohort size (74)
total_subjects = sum(len(f["subject_ids"]) for f in aligned)
print(f"\nTotal subjects across all val folds: {total_subjects} (expect 74)")

