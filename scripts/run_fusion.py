"""
This is the driver script for late fusion analysis. It runs both fusion methods:

    - Weighted averaging via grid search over GNN weight
    - Meta-learner via cross-fold logistic regression

Reports both methods side-by-side and saves all intermediate results and plots.

Inputs that need to exist befire this script is ran:
    - results/fusion/gnn_subject_shared_splits/fold_N/predictions.npz
    - results/fusion/tabular_logistic_compact_full/fold_N_predictions.npz

Outputs (created by this script):
    results/fusion/full_analysis_YYYYMMDD_HHMMSS/
        COMBINED:
            - summary.txt: Human-readable summary of both methods
            - combined_summary.json: Machine-readable summary of both methods
            - fold_N_predictions.npz: Aligned predictions per fold

        WEIGHTED AVERAGING:
            - weighted_averaging_summary.json: weighted averahing specific results
            - per_weight_aucs.csv: Full weight x fold AUC matrix
            - auc_vs_weight.png: Plot of AUC vs GNN weight
            - per_fold_optimal_weights.png: Bar chart of per-fold optimal weights

        META LEARNER:
            - meta_learner_summary.json: Meta-learner-specific results
            - meta_learner_per_fold.csv: Per-fold coefficients and AUC
            - meta_learner_coefficients.png: Bar chart of per-fold coefficients

        COMPARISON
            - method_comparison.png: Bar chart of all methods' AUCs

Run from the project root:
    - python scripts/run_fusion.py
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from agnn.config import load_config
from agnn.fusion.load_predictions import load_aligned_predictions
from agnn.fusion.meta_learner import cross_fold_meta_learner, summarise_meta_learner
from agnn.fusion.weighted_averaging import compute_baseline_aucs, grid_search_weights, summarise_grid_search


def save_aligned_predictions(output_dir: Path, aligned_predictions: list[dict]) -> None:
    """
    Save the aligned per-fold predictions for downstream analysis.
    """
    for fold in aligned_predictions:
        np.savez(
            output_dir/f"fold_{fold['fold']}_predictions.npz",
            y_true=fold["y_true"],
            gnn_prob=fold["gnn_prob"],
            tabular_prob=fold["tabular_prob"],
            subject_ids=np.array(fold["subject_ids"]),
        )


def save_weighted_averaging_outputs(output_dir: Path, grid_result: dict, baseline_aucs: dict, aligned_predictions: list[dict]) -> None:
    """
    Save weighted-averaging-specific outputs: JSON, CSV and plots.
    """
    # JSON summary of weighted-averaging results.
    wa_summary = {
        "cohort_size": sum(len(f["subject_ids"]) for f in aligned_predictions),
        "n_folds": len(aligned_predictions),
        "standalone_gnn_auc": baseline_aucs["gnn_mean_auc"],
        "standalone_tabular_auc": baseline_aucs["tabular_mean_auc"],
        "best_fused_auc": grid_result["best_auc"],
        "best_gnn_weight": grid_result["best_weight"],
        "best_tabular_weight": 1 - grid_result["best_weight"],
        "improvement_over_tabular": grid_result["best_auc"] - baseline_aucs["tabular_mean_auc"],
        "improvement_over_gnn": grid_result["best_auc"] - baseline_aucs["gnn_mean_auc"],
        "per_fold_optimal_gnn_weights": grid_result["per_fold_best_weights"],
        "gnn_per_fold_aucs": baseline_aucs["gnn_per_fold_aucs"].tolist(),
        "tabular_per_fold_aucs": baseline_aucs["tabular_per_fold_aucs"].tolist(),
    }
    with open(output_dir/"weighted_averaging_summary.json", "w") as f:
        json.dump(wa_summary, f, indent=2)

    # Full weight x fold matrix as CSV.
    df = pd.DataFrame(
        grid_result["per_weight_fold_aucs"],
        index=grid_result["weight_grid"],
        columns=[f"fold_{i}" for i in range(grid_result["per_weight_fold_aucs"].shape[1])],
    )

    df.index.name = "gnn_weight"
    df["mean_auc"] = grid_result["mean_aucs"]
    df.to_csv(output_dir/"per_weight_aucs.csv")

    plot_auc_vs_weight(output_dir, grid_result, baseline_aucs)
    plot_per_fold_optimal_weights(output_dir, grid_result)


def save_meta_learner_outputs(output_dir: Path, meta_result: dict) -> None:
    """
    Save meta-learner-specific outputs: JSON, CSV and plot.
    """
    # JSON summary of meta-learner results.
    ml_summary = {
        "mean_auc": meta_result["mean_auc"],
        "std_auc": meta_result["std_auc"],
        "mean_gnn_coef": meta_result["mean_gnn_coef"],
        "mean_tabular_coef": meta_result["mean_tabular_coef"],
        "std_gnn_coef": meta_result["std_gnn_coef"],
        "std_tabular_coef": meta_result["std_tabular_coef"],
        "mean_intercept": meta_result["mean_intercept"],
        "regularisation_C": meta_result["regularisation_C"],
        "per_fold_aucs": [f["auc"] for f in meta_result["per_fold"]],
        "per_fold_gnn_coefs": [f["gnn_coef"] for f in meta_result["per_fold"]],
        "per_fold_tabular_coefs": [f["tabular_coef"] for f in meta_result["per_fold"]],
        "per_fold_intercepts": [f["intercept"] for f in meta_result["per_fold"]],
    }

    with open(output_dir/"meta_learner_summary.json", "w") as f:
        json.dump(ml_summary, f, indent=2)

    # Per-fold details as CSV.
    df = pd.DataFrame([
        {
            "fold": f["fold"],
            "n_train": f["n_train"],
            "n_val": f["n_val"],
            "auc": f["auc"],
            "gnn_coef": f["gnn_coef"],
            "tabular_coef": f["tabular_coef"],
            "intercept": f["intercept"],
        }
        for f in meta_result["per_fold"]
    ])
    df.to_csv(output_dir/"meta_learner_per_fold.csv", index=False)

    plot_meta_learner_coefficients(output_dir, meta_result)


def save_combined_summary(output_dir: Path, baseline_aucs: dict, grid_result: dict, meta_result: dict) -> None:
    """
    Save a combined JSON with both methods for side-by-side comparison.
    """
    combined = {
        "standalone_gnn_auc": baseline_aucs["gnn_mean_auc"],
        "standalone_tabular_auc": baseline_aucs["tabular_mean_auc"],
        "weighted_averaging": {
            "best_auc": grid_result["best_auc"],
            "best_gnn_weight": grid_result["best_weight"],
            "improvement_over_tabular": grid_result["best_auc"] - baseline_aucs["tabular_mean_auc"],
            "per_fold_optimal_gnn_weights": grid_result["per_fold_best_weights"],
        },
        "meta_learner": {
            "mean_auc": meta_result["mean_auc"],
            "std_auc": meta_result["std_auc"],
            "improvement_over_tabular": meta_result["mean_auc"] - baseline_aucs["tabular_mean_auc"],
            "mean_gnn_coef": meta_result["mean_gnn_coef"],
            "mean_tabular_coef": meta_result["mean_tabular_coef"],
        },
    }
    with open(output_dir/"combined_summary.json", "w") as f:
        json.dump(combined, f, indent=2)


def plot_auc_vs_weight(output_dir: Path, grid_result: dict, baseline_aucs: dict) -> None:
    """
    Plot mean AUC as a function of GNN weight, with per-fold curves.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-fold curves as light lines
    n_folds = grid_result["per_weight_fold_aucs"].shape[1]
    for fold_idx in range(n_folds):
        ax.plot(grid_result["weight_grid"], grid_result["per_weight_fold_aucs"][:, fold_idx], alpha=0.3, label=f"Fold {fold_idx}" if fold_idx == 0 else None)

    # Mean AUC as the main series (thick black line)
    ax.plot(grid_result["weight_grid"], grid_result["mean_aucs"], color="black", linewidth=2, label="Mean AUC")

    # Reference lines for standalone baselines.
    ax.axhline(y=baseline_aucs["tabular_mean_auc"], color="blue", linestyle=":", label=f"Tabular alone ({baseline_aucs['tabular_mean_auc']:.3f})")
    ax.axhline(y=baseline_aucs["gnn_mean_auc"], color="green", linestyle=":", label=f"GNN alone ({baseline_aucs['gnn_mean_auc']:.3f})")
    ax.axvline(x=grid_result["best_weight"], color="red", linestyle="--", label=f"Best weight ({grid_result['best_weight']:.2f}, AUC={grid_result['best_auc']:.3f})")

    ax.set_xlabel("GNN weight")
    ax.set_ylabel("AUC")
    ax.set_title("Fusion AUC vs GNN weight (weighted averaging)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(output_dir/"auc_vs_weight.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_fold_optimal_weights(output_dir: Path, grid_result: dict) -> None:
    """
    Bar chart of each fold's individually-optimal GNN weight.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    fold_indices = list(range(len(grid_result["per_fold_best_weights"])))
    ax.bar(fold_indices, grid_result["per_fold_best_weights"], color="steelblue")
    ax.axhline(y=grid_result["best_weight"], color="red", linestyle="--", label=f"Grand-optimum GNN weight ({grid_result['best_weight']:.2f})")

    ax.set_xlabel("Fold")
    ax.set_ylabel("Optimal GNN weight (for just this fold)")
    ax.set_title("Weighted averaging: per-fold optimal weights")
    ax.set_xticks(fold_indices)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir/"per_fold_optimal_weights.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_meta_learner_coefficients(output_dir: Path, meta_result: dict) -> None:
    """
    Bar chart of per-fold learned coefficients for both models.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    n_folds = len(meta_result["per_fold"])
    fold_indices = np.arange(n_folds)
    gnn_coefs = [f["gnn_coef"] for f in meta_result["per_fold"]]
    tab_coefs = [f["tabular_coef"] for f in meta_result["per_fold"]]

    # Side-by-side bars per fold: GNN coefficient vs tabular coefficient.
    width = 0.35
    ax.bar(fold_indices - width / 2, gnn_coefs, width, label="GNN coefficient", color="green")
    ax.bar(fold_indices + width / 2, tab_coefs, width, label="Tabular coefficient", color="blue")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Learned coefficient")
    ax.set_title("Meta-learner: per-fold coefficients")
    ax.set_xticks(fold_indices)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir/"meta_learner_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_method_comparison(output_dir: Path, baseline_aucs: dict, grid_result: dict, meta_result: dict) -> None:
    """
    Bar chart comparing standalone models and both fusion methods.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    method_names = ["GNN\nalone", "Tabular\nalone", "Weighted\naveraging", "Meta-\nlearner"]
    aucs = [baseline_aucs["gnn_mean_auc"], baseline_aucs["tabular_mean_auc"], grid_result["best_auc"], meta_result["mean_auc"]]

    # WA doesn't natively track a std; use per-fold AUCs at the best weight.
    best_weight_idx = int(np.argmax(grid_result["mean_aucs"]))
    wa_std = float(grid_result["per_weight_fold_aucs"][best_weight_idx].std(ddof=1))

    stds = [float(baseline_aucs["gnn_per_fold_aucs"].std(ddof=1)), float(baseline_aucs["tabular_per_fold_aucs"].std(ddof=1)), 
            wa_std, meta_result["std_auc"]]

    colors = ["green", "blue", "orange", "purple"]
    bars = ax.bar(method_names, aucs, yerr=stds, color=colors, capsize=5, alpha=0.8)

    # Overlay each AUC value above its bar.
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{auc:.3f}", ha="center", fontsize=10, fontweight="bold")

    # Chance-level reference line.
    ax.axhline(y=0.5, color="gray", linestyle=":", label="Chance (0.5)")

    ax.set_ylabel("Mean cross-fold AUC")
    ax.set_title("Method comparison: standalone models vs fusion methods")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir/"method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def build_combined_summary_text(baseline_aucs: dict, grid_result: dict, meta_result: dict, cohort_size: int) -> str:
    """
    Create a human-readable summary combining both fusion methods.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Late fusion analysis (full results)")
    lines.append("=" * 70)
    lines.append(f"\nCohort size: {cohort_size} subjects")
    lines.append(f"Number of folds: {len(grid_result['per_fold_best_weights'])}")
    lines.append("")

    lines.append("Standalone baselines:")
    lines.append(f"GNN alone: {baseline_aucs['gnn_mean_auc']:.4f}")
    lines.append(f"Tabular alone: {baseline_aucs['tabular_mean_auc']:.4f}")
    lines.append("")

    lines.append(summarise_grid_search(grid_result, baseline_aucs))
    lines.append("")
    lines.append(summarise_meta_learner(meta_result, baseline_aucs))
    lines.append("")

    lines.append("=" * 70)
    lines.append("Method comparison")
    lines.append("=" * 70)
    lines.append(f"{'Method':<25}{'Mean AUC':<12}{'Improvement over tabular':<30}")
    lines.append(f"{'-'*67}")
    lines.append(f"{'GNN alone':<25}{baseline_aucs['gnn_mean_auc']:<12.4f}"
                 f"{baseline_aucs['gnn_mean_auc'] - baseline_aucs['tabular_mean_auc']:+.4f}")
    lines.append(f"{'Tabular alone':<25}{baseline_aucs['tabular_mean_auc']:<12.4f}"
                 f"{0.0:+.4f} (baseline)")
    lines.append(f"{'Weighted averaging':<25}{grid_result['best_auc']:<12.4f}"
                 f"{grid_result['best_auc'] - baseline_aucs['tabular_mean_auc']:+.4f}")
    lines.append(f"{'Meta-learner':<25}{meta_result['mean_auc']:<12.4f}"
                 f"{meta_result['mean_auc'] - baseline_aucs['tabular_mean_auc']:+.4f}")

    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    results_root = Path(cfg["paths"]["results"])

    # Input directories (both have got to exist
    gnn_dir = results_root/"fusion"/"gnn_subject_shared_splits"
    tabular_dir = results_root/"fusion"/"tabular_logistic_compact_full"

    # Output directory. Timestamped so multiple runs are preserved.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = results_root / "fusion" / f"full_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GNN predictions: {gnn_dir}")
    print(f"Tabular predictions: {tabular_dir}")
    print(f"Output directory: {output_dir}\n")

    # Load and align predictions from both models
    print("Loading and aligning predictions")
    aligned = load_aligned_predictions(gnn_dir, tabular_dir, n_folds=5)
    cohort_size = sum(len(f["subject_ids"]) for f in aligned)
    print(f"Loaded {len(aligned)} folds ({cohort_size} subjects total).")

    # Standalone reference AUCs for both models
    baseline_aucs = compute_baseline_aucs(aligned)
    print(f"Standalone GNN AUC: {baseline_aucs['gnn_mean_auc']:.4f}")
    print(f"Standalone Tabular AUC: {baseline_aucs['tabular_mean_auc']:.4f}\n")

    # Weighted averaging via grid search
    print("Running weighted-averaging grid search:")
    grid_result = grid_search_weights(aligned)
    print(f"Best weighted-averaging AUC: {grid_result['best_auc']:.4f} at GNN weight {grid_result['best_weight']:.2f}\n")

    # Meta-learner via cross-fold logistic regression
    print("Running meta-learner cross-fold analysis:")
    meta_result = cross_fold_meta_learner(aligned)
    print(f"Meta-learner AUC: {meta_result['mean_auc']:.4f} ± {meta_result['std_auc']:.4f}\n")

    # Save aligned predictions and per-method outputs
    save_aligned_predictions(output_dir, aligned)
    save_weighted_averaging_outputs(output_dir, grid_result, baseline_aucs, aligned)
    save_meta_learner_outputs(output_dir, meta_result)
    save_combined_summary(output_dir, baseline_aucs, grid_result, meta_result)

    # Cross-method comparison plot
    plot_method_comparison(output_dir, baseline_aucs, grid_result, meta_result)

    # Human-readable combined summary to console and file
    summary_text = build_combined_summary_text(baseline_aucs, grid_result, meta_result, cohort_size)
    print(summary_text)

    with open(output_dir/"summary.txt", "w") as f:
        f.write(summary_text + "\n")

    print(f"\nAll fusion results saved to: {output_dir}")


if __name__ == "__main__":
    main()