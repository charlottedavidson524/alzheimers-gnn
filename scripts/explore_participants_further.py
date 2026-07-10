"""
Extended exploratory analysis of participants.tsv.

Goals of this file:

- Cross tab second_phase against blood-panel availability to test sub-cohort hypothesis.
- Rank numeric features by their association with APOE e4 carrier status (effect size and FDR-corrected p-values)
- Plot correlation heatmaps of the blood panel and psychometric feature blocks.
- Cross-tab potential confounders (education, smoking, BMI category, AUDIT category, family history) against carrier status

Outputs will be saved in results/extended_exploration/

USAGE
-----
From the project root:
    - python scripts/extended_exploration.py
"""

import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import argparse
 
from agnn.config import load_config

# ──────────────────────────────────────────────────────────────────────
# Command line arguments
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Extended EDA on participants.tsv")
parser.add_argument(
    "--subset",
    choices=["full", "second_phase"],
    default="full",
    help="Which sample to analyse: 'full' (N=192) or "
         "'second_phase' (the N=79 modelling cohort). Default: full.",
)
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────
# Setup (load config, prepare output folder and log helper)
# ──────────────────────────────────────────────────────────────────────

cfg = load_config()
data_root = Path(cfg["paths"]["data_root"])
#results_dir = Path(cfg["paths"]["results"])/"extended_exploration"
#results_dir.mkdir(parents=True, exist_ok=True)

# Filter to the requested subset. Output folder differs so that
# full-sample and modelling-cohort analyses don't overwrite each other.
participants_path = data_root/"participants.tsv"

df = pd.read_csv(participants_path, sep="\t")

if args.subset == "second_phase":
    df = df[df["second_phase"] == 1].copy()
    subset_label = "second_phase = 1 (N=79 modelling cohort)"
    results_dir = Path(cfg["paths"]["results"]) / "extended_exploration_second_phase"
else:
    subset_label = "full sample (N=192)"
    results_dir = Path(cfg["paths"]["results"]) / "extended_exploration"

results_dir.mkdir(parents=True, exist_ok=True)

#participants_path = data_root/"participants.tsv"

if not participants_path.exists():
    print(f"Error: {participants_path} not found")
    print("Run: python scripts/download_data.py --stage 1")
    sys.exit(1)

log_lines = []

def show(message = ""):
    print(message)
    log_lines.append(message)

# ──────────────────────────────────────────────────────────────────────
# Load and prepare data

# Repeat setup from main exploration script instead of importing. Keeps 
# script self-contained 
# ──────────────────────────────────────────────────────────────────────
#df = pd.read_csv(participants_path, sep="\t")

# Strip whitespace from genotype columns
df["APOE_haplotype"] = df["APOE_haplotype"].astype(str).str.strip()
df["PICALM_rs3851179"] = df["PICALM_rs3851179"].astype(str).str.strip()

# Derive carrier flag. It will contain '4'
df["apoe_e4_carrier"] = df["APOE_haplotype"].str.lower().str.contains("4")

show("=" * 70)
show(f"Extended EDA: participants.tsv ({subset_label})")
show("=" * 70)
show(f"N participants: {len(df)}")
show(f"e4 carriers: {df['apoe_e4_carrier'].sum()}")
show("")

# ──────────────────────────────────────────────────────────────────────
# second_phase investigation
#
# The second_phase variable indicates a sub-cohort that had the
# extended protocol including blood draws. So, "has blood data"
# should map onto second_phase cleanly. Checking this.
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("second_phase vs blood data availability")
show("=" * 70)

# All blood columns share the same 116 missing pattern so use one at random
# for a proxy for 'has blood data'. Chose leukocytes.
df["has_blood"] = df["leukocytes"].notna()

# Create a cross-tab. Use fillna to keep NaN visible as it's its own 
# category here
second_phase_display = df["second_phase"].fillna("(missing)")
crosstab = pd.crosstab(second_phase_display, df["has_blood"].map({True: "has blood", False: "no blood"}), margins=True)

# Display
show("Cross-tab of second_phase against blood-data availability:")
show(crosstab.to_string())
show("")

# ──────────────────────────────────────────────────────────────────────
# Feature screening. Ranking numeric features by association with APOE 
# e4 carrier status

# For each numeric variable:
#   - Mean for carriers vs non-carriers
#   - Cohen's d effect size
#   - Independent-samples t-test p-value
#   - FDR-corrected p-value (Benjamini-Hochberg)
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("Feature screening (numeric variables)")
show("=" * 70)

# Get numeric columns. Exclude IDs and the derived carrier flag (target) as well as second_phase and has_blood
# (cohort-membership indicators, not biology). Their original inclusion caused a misleading top of ranking in 
# the original run
EXCLUDED_FROM_SCREENING = {"apoe_e4_carrier", "second_phase", "has_blood"}

numeric_columns = []
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        if "id" not in col.lower() and col not in EXCLUDED_FROM_SCREENING:
            numeric_columns.append(col)
 
show(f"Screened {len(numeric_columns)} numeric variables.")
show("")

def cohens_d(group_a, group_b):
    """
    Cohen's d effect size for two independent samples. Uses pooled standard dev.
    """
    n_a, n_b = len(group_a), len(group_b)

    # Disregard if group size is 0 or 1.
    if n_a < 2 or n_b < 2:
        return np.nan
    
    # Sample variance (using n-1 denominator via ddof=1)
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)

    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n_a-1)*var_a + (n_b-1)*var_b)/(n_a+n_b-2))

    # Return NaN if pooled standard deviation is zero
    if pooled_sd == 0:
        return np.nan
    
    # return Cohen's d
    return (group_a.mean()-group_b.mean())/pooled_sd


def benjamini_hochberg(pvalues):
    """
    Benjamini-HochbergFDR correction. Returns adjusted p-values in the same order as the input.
    NaN inputs pass through as NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    valid_mask = ~np.isnan(p)
    valid_p = p[valid_mask]
    n = len(valid_p)
 
    if n == 0:
        return p
 
    # Rank the valid p-values in ascending order.
    order = np.argsort(valid_p)
    ranked = valid_p[order]
 
    # Compute the BH-adjusted values.
    adjusted_ranked = ranked*n/(np.arange(n)+1)

    # Ensure monotonicity (take running minimum from the right).
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]

    # Cap at 1.
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
 
    # Put back in original order.
    adjusted = np.empty(n)
    adjusted[order] = adjusted_ranked
 
    result = np.full_like(p, np.nan)
    result[valid_mask] = adjusted

    # Return FDRcorrected p-values by Benjamini-Hochberg
    return result

rows = []

# Sort carriers from non-carriers
for col in numeric_columns:
    carriers = df.loc[df["apoe_e4_carrier"], col].dropna()
    non_carriers = df.loc[~df["apoe_e4_carrier"], col].dropna()
 
    # If there are 0 or 1 carriers/non-carriers
    if len(carriers) < 2 or len(non_carriers) < 2:
        rows.append({
            "variable": col,
            "n_carriers": len(carriers),
            "n_non_carriers": len(non_carriers),
            "mean_carriers": carriers.mean() if len(carriers) else np.nan,
            "mean_non_carriers": non_carriers.mean() if len(non_carriers) else np.nan,
            "cohens_d": np.nan,
            "p_value": np.nan,
        })
        continue
 
    # Welch's t-test (does not assume equal variances).
    t_stat, p_val = stats.ttest_ind(carriers, non_carriers, equal_var=False)
    d = cohens_d(carriers, non_carriers)
 
    rows.append({
        "variable": col,
        "n_carriers": len(carriers),
        "n_non_carriers": len(non_carriers),
        "mean_carriers": carriers.mean(),
        "mean_non_carriers": non_carriers.mean(),
        "cohens_d": d,
        "p_value": p_val,
    })

# Create a screening dataframe
screening = pd.DataFrame(rows)

# Add FDR-corrected p-values to the dataframe
screening["p_fdr"] = benjamini_hochberg(screening["p_value"].values)

# Sort by absolute effect size (largest first). Helps find most promising features
screening["abs_d"] = screening["cohens_d"].abs()
screening = screening.sort_values("abs_d", ascending=False, na_position="last")
screening = screening.drop(columns=["abs_d"])

# Save table as CSV.
screening_csv = results_dir/"feature_screening.csv"
screening.to_csv(screening_csv, index=False)

show(f"Full ranked table saved to: {screening_csv}")
show("")

# Show the top 20 in the log.
show("Top 20 variables by absolute Cohen's d:")
show("")

top20 = screening.head(20).copy()

# Make more readable.
# Round to 3 decimal places.
for col in ["mean_carriers", "mean_non_carriers", "cohens_d"]:
    top20[col] = top20[col].round(3)

# Round to 4 sig figs if not na, scientific notation for very small p-values
for col in ["p_value", "p_fdr"]:
    top20[col] = top20[col].apply(lambda x: f"{x:.4g}" if pd.notna(x) else "n/a")

show(top20.to_string(index=False))
show("")

# Number of vars with raw p < 0.05
n_sig_raw = (screening["p_value"] < 0.05).sum()

# Number of vars with FDR-adjusted p-values < 0.05
n_sig_fdr = (screening["p_fdr"] < 0.05).sum()

show(f"Variables with raw p < 0.05: {n_sig_raw}")
show(f"Variables with FDR-adjusted < 0.05: {n_sig_fdr}")
show("")

# ──────────────────────────────────────────────────────────────────────
# Correlation heatmaps of feature blocks

# Heatmaps for blood panel and psychometrics. Should reveal internal 
# redundancies that will help with feature selection. 
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("Featuree block correlation heatmaps")
show("=" * 70)

# Define feature blocks using column names.
# Blood columns
blood_columns = [
    "leukocytes", "erythrocytes", "hemoglobin", "hematocrit", "MCV", "MCH", "MCHC", "RDW-CV", "platelets", "PDW", 
    "MPV", "P-LCR", "neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils", "neutrophils_%", 
    "lymphocytes_%", "monocytes_%", "eosinophils_%", "basophils_%", "total_cholesterol", "cholesterol_HDL",
    "non-HDL_cholesterol", "LDL_cholesterol", "triglycerides",
]

blood_columns = [c for c in blood_columns if c in df.columns]

# Psychometric columns
psychometric_columns = [c for c in df.columns if c.startswith(("BDI", "SES", "RPM", "EHI", "NEO_", "AUDIT", "MINI-COPE", "CVLT"))]

def plot_correlation_heatmap(cols, title, filename):
    """
    Plot and save a correlation heatmap for the provided columns.
    """
    if len(cols) < 2:
        show(f"Skipping {title}: only 0 or 1 columns available.")
        return
    
    # Find correlations
    corr = df[cols].corr()
 
    # Plot
    fig, ax = plt.subplots(figsize=(max(6, len(cols)*0.4), max(5, len(cols)*0.4)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Formatting and labels
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title(title, fontsize=10)
 
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    
    # Save
    out_path = results_dir/filename
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    show(f"Saved: {out_path}")

# Blood column heatmap
plot_correlation_heatmap(blood_columns, "Blood panel correlation heatmap", "corr_blood_panel.png")

# Psychometric column heatmap
plot_correlation_heatmap(psychometric_columns, "Psychometric variables correlation heatmap", "corr_psychometric.png")

# ──────────────────────────────────────────────────────────────────────
# Confounder cross-tabs

# Check whether any lifestyle/demographic variables are associated with
# carrier status by chance.
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("Confounder corss-tabs vs APOE e4 status")
show("=" * 70)

# Define columns that are potential confounders
confounders = [
    "sex",
    "education",
    "smoking_status",
    "coffee_status",
    "dementia_history_parents",
    "learning_deficits",
    "hypertension",
    "diabetes",
    "thyroid_diseases",
    "allergies",
]

confounders = [c for c in confounders if c in df.columns]

carrier_label = df["apoe_e4_carrier"].map({True: "e4 carrier", False: "non-carrier"})
 
for col in confounders:
    show(f"{col}")
    ct = pd.crosstab(df[col].fillna("(missing)"), carrier_label, margins=True)
    show(ct.to_string())

    # Chi-squared test on non-margin cells if there's enough data.
    ct_no_margins = pd.crosstab(df[col], carrier_label)
    if ct_no_margins.size >= 4 and ct_no_margins.values.sum() >= 20:
        try:
            chi2, p, dof, expected = stats.chi2_contingency(ct_no_margins)
            # Warning if counts are too small
            if (expected < 5).any():
                show(f" chi2 test: p = {p:.3g} (some expected counts < 5;")
            else:
                show(f" chi2 test: p = {p:.3g}")
        except Exception as e:
            show(f" chi2 test could not be computed: {e}")
    show("")

# Continuous confounders (BMI and AUDIT)
show("Continuous confounders (t-tests)")
for col in ["BMI", "AUDIT", "age"]:
    if col not in df.columns:
        continue

    # determine carriers and non-carriers
    carriers = df.loc[df["apoe_e4_carrier"], col].dropna()
    non_carriers = df.loc[~df["apoe_e4_carrier"], col].dropna()

    if len(carriers) < 2 or len(non_carriers) < 2:
        continue
    
    # Calculate t-statistic and p-values
    t_stat, p_val = stats.ttest_ind(carriers, non_carriers, equal_var=False)

    # Cohens d
    d = cohens_d(carriers, non_carriers)

    show(f"{col:8s}: mean carriers = {carriers.mean():.2f}, "f"mean non-carriers = {non_carriers.mean():.2f}, "f"d = {d:.3f}, p = {p_val:.3g}")

# ──────────────────────────────────────────────────────────────────────
# Save text log
# ──────────────────────────────────────────────────────────────────────
summary_path = results_dir/"summary.txt"
summary_path.write_text("\n".join(log_lines), encoding="utf-8")

print(f"Text summary saved to: {summary_path}")
