"""
EDA of participants.tsv.
 
Reads the PEARL-Neuro participant table and reports:
    - Sample sizes 
    - APOE e4 carrier vs non-carrier balance 
    - PICALM rs3851179 genotype distribution
    - Missing-data patterns across blood, psychometric and demographic columns
    - Distributions of key variables by APOE e4 status
    - Cross-tabulation of APOE and PICALM (joint risk profile)
 
Plots will be saved to results/exploration/. A plain-text summary will be saved to results/exploration/summary.txt.
 
USAGE
-----
From the project root:
    - python scripts/explore_participants.py
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from agnn.config import load_config

# ──────────────────────────────────────────────────────────────────────
# Setup (loading config, setting paths and preparing output folder)
# ──────────────────────────────────────────────────────────────────────

cfg = load_config()
data_root = Path(cfg["paths"]["data_root"])
results_dir = Path(cfg["paths"]["results"])/"exploration"
results_dir.mkdir(parents=True, exist_ok=True)

participants_path = data_root/"participants.tsv"

if not participants_path.exists():
    print(f"ERROR: {participants_path} not found")
    print("Run: python scripts/download_data.py --stage 1")
    sys.exit(1)

# Collect printed output into a list so it can be saved as a file
log_lines = []

def show(message=""):
    """Print message to console and add it to the log"""
    print(message)
    log_lines.append(message)

# ──────────────────────────────────────────────────────────────────────
# Load participant table
# ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(participants_path, sep="\t")

show("=" * 70)
show("PARTICIPANT TABLE OVERVIEW")
show("=" * 70)
show(f"File: {participants_path}")
show(f"Shape: {df.shape[0]} participants by {df.shape[1]} columns")
show(f"Columns: {list(df.columns)}")

# ──────────────────────────────────────────────────────────────────────
# Find the column names that are needed

# Some datasets name columns slightly differently. Check the columns 
# that actually exist in the file and pick the right one. If the names 
# below are wrong, edit this block (the only place where column names 
# are hard-coded).
# ──────────────────────────────────────────────────────────────────────
columns = list(df.columns)
 
# APOE column (try common variations).
if "APOE" in columns:
    apoe_col = "APOE"
elif "apoe" in columns:
    apoe_col = "apoe"
elif "APOE_haplotype" in columns:
    apoe_col = "APOE_haplotype"
else:
    apoe_col = None
 
# PICALM 
if "PICALM" in columns:
    picalm_col = "PICALM"
elif "picalm" in columns:
    picalm_col = "picalm"
elif "PICALM_rs3851179" in columns:
    picalm_col = "PICALM_rs3851179"
else:
    picalm_col = None
 
# Age 
if "age" in columns:
    age_col = "age"
elif "Age" in columns:
    age_col = "Age"
else:
    age_col = None
 
# Sex 
if "sex" in columns:
    sex_col = "sex"
elif "Sex" in columns:
    sex_col = "Sex"
else:
    sex_col = None
 
# EEG/fMRI (often missing)
if "EEG" in columns:
    eeg_col = "EEG"
elif "has_eeg" in columns:
    eeg_col = "has_eeg"
else:
    eeg_col = None
 
if "fMRI" in columns:
    fmri_col = "fMRI"
elif "has_fmri" in columns:
    fmri_col = "has_fmri"
else:
    fmri_col = None
 
show("")
show("=" * 70)
show("KEY COLUMNS IDENTIFIED")
show("=" * 70)
show(f" APOE column: {apoe_col}")
show(f" PICALM column: {picalm_col}")
show(f" Age column: {age_col}")
show(f" Sex column: {sex_col}")
show(f" EEG flag: {eeg_col}")
show(f" fMRI flag: {fmri_col}")
 
if apoe_col is None:
    show("")
    show("No APOE column found. Check df.columns and update this script.")

# ──────────────────────────────────────────────────────────────────────
# APOE e4 carrier status (target variable)

# Carrier" have minimum one e4 allele (so the genotype string contains a 
# 4, e.g. 'E3/E4', 'e3e4', '34').
# ──────────────────────────────────────────────────────────────────────
if apoe_col is not None:
    show("")
    show("=" * 70)
    show("APOE: FULL SAMPLE")
    show("=" * 70)
    show("Raw genotype counts:")
    show(df[apoe_col].value_counts(dropna=False).to_string())

    # Build carrier flag (lowercase genotype and check for 4)
    apoe_text = df[apoe_col].astype(str).str.lower()
    df["apoe_e4_carrier"] = apoe_text.str.contains("4")

    n_carriers = df["apoe_e4_carrier"].sum()
    n_non_carriers = (~df["apoe_e4_carrier"]).sum()
    total = len(df)

    pct_carriers = 100 * n_carriers / total
 
    show("")
    show(f" e4 carriers: {n_carriers}  ({pct_carriers:.1f}%)")
    show(f" e4 non-carriers: {n_non_carriers}  ({100 - pct_carriers:.1f}%)")

# ──────────────────────────────────────────────────────────────────────
# PICALM rs3851179 genotype
# ──────────────────────────────────────────────────────────────────────
if picalm_col is not None:
    show("")
    show("=" * 70)
    show("PICALM rs3851179: FULL SAMPLE")
    show("=" * 70)
    show("Genotype counts:")
    show(df[picalm_col].value_counts(dropna=False).to_string())

# ──────────────────────────────────────────────────────────────────────
# APOE and PICALM cross-tabulation
# ──────────────────────────────────────────────────────────────────────
if apoe_col is not None and picalm_col is not None:
    show("")
    show("=" * 70)
    show("APOE e4 AND PICALM rs3851179 CROSS-TAB")
    show("=" * 70)
 
    # Map True/False to readable labels for the table.
    carrier_label = df["apoe_e4_carrier"].map({True: "e4 carrier", False: "non-carrier"})
    crosstab = pd.crosstab(carrier_label, df[picalm_col], margins=True, dropna=False)
    show(crosstab.to_string())

# ──────────────────────────────────────────────────────────────────────
# Neuroimaging subset (the participants who had EEG/fMRI scans)
# ──────────────────────────────────────────────────────────────────────
show("")
show("=" * 70)
show("NEUROIMAGING SUBSET")
show("=" * 70)

# Try to find a column which indicates who has EEG/fMRI
neuro_mask = None

if eeg_col is not None:
    flag = df[eeg_col].astype(str).str.lower()
    neuro_mask = flag.isin(["yes", "true", "1", "y"])
    show(f"Using column '{eeg_col}' to flag participants who took part in neuroimaging (EEG)")
elif fmri_col is not None:
    flag = df[fmri_col].astype(str).str.lower()
    neuro_mask = flag.isin(["yes", "true", "1", "y"])
    show(f"Using column '{fmri_col}' to flag participants who took part in neuroimaging (fMRI)")
else:
    show("No EEG/fMRI flag column found in participants.tsv")
    show("The 79-subject subset will need to be identified from on-disk")
    show("files later (after Stage 3 of the download).")

if neuro_mask is not None and apoe_col is not None:
    n_neuro = neuro_mask.sum()
    neuro_df = df[neuro_mask]

    # Work out proportions of apoe carriers vs non carriers from the neuroimaging subset
    n_neuro_carriers = neuro_df["apoe_e4_carrier"].sum()
    n_neuro_non = (~neuro_df["apoe_e4_carrier"]).sum()
    pct_neuro_carriers = (n_neuro_carriers/n_neuro) * 100

    show("")
    show(f"Neuroimaging subset size: {n_neuro}")
    show("APOE e4 status within the neuroimaging subset:")
    show(f"e4 carriers: {n_neuro_carriers} ({pct_neuro_carriers:.1f}%)")
    show(f"e4 non-carriers: {n_neuro_non} ({100 - pct_neuro_carriers:.1f}%)")

# ──────────────────────────────────────────────────────────────────────
# Missing data
# ──────────────────────────────────────────────────────────────────────
show("")
show("=" * 70)
show("MISSING DATA SUMMARY")
show("")

missing_per_column = df.isna().sum()
missing_per_column = missing_per_column[missing_per_column > 0]
missing_per_column = missing_per_column.sort_values(ascending=False)

if len(missing_per_column) == 0:
    show("No missing values in any column")
else:
    show(f"{len(missing_per_column)} columns have missing values.")
    show("Top 20:")
    show(missing_per_column.head(20).to_string())

# ──────────────────────────────────────────────────────────────────────
# Demographic summary
# ──────────────────────────────────────────────────────────────────────
if age_col is not None or sex_col is not None:
    show("")
    show("=" * 70)
    show("DEMOGRAPHIC SUMMARY")
    show("")

    if age_col is not None:
        show(f"Age: {age_col}")
        show(df[age_col].describe().to_string())
        show("")

    if sex_col is not None:
        show(f"Sex ({sex_col}):")
        show(df[sex_col].value_counts(dropna=False).to_string())

# ──────────────────────────────────────────────────────────────────────
# Histograms by APOE e4 status 
#
# There are two overlaid histograms for every numerical column.
# One is for carriers and one is for non-carriers.
# ──────────────────────────────────────────────────────────────────────
if apoe_col is not None:
    show("")
    show("=" * 70)
    show("NUMERIC DISTRIBUTIONS BY APOE e4 STATUS")
    show("=" * 70)
 
    # Get numeric columns. Exclude anything that looks like an ID.
    numeric_columns = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if "id" not in col.lower():
                numeric_columns.append(col)
 
    show(f"Plotting {len(numeric_columns)} numeric columns.")
 
    if len(numeric_columns) > 0:
        # Lay out the plots in a grid (4 columns wide)
        plots_per_row = 4
        n_plots = len(numeric_columns)
        n_rows = (n_plots + plots_per_row - 1) // plots_per_row
 
        fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(plots_per_row * 3, n_rows * 2.2))
        # Flatten 2D axes array into 1D list for easy iteration.
        axes = axes.flatten()
 
        for i, col in enumerate(numeric_columns):
            ax = axes[i]
            carrier_values = df.loc[df["apoe_e4_carrier"], col].dropna()
            non_carrier_values = df.loc[~df["apoe_e4_carrier"], col].dropna()
 
            ax.hist(non_carrier_values, bins=15, alpha=0.5, label="non-carrier", density=True)
            ax.hist(carrier_values, bins=15, alpha=0.5, label="e4 carrier", density=True)
            ax.set_title(col, fontsize=8)
            ax.tick_params(labelsize=7)
 
        # Hide any unused subplots in the bottom-right of the grid.
        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)
 
        axes[0].legend(fontsize=7)
        fig.suptitle("Numeric variable distributions by APOE e4 status", fontsize=10)
        fig.tight_layout()
 
        plot_path = results_dir / "distributions_by_apoe.png"
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
 
        show(f"Saved plot: {plot_path}")

# ──────────────────────────────────────────────────────────────────────
# Save the text summary
# ──────────────────────────────────────────────────────────────────────
summary_path = results_dir / "summary.txt"
summary_text = "\n".join(log_lines)
summary_path.write_text(summary_text, encoding="utf-8")
 
print("")
print(f"Text summary saved to: {summary_path}")







