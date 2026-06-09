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




