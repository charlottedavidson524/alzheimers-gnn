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

