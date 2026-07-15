"""
Data loading for tabular baseline models.

The pipeline loads participants.tsv, filters to the n=79 cohort (second_phase = 1), and applies the feature 
selection decisions that were documented in:
    - docs/decisions/feature_selection/blood-panel.md
    - docs/decisions/feature_selection/cvlt.md
    - docs/decisions/feature_selection/personality-cluster.md
    - docs/decisions/missing-data-imputation.md

There are four feature set variants that need to be supported:
    - compact light: features selected, no blood markers (n=79 participants available)
    - compact full: compact features as well as the 12-marker blood panel (n=76 with blood data)
    - all features light: all available features but no blood (n=79)
    - all features full :  all available features and blood (n=76)

Compact variants feed the logistic regression baselines while all feature variants feed LASSO and random forest
baselines.
 
Returns X (numpy), y (numpy) and feature_names (list) as arrays ready for scikit-learn.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from agnn.config import load_config

# ──────────────────────────────────────────────────────────────────────
# Feature-set definitions 
# 
# Taken from the three decision documents
# ──────────────────────────────────────────────────────────────────────

# These are features that are always included. These are always available for the n=79 modelling cohort with no 
# concerns about missingness.
COMPACT_LIGHT_FEATURES = [
    "age",
    "sex",
    "education",
    "BMI",
    "PICALM_G_count", # This is derived (the number of G alleles at PICALM rs3851179)
    "BDI", # Depression (this is a proxy)
    "CVLT_total_learning", # This is derived (sum of CVLT_1 to CVLT_5)
    "CVLT_13", # Recognition of false alarms
    "RPM", # Intelligence
    "smoking_status",
    "dementia_history_parents"
]
 
# Blood-panel additions. Available only for the ~76 second_phase = 1 participants with blood data. The reason
# for this is justified in the extended-eda section.
COMPACT_FULL_BLOOD_ADDITIONS = [
    "leukocytes",
    "erythrocytes",
    "hemoglobin",
    "hematocrit",
    "platelets",
    "neutrophils_%",
    "lymphocytes_%",
    "monocytes_%",
    "eosinophils_%",
    "basophils_%",
    "total_cholesterol",
    "cholesterol_HDL",
    "triglycerides",
    "HSV_r"
]

# This is for all features (light). It's every non-blood feature that has potential signal. For LASSO and RF
ALL_LIGHT_FEATURES = [
    # Demographic and lifestyle
    "age", 
    "sex", 
    "education", 
    "BMI",
    "smoking_status", 
    "coffee_status", 
    "AUDIT",
    "dementia_history_parents", 
    "hypertension", 
    "diabetes",
    "thyroid_diseases", 
    "allergies", 
    # "learning_deficits",  # Excluded because dataset uses mixed encoding that would need special parsing
    # and it's not central to project scope.
    "EHI",
    # Genetic 
    "PICALM_G_count",
    # Psychometric 
    "BDI", 
    "SES", 
    "RPM",
    "NEO_NEU", 
    "NEO_EXT", 
    "NEO_OPE", 
    "NEO_AGR", 
    "NEO_CON",
    # Psychometric: MINI-COPE 
    "MINI-COPE_1", 
    "MINI-COPE_2", 
    "MINI-COPE_3", 
    "MINI-COPE_4",
    "MINI-COPE_5", 
    "MINI-COPE_6", 
    "MINI-COPE_7", 
    "MINI-COPE_8",
    "MINI-COPE_9", 
    "MINI-COPE_10", 
    "MINI-COPE_11", 
    "MINI-COPE_12",
    "MINI-COPE_13", 
    "MINI-COPE_14",
    # Psychometric: CVLT (all 13 subscores and total_learning)
    "CVLT_1", 
    "CVLT_2", 
    "CVLT_3", 
    "CVLT_4", 
    "CVLT_5",
    "CVLT_6", 
    "CVLT_7", 
    "CVLT_8", 
    "CVLT_9",
    "CVLT_10", 
    "CVLT_11", 
    "CVLT_12", 
    "CVLT_13",
    "CVLT_total_learning"
]

# This is for all features (full). It's everything above as well as the full 28 feature blood panel
ALL_FULL_BLOOD_ADDITIONS = [
    "leukocytes", 
    "erythrocytes", 
    "hemoglobin", 
    "hematocrit",
    "MCV", 
    "MCH", 
    "MCHC", 
    "RDW-CV",
    "platelets", 
    "PDW", 
    "MPV", 
    "P-LCR",
    "neutrophils", 
    "lymphocytes", 
    "monocytes",
    "eosinophils", 
    "basophils",
    "neutrophils_%", 
    "lymphocytes_%", 
    "monocytes_%",
    "eosinophils_%", 
    "basophils_%",
    "total_cholesterol", 
    "cholesterol_HDL",
    "non-HDL_cholesterol", 
    "LDL_cholesterol", 
    "triglycerides",
    "HSV_r"
]

# Columns where misisngness is structural(the measurement wasn't taken). These should not be imputed
NEVER_IMPUTE = set(COMPACT_FULL_BLOOD_ADDITIONS + ALL_FULL_BLOOD_ADDITIONS)

# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────

def load_participants():
    """
    Load participants.tsv from the path in config. Also applies some data-quality fixes, reasoning
    determined earlier.
    """
    cfg = load_config()
    path = Path(cfg["paths"]["data_root"])/"participants.tsv"
    df = pd.read_csv(path, sep="\t")
 
    # Strip whitespace from genotype columns. This is a known data-quality problem. Some entries have trailing 
    # spaces that create needless categories in cross-tabs
    df["APOE_haplotype"] = df["APOE_haplotype"].astype(str).str.strip()
    df["PICALM_rs3851179"] = df["PICALM_rs3851179"].astype(str).str.strip()
 
    return df

# ──────────────────────────────────────────────────────────────────────
# Filter to modelling cohort
# ──────────────────────────────────────────────────────────────────────

def filter_to_modelling_cohort(df):
    """
    Restrict to the n=79 participants who took part in the second phase.
    """
    return df[df["second_phase"] == 1].copy()

# ──────────────────────────────────────────────────────────────────────
# Derive the target and engineer necessary features
# ──────────────────────────────────────────────────────────────────────

def derive_apoe_e4_carrier(df):
    """
    Create the binary target. 
    
    1 if genotype contains a '4' and 0 otherwise.
    """
    return df["APOE_haplotype"].str.lower().str.contains("4").astype(int)
 
 
def derive_picalm_g_count(df):
    """
    Ordinal encoding of PICALM rs3851179.
     
    Done by the number of G alleles (0, 1, or 2).
    """
    return df["PICALM_rs3851179"].str.upper().str.count("G")
 
 
def derive_cvlt_total_learning(df):
    """
    Find the standard CVLT total-learning score.
    
    This is done using a sum of trials 1 through 5
    """
    trial_columns = ["CVLT_1", "CVLT_2", "CVLT_3", "CVLT_4", "CVLT_5"]
    return df[trial_columns].sum(axis=1)
 
 
def add_derived_features(df):
    """
    Add all the derived features expected by the baselines to the dataframe.
    """
    df = df.copy()
    df["PICALM_G_count"] = derive_picalm_g_count(df)
    df["CVLT_total_learning"] = derive_cvlt_total_learning(df)
    return df

# ──────────────────────────────────────────────────────────────────────
# Build a feature matrix for a selected variant
# ──────────────────────────────────────────────────────────────────────

def build_feature_matrix(df, variant):
    """
    Create the feature matrix for the chosen variant.
 
    Parameters
    ----------
        - df : DataFrame with derived features already added (see functions above)
        - variant : 'compact-light', 'compact-full', 'all-features-light' or 'all-features-full'
 
    Returns
    -------
        - X : DataFrame of features
    """
    if variant == "compact-light":
        columns = COMPACT_LIGHT_FEATURES
    elif variant == "compact-full":
        columns = COMPACT_LIGHT_FEATURES + COMPACT_FULL_BLOOD_ADDITIONS
    elif variant == "all-features-light":
        columns = ALL_LIGHT_FEATURES
    elif variant == "all-features-full":
        columns = ALL_LIGHT_FEATURES + ALL_FULL_BLOOD_ADDITIONS
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")
  
    return df[columns].copy()

# ──────────────────────────────────────────────────────────────────────
# Impute any missing data (done after first iteration of log reg)
# ──────────────────────────────────────────────────────────────────────

def impute_missing(X):
    """
    Fill missing values with column median for numerical features and mode for categorical features,
    except for columns in NEVER_IMPUTE (blood panel columns)

    The blood panel columns are excluded because missingness reflects participants who likely never had blood
    drawn, and fabricating biological measurements should be avoided.

    First iteration of logistic regression should've been n=79 but had 10 missing values across 3
    columns, which caused loss of 12% of the cohort. Impute as a result.
    """
    X = X.copy()
    for col in X.columns:
        if col in NEVER_IMPUTE:
            continue  # Make sure to avoid biology-fabricating imputation
        if X[col].isna().any():
            # Categorical columns (integer encoded with not many unique values) get the mode. Continuous get the median.
            if X[col].nunique() <= 10:
                fill_value = X[col].mode().iloc[0]
            else:
                fill_value = X[col].median()
            X[col] = X[col].fillna(fill_value)
    return X

# ──────────────────────────────────────────────────────────────────────
# Drop rows with any missing data
# ──────────────────────────────────────────────────────────────────────

def drop_missing(X, y):
    """
    Drop rows where any feature is missing.
 
    For the compact-light variant this should drop few or no rows because features chosen to be always available.
    For the compact-full variant this drops the 3 participants that don't have blood data. ngoes from 79 to 76.
    """
    complete = X.notna().all(axis=1)
    return X[complete].copy(), y[complete].copy()

# ──────────────────────────────────────────────────────────────────────
# Standardise numeric features for linear models
# ──────────────────────────────────────────────────────────────────────
def standardise(X):
    """
    Z-score standardise every column (mean 0, std 1).
 
    Needed for logistic regression to produce comparable coefficients across features that have different scales.

    Standardise on the whole X here. Simpler for a small baseline. If more rigour was needed, this would be done 
    inside each CV fold using training-set statistics only (with an sklearn Pipeline). This would be worth doing 
    if baselines were to be productionised. For this baseline it's an approximation.
    """
    return (X-X.mean())/X.std()

# ──────────────────────────────────────────────────────────────────────
# Combined fucntion
# ──────────────────────────────────────────────────────────────────────
def load_baseline_data(variant="compact-light"):
    """
    Load, filter, engineer and standardise.
 
    Parameters
    ----------
        - variant : 'compact-light' or 'compact-full'
 
    Returns
    -------
        - X : numpy array of shape (n_samples, n_features)
        - y : numpy array of shape (n_samples,). This is the binary target
        - feature_names : list of str, in the same order as X columns
    """
    df = load_participants()
    df = filter_to_modelling_cohort(df)
    df = add_derived_features(df)

    y = derive_apoe_e4_carrier(df)
    X = build_feature_matrix(df, variant)

    X = impute_missing(X)         
    X, y = drop_missing(X, y)     
    X = standardise(X)

    feature_names = list(X.columns)
    return X.to_numpy(), y.to_numpy(), feature_names