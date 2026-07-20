"""
This is a PICALM-APOE independence check. Want to check if PICALM_G_count and APOE e4 carrier status are independent 
in the n=79 modelling cohort.

Tabular model's feature impotance analysis showed PICALM_G_count was the dominant predictor of APOE e4 status across 
all 3 baseline models that were checked.  These genes should be on different chromosomes and inherit independently
in the general population so need to check for correlation in the sample.

Prints a cross tab of PICALM_G_count against APOE e4 carrier status then a Spearman rank correlation.
"""

from agnn.baselines.data_processing import load_participants, filter_to_modelling_cohort, add_derived_features, derive_apoe_e4_carrier
import pandas as pd
from scipy.stats import spearmanr

df = load_participants()
df = filter_to_modelling_cohort(df)
df = add_derived_features(df)
df["e4_carrier"] = derive_apoe_e4_carrier(df)

# Cross-tab PICALM against carrier status
ct = pd.crosstab(df["PICALM_G_count"], df["e4_carrier"], margins=True)
print("PICALM_G_count vs APOE e4 carrier:")
print(ct)

# Correlation
r, p = spearmanr(df["PICALM_G_count"], df["e4_carrier"])
print(f"\nSpearman correlation: r={r:.3f}, p={p:.3f}")