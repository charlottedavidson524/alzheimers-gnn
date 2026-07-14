Decision on how to encode the 28 blood panel and biomarker features for the tabular baseline models.

The blood panel in `participants.tsv` provides 28 features across 3 groupings.

- Full blood count (CBC): 22 features covering red cells, white cells and platelets.
- Lipid panel: total, HDL, non-HDL, LDL cholestrol and triglycerides.
- Infection marker: 1 feature, HSV_r.

The correlation heatmaps (`docs/findings/extended-eda.md`) identified two redundancies.

- White-cell absolute and percentage counts carry the same information.
- The lipid panel has arithmetic redundancy. Non-HDL cholesterol = total cholesterol − HDL cholesterol.

Blood data is only present for the 76 second-phase participants. The baseline variant that uses these features therefore operates on n=76, not n=79.

Options considered:

- Include everything raw. Could lead to multicollinearity.
- Drop absolute counts, keep percentages (standard clinical convention).
- Drop percentages, keep absolute counts.
- PCA of the CBC block. Low interpretability.
- Compute derived indices (NLR, LDL:HDL ratio) as additional features alongside raw values.

Decision:

Feature selected baseline:

Include the following blood markers:

- Red cell block: leukocytes, erythrocytes, hemoglobin and hematocrit.

- Platelet block: platelets.

- White cell percentages: neutrophils*%, lymphocytes*%, monocytes*%, eosinophils*%, basophils\_%.

- Lipid block: total_cholesterol, HDL_cholesterol, and triglycerides. Non-HDL and LDL cholesterol left out because they're redundant with total and HDL.

- Infection: HSV_r.

The logic behind the exclusions:

- White-cell absolute counts (neutrophils, lymphocytes, monocytes, eosinophils and basophils). They are correlated with percentages.

- MCV, MCH, MCHC and RDW-CV. These are red-cell subtype indices, correlated with each other and with red-cell counts. Can be used for aneamia diagnosis but likely not linked to APOE e4 classification.

- PDW, MPV and P-LCR. These are platelt size measures, all highly correlated with each other. Can just include platelets to represent this info.

- non-HDL cholestrol is redundant with total - HDL.

- LDL_cholestrol: highly correlated with non-HDL and total. Keep these as they're more standard (find sources to support), as well as trigylcerides as an extra dimension.

LASSO baseline will include all 28 features. So will random forest/GNN, same logic as other feature selection files.

Derived features were considered but ultimately won't include them. Two indices that can be linked to alzheimers/inflammation (find sources).

- NLR (neutrophil-to-lymphocyte ratio) = neutrophils/lymphocytes
- LDL:HDL ratio = LDL/HDL

The raw values already capture this information. The interaction can be found using tree-based models without any engineered features, and added complexity should only be brought in if the initial baseline result justifies it. Reintroduce if the raw features underperform.

For reporting:

- Tabular-full baseline sample size n=76 (not 79). Needs to be reported along with the tabular light (n=79) baseline in every comparison.

- Compact baseline uses 12 blood features while LASSO uses 28. Any differences in performance quantify whether raw redundant features add signal beyond hand-selected representatives.

- Feature importances from random forest will indicate which blood markers actually contribute. Will determine if exclusions were needed.

Come back to this if:

- Random forest highlights any of the excluded features as being important.

- LASSO selects non-HDL_cholesterol or LDL_cholesterol consistently over total or HDL.

- Adding NLR or LDL:HDL as derived features improves performance in a significant way.
