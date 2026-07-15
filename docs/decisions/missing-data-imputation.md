Needed to make a decision on how to handle missing values in the feature-selected and all-feature tabular baseline models.

The extended EDA identified small amounts of missingness in the N=79 modelling cohort. Running the initial baseline (compact-light logistic regression) dropped the sample size down to n=69 which is roughly 13% data loss.

A test (`tests/test_missingness.py`) traced the drops to three features in the compact-light set:

| Feature        | Missing (n=79) |
| -------------- | -------------- |
| education      | 8              |
| CVLT_13        | 1              |
| smoking_status | 1              |

When combined, 10 participants have at least one missing value across these features and would be dropped under complete case analysis. At n=79 losing 10 participants is a big loss. Would also create different effective sample sizes across baseline variants (they would use different feature subsets and drop different rows) which would make direct comparison across models very difficult.

There is also some structural missingness. It affects the blood-panel variants. 3 out of 79 second-phase participants are missing blood values.

Options considered:

- Complete case analysis -> drop any row with a missing feature. Easy but loses a big percentage of cohort.

- Median/mode imputation -> fill missing values with median (continuous) or mode (categorical). Keeps n=79 for light variants and n=76 for full variants.,

- Model based imputation -> like iterative imputer. More sophisticated but adds assumptions.

- Remove problematoc features

- Separate models for participants with different missingness patterns. Complex and inappropriate at this sample size.

Decision:

- Decided to do median/mode imputation. Continuous (>10) -> median. Categorical/ordinal (<= 10) -> mode. Categorical features in `participants.tsv` are integer encoded already. 10 chosen as a boundary to distinguish categorical from continuous. Just a heuristic.

For the feature-selected light (n=79) variant, imputation affected:

- education (8 missing -> mode imputed)
- CVLT_13 (1 missing -> median imputed)
- smoking_status (1 missing -> mode imputed)

n stays at 79 this way.

Blood-panel missingness (3 participants) is structural. They likely did not have blood drawn. Imputation would be fabriacting biological measurements. Drop these rows.

Implementation:

The imputation was applied in `src/agnn/baselines/data_processing.py` in the impute_missing() function, immediately before the drop_missing() function. This means:

- Feature matrix is built for the requested variant
- Small-scale missingness (education, CVLT_13, smoking_status) is imputed
- Any remaining missingness (blood panel for 3 participants) is dropped via complete-case filtering
- Standardisation is applied to the imputed matrix

Imputation is computed on the whole modelling cohort, not inside the CV folds. It could be more rigourous to impute using training-fold stats only. At n=79 with 10 missing values across 3 columns, the impact is very small. But if scaling to more rigorous production, model would need to be wrapped in a `sklearn.pipeline.Pipeline` with
`SimpleImputer` and `StandardScaler` inside each fold.

Impact (before and after imputation):

Comparing initial drop to n=69 to n=79 after imputation:

| Metric            | n=69            | n=79            |
| ----------------- | --------------- | --------------- |
| F1                | 0.780 +/- 0.055 | 0.740 +/- 0.073 |
| Balanced accuracy | 0.666 +/- 0.113 | 0.672 +/- 0.145 |
| ROC-AUC           | 0.755 +/- 0.061 | 0.744 +/- 0.138 |

Metrics changed by <0.05 (within CI width). Standard deviations got slightly wider, reflecting increased
per-fold variance from the imputed rows. It's not result distortion. Imputation is filling small gaps without massively affecting behaviour.

What wasn't chosen:

- Complete case analysis: too much data loss
- Model-based imputation (iterative imputer): overkill
- Feature exclusion: `education` is a documented cognitive-reserve variable, shouldn't drop it

Revisit if:

- Cross-fold variance for the compact-light variant stays large (>0.15) across all six baseline variants. Might indicate imputation is destabilising specific folds. Check if imputed participants cluster in one fold.

- Later baselines show big gaps in performance between compact-light (N=79, imputed) and compact-full (N=76, no imputation). Suggests imputation is doing more than thought.

- Final GNN pipeline needs comparable tabular features for fusion. Imputation should be re-considered inside pipeline-wrapped CV folds for rigour.
