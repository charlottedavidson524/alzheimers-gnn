Following on from `docs/findings/tabular_baselines.md`. Feature importance of the tabular baselines was performed in `scripts/analyse_baselines.md`.

All three features found PICALM_G_count to be the most predictive feature:

- Logistic regression coefficient: +1.48 (nearly 2x the next dominant feature)
- Random forest importance: 0.079 (more than 2x the next dominant feature)
- LASSO: chosen in 5/5 CV folds. Mean coefficient +1.78

This showing up across all models is surprising. PICALM and APOE should be independent in the general popul;ation. Did a diagnostic in `tests/test_picalm_apoe.py`. The association is real in this particular cohort.

| Picalm G count | Non-carrier | Carrier | Carrier rate |
| -------------- | ----------- | ------- | ------------ |
| 0 (A/A)        | 7           | 4       | 36%          |
| 1 (G/A)        | 24          | 22      | 48%          |
| 2 (G/G)        | 0           | 22      | 100%         |

Spearman's r = 0.477, p < 0.001.

Every G/G homozygote is an APOE e4carrier. The conecntration is likely due to PEARL-Neuro's deliberately recruiting participant's with a higher rate of dementia. This has implications for generalisability. Will do a sensitivity analysis excluding PICALM_G_count from feature selected full LR baseline to check "unconfounded" tabular performance.

Other findings:

- Affective signiture has been confirmed but isn't that dominant. BDI (LR coefficient +0.88), NEO_NEU, and correlated personality features are chosen across models. Real influence but it's quite modest. Mid-ranked rather than near the top. Actually consistent with extended EDA. the affective/personality cluster nexists at group-mean level but doen't carry individual-level prediction as much as PICALM.

- Inflammatory signal is strong. Eosinophils (absolute and %) rank highly in all three models (LR coefficient +0.95 for eosinophils\_%, RF importance 0.039, LASSO 5/5 folds). Monocytes and the neutrophil–lymphocyte balance also have consistemnt signal. EDA's flag of the eosinophil effect is supported by the multivariate models.

- Triglycerides are a strong positive predictor (LR +0.73, LASSO 5/5 folds). However, Xu et al found no triglyceride difference by e4 status. Total cholesterol has a negative coefficient in LR (-0.37) which is the opposite of what Xu states.

- Of the 78 features available to LASSO, only 11 are selected in all 5 CV folds. Another 11 are in 4/5 folds. 29 are never selected. LASSO is relying on around 15 features and treating the remaining 60 as noise. This is a strong argument that data-driven feature selection at n=79 with 78 candidate features is inherently unstable. Supports earlier finding that hand-selected compact features outperform regularised automatic selection.
