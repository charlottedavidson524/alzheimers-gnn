These are the results from running `scripts/explore_participants_further.py` on both the full n=192 sample of all participants as well as the `second_phase` n=79 modelling cohort.

This documentation covers:

- The resolution of the investigation into the second_phase cohort
- The modelling cohort class balance.
- Univariate featurte screening across both of the samples.
- Correlation structure across both samples.
- Confounder analysis across both samples.

This builds upon earlier findings in `docs/findings/genetic-profile.md`, `docs/findings/demographic-distributions.md` and `docs/findings/histogram-interpretations.md`.

The question about second_phase now has an answer. The full-sample cross tabulation of second_phase against blood data availability was definitive.

| second_phase | has blood | no blood | Total |
| ------------ | --------- | -------- | ----- |
| 0            | 0         | 113      | 113   |
| 1            | 76        | 3        | 79    |
| Total        | 76        | 116      | 192   |

second_phase = 1 is the extended protocol cohort who went through the blood tests (and, according to the paper, the neuroimaging tests). Three second_phase = 1 participants have missing blood values, but this can be put down to an error in collection rather than any structural missingness.

The sample size (79) is an exact match to the neuroimaging subset the paper states. Combine this with the study's description of the second phase protocol and this shows second_phase = 1 is the n=79 neuroimaging cohort. This will be confirmed definitively during Stage 3 download (when EEG files are on disk), but is pretty much certain. Can conclude with an answer to the question raised in `docs/findings/demographic-distributions.md`.

The modelling cohort class balance was also checked. 48 out of 79 participants within `second_phase = 1` are APOE e4 carriers. Non carriers are the minority (31, 39.2%). This is different from the full sample rate (see table below).

| Sample                       | N   | e4 carriers | Rate  |
| ---------------------------- | --- | ----------- | ----- |
| Polish population baseline   | -   | -           | ~20%  |
| PEARL-Neuro full sample      | 192 | 51          | 26.6% |
| PEARL-Neuro modelling cohort | 79  | 48          | 60.8% |

The modelling cohort is around 3x the population carrier rate. This is likely a reflection of the deliberate recuruitment for neuroimaging. Consistent with the study's framing of a middle aged, at risk cohort but maybe higher than expected.

Implications for modelling:

- Class balance is inverted. Minority class is non-carriers, not carriers.
- Imbalance is not too extreme. 60/40. Stratified k-fold CV can likely handle cleanly. Class-weighted loss might not be necessary.
- F1 and ROC-AUC/alternative metrics other than raw accuracy are still the preference. Raw accuracy is not ass bad as it would have been at 27/73 but still less appropriate than F1 for example.
- Any claims about generalisability need to reflect the enrichment. any estimate about performcance is being made with a higher-risk cohort. The screening isn't general population. Should highlight thisd in methodology and discussion.

Implication for multimodal modelling:

- Multimodal fusion documented in `docs/decisions/multimodal-fusion.md` is still sound. The risk-enriched sample must be acknowledged. Question must reflect this, e.g. "in a risk enriched sample, can combining EEG with tabular biomarkers distinguish carriers from non-carriers?"

Feature screening. Comparing full sample and modelling cohort.

The two rankings are quite different. For the full sample there are no biological variables that particularly stand out. The top signals are all modest inflammatory markers with FDR-corrected p-values above 0.7. For the modelling cohort there are psychometric patterns.

n=192 top signals

| Variable                  | Cohen's d | FDR p |
| ------------------------- | --------- | ----- |
| eosinophils\_%            | 0.51      | 0.71  |
| eosinophils               | 0.48      | 0.71  |
| MCH                       | 0.37      | 0.99  |
| MCHC                      | 0.36      | 0.99  |
| smoking_status            | 0.34      | 0.71  |
| MINI-COPE_8               | -0.34     | 0.71  |
| dementia_history_patients | 0.34      | 0.71  |
| EHI (handedness)          | 0.31      | 0.71  |

n=79 top signals

| Variable                    | Cohen's d | Raw p | FDR p |
| --------------------------- | --------- | ----- | ----- |
| eosinophils\_%              | 0.51      | 0.016 | 0.73  |
| eosinophils                 | 0.48      | 0.024 | 0.73  |
| BDI (depression)            | 0.48      | 0.030 | 0.73  |
| NEO_NEU (neuroticism)       | 0.45      | 0.038 | 0.73  |
| MINI-COPE_14                | 0.44      | 0.069 | 0.73  |
| MINI-COPE_1                 | -0.40     | 0.093 | 0.73  |
| CVLT_9                      | 0.40      | 0.070 | 0.73  |
| dementia_history_parents    | 0.39      | 0.078 | 0.73  |
| SES (stress)                | -0.39     | 0.093 | 0.73  |
| MINI-COPE_13                | 0.39      | 0.086 | 0.73  |
| MCH                         | 0.37      | 0.125 | 0.73  |
| NEO_CON (conscientiousness) | -0.37     | 0.111 | 0.73  |
| MCHC                        | 0.36      | 0.159 | 0.73  |
| NEO_EXT (extraversion)      | -0.33     | 0.155 | 0.73  |
| RPM (intelligence)          | -0.32     | 0.126 | 0.73  |
| NEO_AGR (agreeableness)     | -0.29     | 0.200 | 0.74  |

Nothing survives FDR correction in either of the analyses. Not too surprising given there are either a sample of only 78 tests or a presymptomatic sample. There is a pattern in the n=79 cohort that should be reported on though. Carriers in the n=79 cohort show:

- Higher depression (BDI)
- Higher neuroticism (NEO_NEU)
- Lower self-esteem (SES)
- Lower concientiousness (NEO_CON)
- Lower agreeableness (NEO_AGR)
- Lower extraversion (NEO_EXT)
- Lower intelligence score (RPM)

This is a somewhat clear psychometric profile. Check if there is literature associating APOE e4 with behavioural/personality differences in pre-symptomatic for AD propulations. Important notes on the findings above for the dissertation:

- This is an exploratory finding only
- The interpretation is important (no FDR-corrected variable survives; 78 tests; N=79)
- Need to cite relevant literature when making this point

The eosinophil count and percentage are the top biological feature in both runs.

- Check the AD/inflammation literature

Expected there to be lipid signals because of Xu et al. (2023) but this was absent. total_cholestrol and LDL_cholestrol don't appear in the top 20 of either group. Possible explanations could be|:

- Power vs sample size. With n=48 vs n=31 in the modelling cohort, detecting an effect of d = 0.25 needs more power than we have (an 80%-power test at alpha = 0.05 for d = 0.25 needs roughly n = 250 per group).
- The paper is reporting allele-count effects, not carrier-effects.
- The nature of this specific population sample if that of a risk-enhanced Polish pre-symptomatic subgroup. This might not be reflected in a broader-population meta analysis like the one Xu et al. did.

Be sure to acknowledge this surprising result.Also, the GNN/multimodal approach may find a lipid signal that the univariate screening doesn't pick up on.

Correlation structure:

Correlation heatmaps for both the blood tests and the psychometric tests were generated for both the n=192 and n=79 groups. The PNGs can be found in `results/extended_exploration/` and `results/extended_exploration_second_phase/`.

The correlation structure is very similar across both runs. Blood data is only available for `second_phase = 1` so both heatmaps are essentially describing the same people.

Structure visible in both:

- Red cell block: leukocytes, erythrocytes, hemoglobin and hematocrit are mutually correlated (the total cell counts covary)
- Red cell size/content block: MCV, MCH, MCHC form a tight cluster (they all measure red-cell haemoglobin properties)
- Platelet size block: PDW, MPV, P-LCR form a very strong correlated triangle (all measure platelet size/large-cell fraction).
- Lipid panel block: total_cholestrol, non-HDL_cholestrol and LDL_cholestrol form a stromg redundant block.
- White cell counts: absolute vs percentage show near-perfect diagonal correlations for the same cell types.

Psychometric variables:

The full-sample psychometric heatmap shows a strong CVLT block and some MINI-COPE structure but not much else. In the modelling cohort heatmap there are a few new patterns.

Affective/personality cluster:

A coherent block appears in the top-left corner spanning BDI, SES, NEO_NEU, and MINI-COPE_13 and MINI-COPE_14. Notable correlations include:

- BDI and NEO_NEU: strong positive. Depression and neuroticism covary.
- SES and NEO_EXT: strongly negative.
- SES and NEO_CON: negative
- MINI-COPE_14 and NEO_NEU: strongly positive
- MINI-COPE_14 and SES: strongly negative
- MINI-COPE_13 and BDI, NEO_NEU: positive
- MINI-COPE_13 and NEO_CON, NEO_AGR: negative

MINI-COPE_13 and MINI-COPE_14 measure X and Y. Their alignment with the depression/neuroticism/stress cluster is consistent in theory. High-negative-affect people might lean on maladaptive coping strategies. The visible cluster aligns with what was found in the feature screening. It is it's own correlation block rather than just group-mean differences.

Implications for baseline modelling. There is a lot of redundant information in the affective/personality features. When creating the linear tabular baseline, using all seven would produce multicollinearity. Some options are:

- Use a principle component of the affective cluster as a single feature.
- Use only the strongest factors (BDI and NEO_NEU) and drop the other correlated features.
- Use all of the raw features while using a regularised model like ridge or lasso that handles collinearity.
