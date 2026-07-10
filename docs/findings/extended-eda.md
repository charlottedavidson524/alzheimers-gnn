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

Feature screening:

The ranmings of features between the two groups differ significantly.
