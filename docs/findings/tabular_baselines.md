These are the results from the six tabular baseline variants tarined on the n=79 modelling cohort (or n=76 for blood inclusive variants). There are three model types and two feature set variants. They're all evaluated on the same stratified 5-fold corss-validation and metric aggregation .

There are two purposes served by these baseline models:

- Establish the predictive ceiling for tabular data alone. GNN will be compared to this.
- Testing the feature selection decisions to see if they were the right ones.

|                             | tabular-light (N=79)                                                | tabular-full (N=76)                                          |
| --------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------ |
| Logistic (compact features) | Chosen features, no regularisation tuning                           | Compact features + 14-marker blood panel                     |
| Logistic (LASSO)            | All features with L1 penalty (feature selection via regularisation) | All features and full 28-marker blood panel. Uses L1 penalty |
| Random forest               | All features. Tree-based and handles collinearity                   | All features and full 28-marker blood panel                  |

The feature set choices can be found in:

- `docs/decisions/feature-selection/blood-panel.md`
- `docs/decisions/feature-selection/cvlt.md`
- `docs/decisions/feature-selection/personality-cluster.md`
- `docs/decisions/missing-data-imputation.md`

The cross validation set up is stratified 5-fold, seed 42, bootstrap 95% CI over fold results.

Results (ranked in order of ROC-AUC):

| Baseline                | N   | Features | ROC-AUC (95% CI)     | F1 (95% CI)          | Balanced accuracy (95% CI) |
| ----------------------- | --- | -------- | -------------------- | -------------------- | -------------------------- |
| Logistic compact-full   | 76  | 25       | 0.830 [0.781, 0.880] | 0.818 [0.766, 0.871] | 0.763 [0.689, 0.835]       |
| Logistic compact-light  | 79  | 11       | 0.744 [0.640, 0.850] | 0.740 [0.679, 0.801] | 0.672 [0.545, 0.775]       |
| LASSO all-full          | 76  | 78       | 0.735 [0.642, 0.841] | 0.728 [0.675, 0.783] | 0.679 [0.614, 0.752]       |
| Random forest all-light | 79  | 50       | 0.647 [0.507, 0.802] | 0.717 [0.608, 0.813] | 0.575 [0.431, 0.733]       |
| LASSO all-light         | 79  | 50       | 0.645 [0.505, 0.810] | 0.666 [0.549, 0.794] | 0.595 [0.453, 0.753]       |
| Random forest all-full  | 76  | 78       | 0.624 [0.509, 0.761] | 0.684 [0.592, 0.781] | 0.512 [0.395, 0.650]       |

The main result is that feature-selected logistic regression (n=76 with blood markers) achieves ROC-AUC = 0.830. This sets the tabular baseline that the GNN comparison needs to beat to show that EEG-derived features carry information beyond just hand chosen tabular biomarkers.

There are three important findings that come from these tabular results:

Hand selected features beat data driven selection. Feature selected logistic regression outperformed LASSO on the same underlying data, in both feature set variants:

| Feature set       | Feature-selected LR | LASSO | Difference |
| ----------------- | ------------------- | ----- | ---------- |
| Light (no blood)  | 0.744               | 0.645 | +0.099     |
| Full (with blood) | 0.830               | 0.735 | +0.095     |

Feature selected logistic regression wins by roughly 0.1 ROC-AUC on both feature sets.

At n=79 with informative prior knowledge about the biology, hand-selected features enocde domain expertise that data-driven feature selection can't get from 50 to 78 raw features. There are two potential reasons for this:

- Small sample regularisation is unstable
- The biological knowledge used to create the feature set was genuinely useful.

So, data-driven feature selection is not automatically superior to choosing features by hand when the sample size is small.

Blood markers add a lot of predictive value:

Comparing the light vs fullwithin each model type:

| Model Type                  | Light ROC-AUC | Full ROC-AUC | Difference (from adding blood) |
| --------------------------- | ------------- | ------------ | ------------------------------ |
| Logitsic (feature selected) | 0.744         | 0.830        | +0.086                         |
| LASSO                       | 0.645         | 0.735        | +0.090                         |
| Random Forest               | 0.647         | 0.624        | -0.023                         |

The blood markers add an improvement of around 0.09 ROC-AUC . This is a big jump. Outside of 95% CI overlapfor compact-light variant. Blood carries APOE e4 signal that psychometric and demographic features can't capture on their own.

Could be because of lipid panel. Recall Xu et al. (2023) demonstrated APOE e4 carriers show elevated total cholesterol and LDL even in healthy controls. Back with other literature.

Blood markers carry predictive info that purely EEG-based GNN can't access directly. Validates the multimodal fusion framing because models combining both modalities might outperform either singular model.

Non-linearity isn't useful in this case (see random forest underperforming).

Random forest is beaten by logistic regression on both sets of features

| Feature set | Feature selected LR | Random Forest | Difference |
| ----------- | ------------------- | ------------- | ---------- |
| Light       | 0.744               | 0.647         | LR +0.097  |
| Full        | 0.830               | 0.624         | LR +0.206  |

Full featured random forest is quite weak. With 78 features on n=76, RF is probably overfitting. the 500 tress are proabbly memorising parts if the training data. Fold level variance is high (per-fold ROC-AUC ranges between 0.426 to 0.852 for RF all-full).

This is probably because of two things:

- Linear signal
- Overfitting at small N. Could portntially be avoided with bigger N/larger sample.

The signal for the tabular data is well captured by linear models. So, the GNN's value isn't non-linearity because RF couldbe provided it, but instead the spatial strcuture of brain connectivity. Also if using intermediate fusion, feeding the GNN's graph embedding into a final linear classifier would be suitable

Something notable: the fourth fold is consistently hard for high dimensional models

| Baseline                | Fold 4 ROC-AUC |
| ----------------------- | -------------- |
| Logistic compact-light  | 0.722          |
| Logistic compact-full   | 0.833          |
| LASSO all-light         | 0.407          |
| LASSO all-full          | 0.648          |
| Random forest all-light | 0.389          |
| Random forest all-full  | 0.426          |

When feature counts are high (50+) fold 4 collapses. Feature selected baselines handle it with no issue (0.722 and 0.833) but higher dimensional baselines don't. Consistent with high-dimensional overfitting. LASSO and RF latch onto features that appear predictive in the other 4 folds but generalise badly to fold 4.

Threshold vs ranking behaviour. For the winning compact-full LR, F1 (0.818) and balanced accuracy (0.763) are a good bit lower than ROC-AUC (0.830). So, the model's ranking is very good, it correctly orders participants by probability of being a carrier, but the default 0.5 threshold is producing asymmetric errors. For comparative analyses, ROC-AUC is the primary metric. F1 and balanced accuracy carry threshold sensitivity that can be optimised separately.

Limitations (for discussion section):

- Standardisation is applied to the whole X, not inside CV folds. Small statistical inefiicnecy at n=79. Negligible impact. Correct in a production grade pipeline.
- Hyperparameters aren't tuned. Revisit this if final results motivate it.
- Imputation is performed outside of CV folds.

Implications for comparing with GNN -> tabular baselines set the following bar:

- Best tabular ROC-AUC = 0.830 (feature selected full logistic).
- Best tabular light ROC-AUC = 0.744 (feature selected light logistic).

For GNN to be a useful contribution:

- EEG-only GNN ROC-AUC > 0.75 would suggest EEG captures info similarly to tabular biomarkers.
- EEG-only GNN ROC-AUC > 0.83 would suggest EEG captures info better than the best tabular baseline.
- Fusion model ROC-AUC > 0.85 would show the modalities are complementary.

# Update

After running feature improtance, found that PICALM_G_count was consistently highly ranked across all baseline models despite the fact that PICALM and APOE should be uncorrelated in the general public (PEARL-Neuro were searching for those at risk of dementia so correlation is likely higher than in the general public). Re-ran LR, feature selected, full model but without PICALM_G_count and got the following results:

| Baseline                                         | N   | Features | ROC-AUC (95% CI)     | F1 (95% CI)          | Balanced acc. (95% CI) |
| ------------------------------------------------ | --- | -------- | -------------------- | -------------------- | ---------------------- |
| Logistic feature-selected full (previous winner) | 76  | 25       | 0.830 [0.781, 0.880] | 0.818 [0.766, 0.871] | 0.763 [0.689, 0.835]   |
| Logistic feature-selected full, no PICALM        | 76  | 24       | 0.681 [0.622, 0.750] | 0.632 [0.578, 0.676] | 0.571 [0.515, 0.628]   |

There are honestly two baselines for ROC-AUC: 0.830 (with PICALM) (he model's full performance on
this cohort) and 0.681 (without PICALM) (unconfounded tabular signal)

Implications for GNN comparison:

| GNN ROC-AUC | Interpretation                                                                                     |
| ----------- | -------------------------------------------------------------------------------------------------- |
| >= 0.68     | GNN cpatures brain-derived signal comparable to unconfounded tabular-biomarkers                    |
| >= 0.75     | GNN clearly exceeds unconfounded tabular signal. Strong evidence EEG carries additional info       |
| >= 0.83     | GNN matches LR signal without benefiting from PICALM–APOE correlation                              |
| >= 0.85     | GNN beats even inflated tabular signal, ecidence for EEG specific info beyond any tabular baseline |

Note:

Fold 2 was weakest fold in the no-PICALM baseline (ROC-AUC 0.593). Same pattern found in earlier baselines.
Could add to discussion of the results. Could reflect specific participants in the fold-2 test set who are harder to classify from tabular features alone. Whether this pattern persists in th GNN and fusion results remains to be seen.
