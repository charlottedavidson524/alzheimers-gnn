This file will document the decisions made on how to handle seven intercorrelated affective and personality features (BDI, SES, NEO_NEU, NEO_EXT, NEO_CON, NEO_AGR, MINI-COPE_13, MINI-COPE_14) within the tabular baseline models.

Found within the extended EDA (`docs/findings/extended-eda.md`) that:

- A clear personality/affective signiture is present in the modelling cohort (score higher on BDI (depression), NEO_NEU (neuroticism), and MINI-COPE items 13/14 (interpreted as maladaptive coping but need to confirm this). score lower on SES (self-esteem), NEO_CON (conscientiousness), NEO_AGR (agreeableness), and NEO_EXT (extraversion))

- The features form a clear correlation cluster in the n=79 cohort psychiatric heatmap.

- The effect sizes are moderate as no single feature survives FDR correction at N=79 with 78 tests.

Can't include all 7 raw features in a linear model beacuse of potential collinearity, unstable coefficients, and inflated variance. Tree-based and regularised models will be fine with this but logistic regression won't be.

Options:

- BDI alone as affective proxy (largest sample size in the cluster, most clincially well established. Simple and interpretable)

- PCA composite (extract first principle components of the seven features and use that single score). Captures shared variance efficiently but results in a loss of interpretability.

- All raw features with L1 regularisation (LASSO). regularisation will select which features contribute. Note that different features could be selected in different CV folds which would make reporting more complicated.

- All raw features with L2 regularisation (ridge). Handles collinearity by shrinking instead of selecting. Keeps all features. Each coefficient is small and hard to interpret individually.

- Hand select a subset, eg BDI + NEO_NEU + NEO_CON by picking the three most representative. Interpretable but also arbitrary,

Decision:

Two variants. Fewer features logistic regression (just BDI). Reasoning: BDI has largest affect size in the cluster (d=0.48). It's clinically well established, easier to interpret than a composite. Lasso logistic regression (all seven features). Reasoning: Regularisation handles collinearity gracefully. Data driven feature selection. Directly comparable to compact baseline (same underlying data but different feature selection). which features get chosen across CV folds is reportable. Random forest and GNN can handle collinearity so can use all features.

Reasons for rejections:

- PCA composit rejected, because it loses interpretability that motivates tabular models in the first place.
- Hand selected three feature subset was too arbitrary.

Reporting implications:

- Feature selected baseline will attribute predictive contribution to just BDI alone within the affective/personality space.

- LASSO results will report which personality features had non-zero coefficients across CV folds. Could be a
  finding in its own right, e.g. if some are consistently selected.

- Differences between the two performances quantify whether the extra features beyond BDI have any predictive power or if they are redundant.

TO DO:

- CHECK WHAT MINI_COPE-13 AND MINI_COPE-14 CORRESPOND TO.

Revisit this document if:

- MINI_COPE mapping is confimed and 13 and 14 don't meaure maladaptive coping.

- LASSO's cross-fold feature selection is unsteady enough that a subset chosen by hand would make more sense.

- The two varianbt's performances diverge enough that the interpretation changes.
