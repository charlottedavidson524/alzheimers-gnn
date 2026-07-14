Documentation for decision on how to summarise the 13 California Verbal Learning Test subscores (CVLT_1 to CVLT_13) for tabular baseline models.

The CVLT provides 13 numbered subscores in the participant table. From extended EDA (`docs/findings/extended-eda.md`) and `participants.json` verification:

- CVLT_1-8 form a strong correlation block. These are the immediate recall/short-delay recall subscores across learning trials.

- CVLT_9 is distinct. It's the inference-list subscore.

- CVLT_10-12 form a weaker delayed-recall block.

- CVLT_13 measures false recognition alarms. The count of non-list words the participant incorrectly identifies as being on the list. Higher score means more memory errors. Negatively correlates with CVLT_1–8 as expected.

Including all 13 subscores would produce collinearity between the immediate recall block. This adds noise. However, CVLT is one of the most AD-relevant psychometric measures in the dataset. It's subscores carry such predictive value that they should not be discarded.

Options considered:

- Use all 13 raw subscores. Maximum info but adds collinearity.

- Sum-based summaries. Use the total learning score (sum of trials 1-5) as the measure of how much a participant learned.

- Single-subscore summaries. Use either CVLT_5 as a peak learning measure, or CVLT_1 as an initial encoding measure.

- PCA of the CVLT block. Use the first 2-3 principal components. Not as interpretable.

Decision:

The feature selected baseline will use two created features.

- Total learning score: Add together CVLT 1 to 5. Sum of correctly recalled words across the five immediate recall trials. Well established and captures overall encoding capacity.

- Recognition errors: CVLT_13. This is a measure of memory errors. It is possible to have preserved recall performance but higher false alarms. This indicates a source-monitoring or discrimination deficit that could be related to AD.

- Delayed recall isn't included in the feature selected baseline. It's correlated with CVLT_1 to 8. no unique
  signal expected in a small pre-symptomatic sample, which is what the dataset is. Can reintroduce this if the compact baseline underperforms and delayed-recall contribution seems worth testing.

LASSO baseline will use all 13 raw subscores. Lets regularisation choose which contribute. Same logic as for the personality/affective cluster.

For random forest and GNN, they handle multicollinearity fine so all raw features can be included.

What isn't chosen and why:

- All 13 subscores in a compact baseline. Issues with collinearity.

- PCA composite. Rejected for interpetability reasons.

- CVLT_5 on it's own.

Reporting implications:

- Feature selcted baseline will give the coefficient contributions of CVLT_total_learning and CVLT_false_alarms which are two aspects of verbal memory.

- LASSO reports which individual CVLT subscores were selected across the CV folds. If it consistently chooses CVLT_1–5 sum and CVLT_13, this validates the feature selected baseline choices.

- If delayed recall (CVLT_10–12) is seletced by LASSO, this baseline should be revisited.

When to revisit:

- LASSO consistently highlights features that wetrent included in the feature selected baseline.

- Delayed recall or interference measures show surprising discriminative power.
