# Multimodal Fusion

Have decided to treat the project as a multimodal modelling problem. Will report on architectures considered and intended experiments.

PEARL-Neuro provides two kinds of data per participant.

- EEG: Recorded over 128 electrodes with natural spatial and connectivity structure. Graph shaped.
- Tabular: Blood markers (Blood cell types, lipid panel, HSV), psychometric scores (BDI, RPM, NEO, CVLT, MINI-COPE), demographic and lifestyle variables, as well as the PICALM genotype. Flat, unstructured feature vectors.

These two modalities suit different types of models.

- A GNN uses EEG's graph structure. Forcing tabular variables through this architecture will flatten them into graph-level features (no structural benefit) or broadcast them to every node.
- Classic ML models handle tabular data efficiently but can't represent the EEG's graph structure without losing information.

Using a single model class for both modalities can force a compromise. Multimodal modelling uses the right model for each modality and combines their predictions.

## Fusion Strategies

There are three common fusion strategies that can be used. Will look at these three and determine which to use.

### Late fusion

Two independent models are trained separately. In this case this would be a GNN on the EEG graph, and aML model on the tabular features. Each produces a prediction and these are combined using averaging, weighting or a small meta-learner.

Strengths:

- Easiest to implement, reuses ML models and GNN models that will have already been created.
- Each model is independently interpretable.
- Robust at small sample sizes.
- Closest to two clinical tests being combined, which is a natural framing for Alzheimers screening.

Weaknesses:

- The two models can't learn cross-modal interactions.
- Combining predictions requires choices and hyperparameters that need tuning.

### Intermediate fusion

This requires a single end-to-end model. The GNN processes the EEG graph and produces a fixed-dimensional embedding. Tabular features are concatenated to that embedding and a small MLP (1-2 layers) classifies using the combined vector.

Strengths:

- GNN learns EEG-specific representations using the architcetural strengths it has.
- Tabular features bypass GNN and reach the classifier directly.
- Final layer can learn cross-modal interactions,
- A single trainable model.

Weaknesses:

- More parameters than late fusion. Higher overfitting ridsk at n=79. Will likely need regularisation.
- Harder to attribute final prediction to either modality in a clean way.

### Early fusion

This is when tabular features are injected at the GNN's input.

Strengths:

- Maximum opportunity for cross-modal interaction.

Weaknesses:

- Hard to design well
- Often hurts perfromance in practice.
- Not easy to interpret.
- N=79 is likely too small of a sample size.

Rejected this. Too complex for the sample size and not interpretable enough given explainability focus.

## Architectures Chosen

Intermediate fusion is the primary focus but late fusion will also be looked at as a comparator. Intermediate is the standard in modern work. Late fusion will act as a comparator. The two together will show each modality's contribution.

## Experiments Planned

The full set of models that will be done.

| Model               | Inputs         | Architecture                     | Purpose                         |
| ------------------- | -------------- | -------------------------------- | ------------------------------- |
| Tabular (LR)        | Only tabular   | Logistic Regression              | A baseline model but linear     |
| Tabular (RF)        | Only tabular   | Random Forest                    | A baseline model but non-linear |
| GNN                 | EEG graph only | GCN                              | Unimodal EEG ceiling            |
| Late fusion         | Both           | GNN and tabular combined         | Independant modality fusion     |
| Intermediate fusion | Both           | GNN embedding and tabular -> MLP | Joint-representation fusion     |

Twwo sub-variants of each model that uses tabular data:

- The features available for all participants (demographics, psychometrics, genetics. Excludes blood panel).
- Features available for the n=79 subset.

## Implications
