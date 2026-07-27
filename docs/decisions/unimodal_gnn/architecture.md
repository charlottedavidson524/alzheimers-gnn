# Architecture: Primary GNN

First model to be trained, before any potential ablation/hyperparamater tunbing/etc (edge-thresholding k-sweep, node-feature variants, or architecture alternatives) This doc consolidates the six upstream decision docs (frequency-bands.md, edge-thresholding.md, graph-aggregation.md, node-features.md, edge-features.md, and the connectivity-measure choice) into a single architecture guide.

Each of the six decision docs behind this architecture was written to answer one question in isolation, with a mostly defaults being decided instead oif an ablation grid due to time constraints on the project. Combined together, there is implications for what a model should look like but describing that model here for posterity.

NOTE: there was an incompatibility that needed to be solved first. node-features.md decided that each node's feature vector is its relative power across all frequency bands (a 6-dimensional vector: delta, theta, alpha-1, alpha-2, beta, gamma). edge-features.md, written afterward, decided that multi-band connectivity enters the model as parallel per-band GCN branches rather than a single fused graph. Neither doc, on its own, specified if each per-band branch should receive the full 6-dimensional node-feature vector, or only the single band-power value corresponding to that branch's own band. Solution is that each branch receives only its own band's relative power as the node feature (a 1-dimensional value per node, per branch), not the full 6-dimensional vector.

decided this based on mostly precedent, favoured whatever had the strongest literature support. Xu et al. (2025) which is the closest task-matched precedent available throughout this whole set of decisions (multi-band EEG connectivity, GCN-based, Alzheimer's classification, explicitly structured as parallel per-band branches) uses this design. Each of their five band-specific GCN branches receives only that band's own differential-entropy value per node, not a cross-band feature vector. Feeding the full multi-band vector into every branch would be valid (GCN doesnt seem to place any constraints on node feature dimensionality) and isnt wrong per se, but it doesnt have much precedent among the papers surveyed across this project's decision docs. On the other hand, the band-specific, self-contained-branch design is the one paper that most directly matches this project's exact combination of choices does. Given no in-cohort or project-specific evidence favors one over the other, precedent breaks the tie.

## Architecture (end to end)

### Input, per training example (one epoch, one subject)

6 parallel graphs, one per frequency band: delta, theta, alpha-1, alpha-2, beta, gamma (frequency-bands.md). Each graph has the same 127-node topology (EEG electrodes). Edges (per band graph): scalar wPLI value between each electrode pair (connectivity measure decided prior to this doc set), proportionally thresholded at top-20% — Klepl et al. (2022)'s best-performing GNN configuration, used as the sensible single default ahead of edge-thresholding.md's planned k = {10%, 20%, 30%} sweep and MST/OMST comparison. Node features (per band branch): each node's relative power in that branch's own band only. Means a single scalar per node, per branch.

### Per-band branch (x6, one per band)

Use a 2-layer GCN (Kipf & Welling renormalized propagation rule). Also matches Klepl et al. (2022)'s empirically-optimal depth for this task. Each layer: graph convolution -> ReLU -> batch normalization.
Readout = max pooling over the 127 nodes, producing one graph-level embedding vector per band branch. Matches Klepl et al. (2022)'s. Each branch holds its own, independently trained weights. they're not shared across bands, per ME-GCN (Wang et al., 2022)'s controlled ablation showing separated per-stream weights significantly outperform shared weights.

### Fusion

The 6 per-band graph-level embeddings are concatenated into a single vector after graph convolution is complete. Matches Xu et al. (2025)'s flattening step. Bands do not exchange information during message-passing, only at this fusion point (edge-features.md).

### Classifier head

2–3 fully connected layers on the concatenated vector, with dropout between them. Final layer: 2 output units (APOE e4 carrier/non-carrier), softmax activation, cross-entropy loss.

### Training data structure and evaluation protocol

Per-epoch graphs are used as separate training examples rather than ine summary graph per subject. Matches Klepl et al. (2022)'s precedent for making GNN training feasible at comparably small N. Subject-level grouped k-fold cross-validation is a hard requirement. No subject's epochs can appear in both a training and a test/validation partition at any point (fold assignment, hyperparameter selection or final evaluation). Need to make sure this never happens. This is anchored by Brookshire et al. (2024)'s direct demonstration that segment-based (non subject grouped) splitting on an Alzheimer's EEG classifier produced 99.8% accuracy that collapsed to 53.0% under correct subject-based splitting using the exact same model and data. At inference, per-epoch predictions for a held-out subject are aggregated into a single subject-level prediction (majority vote or averaged predicted probability) because the classification is being done per subject not per epoch.

### What's not included

- No cross-band message-passing during convolution. Bands only interact at the final concatenation layer. A genuinely edge-feature-capable architecture (GINEConv, NNConv, EGNN) that could reason across bands at every convolutional step is a legitimate richer alternative. This is the top follow-up if this primary model underperforms.

- No learnable/adaptive graph structure. Edges are fixed, thresholded wPLI values, not learned or refined during training (unlike, e.g., AGGCN's adaptive graph learning module).

- No per subject summary-graph variant is run in parallel. Per epoch graphs are the sole default. The summary-graph alternative is possible (see graph-aggregation.md) as an improvement

- No node-feature ablation run alongside this model. Differential entropy (node-features.md's suggested alternative) is not tested here. No MST/OMST comparator run alongside this model. edge-thresholding.md's top-20% proportional threshold is used as a single fixed default for this first run. The full k-sweep and MST/OMST comparison are follow-up work.

This model is the primary EEG-only pipeline. It is run alongside tabular-only baselines (logistic regression, random forest) and both late and intermediate fusion variants, with the multimodal comparison being the project's main contribution.
