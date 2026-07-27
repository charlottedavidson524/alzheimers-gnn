# Functional Connectivity Summary

1. Connectivity measure is wPLI. Already decided this going in. The weighted phase-lag index is chosen for robustness to volume conduction, this is well established in the AD-EEG-GNN literature (Klepl et al., 2022).

2. Frequency bands: use all five conventional bands (delta, theta, alpha-1, alpha-2, beta, gamma) rather than alpha alone or a single broadband pass. This is inspired by this cohort's own companion analysis (Dzianok et al., 2025). they found real APOE/PICALM-related power differences in delta and alpha-2 specifically, plus corrected evidence that gamma shouldn't be dropped by default (Babiloni et al., 2021; Klepl et al.).

3. Edge thresholding. Proportional (top-k%) thresholding will be used as default, k swept as a hyperparameter (10/20/30%, same as Klepl et al.). This should be gated on an empirical check for whether overall wPLI strength differs by APOE group, informed by wPLI-specific threshold instability evidence (Adamovich et al., 2022) and the case control specific caution about proportional thresholding (van den Heuvel et al., 2017). MST/OMST noted as something to try for comparison, not as the default.

4. Graph aggregation -> per-epoch graphs (not one summary graph per subject) as separate training examples. Matches Klepl et al.'s precedent for small-N GNN training with subject-level cross-validation splitting as a hard requirement that is not up for debate. This is rooted in Brookshire et al. (2024)'s proof that segment-based splitting on an Alzheimer's EEG classifier reduced accuracy from 99.8% to no better thanchance.

5. Node features: each node carries its relative band-power vector (one value per band), rather than differential entropy, full continuous PSD, coordinates, or one-hot identity. this was chosen for its link to this cohort's own power findings, despite DE showing an advantage in a controlled ablation study (Xu et al., 2025).

6. Edge features and multi-band fusion -> each edge carries a single scalar wPLI value. Multi-band information enters as one GCN branch per band (not a multi-dimensional edge vector), with branch outputs concatenated after graph convolution. This corrects an earlier proposal in frequency-bands.md once it became obvious that a vector-valued edge is not really compatible with a standard GCN (default architecture) and instead matches Xu et al. (2025)'s architecture directly.
