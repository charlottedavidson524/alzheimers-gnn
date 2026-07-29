# Functional Connectivity Plan (Revised)

This doc is the revised functional connectivity pipeline for building graph representations of resting-state EEG for GNN-based APOE e4 classification.

## Connectivity measure

The connectivity measure that will be used is wPLI, computed using filter and Hilbert with 5th order Butterworth IIR bandpass. This is because wPLI is volume-conduction robust (Vinck et al. 2011). Filter-Hilbert matches Klepl et al.'s method for AD-EEG-GNN work. Butterworth IIR is chosen over FIR because IIR filter length is not tied to transition bandwidth, allowing analysis of low frequencies (down to 0.5 Hz) within 4-second epochs.

## Frequency bands

Two bands are chosen. Tyhese are delta (0.5-4 Hz) and alpha-2 (10-12 Hz). This is because Dzianok et al. (2025), when analyzing the PEARL-Neuro cohort, identified statistically sig APOE/PICALM effects specifically in delta (global power difference) and alpha-2 (regional trend-level difference). Klepl et al. (2023) reported their best single-band GNN performance in the 7-15 Hz range, overlapping alpha-2. Keeping it to these two bands will align the analysis with cohort-specific evidence as well as precedent found in the literature. Theta, alpha-1, beta, and gamma wont be in the primary model but could be added as sensitivity analyses if there is enough time.

## Edge thresholding

Use the proportional threshold, top 20%. So, keep the 20% strongest wPLI edges per band per epoch. This matches Klepl et al.'s best-reported GNN configuration exactly. A k-sweep (10/20/30%) and MST/OMST comparison are optional follow-ups.

## Graph aggregation

Per-epoch graphs are used as independent training examples, with subject-level grouped cross-validation being a hard requirement. Brookshire et al. (2024) showed that segment-based holdout on Alzheimer's EEG classifiers produces severely inflated accuracy (99.8%) compared to subject-based holdout (just over 50%, basically chance). Any epoch-level splitting would leak subject identity into the test set.

## Node features

Nodde features will be per-branch, band-specific relative power. This means one scalar per node per branch. Each branch receives only its own band's power vector as node features (1-dimensional per node per branch). This keeps branches informationally self-contained:

- Delta branch: delta power per node
- Alpha-2 branch: alpha-2 power per node

This matches Xu et al. (2025)'s architecture. Helps stop cross-band information leakage at the node feature level which forces the model to learn cross-band interactions only through the concatenation stage after graph convolution.

## Edge features

Scalar wPLI per edge, one graph per band. Two parallel per-band graphs (not a multi-dimensional edge vector). Each graph is processed by its own 2-layer GCN branch. Branch outputs are concatenated after graph convolution and passed through an MLP classifier. This is because standard GCN architectures (Kipf & Welling 2017) require scalar edge weights. Multi-band information enters via parallel branches, not via vector-valued edges. This matches Xu et al. (2025)'s architecture.
