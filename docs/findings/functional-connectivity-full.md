# Functional Connectivity Findings -> Full Cohort (n=77)

These are the cohort level results of applying the graph construction pipeline to the preprocessed EEG.

## Usable Cohort

- Preprocessed cohort had 77 subjects (from `docs/findings/eeg-preprocessing/preprocessing-full.md`)
- Successfully built 77 graphss 77 subjects
- 0 subjects failed
- 2 were excluided previously (sub-55, sub-69). In line with PEARL-Neuro

## Graph Structure

Every subject produced graphs with identical structural properties:

- 125 nodes (127 recorded channels minus O9, O10)
- 6 frequency bands (delta, theta, alpha1, alpha2, beta, gamma)
- 3,100 edges per graph (top-20% of 7,750 upper-triangle edges which are symmetrised)
- Node features: 6-dimensional relative band powers per node
- Edge features: scalar wPLI value per edge

## Data Volume

- 74 subjects had 90 epochs each (standard eyes closed protocol)
- 3 subjects with 70 epochs each (sub-19, sub-30, sub-34. These were the 280s fallback subjects)
- There were 5 subjects with reduced epoch counts due to high autoreject rejection:
  - sub-36: 64 epochs
  - sub-54: 61 epochs
  - sub-70: 76 epochs
  - sub-74: 72 epochs
  - sub-76: 56 epochs (lowest in cohort)
- All subjects had 56 or more epochs, which is above statistical adequacy for wPLI

There are approximately 40,000 graphs across the cohort (77x90x6, adjusted for reduced-epoch subjects).

## APOE Label Distribution

From from `participants.tsv` via the `APOE_haplotype` column:

- Non-carriers (label=0): subjects with no e4 allele
- Carriers (label=1): subjects with at least one e4 allele (e3/e4, e4/e4, e2/e4)

Effective cohort class distribution:

- Carriers are 46 subjects (60%)
- Non-carriers = 31 subjects (40%)

This is a 1.5:1 class imbalance (carrier heavy). There are some implications for GNN training:

- Loss weighting is optional given the mild imbalance
- Stratified cross-validation is still needed

## wPLI value distributions

First made graphs for sub-06. Checked wPLI value distribution on sub-06 as a result (because values seemed very high) by histogram inspection of edge weights after top-20% thresholding:

- Delta: peak around 0.55, right-skewed to 0.95
- Theta: peak around 0.40, tail to 1.00
- Alpha1: peak between roughly 0.55-0.60, wide distribution
- Alpha2: peak around 0.55, similar to alpha1
- Beta: peak ariund 0.28. Has a narrower distribution
- Gamma: peak at about 0.20, narrowest tail

Distributions are right-skewed as expected for wPLI. It's because it's the top-20%. No notable patterns that would indicate anything of significance (no spikes at 0 or 1 or any bimodal shapes).

Cross-subject consistency confirmed on skeleton subjects (sub-01 to sub-05) when running the graph construction pipeline on the first five. Alpha1 means ranged 0.55-0.70 and delta means 0.61-0.63. Variation between subjects is small which is a sign that the pipeline is working well.

## Methodology Choices

- wPLI as connectivity measure (`connectivity-measure.md`)
- 6 frequency bands (`frequency-bands.md`): delta [0.5-4], theta [4-7], alpha1 [7.5-9.5], alpha2 [10-12], beta [12-30], gamma [30-45] Hz
- Top-20% edge thresholding per band per epoch (`edge-thresholding.md`)
- Per-epoch graphs as separate training examples (`graph-aggregation.md`)
- Per-epoch relative band-power node features (`node-features.md`)
- Scalar wPLI edge features. Multi-band handled via architecture, not edge vectors (`edge-features.md`)

## Implementation Notes

Made two choiced when implementing that should be documented.

- wPLI computation via `mne_connectivity.spectral_connectivity_time` with CWT-Morlet mode and adaptive `n_cycles`. Cycles capped at 3.0 for higher frequencies, reduced to `freqs x 1.8` at lower frequencies (giving around 0.9 cycles at 0.5 Hz) so wavelet windows fit within the 4-second epoch. Delta-band estimates below 1 Hz use fewer cycles and are noisier.

- wPLI symmetrisation. MNE-connectivity returns wPLI in the lower triangle only. The upper triangle is zeros. The pipeline outright symmetrises using `m + m.T` before top-k thresholding, which is safe because the diagonal is pre-zeroed.

## Cohort Patterns

- Structural uniformity. Every subject produces same 125-node graph structure with the same 3,100 edges after thresholding. Need this to be uniform for cross-subject GNN training.

- wPLI hierarchy across bands. Alpha bands consistently show the highest coupling (mean 0.55-0.70 for top-20% edges), then delta and theta, then beta, then gamma with lowest coupling (mean ~0.24). This matches expectation that alpha rhythm is the dominant resting-state oscillation.

- Individual variation within bands is modest. Within-band means vary approximately +/-0.10 across subjects. Thsi reflects real biological variability rather than pipeline artefacts.

- Subjects with fewer epochs still produce valid graphs. The 5 subjects with reduced epoch counts (56-72 epochs) produce structurally-identical graphs, just less of them. Statistical noise per subject is slightly higher but epoch counts remain above adequacy.

## Implications for GNN

- Fixed 125-node graph structure across cohort.

- 1.5:1 class imbalance. Look into loss weighting or stratified sampling. `StratifiedGroupKFold` probably best for cross-validation.

- Reduced-epoch subjects contribute fewer graphs. Data loader should treat each (subject, epoch, band) triple as a separate example. Subjects with fewer epochs will have fewer training examples, but this is handled naturally by the flat list of graphs structure so shouldn't be a concern.

- Subject-level splits are VITAL. See `graph-aggregation.md` -> all cross-validation splits need to be at the subject level to avoid the leakage documented by Brookshire et al. (2024). Epochs should never be split within a subject.

## Files produced

- `C:/data/pearl-neuro/derivatives/graphs/sub-XX/sub-XX_task-rest_graphs.pt` per subject (77 files, each containing 336-540 PyG Data objects)
- `results/graph_construction_summary_full.csv` (77 rows)
- `results/graph_construction_summary_skeleton.csv` (5 rows from the skeleton run)
