# Epoching

The decision that was made was to epoch the eyes closed segment into 4-second, non-overlapping epochs, then apply autoreject in "interpolate" mode with `random_state=42` to identify and repair or reject bad epochs. Alternatives that were considered include other epoch lengths (e.g. 2s, 8s), different degress of overlap (25%, 50%, 90%), and regarding rejection of epochs looked at fixed peak-to-peak thresholds and manual inspection.

The reasons for this decision include the fact that 4-second non-overlapping epochs match the PEARL-Neuro authors' own analysis of the same dataset (Dzianok et al. 2025 used 75 epochs from 5 minutes of eyes-closed data. My pipeline uses roughly 90 epochs from the full approx 6 minutes). Epoch length provides 0.25 Hz frequency resolution. This is adequate for delta (1-4 Hz) and comfortable for alpha (8-13 Hz). Autoreject (Jas et al. 2017) handles bad-epoch identification using cross-validated per-channel thresholds. This is consistent with the pipeline's earlier bad-channel handling approach (interpolation being preferred to outrightrejection).

Could've had longer epochs, because the wPLI methodological literature (Kim et al., 2022 as an example) prefers 8-second epochs for maximum connectivity stability. Fraschini et al. (2016) also show that epoch length affects both PLI values and reconstructed network topology at the scalp level. The erasons for the shorter epochs are:

- Comparability on the dataset. Dzianok et al. (2025) analysed the exact same PEARL-Neuro recordings using 4-second epochs. Matching their epoch length ensures connectivity estimates from this pipeline are directly comparable with the dataset authors' own analyses. This comparability is more useful than a slight improvement in stability.

- Assumption of stationarity. wPLI computation assumes signal statistics are approximately stable within each epoch. Longer epochs make this assumption progressively less defensible. Fraschini et al. (2016) note this tension explicitly: longer epochs stabilise the estimate but at the cost of the stationarity assumption on which the estimate depends.

- Consistency mitigates length effects. Fraschini et al.'s key methodological warning is that epoch length affects results at the scalp level. Their recommendation is either to move to source space (out of scope for this project) or to keep epoch length consistent across subjects. My pipeline uses identical 4-second epochs for all 79 subjects, which addresses the concern within the sensor-space framework.

Slightly less stable per-epoch wPLI estimates than 8 seconds would provide. This is mitigated by aggregating across roughly 90 epochs per subject, whicg is comfortably above the approx 12-50 epoch adequacy thresholds documented in the literature (Cai et al. 2020; Hardmeier et al. 2014).

The recent AD-EEG literature also includes examples using 50-90% overlap for dynamic connectivity analyses. 50% overlap would double the number of epochs per subject from roughly 90 to around 180. Chose to do 0% overlap. Reasoning for this:

- Autoreject's cross-validation assumes epoch independence. Autoreject estimates rejection thresholds via cross-validation over epochs, treating epochs as statistically distinct trials. Highly overlapping epochs share data, so a 50% overlap means adjacent epochs are 50% identical, which violates the independence assumption underlying cross-validation. Overlapping epochs would confound autoreject's threshold estimation and potentially bias the rejection decisions.

- There is no statistical need for moew epochs. Cai et al. (2020) demonstrated reliable wPLI estimates from as few as 12 four-second epochs in adults. At around 90 non-overlapping epochs per subject, this pipeline has approximately 7 times the minimum adequate epoch count. Adding overlap would provide diminishing returns for a need that doesn't exist.

- Cleanneess in the methodology. Non-overlapping epochs match the PEARL-Neuro authors' approach (Dzianok et al. 2025) and Fraschini et al.'s (2016) rigorous choice. Non-overlapping epochs avoid overlap-induced correlations between wPLI estimates that would complicate downstream statistical interpretation.

There is a trade-off here. There are fewer total epochs than an overlapping scheme would produce. This is not a limitation because of the statistical adequacy that has already been ensured. If the pipeline were extended to dynamic connectivity analysis (tracking how connectivity changes within a single subject's recording), sliding windows with overlap might be needed but this is out of scope for the current project.

Manual visual inspection of each epoch by a trained neurophysiologist is the past gold standard for artefact identification. The Sheffield UK AD-EEG-GNN group (Klepl et al. 2022; Shan et al. 2022; Cao et al. 2024) uses this approach. I am using autoreject. Why?

- Not really possible at n=79 with reproducibility as a requirement. Manual inspection scales linearly with cohort size and requires a trained expert. The Sheffield UK group operates with cohorts of 20-40 subjects, where manual inspection is feasible. At n=79, manual inspection would require days of expert time per pass, and any parameter changes to the pipeline would require re-inspection. I am also not an expert in this domain on top of all of this.

- Manual inspection introduces experimenter subjectivity. Discussed how this reduces reproducability previously. Autoreject is deterministic given a random seed (`random_state=42`), sothe pipeline produces identical outputs on repeated runs. This ensures reproducibility in a way manual inspection can't.

- Autoreject outperforms fixed thresholds and can match manual inspection. Jas et al. (2017) demonstrated on four public datasets that autoreject achieves rejection performance comparable to manual inspection while being fully automated. The cross-validated per-channel thresholds address the main weakness of naive fixed thresholds (e.g.,+/-150 muV). They adapt to each subject's baseline amplitude distribution instead of imposing an arbitrary universal cutoff.

there is also a tradeoff here. There is loss of expert judgment on ambiguous cases. This isitigated by autoreject's "interpolate first, reject as fallback" mode. This is conservative, and it preserves epochs when possible instead of getting rid of them. Interpolation intsead of deletion is consistent with the bad-channel handling choices. When only a few channels in an epoch are affected, autoreject interpolates instead ofdropping the entire epoch. This keeps statistical power. Only epochs with too many bad channels are flat out rejected.

## Pipeline ordering (final)

Epoching happens after ICA has cleaned the continuous signal:

1. Filtering (bandpass, notch)
2. Bad-channel detection and interpolation
3. Common Average Reference
4. Fit ICA on continuous data
5. Classify components (ICLabel), remove artefact components
6. Reconstruct clean continuous signal
7. Extract eyes-closed segment via S 4 -> S 11 markers
8. Epoch into 4-second non-overlapping segments
9. Apply autoreject to identify, repair, or reject bad epochs
10. Produce cleaned epochs ready for connectivity computation

Come back to this if wPLI estimates show high variance across bootstrap resamples per subject. Add 50% overlap to double yhe epoch count (accepting the autoreject caveat above). Also, if many subjects have more than 30% of epochs rejected by autoreject, should tighten upstream preprocessing (ICA, bad-channel handling) instead of loosening rejection criteria.
