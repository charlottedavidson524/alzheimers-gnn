# Artefact Rejection

To deal with artefacts, they are rejected automatically via ICA using the Picard algorithm with EXtended Infomax settings. There is then automated component classification using ICLabel. Compoennts classified as things like eye blink, muscle artifact, heart beat, line noise, or channel noise with p> 0.90 are removed. "Brain" and "other" components are kept.

## Choice of ICA algorithm

Considered FastICA (MNE default) and classic Extended Infoxmax but went with wwhat was stated above. Reasoning: Friebem, Feld & Braun (2018) compared Extended Infomax, FastICA, and TDSEP for muscle artefact removal in EEG. they dfound Extended Infomax best (marginally) but the differences were small relative to the effect of high-pass filter cutoff choice. Makes the 1 Hz cutoff choice (`see docs/decisions/bandpass-filter.md`) more important than algorithm choice, but Extended Infomax is still tye best performing option among the ICA variants tested. Picard (Ablin et al. 2018) with `extended=True`, `ortho=False` is used as the implementation because it produces solutions equivalent to Extended Infomax with much faster convergence. This is key at n=79 subjects.

## Number of components

Artoni, Delorme & Makeig (2018) showed that PCA rank reduction before ICA (even removing only 5% of variance) reduces the number of dipolar components recovered from around 30 to around 10 per subject and drops median IC stability from 90% to 76%. n_components is therefore set to the effective rank of the data, computed per subject via `mne.compute_rank(raw)["eeg"]`. This handles rank reduction from CAR and from interpolated bad channels uniformly and automatically, avoiding both rank deficiency (which produces ghost components) and unnecessary dimensionality reduction.

NOTE AFTER PREPROCESSING SUB-01: `n_components` is set using `numpy.linalg.matrix_rank(raw.get_data())` instead of `mne.compute_rank(rank="info")`. MNE's info-based rank originally remained at 125 after CAR and interpolation, while numpy correctly reported 121. Using info-based rank produces ghost components.

## How to classify components

ICLabel (Pion-Tonachini et al. 2019) is trained on over 200,000 independent components from more than 6,000 EEG recordings and is benchmarked as outperforming MARA, SASICA, and other IC classifiers. At the same time it is 10 times faster than the previous best classifier. Manual visual inspection of ICA components isnt practical at n=79 when keeping reproducability in mind, and introduces subjectivity that ICLabel removes.

## How to reconstruct

A 0.90 probability threshold is used for artefact rejection. Common in ICLabel-based automated pipelines and is balanced. High enough to avoid removing brain components ICLabel is uncertain about, low enough to reject obvious artefacts. Pion-Tonachini et al. (2019) present per-class classification accuracy at multiple confidence levels (Table 4) but dont give a specific threshold. 0.90 is chosen beacuse its a defendable given their reported accuracies. Retaining "other" components is conservative. Its better to keep possible brain signal vs removing too much

## Pipeline order (as of now)

1. Bandpass filter (1–45 Hz)
2. Notch filter (50 Hz)
3. Detect bad channels (LOF, threshold 1.5)
4. Set common average reference (need this for ICLabel because classifier was trained on CAR-referenced data)
5. Interpolate bad channels via spherical spline
6. Fit ICA with `n_components = mne.compute_rank(raw)["eeg"], method="picard", fit_params={"extended": True, "ortho": False}`
7. Classify components via ICLabel
8. Apply ICA to remove artefact components
9. Extract eyes-closed segment
10. Epoch into fixed-length windows

CAR needs to go before ICA because ICLabel needs it to produce reliable classifications.

Come back to this if:

- ICLabel classifications look consistently wrong after inspecting of a few subjects
- Components with obvious neural characteristics are being flagged as "muscle"
