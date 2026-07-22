# Bandpass

It looks likely from EDA of EEG thaat a 1-45Hz bandpass will be needed. However this should be justified from the literature.

1–45 Hz zero-phase FIR bandpass filter should be used, applied to the continuous EEG. Filtered data is used for both ICA fitting and all downstream processing like epoching or wPLI computation.

high-pass cutoff at 1 Hz is chosen to optimise ICA decomposition quality. See systematic evaluations of Winkler et al. (2015) and Klug & Gramann (2021). Winkler et al. (2015) found that high-pass filtering between 1 and 2 Hz consistently produced the best results in terms of SNR, ICA component dipolarity, and single-trial classification accuracy, and that cutoffs below 0.5 Hz were suboptimal.

Klug & Gramann (2021) extend this to high-density montages. They found that cutoffs between 0.5 and 2 Hz produce the best ICA decompositions. Higher cutoffs (roughly 1.25 Hz for 128-channel setups) were prefereed as channel density increases. This dataset's 127-channel montage is directly in the high-density regime they address. Because this pipeline will use ICA for artefact removal (see `docs/decisions/artefact-rejection.md`), ICA-optimal filtering sways the trade-off.

The 45 Hz low-pass cutoff iss above the 40 Hz floor Widmann et al. (2015) recommend for keeping high-frequency neural components and below the 50 Hz mains contamination confirmed present in the raw data (see `docs/findings/eeg-cohort-eda.md`) (handled by a separate notch filter).

A zero-phase FIR implementation is used (MNE's default), matching the filter type tested in both Winkler et al. and Klug & Gramann. The workflow of applying ICA weights back to less-filtered data (Winkler et al. 2015) isntr adopted here. Resting state alpha band connectivity isnt sensitive to the temporal distortions that motivate it in ERP work.

# Notch

Apply a 50 Hz notch filter in addition to the 1–45 Hz bandpass.

This is because the 45 Hz upper cutoff of the bandpass places the analyzed band below the mains contamination frequency. This on it's own should be enough for pipelines that operate just on band-limited analysis (Shan et al. 2022, Cao et al. 2023, and Zhang & Zhu 2025. None of these apply an explicit notch).

However, cohort EDA (`docs/findings/eeg-cohort-eda.md`) saw a sharp 50 Hz spike across all 5 inspected subjects. This is consistent with the PEARL-Neurp paper's report of no notch filter having been applied while recording. Adding a 50 Hz notch stops residual mains energy leaking through the bandpass transition band. This follows the approach of Klepl et al. 2022, whose preprocessing (Butterworth 49–51 Hz stop-band) is the closest thing to this pipeline in the EEG-GNN-for-AD literature. 50 Hz rather than 60 Hz is used because the PEARL-Neuro dataset is Polish where mains supply is 50 Hz.
