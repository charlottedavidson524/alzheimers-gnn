# Downsampling

Downsample the EEG signal from 1000 Hz to 250 Hz after filtering (bandpass and notch), before bad-channel detection. considered keeping the original 1000 Hz sample rate and downsampling to either 128Hz or 500Hz.

The Nyquist-Shannon sampling says the sample rate needs to be 2x the maximum frequency of interest. Bandpass filtering limits the useful signal to 1-45 Hz. A 250 Hz sample rate has a Nyquist frequency of 125 Hz, providing a 5.5 x safety margin above the analysis band. No frequencies relevant to this project are affected by the downsampling.

A lot of the AD-EEG-GNN literature uses 250 Hz as the standard sample rate for connectivity-based analysis. Klepl et al. (2023) and Cao et al. (2024) use 250 Hz. Matching this lets us have comparable methods to other papers.

Downsampling also reduces processing time by 4x in this case. For n=79 subjects, ICA fitting alone is reduced from around 4 hours to approx 1 hour across the cohort. Important in case pipeline needs to be reran.

Also efficient for starage. Preprocessed epochs are around 4 times smaller (from roughly 1.2 GB per subject to 300 MB). this should reduce total dataset storage from arounbd 95 GB to 24 GB.

Resampling should follow the batch and bandpass filters. Resampling after filtering avoids any aliasing risk because the signal is already bandlimited to 45 Hz. This is well below the new Nyquist frequency of 125 Hz. MNE's `raw.resample()` also applies an anti-aliasing low-pass filter internally as a safety measure.

Downsampling will need to be reconsidered if gamma-band connectivity above 100 Hz becomes a target of interest (unlikely for AD/APOE work).
