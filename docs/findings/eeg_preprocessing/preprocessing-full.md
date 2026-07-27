# Preprocessing Findings For Full Cohort (n=77)

These are the results of the 13-step preprocessing pipeline being applied to the PEARL-Neuro resting-state EEG for all usable subjects.

## Usable cohort

- There are 79 subjects (per dataset)
- 77 subjects preprocessed successfully
- 2 subjects were excluded because of missing data
  - sub-55: rest recording missing due to technical problem (documented in PEARL_Neuro paper)
  - sub-69: no consent to publication (also in PEARL_Neuro paper)
- Recovered via fallback: 3 subjects (sub-19, sub-30, sub-34) had truncated events files with missing S 11 markers but full-length raw recordings. Handled via 280s eyes-closed fallback duration

Rate of loss was 2.5% (2/79). This matched the paper's documented missing data. No subjects excluded by the pipeline's 20% bad-channel threshold.

## Data volume

- There were 74 subjects with standard eyes-closed duration (roughly 361 s, S 4 -> S 11)
- There were 3 subjects with fallback duration (280s, S 4 + 280 s)
- The total amount of epochs before autoreject was 6,870 (74 × 90 + 3 × 70)
- The total number of epochs after autoreject is around 6,530 (about 5% average rejection across cohort)

## Bad channel patterns

- Median bad channels per subject = 5
- Range: 0 (sub-11, sub-28, sub-31, sub-71) to 22 (sub-17)
- Highest bad-channel ratio was sub-17 at 17.6% (still below 20% exclusion threshold)
- There was a systematic pattern that was spoted. Frontopolar and frontal electrodes (Fp1, Fp2, AF7, AF8) flagged across approximately 40 subjects. This was consistent with volume-conducted eye artefacts and frontalis muscle activity affecting these positions.

Interpolation via spherical spline handled all detected bad channels. No subject needed any exclusion.

## ICA behaviour

- Effective rank calculation: working correctly (via `numpy.linalg.matrix_rank`)
- Components fit range: 102-124. This is consistent with 125 - 1 (CAR) - n_bad_channels_interpolated
- Components removed had a median of 10 and range of 2-20
- Category distribution across cohort:
  - Eye blinks: 1-5 per subject (most subjects: 2)
  - Muscle artifact: 0-12 per subject (varies a lot)
  - Channel noise: 0-11 per subject (also varies a lot)
  - Heart beat: 0-1 per subject (rare)
  - Line noise: 0 across all subjects (notch filter working)

Line noise never flagged as an ICA component across 77 subjects, confirming the 50 Hz notch filter successfully removed mains contamination before ICA.

## Autoreject behaviour

- Most subjects autorejected: 0-3 epochs rejected
- High rejection subjects include:
  - sub-36: 26 epochs rejected (29%)
  - sub-54: 29 epochs rejected (32%)
  - sub-74: 18 epochs rejected (20%)
  - sub-76: 34 epochs rejected (38%. This is the highest)
  - sub-70: 14 epochs rejected (16%)

All subjects keep 56 epochs or more. This is comfotably above statistical adequacy for wPLI (Cai et al. 2020, Hardmeier et al. 2014)

## Visual sanity check on outlier subjects

Checked out three subjects that were potentially concerining. They were visually inspected using PSD and time-series plots (have screenshots on laptop):

- sub-44 (13 bad channels, only 2 ICA components removed). PSD showed clear alpha peak at around 10 Hz and smooth 1/f falloff. Time series showed no obvious residual artefacts. So, can determine that this is acceptable. The "shallow" ICA cleaning is probably a result of low residual artefact content instead of any under cleaning.

- sub-70 (10 bad channels, 3 ICA components removed, 14 epochs rejected). PSD showed large alpha peak with one localised residual gamma bump in one channel. Time series was clean. Autoreject compensated for ICA's under-cleaning. determined that the subjecy was still acceptable

- sub-76 (34 epochs rejected, highest in cohort). PSD showed very clean spectrum with big alpha peak. Time series was very clean. Autoreject was aggressive but justified because the remaining epochs are very high quality. This is very good, one of the cleanest subject.

Can determine from this visual check that the ICA anc autoreject combination is strong. When one stage under cleans, the other compensates. All inspected outliers produced usable clean data.

## Issues (Identified and Resolved)

Three issues came up during this pipeline:

1. O9 and O10 lack `standard_1005` positions

MNE's `standard_1005` montage has NaN 3D coordinates for O9 and O10 outer-ring occipital electrodes. This originally broke spherical spline interpolation with this error message: `ValueError: array must not contain infs or NaNs`. Fixed it by dropping these two channels consistently for every subject, giving a 125-node graph for all subjects.

2. `mne.compute_rank(rank="info")` was unreliable

Initially set ICA `n_components` via MNE's info-based rank computation. checked on sub-01 that this returned 125 after CAR (which should reduce rank by 1) and bad-channel interpolation (which should reduce rank by n_bad). The actual data rank via `numpy.linalg.matrix_rank(raw.get_data())` correctly reported 121 for sub-01. Fitting ICA with the wrong rank produced 4 ghost components misclassified as "channel noise." Fixed this by computing effective rank from data directly via numpy.

3. Truncated events files with missing S 11 markers

Three subjects (sub-19, sub-30, sub-34) had events files with only the first few events logged. The S 11 marker was missing. Raw EEG recordings were full-length for all three, indicating the events file was truncated during data curation rather than the recording being short. Fixed by adding a 280s fallback duration from the S 4 onset when S 11 is missing, giving all three subjects usable eyes-closed segments.

## Cohort Level Patterns

- Fp1/Fp2 systematically flagged. Around half of all subjects had at least one of Fp1 or Fp2 marked as bad. Not that surpriisng because these frontopolar positions sit above eye muscles and are directly affected by ocular artefacts. Handled by interpolation.

- Eye blink components are near-universal. Every processed subject except one (sub-44) had at least 1 eye blink component removed. Most had exactly 2. This consistency validates that ICA and ICLabel is reliably identifying and removing ocular activity.

- Line noise never appears as ICA component. Across 77 subjects, no ICA component was classified as line noise. So, know that the 50 Hz notch filter is doing its job upstream and stopping mains contamination from reaching the ICA stage.

- Perfect protocol adherence for standard subjects. All 74 subjects with intact events files show exactly 361.29 seconds of eyes-closed data (S 4 to S 11 gap). The recording protocol was executed consistently.

- Autoreject compensates for ICA variation. Subjects with unusually low ICA component removal (e.g., sub-44, sub-70) tend to have autoreject drop more epochs. The two-stage cleaning creates redundancy so when one stage under cleans, the other catches the residuals.

## Implications for Connectivity

- Slightly shortened segments for 3 subjects. Sub-19, sub-30, sub-34 contribute 70 epochs each vs 90 for other subjects. wPLI estimates will probably be slightly noisier for these three, but epoch count still exceeds Cai et al. (2020) adequacy threshold of 12 epochs.

- High autoreject-rejection subjects contribute fewer epochs. Sub-36 (64), sub-54 (61), sub-74 (72), sub-76 (56), sub-70 (76). These are all above 50-epoch adequacy but at the lower end of the cohort distribution.

- Consistent 125-node graph structure. Every subject has the same 125 electrodes with the same node identities. GNN training assumptions about fixed graph topology are met across the cohort.

- Preprocessing artefact patterns are handled. No systematic cohort-wide contamination remaining. Line noise removed by notch, eye blinks removed by ICA, muscle artefacts caught by ICA or autoreject, bad channels interpolated. OK to proceed with functional connectivity.

## Files produced

- `C:/data/pearl-neuro/derivatives/eeg_preprocessed/sub-XX/sub-XX_task-rest_desc-preprocessed_epo.fif` per subject (77 files, approximately 30-100 MB each)
- `C:/data/pearl-neuro/derivatives/eeg_preprocessed/sub-XX/preprocessing_log.txt` per subject (autoreject output log)
- `results/eeg_preprocessing_summary_full.csv` (79 rows: 77 success, 2 error)
- `results/eeg_preprocessing_summary_skeleton.csv` (5 rows: from earlier skeleton run)
