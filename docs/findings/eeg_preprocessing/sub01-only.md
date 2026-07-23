The way I did the EEG pre-processing is as follows:

- Create a function that will preprocess a single subject according the the research/documentation. test on sub-01. Sort of sanity check to be sure that preprocessing is all correct and makes sense

- Create a batch runner function that will preprocess a batch. Also create a driver script that will call this function.

- Test for the 5 subjects on disk after Stage 3 download to be sure driver script/batch runner work.

- Download all n=79 subject's EEG data and batch preprocess. save output.

This file documents the process of the first stage.

The runtime was approx 388 seconds or six and a half minutes.

Key numbers:

- 125 channels (127 recorded, 2 dropped: O9, O10 lack standard_1005 positions)
- 3 bad channels flagged by LOF (Fp1, Fp2, AF8). These wereall frontal which is consistent with eye artefacts
- 121 ICA components fit (effective rank after CAR + interpolation)
- 8 ICA components removed: 4 muscle artifact, 2 eye blink, 2 channel noise
- 90 epochs before autoreject, 89 after (1 rejected, 1.1% rejection rate)
- Eyes-closed duration: 361.3 s (S 4 -> S 11 markers)

Then did a visual sanity check using two graphs (were popups but can find screenshots on disk):

PSD (across all epochs and channels):

- Prominent alpha peak at around 8-9 Hz preserved
- Smooth 1/f falloff
- 50 Hz mains spike successfully removed
- Small residual beta-band bump in some channels. this could be lateral muscle activity ICA didn't fully catch, but is outside AD-relevant bands

Time-series (5 epochs, all channels):

- No large frontal deflections (Fp1 eye blinks removed)
- No high-amplitude bursts in temporal channels (F7, T7, FT9, TP9 muscle artefacts removed)
- Consistent amplitude across all channels

Both confirm the artefacts identified in the raw inspection (`eeg-inspection-sub01.md`) have been cleanly removed.

There were a few bugs identified and fixed

- O9, O10 lacking `standard_1005` positions. MNE's template has NaN positions for these outer-ring occipital channels. This breaks spherical spline interpolation. Fixed by dropping consistently across cohort, giving 125-node graphs.

- ICLabel category name mismatch. Config had `"muscle"` but ICLabel returns `"muscle artifact"`. Easy enough to fix by just updating the config

- `mne.compute_rank(rank="info")` was unreliable. Info-based rank estimation returned 125 after CAR and interpolation. Actual data rank (using `numpy.linalg.matrix_rank`) is 121. Fitting extra components produced 4 ghost components that contaminated ICA decomposition. Fixed by using `numpy.linalg.matrix_rank(raw.get_data())`

There are some implications for the whole cohort:

- 125-node graph structure is fixed and consistent across all subjects
- Runtime of around 6.5 minutes per subject implies roughly 8.5 hours for full n=79 batch. Should run this overnight to make best use of time
- Autoreject rejection rate of around 1% on a clean subject suggests preprocessing isnt too aggressive
- Might be variation in the exact bad-channels and ICA counts across subjects. Pipeline is set up to handle this.

After batch runs:

- Should do cohort level statistics on ICA counts, epoch rejection rates and bad-channel patterns
- Investigate any subjects flagged with unusual status
