# Bad channels

There are several facets of bad channels that need to be dealt with. They need to be detected, then excluded if there are too many of them, and interpolated if there aren't too many to avoid loss of participants where possible. A bad channel could mean a few different things:

- Flat: signal is almost constant -> electrode fell off, amplifier disconnected
- High amplitude: signal is dominated by noise (poor contact or a movement artefact, for example)
- Low correlation: Signal doesn't correlate with neighbours (localised noise)
- Line noise dominated: signal is dominated by 50Hz mains instead of brain activity
- Statistical outliers: signal statistics difer from the norm of the cohort

Different algorithms would detct different types of 'bad' channels.

## Detection

Chose LOF based detection with threshold 1.5.

Also considered FASTER (Nolan et al., 2010) and PREP (Bigdely-Shamlo et al. 2015), clean_rawdata and manual visual inspection.

LOF was chosen (Kumaravel et al. 2022) it's the algorithm underlying MNE's `find_bad_channels_lof` implementation. It has 87.5%average F1-score imrpovement over benchmarks (FASTER's correlation, PREP's RANSAC, and NDR) on adult EEG data. LOF's density-based approach identifies channels that differ from their local neighbourhood rather than from a global distribution, making it more robust to cohort-level variation. Didnt choose visual inspection because:

- It's a specialised clinical research skill. Kumaravel comparison also uses expert visual inspection as ground truth -> LOF still better performs than automated methods calibrated against that ground truth by 87.5% F1.
- Inter-rater and intra-rater variability can make it less reproducible. The same person can look at the same thing at 2pm and 10pm and have different results. Different people might also have different views.

Kumaravel et al. (2022) recommend a threshold of 1.5 for adult EEG based on F1 evaluation across sixteen datasets. This matches MNE's default. Cohort EDA on sub01-05 (bad channel counts of 2–17 out of 127) showed sensible detection at this threshold. This supports its calibration for the PEARL-Neuro data.

## Exclusion

Decided to do subject exclusion when more than 20% of channels are flagged as bad. |This is chosen as a round number that can be supported. More than 20% and spherical spline interpolation quality drops. This is becayse interpolated channels start to rely more on other interpolated channels as tgheir neighbours, and problems with cap fit or quality of the gel severe enough to affect more than 25 channels will probably affect the rest of the channels too.

This is a tradeoff favouring sample size given n=79 and the subtle biological signal being classified (genetic risk, not disease). Per-subject bad channel counts will be reported alongside classification results so this trade-off can be evaluated afterwards. All 5 subjects in the skeleton cohort had less tgan 14% bad channels so no exclusions were needed at this stage.

## Interpolation

Chose spherical spline interpolation via MNE's default implementation (Perrin et al. 1989). This is the standard across FASTER, PREP, HAPPE, and Automagic pipelines. MNE's `raw.interpolate_bads()` uses the Perrin method. Interpolation (rather than deleting) is a requirement. Cross-subject GNN batching needs identical node counts and node ordering across all of the subjects.

Dropping bad channels would result in only being able to use the intersectrion of common channels across all of the participants, which at n=79 with per subject bad channel counts that vary quite a bit would massively reduce the effective node count below 127. This would cause a loss in the spatial coverage that the 128 electrode set up was supposed to provide.

## Limitation

Spherical spline interpolation has been shown to distort local functional connectivity. Kang et al. (2015) showed that interpolation inflates average within-cluster coherence by roughly 21% between interpolated channels and their surrounding neighbours. hese effects were strongest at short inter-electrode distances and got smaller with distance from interpolated channels. This inflation happens because interpolated signals are derived as a weighted sum of neighbour signals. This produced almost zero lag correlations with those neighbours.

## Mitigation

Must use wPLI (Vinck et al. 2011) as connectivity measure. This weights each phase-difference sample by the magnitude of its imaginary component. It results in down weighting of contributions from phase relationships near 0 or 180 degrees. Vinck et al. show that this makes wPLI robust agisnt added common noise sources. They dont say it directly but it can be inferred that this would include the interpolation artefact case. So wPLi is a mitigation for the artefact Kang identified. Interpolation metadata for each subject should be recorded. This means downstream results can be checked for correlation with interpolation.

Come back to bad channels if:

- Cohort stats show unusual patterns like many bad channels. or bad channels concentrating at specific electrodes. In this case tune the LOF threshold.
