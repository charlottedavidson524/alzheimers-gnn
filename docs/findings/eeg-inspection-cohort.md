This corresponds to the `scripts/explore_eeg_cohort.py` code. It is to check that the key findings of `docs/findings/eeg_inspection_sub01` generalises across several subjects before committing to any preprocessing choices.

This extended the sub-01 inspection across 5 subjects. It checked whether:

- Recording metadata
- Event structure
- Spectral properties
- Channel quality

generalise. Analysis was kept to resting state EEG. Task recordings are on disk but out of scope for this pass.

Available files per subject:

- All five have the extneded BrainVision triplet (.eeg, .vhdr, .vmrk) that are under `sub-XX/eeg/` on the disk.
- A per-subject BIDS_events.tsv sidecar wasn't there for sub-02-05, only 01 had one. Initially thought this was missing data. The event info is stored in the .vmrk marker file that ships as part of the BrainVision triplet. Read directly by MNE when header ;loaded. Events pipeline was refactored to use `mne.events_from_annotations/raw` on the loaded Raw object rather than reading tsv sidecars. Should work uniformly across all subjects with no sidecar dependency.

| Recording metadata | Sample rate | Duration | Channels |
| ------------------ | ----------- | -------- | -------- |
| sub-01             | 1000Hz      | 661.5s   | 127      |
| sub-02             | 1000Hz      | 637.7s   | 127      |
| sub-03             | 1000Hz      | 670.8s   | 127      |
| sub-04             | 1000Hz      | 704.1s   | 127      |
| sub-05             | 1000Hz      | 644.9s   | 127      |

Findings that generalise from sub-01:

- 127 channels, not 128
- 1000 Hz sample rate
- Duration roughly 10-12 minutes. Spread is 66.4 seconds. Small, attribute to natural session variability.

Channel list consistency:

- All 5 subjects have an identical channel list (same 127 electrodes, names and orders)
- Implication: cross-subject GNN node alignment is trivial. Every subject's graph will have same 127 nodes in the same order. No channel re-indexing step needed when batching graphs for traininbg.

Event structure:

- All five subjects show the identical 6-event structure predicted by paper's table 5 and confirmed by sub-01.
- S 1 = block boundary/response marker
- S 2 = eyes-open condition onset
- S 4 = eyes-closed condition onset
- S 10 = eyes-closed instruction
- S 11 = end of task and sound effect

MNE's `events_from_annotations` returns these prefixed with `Stimulus/` rather than as raw strings. Normal MNE behaviour from BrainVision annotations.

Two block eyes-open then eyes-closed protocl confirmed for all 5 subjects. All subjects S 4 (eyes-closed onset) happened roughly in the middle of recording. consistent with roughly 4 minutes eyes open and roughly 6 minutes eyes closed.

Extraction of eyes-closed segments:

- for each subject, 60s was extracted from the middle bof the eyes closed block (between S 4 start and S 11 end) which avoided onset/offset transients.

| Subject | Start (s) | Stop (s) |
| ------- | --------- | -------- |
| sub-01  | 300.8     | 360.8    |
| sub-02  | 288.9     | 348.9    |
| sub-03  | 305.4     | 365.4    |
| sub-04  | 322.0     | 382.0    |
| sub-05  | 292.4     | 352.4    |

Start times cluster tightly (289-322s) consistent with an experiment where protocol is fixed and S 4 occurs at a similar point across all the sessions.

Spectral inspection (PSD):

- All 5 subjects show clear alpha peaks in the expected 7-13Hz range, averaged across all channels

| Subject | Alpha peak (Hz) | Peak power (dB) |
| ------- | --------------- | --------------- |
| sub-01  | 8.30            | -105.8          |
| sub-02  | 9.77            | -115.5          |
| sub-03  | 9.77            | -106.5          |
| sub-04  | 10.25           | -112.1          |
| sub-05  | 8.79            | -106.8          |

Findings that generalise from sub-01:

- Prominent alpha peak in every subject
- 50Hz mains contamination is visible in the PSD plot
- 1/f falloff. Classic spectral shape across all 5. no broadband elevation euggestive of pervasive muscle contamination. No dropouts.
- NOTE: sub-02 (-115.5dB) is lower than the other 4. could be a lower amplitude alpha rhythm in the subject, a specific electrode.impedence issue with just them, or averaging over 17 flagged channels (see LOF below)

Amplitude and variance check:

| Subject | Median channel std | Maximum channel std |
| ------- | ------------------ | ------------------- |
| sub-01  | 37.3               | 104.2               |
| sub-02  | 22.7               | 89.0                |
| sub-03  | 39.4               | 186.7               |
| sub-04  | 27.3               | 123.8               |
| sub-05  | 55.0               | 189.4               |

The median of the medians was 37.3 muV. No subjects exceed the 3 times threshold. Theyre all within a reasonable range. But for max channel std, 03 and 05 both have single channels with roughly 190 muV standard deviation (several times cohort median). Consistent with a few noisy channels per subject. see LOF below.

Bad channel screening:

- Automatic bad channel detection via LOF (`mne.preprocessing.find_bad_channels_lof`) flagged bad channels for each subject.

| Subject | Bad channels flagged |
| ------- | -------------------- |
| sub-01  | 5                    |
| sub-02  | 17                   |
| sub-03  | 7                    |
| sub-04  | 2                    |
| sub-05  | 6                    |

The median is 6 bad channels per subject. Sub-02 is an outlier with 17.

Systematically bad channels (more than 2 subjects):

- PPO10h (2)
- POO10h (2)
- TP9 (2)
- AF7 (2)
- P4 (2)

4 of these 5 (all but P4) sit on the outer ring. This is closest to the scalp edge. Could be movement artefacts, muscle contamination from facial/neck muscles, etc. More about cap fit and skin contact.

Note on sub-02: 17/127 flagged. edge of what can be interpolated. Check the literature for what should constitute candidtae exclusion based on number of bad channels.

Findings that generalise:

- 127 channels per subject, same names, same order
- 1000 Hz sample rate
- 10-12 minute recordings
- 6 event 2 block eyes open/eyes closed protocol. S 4 marks the eyees closed onset.
- BrainVision .vmrk-based event annotations available for every subset (BIDS .tsv sidecars not required)
- prominent alpha peak in every subject.
- IAF in normal adult range
- 50Hz mains contamination (may need notch filter)
- Small number of bad channels per subject ( less than 10 usually). outer ring electrodes are overrepresented

What doesn't generalise:

- Bad channel count varies a lot (2-17)
- Sub-02 shows lower alpha peak power

Implications for preprocessing:

- Apply 50Hz notch filter for mains removal
- Apply 1-45Hz bandpass
- rereference
- Handle artefacts (ICA and/or epoch rejection)
- Restrict analysis to condition segments, using eyes closed block extracted via S 4 -> S 11 markers
- Interpolate bad channels, don't drop (just an instinct at this point but validate this with literature)
- Consider per-subject bad-channel threshold.
