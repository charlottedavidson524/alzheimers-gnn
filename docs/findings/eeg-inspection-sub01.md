The point of this file is to describe what was learned by inspecting sub-01's resting state EEG in MNE-python. This was done before writing the preprocessing pipeline.

Files downloaded:

- OpenNeuro file listing was checked directly to see what files exist under sub-01/eeg/. This matched what was downloaded.

| File                        | Status     | Notes                     |
| --------------------------- | ---------- | ------------------------- |
| sub-01_task-rest_eeg.eeg    | Downloaded | BrainVision binary data   |
| sub-01_task-rest_eeg.vhdr   | Downloaded | BrainVision header        |
| sub-01_task-rest_eeg.vmrk   | Downloaded | BrainVision event markers |
| sub-01_task-rest_events.tsv | Downloaded | BIDS event descriptors    |

The standard BIDS sidecars were absent, but CapTrak coordinates exist elsewhere.

The following optional BIDS sidecar files that document recording metadata arent provided under `sub-XX/eeg/` in the public release.

- \_eeg.json: recording metadata sidecar
- \_electrodes.tsv: per-subject electrode positions
- \_coordsystem.json: coord system for electrodes
- \_channels.tsv: per channel metadata

Per subject 3D electrode coordinates do exist, they're just not in BIDS standard location. Implications:

- Recording metadata is read from .vhdr file directly
- Event info is available from both .vmrk markers and BIDS \_events.tsv.
- Bad channel decisions need to be made from the data iytself during preprocessing (no \_channels.tsv flags)

Data loading:

- Loaded via `mne.io.read_raw_brainvision(vhdr_path, preload=True)`.
  - Sample rate: 1000Hz (matches paper)
  - Duration: 661.5s (roughly 11 mins)
  - Channels: 127, not 128
  - Data shape: 127, 661520

- Confirmed recording settings from paper:
  - Amplifier: actiCHamp
  - Cap: high senisty acticap, 128 electrode configuration
  - Sample rate: 1000Hz
  - Online reference: FCz
  - Online filters during recording: low-pass at 280Hz only
  - Impedence: 5-10 ohms, maintained by skin abrasion and gel application.

FCz is the onlibe reference electrode:

- The paper's 128 electrodes is just the physical hardwaRE COUNT. Data file contains 127 channels and FCz is absent. Fz and Cz (FCz neighbours) are both present and FCz ius reference. This is also notes in the paper.

- Implications:
  - Brain graphs are 127 nodes
  - A re-referncing decision is needed during preprocessing
  - Signal at every recorded electrode is currently relative to FCz signal at the same time point. Rereferencing will change this.

Channel naming:

- Follows extended 10-5 system.
- Standard 10-10 backbone el;ectrodes (Fz, Cz, Pz, O1, etc)
- High desnity h suffic half distance electrodes (e.g FFC5h) making this a 128 channel/high density montage
- Outer ring low electrodes (F9/F10, FT9/FT10) etc. See outer circle in Fig 3 PEARL-Neuro.
- No non EEG channels (no ECG, EOG, etc)
- All 127 recorded channels are scalp electrodes

Montage = `standard_1005`

- Applied MNE's standard_1005 montage and used 3 independent checks. All 3 were passed.

1. Requested montage. Requested: standard_1005. Type: DigMontage. Montage defibes 343 possible channel positions. Recording uses 127 of them as a valid subset.
2. Applied positions. Co-ordinate frame: head. Channels with positions: 127/127. Co-ordinate check: [-0.00137, 0.02762, 0.14020]. this is midline, slightly anterior and near the top of the head. Consistent with vertex electrode. No channels without positions.
3. Visual check. The saved topomap (`results/eeg_inspection/montage.png`) matches Figure 3 of the paper. Two concentric electrode rings, central dense grid and expected overall desnity.

So, a generic MNE `standard_1005` montage matches all 127 recorded channels cleanly. Valid first pass choice. Whether to switch to per-subject CapTrak co-ordinates is a decision that will need to be made later depending on the first pass.

Events:

- There are 6 events across the roughly 11 minute recording, agreeing berween .vmrk and \_events.tsv sources.
- Two block eyes-open then eyes-closed resting state protocol is clear.
- \_events.tsv have an obvious two block structure. Sorted the table below by the onset of the events.

| Time (s) | Event         | Intyerpretation                |
| -------- | ------------- | ------------------------------ |
| 23.3     | S1            | Block boundary/response marker |
| 23.3     | S2 (stimulus) | Eyes-open consition onset      |
| 264.7    | S10           | Eyes-closed instruction        |
| 276.1    | S1            | Block boundary/response-marker |
| 276.1    | S4 (stimulus) | Eyes closed condition onset    |
| 637.4    | S11           | End of task and sound effect   |

Interpretations are confirmed by the paper. There are two blocks. Roughly 4 mins of eyes open (Condition A) followed by roughly 6 minutes of eyes closed (Condition B)
. they are separated by a roughly 11 second gap. the total usable recording is around 10.2 minutes. This is confirmed by the paper's resting state session description. Table 5 of the paper documents event codes:

- S 2 = "eyes open condition start"
- S 4 = "eyes closed condition start"
- S 10 = "eyes closed instruction"
- S 11 = "end of task and sound effect"

The stimulus effects (S 2 and S 4) are things presented to the participant (eyes open/eyes closed instructions). Non-stimulus events (S 1, S 10, S 11) are protocol/recording markers with `trial_type` = NaN. This expected BIDS behaviour rather than missing data. Implications:

- Usable segments are 23.3-264.7s (eyes open/4 mins) and 276.1-637.4s (eyes closed/6 minutes). discard silences before, between and after.

- Need to decide whether to analyse conditions separately, concatenated, or to use eyes closed only.

- Eyes closed condition gives strongest alpha signal (see below) so might be preferred.

Signal quality (raw snippet)

- Saved to `results/eeg-inspection/raw-snippet.png` (first 20s, 16 channels)
- Clean baseline with periodic muscle artefacts.
- Two interesting features: a burst of high amplitude, high frequency activity around 8-12s appearing across many channels simultaneously. This is most prominent in temporal channels (F7, FT9, T7, TP9). Its a sign of muscle artefact/EMG. Could be jaw or neck tension (Goncharova et al., 2003). Temporalis muscle sits near these electrodes. Also there are larger deflections in Fp1, particularly near the start of the snippet. Consistent with eye blinks (standard frontal electrode artefact from eyelid movement) (Jung et al., 2000)
- Preprocessing implications: Need to do artefact handling. There are two options: 1 -> ICA based artefact rejection. 2 -> Epoch level rejection. Could also do a hybrid, this is common practice.

Signal quality (power spectral density):

- Plot is saved in `results/eeg-inspection/psd.png`. Averaged across all channels, 0-60Hz, log scale.
- Found that there were three features, all as expected for a healthy-resting state EEG. Prominent alpha peak at roughly 8-9Hz. Individual alpha frequency (IAF) sits within normal adult range (7-13Hz). Peak is obvious, participant genuinely reached resting brain state. Sharp 50Hz mains spike (mains line contamination expected for a Polish recording). Clean 1/f fall off between peaks. Power decreases smoothly with frequency -> healthy EEG spectra.
- Pre-processing implications: 50Hz notch filter probably needed. Bandpass 1-45Hz is likely appropriate. Alpha rhythm is prominent and usable.

So, sub-01 is a viable participant.

Findings that likely generalise beyond sub-01 (run checks on 4 other subjects to verify this):

- There are files available per subject per task (.eeg, .vhdr, .vmrk, \_events.tsv) are in `sub-XX/eeg/`. CapTrak files for 77 of 79 participants.
- FCz reference electrode. Same hardware for whole cohort.
- 10-5 channel naming convention and `standard_1005` montage match up.
- two-block eyes open (4 minutes) and then eyes closed (6 minutes)
- actiCHamp amplifier, 1000Hz sample rate, 280Hz low-pass online filter, 5-10 kilo-ohms impedence should be the same for all subjects.

Reverify all of thee above for 4 more subjects after the stage 3 downloads. There may be some people with unusual characteristics. Want to check that sub-01 is representtaive.
