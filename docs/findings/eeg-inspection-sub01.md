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
