In this file, one subject's EEG data was downloaded and inspected. sub-01's resting state EEG in MNE-python is the focus. This corresponds to the stage 2 download detailed in `docs/download-script.md`

The prupose of this stage/file is to check that the EEG data downloads properly, and also to test the pipeline on one subject before bringing in more data.

To do this, stage 2 of `scripts/download_data.py` was updated to also pull back the full file set for sub-01's resting state EEG. This was done by checking the OpenNeuro file listing to be sure of what exists. What was pulled back is as follows:

| File                          | Status               | Notes                     |
| ----------------------------- | -------------------- | ------------------------- |
| `sub-01_task-rest_eeg.eeg`    | Downloaded (320.5MB) | BrainVision binary data   |
| `sub-01_task-rest_eeg.vhdr`   | Downloaded           | BrainVision header        |
| `sub-01_task-rest_eeg.vmrk`   | Downloaded           | BrainVision event markers |
| `sub-01_task-rest_events.tsv` | Downloaded           | BIDS event descriptions   |

The dataset doesn't include optional BIDS sidecar files that document recording metadata or electrode positions. Common BIDS conventions that aren't in this release:

- \_eeg.json -> recording metadata sidecar
- \_electrodes.tsv -> per-subject electrode positions
- \_coordsystem.json -> coordinate system for the electrodes
- \_channels.tsv -> per-channel metadata

So, per-subject electrode coordinate files aren't provided (see `docs/glossary/eeg-electrode-layout.md`)

The implications of this are as follows:

- Electrode positions must use MNE's generic standard_1005 montage. No per-subject co-ordinates seem to be available. These are idealised electrode positions rather than measured, highly accurate ones. A small hit to accuracy but probably unavoidable.
- Recording metadata (sample rate, filter settings, etc) must be read from the .vhdr file directly, not from a JSON sidecar. MNE handles this when loading BrainVision format.
- Event info is available from both .vmrk markers and the BIDS \_events.tsv. MNE should be able to read either.

Data loading:

Data was loaded via `mne.io.read_raw_brainvision(vhdr_path, preload = True)`

- Sample rate: 1000 Hz
- Duration: 661.5s
- Channels (127, not 128 like expected)
- Data shape: (127, 661520)

One key finding is that FCz is the online reference electrode. The 128 electrodes recorded in the paper was the physical hardware count. A line of code was done to check whcih data channel was missing and it was found to be FCz. Fz and Cz (FCz's neighbours on the frontal midline) are both there. Only FCz is missing, so it can be inferred that it was used as the online refernce electrode. Check whether this is the standard Brain Products antiCAP deafult.

The implications of this are as follows:

- Brain graphs will have 127 nodes, not 128.
- A re-referencing decision will be needed during pre-processing. Such as, common average reference (standard for high-density EEG), linked mastoids, or REST reference. Document this choice in a `docs/decisions/` choice when made.
- The signal at every recorded electrode is currently relative to FCz's signal at the same time point. Re-referencing changes this baseline. Downstream analyses assume the reference is what you say it is.
