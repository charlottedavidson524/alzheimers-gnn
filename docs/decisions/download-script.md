Notes on `scripts/download_data.py` and the approach to the download.

The script downloads PEARL-Neuro files from OpenNeuro's public S3 endpoint via direct HTTPS. Only uses the Python
standard library so no DataLad or AWS CLI.

This is because it involves no new tooling (referenced in `docs/decisions/data-access.md`). It also creates a
reporducible record where the exact files retriveed are listed in a version controlled source rather than lost
in the shell history.

Key design changes are as follows:

- Staged downloads (see --stage argument). There are four stages of download and each increases in size. Start with metadata only -> one test subject to be sure data download is working -> five subjects to test the methodology and determine the pipeline -> full dataset. This lets the approaches taken be validated using small datasets before committing to the whole dataset.
- It is resumable. Files already on the disk are skipped. Downloads use a .tmp file name until they're complete, so partial files never get mixed up for complete ones.
- It is 404 tolerant. The data descriptor makes a note that some files are missing from the public release so the script is designed to log missing files and continue instead of terminating.
- The file lists are editable. Subject lists, EEG tasks, and file extensions are constants at the top of the script. If the scope is expanded (for example if fMRI is added) only a builder function will need to be added rather than needing to rewrite the script.
- Scope is kept to EEG for now. The second through fourth stages are only fetching BrainVision EEG triplet (.eeg/.vhdr/.vmrk). fMRI and per-subject metadata are deferred until they're needed.

Stage 1 ran cleanly and pulled five metadata files (around 150KB) to C:/data/pearl-neuro/

- participants.tsv -> full participant table (192 rows)
- participants.json -> column descriptions for the participant table
- dataset_description.json -> dataset version and authorship
- README, CHANGES -> dataset-level notes

These will be used for EDA and to build tabular baselines without needing any heavy downloads.

Further stages:

- Stage 2: one subject's resting-state EEG ( roughly 500 MB). Confirms confirm BIDS paths and that MNE can read the files.
- Stage 3: proof-of-concept subjects (currently sub-01 to sub-05). All EEG tasks for end-to-end pipeline development.
- Stage 4: all 79 neuroimaging subjects with all EEG tasks. Run once proof-of-concept is verified.
