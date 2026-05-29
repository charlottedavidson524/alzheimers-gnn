# PEARL-Neuro Download (ds004796, v1.1.0)

Data lives outside the repo at C:/data/pearl-neuro (see config/default.yaml).

## 0. Confirm DataLad is installed

- `datalad --version`

## 1. Clone the dataset structure

Small and fast because it's metadata only.

- `datalad clone https://github.com/OpenNeuroDatasets/ds004796.git C:/data/pearl-neuro`
- `cd C:/data/pearl-neuro`

## 2. Get the participants table

This is small so want to do a feasability test before any big downloads.

- `datalad get participants.tsv participants.json`

## 3. Pull data selectively

Only doing what is needed here rather than trying to download 100s of GBs at once.

One subject's EEG to test the pipeline would be:

- `datalad get sub-01/eeg/`

EEG for every participant would be:

- `datalad get */eeg/`

## 4. Fallback if DataLad doesn't work on Windows

Use the OpenNeuro download page (generates an AWS S3/ Node script):

- `https://openneuro.org/datasets/ds004796/versions/1.1.0/download`
