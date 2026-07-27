"""
Download PEARL-Neuro dataset files from OpenNeuro. Doing this using direct HTTPS.

Dataset: ds004796 (PEARL-Neuro), version 1.1.0
Source: https://openneuro.org/datasets/ds004796/versions/1.1.0
Reference: Dzianok, P. and Kublik, E. (2025) 'A Polish Electroencephalography, Alzheimer's Risk-genes, 
           Lifestyle and Neuroimaging (PEARL-Neuro) Database', OpenNeuro. doi:10.18112/openneuro.ds004796.v1.1.0.

USAGE
-----
Run from the project root:
    - python scripts/download_data.py --stage 1

Stages (these will be ran one by one to test data loading and methodology)
    1) Metadata only (participants.tsv, participants.json, dataset_description.json). For EDA.
    2) One subject's EEG triplet (sub-01, resting state). Done to confirm BIDS paths 
       and also that MNE can read the files.
    3) First five subjects EEG (sub-01..sub-05, all tasks). Used to develop the pipeline
       and methodology on small, balanced dataset.
    4) Full EEG for all 79 neuroimaging subjects. Will likely have to run overnight

Files already on the disk are skipped. Missing files (404s as some are missing in the public release) are
noted and skipped rather than treated as an error.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from agnn.config import load_config

# ──────────────────────────────────────────────────────────────────────
# OpenNeuro public S3 endpoint for this dataset.
# <BASE_URL>/<path-inside-dataset>
# ──────────────────────────────────────────────────────────────────────
BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds004796"

# Subjects to fetch in stage 3 (skeleton phase).
SKELETON_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 6)]  # First five subjects

# All 79 subjects (that have EEG/fMRI) with neuroimaging data (stage 4).
FULL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 80)] 

# EEG file extensions in the BrainVision format (data, header and markers).
EEG_EXTS = [".eeg", ".vhdr", ".vmrk"]

# EEG tasks recorded in PEARL-Neuro.
EEG_TASKS = ["rest", "msit", "sternberg"]

# BIDS sidecars (different naming pattern from BrainVision files).
EEG_SIDECARS = ["_events.tsv"]

# ──────────────────────────────────────────────────────────────────────
# Core download 
# ──────────────────────────────────────────────────────────────────────
def download_one(rel_path: str, data_root: Path) -> str:
    """
    Download a single file by its path inside the dataset.

    Returns a status string: 'ok', 'skipped', 'missing', or 'error'. Uses a .tmp file during transfer so 
    partial downloads are obvious and never look like complete files.
    """
    url = f"{BASE_URL}/{rel_path}"
    out = data_root / rel_path
    tmp = out.with_suffix(out.suffix + ".tmp")

    if out.exists() and out.stat().st_size > 0:
        print(f"skip {rel_path}")
        return "skipped"

    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(out)
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"ok {rel_path}  ({size_mb:.1f} MB)")
        return "ok"

    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        if e.code == 404:
            print(f"404 {rel_path} (not in public release)")
            return "missing"
        print(f"ERROR  {rel_path}  HTTP {e.code}")
        return "error"

    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"ERROR {rel_path} {type(e).__name__}: {e}")
        return "error"

# ──────────────────────────────────────────────────────────────────────
# File-list builders for each stage
# ──────────────────────────────────────────────────────────────────────
def files_stage_1() -> list[str]:
    """Top-level metadata only (tiny. run first to validate setup)."""
    return ["participants.tsv", "participants.json", "dataset_description.json", "README", "CHANGES"]

def files_stage_2() -> list[str]:
    """One subject's resting-state EEG + event descriptions (sub-01)."""
    base = "sub-01/eeg/sub-01_task-rest_eeg"
    files = [f"{base}{ext}" for ext in EEG_EXTS]
    files.append("sub-01/eeg/sub-01_task-rest_events.tsv")
    return files


def files_stage_3() -> list[str]:
    """Skeleton subjects, all EEG tasks, plus events sidecars."""
    eeg_files = [
        f"{sub}/eeg/{sub}_task-{task}_eeg{ext}"
        for sub in SKELETON_SUBJECTS
        for task in EEG_TASKS
        for ext in EEG_EXTS
    ]
    events_files = [
        f"{sub}/eeg/{sub}_task-{task}_events.tsv"
        for sub in SKELETON_SUBJECTS
        for task in EEG_TASKS
    ]
    return eeg_files + events_files


def files_stage_4() -> list[str]:
    """All 79 subjects, all EEG tasks, plus events sidecars."""
    eeg_files = [
        f"{sub}/eeg/{sub}_task-{task}_eeg{ext}"
        for sub in FULL_SUBJECTS
        for task in EEG_TASKS
        for ext in EEG_EXTS
    ]
    events_files = [
        f"{sub}/eeg/{sub}_task-{task}_events.tsv"
        for sub in FULL_SUBJECTS
        for task in EEG_TASKS
    ]
    return eeg_files + events_files


STAGES = {
    1: ("Metadata only", files_stage_1),
    2: ("One subject EEG (sub-01 rest)", files_stage_2),
    3: (f"Skeleton EEG ({len(SKELETON_SUBJECTS)} subjects)", files_stage_3),
    4: (f"Full EEG ({len(FULL_SUBJECTS)} subjects)", files_stage_4),
}


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--stage",
        type=int,
        choices=sorted(STAGES.keys()),
        required=True,
        help="Which download stage to run (1=metadata, 2=test subject, 3=skeleton, 4=full).",
    )
    args = parser.parse_args()

    cfg = load_config()
    data_root = Path(cfg["paths"]["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)

    label, builder = STAGES[args.stage]
    files = builder()

    print(f"\nPEARL-Neuro download: stage {args.stage}: {label}")
    print(f"Destination: {data_root}")
    print(f"Files in this stage: {len(files)}\n")

    counts = {"ok": 0, "skipped": 0, "missing": 0, "error": 0}
    for rel in files:
        counts[download_one(rel, data_root)] += 1

    print("\nSummary: ")
    for k, v in counts.items():
        print(f"  {k:8s} {v}")

    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())