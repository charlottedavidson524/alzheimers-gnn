"""
This is a test file for running the full preprocessing pipeline on sub-01. Prints the returned status dict.

Findings are documented in `docs/findings/eeg_preprocessing/sub01-only.md`
"""

from pathlib import Path
from agnn.config import load_config
from agnn.preprocessing.eeg import preprocess_subject

config = load_config()

result = preprocess_subject(
    subject_id="sub-01",
    config=config,
    data_root=Path("C:/data/pearl-neuro"),
    output_root=Path("C:/data/pearl-neuro/derivatives/eeg_preprocessed"),
)

for key, value in result.items():
    print(f"{key}: {value}")