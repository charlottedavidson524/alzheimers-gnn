"""
Quick test: build graphs for sub-06 end-to-end.

Loads config, runs the full graph-construction pipeline on sub-06's preprocessed epochs and prints the returned status dictionary.

Sub-06 chosen because it has typical bad-channel and ICA counts.

Run:
    - python tests/test_build_graphs_sub06.py
"""

from pathlib import Path
from agnn.config import load_config
from agnn.connectivity.build_graphs import build_subject_graphs

# Load project config and derive input and output paths
config = load_config()
data_root = Path(config["paths"]["data_root"])
preprocessed_root = data_root/"derivatives"/"eeg_preprocessed"
output_root = data_root/"derivatives"/"graphs"

# Run the pipeline on sub-06. apoe_label=None because this is just a pipeline check not a training run
result = build_subject_graphs(subject_id="sub-06", config=config, preprocessed_root=preprocessed_root, output_root=output_root, apoe_label=None)

# Print the returned status dictionary. Separate line by line so easy to read
for key, value in result.items():
    print(f"{key}: {value}")