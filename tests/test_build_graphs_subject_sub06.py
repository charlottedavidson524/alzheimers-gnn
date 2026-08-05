"""
This is a manual test building subject level aggregated graphs for sub-06.

Run from the project root:
    - python tests/test_build_graphs_subject_sub06.py
"""

from pathlib import Path
from agnn.config import load_config
from agnn.connectivity.build_graphs_subject import build_aggregated_graphs

# Load the config and resolve the paths needed for sub-06 graph build
config = load_config()
data_root = Path(config["paths"]["data_root"])
preprocessed_root = data_root/"derivatives"/"eeg_preprocessed"
output_root = data_root/"derivatives"/"graphs_subject"

# Build subject level graphs for sub-06. This is just a smooke test.
result = build_aggregated_graphs(subject_id="sub-06", config=config, preprocessed_root=preprocessed_root, output_root=output_root, 
                                 apoe_label=None)

# Print the status dictionary. One field per line for easy inspection
for key, value in result.items():
    print(f" {key}: {value}")