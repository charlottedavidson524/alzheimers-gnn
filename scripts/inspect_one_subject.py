"""
This script is created to inspect sub-01's resting EEG.

It loads the BrainVision data in MNE, checks that the standard 1005 montage is actually applied, and produces
some plots.

Outputs (find in results/eeg_inspection):
    - summary.txt: text findings log
    - raw_snippet.png: snippet of the raw signal across 16 channels
    - psd.png: power spectral density plot
    - montage.png: topomap of applied electrode positions

Run (from project root):
    - python scripts/inspect_one_subject.py
"""

import sys
from pathlib import Path
 
import matplotlib.pyplot as plt
import mne
import pandas as pd
 
from agnn.config import load_config

mne.viz.set_browser_backend("matplotlib")

# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────

cfg = load_config()
data_root = Path(cfg["paths"]["data_root"])
results_dir = Path(cfg["paths"]["results"])/"eeg_inspection"
results_dir.mkdir(parents=True, exist_ok=True)

# Set up paths
vhdr_path = data_root/"sub-01"/"eeg"/"sub-01_task-rest_eeg.vhdr"
events_path = data_root/"sub-01"/"eeg"/"sub-01_task-rest_events.tsv"

# Check vhdr path exists
if not vhdr_path.exists():
    print(f"Error: {vhdr_path} not found.")
    sys.exit(1)
 
log = []

# Messages to add to log
def show(message=""):
    print(message)
    log.append(message)

# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("Loading sub-01 rest EEG check")
show("=" * 70)

# preload = True loads all data into memory. This is acceptable for now because it;s only one subject
raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="error")
 
# Key info about the data
show(f"File: {vhdr_path}")
show(f"Sample rate: {raw.info['sfreq']} Hz")
show(f"Duration: {raw.times[-1]:.1f} seconds ({raw.times[-1]/60:.1f} min)")
show(f"Channels: {len(raw.ch_names)}")
show(f"Data shape: {raw.get_data().shape}")

show("")

# ──────────────────────────────────────────────────────────────────────
# Chwck for reference electrode

# 127 channels when expecting 128. Check that it's because of one being
# designated as a refernce electrode
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("Check for reference electrode")
show("=" * 70)

print("Fz" in raw.ch_names, "Cz" in raw.ch_names, "FCz" in raw.ch_names)

show("")

# ──────────────────────────────────────────────────────────────────────
# Channel names 

# Want to check that they look like the etxned 10-5 layout
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("Channel names")
show("=" * 70)

show(f"All {len(raw.ch_names)} channel names:")

for i in range(0, len(raw.ch_names), 8):
    show(" " + ", ".join(raw.ch_names[i:i+8]))

show("")

# ──────────────────────────────────────────────────────────────────────
# Standard_1005 montage

# Doing three kinds of verification here:
#   - Confirming the requested montage
#   - confirming that the right montage got applied to the data
#   - Saving a topomap for visual comparison against the paper's Figure
#      3
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("Standard 1005 montage")
show("=" * 70)

# Check the requested montage. the string below is the only place the montage identity is chosen. 
# Confirming this makes sure teh code isn't falling back to a different montage.
montage = mne.channels.make_standard_montage("standard_1005")

show(f"Requested montage: standard_1005")
show(f"Montage type: {type(montage).__name__}")
show(f"Number of montage channels: {len(montage.ch_names)}")

# match_case=False in case Brain Products capitalise differently from MNE's built-in montage names.
# Used on_missing="warn" so unmatched channels are logged, not raised.
try:
    raw.set_montage(montage, match_case=False, on_missing="warn", verbose="error")
 
    # Check what got applied to the data.
    show("Check applied montage:")
 
    positions = raw.get_montage().get_positions()
    show(f"Coordinate frame: {positions['coord_frame']}")
 
    ch_pos = positions["ch_pos"]
    n_with_pos = sum(1 for p in ch_pos.values() if p is not None)
    show(f"Channels with positions: {n_with_pos}/{len(raw.ch_names)}")
 
    # Cz's position under standard_1005 is roughly [0, 0, 0.093] metres. This is perfectly cenric laterally and anteroposterior, 
    #  and roughlky 93 mm above the head-centre origin.
    if "Cz" in ch_pos:
        cz = ch_pos["Cz"]
        show(f"Cz coordinates (should be roughly [0, 0, 0.093]): {cz}")
    else:
        show("Cz wasn't found in applied positions.")
 
    # Check if any channels didn't receive positions and if so which ones
    without_pos = [ch for ch, p in ch_pos.items() if p is None]
    if without_pos:
        show(f"Channels without positions: {without_pos}")
    else:
        show("All channels received positions.")

    # Save a topomap for visual comparison against the paper's Figure 3.
    fig = raw.get_montage().plot(show=False, kind="topomap", show_names=False)
    montage_path = results_dir/"montage.png"

    fig.savefig(montage_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    show(f"Saved montage plot to: {montage_path}")
 
except Exception as e:
    show(f"Montage application failed: {e}")

show("")

# ──────────────────────────────────────────────────────────────────────
# Events 

# Check which markers exist
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("Events")
show("=" * 70)

# From the .vmrk markers via MNE:
try:
    events_vmrk, event_id_vmrk = mne.events_from_annotations(raw, verbose="error")
    show(f"Events from .vmrk: {len(events_vmrk)}")
    show(f"Unique event codes: {event_id_vmrk}")
except Exception as e:
    show(f"Reading events from annotations failed: {e}")

# From the BIDS _events.tsv:
if events_path.exists():
    ev_df = pd.read_csv(events_path, sep="\t")
    show(f"\nEvents from _events.tsv: {len(ev_df)} rows")
    show(f"Columns: {list(ev_df.columns)}")
    if "trial_type" in ev_df.columns:
        show("Trial-type counts:")
        show(ev_df["trial_type"].value_counts().to_string())

show("")

# ──────────────────────────────────────────────────────────────────────
# What are in the other four rows of _events.tsv

# Only two rows have the trial_type of stimulus and there are six rows
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("_events.tsv row check")
show("=" * 70)

ev = pd.read_csv("C:/data/pearl-neuro/sub-01/eeg/sub-01_task-rest_events.tsv", sep="\t")
print(ev.to_string())

show("")

# ──────────────────────────────────────────────────────────────────────
# Raw signal snippet plot
# ──────────────────────────────────────────────────────────────────────

show("=" * 70)
show("Plots")
show("=" * 70)
 
# 20 seconds of the first 16 channels. This should be enough to eyeball for obvious problems without 
# creating an overwhelming plot
fig = raw.plot(duration=20, n_channels=16, scalings="auto", show=False, block=False)
snippet_path = results_dir/"raw_snippet.png"

fig.savefig(snippet_path, dpi=100, bbox_inches="tight")
plt.close(fig)

show(f"Saved: {snippet_path}")

# ──────────────────────────────────────────────────────────────────────
# Power spectral density plot
# ──────────────────────────────────────────────────────────────────────
# fmax=60 goes above 50 Hz mains to see whether notch will be needed.
fig = raw.compute_psd(fmax=60, verbose="error").plot(average=True, show=False)
psd_path = results_dir/"psd.png"

fig.savefig(psd_path, dpi=100, bbox_inches="tight")
plt.close(fig)

show(f"Saved: {psd_path}")

show("")

# ──────────────────────────────────────────────────────────────────────
# Save the text log
# ──────────────────────────────────────────────────────────────────────
summary_path = results_dir/"summary.txt"
summary_path.write_text("\n".join(log), encoding="utf-8")

print(f"Text summary saved to: {summary_path}")

