"""
EDA for the first five subject's resting EEG.

Purpose
-------
This code is to verify that the results from the sub-01 inspection (see `docs/findings/eeg-inspection-sub01.md`)
generalise to a wider cohort. The code will answer the following questions:

    - Do all subjects have the same 127 channels in the same order?
    - Does the recording metadata have consistent sample rates and durations?
    - Do all subjects show the 6 event two block eyes-open/eyes-closed protocol?
    - Do all subjects show a clear alpha spike and 50Hz mains spike?
    - Is any subject dramatically out of amplitude or variance range?
    - How many bad channels are there per subject and how many are consistently bad?

Analysis is done using the resting state EEG data only. Task recordings (MSIT, Sternberg) are on disk but aren't 
analysed.

Outputs
-------
All saved under `results/eeg_cohort_eda/`:

    - channel_consistency.csv -> per-subject channel-list summary    
    - metadata_summary.csv -> sample rates and durations
    - event_summary.csv -> event codes and times per subject
    - psd_comparison.png -> overlaid PSDs 
    - amplitude_summary.csv -> per-subject amplitude statistics
    - bad_channels_per_subject.csv -> bad-channel counts per subject
    - bad_channels_per_electrode.csv -> how often each electrode is bad
    - summary.txt -> plain-text log of everything printed

Run from: 
    - python scripts/explore_eeg_cohort.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from agnn.config import load_config

# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────
cfg = load_config()
data_root = Path(cfg["paths"]["data_root"])
results_dir = Path(cfg["paths"]["results"])/"eeg_cohort_eda"
results_dir.mkdir(parents=True, exist_ok=True)
 
# The first five subjects (downloaded in Stage 3)
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 6)]  
 
# Expected event codes 
EXPECTED_EVENT_CODES = {"Stimulus/S  1", "Stimulus/S  2", "Stimulus/S  4", "Stimulus/S 10", "Stimulus/S 11"}
  
# Buffer for the summary log.
log_lines: list[str] = []

# Prints out messages. A summary log 
def show(message: str = "") -> None:
    print(message)
    log_lines.append(message)

# Prints out section titles so won't have to type every time anymore
def section(title: str) -> None:
    show("")
    show("="*70)
    show(f"{title}")
    show("="*70)

# ──────────────────────────────────────────────────────────────────────
# Check all the files exist before starting to write the code
# ──────────────────────────────────────────────────────────────────────
# Path to subject's resting state EEG BrainVision header file
def rest_vhdr(sub: str) -> Path:
    return data_root/sub/"eeg"/f"{sub}_task-rest_eeg.vhdr"

#  Path to subject's resting state BIDS events sidecar
def rest_events_tsv(sub: str) -> Path:
    return data_root/sub/"eeg"/f"{sub}_task-rest_events.tsv"
 
# Check if files are pn the disk
section("Check files are on the disk")
missing = []
for sub in SUBJECTS:
    vhdr = rest_vhdr(sub)
    if vhdr.exists():
        show(f"{sub}: {vhdr.name} is present")
    else:
        show(f"{sub}: is missing {vhdr}")
        missing.append(sub)
 
if missing:
    show("")
    show(f"{len(missing)} subjects are missing rest EEG:")
    show(f"The subjects are: {missing}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# Load headers for all subjects
#
# preload=False only touches the .vhdr file, not the ~300 MB .eeg binary.
# Keeps things qucik
# ──────────────────────────────────────────────────────────────────────
# Check metadata
section("Load headers for all subjects")
raws_headers: dict[str, mne.io.Raw] = {}
for sub in SUBJECTS:
    raw = mne.io.read_raw_brainvision(rest_vhdr(sub), preload=False, verbose="ERROR")
    raws_headers[sub] = raw
    show(f"{sub}: {len(raw.ch_names)} channels, {raw.info['sfreq']:.0f} Hz, {raw.times[-1]:.1f} s")

# ──────────────────────────────────────────────────────────────────────
# Check the consistency of the channel list
# ──────────────────────────────────────────────────────────────────────
section("Channel list consistency")

# Use subject 1 as a reference. Other subjects should match thischannel list so that cross subject GNN node
# alignment is able to work
reference_channels = raws_headers[SUBJECTS[0]].ch_names
show(f"Reference: {SUBJECTS[0]} — {len(reference_channels)} channels")
show("")

# Collect results for each subject here, will be used for csv export
consistency_rows = []
all_consistent = True

# Compare each subject's channel list to a 3D reference. Same count, same names and same order. Need to be consistent
for sub in SUBJECTS:
    ch_names = raws_headers[sub].ch_names
    same_count = len(ch_names) == len(reference_channels)
    same_names = set(ch_names) == set(reference_channels)
    same_order = ch_names == reference_channels
 
    # report failure mode. Ordered by improtance
    if same_count and same_names and same_order:
        status = "All match"
    elif same_names:
        status = "Same names but different order"
        all_consistent = False
    elif same_count:
        status = "Same count but different names"
        all_consistent = False
    else:
        status = f"Different channel count ({len(ch_names)} vs {len(reference_channels)})"
        all_consistent = False
 
    show(f"{sub}: {status}")
    consistency_rows.append({"subject": sub, "n_channels": len(ch_names), "matches_reference": status == "match", "status": status})

# Save the per subject table for reference
pd.DataFrame(consistency_rows).to_csv(results_dir/"channel_consistency.csv", index=False)

# ──────────────────────────────────────────────────────────────────────
# Check the recording metadata is consistemt
# ──────────────────────────────────────────────────────────────────────
section("Consistency of recording metadata")

# Collect per subject recording metadata. Fromat as a list of dictionaries. One row per subject.
# OIncludes channel count for the sake of being complete view (although this was covered in last code block)
metadata_rows = []
for sub in SUBJECTS:
    raw = raws_headers[sub]
    metadata_rows.append({"subject": sub, "sample_rate_hz": raw.info["sfreq"], "duration_s": raw.times[-1], "n_channels": len(raw.ch_names), "n_samples": len(raw.times)})
 
# Build df, print to log and save to results directory
metadata_df = pd.DataFrame(metadata_rows)
show(metadata_df.to_string(index=False))
metadata_df.to_csv(results_dir/"metadata_summary.csv", index=False)
 
# Sample rate should be identical
sfreqs = metadata_df["sample_rate_hz"].unique()

# Duration should be within a reasonable range
durations = metadata_df["duration_s"]

# Check if sample rate is infact identical
show("")
if len(sfreqs) == 1:
    show(f"All subjects are at {sfreqs[0]:.0f} Hz.")
else:
    show(f"Multiple sample rates found: {sfreqs}")

# Check what the duration range is
show(f"Duration range: {durations.min():.1f}-{durations.max():.1f}s (spread {durations.max()-durations.min():.1f} s)")
 
# Expect between 600s and 660s for the 4 and then 6 minute protocols plus  time for the instructions/gap
if 550 <= durations.min() and durations.max() <= 720: # Give a sufficent buffer
    show("All durations are within the expected 10+ minute overhead range.")
else:
    show("At least one duration is outside 550-720s. This will need to be checked.")

# ──────────────────────────────────────────────────────────────────────
# Check that the event structures are consistent.
# ──────────────────────────────────────────────────────────────────────
section("Consistency of event structure")


def onset_of(code: str, events: np.ndarray, event_id: dict, sfreq: float) -> float | None:
    """Return the onset time in seconds of the first occurrence of `code`, or None."""
    if code not in event_id:
        return None
    matches = events[events[:, 2] == event_id[code]]
    if len(matches) == 0:
        return None
    return float(matches[0, 0]) / sfreq


event_rows = []
for sub in SUBJECTS:
    raw = raws_headers[sub]

    # Pull events from the annotations (loaded from the .vmrk marker file)
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    except (ValueError, RuntimeError) as e:
        show(f"{sub}: could not extract events ({type(e).__name__}: {e})")
        event_rows.append({
            "subject": sub, "n_events": None, "codes_present": None,
            "eyes_open_onset_s": None, "eyes_closed_onset_s": None,
            "expected_pattern": False,
        })
        continue

    # First-run diagnostic — shows exactly what code strings MNE produced
    show(f"{sub}: event_id = {event_id}")

    sfreq = raw.info["sfreq"]
    eyes_open_onset = onset_of("Stimulus/S  2", events, event_id, sfreq)
    eyes_closed_onset = onset_of("Stimulus/S  4", events, event_id, sfreq)

    codes_present = set(event_id.keys())
    matches_expected = EXPECTED_EVENT_CODES.issubset(codes_present)

    show(f"{sub}: {len(events)} events, codes {sorted(codes_present)}")
    if eyes_open_onset is not None:
        show(f"  eyes-open onset (S 2): {eyes_open_onset:.1f} s")
    if eyes_closed_onset is not None:
        show(f"  eyes-closed onset (S 4): {eyes_closed_onset:.1f} s")

    event_rows.append({
        "subject": sub,
        "n_events": len(events),
        "codes_present": ",".join(sorted(codes_present)),
        "eyes_open_onset_s": eyes_open_onset,
        "eyes_closed_onset_s": eyes_closed_onset,
        "expected_pattern": matches_expected,
    })

# Save the per-subject events summary
event_df = pd.DataFrame(event_rows)
event_df.to_csv(results_dir/"event_summary.csv", index=False)

show("")
n_matching = event_df["expected_pattern"].sum()
show(f"{n_matching} out of {len(SUBJECTS)} subjects show the expected event pattern.")

# ──────────────────────────────────────────────────────────────────────
# Load data (preload=True) for the analyses below that need it
# ──────────────────────────────────────────────────────────────────────
section("Loading full data for signal focused analyses")
 
raws: dict[str, mne.io.Raw] = {}
for sub in SUBJECTS:
    show(f"{sub}: loading")
    raw = mne.io.read_raw_brainvision(rest_vhdr(sub), preload=True, verbose="ERROR")
    # Apply the standard montage. This gives channel positions so bad channel detection and any topographic 
    # plotting is possible
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, match_case=False, on_missing="warn", verbose="ERROR")
    raws[sub] = raw

# ──────────────────────────────────────────────────────────────────────
# Extract the eyes-closed segment for each subject
#
# Uses the S 4 onset (eyes-closed condition start) and takes 60 s from
# the middle of the eyes-closed segment to avoid onset/offset transients.
# Falls back to a middle-of-recording segment if events are missing.
# ──────────────────────────────────────────────────────────────────────
def extract_eyes_closed_segment(
    raw: mne.io.Raw, seg_length_s: float = 60.0
) -> tuple[mne.io.Raw, float, float]:
    """
    Return a cropped Raw containing seg_length_s from the middle of the
    eyes-closed block, and the (start, stop) times used.

    Uses S 4 (eyes-closed onset) and S 11 (end of task) from the raw's
    annotations. Falls back to a middle-of-recording segment if either
    marker is missing.
    """
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        sfreq = raw.info["sfreq"]

        if "Stimulus/S  4" in event_id and "Stimulus/S 11" in event_id:
            s4_matches = events[events[:, 2] == event_id["Stimulus/S  4"]]
            s11_matches = events[events[:, 2] == event_id["Stimulus/S 11"]]

            if len(s4_matches) > 0 and len(s11_matches) > 0:
                ec_start = float(s4_matches[0, 0]) / sfreq
                ec_end = float(s11_matches[0, 0]) / sfreq

                # Take seg_length_s from the middle of the eyes-closed block
                middle = (ec_start + ec_end) / 2
                start = max(ec_start, middle - seg_length_s / 2)
                stop = min(ec_end, start + seg_length_s)
                return raw.copy().crop(tmin=start, tmax=stop), start, stop
    except (ValueError, RuntimeError):
        pass

    # Fallback: seg_length_s from the middle of the whole recording
    middle = raw.times[-1] / 2
    start = middle - seg_length_s / 2
    stop = middle + seg_length_s / 2
    return raw.copy().crop(tmin=start, tmax=stop), start, stop


section("Extracting 60 s from mid-eyes-closed segment for each subject")
segments: dict[str, mne.io.Raw] = {}
for sub in SUBJECTS:
    seg, start, stop = extract_eyes_closed_segment(raws[sub])
    segments[sub] = seg
    show(f"{sub}: {start:.1f}-{stop:.1f}s ({stop - start:.1f}s)")

# ──────────────────────────────────────────────────────────────────────
# Per-subject PSD comparison
# ──────────────────────────────────────────────────────────────────────
section("PSD comparison by subject")
 
fig, ax = plt.subplots(figsize=(10, 6))
alpha_peaks = []
 
for sub in SUBJECTS:
    seg = segments[sub]
    # Compute PSD across all channels then average out.
    psd = seg.compute_psd(fmin=1.0, fmax=60.0, verbose="ERROR")
    freqs = psd.freqs

    # Average across channels and convert to dB for visibility.
    mean_power = psd.get_data().mean(axis=0)
    mean_power_db = 10*np.log10(mean_power+1e-20)
 
    ax.plot(freqs, mean_power_db, label=sub, alpha=0.75)
 
    # Find the alpha peak. It should max out in the 7-13 Hz band.
    alpha_mask = (freqs >= 7) & (freqs <= 13)
    if alpha_mask.any():
        alpha_freq = freqs[alpha_mask][np.argmax(mean_power[alpha_mask])]
        alpha_amp_db = mean_power_db[alpha_mask][np.argmax(mean_power[alpha_mask])]
    else:
        alpha_freq = None
        alpha_amp_db = None
 
    alpha_peaks.append({"subject": sub, "alpha_peak_hz": alpha_freq, "alpha_peak_power_db": alpha_amp_db})
    show(f"{sub}: alpha peak at {alpha_freq:.2f}Hz (power {alpha_amp_db:.1f}dB)")
 
ax.axvspan(8, 13, alpha=0.1, color="green", label="alpha (8-13 Hz)")
ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="mains (50 Hz)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB)")
ax.set_title("Cohort PSD comparison — 60s eyes-closed segment")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
 
psd_path = results_dir/"psd_comparison.png"
fig.savefig(psd_path, dpi=120, bbox_inches="tight")
plt.close(fig)
show("")
show(f"Saved: {psd_path}")
 
pd.DataFrame(alpha_peaks).to_csv(results_dir/"alpha_peaks.csv", index=False)

# ──────────────────────────────────────────────────────────────────────
# Check amplitude and variance
# ──────────────────────────────────────────────────────────────────────
section("Amplitude and variance sanity check")

# Collect per-subject amplitude stats for comparison
amplitude_rows = []
for sub in SUBJECTS:
    seg = segments[sub]
    data = seg.get_data()*1e6  # convert to microvolts
 
    # Per-channel stats. Will then aggregate across channels
    per_channel_std = data.std(axis=1)
    per_channel_p2p = data.max(axis=1)-data.min(axis=1)

    # Median summarises typical channel, max identidies the worst one
    amplitude_rows.append({"subject": sub, "median_channel_std_uv": float(np.median(per_channel_std)), "median_channel_p2p_uv": float(np.median(per_channel_p2p)), "max_channel_std_uv": float(per_channel_std.max()), "max_channel_p2p_uv": float(per_channel_p2p.max())})

# Build the df, print and save to results directory
amp_df = pd.DataFrame(amplitude_rows)
show(amp_df.to_string(index=False))
amp_df.to_csv(results_dir/"amplitude_summary.csv", index=False)
 
# Flag any subject that has a  median standard deviation of more than 3x the cohort median.
cohort_median = amp_df["median_channel_std_uv"].median()
show("")
show(f"Cohort median std across channels: {cohort_median:.2f} uV")

# This is just a rough heuristi/check rather than anything too rigourous
outliers = amp_df[amp_df["median_channel_std_uv"]>3*cohort_median]
if len(outliers) == 0:
    show("No subject has median amplitude more than 3x cohort median.")
else:
    show(f"{len(outliers)} subjects have unusually high amplitude:")
    show(outliers[["subject", "median_channel_std_uv"]].to_string(index=False))

# ──────────────────────────────────────────────────────────────────────
# Screen for bad channels
#
# This uses MNE's local outlier factor (LOF) method. Flags channels whose 
# stats are very different from neighbours. It then threshold defaults 
# to a moderate value. Log which channels are identified for each subject
# ──────────────────────────────────────────────────────────────────────
section("Bad channel screening (LOF)")

# Per subject recordings for csv output
bad_channels_per_subject = []

# Per channel counter. Will help identify systematically bad electrodes
bad_channel_counts: dict[str, int] = {}

# Run LOF on each subject's eyes-closed segment
for sub in SUBJECTS:
    seg = segments[sub]
    try:
        bads = mne.preprocessing.find_bad_channels_lof(seg, verbose="ERROR")
    except Exception as e:
        show(f"{sub}: LOF failed ({type(e).__name__}: {e})")
        bads = []
    
    # Log the count and then names if there were any
    show(f"{sub}: {len(bads)} bad channels are flagged")
    if bads:
        show(f"{bads} are bad channels")
 
    bad_channels_per_subject.append({"subject": sub, "n_bad": len(bads), "bad_channels": ",".join(bads) if bads else ""})
    
    # Update cohort wide count for each flagged channel
    for ch in bads:
        bad_channel_counts[ch] = bad_channel_counts.get(ch, 0)+1

# Save per-subject summary to results directory
pd.DataFrame(bad_channels_per_subject).to_csv(results_dir/"bad_channels_per_subject.csv", index=False)
 
# Find which channels are consistenly bad
if bad_channel_counts:
    per_channel_df = (pd.DataFrame([{"channel": ch, "n_subjects_flagged": count} for ch, count in bad_channel_counts.items()]).sort_values("n_subjects_flagged", ascending=False).reset_index(drop=True))
    
    show("")
    show("Channels flagged  as bad across multiple subjects:")

    # A channel flagged in two or more subjects is a systematic problem. Could be hardware, gel, fit of the cap etc
    consistently_bad = per_channel_df[per_channel_df["n_subjects_flagged"]>=2]

    # Print systematically bad channels if they exist. If not acknowedge the lack of pattern
    if len(consistently_bad)>0:
        show(consistently_bad.to_string(index=False))
    else:
        show("Bad channels per subject don't seem to show any real pattern")

    # Save per-electrode summary alongside the per subject one
    per_channel_df.to_csv(results_dir / "bad_channels_per_electrode.csv", index=False)

else:
    show("")
    show("No bad channels flagged in any subject.")
 
# ──────────────────────────────────────────────────────────────────────
# Save the results log
# ──────────────────────────────────────────────────────────────────────
summary_path = results_dir/"summary.txt"
summary_path.write_text("\n".join(log_lines), encoding="utf-8")

print("")
print(f"Text summary saved to: {summary_path}")

 
