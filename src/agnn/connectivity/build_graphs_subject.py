"""
This file is for graph construction at the subject level. To do this, average wPLI matrices and node features will be taken 
across all epochs per subject. This produces one graph per band per subject.

This is in contrast with build_graphs.py which produces one graph per epoch per band (so around 360 per subject). This change is
motivated by the fact that pre-symptomatic APOE e4 effects on EEG connectivity are subtle. They most likely operate at the 
subject level instead of per epoch. Per-epoch variability could dilute the signal. There is also the fact that Klepl et al. 
(2022, 2023) use per-epoch training for dementia vs healthy classification. However this is where subject level differences 
dominate epoch-level noise. Subtler tasks (like this one) might benefit from subject-level aggregation.

The output of this file is 2 graphs per subject (one for delta, one for alpha-2) versus roughly 360 per subject in the 
per-epoch pipeline. The total across 77 subjects should be 154 graphs versus the roughly 13,500 from before.
"""

from __future__ import annotations
import time
from pathlib import Path
import mne
import numpy as np
import torch
from torch_geometric.data import Data
from agnn.connectivity.wpli import compute_wpli_all_bands
from agnn.connectivity.features import compute_band_powers, top_k_threshold


def build_aggregated_graphs(subject_id: str, config: dict, preprocessed_root: Path | str, output_root: Path | str, apoe_label: int | None = None) -> dict:
    """
    This function builds subject level aggregated graphs (one per band).

    Parameters
    ----------
    - subject_id: str
        BIDS subject identifier 
    - config: dict
          Loaded project config with a "connectivity" section
    - preprocessed_root: Path or str
          Root of preprocessed epochs
    - output_root: Path or str
          Where to write graph files
    - apoe_label: int or None
          Binary APOE e4 carrier label (0/1). Will be attached as data.y if provided

    Returns
    -------
    - status: dict
          Per subject summary with counts, timings, and status
    """

    # Normalise paths and make sure the subject's output folder exists
    preprocessed_root = Path(preprocessed_root)
    output_root = Path(output_root)
    subject_out = output_root / subject_id
    subject_out.mkdir(parents=True, exist_ok=True)

    # Full path for this subject's graph file
    output_path = subject_out/f"{subject_id}_task-rest_graphs.pt"

    # Initialise the status dictionary with error state as the default (updated upon success)
    status: dict = {
        "subject": subject_id,
        "status": "error",
        "output_path": None,
        "n_epochs_averaged": None,
        "n_bands": None,
        "n_graphs": None,
        "n_nodes": None,
        "edges_per_band": None,
        "apoe_label": apoe_label,
        "runtime_seconds": None,
        "error_message": None,
    }

    # Start the timer and pull the connectivity section from the config
    start_time = time.time()
    cfg = config["connectivity"]

    # Convert bands to tuples for downstream codinh
    bands = {name: tuple(rng) for name, rng in cfg["bands"].items()}

    #Make the edge threshold a float
    k_pct = float(cfg["edge_threshold_pct"])

    try:
        # Load preprocessed epochs from the preprocessing pipeline's output
        epo_path = (preprocessed_root/subject_id/f"{subject_id}_task-rest_desc-preprocessed_epo.fif")
        if not epo_path.exists():
            raise FileNotFoundError(f"Preprocessed epochs not found: {epo_path}")

        # Read epochs and record cohort relevant shape information
        epochs = mne.read_epochs(epo_path, preload=True, verbose="ERROR")
        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)

        # Per-epoch wPLI per band: dict[band_name] -> (n_epochs, n_ch, n_ch)
        wpli_per_epoch = compute_wpli_all_bands(epochs, bands=bands)

        # Average across epochs -> one matrix per band, shape (n_ch, n_ch). This is the key  difference from the per epoch 
        # pipeline. Noise from epoch-to-epoch variability is averaged out. This leaves the subject level connectivity pattern
        wpli_avg = {band: wpli_per_epoch[band].mean(axis=0) for band in bands}

        # Per-epoch node features: (n_epochs, n_channels, n_bands)
        node_powers_per_epoch = compute_band_powers(epochs, bands=bands, relative=True)

        # Average across epochs -> single feature vector per node per band. Shape: (n_channels, n_bands).#
        node_powers_avg = node_powers_per_epoch.mean(axis=0)

        # Node features shared across both band graphs for this subject
        x = torch.from_numpy(node_powers_avg).float()

        # Build one graph per band, all using the same node features
        graphs: list[Data] = []
        edges_per_band: dict[str, int] = {}

        for band_name in bands:
            # Filter-Hilbert wPLI returns a symmetric matrix. Dont need to do any symmetrising
            edge_index, edge_weight = top_k_threshold(wpli_avg[band_name], k_pct=k_pct)
            edges_per_band[band_name] = int(edge_index.shape[1])

            # Scalar edge attribute per edge. Unsqueeze to (E, 1) so downstream GCN layers can treat uniformly
            data = Data(x=x.clone(), edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1))
            data.subject_id = subject_id
            data.epoch_idx = -1  
            data.band = band_name

            # Attach the APOE label if it's provided. If its unalbelled then skip
            if apoe_label is not None:
                data.y = torch.tensor([apoe_label], dtype=torch.long)

            graphs.append(data)

        # Save both band graphs in one file per subject
        torch.save(graphs, output_path)

        # Populate teh status dictionary 
        status["status"] = "success"
        status["output_path"] = str(output_path)
        status["n_epochs_averaged"] = n_epochs
        status["n_bands"] = len(bands)
        status["n_graphs"] = len(graphs)
        status["n_nodes"] = n_channels
        status["edges_per_band"] = edges_per_band

    # Catch any exceptions and record them in the status dictionary instea dof carrying them forwards.
    except Exception as e:
        status["status"] = "error"
        status["error_message"] = f"{type(e).__name__}: {e}"
    finally:
        status["runtime_seconds"] = time.time() - start_time

    return status