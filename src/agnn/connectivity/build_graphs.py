"""
Build all per-epoch, per-band graphs for one subject.

Produces 6 n_epochs PyG Data objects (six bands x approx 90 epochs per subject). Each Data object represents one epoch's connectivity in one frequency 
band.

Design:
    - 6 separate Data objects per epoch, each with its own edges
    - Per band top-k% thresholding 
    - Per epoch node features (relative band powers)
    - data.subject_id, data.epoch_idx, data.band (needed as attributes for CV filtering
    - Symmetric wPLI, zero diagonal. GCN handles self-loops using a  renormalisation trick.
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


def build_subject_graphs(subject_id: str, config: dict, preprocessed_root: Path | str, output_root: Path | str, apoe_label: int | None = None) -> dict:
    """
    Build all per epoch, per band graphs for one subject.

    Parameters
    ----------
    - subject_id : str
          BIDS subject identifier (eg "sub-06")
    - config : dict
          Loaded project config with a "connectivity" section.
    - preprocessed_root : Path or str
          Root of preprocessed epochs 
    - output_root : Path or str
          Where to write graph files 
    - apoe_label : int or None
          Binary APOE e4 carrier label (0/1). Attached as data.y if provided.

    Returns
    -------
    - status : dict
          Per subject summary with counts, timings, and status.
    """
    # Set up input and output graphs for this subject
    preprocessed_root = Path(preprocessed_root)
    output_root = Path(output_root)
    subject_out = output_root/subject_id
    subject_out.mkdir(parents=True, exist_ok=True)

    # Where final graph file gets saved
    output_path = subject_out/f"{subject_id}_task-rest_graphs.pt"

    # Initialise status dict with default values. Gets populated as pipeline progressez. Returned even when it fails so batch runner can log
    status: dict = {"subject": subject_id, "status": "error", "output_path": None, "n_epochs": None, "n_bands": None, "n_graphs": None, "n_nodes": None,
        "mean_edges_per_graph": None, "apoe_label": apoe_label, "runtime_seconds": None, "error_message": None}

    # Start runtime timer
    start_time = time.time()

    # Unpack connectivity settings from config
    cfg = config["connectivity"]
    bands = {name: tuple(rng) for name, rng in cfg["bands"].items()}
    k_pct = float(cfg["edge_threshold_pct"])

    try:
        # Load preprocessed epochs from the preprocessing pipeline's output
        epo_path = (preprocessed_root/subject_id/f"{subject_id}_task-rest_desc-preprocessed_epo.fif")
        if not epo_path.exists():
            raise FileNotFoundError(f"Preprocessed epochs not found: {epo_path}")

        # Read the .fif file into memory and record basic dimensions
        epochs = mne.read_epochs(epo_path, preload=True, verbose="ERROR")
        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)

        # Per-epoch wPLI per band. dict[band_name] -> (n_epochs, n_ch, n_ch)
        wpli = compute_wpli_all_bands(epochs, bands=bands)

        # Per-epoch node features: (n_epochs, n_channels, n_bands). Same feature vector across all 6 band-graphs for a given epoch. Only the edges differ 
        # between the graphs.
        node_powers = compute_band_powers(epochs, bands=bands, relative=True)

        # Build the graphs. For each epoch x band, produce one Data object.
        graphs: list[Data] = []
        edges_by_band: dict[str, list[int]] = {b: [] for b in bands}

        # Loop over epochs, then bands within each epoch. 6 x n_epochs graphs total.
        for epoch_idx in range(n_epochs):
            x = torch.from_numpy(node_powers[epoch_idx]).float()

            for band_name in bands:
                # Filter-Hilbert wPLI returns a fully symmetric matrix, so no
                # need to symmetrise here.
                m = wpli[band_name][epoch_idx]
                edge_index, edge_weight = top_k_threshold(m, k_pct=k_pct)
                edges_by_band[band_name].append(edge_index.shape[1])

                # Scalar edge attribute per edge. Unsqueeze to shape (E, 1) so downstream GCN layers can treat it uniformly.
                data = Data(x=x.clone(), edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1))
                data.subject_id = subject_id
                data.epoch_idx = epoch_idx
                data.band = band_name

                # Attach the APOE carrier label as the target 
                if apoe_label is not None:
                    data.y = torch.tensor([apoe_label], dtype=torch.long)

                # Add graph to running list of graphs for this particular subject
                graphs.append(data)

        # Persist all this subject's graphs in one file. Allows for easy per-subject load during cross-validation.
        torch.save(graphs, output_path)

        # Fill in the status dictionary for the batch runner
        status["status"] = "success"
        status["output_path"] = str(output_path)
        status["n_epochs"] = n_epochs
        status["n_bands"] = len(bands)
        status["n_graphs"] = len(graphs)
        status["n_nodes"] = n_channels
        status["mean_edges_per_graph"] = {b: int(np.mean(e)) for b, e in edges_by_band.items()}

    # Catch any failure and record it so batch will keep running for other subjects
    except Exception as e:
        status["status"] = "error"
        status["error_message"] = f"{type(e).__name__}: {e}"
    # record runtiem
    finally:
        status["runtime_seconds"] = time.time()-start_time

    return status