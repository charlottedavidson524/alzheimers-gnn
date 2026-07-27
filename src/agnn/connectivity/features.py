"""
Node features and edge thresholding for graph construction.

There are two functions created here:

- compute_band_powers: per epoch relative band powers as node features.
- top_k_threshold: prune wPLI matrix to top-k% strongest edges.
"""

from __future__ import annotations
import numpy as np
import torch
import mne


def compute_band_powers(epochs: mne.Epochs, bands: dict[str, tuple[float, float]], relative: bool = True) -> np.ndarray:
    """
    Compute per epoch band powers for each channel and band.

    Parameters
    ----------
    - epochs : mne.Epochs
          Preprocessed epochs
    - bands : dict
          Band name -> (fmin, fmax)
    - relative : bool
          If True, normalise so bands sum to 1 per channel per epoch.

    Returns
    -------
    - powers : array
          Shape (n_epochs, n_channels, n_bands).
    """
    # Get the overall frequency range spanning all bands
    fmin_all = min(f[0] for f in bands.values())
    fmax_all = max(f[1] for f in bands.values())

    # Welch's method with defaults appropriate for 4-second epochs at 250 Hz.
    psd = epochs.compute_psd(method="welch", fmin=fmin_all, fmax=fmax_all, n_fft=256, n_overlap=128, verbose="ERROR")
    psds, freqs = psd.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs)

    # Prepare the output array for band-power values
    n_epochs, n_channels, _ = psds.shape
    n_bands = len(bands)
    powers = np.zeros((n_epochs, n_channels, n_bands), dtype=np.float32)

    # For each band, average PSD across the frequency bins that fall within it.
    for band_idx, (fmin, fmax) in enumerate(bands.values()):
        mask = (freqs>=fmin) & (freqs<=fmax)
        # Mean PSD across freqs within band (integration up to a constant).
        powers[:, :, band_idx] = psds[:, :, mask].mean(axis=-1)

    # Convert absolute powers to relative powers if requested.
    if relative:
        # Normalise each (epoch, channel) to sum to 1 across bands.
        total = powers.sum(axis=-1, keepdims=True)
        total = np.where(total>0, total, 1.0)  # Guards against zero
        powers = powers/total

    return powers


def top_k_threshold(wpli_matrix: np.ndarray, k_pct: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Keep top-k% edges by wPLI value. This is symmetrised.

    Parameters
    ----------
    - wpli_matrix : array, shape (n_channels, n_channels)
          Symmetric wPLI matrix with zero diagonal.
    - k_pct : float
          Percentage of edges to retain, e.g. 20.0 for top-20%.

    Returns
    -------
    - edge_index : LongTensor, shape (2, E)
          Source target node indices. undirected 
    - edge_weight : FloatTensor, shape (E,)
          wPLI values for retained edges (same value on both directions).
    """
    # Number of nodes in the graph (one per electrode)
    n_channels = wpli_matrix.shape[0]

    # Upper triangle (i<j). this avoids double-counting a symmetric matrix
    triu_i, triu_j = np.triu_indices(n_channels, k=1)
    triu_values = wpli_matrix[triu_i, triu_j]

    # Rank edges by magnitude. wPLI is non-negative so magnitude is equal to value
    n_edges_total = len(triu_values)
    n_keep = max(1, int(np.round(n_edges_total*k_pct/100.0)))
    top_idx = np.argpartition(triu_values, -n_keep)[-n_keep:]

    # Extract coordinates and weights of the retained edges
    kept_i = triu_i[top_idx]
    kept_j = triu_j[top_idx]
    kept_w = triu_values[top_idx]

    # Symmetrise by adding both directions for the undirected graph.
    src = np.concatenate([kept_i, kept_j])
    dst = np.concatenate([kept_j, kept_i])
    weight = np.concatenate([kept_w, kept_w])

    # Convert numpy arrays to PyTorch tensors in shapes PyG expects
    edge_index = torch.from_numpy(np.stack([src, dst])).long()
    edge_weight = torch.from_numpy(weight).float()

    return edge_index, edge_weight