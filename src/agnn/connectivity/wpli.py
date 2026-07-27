"""
wPLI computation for cohort connectivity analysis.

Uses mne-connectivity's spectral_connectivity_time with CWT-Morlet spectral estimation to compute per-epoch wPLI (Vinck et al. 2011). Adaptive n_cycles
ensures wavelet windows fit within the 4-second epochs.

- CWT-Morlet is used (not multitaper) because its faster and more reliable than multitaper for this workload. Multitaper was investigated but wouldn't run
  at low-frequency adaptive-n_cycles combinations.
- Per-epoch computation (average=False) matches the graph-aggregation decision (each epoch is a separate training example).
- Bands are computed one at a time so `faverage=True` averages wPLI across the band's frequency bins.

IMPORTANT CAVEAT: For the lowest delta frequencies (0.5-1 Hz), the wavelet window needed to be shortedned to fit within the 4-second epoch, giving fewer 
than 3 cycles. At 0.5 Hz the wavelet uses around 1.75 cycles (3.5-second window). This produces usable but noisier estimates at low delta. Alpha, beta, 
gamma, and delta above 1 Hz use the full 3-cycle window.
"""

from __future__ import annotations
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_time


def compute_wpli_all_bands(epochs: mne.Epochs, bands: dict[str, tuple[float, float]], n_cycles_max: float = 3.0,) -> dict[str, np.ndarray]:
    """
    Compute per-epoch wPLI for each frequency band using CWT-Morlet.

    Parameters
    ----------
    - epochs : mne.Epochs
          Preprocessed epochs for one subject.
    - bands : dict
          Band name -> (fmin, fmax) in Hz.
    - n_cycles_max : float, default 3.0
          Maximum wavelet cycles per frequency. Lower values are used automatically for low frequencies to keep the wavelet window shorter than the epoch.

    Returns
    -------
    - wpli : dict
          band_name -> array of shape (n_epochs, n_channels, n_channels). Diagonal zeroed. Matrix is symmetric.
    """
    # Number of channels and create dictionary for results
    n_channels = len(epochs.ch_names)
    result: dict[str, np.ndarray] = {}

    for band_name, (fmin, fmax) in bands.items():
        # One frequency grid point per 0.5 Hz within the band, at minimum 2 points.
        n_points = max(2, int(np.round((fmax-fmin)/0.5))+1)
        freqs = np.linspace(fmin, fmax, n_points)

        # Adaptive n_cycles: cap at n_cycles_max but reduce for low frequencies so the wavelet window (n_cycles/freq seconds) fits within the epoch.
        adaptive_cycles = np.minimum(n_cycles_max, freqs*1.8)

        con = spectral_connectivity_time(
            epochs,
            freqs=freqs,
            method="wpli",
            average=False, # makes it per epoch
            faverage=True, # averages across freqencies within band
            mode="cwt_morlet", # faster and more reliable than multitaper here
            n_cycles=adaptive_cycles,
            n_jobs=1,
            verbose="ERROR",
        )

        # Dense output shape: (n_epochs, n_ch, n_ch, n_freqs=1 after faverage).
        con_data = con.get_data(output="dense")[..., 0]

        # Numerical noise can produce tiny negatives so clip to [0, 1] for the sake of cleanliness
        con_data = np.clip(con_data, 0.0, 1.0)

        # Zero the diagonal. There;s no self-connectivity by construction.
        for i in range(n_channels):
            con_data[:, i, i] = 0.0

        result[band_name] = con_data.astype(np.float32)

    return result