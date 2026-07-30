"""
wPLI computation using filter-Hilbert for cohort connectivity analysis.

This is an exact match to Klepl et al.'s methodology:
 - Bandpass filter each epoch with 5th-order Butterworth IIR (zero-phase)
 - Hilbert transform to get complex analytic signal
 - Compute wPLI per epoch from cross-spectrum across time samples
"""

from __future__ import annotations
import numpy as np
import mne


def _wpli_from_analytic(analytic: np.ndarray) -> np.ndarray:
    """
    Compute per-epoch wPLI matrices from complex analytic signals.

    Parameters
    ----------
    - analytic: complex array, shape (n_epochs, n_channels, n_times). This is an analytic signal from the Hilbert transform 
      of a bandpass-filtered recording.

    Returns
    -------
    - wpli: real array, shape (n_epochs, n_channels, n_channels). Per-epoch wPLI matrices. These are symmetric. with zero 
      diagonal and values in [0, 1].

    Notes
    -----
    Follows Vinck et al. euation 8:

        wPLI(i, j) = |E{|Im(S_ij)| · sign(Im(S_ij))}| / E{|Im(S_ij)|}

    For each pair of channels, at every point in time in the epoch, want to look at the phase difference between the two signals.
    (ie is channel i ahead of or behind channel j, and by how much). Im(S_ij) is the imaginary part of the cross spectrum, and
    the sign indicates whether i is in fromt of or behind j, and it's size indicates how far from zero lag that particylar 
    moment is.

    wPLI then wants to find out if, across the whole epoch, the phase difference consistenly points the same way (mostly leading
    or mostly lagging) or if it flips back and forth unpredictably. If the relationship is consistent, the signed values will
    reinforce each other, and when averaged will come out closer to 1. Isf inconsistent, positive and negative moments cancel 
    when averaged and wPLi comes out closer to 0.

    The weighted part is important because momemts where the two signals have close to no lag are downweighted. This is because
    true brainconnectivity basically never produces zero lag synchron -> this is mostly a result of volume conduction, which 
    is known to be a possible issue here due to electrode count. By using only the imaginary part of the cross-spectrum (never 
    the real bit) and weighting by size wPLI should be more robust to volume conduction.
    """
    n_epochs, n_channels, _ = analytic.shape
    wpli = np.zeros((n_epochs, n_channels, n_channels), dtype=np.float32)

    for e in range(n_epochs):
        z = analytic[e]  # (n_channels, n_times)

        # Cross spectrum for every channel pair at every time point. S_ij(t) = z_i(t)*conj(z_j(t)).
        # "it,jt->ijt" multiplies each channel i's signal by each channel j's conjugate, keeping the time axis, giving a 
        # (n_channels, n_channels, n_times) array.
        S = np.einsum("it,jt->ijt", z, np.conj(z))
        imag = np.imag(S)

        # Numerator averages the signed imaginary bit over time, then takes the absolute value.
        # This is the bit that shrinks toward 0 when the phase relationship flips direction across the epoch, and stays large when it's consistent
        num = np.abs(np.mean(np.abs(imag)*np.sign(imag), axis=-1))

        # Denominator averages the magnitude of the imaginary part over time, ignoring sign. Normalises wPLI so it stays in [0, 1] 
        # instead of scaling with raw signal amplitude.
        den = np.mean(np.abs(imag), axis=-1)

        # Divide the numerator by denominator for wPLI, avoiding divide-by-zero 
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(den>0, num/den, 0.0)

        # Zero the diagonal explicitly (avpid self-connectivity).
        np.fill_diagonal(m, 0.0)

        # Clip numerical noise into [0, 1].
        wpli[e] = np.clip(m, 0.0, 1.0).astype(np.float32)

    return wpli


def compute_wpli_all_bands(
    epochs: mne.Epochs,
    bands: dict[str, tuple[float, float]],
    iir_order: int = 5,
) -> dict[str, np.ndarray]:
    """
    Compute per-epoch wPLI for each frequency band via filter-Hilbert.

    Parameters
    ----------
    - epochs: mne.Epochs
          Preprocessed epochs for one subject
    - bands: dict
          Band name -> (fmin, fmax) in Hz
    - iir_order: int, default 5
          Butterworth filter order. 5 matches Klepl et al. (2022, 2023).

    Returns
    -------
    - wpli: dict
          band_name -> array of shape (n_epochs, n_channels, n_channels). Symmetric matrix, zero diagonal, values in [0, 1]
    """
    result: dict[str, np.ndarray] = {}

    for band_name, (fmin, fmax) in bands.items():
        # Bandpass filter with zero-phase Butterworth IIR. MNE's method="iir" with default settings applies zero-phase filtering (forward and reverse)
        # so there's no phase distortion introduced by the filter.
        band_epochs = epochs.copy().filter(l_freq=fmin, h_freq=fmax, method="iir", iir_params=dict(order=iir_order, ftype="butter"), verbose="ERROR")

        # Hilbert transform in place. envelope=False keeps the complex analytic signal (need this for wPLI's imaginary cross spectrum computation)
        band_epochs.apply_hilbert(envelope=False)

        # Get complex analytic signal. Shape (n_epochs, n_channels, n_times)
        analytic = band_epochs.get_data()

        result[band_name] = _wpli_from_analytic(analytic)

    return result