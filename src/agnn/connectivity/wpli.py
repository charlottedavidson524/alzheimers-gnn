"""
wPLI computation via filter-Hilbert for cohort connectivity analysis.

Matches Klepl et al. (2022, 2023) methodology exactly:
1. Bandpass filter each epoch with 5th-order Butterworth IIR (zero-phase)
2. Hilbert transform to get complex analytic signal
3. Compute wPLI (Vinck et al. 2011) per epoch from cross-spectrum across time samples

Why IIR Butterworth rather than FIR
------------------------------------
FIR bandpass filters require a filter length of ~3.3/transition_bandwidth. At
a 0.5 Hz lower cutoff with default settings, this requires a 6.6-second filter,
longer than the 4-second epoch. Butterworth IIR is recursive and has no such
length constraint, so it accommodates low-frequency analysis within short epochs.
Klepl et al. explicitly specify 5th-order Butterworth.

Note on absolute wPLI values
----------------------------
Per-epoch wPLI computed over time samples of narrowband signals naturally
produces higher absolute values than cross-trial wPLI reported in some
literature. Each epoch is internally more phase-coherent than an average
across many trials. This is expected for the filter-Hilbert per-epoch
formulation and not a bug — Klepl et al.'s reported values are similarly
elevated.
"""

from __future__ import annotations

import numpy as np
import mne


def _wpli_from_analytic(analytic: np.ndarray) -> np.ndarray:
    """Compute per-epoch wPLI matrices from complex analytic signals.

    Parameters
    ----------
    analytic : complex array, shape (n_epochs, n_channels, n_times)
        Analytic signal from the Hilbert transform of a bandpass-filtered
        recording.

    Returns
    -------
    wpli : real array, shape (n_epochs, n_channels, n_channels)
        Per-epoch wPLI matrices. Symmetric, with zero diagonal, values in [0, 1].

    Notes
    -----
    Follows Vinck et al. (2011) Eq. (8):

        wPLI(i,j) = |E{|Im(S_ij)| · sign(Im(S_ij))}| / E{|Im(S_ij)|}

    Where S_ij(t) = z_i(t) · conj(z_j(t)) is the cross-spectrum between
    channels i and j at time t, and the expectation is taken over time
    samples within an epoch.
    """
    n_epochs, n_channels, _ = analytic.shape
    wpli = np.zeros((n_epochs, n_channels, n_channels), dtype=np.float32)

    for e in range(n_epochs):
        z = analytic[e]  # (n_channels, n_times)

        # Cross-spectrum for all channel pairs at each time point.
        # S[i, j, t] = z[i, t] * conj(z[j, t])
        S = np.einsum("it,jt->ijt", z, np.conj(z))
        imag = np.imag(S)

        # Numerator: absolute value of the mean signed imaginary cross-spectrum.
        # Denominator: mean absolute imaginary cross-spectrum.
        num = np.abs(np.mean(np.abs(imag) * np.sign(imag), axis=-1))
        den = np.mean(np.abs(imag), axis=-1)

        # Guard against divide-by-zero; the diagonal has Im(S)=0 by definition
        # so its wPLI is undefined but forced to zero anyway.
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(den > 0, num / den, 0.0)

        # Zero the diagonal explicitly (no self-connectivity).
        np.fill_diagonal(m, 0.0)

        # Clip numerical noise into [0, 1].
        wpli[e] = np.clip(m, 0.0, 1.0).astype(np.float32)

    return wpli


def compute_wpli_all_bands(
    epochs: mne.Epochs,
    bands: dict[str, tuple[float, float]],
    iir_order: int = 5,
) -> dict[str, np.ndarray]:
    """Compute per-epoch wPLI for each frequency band via filter-Hilbert.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed epochs for one subject.
    bands : dict
        Band name -> (fmin, fmax) in Hz.
    iir_order : int, default 5
        Butterworth filter order. 5 matches Klepl et al. (2022, 2023).

    Returns
    -------
    wpli : dict
        band_name -> array of shape (n_epochs, n_channels, n_channels).
        Symmetric matrix, zero diagonal, values in [0, 1].
    """
    result: dict[str, np.ndarray] = {}

    for band_name, (fmin, fmax) in bands.items():
        # Bandpass filter with zero-phase Butterworth IIR. MNE's method="iir"
        # with default settings applies zero-phase filtering (forward + reverse)
        # so there's no phase distortion introduced by the filter.
        band_epochs = epochs.copy().filter(
            l_freq=fmin,
            h_freq=fmax,
            method="iir",
            iir_params=dict(order=iir_order, ftype="butter"),
            verbose="ERROR",
        )

        # Hilbert transform in place. envelope=False keeps the complex analytic
        # signal (needed for wPLI's imaginary cross-spectrum computation).
        band_epochs.apply_hilbert(envelope=False)

        # Get complex analytic signal: shape (n_epochs, n_channels, n_times).
        analytic = band_epochs.get_data()

        result[band_name] = _wpli_from_analytic(analytic)

    return result