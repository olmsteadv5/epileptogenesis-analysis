"""Magnitude squared coherence (MSC) analysis.

The original notebook hand-rolled a Hann-windowed Welch MSC loop. Here we lean on
``scipy.signal.coherence`` (the same Hann-windowed Welch MSC, but battle-tested
and edge-case-safe) behind a thin ``compute_msc`` wrapper, so the rest of the
codebase keeps a clean interface.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import coherence
from scipy.signal.windows import hann


def compute_msc(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: int = 256,
    noverlap: int = 128,
    nfft: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude squared coherence between ``x`` and ``y``.

    Thin wrapper over ``scipy.signal.coherence`` using a symmetric Hann window
    (matching the original ``0.5 * (1 - cos(...))`` window) of length ``window``,
    ``noverlap`` overlap, and an ``nfft``-point FFT. Returns ``(freqs_hz, msc)``.
    """
    win = hann(window, sym=True)
    freqs, msc = coherence(
        x,
        y,
        fs=fs,
        window=win,
        noverlap=noverlap,
        nfft=nfft,
    )
    return freqs, msc


def _band_average(freqs: np.ndarray, msc: np.ndarray, band, include_low: bool) -> float:
    """Mean MSC over a frequency band.

    ``include_low`` controls the lower edge: ``True`` -> ``[lo, hi]`` (used for the
    low band), ``False`` -> ``(lo, hi]`` (used for the high band) so the shared
    boundary frequency is not counted in both bands. Returns NaN if the band is
    empty.
    """
    lo, hi = band
    if include_low:
        mask = (freqs >= lo) & (freqs <= hi)
    else:
        mask = (freqs > lo) & (freqs <= hi)
    return float(np.mean(msc[mask])) if np.any(mask) else float("nan")


def compute_seizure_coherence(
    data1_full: np.ndarray,
    data2_full: np.ndarray,
    seizure_start_sample: int,
    seizure_end_sample: int,
    pre_start_sample: int,
    pre_end_sample: int,
    samp_freq: float,
    window: int = 256,
    noverlap: int = 128,
    nfft: int = 1024,
    band_low=(0.0, 14.0),
    band_high=(14.0, 55.0),
) -> dict:
    """Compute during-seizure vs pre-seizure coherence for one seizure.

    Slices the two full-channel signals into the seizure window and the
    pre-seizure window (0-based, half-open sample indices), runs ``compute_msc``
    on each, and returns band-limited average MSC (low and high bands, for both
    seizure and pre-seizure) plus the raw freq/MSC arrays needed for plotting.
    """
    sz1 = data1_full[seizure_start_sample:seizure_end_sample]
    sz2 = data2_full[seizure_start_sample:seizure_end_sample]
    pre1 = data1_full[pre_start_sample:pre_end_sample]
    pre2 = data2_full[pre_start_sample:pre_end_sample]

    freqs_seizure, msc_seizure = compute_msc(sz1, sz2, samp_freq, window, noverlap, nfft)
    freqs_pre, msc_pre = compute_msc(pre1, pre2, samp_freq, window, noverlap, nfft)

    return {
        "avg_msc_seizure_low": _band_average(freqs_seizure, msc_seizure, band_low, True),
        "avg_msc_pre_low": _band_average(freqs_pre, msc_pre, band_low, True),
        "avg_msc_seizure_high": _band_average(freqs_seizure, msc_seizure, band_high, False),
        "avg_msc_pre_high": _band_average(freqs_pre, msc_pre, band_high, False),
        "freqs_seizure": freqs_seizure,
        "msc_seizure": msc_seizure,
        "freqs_pre": freqs_pre,
        "msc_pre": msc_pre,
    }
