# Magnitude squared coherence (MSC) analysis.

from __future__ import annotations

import numpy as np
from scipy.signal import coherence, csd
from scipy.signal.windows import hann

def compute_msc(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: int = 256,
    noverlap: int = 128,
    nfft: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:

    # Magnitude squared coherence between x and y.
    # Symmetric Hann window.
    win = hann(window, sym=True)

    # scipy handles the Welch averaging and edge cases.
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
    # Mean MSC over a frequency band.
    lo, hi = band

    # Avoid double-counting the boundary frequency between adjacent bands.
    if include_low:
        mask = (freqs >= lo) & (freqs <= hi)
    else:
        mask = (freqs > lo) & (freqs <= hi)

    # Guard against a band with no bins.
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
    phase_msc_threshold: float = 0.1,
) -> dict:
    # Compute during-seizure vs pre-seizure coherence for one seizure.
    # Slice out the seizure and pre-seizure windows (half-open, 0-based sample indices).
    sz1 = data1_full[seizure_start_sample:seizure_end_sample]
    sz2 = data2_full[seizure_start_sample:seizure_end_sample]
    pre1 = data1_full[pre_start_sample:pre_end_sample]
    pre2 = data2_full[pre_start_sample:pre_end_sample]

    # MSC for the pre and during-seizure windows separately.
    freqs_seizure, msc_seizure = compute_msc(sz1, sz2, samp_freq, window, noverlap, nfft)
    freqs_pre, msc_pre = compute_msc(pre1, pre2, samp_freq, window, noverlap, nfft)

    phase_sz  = compute_phase_delay(sz1, sz2, samp_freq, window, noverlap, nfft, band_low_hz=band_low[0], band_high_hz=band_low[1], msc_threshold=phase_msc_threshold)
    phase_pre = compute_phase_delay(pre1, pre2, samp_freq, window, noverlap, nfft, band_low_hz=band_low[0], band_high_hz=band_low[1], msc_threshold=phase_msc_threshold)

    return {
        # MSC band averages and raw curves.
        "avg_msc_seizure_low": _band_average(freqs_seizure, msc_seizure, band_low, True),
        "avg_msc_pre_low": _band_average(freqs_pre, msc_pre, band_low, True),
        "avg_msc_seizure_high": _band_average(freqs_seizure, msc_seizure, band_high, False),
        "avg_msc_pre_high": _band_average(freqs_pre, msc_pre, band_high, False),
        "freqs_seizure": freqs_seizure,
        "msc_seizure": msc_seizure,
        "freqs_pre": freqs_pre,
        "msc_pre": msc_pre,
        # Phase-delay summary.
        "tau_ms_seizure": phase_sz["tau_ms"],
        "tau_ms_pre": phase_pre["tau_ms"],
        "phase_r2_seizure": phase_sz["r_squared"],
        "phase_r2_pre": phase_pre["r_squared"],
        "phase_nbins_seizure": phase_sz["n_bins_used"],
        "phase_nbins_pre": phase_pre["n_bins_used"],
        # Raw phase-delay points for plotting.
        "phase_sz_freqs": phase_sz["freqs_used"],
        "phase_sz_values": phase_sz["phase_used"],
        "phase_pre_freqs": phase_pre["freqs_used"],
        "phase_pre_values": phase_pre["phase_used"],
    }

def compute_phase_delay(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: int = 256,
    noverlap: int = 128,
    nfft: int = 1024,
    band_low_hz: float = 0.0,
    band_high_hz: float = 14.0,
    msc_threshold: float = 0.1,
) -> dict:
    win = hann(window, sym=True)

    # The cross-spectrum is complex; its angle is the phase difference per frequency.
    freqs, Pxy = csd(x, y, fs = fs, window = win, noverlap = noverlap, nfft = nfft)

    # MSC, used below to gate which bins are trustworthy.
    _, msc = compute_msc(x, y, fs, window, noverlap, nfft)

    # Keep only bins in the band that also clear the coherence threshold.
    band_mask = (freqs >= band_low_hz) & (freqs <= band_high_hz)
    msc_mask = msc >= msc_threshold
    mask = band_mask & msc_mask

    # Need at least 3 surviving bins to fit a meaningful straight line.
    n_bins = int(np.sum(mask))

    if n_bins < 3:
        return {
            "tau_ms": float("nan"),
            "n_bins_used": n_bins,
            "r_squared": float("nan"),
            "freqs_used": np.array([]),
            "phase_used": np.array([]),
            "slope": float("nan"),
        }

    # Bins that passed the mask.
    freqs_masked = freqs[mask]

    # Phase angle from the complex cross-spectrum, wrapped into [-pi, pi].
    phase_raw = np.angle(Pxy[mask])

    # Unwrap so the phase is continuous across bins.
    phase_unwrap = np.unwrap(phase_raw)

    # Weight each bin by its MSC so high-coherence bins drive the fit.
    weights = msc[mask]

    # Weighted straight-line fit of phase vs frequency.
    coeffs = np.polyfit(freqs_masked, phase_unwrap, 1, w=weights)
    slope = coeffs[0]

    # Convert the phase slope to a time delay (see arxiv.org/abs/0712.2352).
    tau_s = -slope / (2 * np.pi)
    tau_ms = tau_s * 1000

    # R-squared of the fit.
    predicted = np.polyval(coeffs, freqs_masked)
    ss_res = np.sum(weights * (phase_unwrap - predicted) ** 2)
    ss_tot = np.sum(weights * (phase_unwrap - np.average(phase_unwrap, weights=weights)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "tau_ms": tau_ms,
        "n_bins_used": n_bins,
        "r_squared": r_squared,
        "freqs_used": freqs_masked,
        "phase_used": phase_unwrap,
        "slope": slope,  # rad/Hz (= -2*pi*tau)
    }
