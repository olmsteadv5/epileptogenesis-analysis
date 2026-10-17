
from __future__ import annotations

import numpy as np
from scipy.signal import coherence, csd
from scipy.signal.windows import hann

# decision: bin count where support hits half weight in the confidence score
PHASE_BIN_HALF_SAT = 10

def compute_msc(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: int = 256,
    noverlap: int = 128,
    nfft: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:

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
    lo, hi = band

    # decision: half-open band so the boundary bin isn't counted twice
    if include_low:
        mask = (freqs >= lo) & (freqs <= hi)
    else:
        mask = (freqs > lo) & (freqs <= hi)

    # explainer: a band can come out empty
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
    # explainer: half-open 0-based sample slices
    sz1 = data1_full[seizure_start_sample:seizure_end_sample]
    sz2 = data2_full[seizure_start_sample:seizure_end_sample]
    pre1 = data1_full[pre_start_sample:pre_end_sample]
    pre2 = data2_full[pre_start_sample:pre_end_sample]

    freqs_seizure, msc_seizure = compute_msc(sz1, sz2, samp_freq, window, noverlap, nfft)
    freqs_pre, msc_pre = compute_msc(pre1, pre2, samp_freq, window, noverlap, nfft)

    phase_sz  = compute_phase_delay(sz1, sz2, samp_freq, window, noverlap, nfft, band_low_hz=band_low[0], band_high_hz=band_low[1], msc_threshold=phase_msc_threshold)
    phase_pre = compute_phase_delay(pre1, pre2, samp_freq, window, noverlap, nfft, band_low_hz=band_low[0], band_high_hz=band_low[1], msc_threshold=phase_msc_threshold)

    return {
        "avg_msc_seizure_low": _band_average(freqs_seizure, msc_seizure, band_low, True),
        "avg_msc_pre_low": _band_average(freqs_pre, msc_pre, band_low, True),
        "avg_msc_seizure_high": _band_average(freqs_seizure, msc_seizure, band_high, False),
        "avg_msc_pre_high": _band_average(freqs_pre, msc_pre, band_high, False),
        "freqs_seizure": freqs_seizure,
        "msc_seizure": msc_seizure,
        "freqs_pre": freqs_pre,
        "msc_pre": msc_pre,
        "tau_ms_seizure": phase_sz["tau_ms"],
        "tau_ms_pre": phase_pre["tau_ms"],
        "phase_r2_seizure": phase_sz["r_squared"],
        "phase_r2_pre": phase_pre["r_squared"],
        "phase_nbins_seizure": phase_sz["n_bins_used"],
        "phase_nbins_pre": phase_pre["n_bins_used"],
        "phase_confidence_seizure": phase_sz["confidence"],
        "phase_confidence_pre": phase_pre["confidence"],
        "phase_sz_freqs": phase_sz["freqs_used"],
        "phase_sz_values": phase_sz["phase_used"],
        "phase_pre_freqs": phase_pre["freqs_used"],
        "phase_pre_values": phase_pre["phase_used"],
        # decision: sub-threshold bins kept only so the plot can show them in grey
        "phase_sz_unused_freqs": phase_sz["freqs_unused"],
        "phase_sz_unused_values": phase_sz["phase_unused"],
        "phase_pre_unused_freqs": phase_pre["freqs_unused"],
        "phase_pre_unused_values": phase_pre["phase_unused"],
        # decision: fit coefficients saved so the line can be redrawn per seizure
        "phase_sz_slope": phase_sz["slope"],
        "phase_sz_intercept": phase_sz["intercept"],
        "phase_pre_slope": phase_pre["slope"],
        "phase_pre_intercept": phase_pre["intercept"],
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

    # explainer: cross-spectrum angle is the phase difference per frequency
    freqs, Pxy = csd(x, y, fs = fs, window = win, noverlap = noverlap, nfft = nfft)

    _, msc = compute_msc(x, y, fs, window, noverlap, nfft)

    # decision: only bins in the band that also clear the coherence gate
    band_mask = (freqs >= band_low_hz) & (freqs <= band_high_hz)
    msc_mask = msc >= msc_threshold
    mask = band_mask & msc_mask

    unused_mask = band_mask & ~msc_mask
    freqs_unused = freqs[unused_mask]
    phase_unused = np.angle(Pxy[unused_mask])

    # decision: need at least 3 bins for a line worth fitting
    n_bins = int(np.sum(mask))

    if n_bins < 3:
        return {
            "tau_ms": float("nan"),
            "n_bins_used": n_bins,
            "r_squared": float("nan"),
            "confidence": float("nan"),
            "freqs_used": np.array([]),
            "phase_used": np.array([]),
            "freqs_unused": freqs_unused,
            "phase_unused": phase_unused,
            "slope": float("nan"),
            "intercept": float("nan"),
        }

    freqs_masked = freqs[mask]

    phase_raw = np.angle(Pxy[mask])

    # explainer: unwrap so phase is continuous across bins
    phase_unwrap = np.unwrap(phase_raw)

    # decision: weight bins by MSC so coherent bins drive the fit
    weights = msc[mask]

    coeffs = np.polyfit(freqs_masked, phase_unwrap, 1, w=weights)
    slope = coeffs[0]
    intercept = coeffs[1]

    # explainer: phase slope to time delay, see arxiv.org/abs/0712.2352
    tau_s = -slope / (2 * np.pi)
    tau_ms = tau_s * 1000

    predicted = np.polyval(coeffs, freqs_masked)
    ss_res = np.sum(weights * (phase_unwrap - predicted) ** 2)
    ss_tot = np.sum(weights * (phase_unwrap - np.average(phase_unwrap, weights=weights)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # decision: confidence in [0, 1] = geometric mean of fit linearity, bin coherence and bin support
    if np.isfinite(r_squared):
        r2_term = min(max(r_squared, 0.0), 1.0)
        msc_term = float(np.average(weights))
        # explainer: saturates toward 1, half weight at PHASE_BIN_HALF_SAT bins
        support_term = n_bins / (n_bins + PHASE_BIN_HALF_SAT)
        confidence = float((r2_term * msc_term * support_term) ** (1.0 / 3.0))
    else:
        confidence = float("nan")

    return {
        "tau_ms": tau_ms,
        "n_bins_used": n_bins,
        "r_squared": r_squared,
        "confidence": confidence,
        "freqs_used": freqs_masked,
        "phase_used": phase_unwrap,
        "freqs_unused": freqs_unused,
        "phase_unused": phase_unused,
        "slope": slope,  # explainer: rad/Hz, = -2*pi*tau
        "intercept": intercept,  # explainer: rad, phase at 0 Hz
    }
