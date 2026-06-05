"""All plot builders for the pipeline.

Every function returns a Matplotlib ``Figure`` (or draws onto supplied axes) and
performs no I/O; saving and displaying are the caller's job. A shared style block
at the top keeps fonts, sizes, and colours consistent across every figure.
"""

from __future__ import annotations

import math
from datetime import datetime

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Shared style / theme
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2,
    }
)

CH1_COLOR = "red"          # First EEG channel
CH2_COLOR = "blue"         # Second EEG channel
SEIZURE_COLOR = "red"      # Seizure onset/offset markers
THRESHOLD_COLOR = "blue"   # Correlation threshold lines
MEAN_COLOR = "red"         # Mean reference lines
PRE_COLOR = "blue"         # Pre-seizure series
DURING_COLOR = "red"       # During-seizure series
# Defaults below mirror config.py; the notebook threads the config values in as
# args (keeping this module config-free, like the analysis modules).
RACINE_CMAP = "viridis"      # Default colormap for Racine-coded scatter
RACINE_SCALE_MAX = 7         # Default Racine severity scale upper bound
ANIMAL_CMAP = "tab10"        # Qualitative colormap for per-animal series
R2_CMAP = "viridis"          # Fit-quality (R^2) colour scale for phase-delay points
# Marker shapes used to distinguish animals when colour already encodes R^2.
_ANIMAL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
# Marker-area mapping for coherent-bin counts: area = base + scale * bins.
_BIN_SIZE_BASE = 25.0
_BIN_SIZE_SCALE = 7.0

# (year, month, day) -- must match config.DAY_ZERO / offset_to_day_zero's epoch.
_DEFAULT_DAY_ZERO = (2000, 1, 1)


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------
def _format_date_axis(ax) -> None:
    """Apply the shared mm/dd HH:MM datetime formatting + rotation to an axis."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")


def _format_day_axis(ax, day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO) -> None:
    """Label an axis of normalized datetimes as 'Day N' from the Day-0 epoch.

    ``day_zero`` is a ``(year, month, day)`` tuple and must match the epoch used
    by ``offset_to_day_zero`` when the ``Normalized_DT`` values were built.
    """
    epoch = datetime(*day_zero)

    def _fmt(x, _pos):
        days = (mdates.num2date(x).replace(tzinfo=None) - epoch).days
        return f"Day {days}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt))


def _animal_colors(animal_ids):
    """Return a stable {animal_id: color} mapping from the qualitative cmap."""
    cmap = plt.get_cmap(ANIMAL_CMAP)
    uniq = list(dict.fromkeys(animal_ids))
    return {a: cmap(i % cmap.N) for i, a in enumerate(uniq)}


def _bins_to_size(bins):
    """Map coherent-bin counts to scatter marker areas (more bins -> bigger)."""
    return _BIN_SIZE_BASE + _BIN_SIZE_SCALE * np.asarray(bins, dtype=float)


def _size_to_bins(sizes):
    """Inverse of ``_bins_to_size`` -- used to label the marker-size legend."""
    return (np.asarray(sizes) - _BIN_SIZE_BASE) / _BIN_SIZE_SCALE


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def plot_overview(
    data1,
    data2,
    seizures: pd.DataFrame,
    corrected_start,
    samp_freq: float,
    ch1: int,
    ch2: int,
    file_name: str,
    animal_no,
    rec_order,
    downsample_factor: int,
    num_mins_per_subplot: int,
    num_subplots: int,
):
    """60-minute EEG overview: ``num_subplots`` stacked decimated traces.

    Each subplot shows ``num_mins_per_subplot`` minutes of both channels with
    seizure onset/offset markers and labels overlaid where they fall.
    """
    samples_per_subplot = int(num_mins_per_subplot * 60 * samp_freq)

    fig, axes = plt.subplots(num_subplots, 1, figsize=(14, 11))
    fig.suptitle(
        f"File: {file_name} | Animal: {animal_no} | "
        f"Rec Order: {rec_order} (Ch {ch1}, {ch2})"
    )

    for plot_number, ax in enumerate(axes):
        start_idx = plot_number * samples_per_subplot
        end_idx = start_idx + samples_per_subplot
        if end_idx > len(data1):
            ax.set_visible(False)
            continue

        d1_slice = data1[start_idx:end_idx:downsample_factor]
        d2_slice = data2[start_idx:end_idx:downsample_factor]
        time_axis = np.linspace(0, num_mins_per_subplot, len(d1_slice))

        ax.plot(time_axis, d1_slice, color=CH1_COLOR, label=f"Ch {ch1}", linewidth=0.6)
        ax.plot(time_axis, d2_slice, color=CH2_COLOR, label=f"Ch {ch2}", linewidth=0.6)

        min_start = plot_number * num_mins_per_subplot
        min_end = (plot_number + 1) * num_mins_per_subplot

        for _, row in seizures.iterrows():
            start_off_min = (row["SeizureStartDT"] - corrected_start).total_seconds() / 60.0
            end_off_min = (row["SeizureEndDT"] - corrected_start).total_seconds() / 60.0

            if min_start <= start_off_min < min_end:
                local_start = start_off_min - min_start
                ax.axvline(local_start, color="black", linewidth=1.5)
                ax.annotate(
                    f" Seizure:{row['Seizure']} (Racine:{row['Racine_Scale']}) ",
                    xy=(local_start, 4.5),
                    color="black",
                    fontsize=8,
                    ha="left",
                )
            if min_start <= end_off_min < min_end:
                ax.axvline(end_off_min - min_start, color="black", linewidth=1.0)

        ax.set_ylim(-5, 5)
        ax.set_xticks(range(0, num_mins_per_subplot + 1))
        ax.set_ylabel(f"Min {min_start}-{min_end}")

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


# ---------------------------------------------------------------------------
# Correlation + zoom
# ---------------------------------------------------------------------------
def plot_correlation(
    corrs,
    time_axis,
    seizure_start: float,
    seizure_end: float,
    corr_threshold: float,
    title: str | None = None,
    ax=None,
):
    """Rolling correlation trace with seizure span and threshold lines.

    ``time_axis``, ``seizure_start`` and ``seizure_end`` are all in seconds from
    the start of the correlation window. Draws onto ``ax`` if given, otherwise
    creates and returns a standalone figure's axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))

    ax.plot(time_axis, corrs, color="black", linewidth=1.2)
    ax.axvline(seizure_start, color=SEIZURE_COLOR, linewidth=2)
    ax.axvline(seizure_end, color=SEIZURE_COLOR, linewidth=2)
    ax.axhline(corr_threshold, color=THRESHOLD_COLOR, linestyle="--")
    ax.axhline(-corr_threshold, color=THRESHOLD_COLOR, linestyle="--")
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Correlation")
    if title:
        ax.set_title(title)
    return ax


def plot_zoomed_segments(
    raw_seiz1,
    raw_seiz2,
    samp_freq: float,
    segment_secs: float,
    axes=None,
    fig=None,
):
    """Zoomed overlays of the raw seizure signal, one axis per time segment.

    Splits the seizure into ``segment_secs`` chunks and plots both channels in
    each. Draws onto ``axes`` if supplied; otherwise builds a standalone figure.
    """
    samples_per_seg = int(math.floor(segment_secs * samp_freq))
    num_segments = max(1, math.ceil(len(raw_seiz1) / samples_per_seg))

    if axes is None:
        fig, axes = plt.subplots(num_segments, 1, figsize=(9, 2 * num_segments))
        if num_segments == 1:
            axes = [axes]

    for i, ax in enumerate(axes):
        s_ptr = i * samples_per_seg
        e_ptr = min((i + 1) * samples_per_seg, len(raw_seiz1))
        seg1 = raw_seiz1[s_ptr:e_ptr]
        seg2 = raw_seiz2[s_ptr:e_ptr]
        t_slice = i * segment_secs + np.arange(len(seg1)) / samp_freq

        ax.plot(t_slice, seg1, color=CH1_COLOR, linewidth=0.6, alpha=0.8)
        ax.plot(t_slice, seg2, color=CH2_COLOR, linewidth=0.6, alpha=0.8)
        ax.set_ylim(-2, 2)
        ax.set_ylabel(f"Sec {int(i * segment_secs)}")

        if i < len(axes) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Seconds from Start")

    return axes


def assemble_correlation_figure(
    corrs,
    time_axis,
    seizure_start: float,
    seizure_end: float,
    raw_seiz1,
    raw_seiz2,
    samp_freq: float,
    corr_threshold: float,
    segment_secs: float,
    title: str | None = None,
):
    """Combine the correlation trace and the zoomed segments into one figure.

    Matplotlib cannot nest pre-built figures, so this builds a single
    ``GridSpec`` figure (correlation on top, one zoom row per segment) and lets
    ``plot_correlation`` / ``plot_zoomed_segments`` draw onto its axes.
    """
    samples_per_seg = int(math.floor(segment_secs * samp_freq))
    num_segments = max(1, math.ceil(len(raw_seiz1) / samples_per_seg))

    img_height = (300 + num_segments * 150) / 100.0
    fig = plt.figure(figsize=(9, img_height))
    height_ratios = [0.25] + [0.75 / num_segments] * num_segments
    gs = GridSpec(num_segments + 1, 1, height_ratios=height_ratios, figure=fig)

    corr_ax = fig.add_subplot(gs[0])
    plot_correlation(
        corrs, time_axis, seizure_start, seizure_end, corr_threshold,
        title=title, ax=corr_ax,
    )

    zoom_axes = [fig.add_subplot(gs[i + 1]) for i in range(num_segments)]
    plot_zoomed_segments(raw_seiz1, raw_seiz2, samp_freq, segment_secs, axes=zoom_axes)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Coherence (single seizure)
# ---------------------------------------------------------------------------
def plot_coherence(
    freqs_pre,
    msc_pre,
    freqs_sz,
    msc_sz,
    samp_freq: float,
    pre_seizure_secs: float = 30,
    title: str | None = None,
):
    """Pre-seizure vs during-seizure MSC across frequency for one seizure."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs_pre, msc_pre, color=PRE_COLOR, linewidth=1.5,
            label=f"Pre-Seizure ({pre_seizure_secs:g}s)")
    ax.plot(freqs_sz, msc_sz, color=DURING_COLOR, linewidth=1.5, label="During Seizure")
    ax.set_xlim(0, samp_freq / 2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("MSC")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Correlation summaries
# ---------------------------------------------------------------------------
def _trend_scatter(ax, x, y, racine, title, ylabel, cmap=RACINE_CMAP, racine_max=RACINE_SCALE_MAX):
    """Shared trend-line + Racine-coloured scatter + mean line panel."""
    ax.plot(x, y, color="gray", linewidth=1.5, alpha=0.4, label="Trend")
    sc = ax.scatter(
        x, y, c=racine, cmap=cmap, vmin=1, vmax=racine_max,
        s=49, zorder=3,
    )
    mean_val = float(np.mean(y))
    ax.axhline(
        mean_val, color=MEAN_COLOR, linestyle="--", linewidth=2,
        label=f"Mean: {round(mean_val, 3)}",
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    return sc


def plot_animal_summary(
    group_df: pd.DataFrame,
    corr_threshold: float,
    cmap: str = RACINE_CMAP,
    racine_max: int = RACINE_SCALE_MAX,
):
    """Per-animal longitudinal correlation summary (4 stacked panels).

    Panels: seizure duration, seconds above threshold, average correlation
    strength, and Racine severity, plotted against recording time.
    """
    group = group_df.sort_values("File_Start_DT")
    animal = group["Animal_ID"].iloc[0]
    ch1 = group["Ch1"].iloc[0]
    ch2 = group["Ch2"].iloc[0]
    x = group["File_Start_DT"]
    racine = group["Racine_Scale"]

    fig, axes = plt.subplots(4, 1, figsize=(9, 16))

    sc = _trend_scatter(
        axes[0], x, group["Seizure_Duration_Sec"], racine,
        f"Seizure Duration | Animal: {animal} (Ch {ch1}, {ch2})", "Duration (Seconds)",
        cmap=cmap, racine_max=racine_max,
    )
    _trend_scatter(
        axes[1], x, group["Seconds_Above_Threshold"], racine,
        f"Time Above Threshold | Animal: {animal} (Ch {ch1}, {ch2})",
        f"Seconds Above ({corr_threshold})",
        cmap=cmap, racine_max=racine_max,
    )
    _trend_scatter(
        axes[2], x, group["Avg_Corr_Strength"], racine,
        f"Correlation Strength | Animal: {animal} (Ch {ch1}, {ch2})", "Avg |Correlation|",
        cmap=cmap, racine_max=racine_max,
    )
    axes[2].set_ylim(0, 1.05)

    # Racine progression (no colour coding -- it is the colour axis elsewhere).
    axes[3].plot(x, racine, color="gray", linewidth=1.5, alpha=0.5)
    axes[3].scatter(x, racine, color="black", s=49, zorder=3)
    axes[3].set_title(f"Seizure Severity | Animal: {animal} (Ch {ch1}, {ch2})")
    axes[3].set_ylabel("Racine Scale")
    axes[3].set_ylim(0, racine_max + 1)
    axes[3].set_yticks(range(0, racine_max + 1))

    for ax in axes:
        ax.set_xlabel("Date/Time")
        _format_date_axis(ax)

    fig.colorbar(sc, ax=axes.tolist(), label="Racine", fraction=0.025, pad=0.02)
    fig.suptitle(f"Longitudinal Analysis | Animal {animal}")
    return fig


def plot_global_summary(
    normalized_df: pd.DataFrame,
    racine_max: int = RACINE_SCALE_MAX,
    day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO,
):
    """Global correlation summary: all animals offset to Day 0, coloured by animal."""
    colors = _animal_colors(normalized_df["Animal_ID"])
    x_all = normalized_df["Normalized_DT"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 16))
    panels = [
        ("Seizure_Duration_Sec", "Global Duration (Normalized Start)", "Seconds", None),
        ("Seconds_Above_Threshold", "Global Time Above Threshold", "Seconds", None),
        ("Avg_Corr_Strength", "Global Correlation Strength", "Avg |Corr|", (0, 1.05)),
        ("Racine_Scale", "Global Seizure Severity", "Racine", (0, racine_max + 1)),
    ]

    for ax, (col, title, ylabel, ylim) in zip(axes, panels):
        ax.plot(x_all, normalized_df[col], color="gray", linewidth=1.0, alpha=0.2)
        for animal, group in normalized_df.groupby("Animal_ID"):
            ax.scatter(
                group["Normalized_DT"], group[col],
                color=colors[animal], s=36, alpha=0.7, label=str(animal),
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        _format_day_axis(ax, day_zero)

    axes[3].set_yticks(range(0, racine_max + 1))
    axes[0].legend(title="Animal", loc="center left", bbox_to_anchor=(1.0, 0.5))
    axes[3].set_xlabel("Experimental Days (Starting from Day 0)")

    fig.suptitle("Population Analysis: Offset to Experimental Day 0")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


# ---------------------------------------------------------------------------
# Coherence summaries (mirror the correlation summaries)
# ---------------------------------------------------------------------------
def _band_label(band) -> str:
    """Human-readable band label, e.g. (0.0, 14.0) -> '0-14 Hz'."""
    lo, hi = band
    return f"{lo:g}-{hi:g} Hz"


def plot_coherence_animal_summary(
    group_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    band_high=(14.0, 55.0),
    pre_seizure_secs: float = 30,
):
    """Per-animal average-coherence trends over time, split by frequency band.

    Two stacked panels (low band on top, high band below); each shows the
    during-seizure vs pre-seizure band-averaged MSC against recording time.
    """
    group = group_df.sort_values("File_Start_DT")
    animal = group["Animal_ID"].iloc[0]
    x = group["File_Start_DT"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    panels = [
        (band_low, "Avg_MSC_Seizure_Low", "Avg_MSC_Pre_Low"),
        (band_high, "Avg_MSC_Seizure_High", "Avg_MSC_Pre_High"),
    ]

    for ax, (band, sz_col, pre_col) in zip(axes, panels):
        ax.plot(x, group[sz_col], color=DURING_COLOR, marker="o",
                linewidth=1.5, label="During Seizure")
        ax.plot(x, group[pre_col], color=PRE_COLOR, marker="o",
                linewidth=1.5, label=f"Pre-Seizure ({pre_seizure_secs:g}s)")
        ax.set_title(f"Average Coherence ({_band_label(band)}) | Animal: {animal}")
        ax.set_ylabel("Avg MSC")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Date/Time")
        ax.legend(loc="upper left")
        _format_date_axis(ax)

    fig.suptitle(f"Longitudinal Coherence | Animal {animal}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_coherence_ratio_animal_summary(
    group_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    band_high=(14.0, 55.0),
):
    """Per-animal seizure/pre-seizure MSC ratio over time, split by frequency band.

    Two stacked panels (low band on top, high band below); each shows the ratio
    of during-seizure to pre-seizure band-averaged MSC against recording time.
    A dashed reference line at ratio=1 marks equal coherence (above => the
    seizure is more coherent than its 30s baseline).
    """
    group = group_df.sort_values("File_Start_DT")
    animal = group["Animal_ID"].iloc[0]
    x = group["File_Start_DT"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    panels = [
        (band_low, "Avg_MSC_Seizure_Low", "Avg_MSC_Pre_Low"),
        (band_high, "Avg_MSC_Seizure_High", "Avg_MSC_Pre_High"),
    ]

    for ax, (band, sz_col, pre_col) in zip(axes, panels):
        ratio = group[sz_col] / group[pre_col]
        ax.plot(x, ratio, color=DURING_COLOR, marker="o", linewidth=1.5,
                label="Seizure / Pre-Seizure")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Equal coherence")
        ax.set_title(f"Coherence Ratio ({_band_label(band)}) | Animal: {animal}")
        ax.set_ylabel("Seizure MSC / Pre-Seizure MSC")
        ax.set_xlabel("Date/Time")
        ax.legend(loc="upper left")
        _format_date_axis(ax)

    fig.suptitle(f"Longitudinal Coherence Ratio | Animal {animal}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_coherence_global_summary(
    normalized_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    band_high=(14.0, 55.0),
    day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO,
):
    """Global band-averaged coherence: all animals offset to Day 0, by animal.

    Two stacked panels (low band, high band). Within each, during-seizure values
    use filled markers and pre-seizure values use faded open markers in the same
    per-animal colour.
    """
    colors = _animal_colors(normalized_df["Animal_ID"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    panels = [
        (band_low, "Avg_MSC_Seizure_Low", "Avg_MSC_Pre_Low"),
        (band_high, "Avg_MSC_Seizure_High", "Avg_MSC_Pre_High"),
    ]

    for ax, (band, sz_col, pre_col) in zip(axes, panels):
        for animal, group in normalized_df.groupby("Animal_ID"):
            ax.scatter(group["Normalized_DT"], group[sz_col], color=colors[animal],
                       s=42, alpha=0.85, label=f"{animal} (seiz)")
            ax.scatter(group["Normalized_DT"], group[pre_col], facecolors="none",
                       edgecolors=colors[animal], s=42, alpha=0.5, label=f"{animal} (pre)")
        ax.set_title(f"Global Average Coherence ({_band_label(band)})")
        ax.set_ylabel("Avg MSC")
        ax.set_ylim(0, 1.05)
        _format_day_axis(ax, day_zero)

    axes[0].legend(title="Animal", loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    axes[1].set_xlabel("Experimental Days (Starting from Day 0)")

    fig.suptitle("Population Coherence: Offset to Experimental Day 0")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_coherence_ratio_global_summary(
    normalized_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    band_high=(14.0, 55.0),
    day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO,
):
    """Global seizure/pre-seizure MSC ratio: all animals offset to Day 0, by animal.

    Two stacked panels (low band, high band). Each animal's seizure-to-pre ratio
    is scattered against experimental day in its own colour, with a dashed
    reference line at ratio=1 (above => seizure more coherent than its baseline).
    """
    colors = _animal_colors(normalized_df["Animal_ID"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    panels = [
        (band_low, "Avg_MSC_Seizure_Low", "Avg_MSC_Pre_Low"),
        (band_high, "Avg_MSC_Seizure_High", "Avg_MSC_Pre_High"),
    ]

    for ax, (band, sz_col, pre_col) in zip(axes, panels):
        for animal, group in normalized_df.groupby("Animal_ID"):
            ratio = group[sz_col] / group[pre_col]
            ax.scatter(group["Normalized_DT"], ratio, color=colors[animal],
                       s=42, alpha=0.85, label=f"{animal}")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Global Coherence Ratio ({_band_label(band)})")
        ax.set_ylabel("Seizure MSC / Pre-Seizure MSC")
        _format_day_axis(ax, day_zero)

    axes[0].legend(title="Animal", loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    axes[1].set_xlabel("Experimental Days (Starting from Day 0)")

    fig.suptitle("Population Coherence Ratio: Offset to Experimental Day 0")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


# ---------------------------------------------------------------------------
# Phase time-delay summaries (mirror the coherence summaries)
# ---------------------------------------------------------------------------
def _bin_size_legend(ax, sc, loc):
    """Attach a marker-size legend (coherent bins used) and return the artist."""
    handles, labels = sc.legend_elements(
        prop="sizes", num=4, alpha=0.6, color="gray",
        func=_size_to_bins, fmt="{x:.0f}",
    )
    leg = ax.legend(handles, labels, title="Bins used", loc=loc,
                    labelspacing=1.3, borderpad=0.8, framealpha=0.9)
    ax.add_artist(leg)
    return leg


def plot_phase_delay_animal_summary(
    group_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    pre_seizure_secs: float = 30,
    cmap: str = R2_CMAP,
):
    """Per-animal phase-derived time delay over time, weighted by fit confidence.

    Two stacked panels against recording time -- during-seizure on top,
    pre-seizure baseline below -- one point per seizure in each. Every point is
    the inter-channel time delay (ms), resting on a dashed zero line (positive
    => channel 1 leads), **coloured by the phase-slope fit R²** (shared colorbar)
    and **sized by the number of coherent frequency bins** that entered the fit,
    so unreliable estimates (dark, small) visually recede and trustworthy ones
    (bright, large) stand out. The delay is fit over the ``band_low`` band only;
    seizures with too few coherent bins are NaN and simply don't appear.
    """
    group = group_df.sort_values("File_Start_DT")
    animal = group["Animal_ID"].iloc[0]
    x = group["File_Start_DT"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    panels = [
        ("Tau_ms_Seizure", "Phase_R2_Seizure", "Phase_Nbins_Seizure",
         f"During Seizure ({_band_label(band_low)})"),
        ("Tau_ms_Pre", "Phase_R2_Pre", "Phase_Nbins_Pre",
         f"Pre-Seizure {pre_seizure_secs:g}s ({_band_label(band_low)})"),
    ]

    sc = None
    for ax, (tau_col, r2_col, nbin_col, title) in zip(axes, panels):
        # colour = fit R^2, size = coherent bins used.
        sc = ax.scatter(
            x, group[tau_col], c=group[r2_col], cmap=cmap, vmin=0.0, vmax=1.0,
            s=_bins_to_size(group[nbin_col]), alpha=0.9, edgecolors="black",
            linewidths=0.4, zorder=3,
        )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{title} | Animal: {animal}")
        ax.set_ylabel("Time Delay (ms)")
        _bin_size_legend(ax, sc, loc="upper right")
        _format_date_axis(ax)

    axes[-1].set_xlabel("Date/Time")

    # One colorbar shared across both panels (both use the same 0-1 R^2 scale).
    cbar = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("Phase-slope fit R²")

    fig.suptitle(f"Longitudinal Phase Delay | Animal {animal}")
    return fig


def plot_phase_delay_global_summary(
    normalized_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO,
    cmap: str = R2_CMAP,
):
    """Global phase-derived time delay: all animals offset to Day 0.

    A single panel against experimental day, one point per seizure (during
    seizure only). Confidence is encoded the same way as the per-animal plot:
    **colour = phase-slope fit R²** (colorbar) and **size = coherent bins used**.
    Animals are distinguished by **marker shape** (legend), so the colour channel
    stays free for fit quality. A dashed zero line marks no delay.
    """
    animals = list(dict.fromkeys(normalized_df["Animal_ID"]))
    marker_for = {a: _ANIMAL_MARKERS[i % len(_ANIMAL_MARKERS)] for i, a in enumerate(animals)}

    fig, ax = plt.subplots(figsize=(12, 7))

    sc = None
    for a in animals:
        g = normalized_df[normalized_df["Animal_ID"] == a]
        sc = ax.scatter(
            g["Normalized_DT"], g["Tau_ms_Seizure"], c=g["Phase_R2_Seizure"],
            cmap=cmap, vmin=0.0, vmax=1.0, s=_bins_to_size(g["Phase_Nbins_Seizure"]),
            marker=marker_for[a], alpha=0.9, edgecolors="black", linewidths=0.4,
        )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Global Phase Time Delay ({_band_label(band_low)})")
    ax.set_ylabel("Time Delay (ms)")
    ax.set_xlabel("Experimental Days (Starting from Day 0)")
    _format_day_axis(ax, day_zero)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Phase-slope fit R²")

        # Animal legend = marker shape (neutral colour); bin-size key beneath it.
        shape_handles = [
            plt.Line2D([0], [0], marker=marker_for[a], linestyle="none",
                       markerfacecolor="gray", markeredgecolor="black",
                       markersize=8, label=str(a))
            for a in animals
        ]
        animal_leg = ax.legend(handles=shape_handles, title="Animal", loc="upper left",
                               framealpha=0.9)
        ax.add_artist(animal_leg)
        _bin_size_legend(ax, sc, loc="upper right")

    fig.suptitle("Population Phase Delay: Offset to Experimental Day 0")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig
