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
RACINE_CMAP = "viridis"    # Colormap for Racine-coded scatter
RACINE_CLIM = (1, 7)       # Colorbar limits for the Racine scale
ANIMAL_CMAP = "tab10"      # Qualitative colormap for per-animal series

_DAY_ZERO = datetime(2000, 1, 1)


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------
def _format_date_axis(ax) -> None:
    """Apply the shared mm/dd HH:MM datetime formatting + rotation to an axis."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")


def _format_day_axis(ax) -> None:
    """Label an axis of normalized datetimes as 'Day N' from the Day-0 epoch."""

    def _fmt(x, _pos):
        days = (mdates.num2date(x).replace(tzinfo=None) - _DAY_ZERO).days
        return f"Day {days}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt))


def _animal_colors(animal_ids):
    """Return a stable {animal_id: color} mapping from the qualitative cmap."""
    cmap = plt.get_cmap(ANIMAL_CMAP)
    uniq = list(dict.fromkeys(animal_ids))
    return {a: cmap(i % cmap.N) for i, a in enumerate(uniq)}


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
    title: str | None = None,
):
    """Pre-seizure vs during-seizure MSC across frequency for one seizure."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs_pre, msc_pre, color=PRE_COLOR, linewidth=1.5, label="Pre-Seizure (30s)")
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
def _trend_scatter(ax, x, y, racine, title, ylabel):
    """Shared trend-line + Racine-coloured scatter + mean line panel."""
    ax.plot(x, y, color="gray", linewidth=1.5, alpha=0.4, label="Trend")
    sc = ax.scatter(
        x, y, c=racine, cmap=RACINE_CMAP, vmin=RACINE_CLIM[0], vmax=RACINE_CLIM[1],
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


def plot_animal_summary(group_df: pd.DataFrame, corr_threshold: float):
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
    )
    _trend_scatter(
        axes[1], x, group["Seconds_Above_Threshold"], racine,
        f"Time Above Threshold | Animal: {animal} (Ch {ch1}, {ch2})",
        f"Seconds Above ({corr_threshold})",
    )
    _trend_scatter(
        axes[2], x, group["Avg_Corr_Strength"], racine,
        f"Correlation Strength | Animal: {animal} (Ch {ch1}, {ch2})", "Avg |Correlation|",
    )
    axes[2].set_ylim(0, 1.05)

    # Racine progression (no colour coding -- it is the colour axis elsewhere).
    axes[3].plot(x, racine, color="gray", linewidth=1.5, alpha=0.5)
    axes[3].scatter(x, racine, color="black", s=49, zorder=3)
    axes[3].set_title(f"Seizure Severity | Animal: {animal} (Ch {ch1}, {ch2})")
    axes[3].set_ylabel("Racine Scale")
    axes[3].set_ylim(0, 8)
    axes[3].set_yticks(range(0, 8))

    for ax in axes:
        ax.set_xlabel("Date/Time")
        _format_date_axis(ax)

    fig.colorbar(sc, ax=axes.tolist(), label="Racine", fraction=0.025, pad=0.02)
    fig.suptitle(f"Longitudinal Analysis | Animal {animal}")
    return fig


def plot_global_summary(normalized_df: pd.DataFrame):
    """Global correlation summary: all animals offset to Day 0, coloured by animal."""
    colors = _animal_colors(normalized_df["Animal_ID"])
    x_all = normalized_df["Normalized_DT"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 16))
    panels = [
        ("Seizure_Duration_Sec", "Global Duration (Normalized Start)", "Seconds", None),
        ("Seconds_Above_Threshold", "Global Time Above Threshold", "Seconds", None),
        ("Avg_Corr_Strength", "Global Correlation Strength", "Avg |Corr|", (0, 1.05)),
        ("Racine_Scale", "Global Seizure Severity", "Racine", (0, 8)),
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
        _format_day_axis(ax)

    axes[3].set_yticks(range(0, 8))
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
                linewidth=1.5, label="Pre-Seizure (30s)")
        ax.set_title(f"Average Coherence ({_band_label(band)}) | Animal: {animal}")
        ax.set_ylabel("Avg MSC")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Date/Time")
        ax.legend(loc="upper left")
        _format_date_axis(ax)

    fig.suptitle(f"Longitudinal Coherence | Animal {animal}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_coherence_global_summary(
    normalized_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    band_high=(14.0, 55.0),
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
        _format_day_axis(ax)

    axes[0].legend(title="Animal", loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    axes[1].set_xlabel("Experimental Days (Starting from Day 0)")

    fig.suptitle("Population Coherence: Offset to Experimental Day 0")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig
