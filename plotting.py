"""All plot builders for the pipeline.

Every function returns a Matplotlib ``Figure`` (or draws onto supplied axes) and
performs no I/O; saving and displaying are the caller's job. A shared style block
at the top keeps fonts, sizes, and colours consistent across every figure.

The longitudinal *summary* plots (per-animal and global) are all the same figure
skeleton -- a vertical stack of time-series panels with a shared x-axis, suptitle
and layout. That skeleton lives once in ``longitudinal_animal_figure`` /
``longitudinal_global_figure``; each summary only supplies a list of ``Panel``
specs describing what to draw in each panel. Adding a new metric is one ``Panel``,
not a new bespoke function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.colors as mcolors
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
ANIMAL_CMAP = "Accent"        # Qualitative colormap for per-animal series
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


def _band_label(band) -> str:
    """Human-readable band label, e.g. (0.0, 14.0) -> '0-14 Hz'."""
    lo, hi = band
    return f"{lo:g}-{hi:g} Hz"


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


def plot_msc_by_animal(
    coherence_details,
    band_low=(0.0, 14.0),
    which: str = "seizure",
    msc_ylim=(0.0, 1.05),
    line_alpha: float = 0.45,
):
    """One raw-MSC-vs-frequency panel per animal, matching plot_phase_by_animal.

    Each seizure is one translucent MSC curve over ``band_low``, coloured by
    animal. Same faceting and sorted animal order as the phase grid so the two
    stack column-for-column.
    """
    if which == "seizure":
        f_msc, v_msc = "freqs_seizure", "msc_seizure"
    else:
        f_msc, v_msc = "freqs_pre", "msc_pre"

    lo, hi = band_low
    animals = sorted({d["Animal_ID"] for d in coherence_details})
    colors = _animal_colors(animals)
    n = len(animals)

    fig, raw = plt.subplots(1, n, figsize=(4.2 * n, 4.2),
                            sharex=True, sharey=True, squeeze=False)
    axes = raw[0]
    for ax, animal in zip(axes, animals):
        color = colors[animal]
        for d in coherence_details:
            if d["Animal_ID"] != animal:
                continue
            freqs = np.asarray(d[f_msc]); msc = np.asarray(d[v_msc])
            m = (freqs >= lo) & (freqs <= hi)
            ax.plot(freqs[m], msc[m], color=color, alpha=line_alpha, linewidth=1.0)
        ax.set_title(f"Animal {animal}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(*msc_ylim)
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("MSC")
    fig.suptitle(f"Raw MSC by animal ({_band_label(band_low)}, {which})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig

def plot_phase_by_animal(
    coherence_details,
    band_low=(0.0, 14.0),
    which: str = "seizure",
    align_phase: bool = True,
    phase_ylim=None,
    color_by: str = "msc",     # "msc" -> opacity ; "colorbar" -> viridis + colorbar
    alpha_floor: float = 0.08,
):
    """One phase-vs-frequency panel per animal; each point encodes its own MSC.

    ``coherence_details`` is the list of per-seizure dicts (compute_seizure_coherence
    output + an "Animal_ID" key). With ``color_by="msc"`` a point's opacity is its
    MSC (solid = trustworthy). With ``color_by="colorbar"`` points are coloured by
    MSC on a shared viridis scale instead.
    """
    if which == "seizure":
        f_ph, v_ph = "phase_sz_freqs", "phase_sz_values"
        f_msc, v_msc = "freqs_seizure", "msc_seizure"
    else:
        f_ph, v_ph = "phase_pre_freqs", "phase_pre_values"
        f_msc, v_msc = "freqs_pre", "msc_pre"

    lo, hi = band_low
    animals = sorted({d["Animal_ID"] for d in coherence_details})
    colors = _animal_colors(animals)

    n = len(animals)
    fig, raw_axes = plt.subplots(
        1, n, figsize=(4.2 * n, 4.2), sharex=True, sharey=True, squeeze=False,
        layout="constrained" if color_by == "colorbar" else None,
    )
    axes = raw_axes[0]
    mappable = None

    for ax, animal in zip(axes, animals):
        base = mcolors.to_rgb(colors[animal])
        for d in coherence_details:
            if d["Animal_ID"] != animal:
                continue
            pf = np.asarray(d[f_ph], dtype=float)
            pv = np.asarray(d[v_ph], dtype=float)
            if pf.size < 2:
                continue
            if align_phase:
                pv = pv - pv[0]
            msc_pts = np.interp(pf, np.asarray(d[f_msc]), np.asarray(d[v_msc]))

            if color_by == "msc":
                a = np.clip(msc_pts, alpha_floor, 1.0)
                rgba = np.zeros((len(a), 4)); rgba[:, :3] = base; rgba[:, 3] = a
                ax.plot(pf, pv, color=base, alpha=0.12, linewidth=0.8)
                ax.scatter(pf, pv, c=rgba, s=12, edgecolors="none")
            else:
                ax.plot(pf, pv, color="gray", alpha=0.15, linewidth=0.8)
                mappable = ax.scatter(pf, pv, c=msc_pts, cmap="viridis",
                                      vmin=0.0, vmax=1.0, s=12, edgecolors="none")

        ax.set_title(f"Animal {animal}")
        ax.set_xlim(lo, hi)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        if phase_ylim is not None:
            ax.set_ylim(*phase_ylim)
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Phase (rad)" + (", aligned to 0" if align_phase else ""))

    if color_by == "colorbar" and mappable is not None:
        fig.colorbar(mappable, ax=list(axes), fraction=0.025, pad=0.02).set_label("MSC")
    fig.suptitle(f"Phase responses by animal ({_band_label(band_low)}, {which})")
    if color_by != "colorbar":
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig

def plot_delay_distribution(
    df: pd.DataFrame,
    group_col: str = "Animal_ID",
    value_col: str = "Tau_ms_Seizure",
    r2_col: str = "Phase_R2_Seizure",
    nbins_col: str = "Phase_Nbins_Seizure",
    r2_min: float = 0.0,
    nbins_min: int = 3,
    title: str | None = None,
):
    """Box + strip plot of per-seizure phase delay, grouped by animal.

    Keeps only rows whose fit cleared ``r2_min`` and ``nbins_min`` so the
    distribution reflects trustworthy delays. Dots are individual seizures.
    """
    d = df[df[value_col].notna()].copy()
    if r2_col in d:
        d = d[d[r2_col].fillna(0) >= r2_min]
    if nbins_col in d:
        d = d[d[nbins_col].fillna(0) >= nbins_min]

    groups = sorted(d[group_col].unique())
    data = [d.loc[d[group_col] == g, value_col].values for g in groups]
    colors = _animal_colors(groups)

    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 3, 5))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    bp = ax.boxplot(data, positions=range(len(groups)), widths=0.55,
                    showfliers=False, patch_artist=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(colors[g]); patch.set_alpha(0.35)
    for i, (g, arr) in enumerate(zip(groups, data)):
        jitter = (np.random.default_rng(i).random(len(arr)) - 0.5) * 0.25
        ax.scatter(np.full(len(arr), i) + jitter, arr, color=colors[g],
                   s=18, alpha=0.8, edgecolors="black", linewidths=0.3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"Animal {g}\n(n={len(a)})" for g, a in zip(groups, data)])
    ax.set_ylabel("Phase delay (ms)")
    ax.set_title(title or f"Per-seizure delay by animal "
                          f"(r2>={r2_min}, bins>={nbins_min})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Longitudinal summary framework
# ---------------------------------------------------------------------------
# Every per-animal / global summary is the same figure: a vertical stack of
# time-series panels sharing an x-axis (recording datetime for per-animal views,
# Day-0-normalized datetime for global views) plus a suptitle and shared layout.
# A ``Panel`` describes one row -- its title, y-label, optional y-limits/ticks,
# the ``draw`` callable that actually plots it, and an optional ``after`` hook
# for extra per-panel decoration (e.g. a marker-size legend). ``draw`` may return
# a "mappable" (a scatter) which the builder turns into one shared colorbar.
@dataclass
class Panel:
    title: str
    ylabel: str
    draw: Callable                 # animal: (ax, x, group_df) -> mappable | None
                                   # global: (ax, normalized_df, colors) -> mappable | None
    ylim: tuple | None = None
    yticks: range | None = None
    after: Callable | None = None  # (ax, mappable) -> None


def _decorate(ax, panel: Panel) -> None:
    ax.set_title(panel.title)
    ax.set_ylabel(panel.ylabel)
    if panel.ylim is not None:
        ax.set_ylim(*panel.ylim)
    if panel.yticks is not None:
        ax.set_yticks(panel.yticks)


def _as_list(axes, n):
    """Normalize ``plt.subplots`` return to a list (it's a bare Axes when n==1)."""
    return [axes] if n == 1 else list(axes)


def longitudinal_animal_figure(
    group_df: pd.DataFrame,
    panels: list[Panel],
    *,
    suptitle: str,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    colorbar_label: str | None = None,
):
    """Build a per-animal summary: panels stacked against recording datetime.

    Rows are sorted by ``File_Start_DT`` and that column is the shared x-axis.
    When ``colorbar_label`` is given the figure uses constrained layout and adds
    one colorbar from the last scatter a panel returned.
    """
    group = group_df.sort_values("File_Start_DT")
    x = group["File_Start_DT"]
    constrained = colorbar_label is not None

    fig, raw_axes = plt.subplots(
        len(panels), 1, figsize=figsize or (9, 4 * len(panels)),
        sharex=sharex, layout="constrained" if constrained else None,
    )
    axes = _as_list(raw_axes, len(panels))

    mappable = None
    for ax, panel in zip(axes, panels):
        m = panel.draw(ax, x, group)
        mappable = m if m is not None else mappable
        _decorate(ax, panel)
        _format_date_axis(ax)
        if panel.after is not None:
            panel.after(ax, m)

    axes[-1].set_xlabel("Date/Time")
    if colorbar_label and mappable is not None:
        fig.colorbar(mappable, ax=axes, fraction=0.046, pad=0.02).set_label(colorbar_label)

    fig.suptitle(suptitle)
    if not constrained:
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def longitudinal_global_figure(
    normalized_df: pd.DataFrame,
    panels: list[Panel],
    *,
    day_zero: tuple[int, int, int],
    suptitle: str,
    figsize: tuple[float, float] | None = None,
    colorbar_label: str | None = None,
):
    """Build a global summary: panels stacked against Day-0-normalized datetime.

    A stable per-animal colour map is passed to every panel's ``draw`` so each
    can colour its own series; the x-axis is labelled in experimental days.
    """
    colors = _animal_colors(normalized_df["Animal_ID"])
    constrained = colorbar_label is not None

    fig, raw_axes = plt.subplots(
        len(panels), 1, figsize=figsize or (11, 4 * len(panels)),
        layout="constrained" if constrained else None,
    )
    axes = _as_list(raw_axes, len(panels))

    mappable = None
    for ax, panel in zip(axes, panels):
        m = panel.draw(ax, normalized_df, colors)
        mappable = m if m is not None else mappable
        _decorate(ax, panel)
        _format_day_axis(ax, day_zero)
        if panel.after is not None:
            panel.after(ax, m)

    axes[-1].set_xlabel("Experimental Days (Starting from Day 0)")
    if colorbar_label and mappable is not None:
        fig.colorbar(mappable, ax=axes, fraction=0.046, pad=0.02).set_label(colorbar_label)

    fig.suptitle(suptitle)
    if not constrained:
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


# ---------------------------------------------------------------------------
# Per-panel drawing strategies
# ---------------------------------------------------------------------------
# Each factory returns a ``draw`` callable matching the signature its builder
# expects (per-animal: (ax, x, group); global: (ax, df, colors)).
def _draw_trend_racine(col, cmap, racine_max):
    """Trend line + Racine-coloured scatter + mean line (per-animal)."""
    def draw(ax, x, g):
        y = g[col]
        ax.plot(x, y, color="gray", linewidth=1.5, alpha=0.4, label="Trend")
        sc = ax.scatter(x, y, c=g["Racine_Scale"], cmap=cmap, vmin=1,
                        vmax=racine_max, s=49, zorder=3)
        mean_val = float(np.mean(y))
        ax.axhline(mean_val, color=MEAN_COLOR, linestyle="--", linewidth=2,
                   label=f"Mean: {round(mean_val, 3)}")
        ax.legend(loc="upper left")
        return sc
    return draw


def _draw_racine_progression():
    """Plain Racine-severity line + black markers (per-animal)."""
    def draw(ax, x, g):
        r = g["Racine_Scale"]
        ax.plot(x, r, color="gray", linewidth=1.5, alpha=0.5)
        ax.scatter(x, r, color="black", s=49, zorder=3)
        return None
    return draw


def _draw_seizure_vs_pre(sz_col, pre_col, pre_secs):
    """During-seizure vs pre-seizure line markers (per-animal)."""
    def draw(ax, x, g):
        ax.plot(x, g[sz_col], color=DURING_COLOR, marker="o", linewidth=1.5,
                label="During Seizure")
        ax.plot(x, g[pre_col], color=PRE_COLOR, marker="o", linewidth=1.5,
                label=f"Pre-Seizure ({pre_secs:g}s)")
        ax.legend(loc="upper left")
        return None
    return draw


def _draw_ratio(sz_col, pre_col):
    """Seizure/pre-seizure ratio line + reference line at 1 (per-animal)."""
    def draw(ax, x, g):
        ratio = g[sz_col] / g[pre_col]
        ax.plot(x, ratio, color=DURING_COLOR, marker="o", linewidth=1.5,
                label="Seizure / Pre-Seizure")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Equal coherence")
        ax.legend(loc="upper left")
        return None
    return draw


def _draw_confidence(tau_col, r2_col, nbin_col, cmap):
    """R^2-coloured, bin-sized time-delay scatter + zero line (per-animal)."""
    def draw(ax, x, g):
        sc = ax.scatter(
            x, g[tau_col], c=g[r2_col], cmap=cmap, vmin=0.0, vmax=1.0,
            s=_bins_to_size(g[nbin_col]), alpha=0.9, edgecolors="black",
            linewidths=0.4, zorder=3,
        )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        return sc
    return draw


def _gdraw_animal_scatter(col):
    """Faint all-points trend line + per-animal coloured scatter (global)."""
    def draw(ax, df, colors):
        ax.plot(df["Normalized_DT"], df[col], color="gray", linewidth=1.0, alpha=0.2)
        for animal, group in df.groupby("Animal_ID"):
            ax.scatter(group["Normalized_DT"], group[col], color=colors[animal],
                       s=36, alpha=0.7, label=str(animal))
        return None
    return draw


def _gdraw_seizure_pre(sz_col, pre_col):
    """Per-animal seizure (filled) vs pre-seizure (open) scatter (global)."""
    def draw(ax, df, colors):
        for animal, group in df.groupby("Animal_ID"):
            ax.scatter(group["Normalized_DT"], group[sz_col], color=colors[animal],
                       s=42, alpha=0.85, label=f"{animal} (seiz)")
            ax.scatter(group["Normalized_DT"], group[pre_col], facecolors="none",
                       edgecolors=colors[animal], s=42, alpha=0.5, label=f"{animal} (pre)")
        return None
    return draw


def _gdraw_ratio(sz_col, pre_col):
    """Per-animal seizure/pre ratio scatter + reference line at 1 (global)."""
    def draw(ax, df, colors):
        for animal, group in df.groupby("Animal_ID"):
            ratio = group[sz_col] / group[pre_col]
            ax.scatter(group["Normalized_DT"], ratio, color=colors[animal],
                       s=42, alpha=0.85, label=f"{animal}")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        return None
    return draw


def _animal_legend(loc="center left", bbox=(1.0, 0.5), **kw):
    """An ``after`` hook adding the per-animal legend to a (first) panel."""
    def after(ax, _mappable):
        ax.legend(title="Animal", loc=loc, bbox_to_anchor=bbox, **kw)
    return after


# ---------------------------------------------------------------------------
# Correlation summaries
# ---------------------------------------------------------------------------
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
    animal = group_df["Animal_ID"].iloc[0]
    ch1, ch2 = group_df["Ch1"].iloc[0], group_df["Ch2"].iloc[0]
    tag = f"Animal: {animal} (Ch {ch1}, {ch2})"

    panels = [
        Panel(f"Seizure Duration | {tag}", "Duration (Seconds)",
              _draw_trend_racine("Seizure_Duration_Sec", cmap, racine_max)),
        Panel(f"Time Above Threshold | {tag}", f"Seconds Above ({corr_threshold})",
              _draw_trend_racine("Seconds_Above_Threshold", cmap, racine_max)),
        Panel(f"Correlation Strength | {tag}", "Avg |Correlation|",
              _draw_trend_racine("Avg_Corr_Strength", cmap, racine_max), ylim=(0, 1.05)),
        Panel(f"Seizure Severity | {tag}", "Racine Scale",
              _draw_racine_progression(), ylim=(0, racine_max + 1),
              yticks=range(0, racine_max + 1)),
    ]
    return longitudinal_animal_figure(
        group_df, panels, figsize=(9, 16), colorbar_label="Racine",
        suptitle=f"Longitudinal Analysis | Animal {animal}",
    )


def plot_global_summary(
    normalized_df: pd.DataFrame,
    racine_max: int = RACINE_SCALE_MAX,
    day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO,
):
    """Global correlation summary: all animals offset to Day 0, coloured by animal."""
    panels = [
        Panel("Global Duration (Normalized Start)", "Seconds",
              _gdraw_animal_scatter("Seizure_Duration_Sec"), after=_animal_legend()),
        Panel("Global Time Above Threshold", "Seconds",
              _gdraw_animal_scatter("Seconds_Above_Threshold")),
        Panel("Global Correlation Strength", "Avg |Corr|",
              _gdraw_animal_scatter("Avg_Corr_Strength"), ylim=(0, 1.05)),
        Panel("Global Seizure Severity", "Racine",
              _gdraw_animal_scatter("Racine_Scale"), ylim=(0, racine_max + 1),
              yticks=range(0, racine_max + 1)),
    ]
    return longitudinal_global_figure(
        normalized_df, panels, day_zero=day_zero, figsize=(11, 16),
        suptitle="Population Analysis: Offset to Experimental Day 0",
    )


# ---------------------------------------------------------------------------
# Coherence summaries (mirror the correlation summaries)
# ---------------------------------------------------------------------------
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
    animal = group_df["Animal_ID"].iloc[0]
    panels = [
        Panel(f"Average Coherence ({_band_label(band_low)}) | Animal: {animal}", "Avg MSC",
              _draw_seizure_vs_pre("Avg_MSC_Seizure_Low", "Avg_MSC_Pre_Low", pre_seizure_secs),
              ylim=(0, 1.05)),
        Panel(f"Average Coherence ({_band_label(band_high)}) | Animal: {animal}", "Avg MSC",
              _draw_seizure_vs_pre("Avg_MSC_Seizure_High", "Avg_MSC_Pre_High", pre_seizure_secs),
              ylim=(0, 1.05)),
    ]
    return longitudinal_animal_figure(
        group_df, panels, figsize=(9, 9),
        suptitle=f"Longitudinal Coherence | Animal {animal}",
    )


def plot_coherence_ratio_animal_summary(
    group_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    band_high=(14.0, 55.0),
):
    """Per-animal seizure/pre-seizure MSC ratio over time, split by frequency band.

    Two stacked panels (low band on top, high band below); each shows the ratio
    of during-seizure to pre-seizure band-averaged MSC against recording time.
    A dashed reference line at ratio=1 marks equal coherence (above => the
    seizure is more coherent than its baseline).
    """
    animal = group_df["Animal_ID"].iloc[0]
    panels = [
        Panel(f"Coherence Ratio ({_band_label(band_low)}) | Animal: {animal}",
              "Seizure MSC / Pre-Seizure MSC",
              _draw_ratio("Avg_MSC_Seizure_Low", "Avg_MSC_Pre_Low")),
        Panel(f"Coherence Ratio ({_band_label(band_high)}) | Animal: {animal}",
              "Seizure MSC / Pre-Seizure MSC",
              _draw_ratio("Avg_MSC_Seizure_High", "Avg_MSC_Pre_High")),
    ]
    return longitudinal_animal_figure(
        group_df, panels, figsize=(9, 9),
        suptitle=f"Longitudinal Coherence Ratio | Animal {animal}",
    )


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
    panels = [
        Panel(f"Global Average Coherence ({_band_label(band_low)})", "Avg MSC",
              _gdraw_seizure_pre("Avg_MSC_Seizure_Low", "Avg_MSC_Pre_Low"),
              ylim=(0, 1.05), after=_animal_legend(ncol=1)),
        Panel(f"Global Average Coherence ({_band_label(band_high)})", "Avg MSC",
              _gdraw_seizure_pre("Avg_MSC_Seizure_High", "Avg_MSC_Pre_High"),
              ylim=(0, 1.05)),
    ]
    return longitudinal_global_figure(
        normalized_df, panels, day_zero=day_zero, figsize=(11, 9),
        suptitle="Population Coherence: Offset to Experimental Day 0",
    )


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
    panels = [
        Panel(f"Global Coherence Ratio ({_band_label(band_low)})",
              "Seizure MSC / Pre-Seizure MSC",
              _gdraw_ratio("Avg_MSC_Seizure_Low", "Avg_MSC_Pre_Low"),
              after=_animal_legend(ncol=1)),
        Panel(f"Global Coherence Ratio ({_band_label(band_high)})",
              "Seizure MSC / Pre-Seizure MSC",
              _gdraw_ratio("Avg_MSC_Seizure_High", "Avg_MSC_Pre_High")),
    ]
    return longitudinal_global_figure(
        normalized_df, panels, day_zero=day_zero, figsize=(11, 9),
        suptitle="Population Coherence Ratio: Offset to Experimental Day 0",
    )


# ---------------------------------------------------------------------------
# Phase time-delay summaries (mirror the coherence summaries)
# ---------------------------------------------------------------------------
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
    animal = group_df["Animal_ID"].iloc[0]
    bin_legend = lambda ax, sc: _bin_size_legend(ax, sc, loc="upper right")

    panels = [
        Panel(f"During Seizure ({_band_label(band_low)}) | Animal: {animal}",
              "Time Delay (ms)",
              _draw_confidence("Tau_ms_Seizure", "Phase_R2_Seizure",
                               "Phase_Nbins_Seizure", cmap),
              after=bin_legend),
        Panel(f"Pre-Seizure {pre_seizure_secs:g}s ({_band_label(band_low)}) | Animal: {animal}",
              "Time Delay (ms)",
              _draw_confidence("Tau_ms_Pre", "Phase_R2_Pre", "Phase_Nbins_Pre", cmap),
              after=bin_legend),
    ]
    return longitudinal_animal_figure(
        group_df, panels, figsize=(11, 10), sharex=True,
        colorbar_label="Phase-slope fit R²",
        suptitle=f"Longitudinal Phase Delay | Animal {animal}",
    )


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

    def draw(ax, df, _colors):
        sc = None
        for a in animals:
            g = df[df["Animal_ID"] == a]
            sc = ax.scatter(
                g["Normalized_DT"], g["Tau_ms_Seizure"], c=g["Phase_R2_Seizure"],
                cmap=cmap, vmin=0.0, vmax=1.0, s=_bins_to_size(g["Phase_Nbins_Seizure"]),
                marker=marker_for[a], alpha=0.9, edgecolors="black", linewidths=0.4,
            )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        return sc

    def after(ax, sc):
        if sc is None:
            return
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

    panels = [
        Panel(f"Global Phase Time Delay ({_band_label(band_low)})", "Time Delay (ms)",
              draw, after=after),
    ]
    return longitudinal_global_figure(
        normalized_df, panels, day_zero=day_zero, figsize=(12, 7),
        colorbar_label="Phase-slope fit R²",
        suptitle="Population Phase Delay: Offset to Experimental Day 0",
    )
