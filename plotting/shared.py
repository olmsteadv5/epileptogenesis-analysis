# Shared theme, colour/marker constants, small helpers, and the longitudinal-summary framework.
# Every plot module pulls its styling and stacked-panel scaffolding from here.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import matplotlib.dates as mdates
import numpy as np
from matplotlib import pyplot as plt


# Shared style / theme
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
# Defaults mirror config.py; the notebook passes the actual config values in as args.
RACINE_CMAP = "viridis"      # Default colormap for Racine-coded scatter
RACINE_SCALE_MAX = 7         # Default Racine severity scale upper bound
ANIMAL_CMAP = "Accent"        # Qualitative colormap for per-animal series
R2_CMAP = "viridis"          # Fit-quality (R^2) colour scale for phase-delay points
# Marker shapes used to distinguish animals when colour already encodes R^2.
_ANIMAL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
# Marker-area mapping for coherent-bin counts: area = base + scale * bins.
_BIN_SIZE_BASE = 25.0
_BIN_SIZE_SCALE = 7.0
# Marker-area mapping for seizure duration (seconds): area = base + scale * seconds.
_DUR_SIZE_BASE = 25.0
_DUR_SIZE_SCALE = 1.5

# (year, month, day); must match config.DAY_ZERO and offset_to_day_zero's epoch.
_DEFAULT_DAY_ZERO = (2000, 1, 1)


# Small internal helpers
def _format_date_axis(ax) -> None:
    # Apply the shared mm/dd HH:MM datetime formatting + rotation to an axis.
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")


def _format_day_axis(ax, day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO) -> None:
    # Label an axis of normalized datetimes as 'Day N' from the Day-0 epoch.
    # day_zero must match the epoch offset_to_day_zero used to build Normalized_DT.
    epoch = datetime(*day_zero)

    def _fmt(x, _pos):
        days = (mdates.num2date(x).replace(tzinfo=None) - epoch).days
        return f"Day {days}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt))


def _animal_colors(animal_ids):
    # Return a stable {animal_id: color} mapping from the qualitative cmap.
    cmap = plt.get_cmap(ANIMAL_CMAP)
    uniq = list(dict.fromkeys(animal_ids))
    return {a: cmap(i % cmap.N) for i, a in enumerate(uniq)}


def _bins_to_size(bins):
    # Map coherent-bin counts to scatter marker areas (more bins -> bigger).
    return _BIN_SIZE_BASE + _BIN_SIZE_SCALE * np.asarray(bins, dtype=float)


def _size_to_bins(sizes):
    # Inverse of _bins_to_size, used to label the marker-size legend.
    return (np.asarray(sizes) - _BIN_SIZE_BASE) / _BIN_SIZE_SCALE


def _band_label(band) -> str:
    # Human-readable band label, e.g. (0.0, 14.0) -> '0-14 Hz'.
    lo, hi = band
    return f"{lo:g}-{hi:g} Hz"


def _bin_size_legend(ax, sc, loc):
    # Attach a marker-size legend (coherent bins used) and return the artist.
    handles, labels = sc.legend_elements(
        prop="sizes", num=4, alpha=0.6, color="gray",
        func=_size_to_bins, fmt="{x:.0f}",
    )
    leg = ax.legend(handles, labels, title="Bins used", loc=loc,
                    labelspacing=1.3, borderpad=0.8, framealpha=0.9)
    ax.add_artist(leg)
    return leg


def _duration_to_size(secs):
    # Map seizure duration (seconds) to scatter marker areas (longer -> bigger).
    return _DUR_SIZE_BASE + _DUR_SIZE_SCALE * np.asarray(secs, dtype=float)


def _size_to_duration(sizes):
    # Inverse of _duration_to_size, used to label the marker-size legend.
    return (np.asarray(sizes) - _DUR_SIZE_BASE) / _DUR_SIZE_SCALE


def _duration_size_legend(ax, sc, loc):
    # Attach a marker-size legend (seizure duration) and return the artist.
    handles, labels = sc.legend_elements(
        prop="sizes", num=4, alpha=0.6, color="gray",
        func=_size_to_duration, fmt="{x:.0f}s",
    )
    leg = ax.legend(handles, labels, title="Seizure dur.", loc=loc,
                    labelspacing=1.3, borderpad=0.8, framealpha=0.9)
    ax.add_artist(leg)
    return leg


# Longitudinal summary framework.
# Each summary is a vertical stack of time-series panels sharing an x-axis. A Panel describes one
# row: its title, y-label, optional limits/ticks, the draw callable, and an optional after hook.
# draw may return a scatter that the builder turns into one shared colorbar.
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
    # Normalize plt.subplots return to a list (it's a bare Axes when n==1).
    return [axes] if n == 1 else list(axes)


def longitudinal_animal_figure(
    group_df,
    panels: list[Panel],
    *,
    suptitle: str,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    colorbar_label: str | None = None,
):
    # Build a per-animal summary: panels stacked against recording datetime.
    # Rows are sorted by File_Start_DT and that column is the shared x-axis. When colorbar_label
    # is given the figure uses constrained layout and adds one colorbar from the last scatter.
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
    normalized_df,
    panels: list[Panel],
    *,
    day_zero: tuple[int, int, int],
    suptitle: str,
    figsize: tuple[float, float] | None = None,
    colorbar_label: str | None = None,
    xlim: tuple | None = None,
):
    # Build a global summary: panels stacked against Day-0-normalized datetime.
    # A stable per-animal colour map is passed to every panel's draw so each can colour its
    # own series; the x-axis is labelled in experimental days.
    # xlim locks the day axis so a sweep of frames stays on one comparable scale.
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
        if xlim is not None:
            ax.set_xlim(*xlim)
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


def _animal_legend(loc="center left", bbox=(1.0, 0.5), **kw):
    # An after hook adding the per-animal legend to a (first) panel.
    def after(ax, _mappable):
        ax.legend(title="Animal", loc=loc, bbox_to_anchor=bbox, **kw)
    return after
