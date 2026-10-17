# explainer: theme, constants, helpers, and the stacked-panel summary builder every plot module uses

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import matplotlib.dates as mdates
import numpy as np
from matplotlib import pyplot as plt


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

CH1_COLOR = "red"
CH2_COLOR = "blue"
SEIZURE_COLOR = "red"
THRESHOLD_COLOR = "blue"
MEAN_COLOR = "red"
PRE_COLOR = "blue"
DURING_COLOR = "red"
# decision: defaults mirror config.py, the notebook passes the real values in
RACINE_CMAP = "viridis"
RACINE_SCALE_MAX = 7
ANIMAL_CMAP = "Accent"
R2_CMAP = "viridis"
# decision: marker shape tells animals apart when colour already means R^2
_ANIMAL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
# explainer: area = base + scale * bins
_BIN_SIZE_BASE = 25.0
_BIN_SIZE_SCALE = 7.0
# explainer: area = base + scale * seconds
_DUR_SIZE_BASE = 25.0
_DUR_SIZE_SCALE = 1.5

# explainer: must match config.DAY_ZERO and offset_to_day_zero's epoch
_DEFAULT_DAY_ZERO = (2000, 1, 1)


def _format_date_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")


def _format_day_axis(ax, day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO) -> None:
    # explainer: day_zero must match the epoch offset_to_day_zero used
    epoch = datetime(*day_zero)

    def _fmt(x, _pos):
        days = (mdates.num2date(x).replace(tzinfo=None) - epoch).days
        return f"Day {days}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt))


def _animal_colors(animal_ids):
    # decision: stable per-animal colours so every figure agrees
    cmap = plt.get_cmap(ANIMAL_CMAP)
    uniq = list(dict.fromkeys(animal_ids))
    return {a: cmap(i % cmap.N) for i, a in enumerate(uniq)}


def _bins_to_size(bins):
    return _BIN_SIZE_BASE + _BIN_SIZE_SCALE * np.asarray(bins, dtype=float)


def _size_to_bins(sizes):
    # explainer: inverse of _bins_to_size, for the legend labels
    return (np.asarray(sizes) - _BIN_SIZE_BASE) / _BIN_SIZE_SCALE


def _band_label(band) -> str:
    lo, hi = band
    return f"{lo:g}-{hi:g} Hz"


def _bin_size_legend(ax, sc, loc):
    handles, labels = sc.legend_elements(
        prop="sizes", num=4, alpha=0.6, color="gray",
        func=_size_to_bins, fmt="{x:.0f}",
    )
    leg = ax.legend(handles, labels, title="Bins used", loc=loc,
                    labelspacing=1.3, borderpad=0.8, framealpha=0.9)
    ax.add_artist(leg)
    return leg


def _duration_to_size(secs):
    return _DUR_SIZE_BASE + _DUR_SIZE_SCALE * np.asarray(secs, dtype=float)


def _size_to_duration(sizes):
    # explainer: inverse of _duration_to_size, for the legend labels
    return (np.asarray(sizes) - _DUR_SIZE_BASE) / _DUR_SIZE_SCALE


def _duration_size_legend(ax, sc, loc):
    handles, labels = sc.legend_elements(
        prop="sizes", num=4, alpha=0.6, color="gray",
        func=_size_to_duration, fmt="{x:.0f}s",
    )
    leg = ax.legend(handles, labels, title="Seizure dur.", loc=loc,
                    labelspacing=1.3, borderpad=0.8, framealpha=0.9)
    ax.add_artist(leg)
    return leg


# explainer: a summary is a stack of panels on one x-axis. a scatter returned from draw becomes the shared colorbar
@dataclass
class Panel:
    title: str
    ylabel: str
    draw: Callable  # explainer: per animal: (ax, x, group_df) -> mappable or None
                                   # explainer: global: (ax, normalized_df, colors) -> mappable or None
    ylim: tuple | None = None
    yticks: range | None = None
    after: Callable | None = None  # explainer: (ax, mappable) -> None


def _decorate(ax, panel: Panel) -> None:
    ax.set_title(panel.title)
    ax.set_ylabel(panel.ylabel)
    if panel.ylim is not None:
        ax.set_ylim(*panel.ylim)
    if panel.yticks is not None:
        ax.set_yticks(panel.yticks)


def _as_list(axes, n):
    # explainer: plt.subplots returns a bare Axes when n == 1
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
    # explainer: x is File_Start_DT. colorbar_label switches to constrained layout and one colorbar from the last scatter
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
    # decision: xlim locks the day axis so every sweep frame shares one scale
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
    def after(ax, _mappable):
        ax.legend(title="Animal", loc=loc, bbox_to_anchor=bbox, **kw)
    return after
