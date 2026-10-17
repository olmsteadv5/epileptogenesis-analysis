
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .shared import (
    DURING_COLOR,
    PRE_COLOR,
    _DEFAULT_DAY_ZERO,
    Panel,
    _animal_colors,
    _animal_legend,
    _band_label,
    longitudinal_animal_figure,
    longitudinal_global_figure,
)


def plot_coherence(
    freqs_pre,
    msc_pre,
    freqs_sz,
    msc_sz,
    samp_freq: float,
    pre_seizure_secs: float = 30,
    title: str | None = None,
):
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
    # decision: same faceting and animal order as plot_phase_by_animal so the two grids stack column for column
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


def _draw_seizure_vs_pre(sz_col, pre_col, pre_secs):
    def draw(ax, x, g):
        ax.plot(x, g[sz_col], color=DURING_COLOR, marker="o", linewidth=1.5,
                label="During Seizure")
        ax.plot(x, g[pre_col], color=PRE_COLOR, marker="o", linewidth=1.5,
                label=f"Pre-Seizure ({pre_secs:g}s)")
        ax.legend(loc="upper left")
        return None
    return draw


def _draw_ratio(sz_col, pre_col):
    def draw(ax, x, g):
        ratio = g[sz_col] / g[pre_col]
        ax.plot(x, ratio, color=DURING_COLOR, marker="o", linewidth=1.5,
                label="Seizure / Pre-Seizure")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Equal coherence")
        ax.legend(loc="upper left")
        return None
    return draw


def _gdraw_seizure_pre(sz_col, pre_col):
    def draw(ax, df, colors):
        for animal, group in df.groupby("Animal_ID"):
            ax.scatter(group["Normalized_DT"], group[sz_col], color=colors[animal],
                       s=42, alpha=0.85, label=f"{animal} (seiz)")
            ax.scatter(group["Normalized_DT"], group[pre_col], facecolors="none",
                       edgecolors=colors[animal], s=42, alpha=0.5, label=f"{animal} (pre)")
        return None
    return draw


def _gdraw_ratio(sz_col, pre_col):
    def draw(ax, df, colors):
        for animal, group in df.groupby("Animal_ID"):
            ratio = group[sz_col] / group[pre_col]
            ax.scatter(group["Normalized_DT"], ratio, color=colors[animal],
                       s=42, alpha=0.85, label=f"{animal}")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        return None
    return draw


def plot_coherence_animal_summary(
    group_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    band_high=(14.0, 55.0),
    pre_seizure_secs: float = 30,
):
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
    # explainer: above the dashed ratio = 1 line means the seizure is more coherent
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
    # decision: during-seizure filled, pre-seizure faded and open, same animal colour
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
