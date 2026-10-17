
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from .shared import (
    CH1_COLOR,
    CH2_COLOR,
    MEAN_COLOR,
    RACINE_CMAP,
    RACINE_SCALE_MAX,
    SEIZURE_COLOR,
    THRESHOLD_COLOR,
    _DEFAULT_DAY_ZERO,
    Panel,
    _animal_legend,
    longitudinal_animal_figure,
    longitudinal_global_figure,
)


def plot_correlation(
    corrs,
    time_axis,
    seizure_start: float,
    seizure_end: float,
    corr_threshold: float,
    title: str | None = None,
    ax=None,
):
    # explainer: times are seconds from the window start
    if ax is None:  # explainer: draws onto ax if given, else makes its own figure
        _, ax = plt.subplots(figsize=(9, 3))

    ax.plot(time_axis, corrs, color="white", linewidth=1.2)
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
    # explainer: seizure split into segment_secs chunks, both channels in each
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
    samples_per_seg = int(math.floor(segment_secs * samp_freq))
    num_segments = max(1, math.ceil(len(raw_seiz1) / samples_per_seg))

    # decision: matplotlib can't nest figures, so one GridSpec with the trace on top and a zoom row per segment
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


def _draw_trend_racine(col, cmap, racine_max):
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
    def draw(ax, x, g):
        r = g["Racine_Scale"]
        ax.plot(x, r, color="gray", linewidth=1.5, alpha=0.5)
        ax.scatter(x, r, color="black", s=49, zorder=3)
        return None
    return draw


def _gdraw_animal_scatter(col):
    def draw(ax, df, colors):
        ax.plot(df["Normalized_DT"], df[col], color="gray", linewidth=1.0, alpha=0.2)
        for animal, group in df.groupby("Animal_ID"):
            ax.scatter(group["Normalized_DT"], group[col], color=colors[animal],
                       s=36, alpha=0.7, label=str(animal))
        return None
    return draw


def plot_animal_summary(
    group_df: pd.DataFrame,
    corr_threshold: float,
    cmap: str = RACINE_CMAP,
    racine_max: int = RACINE_SCALE_MAX,
):
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
