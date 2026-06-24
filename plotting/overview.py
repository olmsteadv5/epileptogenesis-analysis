# 60-minute decimated EEG overview with seizure markers.

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .shared import CH1_COLOR, CH2_COLOR


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
    # 60-minute EEG overview: num_subplots stacked decimated traces, each showing
    # num_mins_per_subplot minutes of both channels.
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

        # Overlay seizure onset/offset markers that fall in this subplot's minute range.
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
