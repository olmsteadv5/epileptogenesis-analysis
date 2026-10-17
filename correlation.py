# explainer: datetime_to_sample returns 0-based indices meant for numpy half-open slices

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from obe_io import read_data

# explainer: keep in sync with config.DAY_ZERO, the plot day axis assumes the same epoch
_DEFAULT_DAY_ZERO = (2000, 1, 1)


def channel_indices(
    rec_order: int,
    ch1_offset: int,
    ch2_offset: int,
    stride: int = 10,
) -> tuple[int, int]:
    # explainer: 1-based channel numbers
    ch1 = int((rec_order - 1) * stride + ch1_offset)
    ch2 = int((rec_order - 1) * stride + ch2_offset)
    return ch1, ch2


# explainer: can go negative (before the recording) or past the end, callers check bounds
def datetime_to_sample(dt: datetime, ref_start: datetime, samp_freq: float) -> int:
    seconds = (dt - ref_start).total_seconds()
    return int(math.floor(seconds * samp_freq))


def compute_rolling_correlation(
    data1: np.ndarray,
    data2: np.ndarray,
    samp_freq: float,
    window_secs: float,
    step_secs: float,
) -> tuple[np.ndarray, np.ndarray]:
    win_size = int(math.floor(window_secs * samp_freq))
    step_size = int(math.floor(step_secs * samp_freq))

    # explainer: empty if the data is shorter than one window
    starts = range(0, len(data1) - win_size, step_size)
    corrs = np.array(
        [
            np.corrcoef(data1[k : k + win_size], data2[k : k + win_size])[0, 1]
            for k in starts
        ]
    )
    # explainer: window start offsets, seconds from the start of data1
    time_axis = np.arange(len(corrs)) * step_secs
    return corrs, time_axis


def offset_to_day_zero(
    df: pd.DataFrame,
    time_col: str = "File_Start_DT",
    animal_col: str = "Animal_ID",
    day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO,
) -> pd.DataFrame:
    # decision: each animal's first day maps onto day_zero, time of day and day spacing kept, so animals overlay
    out = df.copy()
    day_zero_ts = pd.Timestamp(*day_zero)
    normalized = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    for _, group in out.groupby(animal_col):
        first_date = group[time_col].min().normalize()
        day_offset = group[time_col].dt.normalize() - first_date
        time_of_day = group[time_col] - group[time_col].dt.normalize()
        normalized.loc[group.index] = day_zero_ts + day_offset + time_of_day

    out["Normalized_DT"] = normalized
    return out.sort_values("Normalized_DT").reset_index(drop=True)


def filter_by_baseline(
    matched_results,
    folder_path: str,
    *,
    baseline_duration_secs: float,
    baseline_corr_threshold: float,
    rolling_window_secs: float,
    rolling_step_secs: float,
    ch1_offset: int,
    ch2_offset: int,
    channel_stride: int = 10,
):
    # decision: drop seizures whose pre-seizure window is out of bounds, too short, or already too correlated
    import os

    print("Checking baseline correlation for each seizure...")

    seizures_to_analyze: list[tuple[int, int, float]] = []
    excluded_seizures: list[dict] = []

    total_seizures = sum(len(m.seizures) for m in matched_results)
    excluded_count = 0
    included_count = 0

    for match_idx, match in enumerate(matched_results):
        full_path = os.path.join(folder_path, match.filename)
        hdr = match.header
        samp_freq = hdr.samp_freq
        rec_order = match.seizures["Recording_order"].iloc[0]
        ch1_idx, ch2_idx = channel_indices(
            rec_order, ch1_offset, ch2_offset, channel_stride
        )

        for row_idx in range(len(match.seizures)):
            row = match.seizures.iloc[row_idx]
            reason = ""
            avg_baseline_corr = float("nan")

            try:
                baseline_start = row["SeizureStartDT"] - timedelta(
                    seconds=baseline_duration_secs
                )
                baseline_end = row["SeizureStartDT"]

                start_samp = datetime_to_sample(
                    baseline_start, match.corrected_start, samp_freq
                )
                end_samp = datetime_to_sample(
                    baseline_end, match.corrected_start, samp_freq
                )

                if start_samp < 0 or end_samp > hdr.samples:
                    reason = "Out of Bounds"
                else:
                    baseline_data1 = read_data(full_path, ch1_idx)[start_samp:end_samp]
                    baseline_data2 = read_data(full_path, ch2_idx)[start_samp:end_samp]

                    baseline_corrs, _ = compute_rolling_correlation(
                        baseline_data1,
                        baseline_data2,
                        samp_freq,
                        rolling_window_secs,
                        rolling_step_secs,
                    )

                    if len(baseline_corrs) == 0:
                        reason = "Baseline Too Short"
                    else:
                        avg_baseline_corr = float(np.mean(np.abs(baseline_corrs)))

                        # decision: baseline already too correlated to trust as a quiet reference
                        if avg_baseline_corr > baseline_corr_threshold:
                            reason = f"High Baseline Corr ({round(avg_baseline_corr, 3)})"

            except Exception as e:
                reason = f"Processing Error: {e}"

            if reason == "":
                seizures_to_analyze.append((match_idx, row_idx, avg_baseline_corr))
                included_count += 1
            else:
                excluded_count += 1
                excluded_seizures.append(
                    {
                        "Animal": row["Animal_no"],
                        "Seizure": row["Seizure"],
                        "Order": rec_order,
                        "Reason": reason,
                    }
                )
                print(
                    f"  Excluded Animal {row['Animal_no']} Seiz {row['Seizure']}: {reason}"
                )

    print(f"\nExcluded {excluded_count} of {total_seizures} seizures; {included_count} to analyze.")

    if excluded_seizures:
        print("Exclusions by animal:")
        animal_ids = sorted({exc["Animal"] for exc in excluded_seizures})
        for animal_id in animal_ids:
            reasons = {
                exc["Reason"]
                for exc in excluded_seizures
                if exc["Animal"] == animal_id
            }
            count_val = sum(1 for exc in excluded_seizures if exc["Animal"] == animal_id)
            print(f"  Animal {animal_id}: {count_val} removed ({sorted(reasons)})")

    return seizures_to_analyze, excluded_seizures
