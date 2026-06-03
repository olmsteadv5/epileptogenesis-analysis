"""Rolling correlation and shared signal/index helpers.

These helpers consolidate index math that was duplicated throughout the original
notebook. In particular ``datetime_to_sample`` returns a **0-based** index suited
to NumPy half-open slicing, which reproduces the Julia 1-based inclusive slices
without the per-call ``+1`` / no-``+1`` bookkeeping.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from obe_io import read_data

# Dummy "Day 0" all animals are offset onto for the global summaries.
_DAY_ZERO = pd.Timestamp(2000, 1, 1)


def channel_indices(
    rec_order: int,
    ch1_offset: int,
    ch2_offset: int,
    stride: int = 10,
) -> tuple[int, int]:
    """Return the (1-based) ``(ch1, ch2)`` channel numbers for a recording order.

    Replaces the ``int((rec_order - 1) * 10 + offset)`` expression that appeared
    in several places in the original notebook.
    """
    ch1 = int((rec_order - 1) * stride + ch1_offset)
    ch2 = int((rec_order - 1) * stride + ch2_offset)
    return ch1, ch2


def datetime_to_sample(dt: datetime, ref_start: datetime, samp_freq: float) -> int:
    """Convert a datetime to a 0-based sample index relative to ``ref_start``.

    Equivalent to the Julia ``floor(Int, seconds * samp_freq)`` used for slice
    bounds. Use the result directly as a NumPy slice bound (start or end); a
    negative result means "before the recording", a result greater than the
    sample count means "past the end".
    """
    seconds = (dt - ref_start).total_seconds()
    return int(math.floor(seconds * samp_freq))


def compute_rolling_correlation(
    data1: np.ndarray,
    data2: np.ndarray,
    samp_freq: float,
    window_secs: float,
    step_secs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Pearson correlation between two signals.

    Slides a ``window_secs`` window in ``step_secs`` hops and computes the
    Pearson correlation in each window. Returns ``(correlations, time_axis)``
    where ``time_axis`` is the window-start offset in seconds from the start of
    ``data1``. An empty result means the data was shorter than one window.
    """
    win_size = int(math.floor(window_secs * samp_freq))
    step_size = int(math.floor(step_secs * samp_freq))

    starts = range(0, len(data1) - win_size, step_size)
    corrs = np.array(
        [
            np.corrcoef(data1[k : k + win_size], data2[k : k + win_size])[0, 1]
            for k in starts
        ]
    )
    time_axis = np.arange(len(corrs)) * step_secs
    return corrs, time_axis


def offset_to_day_zero(
    df: pd.DataFrame,
    time_col: str = "File_Start_DT",
    animal_col: str = "Animal_ID",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``Normalized_DT`` column offset to Day 0.

    Each animal's earliest calendar day is mapped onto 2000-01-01 while the time
    of day and the day-to-day spacing are preserved, so animals recorded on
    different real dates can be overlaid on a common experimental timeline.
    """
    out = df.copy()
    normalized = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    for _, group in out.groupby(animal_col):
        first_date = group[time_col].min().normalize()
        day_offset = group[time_col].dt.normalize() - first_date
        time_of_day = group[time_col] - group[time_col].dt.normalize()
        normalized.loc[group.index] = _DAY_ZERO + day_offset + time_of_day

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
    """Pre-filter seizures by their pre-seizure baseline correlation.

    For every seizure, the ``baseline_duration_secs`` window immediately before
    onset is scored with the rolling correlation. A seizure is excluded if that
    window is out of bounds, too short for a single correlation window, or has a
    mean ``|correlation|`` above ``baseline_corr_threshold``.

    Returns ``(seizures_to_analyze, excluded_seizures)`` where
    ``seizures_to_analyze`` is a list of ``(match_idx, row_idx, avg_baseline_corr)``
    and ``excluded_seizures`` is a list of dicts describing each exclusion.
    """
    import os

    print("\n=== PRE-FILTERING: Baseline Correlation Check ===")

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

                # CHECK 1: Out of bounds
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

                    # CHECK 2: Window too short
                    if len(baseline_corrs) == 0:
                        reason = "Baseline Too Short"
                    else:
                        avg_baseline_corr = float(np.mean(np.abs(baseline_corrs)))

                        # CHECK 3: High Correlation
                        if avg_baseline_corr > baseline_corr_threshold:
                            reason = f"High Baseline Corr ({round(avg_baseline_corr, 3)})"

            except Exception as e:  # noqa: BLE001 - mirror Julia's broad catch
                reason = f"Processing Error: {e}"

            # FINAL DECISION
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
                    f"  x EXCLUDED: Animal {row['Animal_no']}, "
                    f"Seiz {row['Seizure']} - Reason: {reason}"
                )

    print("\n--- PRE-FILTERING SUMMARY ---")
    print(f"Total seizures found: {total_seizures}")
    print(f"Total Excluded: {excluded_count}")
    print(f"Total to be analyzed: {included_count}")

    if excluded_seizures:
        print("\nExclusions by Animal:")
        animal_ids = sorted({exc["Animal"] for exc in excluded_seizures})
        for animal_id in animal_ids:
            reasons = {
                exc["Reason"]
                for exc in excluded_seizures
                if exc["Animal"] == animal_id
            }
            count_val = sum(1 for exc in excluded_seizures if exc["Animal"] == animal_id)
            print(
                f"  Animal {animal_id}: {count_val} seizures removed "
                f"(Reasons: {sorted(reasons)})"
            )

    print(f"\nCorrelation analysis will run on {included_count} seizures.")
    return seizures_to_analyze, excluded_seizures
