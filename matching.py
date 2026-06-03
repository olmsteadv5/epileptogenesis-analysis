"""CSV metadata loading and OBE-file-to-seizure matching."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from obe_io import OBEHeader, read_hdr

# CSV date/time formats, e.g. "6/18/2013" and "1:13:24 PM".
_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


@dataclass
class OBEMatch:
    """One ``.OBE`` file matched to the seizures that overlap its recording."""

    filename: str
    header: OBEHeader
    corrected_start: datetime          # header start time after the DST correction
    seizures: pd.DataFrame             # rows of the metadata DataFrame that overlap


def load_seizure_metadata(csv_path: str) -> pd.DataFrame:
    """Load the seizure spreadsheet and add parsed start/end datetimes.

    Adds ``SeizureStartDT`` and ``SeizureEndDT`` columns built from the
    ``Date_Start``/``Time_Start`` and ``Date_End``/``Time_End`` text columns.
    """
    df = pd.read_csv(csv_path)

    df["SeizureStartDT"] = pd.to_datetime(
        df["Date_Start"].astype(str) + " " + df["Time_Start"].astype(str),
        format=_DATETIME_FORMAT,
    )
    df["SeizureEndDT"] = pd.to_datetime(
        df["Date_End"].astype(str) + " " + df["Time_End"].astype(str),
        format=_DATETIME_FORMAT,
    )
    return df


def match_obe_files(
    folder_path: str,
    seizure_df: pd.DataFrame,
    dst_offset_hours: int = 1,
) -> list[OBEMatch]:
    """Match every ``.OBE`` file in ``folder_path`` to overlapping seizures.

    For each file: read the header, subtract the daylight-savings offset from
    the start time, compute the recording end from ``Samples / SampFreq``, and
    collect seizures whose interval overlaps the recording. Files whose
    overlapping seizures span more than one ``Recording_order`` are skipped.
    """
    obe_files = sorted(
        f for f in os.listdir(folder_path) if f.upper().endswith(".OBE")
    )
    matched: list[OBEMatch] = []

    print(f"Starting search in {len(obe_files)} files...")

    for file in obe_files:
        full_path = os.path.join(folder_path, file)
        hdr = read_hdr(full_path)

        # Offset correction for daylight savings.
        obe_start = hdr.start_time - timedelta(hours=dst_offset_hours)

        # Recording end time from sample count and sampling frequency.
        duration_seconds = hdr.samples / hdr.samp_freq
        obe_end = obe_start + timedelta(seconds=duration_seconds)

        overlapping = seizure_df[
            (seizure_df["SeizureStartDT"] < obe_end)
            & (seizure_df["SeizureEndDT"] > obe_start)
        ]

        if overlapping.empty:
            continue

        unique_orders = overlapping["Recording_order"].unique()
        if len(unique_orders) > 1:
            print(
                f"Conflict in {file}: Found multiple Recording_orders "
                f"{list(unique_orders)}. Skipping."
            )
            continue

        matched.append(
            OBEMatch(
                filename=file,
                header=hdr,
                corrected_start=obe_start,
                seizures=overlapping.reset_index(drop=True),
            )
        )

    print("\n--- Matching Complete ---")
    print(f"Successfully matched {len(matched)} OBE files.")
    return matched
