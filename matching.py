# CSV metadata loading and OBE-file-to-seizure matching.

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from obe_io import OBEHeader, read_hdr

# CSV date/time formats, e.g. "6/18/2013" and "1:13:24 PM".
_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


# One .OBE file matched to the seizures that overlap its recording.
@dataclass
class OBEMatch:
    filename: str
    header: OBEHeader
    corrected_start: datetime          # header start time after the DST correction
    seizures: pd.DataFrame             # rows of the metadata DataFrame that overlap


# Load the seizure spreadsheet and add parsed start/end datetimes.
# Adds SeizureStartDT and SeizureEndDT from the Date_Start/Time_Start and Date_End/Time_End columns.
def load_seizure_metadata(csv_path: str) -> pd.DataFrame:
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
    # Match every .OBE file in folder_path to the seizures that overlap it.
    obe_files = sorted(
        f for f in os.listdir(folder_path) if f.upper().endswith(".OBE")
    )
    matched: list[OBEMatch] = []

    print(f"Searching {len(obe_files)} files...")

    for file in obe_files:
        full_path = os.path.join(folder_path, file)
        hdr = read_hdr(full_path)

        # Correct the header start time for daylight savings.
        obe_start = hdr.start_time - timedelta(hours=dst_offset_hours)

        # Recording end comes from the sample count and sampling frequency.
        duration_seconds = hdr.samples / hdr.samp_freq
        obe_end = obe_start + timedelta(seconds=duration_seconds)

        # Keep seizures whose interval overlaps this recording.
        overlapping = seizure_df[
            (seizure_df["SeizureStartDT"] < obe_end)
            & (seizure_df["SeizureEndDT"] > obe_start)
        ]

        if overlapping.empty:
            continue

        # A file should map to a single recording order; skip it if not.
        unique_orders = overlapping["Recording_order"].unique()
        if len(unique_orders) > 1:
            print(f"Skipping {file}: multiple recording orders {list(unique_orders)}")
            continue

        matched.append(
            OBEMatch(
                filename=file,
                header=hdr,
                corrected_start=obe_start,
                seizures=overlapping.reset_index(drop=True),
            )
        )

    print(f"Matched {len(matched)} of {len(obe_files)} files.")
    return matched
