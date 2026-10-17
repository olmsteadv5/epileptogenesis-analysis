
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from obe_io import OBEHeader, read_hdr

_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


@dataclass
class OBEMatch:
    filename: str
    header: OBEHeader
    corrected_start: datetime  # explainer: after the DST correction
    seizures: pd.DataFrame  # explainer: metadata rows that overlap this file


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
    obe_files = sorted(
        f for f in os.listdir(folder_path) if f.upper().endswith(".OBE")
    )
    matched: list[OBEMatch] = []

    print(f"Searching {len(obe_files)} files...")

    for file in obe_files:
        full_path = os.path.join(folder_path, file)
        hdr = read_hdr(full_path)

        obe_start = hdr.start_time - timedelta(hours=dst_offset_hours)

        # explainer: recording end from sample count and sampling rate
        duration_seconds = hdr.samples / hdr.samp_freq
        obe_end = obe_start + timedelta(seconds=duration_seconds)

        overlapping = seizure_df[
            (seizure_df["SeizureStartDT"] < obe_end)
            & (seizure_df["SeizureEndDT"] > obe_start)
        ]

        if overlapping.empty:
            continue

        # decision: skip files that map to more than one recording order
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
