# explainer: .OBE format: little-endian header at fixed byte offsets, then float32 channel blocks one after another

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

# explainer: start time is an Excel/Delphi serial date
_BASE_DATE = datetime(1899, 12, 30)
_MS_PER_DAY = 86_400_000


@dataclass
class OBEHeader:
    samples: int
    channels: int
    seg_length: int
    segments: int
    ref_channel: int
    eeg_samp_freq: float
    samp_freq: float
    start_time: datetime
    identifier: int
    channel_labels_loc: int
    data_location: int


def read_hdr(filename: str) -> OBEHeader:
    # explainer: read up to the float64 start time at byte 100
    with open(filename, "rb") as io:
        head = io.read(108)

    identifier = struct.unpack_from("<i", head, 0)[0]
    samples = struct.unpack_from("<i", head, 4)[0]
    channels = struct.unpack_from("<i", head, 8)[0]
    seg_length = struct.unpack_from("<i", head, 12)[0]
    segments = struct.unpack_from("<i", head, 16)[0]
    # explainer: bytes 20 and 24 (segs_per_blk, blocks) exist but aren't used
    samp_freq = struct.unpack_from("<d", head, 28)[0]
    eeg_samp_freq = struct.unpack_from("<d", head, 36)[0]
    ref_channel = struct.unpack_from("<i", head, 44)[0]
    start_time_val = struct.unpack_from("<d", head, 100)[0]

    start_time = _BASE_DATE + timedelta(milliseconds=round(start_time_val * _MS_PER_DAY))

    # decision: legacy identifier normalized to 1
    if identifier == 15101051:
        identifier = 1

    # explainer: data offsets depend on the file identifier
    channel_labels_loc = 0
    data_location = 0
    if identifier == 1:
        channel_labels_loc = 5000
        data_location = 10000
    elif identifier == 2:
        channel_labels_loc = 15700
        data_location = 131072

    return OBEHeader(
        samples=samples,
        channels=channels,
        seg_length=seg_length,
        segments=segments,
        ref_channel=ref_channel,
        eeg_samp_freq=eeg_samp_freq,
        samp_freq=samp_freq,
        start_time=start_time,
        identifier=identifier,
        channel_labels_loc=channel_labels_loc,
        data_location=data_location,
    )


def read_data(
    filename: str,
    channel: int | None = None,
    segment: int | None = None,
    start_sample: int | None = None,
    num_to_read: int | None = None,
) -> np.ndarray:
    # explainer: channel and segment are 1-based
    hdr = read_hdr(filename)
    samples = hdr.samples
    channels = hdr.channels
    seg_length = hdr.seg_length
    segments = hdr.segments
    data_loc = hdr.data_location

    two_dim = segments > 0 and seg_length > 1

    if channel is None:
        # explainer: no channel = every channel as one flat block
        curr_channel = 1
        curr_start = 1
        curr_num = samples * channels
    elif segment is None:
        curr_channel = channel
        curr_start = 1
        curr_num = seg_length * segments
    elif start_sample is None:
        curr_channel = channel
        curr_start = 1 if not two_dim else segment * seg_length + 1
        curr_num = seg_length
    else:
        curr_channel = channel
        curr_start = start_sample
        curr_num = num_to_read

    start_address = data_loc + (((curr_channel - 1) * samples + curr_start - 1) * 4)

    with open(filename, "rb") as io:
        io.seek(start_address)
        data = np.fromfile(io, dtype="<f4", count=curr_num)

    # explainer: flat all-channel read, reshape to (samples, channels) column-major
    if channel is None:
        return np.reshape(data, (samples, channels), order="F")
    return data
