"""Readers for the binary ``.OBE`` EEG file format.

Ported from ``read_hdr.jl`` / ``read_data.jl``. The header is a little-endian
mix of Int32 and Float64 fields at fixed byte offsets; channel data is stored as
contiguous Float32 blocks, one channel after another.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

# Excel/Delphi serial-date epoch used by the OBE start-time field.
_BASE_DATE = datetime(1899, 12, 30)
_MS_PER_DAY = 86_400_000


@dataclass
class OBEHeader:
    """Parsed ``.OBE`` header. Field order matches ``read_hdr.jl``'s return."""

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
    """Read and parse the header of an ``.OBE`` file.

    Mirrors ``readhdr`` from ``read_hdr.jl``: sequential Int32/Float64 fields,
    a serial start-time at byte offset 100 (``25 * 4``), the
    ``15101051 -> 1`` identifier remap, and identifier-dependent data offsets.
    """
    with open(filename, "rb") as io:
        head = io.read(108)  # through the Float64 start-time at offset 100

    identifier = struct.unpack_from("<i", head, 0)[0]
    samples = struct.unpack_from("<i", head, 4)[0]
    channels = struct.unpack_from("<i", head, 8)[0]
    seg_length = struct.unpack_from("<i", head, 12)[0]
    segments = struct.unpack_from("<i", head, 16)[0]
    # bytes 20 (segs_per_blk) and 24 (blocks) are read but unused downstream
    samp_freq = struct.unpack_from("<d", head, 28)[0]
    eeg_samp_freq = struct.unpack_from("<d", head, 36)[0]
    ref_channel = struct.unpack_from("<i", head, 44)[0]
    start_time_val = struct.unpack_from("<d", head, 100)[0]

    start_time = _BASE_DATE + timedelta(milliseconds=round(start_time_val * _MS_PER_DAY))

    if identifier == 15101051:
        identifier = 1

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
    """Read Float32 channel data from an ``.OBE`` file.

    Ported from ``readdata`` in ``read_data.jl``; ``channel``/``segment`` are
    1-based to match the Julia indexing. Return shape depends on arguments:

    * ``channel is None``        -> 2-D ``(samples, channels)`` array of all channels.
    * ``channel`` given only     -> 1-D array with that channel's full data.
    * ``segment``/``start_sample`` given -> 1-D array of the requested slice.
    """
    hdr = read_hdr(filename)
    samples = hdr.samples
    channels = hdr.channels
    seg_length = hdr.seg_length
    segments = hdr.segments
    data_loc = hdr.data_location

    two_dim = segments > 0 and seg_length > 1

    if channel is None:
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

    if channel is None:
        # Julia fills column-major: column j == channel j -> use Fortran order.
        return np.reshape(data, (samples, channels), order="F")
    # Single channel (and the unused segment/slice branches) are consumed as 1-D.
    return data
