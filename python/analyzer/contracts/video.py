"""Typed contracts for video ingestion.

Like `health.py`, these Pydantic models are the single source of truth for the
Rust <-> Python boundary and the TypeScript is generated from them.

Two facts about consumer video drive the shape of everything here, and both are
silent failures rather than loud ones if they are got wrong:

**Rotation.** Phone recordings store upright frames sideways and attach a
display matrix saying how to present them. Decoders disagree about whether they
apply it. Getting it wrong does not produce an error -- it produces a sideways
skeleton and biases every angle the biomechanics layer computes.

**Variable frame rate.** Phone capture frequently varies the interval between
frames, so `frame_index / fps` is not the time the frame was taken. Getting it
wrong does not produce an error either -- it produces a tempo ratio and a
club-head speed that are simply wrong, with no indication that they are.

So rotation is recorded in one explicitly-named direction, and time comes from
presentation timestamps that were measured, with the measurement reported
alongside the conclusion drawn from it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from analyzer.contracts.cache import ContentKey

# Bump on any breaking change to the models in this module. Cached metadata
# carrying a different version is recomputed rather than reinterpreted.
VIDEO_SCHEMA_VERSION = 1

# Counter-clockwise, because that is the direction ffmpeg's display matrix is
# expressed in and translating it would only add a place to get the sign wrong.
# The value is what must be applied to the stored frame to display it upright.
RotationDegrees = Literal[0, 90, 180, 270]


class TimestampSource(StrEnum):
    """Where a clip's per-frame times came from.

    PACKET_PTS      - presentation timestamps read from the container index.
                      Authoritative: these are the times the frames are to be
                      shown at, whether or not they are evenly spaced.
    CONTAINER_RATE  - synthesised from the container's declared average frame
                      rate because no usable timestamps were found. A fallback,
                      and one that makes variable frame rate undetectable, so
                      anything derived from it is flagged.
    """

    PACKET_PTS = "packet_pts"
    CONTAINER_RATE = "container_rate"


class IntervalStats(BaseModel):
    """Measured spacing between consecutive presentation timestamps.

    Reported so the `is_vfr` verdict can be checked rather than trusted: a clip
    with one irregular interval out of 7000 and a clip that switches rate
    half-way through are both "variable", and these numbers are what tells them
    apart.
    """

    median_s: float
    min_s: float
    max_s: float
    quantum_s: float = Field(
        description=(
            "One tick of the container's time base. Intervals cannot be resolved "
            "more finely than this, so it is the tolerance below which a "
            "difference is rounding rather than a real rate change."
        )
    )
    irregular_count: int = Field(
        description="Intervals differing from the median by more than one time-base tick."
    )
    irregular_fraction: float


class VideoTiming(BaseModel):
    """When each frame is presented, and whether that is evenly spaced."""

    source: TimestampSource
    frame_count: int = Field(description="Frames found in the container index.")
    first_timestamp_s: float
    last_timestamp_s: float
    timestamp_span_s: float = Field(
        description="last - first. Measured exactly; assumes nothing about the final frame."
    )
    duration_s: float = Field(
        description=(
            "Estimated playback duration: the timestamp span plus the last observed "
            "interval, since how long the final frame is displayed is not recorded."
        )
    )
    container_duration_s: float | None = Field(
        default=None,
        description="Duration as declared by the container, for comparison. Not used for analysis.",
    )
    nominal_fps: float | None = Field(
        default=None,
        description="The container's declared average frame rate. Display only.",
    )
    measured_fps: float | None = Field(
        default=None,
        description=(
            "(frame_count - 1) / timestamp_span_s. Meaningful as an average even on "
            "variable-rate clips; None when there are fewer than two frames."
        ),
    )
    intervals: IntervalStats | None = Field(
        default=None,
        description="None when timestamps were synthesised, or when there are fewer than two frames.",
    )
    is_vfr: bool | None = Field(
        default=None,
        description=(
            "True when any interval differs from the median by more than one time-base "
            "tick. None means it could not be determined, which is not the same as False."
        ),
    )


class VideoStreamInfo(BaseModel):
    """The video stream's coding and geometry, as the container declares them."""

    codec_name: str
    codec_long_name: str | None = None
    profile: str | None = None
    pix_fmt: str | None = None
    coded_width: int = Field(description="Width as stored, before any display rotation.")
    coded_height: int = Field(description="Height as stored, before any display rotation.")
    display_width: int = Field(description="Width after rotation. What a frame source yields.")
    display_height: int = Field(description="Height after rotation. What a frame source yields.")
    rotation_ccw_degrees: RotationDegrees = Field(
        description=(
            "Counter-clockwise rotation to apply to a stored frame to display it "
            "upright, taken from the container's display matrix."
        )
    )
    rotation_source: str | None = Field(
        default=None,
        description="Which container field the rotation came from: 'display_matrix' or 'rotate_tag'.",
    )
    sample_aspect_ratio: str | None = None
    bit_rate: int | None = None
    time_base: str = Field(description="Stream time base as a rational string, e.g. '1/15360'.")


class VideoMetadata(BaseModel):
    """Everything known about a video file without decoding it.

    Produced by `probe_video`. Every field is read from the container by ffprobe
    or computed from timestamps that were; nothing here is assumed.
    """

    schema_version: int = VIDEO_SCHEMA_VERSION
    path: str
    file_size_bytes: int
    content_key: ContentKey
    probed_at: datetime
    container_format: str = Field(
        description="ffprobe's format_name, e.g. 'mov,mp4,m4a,3gp,3g2,mj2'."
    )
    stream: VideoStreamInfo
    timing: VideoTiming
    warnings: list[str] = Field(
        default_factory=list,
        description="Caveats that affect how the clip may be analysed, e.g. variable frame rate.",
    )
