"""Per-landmark time series, the shape the filtering layer consumes.

A `PoseSequence` is organised by frame, which is how it is produced and stored.
Every numerical operation in Phase 3 -- smoothing, differentiating, gap
handling -- runs along one landmark's trajectory through time instead, so the
transpose happens once, here, into contiguous numpy arrays.

Missing values are NaN, not zero and not interpolated. A landmark that was never
detected and a landmark detected at the origin are different facts, and only NaN
keeps them different all the way to the filter that has to decide what to do
about the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.pose import (
    Landmark,
    LandmarkPoint,
    LandmarkSpace,
    PoseSequence,
)
from analyzer.coordinates import image_to_frame_widths, require_reachable


@dataclass(frozen=True)
class LandmarkSeries:
    """One landmark's trajectory through a clip, in one coordinate space."""

    landmark: Landmark
    space: LandmarkSpace
    timestamps_s: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    visibility: NDArray[np.float64]
    presence: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.timestamps_s.size)

    @property
    def observed(self) -> NDArray[np.bool_]:
        """Frames in which this landmark was actually detected."""
        return ~np.isnan(self.x)

    @property
    def observed_fraction(self) -> float:
        """Proportion of frames with a value. 0.0 for an empty series."""
        if len(self) == 0:
            return 0.0
        return float(np.count_nonzero(self.observed) / len(self))


def _points_for(sequence: PoseSequence, space: LandmarkSpace) -> list[list[LandmarkPoint]]:
    """The stored points a space is read from.

    FRAME_WIDTHS is derived rather than stored, so it reads the IMAGE points and
    is converted below. The store keeps what the estimator emitted and nothing
    else, which is what lets a convention change here without re-extracting.
    """
    field = "hip_local" if space is LandmarkSpace.HIP_LOCAL else "image"
    return [getattr(frame, field) for frame in sequence.frames]


def landmark_series(
    sequence: PoseSequence,
    landmark: Landmark,
    space: LandmarkSpace = LandmarkSpace.FRAME_WIDTHS,
    slow_motion_factor: float = 1.0,
) -> LandmarkSeries:
    """Extract one landmark's trajectory, NaN where it was not detected.

    Defaults to FRAME_WIDTHS, the frame every measurement is taken in. The
    conversion from the stored IMAGE coordinates happens here, once, below the
    filter -- which is what makes the derivatives come out correctly signed
    without a second correction anywhere above (see `analyzer/coordinates.py`).

    `slow_motion_factor` divides the timestamps, which is the whole of what it
    takes to analyse slow-motion footage: every duration, every velocity and the
    filter's own window are then in real seconds. It is applied here for the
    same reason the coordinate conversion is -- once, below everything, so that
    no layer above has to know about it or can forget it.
    """
    require_reachable(space)
    if not np.isfinite(slow_motion_factor) or slow_motion_factor <= 0.0:
        raise ValueError(
            f"slow_motion_factor must be a positive number, got {slow_motion_factor!r}. "
            "It is how many times slower than real time the clip plays: 1 for an "
            "ordinary recording, 8 for eight-times slow motion."
        )
    count = len(sequence.frames)
    timestamps = np.empty(count, dtype=np.float64)
    values = {
        name: np.full(count, np.nan, dtype=np.float64)
        for name in ("x", "y", "z", "visibility", "presence")
    }

    for row, (frame, points) in enumerate(
        zip(sequence.frames, _points_for(sequence, space), strict=True)
    ):
        timestamps[row] = frame.timestamp_s / slow_motion_factor
        if not frame.detected or landmark >= len(points):
            continue
        point = points[landmark]
        values["x"][row] = point.x
        values["y"][row] = point.y
        values["z"][row] = point.z
        values["visibility"][row] = point.visibility
        values["presence"][row] = point.presence

    if space is LandmarkSpace.FRAME_WIDTHS:
        # NaN passes through the conversion unchanged, so an undetected frame
        # stays undetected rather than becoming a coordinate at the origin.
        stacked = image_to_frame_widths(
            np.stack((values["x"], values["y"], values["z"]), axis=-1), sequence.geometry
        )
        values["x"], values["y"], values["z"] = (stacked[:, axis] for axis in range(3))

    return LandmarkSeries(
        landmark=landmark,
        space=space,
        timestamps_s=timestamps,
        x=values["x"],
        y=values["y"],
        z=values["z"],
        visibility=values["visibility"],
        presence=values["presence"],
    )


def all_series(
    sequence: PoseSequence,
    space: LandmarkSpace = LandmarkSpace.FRAME_WIDTHS,
    slow_motion_factor: float = 1.0,
) -> dict[Landmark, LandmarkSeries]:
    """Every landmark's trajectory, keyed by landmark."""
    return {
        landmark: landmark_series(sequence, landmark, space, slow_motion_factor)
        for landmark in Landmark
    }
