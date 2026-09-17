"""The subject, ready to be measured.

One object built once per clip and read by every metric module, so that the
choice of scale and the set of landmarks a metric may use are decided in one
place rather than per module.

It no longer converts anything. Until Phase 6 this was also where the aspect
correction and the y flip happened, which made it the only safe way into the
filtered sequence; both now happen below the filter, so the positions arriving
here are already isotropic and upward-positive. What is left is a slice to x and
y, and the scale below.

## Scale

Distances here are reported in **torso lengths**, not frame widths. A frame
width is a property of where the camera was put -- move it twice as far away and
every distance in frame halves -- while the subject's own torso does not change
between recordings of the same person. Dividing by it is what makes a hip sway
measured on one clip comparable with the same sway measured on another, and it
is the same argument Phase 4 makes for judging hand travel against a torso
rather than against the frame.

The torso length itself is a **median over the clip**, not a per-frame value.
Taken per frame it shrinks whenever the subject bends or turns away from the
camera, and dividing a distance by a shrinking scale would report a growing
number for a constant motion. The median is dominated by the frames where the
subject stands square, which is the length being appealed to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.biomechanics.geometry import distance, midpoint
from analyzer.contracts.phases import HandSource
from analyzer.contracts.pose import FrameGeometry, Landmark, LandmarkSpace
from analyzer.filtering.landmarks import FilteredSequence
from analyzer.phases.signals import choose_hand

# The landmarks every metric in this phase is built from. Restricting the
# conversion to these keeps the object small and makes the dependency explicit:
# adding a metric that needs an ankle is a change to this tuple, which is a
# change a reviewer sees.
TRACKED: tuple[Landmark, ...] = (
    Landmark.NOSE,
    Landmark.LEFT_SHOULDER,
    Landmark.RIGHT_SHOULDER,
    Landmark.LEFT_ELBOW,
    Landmark.RIGHT_ELBOW,
    Landmark.LEFT_WRIST,
    Landmark.RIGHT_WRIST,
    Landmark.LEFT_HIP,
    Landmark.RIGHT_HIP,
    Landmark.LEFT_KNEE,
    Landmark.RIGHT_KNEE,
    Landmark.LEFT_ANKLE,
    Landmark.RIGHT_ANKLE,
)


class BodyError(ValueError):
    """The subject could not be assembled from a filtered sequence."""


@dataclass(frozen=True)
class HandTrack:
    """The tracked hand point, in frame widths.

    Which landmark it came from was decided by Phase 4, once for the clip, and
    is carried here rather than re-decided: the two layers reasoning about
    "the hands" from different points would make a hand path and a hand speed
    that do not describe the same trajectory.
    """

    source: HandSource
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    valid: NDArray[np.bool_]
    visibility: NDArray[np.float64]


@dataclass(frozen=True)
class Body:
    """One clip's subject: every tracked landmark in frame widths, plus scale."""

    t: NDArray[np.float64]
    geometry: FrameGeometry
    points: dict[Landmark, NDArray[np.float64]]
    valid: dict[Landmark, NDArray[np.bool_]]
    visibility: dict[Landmark, NDArray[np.float64]]
    hand: HandTrack
    torso_length: float

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def interval_s(self) -> float:
        """Median time between frames, for judging how finely a duration is divided."""
        if self.t.size < 2:
            return 0.0
        return float(np.median(np.diff(self.t)))

    def tracked(self, *landmarks: Landmark) -> NDArray[np.bool_]:
        """Frames where every named landmark carries a position."""
        mask = np.ones(len(self), dtype=bool)
        for landmark in landmarks:
            mask &= self.valid[landmark]
        return mask

    def masked(self, series: NDArray[np.float64], *landmarks: Landmark) -> NDArray[np.float64]:
        """A per-frame series, NaN wherever any landmark it needs is missing.

        The guard that keeps a metric from being computed out of a position the
        filter never produced. Arithmetic on NaN already yields NaN, but a
        landmark can be invalid while its coordinates are finite -- a partially
        filled gap, for instance -- so the mask is applied explicitly rather
        than assumed.
        """
        return np.where(self.tracked(*landmarks), series, np.nan)

    # --- the segments metrics are built from -----------------------------

    @property
    def shoulder_mid(self) -> NDArray[np.float64]:
        return midpoint(self.points[Landmark.LEFT_SHOULDER], self.points[Landmark.RIGHT_SHOULDER])

    @property
    def hip_mid(self) -> NDArray[np.float64]:
        return midpoint(self.points[Landmark.LEFT_HIP], self.points[Landmark.RIGHT_HIP])

    @property
    def shoulder_span(self) -> NDArray[np.float64]:
        """Projected distance between the shoulders, per frame, in frame widths."""
        return self.masked(
            distance(self.points[Landmark.LEFT_SHOULDER], self.points[Landmark.RIGHT_SHOULDER]),
            Landmark.LEFT_SHOULDER,
            Landmark.RIGHT_SHOULDER,
        )

    @property
    def hip_span(self) -> NDArray[np.float64]:
        """Projected distance between the hips, per frame, in frame widths."""
        return self.masked(
            distance(self.points[Landmark.LEFT_HIP], self.points[Landmark.RIGHT_HIP]),
            Landmark.LEFT_HIP,
            Landmark.RIGHT_HIP,
        )

    def in_torso_lengths(self, series: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert a series of frame-width distances to torso lengths."""
        return series / self.torso_length


def _torso_length(points: dict[Landmark, NDArray[np.float64]], usable: NDArray[np.bool_]) -> float:
    """Median shoulder-midpoint to hip-midpoint distance over the frames that have one."""
    if not np.any(usable):
        return float("nan")
    shoulders = midpoint(points[Landmark.LEFT_SHOULDER], points[Landmark.RIGHT_SHOULDER])
    hips = midpoint(points[Landmark.LEFT_HIP], points[Landmark.RIGHT_HIP])
    return float(np.median(distance(shoulders, hips)[usable]))


def body_from(filtered: FilteredSequence) -> Body:
    """Build the measurable subject from a filtered pose sequence.

    Refuses anything but FRAME_WIDTHS. IMAGE is anisotropic and upside down, so
    every angle here would be wrong by the shape of the frame and every height
    inverted. HIP_LOCAL is hip-centred, so a hip sway measured in it is zero by
    construction and a head lift is measured from an origin that is itself
    moving.
    """
    if filtered.space is not LandmarkSpace.FRAME_WIDTHS:
        raise BodyError(
            f"Biomechanics reads FRAME_WIDTHS trajectories; this sequence is in "
            f"{filtered.space.value}. IMAGE has a different unit on each axis and y "
            "pointing down; HIP_LOCAL is centred on the hips, so a displacement in it "
            "is relative to a moving origin."
        )

    missing = [landmark for landmark in TRACKED if landmark not in filtered.landmarks]
    if missing:
        raise BodyError(
            "Biomechanics needs the whole tracked skeleton; this sequence was filtered "
            f"without {', '.join(entry.name.lower() for entry in missing)}."
        )

    # Sliced to x and y rather than converted. The frame is already isotropic
    # and upward-positive; the z channel is dropped because in it MediaPipe's
    # single-camera depth guess would enter a distance as though it were a
    # measurement.
    points = {landmark: filtered[landmark].position[:, :2] for landmark in TRACKED}
    valid = {landmark: filtered[landmark].valid for landmark in TRACKED}
    visibility = {landmark: filtered[landmark].visibility for landmark in TRACKED}

    torso_usable = (
        valid[Landmark.LEFT_SHOULDER]
        & valid[Landmark.RIGHT_SHOULDER]
        & valid[Landmark.LEFT_HIP]
        & valid[Landmark.RIGHT_HIP]
    )

    raw_hand = choose_hand(filtered)
    return Body(
        t=filtered.t,
        geometry=filtered.geometry,
        points=points,
        valid=valid,
        visibility=visibility,
        hand=HandTrack(
            source=raw_hand.source,
            position=raw_hand.position[:, :2],
            # No special case any more. The conversion now happens on positions
            # before the fit, and the filter is linear, so the velocity it
            # produces is already in the same frame -- where converting after the
            # fit meant scaling and flipping the derivative separately and
            # keeping that in step with the position conversion by hand.
            velocity=raw_hand.velocity[:, :2],
            valid=raw_hand.valid,
            visibility=raw_hand.visibility,
        ),
        torso_length=_torso_length(points, torso_usable),
    )
