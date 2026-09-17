"""Signals a swing is read from.

The first code in this engine that knows what a golf swing is. Everything below
produces general trajectories; this reduces them to the handful of scalar
signals whose shape identifies the events.

**Hand speed** is the primary signal, and the whole detection rests on one fact
about it: the hands are nearly stationary at address, accelerate through the
backswing, come momentarily to rest at the top, accelerate again far harder
through the downswing, and decay through the follow-through. That gives a
minimum between two maxima, which is a structure worth looking for rather than a
threshold worth tuning.

**Hand height** disambiguates. A speed minimum alone could be a pause anywhere;
a speed minimum at the highest point the hands reach is the top of a backswing.

**Shoulder and hip line angles** are carried for Phase 5 and for corroboration.
They are *projected* angles -- the angle of a line in the image plane, under
whatever viewing geometry the camera happened to have -- and are not body
rotation. `contracts/phases.py` and Phase 5 are where that distinction is
enforced; here they are simply signals.

## Image coordinates point down

The single most dangerous thing in this module. IMAGE space has y increasing
*downward*, so a larger y is a lower hand. Every rule here is about the hands
being high or low, and a sign error would invert the top of the backswing into
the bottom without failing anywhere. So y is converted once, here, into an
explicitly named `height` that increases upward, and the raw y is not used again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.phases import HandSource
from analyzer.contracts.pose import Landmark, LandmarkSpace
from analyzer.filtering.landmarks import FilteredLandmark, FilteredSequence


def _torso_length(filtered: FilteredSequence) -> float:
    """Median distance from the shoulder midpoint to the hip midpoint.

    The scale everything else is judged against. It has to come from the subject
    rather than from the frame, because how much of the frame a person fills
    depends only on where the camera was put, and it has to come from the
    subject's *body* rather than from the filter's noise estimate, because that
    estimate is zero whenever the smoothing window holds exactly as many samples
    as the polynomial has coefficients -- which is the case on every clip below
    about 60 fps at the shipped defaults. A torso is always the same torso.
    """
    landmarks = filtered.landmarks
    left_shoulder = landmarks.get(Landmark.LEFT_SHOULDER)
    right_shoulder = landmarks.get(Landmark.RIGHT_SHOULDER)
    left_hip = landmarks.get(Landmark.LEFT_HIP)
    right_hip = landmarks.get(Landmark.RIGHT_HIP)
    if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
        return float("nan")

    usable = left_shoulder.valid & right_shoulder.valid & left_hip.valid & right_hip.valid
    if not np.any(usable):
        return float("nan")

    shoulders = (left_shoulder.position + right_shoulder.position) / 2.0
    hips = (left_hip.position + right_hip.position) / 2.0
    spans = np.linalg.norm((shoulders - hips)[usable, :2], axis=1)
    return float(np.median(spans))


class SignalError(ValueError):
    """The signals a swing is read from could not be built."""


@dataclass(frozen=True)
class HandTrack:
    """The tracked hand point for one clip, and where it came from."""

    source: HandSource
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    visibility: NDArray[np.float64]
    valid: NDArray[np.bool_]
    observed: NDArray[np.bool_]

    def __len__(self) -> int:
        return int(self.valid.size)


@dataclass(frozen=True)
class SwingSignals:
    """Scalar signals through time, all sampled on the clip's own timestamps."""

    t: NDArray[np.float64]
    hand: HandTrack
    speed: NDArray[np.float64]
    height: NDArray[np.float64]
    shoulder_angle_deg: NDArray[np.float64]
    hip_angle_deg: NDArray[np.float64]
    torso_length: float

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def peak_speed(self) -> float:
        """Fastest the hands moved, or NaN if they were never tracked."""
        if not np.any(np.isfinite(self.speed)):
            return float("nan")
        return float(np.nanmax(self.speed))

    @property
    def travel(self) -> float:
        """How far the hands ranged, as the diagonal of the box they stayed inside.

        A swing sweeps the hands across a large part of the frame. A subject
        standing still does not, whatever their landmarks are doing frame to
        frame, which is what makes this the quantity worth comparing against the
        noise.
        """
        usable = self.hand.valid
        if not np.any(usable):
            return float("nan")
        x = self.hand.position[usable, 0]
        height = self.height[usable]
        return float(np.hypot(np.ptp(x), np.ptp(height)))

    @property
    def travel_ratio(self) -> float:
        """How far the hands ranged, in torso lengths.

        The measurement that decides whether a swing happened at all. Judging
        travel against the subject's own body makes it independent of framing:
        the same swing filmed from twice as far away halves every distance in
        the frame, including the torso, and leaves this unchanged.

        A ratio of *speeds* was tried first and abandoned. Peak hand speed
        against the speed seen during the quietest frames sounds equivalent, and
        on a clip of somebody standing still it compares the largest noise spike
        against the median noise -- comfortably above ten, reporting a swing
        where nobody moved.
        """
        distance, scale = self.travel, self.torso_length
        if not np.isfinite(distance) or not np.isfinite(scale) or scale <= 0:
            return float("nan")
        return float(distance / scale)


def _midpoint(left: FilteredLandmark, right: FilteredLandmark) -> HandTrack:
    valid = left.valid & right.valid
    return HandTrack(
        source=HandSource.MIDPOINT,
        position=(left.position + right.position) / 2.0,
        velocity=(left.velocity + right.velocity) / 2.0,
        visibility=np.fmin(left.visibility, right.visibility),
        valid=valid,
        observed=left.observed & right.observed,
    )


def _single(entry: FilteredLandmark, source: HandSource) -> HandTrack:
    return HandTrack(
        source=source,
        position=entry.position,
        velocity=entry.velocity,
        visibility=entry.visibility,
        valid=entry.valid,
        observed=entry.observed,
    )


def choose_hand(filtered: FilteredSequence) -> HandTrack:
    """Pick the hand point for a clip: the option tracked on the most frames.

    Decided once for the whole clip. The alternative -- taking the midpoint
    where both wrists are seen and a single wrist elsewhere -- sounds strictly
    better and is not: the midpoint and a wrist are points about half a hand's
    width apart, so every switch injects a step into the position signal, which
    the differentiator reports as a velocity spike. Those spikes land exactly
    where tracking is hardest, which on a golf swing is around impact.

    Ties go to the midpoint, which averages two independent estimates and is
    therefore the quieter signal when it is available at all.
    """
    left = filtered.landmarks.get(Landmark.LEFT_WRIST)
    right = filtered.landmarks.get(Landmark.RIGHT_WRIST)
    if left is None or right is None:
        raise SignalError(
            "Phase detection needs both wrist landmarks to be filtered; the sequence "
            "supplied was restricted to a subset that excludes one of them."
        )

    candidates = (
        _midpoint(left, right),
        _single(left, HandSource.LEFT_WRIST),
        _single(right, HandSource.RIGHT_WRIST),
    )
    return max(candidates, key=lambda track: int(np.count_nonzero(track.valid)))


def _line_angle_deg(
    start: FilteredLandmark | None, end: FilteredLandmark | None, count: int
) -> NDArray[np.float64]:
    """Tilt of the line through two landmarks, in degrees, positive end-up.

    Negated relative to raw image coordinates so that a positive angle means the
    `end` point is *higher* in the frame, matching how the rest of this module
    speaks about the body. Zero is level.

    Folded onto (-90, 90] because a shoulder line has an **orientation, not a
    direction**. Taken as a vector angle, a nearly level line sits beside the
    ±180 discontinuity, and the signal flips the full 360 between consecutive
    frames every time the tilt crosses zero -- which for shoulders and hips is
    most of a swing. Folding moves the discontinuity to vertical, where these
    lines never go.
    """
    angles = np.full(count, np.nan, dtype=np.float64)
    if start is None or end is None:
        return angles

    delta = end.position - start.position
    usable = start.valid & end.valid
    directed = np.degrees(np.arctan2(-delta[usable, 1], delta[usable, 0]))
    angles[usable] = (directed + 90.0) % 180.0 - 90.0
    return angles


def swing_signals(filtered: FilteredSequence) -> SwingSignals:
    """Reduce a filtered sequence to the signals a swing is read from."""
    if filtered.space is not LandmarkSpace.IMAGE:
        # HIP_LOCAL is hip-centred, so the hands' height in it is measured from a
        # point that moves with the body. "Highest the hands reached" then means
        # something other than what the rules below assume.
        raise SignalError(
            f"Phase detection reads IMAGE-space trajectories; this sequence is in "
            f"{filtered.space.value}. HIP_LOCAL is centred on the hips, so hand height "
            "in it is relative to a moving origin rather than to the frame."
        )

    count = len(filtered.t)
    hand = choose_hand(filtered)

    # Speed in the image plane only. The z channel of IMAGE space is MediaPipe's
    # own depth estimate from a single camera, on an unstated scale; including it
    # would mix a measured quantity with a guessed one inside one number.
    speed = np.linalg.norm(hand.velocity[:, :2], axis=1)
    speed[~hand.valid] = np.nan

    # y grows downward in IMAGE space. Converted once, here.
    height = np.where(hand.valid, 1.0 - hand.position[:, 1], np.nan)

    return SwingSignals(
        t=filtered.t,
        hand=hand,
        speed=speed,
        height=height,
        torso_length=_torso_length(filtered),
        shoulder_angle_deg=_line_angle_deg(
            filtered.landmarks.get(Landmark.LEFT_SHOULDER),
            filtered.landmarks.get(Landmark.RIGHT_SHOULDER),
            count,
        ),
        hip_angle_deg=_line_angle_deg(
            filtered.landmarks.get(Landmark.LEFT_HIP),
            filtered.landmarks.get(Landmark.RIGHT_HIP),
            count,
        ),
    )
