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

## Coordinates arrive already corrected

This module reads **FRAME_WIDTHS**, in which y already increases upward and both
axes are in the same unit. Neither is true of the IMAGE coordinates a pose
estimator emits, and both used to be fixed here: `height` was computed as
`1 - y`, and the line angles negated their own y.

Both corrections moved below the filter in Phase 6, which is a better place for
two reasons. A sign flip applied to positions before fitting emerges correctly
signed in the velocity and the acceleration, where applying it afterwards needs
a second correction kept in step with the first. And an anisotropic frame makes
every distance here -- hand travel, torso length, the speed the whole detection
keys on -- wrong by a factor that depends on the shape of the recording, which
is not something a detector should have to think about.

`height` survives as a name because the rules are about the hands being high or
low and reading `position[:, 1]` at each of them would be worse. It is now
simply the y channel.
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

    Chosen by the **longest unbroken run** of tracked frames rather than by how
    many frames are tracked in total. A swing is one continuous event: address,
    backswing, downswing and finish all have to fall inside a single run, and a
    trajectory with a hole in the middle cannot contain one across the hole
    however well tracked it is either side.

    The two are not the same choice, and the difference decides whole clips.
    Down-the-line footage hides one wrist behind the other, and which one it
    hides changes through the swing: on `data/dtl/iron_dtl.mp4` the right wrist
    is visible through address and the backswing while the left is not, and the
    left takes over once the right disappears after impact. Counting frames
    picks the left (59 against 48) and misses the entire address and backswing,
    so no swing is found. Counting the longest run picks the right (48 against
    44) and the swing is detected with every event confident.

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
    # Total coverage breaks a tie in the run length, and `max` keeps the first
    # of equals, so the midpoint wins a tie in both.
    return max(
        candidates,
        key=lambda track: (
            longest_run(track.valid),
            int(np.count_nonzero(track.valid)),
        ),
    )


def longest_run(mask: NDArray[np.bool_]) -> int:
    """Length of the longest unbroken run of True."""
    if mask.size == 0:
        return 0
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _line_angle_deg(
    start: FilteredLandmark | None, end: FilteredLandmark | None, count: int
) -> NDArray[np.float64]:
    """Tilt of the line through two landmarks away from level, in degrees, on [-90, 90].

    A shoulder line has an **orientation, not a direction**, so the answer must
    not depend on which landmark was passed first. It is made canonical by
    measuring left to right *across the frame*: whichever landmark has the
    smaller x is treated as the start. **Positive means the landmark further
    right in the frame is the higher one.** Zero is level.

    No sign flip here any more: FRAME_WIDTHS already has y pointing up.

    This convention is shared with `biomechanics.geometry.line_tilt_deg` and the
    two must not diverge: a reader comparing a plotted shoulder angle against a
    reported metric should not have to work out which sign each used.

    An earlier version folded the plain vector angle onto (-90, 90], which fixes
    the +-180 jump a near-level line otherwise produces and quietly redefines the
    sign while doing it -- folding reverses any leftward vector, so "positive
    means `end` is higher" held only while `end` was on the right of the frame.
    A player facing the camera has their right shoulder on the frame's left, so
    the stated meaning inverted for precisely the footage this system reads.
    """
    angles = np.full(count, np.nan, dtype=np.float64)
    if start is None or end is None:
        return angles

    delta = end.position - start.position
    usable = start.valid & end.valid
    dx, dy = delta[usable, 0], delta[usable, 1]
    # Point the segment rightwards across the frame, so the tilt describes the
    # line rather than the order the landmarks arrived in.
    leftwards = dx < 0.0
    angles[usable] = np.degrees(
        np.arctan2(np.where(leftwards, -dy, dy), np.where(leftwards, -dx, dx))
    )
    return angles


def swing_signals(filtered: FilteredSequence) -> SwingSignals:
    """Reduce a filtered sequence to the signals a swing is read from."""
    if filtered.space is not LandmarkSpace.FRAME_WIDTHS:
        # Two different objections, and both matter. IMAGE is anisotropic and its
        # y points down, so every distance here would be wrong by the shape of
        # the frame and every height inverted. HIP_LOCAL is hip-centred, so the
        # hands' height in it is measured from a point that moves with the body,
        # and "the highest the hands reached" means something other than what the
        # rules below assume.
        raise SignalError(
            f"Phase detection reads FRAME_WIDTHS trajectories; this sequence is in "
            f"{filtered.space.value}. IMAGE has y pointing down and a different unit on "
            "each axis, and HIP_LOCAL is centred on a moving origin -- so hand height "
            "and hand travel would both mean something else."
        )

    count = len(filtered.t)
    hand = choose_hand(filtered)

    # Speed in the image plane only. The z channel is MediaPipe's own depth
    # estimate from a single camera, on an unstated scale; including it would mix
    # a measured quantity with a guessed one inside one number.
    speed = np.linalg.norm(hand.velocity[:, :2], axis=1)
    speed[~hand.valid] = np.nan

    # Already upward-positive, and in the same unit as x. Both were this module's
    # job before Phase 6 moved the conversion below the filter.
    height = np.where(hand.valid, hand.position[:, 1], np.nan)

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
