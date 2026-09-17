"""Hand and arm metrics, and working out which arm is which.

## Lead and trail are not left and right

Every coaching quantity about the arms is stated for the *lead* arm -- the one
nearer the target, which stays straight through the backswing -- or the *trail*
arm behind it. Which physical arm that is depends on the player's handedness,
and a pose sequence does not say. Assuming right-handed would silently mislabel
every left-handed player's swing, and asking the user for something the video
already shows is worse than measuring it.

It is measurable. **At the top of the backswing the hands sit over the trail
shoulder**, for either handedness and whether the camera is in front of the
player or behind them: that is what taking the club back means. Projecting the
hands' offset from the middle of the chest onto the shoulder line names that
shoulder, and how large the projection is says how confidently.

The method declines cleanly on the view where it should. Down the line, the
shoulder line points nearly at the camera, so it projects to almost nothing and
the offset along it is noise -- and the answer returned is "undecidable", which
is correct, because that view genuinely does not show which side the hands went.
The arm metrics that need a side are then refused rather than assigned by a coin
flip, and the left and right arm angles remain available to anyone who wants
them under those names.

## Hand path

Summed frame to frame rather than measured end to end. A swing's hands travel an
arc; the straight line between its ends is the chord, which is shorter by
roughly a third on a full backswing, and reporting one as the other would be a
quiet and consistent underestimate.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from analyzer.biomechanics.anchors import Anchors
from analyzer.biomechanics.body import Body
from analyzer.biomechanics.geometry import (
    interior_angle_deg,
    lengths,
    midpoint,
    project_onto,
)
from analyzer.biomechanics.registry import (
    Anchor,
    MetricName,
    build_metric,
    in_plane_fraction,
    measure_series,
    visibility_factor,
)
from analyzer.contracts.metrics import (
    BodySide,
    LeadSide,
    Metric,
    MetricConfig,
    RefusedMetric,
)
from analyzer.contracts.pose import Landmark

_ARMS: dict[BodySide, tuple[Landmark, Landmark, Landmark]] = {
    BodySide.LEFT: (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
    BodySide.RIGHT: (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
}

_UNDECIDED = LeadSide(
    side=None,
    margin=float("nan"),
    shoulder_span_ratio=float("nan"),
    methodology=(
        "Not determined: the top of the backswing was not located, or the shoulders "
        "were not tracked there."
    ),
)


def infer_lead_side(body: Body, anchors: Anchors, config: MetricConfig) -> LeadSide:
    """Name the leading side from where the hands sat at the top, or decline to.

    Returns the **lead** side: the trail shoulder is the one the hands went over,
    so the lead side is the other one.
    """
    if anchors.top is None or not anchors.top.frames:
        return _UNDECIDED

    frame = anchors.top.frames[0]
    left, right = body.points[Landmark.LEFT_SHOULDER], body.points[Landmark.RIGHT_SHOULDER]
    if not (
        body.valid[Landmark.LEFT_SHOULDER][frame]
        and body.valid[Landmark.RIGHT_SHOULDER][frame]
        and body.hand.valid[frame]
    ):
        return _UNDECIDED

    line = right[frame] - left[frame]
    half_span = float(lengths(line)) / 2.0
    span_ratio = float(lengths(line)) / body.torso_length
    if half_span <= 0.0:
        return _UNDECIDED

    offset = float(project_onto(body.hand.position[frame] - midpoint(left, right)[frame], line))
    margin = float(np.clip(abs(offset) / half_span, 0.0, 1.0))

    explanation = (
        "The hands sit over the trail shoulder at the top of the backswing, so the "
        "lead side is the other one. Measured by projecting the hands' offset from "
        f"the chest midpoint onto the shoulder line on frame {frame}: "
        f"{margin:.2f} of a half shoulder span, across a line projecting "
        f"{span_ratio:.2f} torso lengths."
    )

    if span_ratio < config.min_shoulder_span_ratio:
        return LeadSide(
            side=None,
            margin=margin,
            shoulder_span_ratio=span_ratio,
            methodology=(
                f"{explanation} Below the {config.min_shoulder_span_ratio:g} torso "
                "lengths this needs: the shoulder line is pointing nearly at the "
                "camera, which is what a down-the-line recording looks like, and it "
                "does not show which side the hands went."
            ),
        )
    if margin < config.min_lead_side_margin:
        return LeadSide(
            side=None,
            margin=margin,
            shoulder_span_ratio=span_ratio,
            methodology=(
                f"{explanation} Below the {config.min_lead_side_margin:g} needed to "
                "name a side: the hands are near enough the middle of the chest that "
                "the answer would be a coin flip."
            ),
        )

    # offset > 0 means the hands are towards the right shoulder, making that the
    # trail side, so the lead side is the left.
    side = BodySide.LEFT if offset > 0 else BodySide.RIGHT
    return LeadSide(
        side=side,
        margin=margin,
        shoulder_span_ratio=span_ratio,
        methodology=explanation,
    )


def arm_angle_deg(body: Body, side: BodySide) -> NDArray[np.float64]:
    """Projected shoulder-elbow-wrist angle on one side, per frame. 180 is straight."""
    shoulder, elbow, wrist = _ARMS[side]
    angle = interior_angle_deg(body.points[shoulder], body.points[elbow], body.points[wrist])
    return body.masked(angle, shoulder, elbow, wrist)


def arm_in_plane(body: Body, side: BodySide) -> NDArray[np.float64]:
    """How squarely the arm is seen, per frame: the weaker of upper arm and forearm."""
    shoulder, elbow, wrist = _ARMS[side]
    upper = in_plane_fraction(
        body.masked(lengths(body.points[elbow] - body.points[shoulder]), shoulder, elbow)
    )
    fore = in_plane_fraction(
        body.masked(lengths(body.points[wrist] - body.points[elbow]), elbow, wrist)
    )
    return np.fmin(upper, fore)


def hand_path_length(body: Body, frames: tuple[int, ...]) -> tuple[float, list[int]]:
    """Distance the hands travelled along their path over a span, in torso lengths.

    Summed step by step over consecutive tracked frames. A step across a gap in
    tracking is not counted: the hands did travel during it, but by an unknown
    amount, and adding the straight line across the hole would report a guess as
    a measurement. The frames actually summed are returned so the caller can say
    which they were.
    """
    usable = [frame for frame in frames if 0 <= frame < len(body) and body.hand.valid[frame]]
    if len(usable) < 2:
        return float("nan"), usable

    total = 0.0
    counted: list[int] = [usable[0]]
    for previous, current in itertools.pairwise(usable):
        if current != previous + 1:
            continue
        total += float(lengths(body.hand.position[current] - body.hand.position[previous]))
        counted.append(current)
    return total / body.torso_length, counted


def metrics(
    body: Body, anchors: Anchors, lead: LeadSide
) -> tuple[list[Metric], list[RefusedMetric]]:
    """Every hand and arm metric this clip supports."""
    produced: list[Metric] = []
    refused: list[RefusedMetric] = []

    _arm_metrics(body, anchors, lead, produced, refused)
    _hand_metrics(body, anchors, produced, refused)
    return produced, refused


def _arm_metrics(
    body: Body,
    anchors: Anchors,
    lead: LeadSide,
    produced: list[Metric],
    refused: list[RefusedMetric],
) -> None:
    instants = [anchor for anchor in (anchors.top, anchors.impact) if anchor is not None]

    if lead.side is None:
        for name in (MetricName.LEAD_ARM_ANGLE, MetricName.TRAIL_ARM_ANGLE):
            refused.append(
                RefusedMetric(
                    name=name,
                    reason=(
                        "Which arm leads could not be determined, and reporting a lead "
                        f"arm angle without knowing which arm that is would be a guess. {lead.methodology}"
                    ),
                )
            )
        return

    trail = BodySide.RIGHT if lead.side is BodySide.LEFT else BodySide.LEFT
    for name, side in ((MetricName.LEAD_ARM_ANGLE, lead.side), (MetricName.TRAIL_ARM_ANGLE, trail)):
        series, plane = arm_angle_deg(body, side), arm_in_plane(body, side)
        for anchor in instants:
            found = measure_series(
                name,
                series,
                anchor,
                visibility=body.visibility,
                landmarks=_ARMS[side],
                method=plane,
                methodology=(
                    f"Projected shoulder-elbow-wrist angle on the {side.value} side, "
                    "which this clip's hand position at the top identifies as the "
                    f"{'leading' if side is lead.side else 'trailing'} arm. 180 degrees "
                    "is a straight arm. Scored by the less squarely seen of the upper "
                    "arm and the forearm."
                ),
            )
            _record(produced, refused, found, name, anchor, f"the {side.value} arm")


def _hand_metrics(
    body: Body,
    anchors: Anchors,
    produced: list[Metric],
    refused: list[RefusedMetric],
) -> None:
    for anchor in (anchors.backswing, anchors.downswing):
        if anchor is None:
            continue
        value, frames = hand_path_length(body, anchor.frames)
        if not np.isfinite(value) or len(frames) < 2:
            refused.append(
                RefusedMetric(
                    name=MetricName.HAND_PATH_LENGTH,
                    reason=(
                        f"The hands were tracked on fewer than two frames {anchor.label}, "
                        "so no path could be summed."
                    ),
                )
            )
            continue
        produced.append(
            build_metric(
                MetricName.HAND_PATH_LENGTH,
                value,
                anchor,
                source_frames=frames,
                observation=_hand_observation(body, frames),
                method=1.0,
                methodology=(
                    f"Sum of the frame-to-frame distances travelled by the tracked hand "
                    f"point ({body.hand.source.value}) {anchor.label}, in torso lengths. "
                    "Summed along the path rather than measured end to end, so an arc is "
                    "not reported as its chord. Steps across untracked frames are left "
                    "out rather than bridged."
                ),
            )
        )

    downswing = anchors.downswing
    if downswing is None:
        refused.append(
            RefusedMetric(
                name=MetricName.PEAK_HAND_SPEED,
                reason="No downswing phase was detected, so there is no interval to take a peak over.",
            )
        )
        return

    speed = np.where(body.hand.valid, lengths(body.hand.velocity), np.nan)
    frames = [frame for frame in downswing.frames if 0 <= frame < speed.size]
    finite = [frame for frame in frames if np.isfinite(speed[frame])]
    if not finite:
        refused.append(
            RefusedMetric(
                name=MetricName.PEAK_HAND_SPEED,
                reason="The hands carried no filtered velocity anywhere in the downswing.",
            )
        )
        return

    fastest = max(finite, key=lambda frame: speed[frame])
    produced.append(
        build_metric(
            MetricName.PEAK_HAND_SPEED,
            float(speed[fastest]) / body.torso_length,
            downswing,
            source_frames=[fastest],
            observation=_hand_observation(body, [fastest]),
            method=1.0,
            methodology=(
                f"Greatest speed of the tracked hand point ({body.hand.source.value}) in "
                "the image plane during the downswing, in torso lengths per second. Taken "
                "from the filter's own velocity coefficient rather than from differences "
                "of smoothed positions. Flattened by the smoothing window: a window as "
                "wide as a large fraction of the downswing under-reports the peak, which "
                "the filtering report quantifies."
            ),
        )
    )


def _hand_observation(body: Body, frames: list[int]) -> float:
    """Visibility of the hand point over given frames.

    Read from the hand track's own channel rather than from a named landmark,
    because which landmark the track came from was Phase 4's choice and may be
    one wrist, the other, or the midpoint of both.
    """
    return visibility_factor((body.hand.visibility,), frames)


def _record(
    produced: list[Metric],
    refused: list[RefusedMetric],
    found: Metric | None,
    name: MetricName,
    anchor: Anchor,
    subject: str,
) -> None:
    if found is not None:
        produced.append(found)
        return
    refused.append(
        RefusedMetric(
            name=name,
            event=anchor.event,
            reason=(
                f"No filtered position for {subject} {anchor.label}. The landmarks were "
                "either gated out as unreliable or left unfilled across a gap too long "
                "to bridge."
            ),
        )
    )
