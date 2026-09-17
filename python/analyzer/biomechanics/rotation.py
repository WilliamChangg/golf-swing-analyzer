"""Rotation metrics: shoulder turn, pelvis turn, X-factor, and the line tilts.

Turn is the quantity a single camera is worst at, and the one golf cares most
about. The approach here is foreshortening: a shoulder line of fixed real width
projects to less than that width as it turns away from the camera, by a factor
of the cosine of the turn, so two measured lengths give the angle with no camera
parameters at all.

    span(t) = span_0 * cos(theta(t))    =>    theta = arccos(span / span_0)

What that buys and what it costs are both worth stating plainly.

**It is a real measurement.** The projected span is measured every frame, and
`span_0` is measured too. Nothing is assumed about focal length, distance or
sensor size, because the ratio of two lengths in the same image cancels all of
them.

**It cannot tell direction.** The cosine is even, so a turn away from the camera
and a turn towards it shorten the line identically. Every value here is a
magnitude. Recovering the sign needs depth, which is Phase 9.

**It rests on one assumption, and the assumption is checked.** `span_0` is the
line's width when square to the camera, and it is taken as the **median of the
address phase** -- because that is where the method's premise says the player is
square, set up to the ball with the camera facing them.

Taking it instead as the widest view anywhere in the clip is the obvious
alternative and is worse. A maximum over a clip is a maximum over that clip's
landmark noise, so it selects the one frame where the estimator most overstated
the span: on the face-on reference clip that is a frame just past impact where
the shoulders read 12% wider than the player can be, and every rotation in the
clip is then measured against an error.

The premise is checked separately, and a failure is **refused** rather than
warned about. If the line projects more than `max_reference_excess` times wider
anywhere in the clip than it did at address, the player was not square there,
and the result would not be a slightly-wrong version of the truth -- it would be
the angle away from a pose they never held.

That is what a down-the-line clip triggers. Seen along the target line the
shoulders start nearly end-on and open through the backswing, so they project
over ten times wider at the top than at address. The engine notices and says so.

Line **tilts** are a different quantity and are kept separate. The tilt is the
angle of the shoulder line away from level in the image -- a projected angle,
available from any view, and not a turn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.biomechanics.anchors import Anchors
from analyzer.biomechanics.body import Body
from analyzer.biomechanics.geometry import foreshortening_angle_deg, line_tilt_deg
from analyzer.biomechanics.registry import (
    Anchor,
    MetricName,
    in_plane_fraction,
    measure_series,
)
from analyzer.contracts.metrics import (
    CameraView,
    Metric,
    MetricConfig,
    RefusedMetric,
    RotationReference,
)
from analyzer.contracts.pose import Landmark

_SHOULDERS = (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER)
_HIPS = (Landmark.LEFT_HIP, Landmark.RIGHT_HIP)


@dataclass(frozen=True)
class Segment:
    """One body line, and everything the rotation of it is measured against."""

    label: str
    landmarks: tuple[Landmark, Landmark]
    span: NDArray[np.float64]
    reference: RotationReference | None
    turn_deg: NDArray[np.float64]
    method: NDArray[np.float64]


def _segment(
    body: Body,
    label: str,
    landmarks: tuple[Landmark, Landmark],
    span: NDArray[np.float64],
    address: Anchor | None,
    config: MetricConfig,
) -> Segment:
    """Measure one line's turn against its projected width at address."""
    empty = Segment(label, landmarks, span, None, np.full_like(span, np.nan), np.zeros_like(span))
    if not np.any(np.isfinite(span)) or address is None:
        return empty

    seen = [
        frame for frame in address.frames if 0 <= frame < span.size and np.isfinite(span[frame])
    ]
    if not seen:
        return empty

    # Median over the address phase, not a single frame of it. The subject is
    # still there, so every frame is a measurement of the same pose, and the
    # median of them is the estimate that one jumped landmark cannot move.
    baseline = float(np.median(span[seen]))
    if baseline <= 0.0:
        return empty

    widest_frame = int(np.nanargmax(span))
    widest = float(span[widest_frame])
    excess = widest / baseline

    turn = foreshortening_angle_deg(span, baseline)

    # The sine of the angle found, which is exactly how sharply the arccos turns
    # a change in length into a change in angle there: d(theta)/d(span) goes as
    # 1/sin(theta). A large turn is measured sharply and a small one is swamped
    # by landmark noise, and this says which of the two a given value is.
    method = np.abs(np.sin(np.radians(turn)))

    return Segment(
        label=label,
        landmarks=landmarks,
        span=span,
        reference=RotationReference(
            landmarks=label,
            span=baseline / body.torso_length,
            frames=seen,
            widest_span=widest / body.torso_length,
            widest_frame=widest_frame,
            excess=excess,
            square_at_address=excess <= config.max_reference_excess,
        ),
        turn_deg=turn,
        method=method,
    )


def segments(body: Body, anchors: Anchors, config: MetricConfig) -> tuple[Segment, Segment]:
    """The shoulder and hip lines, each with its baseline and turn series."""
    return (
        _segment(body, "shoulders", _SHOULDERS, body.shoulder_span, anchors.address, config),
        _segment(body, "hips", _HIPS, body.hip_span, anchors.address, config),
    )


def _reference_problem(segment: Segment, config: MetricConfig) -> str | None:
    """Why this segment's turn cannot be measured, or None if it can."""
    reference = segment.reference
    if reference is None:
        return (
            f"The {segment.label} were not tracked through an address phase, so there is "
            "no square-to-camera baseline to measure a rotation away from."
        )
    # No check here on how wide the line projects at address. That measurement
    # is what decides the view, and Phase 6 made the view the single owner of
    # "this recording does not contain this quantity" -- a second threshold on
    # the same number, a few hundredths away from the first, would refuse the
    # same clips for a narrower-sounding reason.
    if not reference.square_at_address:
        return (
            f"The {segment.label} projected {reference.excess:.2f} times wider on frame "
            f"{reference.widest_frame} than they did at address, over the "
            f"{config.max_reference_excess:g} allowed. Foreshortening measures rotation "
            "away from a square view, and takes address as that view because the player "
            "sets up facing the camera. A line that opens up well past its address width "
            "mid-swing was not square at address -- which is what a down-the-line "
            "recording looks like -- so every turn would be measured from a pose the "
            "player never held."
        )
    return None


def metrics(
    body: Body, anchors: Anchors, config: MetricConfig, view: CameraView
) -> tuple[list[Metric], list[RefusedMetric], list[RotationReference]]:
    """Every rotation metric this clip supports, with the references they used."""
    produced: list[Metric] = []
    refused: list[RefusedMetric] = []

    shoulders, hips = segments(body, anchors, config)
    references = [
        segment.reference for segment in (shoulders, hips) if segment.reference is not None
    ]

    instants: list[Anchor] = [
        anchor for anchor in (anchors.address, anchors.top, anchors.impact) if anchor is not None
    ]
    # Turns are not reported at address. Address *is* the baseline they are
    # measured against, so the value there is zero by construction -- a
    # restatement of the method rather than a measurement of the swing.
    turned: list[Anchor] = [
        anchor for anchor in (anchors.top, anchors.impact) if anchor is not None
    ]

    # --- tilts: available from any view, so computed before the turn checks ---
    for name, segment in (
        (MetricName.SHOULDER_TILT, shoulders),
        (MetricName.PELVIS_TILT, hips),
    ):
        left, right = segment.landmarks
        tilt = body.masked(line_tilt_deg(body.points[left], body.points[right]), left, right)
        plane = in_plane_fraction(segment.span)
        for anchor in instants:
            found = measure_series(
                name,
                tilt,
                anchor,
                view=view,
                visibility=body.visibility,
                landmarks=segment.landmarks,
                method=plane,
                methodology=(
                    f"Angle of the {segment.label} line away from level in the image, "
                    "measured left to right across the frame because the line has an "
                    "orientation rather than a direction. Positive means the landmark "
                    "further right in the frame is the higher of the two, which is a "
                    "statement about the picture rather than about which side of the "
                    "body is raised."
                ),
            )
            _record(produced, refused, found, name, anchor, f"the {segment.label}")

    # --- turns, and X-factor, which needs both -----------------------------
    problems = {segment.label: _reference_problem(segment, config) for segment in (shoulders, hips)}

    for name, segment in (
        (MetricName.SHOULDER_TURN, shoulders),
        (MetricName.PELVIS_TURN, hips),
    ):
        problem = problems[segment.label]
        if problem is not None:
            refused.append(RefusedMetric(name=name, reason=problem))
            continue
        assert segment.reference is not None  # noqa: S101 - implied by problem being None
        for anchor in turned:
            found = measure_series(
                name,
                segment.turn_deg,
                anchor,
                view=view,
                visibility=body.visibility,
                landmarks=segment.landmarks,
                method=segment.method,
                methodology=(
                    f"arccos of the projected {segment.label} span divided by the "
                    f"{segment.reference.span:.2f} torso lengths it spanned at address, "
                    "where the player is square to a face-on camera. A magnitude: "
                    "foreshortening cannot distinguish a turn from its mirror image. "
                    "Confidence falls with the sine of the angle, which is how sharply the "
                    "arccos resolves it there."
                ),
            )
            _record(produced, refused, found, name, anchor, f"the {segment.label}")

    blocking = [problem for problem in problems.values() if problem is not None]
    if blocking:
        refused.append(
            RefusedMetric(
                name=MetricName.X_FACTOR,
                reason=(
                    "X-factor is the difference of the shoulder and pelvis turns, and "
                    f"at least one of those could not be measured. {blocking[0]}"
                ),
            )
        )
        return produced, refused, references

    separation = shoulders.turn_deg - hips.turn_deg
    # The weaker of the two conditioning factors: a difference is no better
    # determined than its worse-determined term.
    method = np.fmin(shoulders.method, hips.method)
    for anchor in turned:
        found = measure_series(
            MetricName.X_FACTOR,
            separation,
            anchor,
            view=view,
            visibility=body.visibility,
            landmarks=(*_SHOULDERS, *_HIPS),
            method=method,
            methodology=(
                "Shoulder turn minus pelvis turn, both measured by foreshortening "
                "against their address widths. Both terms are magnitudes, so this says "
                "how far apart the two are rather than which way either went; on a swing "
                "where both turn the same way that is the separation, and it is an "
                "approximation labelled as one."
            ),
        )
        _record(produced, refused, found, MetricName.X_FACTOR, anchor, "the torso")

    return produced, refused, references


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
