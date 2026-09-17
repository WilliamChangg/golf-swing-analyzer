"""Posture metrics: spine angle, knee flexion, and how far the body moved.

Two kinds of quantity live here and they behave differently.

**Angles** -- spine tilt, knee flex -- are projected. The angle between two
segments in the image equals the angle between them in the world only when both
lie in the image plane, and is smaller otherwise. Nothing here can measure how
far from that they are, but the *fraction of each segment that is in the plane*
is measurable, and it is what scores the confidence: an angle read across a
thigh seen nearly end-on is both further from the truth and noisier, because the
same landmark error subtends more angle across a shorter baseline.

**Displacements** -- hip sway, head sway, head lift -- are differences from
where the body was at address, in torso lengths. These are exact statements
about the image: this is how far the head moved across the frame. What they
cannot see is motion towards or away from the camera, which contributes nothing
to them at all. That is a property of one camera, not of the arithmetic, and it
is what `MetricBasis.IMAGE_PLANE` records.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from analyzer.biomechanics.anchors import Anchors, reference_point
from analyzer.biomechanics.body import Body
from analyzer.biomechanics.geometry import (
    distance,
    interior_angle_deg,
    tilt_from_vertical_deg,
)
from analyzer.biomechanics.registry import (
    Anchor,
    MetricName,
    in_plane_fraction,
    measure_series,
)
from analyzer.contracts.metrics import CameraView, Metric, RefusedMetric
from analyzer.contracts.pose import Landmark

_STRAIGHT_LEG_DEG = 180.0

_SPINE_LANDMARKS = (
    Landmark.LEFT_SHOULDER,
    Landmark.RIGHT_SHOULDER,
    Landmark.LEFT_HIP,
    Landmark.RIGHT_HIP,
)

_KNEES: dict[MetricName, tuple[Landmark, Landmark, Landmark]] = {
    MetricName.LEFT_KNEE_FLEX: (Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
    MetricName.RIGHT_KNEE_FLEX: (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
}


def spine_tilt_deg(body: Body) -> NDArray[np.float64]:
    """Lean of the hip-midpoint-to-shoulder-midpoint line from vertical, per frame."""
    return body.masked(tilt_from_vertical_deg(body.hip_mid, body.shoulder_mid), *_SPINE_LANDMARKS)


def spine_in_plane(body: Body) -> NDArray[np.float64]:
    """Fraction of the torso segment lying in the image plane, per frame."""
    return in_plane_fraction(
        body.masked(distance(body.hip_mid, body.shoulder_mid), *_SPINE_LANDMARKS)
    )


def knee_flex_deg(body: Body, name: MetricName) -> NDArray[np.float64]:
    """Flexion at one knee, per frame. Zero is a straight leg.

    Reported as flexion rather than as the interior angle so that larger means
    more bent, which is how the quantity is spoken about. The interior angle is
    what is measured; the subtraction is named here rather than left for a
    reader to reverse.
    """
    hip, knee, ankle = _KNEES[name]
    angle = interior_angle_deg(body.points[hip], body.points[knee], body.points[ankle])
    return body.masked(_STRAIGHT_LEG_DEG - angle, hip, knee, ankle)


def knee_in_plane(body: Body, name: MetricName) -> NDArray[np.float64]:
    """How squarely the leg is seen, per frame: the weaker of thigh and shin.

    The weaker of the two, because an angle is only as well determined as its
    worse-seen arm -- scoring it by the better one would report a confident
    angle whenever either segment happened to lie in the plane.
    """
    hip, knee, ankle = _KNEES[name]
    thigh = in_plane_fraction(body.masked(distance(body.points[hip], body.points[knee]), hip, knee))
    shin = in_plane_fraction(
        body.masked(distance(body.points[knee], body.points[ankle]), knee, ankle)
    )
    return np.fmin(thigh, shin)


def _displacement(
    body: Body, positions: NDArray[np.float64], origin: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    """Travel along one axis from a reference point, in torso lengths."""
    return body.in_torso_lengths(positions[:, axis] - origin[axis])


def metrics(
    body: Body, anchors: Anchors, view: CameraView
) -> tuple[list[Metric], list[RefusedMetric]]:
    """Every posture metric this clip supports, and the reasons for those it does not."""
    produced: list[Metric] = []
    refused: list[RefusedMetric] = []

    instants: list[Anchor] = [
        anchor for anchor in (anchors.address, anchors.top, anchors.impact) if anchor is not None
    ]

    # --- angles ---------------------------------------------------------
    tilt, tilt_plane = spine_tilt_deg(body), spine_in_plane(body)
    for anchor in instants:
        found = measure_series(
            MetricName.SPINE_TILT,
            tilt,
            anchor,
            view=view,
            visibility=body.visibility,
            landmarks=_SPINE_LANDMARKS,
            method=tilt_plane,
            methodology=(
                "Angle between vertical and the line from the hip midpoint to the "
                "shoulder midpoint, in the image plane, positive leaning towards the "
                "right of the frame. A projected angle: which anatomical bend it "
                "represents depends on where the camera stood."
            ),
        )
        _record(produced, refused, found, MetricName.SPINE_TILT, anchor, "the torso")

    for name in (MetricName.LEFT_KNEE_FLEX, MetricName.RIGHT_KNEE_FLEX):
        series, plane = knee_flex_deg(body, name), knee_in_plane(body, name)
        for anchor in instants:
            found = measure_series(
                name,
                series,
                anchor,
                view=view,
                visibility=body.visibility,
                landmarks=_KNEES[name],
                method=plane,
                methodology=(
                    "180 degrees minus the projected hip-knee-ankle angle. Scored by "
                    "the less squarely seen of the thigh and the shin, since an angle "
                    "is no better determined than its worse-seen arm."
                ),
            )
            _record(produced, refused, found, name, anchor, "the leg")

    # --- displacements from address --------------------------------------
    if anchors.address is None:
        for name in (MetricName.HIP_SWAY, MetricName.HEAD_SWAY, MetricName.HEAD_LIFT):
            refused.append(
                RefusedMetric(
                    name=name,
                    reason=(
                        "Measured as travel from the address pose, and this clip has no "
                        "address phase -- the hands were already moving in its first "
                        "tracked frame, so there is no still pose to measure from."
                    ),
                )
            )
        return produced, refused

    moved: list[Anchor] = [anchor for anchor in (anchors.top, anchors.impact) if anchor is not None]
    hip_origin = reference_point(body.hip_mid, anchors.address)
    head_origin = reference_point(body.points[Landmark.NOSE], anchors.address)

    displacements = (
        (
            MetricName.HIP_SWAY,
            _displacement(body, body.hip_mid, hip_origin, 0),
            (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
            "hip midpoint",
            "across the frame",
        ),
        (
            MetricName.HEAD_SWAY,
            _displacement(body, body.points[Landmark.NOSE], head_origin, 0),
            (Landmark.NOSE,),
            "nose",
            "across the frame",
        ),
        (
            MetricName.HEAD_LIFT,
            _displacement(body, body.points[Landmark.NOSE], head_origin, 1),
            (Landmark.NOSE,),
            "nose",
            "up the frame",
        ),
    )

    for name, series, landmarks, point, direction in displacements:
        masked = body.masked(series, *landmarks)
        for anchor in moved:
            found = measure_series(
                name,
                masked,
                anchor,
                view=view,
                visibility=body.visibility,
                landmarks=landmarks,
                # An image-plane displacement is exactly what it claims to be: the
                # distance between two points as the camera saw them. Nothing about
                # the method degrades it, so the factor is 1 and the limitation --
                # that motion towards the camera is invisible to it -- is carried by
                # the basis rather than hidden inside a number.
                method=1.0,
                methodology=(
                    f"Travel of the {point} {direction} from its median position over "
                    "the address phase, divided by the subject's median torso length. "
                    "Measures the projection only: motion towards or away from the "
                    "camera does not appear in it."
                ),
            )
            _record(produced, refused, found, name, anchor, f"the {point}")

    return produced, refused


def _record(
    produced: list[Metric],
    refused: list[RefusedMetric],
    found: Metric | None,
    name: MetricName,
    anchor: Anchor,
    subject: str,
) -> None:
    """File a computed metric, or the reason there is not one."""
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
