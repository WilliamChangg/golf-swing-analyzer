"""Metrics measured in three dimensions, from a reconstructed swing.

The first family in this engine whose numbers are statements about a body rather
than about a picture of one, and the first that declares a `requires` -- which is
what turns the calibration gate Phase 8 built, enforced and tested against
nothing into a gate that blocks something real.

## Every quantity here is chosen to need no scene frame

Triangulation produces metres in the **reference camera's** frame. It does not
produce a gravity direction or a target line, so "vertical" and "towards the
target" are not available, and a metric defined against either would be defined
against an axis nobody measured. See `analyzer/contracts/reconstruction.py`.

That rules out a 3D spine tilt, which is genuinely a loss -- forward posture
angle is a real coaching quantity and it needs a vertical. It does not rule out
the rotations, because a rotation can be measured about the **body's own axis**:
the address hip-midpoint to shoulder-midpoint direction is a measured direction
in the reconstruction, and a shoulder turn about it is the same number whichever
way the camera was pointing or which way up the room was. Nor does it rule out
joint angles or speeds, which are invariant to the frame entirely.

So the family is: three rotations about the measured spine axis, two true elbow
angles, and one speed in metres per second.

## The sign of a turn is measured, not assumed

A signed rotation needs a positive direction, and the obvious ones are not
available: "away from the target" needs the target line, and "clockwise from
above" needs an up. What *is* available is the swing itself. The shoulders turn
one way during the backswing, so the sense of the rotation at the top defines
positive for the whole clip, and every other anchor is reported in that sense.

That makes the numbers comparable across anchors within a clip -- a shoulder turn
of +88 at the top and +12 at impact says the player has unwound most of the way
-- and it makes them comparable across clips of the same player, since a swing's
backswing direction does not change. It does not make a left-handed player's
numbers comparable with a right-handed one's without noting which is which, and
nothing here pretends otherwise.

## Uncertainty, in the metric's own unit

Every point carries a propagated positional uncertainty in metres, and the
angular consequence of that is a division: a direction between two points
`L` apart, each uncertain by `sigma`, is uncertain by about `sqrt(2) sigma / L`
radians. The turns use the **projected** length of the body line onto the plane
perpendicular to the spine, which is the baseline the angle is actually read
across, so the uncertainty grows on its own as the geometry degenerates rather
than needing a separate check for it.

The `method` confidence factor is the sine of the ray convergence angle at the
landmarks used, which is the measured conditioning of the triangulation that
produced them -- the same quantity `ReconstructionQuality` gates on, read where
the metric was taken rather than averaged over the clip.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from analyzer.biomechanics.anchors import Anchors
from analyzer.biomechanics.registry import Anchor, build_metric, definition, observation_factor
from analyzer.contracts.metrics import (
    BodySide,
    CameraView,
    LeadSide,
    Metric,
    MetricConfig,
    MetricName,
    RefusedMetric,
)
from analyzer.contracts.pose import Landmark
from analyzer.reconstruction import ReconstructedSequence

_NO_RECONSTRUCTION = (
    "No 3D reconstruction was supplied for this clip, so there are no spatial "
    "positions to measure. A reconstruction needs both cameras calibrated, their "
    "relative pose measured, and the two clips aligned."
)

# Direction uncertainty from two endpoint uncertainties: the two combine in
# quadrature across a baseline, which is sqrt(2) * sigma / length for equal ends.
_ENDPOINT_PAIR = float(np.sqrt(2.0))


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector * np.nan


def _median_point(
    reconstruction: ReconstructedSequence, landmark: Landmark, frames: tuple[int, ...]
) -> tuple[NDArray[np.float64], float]:
    """Where a landmark sat over an anchor's frames, and how well it is known.

    The median across the frames, per axis, and the median of their propagated
    uncertainties -- **not** the uncertainty of the median. Reducing over N
    frames does narrow it, by roughly sqrt(N) if the errors were independent, and
    they are not: a landmark the two cameras disagree about is one they disagree
    about in the same direction for as long as the occlusion lasts. Claiming the
    sqrt(N) would understate exactly the error this phase exists to surface.
    """
    column = reconstruction.index[landmark]
    usable = [
        frame
        for frame in frames
        if 0 <= frame < len(reconstruction) and reconstruction.valid[frame, column]
    ]
    if not usable:
        return np.full(3, np.nan), float("nan")
    return (
        np.median(reconstruction.points[usable, column, :], axis=0),
        float(np.median(reconstruction.uncertainty_m[usable, column])),
    )


def _method_factor(
    reconstruction: ReconstructedSequence,
    landmarks: tuple[Landmark, ...],
    frames: tuple[int, ...],
) -> float:
    """Sine of the ray convergence angle, over the points that entered the value.

    The measured conditioning of the triangulation. At a right angle it is 1 and
    a pixel of landmark error costs a pixel's worth of depth; at 15 degrees it is
    0.26 and the same error costs four times as much. Nothing here is a chosen
    constant, which is the standing rule for a `method` factor in this engine.
    """
    values: list[float] = []
    for landmark in landmarks:
        column = reconstruction.index[landmark]
        for frame in frames:
            if 0 <= frame < len(reconstruction) and reconstruction.valid[frame, column]:
                angle = reconstruction.convergence_deg[frame, column]
                if np.isfinite(angle):
                    values.append(float(np.sin(np.radians(angle))))
    if not values:
        return 0.0
    return float(np.clip(np.mean(values), 0.0, 1.0))


def _visibility_channels(
    reconstruction: ReconstructedSequence,
) -> dict[Landmark, NDArray[np.float64]]:
    """Per-landmark visibility series, as `observation_factor` consumes them.

    The weaker of the two views at each instant, which the reconstruction already
    carries: a point is located by the disagreement between two cameras, so it is
    no better observed than the camera that saw it worse.
    """
    return {
        landmark: reconstruction.visibility[:, column]
        for landmark, column in reconstruction.index.items()
    }


def _spine_axis(
    reconstruction: ReconstructedSequence, address: Anchor
) -> tuple[NDArray[np.float64], float]:
    """The body's own rotation axis, measured at address, and its length.

    Hip midpoint to shoulder midpoint. Measured rather than assumed because the
    reconstruction supplies no vertical, and this is the axis a golf rotation is
    conventionally about anyway -- the shoulders turn around the spine, not
    around the room.
    """
    shoulders = [
        _median_point(reconstruction, landmark, address.frames)
        for landmark in (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER)
    ]
    hips = [
        _median_point(reconstruction, landmark, address.frames)
        for landmark in (Landmark.LEFT_HIP, Landmark.RIGHT_HIP)
    ]
    shoulder_mid = 0.5 * (shoulders[0][0] + shoulders[1][0])
    hip_mid = 0.5 * (hips[0][0] + hips[1][0])

    axis = shoulder_mid - hip_mid
    return _unit(axis), float(np.linalg.norm(axis))


def _line_direction(
    reconstruction: ReconstructedSequence,
    pair: tuple[Landmark, Landmark],
    frames: tuple[int, ...],
    axis: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    """A body line's direction in the plane perpendicular to the spine axis.

    Returns the unit direction, the length of the projected line, and the
    positional uncertainty of its ends. The projected length is the baseline the
    turn angle is read across, so it -- not the full shoulder width -- is what
    the angular uncertainty divides by. A shoulder line that happened to lie
    along the spine axis would project to nothing and produce an angle with no
    meaning, and this is where that becomes an enormous uncertainty rather than a
    confident number.
    """
    first, first_sigma = _median_point(reconstruction, pair[0], frames)
    second, second_sigma = _median_point(reconstruction, pair[1], frames)
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        return np.full(3, np.nan), float("nan"), float("nan")

    line = second - first
    projected = line - axis * float(line @ axis)
    length = float(np.linalg.norm(projected))
    sigma = float(np.nanmax([first_sigma, second_sigma]))
    return (projected / length if length > 0.0 else projected * np.nan), length, sigma


def _signed_turn(
    reference: NDArray[np.float64], current: NDArray[np.float64], axis: NDArray[np.float64]
) -> float:
    """Signed angle from one direction to another, about an axis, in degrees."""
    if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(current))):
        return float("nan")
    return float(
        np.degrees(
            np.arctan2(float(axis @ np.cross(reference, current)), float(reference @ current))
        )
    )


def _angular_sigma_deg(sigma_m: float, length_m: float) -> float:
    """How well a direction between two uncertain points is known, in degrees."""
    if not np.isfinite(sigma_m) or not np.isfinite(length_m) or length_m <= 0.0:
        return float("inf")
    return float(np.degrees(_ENDPOINT_PAIR * sigma_m / length_m))


def _turn_metrics(
    reconstruction: ReconstructedSequence,
    anchors: Anchors,
    address: Anchor,
    axis: NDArray[np.float64],
    config: MetricConfig,
    view: CameraView,
) -> tuple[list[Metric], list[RefusedMetric]]:
    """Shoulder turn, pelvis turn and X-factor, at every anchor that supports them."""
    produced: list[Metric] = []
    refused: list[RefusedMetric] = []
    visibility = _visibility_channels(reconstruction)

    lines = {
        MetricName.SHOULDER_TURN_3D: (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER),
        MetricName.PELVIS_TURN_3D: (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
    }
    baselines = {
        name: _line_direction(reconstruction, pair, address.frames, axis)
        for name, pair in lines.items()
    }

    # The positive sense, measured from this swing rather than assumed: whichever
    # way the shoulders had turned at the top is positive everywhere in the clip.
    sense = 1.0
    if anchors.top is not None:
        at_top, _, _ = _line_direction(
            reconstruction, lines[MetricName.SHOULDER_TURN_3D], anchors.top.frames, axis
        )
        turn = _signed_turn(baselines[MetricName.SHOULDER_TURN_3D][0], at_top, axis)
        if np.isfinite(turn) and turn < 0.0:
            sense = -1.0

    targets = [
        entry for entry in (anchors.top, anchors.impact, anchors.finish) if entry is not None
    ]

    for anchor in targets:
        turns: dict[MetricName, float] = {}
        spreads: dict[MetricName, float] = {}

        for name, pair in lines.items():
            baseline, baseline_length, baseline_sigma = baselines[name]
            current, length, sigma = _line_direction(reconstruction, pair, anchor.frames, axis)
            value = sense * _signed_turn(baseline, current, axis)
            if not np.isfinite(value):
                refused.append(
                    RefusedMetric(
                        name=name,
                        event=anchor.event,
                        reason=(
                            f"{definition(name).label} {anchor.label} needs both ends of the "
                            "line reconstructed at address and at that instant, and at least "
                            "one was not."
                        ),
                    )
                )
                continue

            spread = float(
                np.hypot(
                    _angular_sigma_deg(baseline_sigma, baseline_length),
                    _angular_sigma_deg(sigma, length),
                )
            )
            if spread > config.max_rotation_uncertainty_deg:
                refused.append(
                    RefusedMetric(
                        name=name,
                        event=anchor.event,
                        reason=(
                            f"{definition(name).label} {anchor.label} is uncertain by "
                            f"{spread:.0f} degrees, past the "
                            f"{config.max_rotation_uncertainty_deg:.0f} degree bound. The "
                            "reconstruction does not pin those landmarks down well enough "
                            "there to read a rotation across them."
                        ),
                    )
                )
                continue

            turns[name] = value
            spreads[name] = spread
            produced.append(
                build_metric(
                    name,
                    value,
                    anchor,
                    view=view,
                    source_frames=anchor.frames,
                    observation=observation_factor(visibility, pair, anchor.frames),
                    method=_method_factor(reconstruction, pair, anchor.frames),
                    uncertainty=spread,
                    methodology=(
                        f"Angle from the address direction of the reconstructed "
                        f"{definition(name).label.split()[0].lower()} line to its direction "
                        f"{anchor.label}, both projected onto the plane perpendicular to the "
                        f"address spine axis, taken about that axis. Positive is the "
                        f"direction the shoulders turned during this swing's backswing. "
                        f"Uncertainty is the two directions' own, from the propagated "
                        f"positional uncertainty over the {length:.2f} m projected baseline."
                    ),
                )
            )

        if len(turns) == 2:
            value = turns[MetricName.SHOULDER_TURN_3D] - turns[MetricName.PELVIS_TURN_3D]
            spread = float(
                np.hypot(spreads[MetricName.SHOULDER_TURN_3D], spreads[MetricName.PELVIS_TURN_3D])
            )
            landmarks = (*lines[MetricName.SHOULDER_TURN_3D], *lines[MetricName.PELVIS_TURN_3D])
            if spread > config.max_rotation_uncertainty_deg:
                refused.append(
                    RefusedMetric(
                        name=MetricName.X_FACTOR_3D,
                        event=anchor.event,
                        reason=(
                            f"X-factor {anchor.label} is a difference of two rotations and "
                            f"inherits both uncertainties: {spread:.0f} degrees, past the "
                            f"{config.max_rotation_uncertainty_deg:.0f} degree bound."
                        ),
                    )
                )
            else:
                produced.append(
                    build_metric(
                        MetricName.X_FACTOR_3D,
                        value,
                        anchor,
                        view=view,
                        source_frames=anchor.frames,
                        observation=observation_factor(visibility, landmarks, anchor.frames),
                        method=_method_factor(reconstruction, landmarks, anchor.frames),
                        uncertainty=spread,
                        methodology=(
                            "Shoulder turn minus pelvis turn, both measured about the same "
                            "address spine axis at the same instant, so the subtraction is "
                            "between two angles in one frame. Uncertainty is the two "
                            "rotations' own, combined in quadrature."
                        ),
                    )
                )

    return produced, refused


def _arm_metrics(
    reconstruction: ReconstructedSequence,
    anchors: Anchors,
    lead: LeadSide | None,
    config: MetricConfig,
    view: CameraView,
) -> tuple[list[Metric], list[RefusedMetric]]:
    """True elbow angles, at the anchors where both arms are reconstructed."""
    produced: list[Metric] = []
    refused: list[RefusedMetric] = []
    visibility = _visibility_channels(reconstruction)

    names = (MetricName.LEAD_ARM_ANGLE_3D, MetricName.TRAIL_ARM_ANGLE_3D)
    if lead is None or lead.side is None:
        refused.extend(
            RefusedMetric(
                name=name,
                reason=(
                    "Which arm leads could not be determined, so a lead and trail angle "
                    "cannot be assigned. " + (lead.methodology if lead is not None else "")
                ).strip(),
            )
            for name in names
        )
        return produced, refused

    sides = {
        MetricName.LEAD_ARM_ANGLE_3D: lead.side,
        MetricName.TRAIL_ARM_ANGLE_3D: (
            BodySide.RIGHT if lead.side is BodySide.LEFT else BodySide.LEFT
        ),
    }

    for anchor in [entry for entry in (anchors.top, anchors.impact) if entry is not None]:
        for name, side in sides.items():
            prefix = side.value.upper()
            chain = (
                Landmark[f"{prefix}_SHOULDER"],
                Landmark[f"{prefix}_ELBOW"],
                Landmark[f"{prefix}_WRIST"],
            )
            points = [_median_point(reconstruction, landmark, anchor.frames) for landmark in chain]
            if not all(np.all(np.isfinite(point)) for point, _ in points):
                refused.append(
                    RefusedMetric(
                        name=name,
                        event=anchor.event,
                        reason=(
                            f"{definition(name).label} {anchor.label} needs the shoulder, "
                            "elbow and wrist all reconstructed there, and at least one was "
                            "not."
                        ),
                    )
                )
                continue

            upper = points[0][0] - points[1][0]
            lower = points[2][0] - points[1][0]
            upper_length = float(np.linalg.norm(upper))
            lower_length = float(np.linalg.norm(lower))
            cosine = float(_unit(upper) @ _unit(lower))
            value = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

            sigma = float(np.nanmax([sigma for _, sigma in points]))
            spread = float(
                np.hypot(
                    _angular_sigma_deg(sigma, upper_length),
                    _angular_sigma_deg(sigma, lower_length),
                )
            )
            if spread > config.max_rotation_uncertainty_deg:
                refused.append(
                    RefusedMetric(
                        name=name,
                        event=anchor.event,
                        reason=(
                            f"{definition(name).label} {anchor.label} is uncertain by "
                            f"{spread:.0f} degrees over a {upper_length:.2f} m upper arm, "
                            f"past the {config.max_rotation_uncertainty_deg:.0f} degree bound."
                        ),
                    )
                )
                continue

            produced.append(
                build_metric(
                    name,
                    value,
                    anchor,
                    view=view,
                    source_frames=anchor.frames,
                    observation=observation_factor(visibility, chain, anchor.frames),
                    method=_method_factor(reconstruction, chain, anchor.frames),
                    uncertainty=spread,
                    methodology=(
                        f"Interior angle at the reconstructed {side.value} elbow between the "
                        f"{upper_length:.2f} m upper arm and the {lower_length:.2f} m forearm, "
                        "in three dimensions. 180 degrees is a straight arm. Uncertainty is "
                        "the two segments' direction uncertainties, from the propagated "
                        "positional uncertainty of the three joints."
                    ),
                )
            )

    return produced, refused


def _hand_speed_metric(
    reconstruction: ReconstructedSequence, anchors: Anchors, view: CameraView
) -> tuple[list[Metric], list[RefusedMetric]]:
    """Peak speed of the reconstructed hand midpoint over the downswing, in m/s.

    The midpoint of the two wrists rather than one of them, matching Phase 5's
    hand point, and reduced with a **maximum** rather than a median because the
    quantity is a peak: taking the median over the downswing would report the
    hands' typical speed and call it their fastest.
    """
    anchor = anchors.downswing
    if anchor is None:
        return [], [
            RefusedMetric(
                name=MetricName.PEAK_HAND_SPEED_3D,
                reason="No downswing was detected, so there is no interval to take a peak over.",
            )
        ]

    wrists = (Landmark.LEFT_WRIST, Landmark.RIGHT_WRIST)
    columns = [reconstruction.index[landmark] for landmark in wrists]
    frames = [
        frame
        for frame in anchor.frames
        if 0 <= frame < len(reconstruction)
        and any(reconstruction.valid[frame, column] for column in columns)
    ]
    if not frames:
        return [], [
            RefusedMetric(
                name=MetricName.PEAK_HAND_SPEED_3D,
                reason="Neither wrist was reconstructed at any frame of the downswing.",
            )
        ]

    speeds: list[float] = []
    for frame in frames:
        velocities = [
            reconstruction.velocity[frame, column]
            for column in columns
            if reconstruction.valid[frame, column]
            and np.all(np.isfinite(reconstruction.velocity[frame, column]))
        ]
        if velocities:
            speeds.append(float(np.linalg.norm(np.mean(velocities, axis=0))))

    if not speeds:
        return [], [
            RefusedMetric(
                name=MetricName.PEAK_HAND_SPEED_3D,
                reason=(
                    "The wrists were reconstructed over the downswing but carried no "
                    "velocity, which happens where the filter had too little support to fit "
                    "one in either view."
                ),
            )
        ]

    peak = max(speeds)
    return [
        build_metric(
            MetricName.PEAK_HAND_SPEED_3D,
            peak,
            anchor,
            view=view,
            source_frames=frames,
            observation=observation_factor(_visibility_channels(reconstruction), wrists, frames),
            method=_method_factor(reconstruction, wrists, tuple(frames)),
            methodology=(
                "Greatest speed of the reconstructed wrist midpoint over the downswing, in "
                "metres per second. The velocity comes from the two views' own fitted image "
                "velocities carried through the triangulation's least-squares condition, not "
                "from differencing the reconstructed positions -- the same rule Phase 3 "
                "states, applied one layer up. No uncertainty is quoted: propagating one "
                "would need the image velocities' own covariance, which the filter does not "
                "report."
            ),
        )
    ], []


def metrics(
    reconstruction: ReconstructedSequence | None,
    anchors: Anchors,
    lead: LeadSide | None,
    config: MetricConfig,
    view: CameraView,
) -> tuple[list[Metric], list[RefusedMetric]]:
    """Every spatial metric this clip's reconstruction supports.

    Returns refusals rather than nothing when there is no reconstruction, because
    an absent metric and one that could not honestly be computed look identical
    in a list of results -- and here the fix is a specific and actionable one:
    calibrate the pair, or align it.
    """
    names = (
        MetricName.SHOULDER_TURN_3D,
        MetricName.PELVIS_TURN_3D,
        MetricName.X_FACTOR_3D,
        MetricName.LEAD_ARM_ANGLE_3D,
        MetricName.TRAIL_ARM_ANGLE_3D,
        MetricName.PEAK_HAND_SPEED_3D,
    )

    if reconstruction is None:
        return [], [RefusedMetric(name=name, reason=_NO_RECONSTRUCTION) for name in names]

    if anchors.address is None:
        return [], [
            RefusedMetric(
                name=name,
                reason=(
                    "No address phase was detected, and every rotation here is measured "
                    "against the spine axis and the body lines as they were at address."
                ),
            )
            for name in (
                MetricName.SHOULDER_TURN_3D,
                MetricName.PELVIS_TURN_3D,
                MetricName.X_FACTOR_3D,
            )
        ] + [
            RefusedMetric(
                name=name,
                reason="No address phase was detected, so the swing's anchors are incomplete.",
            )
            for name in (
                MetricName.LEAD_ARM_ANGLE_3D,
                MetricName.TRAIL_ARM_ANGLE_3D,
                MetricName.PEAK_HAND_SPEED_3D,
            )
        ]

    axis, axis_length = _spine_axis(reconstruction, anchors.address)
    if not np.all(np.isfinite(axis)) or axis_length <= 0.0:
        return [], [
            RefusedMetric(
                name=name,
                reason=(
                    "The spine axis could not be measured at address -- the shoulders or "
                    "the hips were not reconstructed there -- and it is the axis every "
                    "rotation in this family is taken about."
                ),
            )
            for name in (
                MetricName.SHOULDER_TURN_3D,
                MetricName.PELVIS_TURN_3D,
                MetricName.X_FACTOR_3D,
            )
        ]

    turn_metrics, turn_refused = _turn_metrics(
        reconstruction, anchors, anchors.address, axis, config, view
    )
    arm_metrics, arm_refused = _arm_metrics(reconstruction, anchors, lead, config, view)
    speed_metrics, speed_refused = _hand_speed_metric(reconstruction, anchors, view)

    return (
        [*turn_metrics, *arm_metrics, *speed_metrics],
        [*turn_refused, *arm_refused, *speed_refused],
    )
