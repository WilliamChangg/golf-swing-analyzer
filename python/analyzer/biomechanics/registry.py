"""The metric registry, and how a measurement becomes a `Metric`.

Every quantity this engine can produce is declared once, here, with its unit,
its basis and the words describing it. The code that computes a value never
states any of those: it hands the value to `measure_series` or `measure_value`
along with its definition, and the `Metric` is assembled from the declaration.

That indirection exists for one reason. A unit written next to the computation
drifts from the unit written in the documentation, and the failure is silent --
the number still renders, still looks reasonable, and is now labelled wrongly.
Declaring it in one place makes that impossible rather than unlikely, and gives
the test suite something exhaustive to check: every name has a definition, and
every emitted metric matches its own.

## Anchors, and confidence propagation

A metric is measured somewhere: at an instant, or over an interval. `Anchor`
carries that -- which frames, which event or phase, and **the confidence Phase 4
assigned to it**. That last part is the propagation. A spine tilt measured
perfectly at a frame that is not really the top of the backswing is not a good
measurement of the spine tilt at the top; it is a good measurement of something
else. So the event's own confidence multiplies into the metric's, and a metric
can never be more trustworthy than the instant it claims to describe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.metrics import (
    Metric,
    MetricBasis,
    MetricConfidence,
    MetricGroup,
    MetricName,
    MetricUnit,
)
from analyzer.contracts.phases import SwingEvent, SwingPhase
from analyzer.contracts.pose import Landmark


@dataclass(frozen=True)
class MetricDefinition:
    """What a metric is, independent of any particular clip."""

    name: MetricName
    group: MetricGroup
    label: str
    unit: MetricUnit
    basis: MetricBasis
    summary: str = ""
    anchored: bool = True
    """Whether the display label needs the anchor appended.

    True for a quantity that could be measured at several instants, where
    "spine tilt" alone is ambiguous. False for one whose own name already fixes
    the interval -- "backswing duration over the backswing" tells a reader
    nothing they did not have.
    """


@dataclass(frozen=True)
class Anchor:
    """Where a metric was measured, and what that instant is worth.

    `frames` is every frame the value may be read from -- one frame for a metric
    taken at an event, the whole interval for one taken over a phase. Reducing
    over an interval is what makes an address measurement robust: the subject is
    still there, so the median over the phase is a better estimate of the pose
    than any single frame of it, and the frames are all reported so the reduction
    can be checked.
    """

    label: str
    frames: tuple[int, ...]
    confidence: float
    event: SwingEvent | None = None
    phase: SwingPhase | None = None


# Every quantity the engine knows how to measure. Ordered within each group the
# way a reader works through a swing: the body first, then what it did, then
# when.
_DEFINITIONS: tuple[MetricDefinition, ...] = (
    # --- posture ---------------------------------------------------------
    MetricDefinition(
        name=MetricName.SPINE_TILT,
        group=MetricGroup.POSTURE,
        label="Spine tilt",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.PROJECTED_ANGLE,
        summary=(
            "Lean of the hip-midpoint-to-shoulder-midpoint line away from vertical, "
            "positive towards the right of the frame. Which anatomical direction "
            "that is depends on where the camera stood, which this phase does not "
            "know: face-on it is side bend, down the line it is forward bend."
        ),
    ),
    MetricDefinition(
        name=MetricName.LEFT_KNEE_FLEX,
        group=MetricGroup.POSTURE,
        label="Left knee flex",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.PROJECTED_ANGLE,
        summary="180 degrees minus the projected hip-knee-ankle angle. Zero is a straight leg.",
    ),
    MetricDefinition(
        name=MetricName.RIGHT_KNEE_FLEX,
        group=MetricGroup.POSTURE,
        label="Right knee flex",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.PROJECTED_ANGLE,
        summary="180 degrees minus the projected hip-knee-ankle angle. Zero is a straight leg.",
    ),
    MetricDefinition(
        name=MetricName.HIP_SWAY,
        group=MetricGroup.POSTURE,
        label="Hip sway",
        unit=MetricUnit.TORSO_LENGTHS,
        basis=MetricBasis.IMAGE_PLANE,
        summary=(
            "Sideways travel of the hip midpoint from where it sat at address, "
            "positive towards the right of the frame."
        ),
    ),
    MetricDefinition(
        name=MetricName.HEAD_SWAY,
        group=MetricGroup.POSTURE,
        label="Head sway",
        unit=MetricUnit.TORSO_LENGTHS,
        basis=MetricBasis.IMAGE_PLANE,
        summary="Sideways travel of the nose from its address position.",
    ),
    MetricDefinition(
        name=MetricName.HEAD_LIFT,
        group=MetricGroup.POSTURE,
        label="Head lift",
        unit=MetricUnit.TORSO_LENGTHS,
        basis=MetricBasis.IMAGE_PLANE,
        summary="Vertical travel of the nose from its address position, positive upwards.",
    ),
    # --- rotation --------------------------------------------------------
    MetricDefinition(
        name=MetricName.SHOULDER_TURN,
        group=MetricGroup.ROTATION,
        label="Shoulder turn",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.FORESHORTENED_ANGLE,
        summary=(
            "Rotation of the shoulder line away from its projected width at "
            "address, inferred from how much it shortened. A magnitude: the "
            "method cannot tell a turn from its mirror image."
        ),
    ),
    MetricDefinition(
        name=MetricName.PELVIS_TURN,
        group=MetricGroup.ROTATION,
        label="Pelvis turn",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.FORESHORTENED_ANGLE,
        summary="The same measurement made on the hip line.",
    ),
    MetricDefinition(
        name=MetricName.X_FACTOR,
        group=MetricGroup.ROTATION,
        label="X-factor",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.FORESHORTENED_ANGLE,
        summary=(
            "Shoulder turn minus pelvis turn: how much the upper body has "
            "outrun the lower. An approximation, and signed only in the sense "
            "that both terms are magnitudes -- it says how far apart they are, "
            "not which way either went."
        ),
    ),
    MetricDefinition(
        name=MetricName.SHOULDER_TILT,
        group=MetricGroup.ROTATION,
        label="Shoulder tilt",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.PROJECTED_ANGLE,
        summary=(
            "Tilt of the shoulder line away from level, measured left to right "
            "across the frame because the line has an orientation rather than a "
            "direction. Positive means the shoulder further right in the frame "
            "is the higher one -- an image-frame statement, not an anatomical "
            "one: which shoulder that is depends on which way the player faces."
        ),
    ),
    MetricDefinition(
        name=MetricName.PELVIS_TILT,
        group=MetricGroup.ROTATION,
        label="Pelvis tilt",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.PROJECTED_ANGLE,
        summary="The same measurement made on the hip line.",
    ),
    # --- hands and arms --------------------------------------------------
    MetricDefinition(
        name=MetricName.HAND_PATH_LENGTH,
        group=MetricGroup.ARMS,
        label="Hand path length",
        unit=MetricUnit.TORSO_LENGTHS,
        basis=MetricBasis.IMAGE_PLANE,
        summary=(
            "Distance the hands travelled along their path, summed frame to "
            "frame rather than measured end to end, so an arc is not reported as "
            "its chord."
        ),
    ),
    MetricDefinition(
        name=MetricName.PEAK_HAND_SPEED,
        group=MetricGroup.ARMS,
        label="Peak hand speed",
        unit=MetricUnit.TORSO_LENGTHS_PER_S,
        basis=MetricBasis.IMAGE_PLANE,
        summary=(
            "Fastest the tracked hand point moved in the image plane. In torso "
            "lengths per second rather than frame widths per second, so it is "
            "comparable between recordings made at different distances."
        ),
    ),
    MetricDefinition(
        name=MetricName.LEAD_ARM_ANGLE,
        group=MetricGroup.ARMS,
        label="Lead arm angle",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.PROJECTED_ANGLE,
        summary=(
            "Projected shoulder-elbow-wrist angle on the leading side. "
            "180 degrees is a straight arm."
        ),
    ),
    MetricDefinition(
        name=MetricName.TRAIL_ARM_ANGLE,
        group=MetricGroup.ARMS,
        label="Trail arm angle",
        unit=MetricUnit.DEGREES,
        basis=MetricBasis.PROJECTED_ANGLE,
        summary="The same measurement made on the trailing side.",
    ),
    # --- timing ----------------------------------------------------------
    MetricDefinition(
        name=MetricName.BACKSWING_DURATION,
        group=MetricGroup.TIMING,
        label="Backswing duration",
        anchored=False,
        unit=MetricUnit.SECONDS,
        basis=MetricBasis.TEMPORAL,
        summary="Takeaway to top, from the clip's presentation timestamps.",
    ),
    MetricDefinition(
        name=MetricName.DOWNSWING_DURATION,
        group=MetricGroup.TIMING,
        label="Downswing duration",
        anchored=False,
        unit=MetricUnit.SECONDS,
        basis=MetricBasis.TEMPORAL,
        summary="Top to the estimated impact.",
    ),
    MetricDefinition(
        name=MetricName.FOLLOW_THROUGH_DURATION,
        group=MetricGroup.TIMING,
        label="Follow-through duration",
        anchored=False,
        unit=MetricUnit.SECONDS,
        basis=MetricBasis.TEMPORAL,
        summary="Impact to the finish.",
    ),
    MetricDefinition(
        name=MetricName.TAKEAWAY_TO_IMPACT,
        group=MetricGroup.TIMING,
        label="Takeaway to impact",
        anchored=False,
        unit=MetricUnit.SECONDS,
        basis=MetricBasis.TEMPORAL,
        summary="The whole swing, from the hands first moving to the estimated impact.",
    ),
    MetricDefinition(
        name=MetricName.TEMPO_RATIO,
        group=MetricGroup.TIMING,
        label="Tempo ratio",
        anchored=False,
        unit=MetricUnit.RATIO,
        basis=MetricBasis.TEMPORAL,
        summary=(
            "Backswing duration divided by downswing duration. Unitless, so it "
            "is the one quantity here that a frame rate, a camera position and a "
            "subject's size all leave completely alone."
        ),
    ),
)

REGISTRY: Mapping[MetricName, MetricDefinition] = {
    definition.name: definition for definition in _DEFINITIONS
}


def definition(name: MetricName) -> MetricDefinition:
    """The declaration for a metric. Raises if the name has none."""
    try:
        return REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - a missing entry fails the suite first
        raise KeyError(f"No metric definition registered for {name!r}.") from exc


def visibility_factor(channels: Sequence[NDArray[np.float64]], frames: Sequence[int]) -> float:
    """Mean reported visibility across some per-frame channels, over some frames.

    The estimator's own confidence that a landmark was unoccluded, averaged over
    exactly the (channel, frame) pairs that entered the value -- not over the
    clip, and not over the landmark set as a whole. An angle at the top computed
    from a wrist the model could barely see should say so, and a clip-wide
    average is the one number guaranteed not to.

    Takes channels rather than landmarks because not every quantity is read from
    a named landmark: the hand point Phase 4 chose may be one wrist, the other,
    or the midpoint of both, and its visibility is its own channel.
    """
    values: list[float] = []
    for channel in channels:
        for frame in frames:
            if 0 <= frame < channel.size and np.isfinite(channel[frame]):
                values.append(float(channel[frame]))
    if not values:
        return 0.0
    return float(np.clip(np.mean(values), 0.0, 1.0))


def observation_factor(
    visibility: Mapping[Landmark, NDArray[np.float64]],
    landmarks: Sequence[Landmark],
    frames: Sequence[int],
) -> float:
    """`visibility_factor` for a metric read from a set of named landmarks."""
    channels = [
        channel
        for channel in (visibility.get(landmark) for landmark in landmarks)
        if channel is not None
    ]
    return visibility_factor(channels, frames)


def duration_method_factor(duration_s: float, interval_s: float) -> float:
    """How finely the frame rate divides a duration.

    An event is located to within a frame, so a duration between two of them
    carries about one frame interval of uncertainty. The factor is the fraction
    of the duration that is *not* that uncertainty, which falls to zero as a
    phase shrinks towards a single frame -- and a phase one frame long is a
    statement about the capture rate rather than about the swing.
    """
    if not np.isfinite(duration_s) or duration_s <= 0.0 or interval_s <= 0.0:
        return 0.0
    return float(np.clip(1.0 - interval_s / duration_s, 0.0, 1.0))


def _reduce(series: NDArray[np.float64], frames: Sequence[int]) -> tuple[float, list[int]]:
    """Median of a per-frame series over the frames that carry a value.

    Returns the value and the frames it was taken from, which are reported as
    the metric's `source_frames`. A single frame reduces to itself; an interval
    reduces to its median, which is what makes a measurement taken across the
    address phase robust to one bad frame in it.
    """
    usable = [frame for frame in frames if 0 <= frame < series.size and np.isfinite(series[frame])]
    if not usable:
        return float("nan"), []
    return float(np.median(series[usable])), usable


def build_metric(
    name: MetricName,
    value: float,
    anchor: Anchor,
    *,
    source_frames: Sequence[int],
    observation: float,
    method: float,
    methodology: str,
) -> Metric:
    """Assemble a `Metric` from its declaration and one measurement.

    The unit, basis, group and label come from the registry; only the value, the
    anchor and the confidence factors come from the caller. `overall` is the
    product of the three factors, as in Phase 4 and for the same reason: a metric
    needs all of them, and a product says so where an average would let a strong
    factor cover for a fatal one.
    """
    entry = definition(name)
    factors = tuple(float(np.clip(f, 0.0, 1.0)) for f in (observation, anchor.confidence, method))

    return Metric(
        name=entry.name,
        group=entry.group,
        label=f"{entry.label} {anchor.label}" if entry.anchored else entry.label,
        value=value,
        unit=entry.unit,
        basis=entry.basis,
        event=anchor.event,
        phase=anchor.phase,
        source_frames=list(source_frames),
        confidence=MetricConfidence(
            overall=float(np.clip(factors[0] * factors[1] * factors[2], 0.0, 1.0)),
            observation=factors[0],
            anchor=factors[1],
            method=factors[2],
        ),
        methodology=methodology,
    )


def in_plane_fraction(span: NDArray[np.float64]) -> NDArray[np.float64]:
    """How much of a segment lies in the image plane, per frame, on [0, 1].

    A rigid segment projects to its full length only when it is perpendicular to
    the optical axis, so the fraction of its greatest observed projection is a
    measured lower bound on how squarely it is being seen. It is the right
    confidence factor for a projected angle twice over: an angle read across a
    segment seen nearly end-on is both further from the true one and noisier,
    because the same landmark error subtends a larger angle across a shorter
    baseline.

    A lower bound, because the greatest projection observed is only the
    squarest view that happened to occur in this clip.
    """
    if not np.any(np.isfinite(span)):
        return np.zeros_like(span)
    peak = float(np.nanmax(span))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(span)
    return np.clip(span / peak, 0.0, 1.0)


def measure_series(
    name: MetricName,
    series: NDArray[np.float64],
    anchor: Anchor,
    *,
    visibility: Mapping[Landmark, NDArray[np.float64]],
    landmarks: Sequence[Landmark],
    method: float | NDArray[np.float64],
    methodology: str,
) -> Metric | None:
    """Reduce a per-frame series at an anchor and build the metric, or None.

    None means the series carried no value at any of the anchor's frames, which
    is the refusal this layer is built around: the landmarks were gated out, or
    the filter had too little support there, and there is nothing to report.

    `method` may be a per-frame series of its own, in which case it is reduced
    over the same frames as the value. A confidence factor that varies through
    the clip -- how squarely a segment is being seen, for instance -- must be
    read where the measurement was taken, not averaged over the whole clip.
    """
    value, frames = _reduce(series, anchor.frames)
    if not frames:
        return None

    if isinstance(method, np.ndarray):
        reduced, _ = _reduce(method, tuple(frames))
        method_factor = 0.0 if np.isnan(reduced) else reduced
    else:
        method_factor = float(method)

    return build_metric(
        name,
        value,
        anchor,
        source_frames=frames,
        observation=observation_factor(visibility, landmarks, frames),
        method=method_factor,
        methodology=methodology,
    )
