"""What has to be true before two numbers may be called different.

Phase 13 asked what a measurement has to clear before it may be placed on one
side of a published band. This asks the harder version: what two measurements of
the same quantity, made from two recordings, have to clear before the difference
between them is a fact about the swings rather than about the recordings.

Four things can differ between two clips without either swing differing, and each
is a gate here:

**The camera position.** A projected angle is a fact about where the camera stood
as much as about the body. The view detector's three labels are far too coarse to
settle this -- a tripod can move twenty degrees round a player and stay inside
`face_on` -- so the address shoulder span is used instead, which is the same
number the view is decided from and the same number every foreshortening rotation
is referred to. `camera_term_deg` then computes what a residual azimuth
difference alone would do to a reported rotation, exactly, and adds it to the
bracket.

**The lens.** Phase 8 measured undistortion moving peak hand speed by 3.0% and a
rotation by up to 1.2 degrees on the reference clip. Comparing an undistorted
clip against a raw one puts that straight into the difference, so a mismatch
refuses rather than warns.

**The clock.** A slow-motion factor is supplied rather than measured, so a
duration from such a clip is a measurement multiplied by a guess and a difference
of two of them is a difference of two guesses. A ratio of two durations from one
clip divides the factor out and survives -- the same gate, and the same argument,
as `FindingRefusal.SUPPLIED_TIMEBASE`.

**The resolution.** Each clip's own bracket, from `coaching.bracket`, so that the
two layers cannot come to different conclusions about what one recording can
resolve. The two are **added** rather than combined in quadrature, which is where
this parts company with `coaching.combined_bracket`: that function bounds the
difference of two measurements sharing a systematic error, which largely cancels,
and two separate recordings share nothing. A sum of bounds is a bound.
"""

from __future__ import annotations

import math

import numpy as np

from analyzer.coaching.bracket import bracket_for
from analyzer.contracts.comparison import (
    BracketTerm,
    CameraAgreement,
    ComparisonConfig,
    DifferenceRefusal,
    Direction,
    MetricDifference,
    RefusedDifference,
)
from analyzer.contracts.metrics import (
    CameraView,
    Metric,
    MetricBasis,
    MetricName,
    MetricSet,
    MetricUnit,
)
from analyzer.contracts.phases import SwingEvent

_PROJECTED = frozenset(
    {
        MetricBasis.IMAGE_PLANE,
        MetricBasis.PROJECTED_ANGLE,
        MetricBasis.FORESHORTENED_ANGLE,
    }
)
"""The bases whose value depends on where the camera stood.

`TEMPORAL` is absent because a stopwatch does not care, and `SPATIAL` is absent
because a triangulated distance is the same distance from anywhere -- that is the
property Phase 9 exists to buy, and the reason its metrics carry one anatomical
reading instead of one per view.
"""


def camera_agreement(
    reference: MetricSet, target: MetricSet, config: ComparisonConfig
) -> CameraAgreement:
    """Whether the two clips were filmed from the same place.

    Both numbers come from `ViewEstimate`, which measures them at address: the
    projected shoulder span in torso lengths, and its ratio to the widest that
    line was ever seen in the clip. The second is the cosine of how far off
    broadside the shoulders were, so its arccosine estimates the camera's azimuth
    -- unsigned, because foreshortening is identical from either side of
    broadside and nothing here separates them.
    """
    reference_view = reference.view
    target_view = target.view
    if reference_view is None or target_view is None:
        return CameraAgreement(
            consistent=False,
            methodology=(
                "At least one clip produced no view estimate, so nothing here says "
                "where its camera stood and no projected quantity may be compared. "
                "A clip with no detected swing has no address phase to measure the "
                "shoulder span over, which is the usual way this happens."
            ),
        )

    spans = (reference_view.shoulder_span_ratio, target_view.shoulder_span_ratio)
    mean = (spans[0] + spans[1]) / 2.0
    disagreement = abs(spans[0] - spans[1]) / mean if mean > 0.0 else None

    reference_azimuth = _azimuth_deg(reference_view.openness)
    target_azimuth = _azimuth_deg(target_view.openness)
    # The **sum**, not the difference. Both azimuths are unsigned, so two cameras
    # each five degrees off broadside may have stood in one place or ten degrees
    # apart, and a bound has to assume the second.
    separation = (
        reference_azimuth + target_azimuth
        if reference_azimuth is not None and target_azimuth is not None
        else None
    )

    same_class = reference_view.view is target_view.view and reference_view.view is not (
        CameraView.UNKNOWN
    )
    consistent = (
        same_class and disagreement is not None and disagreement <= config.max_span_disagreement
    )

    return CameraAgreement(
        reference_span=spans[0],
        target_span=spans[1],
        span_disagreement=disagreement,
        reference_openness=reference_view.openness,
        target_openness=target_view.openness,
        reference_azimuth_deg=reference_azimuth,
        target_azimuth_deg=target_azimuth,
        azimuth_separation_deg=separation,
        consistent=consistent,
        methodology=(
            "Projected shoulder span at address, in torso lengths, from each "
            "clip's own view estimate; the azimuth is the arccosine of that span "
            "divided by the widest the line was seen in the same clip. The verdict "
            "is on the spans, which are measured; the azimuths are reported and "
            "used to widen a bracket, because the arccosine is flat near broadside "
            "and would turn a percent of landmark noise into eight degrees. It "
            "assumes both recordings show the same body, since a broader player "
            "projects a broader line from the same place, and it depends on each "
            "clip containing a frame where the shoulders come square."
        ),
    )


def _azimuth_deg(openness: float) -> float | None:
    """How far off broadside a camera stood, from the openness it produced.

    None outside (0, 1]. Above 1 the address view was wider than the widest frame
    in the clip, which is landmark noise rather than geometry; at or below 0 the
    line never projected at all. Both are reported as unknown rather than clamped
    to an angle, because a clamp would turn a measurement failure into a confident
    zero.
    """
    if not math.isfinite(openness) or openness <= 0.0 or openness > 1.0:
        return None
    return float(np.degrees(np.arccos(openness)))


def camera_term_deg(
    value_deg: float, reference_azimuth_deg: float | None, target_azimuth_deg: float | None
) -> float:
    """How much of a foreshortened rotation's difference the cameras alone explain.

    A rotation measured by foreshortening compares a body line's projected span
    against its span at address. From a camera `a` degrees off broadside, a line
    that has truly turned `t` projects `cos(a + t)` against an address span of
    `cos(a)`, so the reported angle is `arccos(cos(a + t) / cos(a))` -- which is
    not `t` unless `a` is zero, and departs from it quickly.

    Worked at the reference clips' own numbers: a fifty-degree turn seen from ten
    degrees off broadside reads as fifty-six, and from twenty degrees as
    sixty-five. **Two cameras twenty degrees apart report a fifteen-degree
    difference in shoulder turn on a swing that did not change**, which is larger
    than most differences anybody would want to read from this system.

    So the term is computed rather than assumed away: the true turn is recovered
    from the reference clip's own azimuth, re-read from the target's, and the
    difference returned. Both azimuths are unsigned and both branches of the
    recovery are real, so the worst case over the sign assignments is taken --
    a bracket is a bound, and nothing here knows which side of the player either
    camera stood on.

    **Two equal azimuths therefore do not give zero**, and that is not a defect
    to be tidied away. Two cameras each five degrees off broadside may have stood
    in the same place or ten degrees apart, foreshortening cannot tell those
    apart, and a term that assumed the first would be assuming the answer. Only a
    pair whose openness puts both cameras square returns zero.
    """
    if reference_azimuth_deg is None or target_azimuth_deg is None:
        return 0.0
    if not math.isfinite(value_deg) or value_deg < 0.0:
        return 0.0

    theta = math.radians(min(value_deg, 90.0))
    worst = 0.0
    for sign_a in (1.0, -1.0):
        alpha = sign_a * math.radians(reference_azimuth_deg)
        recovered = math.cos(alpha) * math.cos(theta)
        if abs(recovered) > 1.0:
            continue
        angle = math.acos(recovered)
        for branch in (angle, -angle):
            turn = branch - alpha
            for sign_b in (1.0, -1.0):
                beta = sign_b * math.radians(target_azimuth_deg)
                if abs(math.cos(beta)) < 1e-9:
                    continue
                projected = abs(math.cos(beta + turn)) / abs(math.cos(beta))
                if projected > 1.0:
                    continue
                worst = max(worst, abs(math.degrees(math.acos(projected)) - math.degrees(theta)))
    return min(worst, 90.0)


def compare_metrics(
    reference: MetricSet,
    target: MetricSet,
    camera: CameraAgreement,
    config: ComparisonConfig,
    *,
    reference_interval_s: float | None,
    target_interval_s: float | None,
) -> tuple[list[MetricDifference], list[RefusedDifference], int]:
    """Every quantity either clip measured, compared or refused by name.

    Iterates the quantities at least one clip **produced**, not the whole
    registry. A metric neither recording could measure was already refused one
    layer down with a reason specific to that recording -- "a down-the-line camera
    does not contain this rotation" -- and restating it here in the comparison
    layer's voice would give a reader the same fact twice with the less specific
    wording second.
    """
    differences: list[MetricDifference] = []
    refused: list[RefusedDifference] = []

    for name, event in _quantities(reference, target):
        left = reference.get(name, event)
        right = target.get(name, event)
        outcome = _compare_one(
            left,
            right,
            reference,
            target,
            camera,
            config,
            reference_interval_s=reference_interval_s,
            target_interval_s=target_interval_s,
        )
        if isinstance(outcome, MetricDifference):
            differences.append(outcome)
        else:
            refused.append(outcome)

    return differences, refused, len(differences) + len(refused)


def _quantities(
    reference: MetricSet, target: MetricSet
) -> list[tuple[MetricName, SwingEvent | None]]:
    """The name/anchor pairs either clip produced, in the reference's order first."""
    seen: list[tuple[MetricName, SwingEvent | None]] = []
    for metrics in (reference.metrics, target.metrics):
        for metric in metrics:
            key = (metric.name, metric.event)
            if key not in seen:
                seen.append(key)
    return seen


def _compare_one(  # one return per gate; collapsing them would hide the order
    left: Metric | None,
    right: Metric | None,
    reference: MetricSet,
    target: MetricSet,
    camera: CameraAgreement,
    config: ComparisonConfig,
    *,
    reference_interval_s: float | None,
    target_interval_s: float | None,
) -> MetricDifference | RefusedDifference:
    """One quantity, through the gates in the order that makes a refusal useful.

    The order is load-bearing, the way Phase 13 found `BASIS_NOT_PERMITTED` had to
    sit above `NO_METRIC`. A reader told "this quantity is missing from one clip"
    goes and re-films it; if the two cameras were in different places, the
    re-filmed clip is refused for the camera instead, and they have shot a clip to
    learn something the first answer could have told them. So every gate that is a
    fact about the *pair* runs before every gate that is a fact about one clip.
    """
    present = left or right
    assert present is not None  # noqa: S101 - callers only pass produced quantities
    label, name, event = present.label, present.name, present.event
    projected = present.basis in _PROJECTED

    if projected and not _same_view(reference, target):
        return RefusedDifference(
            name=name,
            label=label,
            event=event,
            refusal=DifferenceRefusal.VIEW_MISMATCH,
            reason=(
                f"{label} is measured in the image plane, and these clips were "
                f"filmed from different positions ("
                f"{_view_name(reference)} and {_view_name(target)}). The same "
                "computation means different things from the two, so the numbers "
                "are not two measurements of one quantity."
            ),
        )

    if projected and not camera.consistent:
        return RefusedDifference(
            name=name,
            label=label,
            event=event,
            refusal=DifferenceRefusal.CAMERA_MOVED,
            reason=(
                f"{label} depends on where the camera stood, and these two "
                "recordings disagree about that: " + _camera_evidence(camera)
            ),
        )

    if projected and reference.calibration is not target.calibration:
        return RefusedDifference(
            name=name,
            label=label,
            event=event,
            refusal=DifferenceRefusal.CALIBRATION_MISMATCH,
            reason=(
                f"One clip's landmarks were corrected for the lens and the other's "
                f"were not ({reference.calibration.value} against "
                f"{target.calibration.value}). That correction moves an image-plane "
                "measurement on its own, so the difference would be partly the two "
                "lenses."
            ),
        )

    if present.unit is MetricUnit.SECONDS and not _one_timebase(reference, target):
        return RefusedDifference(
            name=name,
            label=label,
            event=event,
            refusal=DifferenceRefusal.SUPPLIED_TIMEBASE,
            reason=(
                f"{label} is a duration, and a slow-motion factor was supplied "
                f"rather than measured ({reference.slow_motion_factor:g}x and "
                f"{target.slow_motion_factor:g}x). Each duration is a measured "
                "number multiplied by a guess, so their difference is a difference "
                "of guesses. A ratio of two durations from one clip divides the "
                "factor back out and is compared."
            ),
        )

    if left is None or right is None:
        missing = "target" if right is None else "reference"
        produced = target if right is None else reference
        return RefusedDifference(
            name=name,
            label=label,
            event=event,
            refusal=DifferenceRefusal.MISSING,
            reason=(
                f"{label} was measured in one clip and not the other; the "
                f"{missing} clip did not produce it. " + _why_missing(produced, name, event)
            ),
        )

    weakest = min(left.confidence.overall, right.confidence.overall)
    if weakest < config.min_confidence:
        return RefusedDifference(
            name=name,
            label=label,
            event=event,
            refusal=DifferenceRefusal.LOW_CONFIDENCE,
            reason=(
                f"The weaker of the two measurements scores {weakest:.2f}, below "
                f"{config.min_confidence:.2f}. A difference of two numbers is "
                "supported by less than either of them, not more."
            ),
            reference_value=left.value,
            target_value=right.value,
        )

    left_bracket, left_reason = bracket_for(left, reference, reference_interval_s)
    right_bracket, right_reason = bracket_for(right, target, target_interval_s)
    if left_bracket is None or right_bracket is None:
        which = "reference" if left_bracket is None else "target"
        return RefusedDifference(
            name=name,
            label=label,
            event=event,
            refusal=DifferenceRefusal.NO_BRACKET,
            reason=(
                f"No distance this difference could be required to clear. On the "
                f"{which} clip: " + (left_reason if left_bracket is None else right_reason)
            ),
            reference_value=left.value,
            target_value=right.value,
        )

    terms = [
        BracketTerm(source="reference clip", value=left_bracket, reason=left_reason),
        BracketTerm(source="target clip", value=right_bracket, reason=right_reason),
    ]
    if present.basis is MetricBasis.FORESHORTENED_ANGLE:
        camera_term = camera_term_deg(
            max(abs(left.value), abs(right.value)),
            camera.reference_azimuth_deg,
            camera.target_azimuth_deg,
        )
        # A millidegree, because the recovery and the re-reading are an arccosine
        # run forwards and backwards and leave floating dust behind when the two
        # cameras really are square. A term nobody can see is not a term, and one
        # listed at 7e-15 degrees would make the itemised bracket unreadable.
        if camera_term > 1e-3:
            terms.append(
                BracketTerm(
                    source="camera azimuth",
                    value=camera_term,
                    reason=(
                        "What this rotation would read as from the other clip's "
                        "camera if the body had not moved at all. The two cameras "
                        "stood at most "
                        f"{camera.azimuth_separation_deg:.1f} degrees apart, and a "
                        "foreshortened rotation is measured against a baseline "
                        "that moves with them."
                    ),
                )
            )

    bracket = sum(term.value for term in terms)
    difference = right.value - left.value
    margin = abs(difference) - bracket

    if not (margin > 0.0):
        return RefusedDifference(
            name=name,
            label=label,
            event=event,
            refusal=DifferenceRefusal.UNRESOLVED,
            reason=(
                f"The two values differ by {abs(difference):.3g} "
                f"{present.unit.value}, and these recordings cannot resolve a "
                f"difference below {bracket:.3g}. Not a finding that the swings "
                "agree: a finding that this pair of recordings cannot tell them "
                "apart. " + _what_would_help(terms)
            ),
            reference_value=left.value,
            target_value=right.value,
            bracket=bracket,
        )

    return MetricDifference(
        name=name,
        label=label,
        group=present.group,
        unit=present.unit,
        basis=present.basis,
        event=event,
        reference_value=left.value,
        target_value=right.value,
        difference=difference,
        direction=Direction.HIGHER if difference > 0.0 else Direction.LOWER,
        bracket=bracket,
        terms=terms,
        margin=margin,
        confidence=weakest,
        reference_frames=list(left.source_frames),
        target_frames=list(right.source_frames),
        interpretation=left.interpretation,
        methodology=(
            f"{left.methodology} Both clips were measured that way; the difference "
            "is the target's value minus the reference's, and it had to clear the "
            "sum of what each recording can resolve."
        ),
    )


def _same_view(reference: MetricSet, target: MetricSet) -> bool:
    """Whether both clips were filmed from the same, identified, camera position.

    Two `unknown` views are not a match. An oblique camera is a real thing to have
    recorded and it supports some measurements, but two oblique cameras are two
    unrelated obliques, and treating the shared label as agreement would compare
    them as though it meant something.
    """
    if reference.view is None or target.view is None:
        return False
    return reference.view.view is target.view.view and reference.view.view is not CameraView.UNKNOWN


def _view_name(metrics: MetricSet) -> str:
    return metrics.view.view.value if metrics.view is not None else "unmeasured"


def _one_timebase(reference: MetricSet, target: MetricSet) -> bool:
    """Whether both clips play at real time, so a duration in seconds is measured."""
    return reference.slow_motion_factor == 1.0 and target.slow_motion_factor == 1.0


def _camera_evidence(camera: CameraAgreement) -> str:
    """The numbers behind a `CAMERA_MOVED` refusal, in words."""
    if camera.reference_span is None or camera.target_span is None:
        return camera.methodology
    parts = [
        f"the shoulders span {camera.reference_span:.2f} torso lengths at address "
        f"in one and {camera.target_span:.2f} in the other"
    ]
    if camera.azimuth_separation_deg is not None:
        parts.append(
            f"which puts them up to {camera.azimuth_separation_deg:.1f} degrees "
            "apart round the player"
        )
    return ", ".join(parts) + ". Film both from one position, or compare only the timings."


def _why_missing(produced: MetricSet, name: MetricName, event: SwingEvent | None) -> str:
    """The other clip's own reason for not producing this quantity, where it gave one."""
    entry = next(
        (item for item in produced.refused if item.name is name and item.event is event), None
    )
    if entry is not None:
        return entry.reason
    return (
        "That clip's own metric set does not say why, which means the quantity was "
        "not asked for there rather than refused."
    )


def _what_would_help(terms: list[BracketTerm]) -> str:
    """Which term dominates the bracket, since that is what a reader can act on."""
    largest = max(terms, key=lambda term: term.value)
    if largest.source == "camera azimuth":
        return "Most of that comes from the cameras not having stood in the same place."
    return f"Most of that comes from the {largest.source}."
