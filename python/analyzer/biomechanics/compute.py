"""Computing every metric for one clip.

The entry point to the biomechanics engine. It does three things before any
measurement happens, and each of them can stop the run:

1. **No swing, no metrics.** Metrics are anchored to swing events. On a clip
   where Phase 4 found none, there is no top of a backswing to measure a
   shoulder turn at, and inventing the instant so that something can be reported
   is exactly the failure this system is built to avoid.
2. **No scale, no metrics.** Every length here is in torso lengths. Without a
   tracked torso there is nothing to divide by, and a distance in frame widths
   would be a number about the camera rather than about the swing.
3. **No frame geometry, no metrics.** Guaranteed by the contract rather than
   checked here, since `PoseSequence` cannot be built without it -- but it is
   the reason the pose schema version moved, and worth naming where a reader
   will be looking for it.

What survives all three is then measured by the four family modules, each of
which returns what it produced *and what it refused*. Both halves reach the
caller. An absent metric and one that could not honestly be computed look
identical in a list of results, and they call for completely different
responses.
"""

from __future__ import annotations

import numpy as np

from analyzer.biomechanics import arms, posture, rotation, timing
from analyzer.biomechanics.anchors import build_anchors
from analyzer.biomechanics.body import Body, body_from
from analyzer.biomechanics.registry import REGISTRY
from analyzer.contracts.metrics import (
    MetricConfig,
    MetricName,
    MetricSet,
    RefusedMetric,
)
from analyzer.contracts.phases import SwingPhases
from analyzer.filtering.landmarks import FilteredSequence


def _refuse_all(reason: str) -> list[RefusedMetric]:
    """Every registered metric, refused for the same reason."""
    return [RefusedMetric(name=name, reason=reason) for name in REGISTRY]


def _empty(
    filtered: FilteredSequence,
    config: MetricConfig,
    torso_length: float,
    reason: str,
) -> MetricSet:
    return MetricSet(
        computed=False,
        refused=_refuse_all(reason),
        torso_length=torso_length,
        geometry=filtered.geometry,
        frames=int(filtered.t.size),
        config=config,
        warnings=[reason],
    )


def _view_warnings(body: Body, result: MetricSet) -> list[str]:
    """Facts about the recording a reader should not have to derive from the refusals."""
    warnings: list[str] = []

    off_baseline = [reference for reference in result.references if not reference.square_at_address]
    if off_baseline:
        names = " and ".join(reference.landmarks for reference in off_baseline)
        worst = max(reference.excess for reference in off_baseline)
        warnings.append(
            f"The {names} projected up to {worst:.1f} times wider mid-swing than at "
            "address, so the player was not square to the camera at address and this does "
            "not look like a face-on recording. Rotation by foreshortening has been "
            "refused rather than measured against a pose the player never held. Face-on "
            "capture is what makes shoulder and pelvis turn available."
        )

    if result.lead_side is not None and result.lead_side.side is None:
        warnings.append(
            "Which arm leads could not be determined, so the lead and trail arm angles "
            f"were refused. {result.lead_side.methodology}"
        )

    return warnings


def compute_metrics(
    filtered: FilteredSequence,
    phases: SwingPhases,
    config: MetricConfig | None = None,
) -> MetricSet:
    """Measure every biomechanics metric this clip supports."""
    resolved = config or MetricConfig()

    if not phases.detected:
        return _empty(
            filtered,
            resolved,
            float("nan"),
            "No swing was detected in this clip, so there are no events to measure metrics "
            "at. Every metric here is anchored to an instant or an interval of a swing; "
            "without one, a reported value would be a measurement of an invented moment.",
        )

    body = body_from(filtered)

    if not np.isfinite(body.torso_length) or body.torso_length < resolved.min_torso_length:
        return _empty(
            filtered,
            resolved,
            body.torso_length,
            "The subject's torso was never tracked, so there is no scale to express a "
            "distance in. Lengths here are in torso lengths precisely so that they do not "
            "depend on where the camera was put, and that requires a torso to measure.",
        )

    anchors = build_anchors(phases)
    lead = arms.infer_lead_side(body, anchors, resolved)

    posture_metrics, posture_refused = posture.metrics(body, anchors)
    rotation_metrics, rotation_refused, references = rotation.metrics(body, anchors, resolved)
    arm_metrics, arm_refused = arms.metrics(body, anchors, lead)
    timing_metrics, timing_refused = timing.metrics(body, anchors, phases)

    result = MetricSet(
        computed=True,
        metrics=[*posture_metrics, *rotation_metrics, *arm_metrics, *timing_metrics],
        refused=[*posture_refused, *rotation_refused, *arm_refused, *timing_refused],
        lead_side=lead,
        references=references,
        torso_length=body.torso_length,
        geometry=filtered.geometry,
        frames=len(body),
        config=resolved,
    )
    result.warnings = _view_warnings(body, result)
    return result


def known_metrics() -> tuple[MetricName, ...]:
    """Every metric name the engine can produce, in registry order."""
    return tuple(REGISTRY)
