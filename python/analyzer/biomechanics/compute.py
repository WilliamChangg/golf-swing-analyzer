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

from analyzer.biomechanics import arms, posture, rotation, spatial, timing
from analyzer.biomechanics.anchors import build_anchors
from analyzer.biomechanics.body import body_from
from analyzer.biomechanics.registry import REGISTRY, definition
from analyzer.biomechanics.view import infer_view
from analyzer.contracts.calibration import CalibrationStatus
from analyzer.contracts.metrics import (
    CameraView,
    Metric,
    MetricConfig,
    MetricName,
    MetricSet,
    RefusedMetric,
)
from analyzer.contracts.phases import SwingPhases
from analyzer.filtering.landmarks import FilteredSequence
from analyzer.reconstruction import ReconstructedSequence


def _refuse_all(reason: str) -> list[RefusedMetric]:
    """Every registered metric, refused for the same reason."""
    return [RefusedMetric(name=name, reason=reason) for name in REGISTRY]


def _empty(
    filtered: FilteredSequence,
    config: MetricConfig,
    torso_length: float,
    reason: str,
    calibration: CalibrationStatus,
) -> MetricSet:
    return MetricSet(
        computed=False,
        refused=_refuse_all(reason),
        torso_length=torso_length,
        geometry=filtered.geometry,
        frames=int(filtered.t.size),
        calibration=calibration,
        slow_motion_factor=filtered.slow_motion_factor,
        config=config,
        warnings=[reason],
    )


def _apply_view_gate(
    produced: list[Metric], refused: list[RefusedMetric], view: CameraView
) -> tuple[list[Metric], list[RefusedMetric]]:
    """Let the camera position have the last word on what this clip can report.

    Applied centrally rather than inside each family module, so that a family
    added later cannot forget it, and applied to the *refusals* as well as the
    values. A quantity its view does not contain gets exactly one refusal saying
    so, replacing whatever more specific complaint the measurement itself
    raised: on a down-the-line clip the shoulder line is also too short to read
    an angle across, which is true, a consequence of the same fact, and not the
    thing worth telling someone holding a camera.

    `UNKNOWN` blocks nothing. An oblique camera does support some of these, and
    which ones is better decided by each metric's own conditions -- the
    foreshortening baseline check, the shoulder span the lead side needs -- than
    by a label the clip could not establish.
    """
    blocked = {name for name, entry in REGISTRY.items() if entry.refuses(view)}
    if not blocked:
        return produced, refused

    named = view.value.replace("_", "-")
    survivors = [entry for entry in refused if entry.name not in blocked]
    survivors.extend(
        RefusedMetric(
            name=name,
            reason=(
                f"{definition(name).label} cannot be measured from a {named} "
                f"recording. {definition(name).summary}"
            ),
        )
        for name in sorted(blocked, key=lambda entry: entry.value)
    )
    return [metric for metric in produced if metric.name not in blocked], survivors


def _apply_calibration_gate(
    produced: list[Metric], refused: list[RefusedMetric], status: CalibrationStatus
) -> tuple[list[Metric], list[RefusedMetric]]:
    """Let what is known about the camera's geometry have the last word too.

    The second of the two central gates, and the mirror of `_apply_view_gate`:
    one asks whether the recording *contains* a quantity, the other whether this
    build is entitled to *claim* it. They are separate because they fail
    separately -- a face-on camera contains a shoulder turn whether or not it is
    calibrated, and a stereo rig cannot recover a quantity its footage never saw.

    **It now blocks something.** Through Phase 8 every metric in the registry
    measured the image plane, which an uncalibrated camera supplies, so the gate
    was built, enforced and tested against a set it refused nothing from -- on
    purpose, so that Phase 9's metrics would meet machinery that predated them
    rather than a check written on the day the first metric needed it, when the
    metric is the reason to weaken it. Phase 9 added six metrics declaring
    `STEREO`, and on an uncalibrated project this is where they are refused.

    The refusal names the calibration rather than the symptom, for the same
    reason the view gate does: "X-factor (3D) needs a stereo calibration" is a
    thing a person can act on, and "no reconstruction was supplied" is a
    restatement of the same fact one layer further from its cause.
    """
    blocked = {name for name, entry in REGISTRY.items() if not status.at_least(entry.requires)}
    if not blocked:
        return produced, refused

    survivors = [entry for entry in refused if entry.name not in blocked]
    survivors.extend(
        RefusedMetric(
            name=name,
            reason=(
                f"{definition(name).label} needs a {definition(name).requires.value} "
                f"calibration and this project has {status.value}. "
                f"{definition(name).summary}"
            ),
        )
        for name in sorted(blocked, key=lambda entry: entry.value)
    )
    return [metric for metric in produced if metric.name not in blocked], survivors


def _calibration_warnings(status: CalibrationStatus) -> list[str]:
    """What the calibration state means for reading these numbers.

    Stated on every result rather than only when something was refused, because
    the interesting case is the one where nothing was: a reader who sees no
    refusals should not conclude that the geometry was known.
    """
    if status is CalibrationStatus.NONE:
        return [
            "This camera is not calibrated, so the landmarks these metrics are measured "
            "from still carry the lens's distortion -- tens of pixels near the frame edge "
            "on an ordinary phone, and nothing in the numbers shows it. Calibrating the "
            "camera removes it and makes every angle and distance below a cleaner "
            "statement about the image plane."
        ]
    if status is CalibrationStatus.INTRINSICS:
        return [
            "The lens has been measured and removed from these landmarks. They remain "
            "measurements of the image plane: a calibrated camera knows which direction "
            "a pixel came from and not how far away it was, so no value here is a "
            "metric-scale claim about the body. That needs two calibrated views of the "
            "same instant, which is Phase 9."
        ]
    return [
        "Both cameras are calibrated and their relative pose is known, so the metrics "
        "whose basis is `spatial` are measurements of the body in metres rather than of "
        "its picture. Everything else below is still a single-view projected quantity "
        "measured on this clip alone, and the two are not interchangeable -- where a "
        "projected and a reconstructed version of the same quantity both appear, they "
        "will disagree, and the reconstructed one carries the uncertainty that says by "
        "how much it could."
    ]


def _view_warnings(result: MetricSet) -> list[str]:
    """Facts about the recording a reader should not have to derive from the refusals.

    The view comes first and absorbs what it explains. On a down-the-line clip
    the shoulders are also not square at address and the lead side also cannot be
    named; both are consequences of where the camera was, and listing them as
    separate findings would read as three problems where there is one fact.
    """
    warnings: list[str] = []
    view = result.view.view if result.view is not None else CameraView.UNKNOWN

    if view is CameraView.DOWN_THE_LINE:
        warnings.append(
            "This is a down-the-line recording. Posture, hand depth, hand path and "
            "timing are all measurable from it; shoulder turn, pelvis turn and X-factor "
            "are not, because the shoulder line points at the camera and its rotation "
            "does not appear in the picture. A second camera placed face-on is what "
            "makes those available."
        )
    elif view is CameraView.UNKNOWN and result.view is not None:
        warnings.append(
            f"The camera view could not be established. {result.view.methodology} "
            "Metrics are still reported, but what each one corresponds to on the body "
            "depends on where the camera stood, so they carry no anatomical reading."
        )

    # Only where the view has not already accounted for it: on a down-the-line
    # clip an off-square baseline is the expected consequence, not a finding.
    if view is not CameraView.DOWN_THE_LINE:
        off_baseline = [
            reference for reference in result.references if not reference.square_at_address
        ]
        if off_baseline:
            names = " and ".join(reference.landmarks for reference in off_baseline)
            worst = max(reference.excess for reference in off_baseline)
            warnings.append(
                f"The {names} projected up to {worst:.1f} times wider mid-swing than at "
                "address, so the player was not square to the camera there and rotation "
                "by foreshortening has been refused rather than measured against a pose "
                "they never held."
            )

        if result.lead_side is not None and result.lead_side.side is None:
            warnings.append(
                "Which arm leads could not be determined, so the lead and trail arm "
                f"angles were refused. {result.lead_side.methodology}"
            )

    return warnings


def compute_metrics(
    filtered: FilteredSequence,
    phases: SwingPhases,
    config: MetricConfig | None = None,
    calibration: CalibrationStatus = CalibrationStatus.NONE,
    reconstruction: ReconstructedSequence | None = None,
) -> MetricSet:
    """Measure every biomechanics metric this clip supports.

    `calibration` says what is known about this camera's geometry. It does not
    change how anything is computed -- the lens, if one was measured, came off
    the landmarks far below this, in `pose/series.py` -- and it decides what may
    be reported, which is a different question asked in a different place.

    `reconstruction` is the 3D positions Phase 9 triangulated from this clip and
    its partner, indexed by **this** clip's frames. It is an argument rather than
    something computed here for the reason the calibration status is: this
    function measures one clip, and a reconstruction is a fact about two. Absent,
    the spatial metrics are refused by name -- and on a project without a stereo
    rig they are refused by the calibration gate before that, which is the more
    useful of the two answers and so the one that wins.
    """
    resolved = config or MetricConfig()

    if not phases.detected:
        return _empty(
            filtered,
            resolved,
            float("nan"),
            "No swing was detected in this clip, so there are no events to measure metrics "
            "at. Every metric here is anchored to an instant or an interval of a swing; "
            "without one, a reported value would be a measurement of an invented moment.",
            calibration,
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
            calibration,
        )

    anchors = build_anchors(phases)

    # Before anything is measured, because every measurement is tagged with it
    # and some are not available from every camera position.
    estimate = infer_view(body, anchors, resolved)
    view = estimate.view

    lead = arms.infer_lead_side(body, anchors, resolved)

    posture_metrics, posture_refused = posture.metrics(body, anchors, view)
    rotation_metrics, rotation_refused, references = rotation.metrics(body, anchors, resolved, view)
    arm_metrics, arm_refused = arms.metrics(body, anchors, lead, view)
    timing_metrics, timing_refused = timing.metrics(body, anchors, phases, view)
    spatial_metrics, spatial_refused = spatial.metrics(
        reconstruction, anchors, lead, resolved, view
    )

    produced, all_refused = _apply_view_gate(
        [
            *posture_metrics,
            *rotation_metrics,
            *arm_metrics,
            *timing_metrics,
            *spatial_metrics,
        ],
        [
            *posture_refused,
            *rotation_refused,
            *arm_refused,
            *timing_refused,
            *spatial_refused,
        ],
        view,
    )
    produced, all_refused = _apply_calibration_gate(produced, all_refused, calibration)

    result = MetricSet(
        computed=True,
        metrics=produced,
        refused=all_refused,
        view=estimate,
        lead_side=lead,
        references=references,
        torso_length=body.torso_length,
        geometry=filtered.geometry,
        frames=len(body),
        calibration=calibration,
        slow_motion_factor=filtered.slow_motion_factor,
        config=resolved,
    )
    result.warnings = [*_view_warnings(result), *_calibration_warnings(calibration)]
    return result


def known_metrics() -> tuple[MetricName, ...]:
    """Every metric name the engine can produce, in registry order."""
    return tuple(REGISTRY)
