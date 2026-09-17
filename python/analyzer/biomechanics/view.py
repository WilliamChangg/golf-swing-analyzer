"""Working out where the camera stood, because it decides what the numbers mean.

Every measurement in this engine is a projection, and a projection is only
interpretable once you know the direction it was taken from. The same spine tilt
is lateral side bend from in front of the player and forward posture angle from
behind them down the target line. The same hip displacement is a slide towards
the target in one view and a move towards the ball in the other. The arithmetic
is identical; the anatomy is not.

So the view is **measured**, not configured, and every metric carries it.

## What it is measured from

The projected width of the shoulder line at address, in torso lengths.

Seen face-on the shoulders lie broadside to the camera and span most of a torso
length or more. Seen down the line they point almost directly at it and collapse
to nearly nothing. On the reference clips those are 0.83 and 0.10 -- a factor of
eight, which is a verdict rather than a close call.

The thresholds have an anatomical basis rather than a measured one, and that is
worth being explicit about since almost nothing else in this engine does. An
adult's shoulder width is a fairly stable multiple of the distance from their
shoulders to their hips, so a shoulder line spanning more than half that
distance cannot be pointing at the camera, and one spanning less than a third of
it cannot be broadside. Between the two the answer is `UNKNOWN`, which is the
honest report for an oblique camera rather than a failure.

## Corroboration

`openness` -- the address span divided by the widest the shoulder line was ever
seen in the clip -- is the cosine of how far off broadside the shoulders were at
address, and it is measured entirely within the clip with no anatomical
assumption at all. A swing turns the shoulders through roughly a right angle, so
both views contain a frame where the line is nearly square, which is what makes
the comparison meaningful.

It is corroboration rather than the verdict because it depends on that frame
existing and on the estimator not overstating the span there. Both are true
often enough to be useful and not always: on the face-on reference clip the
widest frame is a post-impact artefact reading 12% wider than the player can be,
which is the same defect that moved the rotation baseline to address in Phase 5.
"""

from __future__ import annotations

import numpy as np

from analyzer.biomechanics.anchors import Anchors
from analyzer.biomechanics.body import Body
from analyzer.contracts.metrics import CameraView, MetricConfig, ViewEstimate

_UNDETERMINED = ViewEstimate(
    view=CameraView.UNKNOWN,
    confidence=0.0,
    shoulder_span_ratio=float("nan"),
    hip_span_ratio=float("nan"),
    openness=float("nan"),
    frames=[],
    methodology=(
        "Not determined: the clip has no address phase, or the shoulders were not "
        "tracked through it. The view is read from how wide the shoulder line "
        "projects while the player is set up to the ball, and without those frames "
        "there is nothing to read it from."
    ),
)


def _median_over(series: np.ndarray, frames: tuple[int, ...]) -> tuple[float, list[int]]:
    usable = [frame for frame in frames if 0 <= frame < series.size and np.isfinite(series[frame])]
    if not usable:
        return float("nan"), []
    return float(np.median(series[usable])), usable


def infer_view(body: Body, anchors: Anchors, config: MetricConfig) -> ViewEstimate:
    """Decide which view a clip was shot from, or report that it is undecidable."""
    if anchors.address is None or not anchors.address.frames:
        return _UNDETERMINED

    shoulder_span = body.shoulder_span
    span, frames = _median_over(shoulder_span, anchors.address.frames)
    if not frames or not np.isfinite(body.torso_length) or body.torso_length <= 0.0:
        return _UNDETERMINED

    ratio = span / body.torso_length
    hip_span, _ = _median_over(body.hip_span, anchors.address.frames)
    hip_ratio = hip_span / body.torso_length

    widest = float(np.nanmax(shoulder_span)) if np.any(np.isfinite(shoulder_span)) else float("nan")
    openness = span / widest if np.isfinite(widest) and widest > 0 else float("nan")

    # The width of the undecided band, used to scale how clear the answer is.
    # A full band clear of the nearer threshold scores 1.
    band = max(config.face_on_span_ratio - config.down_the_line_span_ratio, 1e-9)

    if ratio >= config.face_on_span_ratio:
        view = CameraView.FACE_ON
        confidence = float(np.clip((ratio - config.face_on_span_ratio) / band, 0.0, 1.0))
        verdict = (
            "broadside to the camera, so this is a face-on recording: rotation is "
            "measurable by foreshortening and the player's left and right are "
            "distinguishable."
        )
    elif ratio <= config.down_the_line_span_ratio:
        view = CameraView.DOWN_THE_LINE
        confidence = float(np.clip((config.down_the_line_span_ratio - ratio) / band, 0.0, 1.0))
        verdict = (
            "pointing nearly at the camera, so this is a down-the-line recording: "
            "the frame's horizontal axis runs towards and away from the ball, and "
            "rotation about the spine is not recoverable from it."
        )
    else:
        view = CameraView.UNKNOWN
        confidence = 0.0
        verdict = (
            f"between the {config.down_the_line_span_ratio:g} and "
            f"{config.face_on_span_ratio:g} torso lengths that would settle it, so the "
            "camera is oblique and the view is reported as unknown rather than rounded "
            "to the nearer label."
        )

    return ViewEstimate(
        view=view,
        confidence=confidence,
        shoulder_span_ratio=ratio,
        hip_span_ratio=hip_ratio,
        openness=openness,
        frames=frames,
        methodology=(
            f"The shoulder line projects {ratio:.2f} torso lengths across the "
            f"{len(frames)} frames of the address phase -- {verdict} Corroborated by its "
            f"width there against the widest in the clip, {openness:.2f}, which is the "
            "cosine of how far off broadside it was at address."
        ),
    )
