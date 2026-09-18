"""Running a ball detector over a clip, and assembling the report.

The only module here that knows about both video and poses, which is
`club/extract.py`'s role one package along and is kept separate for the same
reason: the detector stays unaware of where frames come from, the tracker stays
unaware of pixels, and a test can drive the whole path with a fake detector and a
synthetic pose sequence without decoding anything.

## The search region is anchored to the ground, and does not move

Two decisions, and the reference footage forced the second one.

**It does not follow the hands.** `club/extract.py` anchors its search on the
filtered hand position at each frame, because a club is held and therefore goes
where the hands go. A teed ball does not, so a region that tracked the hands
would sweep away from the ball exactly when the hands move fastest -- the ball
would go missing through the downswing and come back in the follow-through, which
is a departure, in the right phase, with a plausible confidence. The anchor is
therefore taken once, at address, and held for the clip, including through the
frames after impact where the permanence check does its work.

**It is anchored at the feet rather than at the hands**, which is the stronger
structural fact and was not the first thing tried. A teed ball sits on the
ground, and the pose layer knows where the ground is because it knows where the
ankles are; anchoring at the hands instead says only "within about a club length",
which on a vertical phone clip is a disc covering the whole picture -- sky, trees
and all.

That is not a cost in work, it is a cost in **correctness**, and on
`data/face-on/rory_face_on.mp4` it decided the clip. The identification looks for
a small round thing that sits perfectly still for hundreds of frames and then
stops being there, and a hands-anchored region offers it the sky: a patch of
cloud between two branches is rounder, better resolved and stiller than a golf
ball 4 px across, and it eventually drifts behind a branch. The clip established
on a point 126 px from the top of the frame and reported an impact 117 frames
late. Anchored at the ankles the same clip finds the ball.

The pose layer still supplies the two other things the search needs: a torso
length, which is the scale every bound in `BallConfig` is stated in, and a clock
in real seconds.

## What is not cached

Nothing, for the reasons `club/extract.py` gives and one more that is specific to
this phase. The work is a morphological opening and a contour pass over one
region per frame, which `scripts/benchmark_ball.py` measures at a fraction of
what pose estimation costs on the same footage; and the output is a handful of
numbers per frame plus one instant, which is smaller than the key that would
index it.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from analyzer.ball.detector import (
    BallAnchor,
    BallCandidate,
    BallDetectionError,
    BallDetector,
)
from analyzer.ball.track import BallCandidateView, BallEvidence, BallTrackResult, track_ball
from analyzer.contracts.ball import BallConfig, BallRefusal, BallTrackingReport
from analyzer.contracts.phases import SwingEvent, SwingPhases
from analyzer.contracts.pose import FrameGeometry, Landmark
from analyzer.coordinates import frame_widths_to_pixels, pixels_to_frame_widths
from analyzer.filtering.landmarks import FilteredSequence
from analyzer.ingestion.probe import probe
from analyzer.ingestion.reader import OpenCVFrameSource
from analyzer.phases.signals import SwingSignals, swing_signals
from analyzer.progress import NullReporter, ProgressReporter, ProgressTracker

TASK_NAME = "detect_ball"

# Frames are consumed once, in order, and never revisited -- the tracker works on
# the candidates rather than on the pixels -- so the frame cache would only hold
# memory it can never hand back a hit for. The same reasoning, and the same value,
# as `pose/extract.py` and `club/extract.py`.
_FRAME_CACHE_BYTES = 1

# Fraction of a clip treated as address when no swing was detected, for the
# purpose of placing the search region. A guess, and labelled as one: it is used
# only to point a region a couple of torso lengths across at a player who is
# standing still, so being wrong by a good margin still covers the ball.
_ASSUMED_ADDRESS_FRACTION = 0.25


def detect_ball(
    video: Path,
    filtered: FilteredSequence,
    detector: BallDetector,
    *,
    phases: SwingPhases | None = None,
    config: BallConfig | None = None,
    reporter: ProgressReporter | None = None,
    request_id: int | str | None = None,
) -> BallTrackingReport:
    """Find the ball in `video` and the frame at which it stopped being there.

    The detector is passed in rather than constructed here, for the reason
    `extract_poses` and `track_club` pass theirs in: it is the thing most likely
    to be swapped, and a test that supplies a scripted one exercises everything
    around it without an image.
    """
    resolved = config or BallConfig()
    signals = swing_signals(filtered)
    geometry = filtered.geometry

    if not math.isfinite(signals.torso_length) or signals.torso_length <= 0.0:
        raise BallDetectionError(
            "The subject's shoulder-to-hip span could not be measured from this clip, "
            "so there is no scale to state the search bounds in.",
            remediation=(
                "Check that the whole body is in frame and that pose extraction found "
                "shoulders and hips. `analyzer filter` reports which landmarks were tracked."
            ),
        )

    anchor_px = _address_anchor(filtered, signals, geometry, phases)
    if anchor_px is None:
        raise BallDetectionError(
            "Neither the player's feet nor their hands were tracked anywhere in this clip, "
            "so there is nowhere to put a search region: a ball is found on the ground where "
            "the player is standing, and nothing here knows where that is.",
            remediation=(
                "Check the filtering report. A clip framed from the waist up has no ankles in "
                "it, and one where the hands were lost throughout is too blurred or too "
                "tightly framed for any of this layer to run."
            ),
        )

    started = time.perf_counter()
    evidence = _collect_evidence(
        video, signals, geometry, anchor_px, detector, reporter, request_id
    )
    elapsed = time.perf_counter() - started

    result = track_ball(evidence, resolved, phases=phases, torso_length=signals.torso_length)
    return _report(
        video=video,
        detector=detector,
        geometry=geometry,
        signals=signals,
        result=result,
        config=resolved,
        elapsed_s=elapsed,
        phases=phases,
    )


def _address_span(signals: SwingSignals, phases: SwingPhases | None) -> int:
    """How many frames from the start count as address, for placing the region.

    Bounded by the takeaway where a swing was detected, and by a fixed fraction of
    the clip where none was. The fraction is a guess and is labelled as one: it is
    used only to point a region a couple of torso lengths across at a player who
    is standing still, so being wrong by a good margin still covers the ball.
    """
    frames = len(signals.t)
    if phases is not None and phases.detected:
        takeaway = phases.event(SwingEvent.TAKEAWAY)
        if takeaway is not None and takeaway.frame_index > 0:
            return min(takeaway.frame_index, frames)
    return max(1, int(_ASSUMED_ADDRESS_FRACTION * frames))


def _median_position(
    filtered: FilteredSequence, landmark: Landmark, stop: int
) -> tuple[float, float] | None:
    """Where one landmark sat over the address frames, in frame widths.

    A median rather than the value at any single frame: a player waggles, shifts
    their weight and re-sets their feet, and one frame's reading puts the region's
    centre wherever that happened to be. Over a span it is where they stood.
    """
    track = filtered.landmarks.get(landmark)
    if track is None:
        return None
    usable = track.valid[:stop]
    if not np.any(usable):
        return None
    position = track.position[:stop][usable]
    return float(np.median(position[:, 0])), float(np.median(position[:, 1]))


def _address_anchor(
    filtered: FilteredSequence,
    signals: SwingSignals,
    geometry: FrameGeometry,
    phases: SwingPhases | None,
) -> tuple[float, float] | None:
    """Where the ball is expected to be at address, in pixels. Held for the clip.

    The **ankle midpoint**, which is the ground under the player. See the module
    docstring on why this rather than the hands: a ball sits on the ground, and
    the difference between "on the ground near the feet" and "within a club length
    of the hands" is the difference between a region a couple of torso lengths
    across and one that contains the sky.

    Falls back to the hands when the ankles were never tracked -- a clip framed
    from the waist up, or a player standing in deep rough. The fallback is worse
    for the reason above and it is still better than refusing, because the
    identification's own bounds are what decide whether anything is reported.
    """
    stop = _address_span(signals, phases)

    left = _median_position(filtered, Landmark.LEFT_ANKLE, stop)
    right = _median_position(filtered, Landmark.RIGHT_ANKLE, stop)
    found = [point for point in (left, right) if point is not None]
    if found:
        centre = np.array(
            [
                float(np.mean([point[0] for point in found])),
                float(np.mean([point[1] for point in found])),
            ]
        )
    else:
        usable = signals.hand.valid[:stop]
        if not np.any(usable):
            usable = signals.hand.valid
            stop = len(signals.t)
            if not np.any(usable):
                return None
        position = signals.hand.position[:stop][usable]
        centre = np.array([float(np.median(position[:, 0])), float(np.median(position[:, 1]))])

    pixels: NDArray[np.float64] = frame_widths_to_pixels(centre, geometry)
    return float(pixels[0]), float(pixels[1])


def _collect_evidence(
    video: Path,
    signals: SwingSignals,
    geometry: FrameGeometry,
    anchor_px: tuple[float, float],
    detector: BallDetector,
    reporter: ProgressReporter | None,
    request_id: int | str | None,
) -> list[BallEvidence]:
    """Decode every frame once and ask the detector what it offers.

    Detection runs over the whole clip before any of it is reasoned about, which
    is what makes the identification possible at all: the ball cannot be picked
    out of the candidates without having seen which candidate positions recur.
    The candidates are a few numbers each, so holding a clip's worth of them costs
    nothing next to the frames they came from -- which are released as they are
    consumed.

    Unlike the club's equivalent, a frame with no tracked hand is **still
    searched**. The anchor is fixed at address and does not need this frame's
    hands, and the frames where the pose layer loses the hands are exactly the
    frames around impact -- which is the only part of the clip this phase is
    measuring.
    """
    probed = probe(video)
    total = probed.metadata.timing.frame_count
    torso_px = signals.torso_length * geometry.width
    anchor = BallAnchor(x=anchor_px[0], y=anchor_px[1], torso_px=torso_px)

    tracker = ProgressTracker(reporter or NullReporter(), task=TASK_NAME, request_id=request_id)
    tracker.report("starting", 0, total, detail=detector.info.name)

    evidence: list[BallEvidence] = []
    with OpenCVFrameSource(probed, cache_bytes=_FRAME_CACHE_BYTES) as source:
        for frame in source.frames():
            position = frame.index
            timestamp = (
                float(signals.t[position])
                if position < signals.t.size
                else float(frame.timestamp_s)
            )
            detection = detector.detect(frame, anchor)
            evidence.append(
                BallEvidence(
                    frame_index=position,
                    timestamp_s=timestamp,
                    candidates=tuple(
                        _to_frame_widths(candidate, geometry, signals.torso_length)
                        for candidate in detection.candidates
                    ),
                    refusal=detection.refusal,
                )
            )
            tracker.report("detecting", position + 1, total)

    tracker.report("done", len(evidence), total)
    return evidence


def _to_frame_widths(
    candidate: BallCandidate, geometry: FrameGeometry, torso_length: float
) -> BallCandidateView:
    """A pixel-space candidate in the frame everything above Phase 3 measures in.

    Pixels are isotropic and frame widths are isotropic, so the change is a scale
    and a flip of y, and a radius converts by the scale alone -- there is no
    aspect correction because there is no anisotropy to correct. The pixel radius
    is carried through unchanged, because whether the ball has any shape left is a
    fact about the sensor grid and not about any normalised frame.
    """
    centre = pixels_to_frame_widths(np.array([candidate.x, candidate.y]), geometry)
    radius = candidate.radius_px / float(geometry.width)
    return BallCandidateView(
        x=float(centre[0]),
        y=float(centre[1]),
        radius=radius,
        radius_px=candidate.radius_px,
        radius_torso=radius / torso_length if torso_length > 0 else radius,
        contrast=candidate.contrast,
        circularity=candidate.circularity,
    )


def _report(
    *,
    video: Path,
    detector: BallDetector,
    geometry: FrameGeometry,
    signals: SwingSignals,
    result: BallTrackResult,
    config: BallConfig,
    elapsed_s: float,
    phases: SwingPhases | None,
) -> BallTrackingReport:
    """Assemble what crosses the engine boundary."""
    frames = result.frames
    count = len(frames)
    warnings = list(result.warnings)

    if phases is None or not phases.detected:
        warnings.append(
            "No swing was detected in this clip. The identification still ran, and it "
            "searched every frame rather than only the frames before the top -- which "
            "dilutes the agreement a ball is established on, and can only make this refuse "
            "rather than make it confident about the wrong position."
        )

    return BallTrackingReport(
        detected=result.observed_frames > 0,
        video_path=str(video),
        detector=detector.info,
        geometry=geometry,
        torso_length=signals.torso_length,
        slow_motion_factor=signals.slow_motion_factor,
        frame_count=count,
        observed_frames=result.observed_frames,
        pre_departure_coverage=result.pre_departure_coverage,
        established=result.established,
        departure=result.departure,
        quality=result.quality,
        frames=frames,
        refusals=result.refusals,
        tracked_at=datetime.now(UTC),
        elapsed_s=elapsed_s,
        ms_per_frame=(elapsed_s * 1000.0 / count) if count else 0.0,
        config=config,
        refusal=(
            None
            if result.observed_frames
            else (
                "No position in this clip held something of a ball's size and shape often "
                "enough to be called a ball. The refusal counts say which check: "
                "out_of_frame is the framing -- the ball sits about a club length below the "
                "hands and a clip cropped to the upper body does not contain it -- and "
                "no_candidate is contrast, resolution, or a ball that is simply not there."
            )
        ),
        warnings=warnings,
    )


def frame_evidence(
    frame_index: int,
    timestamp_s: float,
    candidates: tuple[BallCandidate, ...],
    *,
    geometry: FrameGeometry,
    torso_length: float,
    refusal: BallRefusal | None = None,
) -> BallEvidence:
    """Express one frame's candidates in frame widths, without decoding anything.

    The boundary between the pixel grid and the frame every measurement is taken
    in, for exactly one frame. Public because it is the whole of what a benchmark
    or a test needs: `scripts/benchmark_ball.py` renders a ball that vanishes at a
    known frame and drives this directly, which is how the located instant gets
    measured against a known one without encoding a video first.
    """
    return BallEvidence(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        candidates=tuple(
            _to_frame_widths(candidate, geometry, torso_length) for candidate in candidates
        ),
        refusal=refusal,
    )
