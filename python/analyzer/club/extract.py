"""Running a club detector over a clip, and assembling the report.

The only module here that knows about both video and poses, which is
`pose/extract.py`'s role one layer along and is kept separate for the same
reason: the detector stays unaware of where frames come from, the tracker stays
unaware of pixels, and a test can drive the whole path with a fake detector and
a synthetic pose sequence without decoding anything.

## Why this needs the poses at all

A shaft detector anchored at nothing is a line detector, and a line detector
pointed at a driving range returns the horizon. The hands are the one fact about
a golf club that no amount of image processing recovers and no background feature
can imitate, so they arrive here from Phase 3's filtered trajectories -- which
also supply the two other things the search needs: a torso length, which is the
scale every bound in `ClubConfig` is stated in, and a clock in real seconds.

Using the **filtered** hand position rather than the raw landmark is deliberate.
The raw wrist jitters by a few pixels frame to frame, and the search region, the
support ray and the reported grip all hang off it; a jittering anchor would put
that jitter into the shaft angle, where it would be indistinguishable from the
club moving. It also means the anchor exists exactly where Phase 3 says a
position is supported, so a frame the filter refused arrives here as `NO_GRIP`
rather than as a search around a guess.

## What is not cached

Nothing. Tracking a clip costs a few milliseconds a frame against the ~17 ms a
frame pose estimation costs on the same footage, and a cache would add a second
schema to version and invalidate for a fraction of the work that has already been
done to get here. That is the same decision Phases 3, 5, 7 and 9 made and it
rests on the same kind of evidence -- `scripts/benchmark_club.py` measures it --
so it changes if the measurement does.

The one thing that would change it is a detector that is not classical. A learned
shaft detector would cost what a pose model costs, and at that point the club
track belongs in the content-keyed cache beside the landmarks.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from analyzer.club.detector import (
    ClubDetectionError,
    ClubDetector,
    GripAnchor,
    ShaftCandidate,
)
from analyzer.club.geometry import angle_deg, to_frame_widths
from analyzer.club.track import CandidateView, FrameEvidence, TrackResult, track_shafts
from analyzer.contracts.club import ClubConfig, ClubTrackingReport, ShaftRefusal
from analyzer.contracts.phases import SwingPhases
from analyzer.contracts.pose import FrameGeometry
from analyzer.coordinates import frame_widths_to_pixels
from analyzer.filtering.landmarks import FilteredSequence
from analyzer.ingestion.probe import probe
from analyzer.ingestion.reader import OpenCVFrameSource, VideoFrame
from analyzer.phases.signals import SwingSignals, swing_signals
from analyzer.progress import NullReporter, ProgressReporter, ProgressTracker

TASK_NAME = "track_club"

# Frames are consumed once, in order, and never revisited -- the tracker works on
# the candidates rather than on the pixels -- so the frame cache would only hold
# memory it can never hand back a hit for. The same reasoning, and the same
# value, as `pose/extract.py`.
_FRAME_CACHE_BYTES = 1


def track_club(
    video: Path,
    filtered: FilteredSequence,
    detector: ClubDetector,
    *,
    phases: SwingPhases | None = None,
    config: ClubConfig | None = None,
    reporter: ProgressReporter | None = None,
    request_id: int | str | None = None,
) -> ClubTrackingReport:
    """Track the club through `video`, anchored on `filtered`'s hand trajectory.

    The detector is passed in rather than constructed here, for the reason
    `extract_poses` passes its estimator in: it is the thing most likely to be
    swapped, and a test that supplies a scripted one exercises everything around
    it without an image.
    """
    resolved = config or ClubConfig()
    signals = swing_signals(filtered)
    geometry = filtered.geometry

    if not math.isfinite(signals.torso_length) or signals.torso_length <= 0.0:
        raise ClubDetectionError(
            "The subject's shoulder-to-hip span could not be measured from this clip, "
            "so there is no scale to state the search bounds in.",
            remediation=(
                "Check that the whole body is in frame and that pose extraction found "
                "shoulders and hips. `analyzer filter` reports which landmarks were tracked."
            ),
        )

    started = time.perf_counter()
    evidence = _collect_evidence(video, signals, geometry, detector, resolved, reporter, request_id)
    elapsed = time.perf_counter() - started

    result = track_shafts(
        evidence,
        resolved,
        phases=phases,
        frame_interval_s=_median_interval_s(signals),
    )
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


def _median_interval_s(signals: SwingSignals) -> float | None:
    """The clip's own frame interval, in real seconds.

    The factor that turns an image speed into an upper bound on motion blur: an
    exposure cannot outlast the interval between frames. Taken as a median rather
    than from the declared frame rate because variable-rate footage is the case
    Phase 1 exists to handle, and `frame / fps` is not when a frame was taken.
    """
    if signals.t.size < 2:
        return None
    intervals = np.diff(signals.t)
    finite = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    return float(np.median(finite)) if finite.size else None


def _collect_evidence(
    video: Path,
    signals: SwingSignals,
    geometry: FrameGeometry,
    detector: ClubDetector,
    config: ClubConfig,
    reporter: ProgressReporter | None,
    request_id: int | str | None,
) -> list[FrameEvidence]:
    """Decode every frame once and ask the detector what it offers.

    Detection runs over the whole clip before any of it is tracked, which is what
    makes the tracker's seeding possible: it cannot start at the frame with the
    best evidence without having seen every frame's evidence. The candidates are
    a few numbers each, so holding a clip's worth of them costs nothing next to
    the frames they came from -- which are released as they are consumed.
    """
    probed = probe(video)
    total = probed.metadata.timing.frame_count
    torso_px = signals.torso_length * geometry.width

    tracker = ProgressTracker(reporter or NullReporter(), task=TASK_NAME, request_id=request_id)
    tracker.report("starting", 0, total, detail=detector.info.name)

    anchors_px = frame_widths_to_pixels(signals.hand.position[:, :2], geometry)
    evidence: list[FrameEvidence] = []

    with OpenCVFrameSource(probed, cache_bytes=_FRAME_CACHE_BYTES) as source:
        for frame in source.frames():
            position = frame.index
            if position >= signals.t.size:
                # The container reported more frames than the pose extraction
                # covered. Recorded as a frame with no anchor rather than dropped,
                # so frame indices keep meaning frame indices.
                evidence.append(
                    FrameEvidence(
                        frame_index=position,
                        timestamp_s=float(frame.timestamp_s),
                        grip=None,
                        refusal=ShaftRefusal.NO_GRIP,
                    )
                )
                continue

            timestamp = float(signals.t[position])
            if not signals.hand.valid[position]:
                evidence.append(
                    FrameEvidence(
                        frame_index=position,
                        timestamp_s=timestamp,
                        grip=None,
                        refusal=ShaftRefusal.NO_GRIP,
                    )
                )
                tracker.report("detecting", position + 1, total)
                continue

            anchor = GripAnchor(
                x=float(anchors_px[position, 0]),
                y=float(anchors_px[position, 1]),
                torso_px=torso_px,
            )
            evidence.append(
                frame_evidence(
                    frame,
                    anchor,
                    detector,
                    geometry=geometry,
                    torso_length=signals.torso_length,
                    timestamp_s=timestamp,
                )
            )
            tracker.report("detecting", position + 1, total)

    tracker.report("done", len(evidence), total)
    return evidence


def frame_evidence(
    frame: VideoFrame,
    anchor: GripAnchor,
    detector: ClubDetector,
    *,
    geometry: FrameGeometry,
    torso_length: float,
    timestamp_s: float | None = None,
) -> FrameEvidence:
    """Ask the detector about one frame and express what it offers in frame widths.

    The boundary between the pixel grid and the frame every measurement is taken
    in, for exactly one frame. Public because it is also the whole of what a
    benchmark or a test needs: `scripts/benchmark_club.py` renders shafts at
    known angles and drives this directly, which is how detection rate against
    blur gets measured without encoding a video first.
    """
    detection = detector.detect(frame, anchor)
    grip_fw = to_frame_widths((anchor.x, anchor.y), geometry)
    return FrameEvidence(
        frame_index=frame.index,
        timestamp_s=frame.timestamp_s if timestamp_s is None else timestamp_s,
        grip=grip_fw,
        candidates=tuple(
            _to_frame_widths(candidate, grip_fw, geometry, torso_length)
            for candidate in detection.candidates
        ),
        refusal=detection.refusal,
    )


def _to_frame_widths(
    candidate: ShaftCandidate,
    grip_fw: tuple[float, float],
    geometry: FrameGeometry,
    torso_length: float,
) -> CandidateView:
    """A pixel-space candidate in the frame everything above Phase 3 measures in.

    The angle is **recomputed** from the converted endpoints rather than taken
    from the detector's own and negated. The two are equal -- `angle_to_frame_widths`
    is that negation and `test_club_geometry` pins the equality to floating-point
    precision -- but only one of them can be the definition, and the one derived
    from the coordinates actually reported is the one that cannot drift from
    them. This is the sign that turns a backswing into a downswing without
    failing anywhere, and this codebase has been caught by exactly that before:
    Phase 9's `look_at` built two cameras upside down and the reconstruction
    could not see it.
    """
    tip_px = (candidate.tip_x, candidate.tip_y)
    tip_fw = to_frame_widths(tip_px, geometry)
    dx, dy = tip_fw[0] - grip_fw[0], tip_fw[1] - grip_fw[1]
    length = math.hypot(dx, dy)
    return CandidateView(
        tip=tip_fw,
        tip_px=tip_px,
        angle_deg=angle_deg(dx, dy),
        length=length,
        length_torso=length / torso_length,
        support=candidate.support,
    )


def _report(
    *,
    video: Path,
    detector: ClubDetector,
    geometry: FrameGeometry,
    signals: SwingSignals,
    result: TrackResult,
    config: ClubConfig,
    elapsed_s: float,
    phases: SwingPhases | None,
) -> ClubTrackingReport:
    """Assemble what crosses the engine boundary."""
    frames = result.frames
    count = len(frames)
    warnings = list(result.warnings)

    if phases is None or not phases.detected:
        warnings.append(
            "No swing was detected in this clip, so coverage is reported clip-wide only. "
            "That number is dominated by the frames where the club is nearly still, and it "
            "is not evidence that the downswing was tracked."
        )

    return ClubTrackingReport(
        tracked=result.tracked_frames > 0,
        video_path=str(video),
        detector=detector.info,
        geometry=geometry,
        torso_length=signals.torso_length,
        slow_motion_factor=signals.slow_motion_factor,
        frame_count=count,
        tracked_frames=result.tracked_frames,
        coverage=result.tracked_frames / count if count else 0.0,
        unanchored_frames=result.unanchored_frames,
        phase_coverage=result.phase_coverage,
        quality=result.quality,
        impact=result.impact,
        frames=frames,
        refusals=result.refusals,
        tracked_at=datetime.now(UTC),
        elapsed_s=elapsed_s,
        ms_per_frame=(elapsed_s * 1000.0 / count) if count else 0.0,
        config=config,
        refusal=(
            None
            if result.tracked_frames
            else (
                "No frame produced a shaft that survived every check. The refusal counts say "
                "which check: no_grip is the pose layer, no_edges is lighting, no_candidate is "
                "blur or framing, ambiguous is the background, and low_confidence is all three "
                "at once."
            )
        ),
        warnings=warnings,
    )
