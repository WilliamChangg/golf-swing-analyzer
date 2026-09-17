"""Running a pose estimator over a clip and storing the result.

This is the only place that knows about both video and poses, which keeps the
frame source unaware of what reads it and the estimator unaware of where frames
come from. Both arrive as parameters, so a test can supply a fake estimator and
exercise the whole path without loading a model.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from analyzer.contracts.pose import (
    LANDMARK_COUNT,
    PoseExtractionResult,
    PoseExtractionStats,
    PoseFrame,
    PoseSequence,
)
from analyzer.ingestion.probe import probe
from analyzer.ingestion.reader import OpenCVFrameSource
from analyzer.paths import cache_dir
from analyzer.pose.estimator import PoseEstimator
from analyzer.pose.store import write_sequence
from analyzer.progress import NullReporter, ProgressReporter, ProgressTracker

TASK_NAME = "extract_poses"

# Decoded frames are consumed once, in order, and never revisited, so the frame
# cache would only hold memory it can never hand back a hit for.
_FRAME_CACHE_BYTES = 1


def _mean_visibility(frames: list[PoseFrame]) -> float | None:
    """Mean visibility over detected landmarks, or None if nothing was detected."""
    values = [point.visibility for frame in frames if frame.detected for point in frame.image]
    if not values:
        return None
    return float(np.mean(values))


def _collect_warnings(stats: PoseExtractionStats, nudges: int) -> list[str]:
    warnings: list[str] = []

    if stats.frames_detected == 0:
        warnings.append(
            "No pose was detected in any frame. Nothing downstream can be computed from "
            "this extraction. Check that the subject is in frame and adequately lit."
        )
    elif stats.detection_rate < 0.5:
        warnings.append(
            f"A pose was found in only {stats.detection_rate:.0%} of frames. Metrics derived "
            "from this clip will have large gaps."
        )

    if nudges:
        warnings.append(
            f"{nudges} frames were closer together than one millisecond, so the clock handed "
            "to the estimator was advanced to keep it strictly increasing. Recorded frame "
            "times are unaffected: they come from the container index."
        )

    return warnings


def extract_poses(
    video: Path,
    estimator: PoseEstimator,
    *,
    reporter: ProgressReporter | None = None,
    request_id: int | str | None = None,
) -> PoseSequence:
    """Run `estimator` over every frame of `video`, in order.

    The estimator is passed in rather than constructed here: it is stateful,
    expensive to create, and the thing most likely to be swapped.
    """
    probed = probe(video)
    total_frames = probed.metadata.timing.frame_count

    tracker = ProgressTracker(reporter or NullReporter(), task=TASK_NAME, request_id=request_id)
    tracker.report("starting", 0, total_frames, detail=estimator.info.name)

    frames: list[PoseFrame] = []
    started = time.perf_counter()

    with OpenCVFrameSource(probed, cache_bytes=_FRAME_CACHE_BYTES) as source:
        for frame in source.frames():
            frames.append(estimator.estimate(frame))
            tracker.report("estimating", len(frames), total_frames)

    elapsed = time.perf_counter() - started
    detected = sum(1 for frame in frames if frame.detected)

    stats = PoseExtractionStats(
        frames_processed=len(frames),
        frames_detected=detected,
        detection_rate=detected / len(frames) if frames else 0.0,
        elapsed_s=elapsed,
        ms_per_frame=(elapsed * 1000 / len(frames)) if frames else 0.0,
        mean_visibility=_mean_visibility(frames),
    )
    tracker.report("done", len(frames), total_frames)

    return PoseSequence(
        video_path=probed.metadata.path,
        video_content_key=probed.metadata.content_key,
        model=estimator.info,
        extracted_at=datetime.now(UTC),
        stats=stats,
        frames=frames,
    )


def output_path_for(sequence: PoseSequence) -> Path:
    """Where a sequence is stored.

    Under the content key of the *video*, so re-extracting the same clip
    overwrites its own result and a different clip never collides with it. The
    model name is in the filename because two models over one clip are two
    different results, both worth keeping.
    """
    return (
        cache_dir()
        / "poses"
        / sequence.video_content_key.as_path_segment()
        / f"{sequence.model.name}.parquet"
    )


def extract_and_store(
    video: Path,
    estimator: PoseEstimator,
    *,
    output: Path | None = None,
    reporter: ProgressReporter | None = None,
    request_id: int | str | None = None,
) -> PoseExtractionResult:
    """Extract, persist, and return what the desktop app needs to display.

    The landmarks themselves are not in the return value -- a 240 fps clip is
    tens of thousands of frames of 33 landmarks in two spaces, which belongs in
    a columnar file rather than a JSON-RPC response. The caller gets the path.
    """
    sequence = extract_poses(video, estimator, reporter=reporter, request_id=request_id)
    destination = output or output_path_for(sequence)
    write_sequence(sequence, destination, landmark_count=LANDMARK_COUNT)

    nudges = getattr(estimator, "timestamp_nudges", 0)
    return PoseExtractionResult(
        video_path=sequence.video_path,
        output_path=str(destination),
        model=sequence.model,
        extracted_at=sequence.extracted_at,
        stats=sequence.stats,
        warnings=_collect_warnings(sequence.stats, int(nudges)),
    )
