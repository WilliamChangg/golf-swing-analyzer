"""MediaPipe implementation of `PoseEstimator`.

Everything MediaPipe-shaped is confined to this file. Three details here are
easy to get wrong and silent when wrong, so each is handled explicitly rather
than left to a default.

**Colour order.** Frames arrive BGR, because that is what OpenCV decodes to.
MediaPipe's `SRGB` format means what it says. Handing it BGR does not fail --
it degrades detection on skin tones and shadowed limbs in a way that looks like
an ordinary bad result -- so the conversion is done here, once.

**Running mode.** `VIDEO` rather than `IMAGE`, so the landmarker tracks between
frames instead of re-detecting from scratch. That is both faster and steadier,
but it makes the estimator stateful: frames must arrive in order.

**Timestamps.** `detect_for_video` takes integer milliseconds and requires them
to increase strictly. At 240 fps frames are 4.17 ms apart and at 480 fps 2.08
ms, so rounding alone will eventually repeat a value and MediaPipe rejects the
call. The clock handed to MediaPipe is therefore forced to advance, while every
timestamp that reaches a result stays the real one from the container index --
see `_MonotonicMilliseconds`.
"""

from __future__ import annotations

from types import TracebackType
from typing import cast

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np
from mediapipe.tasks.python import BaseOptions  # type: ignore[import-untyped]
from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import-untyped]
from numpy.typing import NDArray

from analyzer.contracts.pose import (
    LANDMARK_COUNT,
    LandmarkPoint,
    PoseFrame,
    PoseModelInfo,
)
from analyzer.environment.hardware import mediapipe_delegate
from analyzer.ingestion.reader import VideoFrame
from analyzer.pose.estimator import (
    PoseEstimationError,
    model_info,
    resolve_model,
)

# Defaults are MediaPipe's own. They are named here rather than left implicit
# because they are recorded on every result, and a result is only reproducible
# if the thresholds that produced it are known.
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
DEFAULT_MIN_PRESENCE_CONFIDENCE = 0.5
DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5


class _MonotonicMilliseconds:
    """Converts real timestamps to the strictly-increasing integers MediaPipe wants.

    When two frames round to the same millisecond, the second is nudged forward
    by one. That makes the value handed to MediaPipe drift from real time by up
    to a millisecond on very high frame rates.

    This is safe, and it is worth being precise about why: the clock is used
    only by MediaPipe's internal tracker to order frames. No measurement is ever
    taken from it. Every timestamp that reaches a `PoseFrame`, a stored
    sequence, or a metric comes from the container's packet index instead.
    """

    def __init__(self) -> None:
        self._previous: int | None = None
        self._nudges = 0

    @property
    def nudges(self) -> int:
        """How many frames had to be pushed forward to stay strictly increasing."""
        return self._nudges

    def next(self, timestamp_s: float) -> int:
        candidate = round(timestamp_s * 1000)
        if self._previous is not None and candidate <= self._previous:
            candidate = self._previous + 1
            self._nudges += 1
        self._previous = candidate
        return candidate


def _to_points(landmarks: object) -> list[LandmarkPoint]:
    """Map MediaPipe landmarks onto the contract, dropping MediaPipe's types."""
    points: list[LandmarkPoint] = []
    for landmark in landmarks:  # type: ignore[attr-defined]
        points.append(
            LandmarkPoint(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                # Either confidence can be absent depending on the model build.
                # Absent is reported as 0.0, which the filtering layer treats as
                # "do not trust", rather than as 1.0, which would invent trust.
                visibility=float(landmark.visibility or 0.0),
                presence=float(landmark.presence or 0.0),
            )
        )
    return points


class MediaPipePoseEstimator:
    """Pose estimation via MediaPipe Tasks, in VIDEO running mode."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        min_pose_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence: float = DEFAULT_MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE,
    ) -> None:
        entry, path = resolve_model(model_name)

        delegate = mediapipe_delegate()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(path), delegate=getattr(BaseOptions.Delegate, delegate.upper())
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            # One pose: a swing video has one subject, and accepting more would
            # mean choosing between them later with no basis for the choice.
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )

        try:
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        # Boundary: MediaPipe raises bare RuntimeError for everything from a
        # corrupt model to a bad option, so all of it becomes one typed error.
        except Exception as exc:
            raise PoseEstimationError(
                f"MediaPipe could not load '{entry.name}': {type(exc).__name__}: {exc}",
                remediation=(
                    f"The model file may be corrupt. Re-download it with "
                    f"`python scripts/download_models.py --only {entry.name}`."
                ),
            ) from exc

        self._info = model_info(
            entry,
            path,
            delegate=delegate,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._clock = _MonotonicMilliseconds()
        self._closed = False

    @property
    def info(self) -> PoseModelInfo:
        return self._info

    @property
    def timestamp_nudges(self) -> int:
        """Frames whose MediaPipe clock value had to be forced forward."""
        return self._clock.nudges

    def estimate(self, frame: VideoFrame) -> PoseFrame:
        if self._closed:
            raise PoseEstimationError("This estimator has been closed.")

        # MediaPipe's SRGB means what it says; frames arrive BGR from the decoder.
        rgb = cast(NDArray[np.uint8], cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect_for_video(image, self._clock.next(frame.timestamp_s))
        except Exception as exc:
            raise PoseEstimationError(
                f"MediaPipe failed on frame {frame.index}: {type(exc).__name__}: {exc}"
            ) from exc

        if not result.pose_landmarks:
            # No detection is a recorded fact, not a gap. See `PoseFrame`.
            return PoseFrame(frame_index=frame.index, timestamp_s=frame.timestamp_s, detected=False)

        image_points = _to_points(result.pose_landmarks[0])
        hip_local_points = (
            _to_points(result.pose_world_landmarks[0]) if result.pose_world_landmarks else []
        )

        if len(image_points) != LANDMARK_COUNT:
            raise PoseEstimationError(
                f"MediaPipe returned {len(image_points)} landmarks, expected {LANDMARK_COUNT}. "
                "The model does not match the Landmark enum this build was written against."
            )

        return PoseFrame(
            frame_index=frame.index,
            timestamp_s=frame.timestamp_s,
            detected=True,
            image=image_points,
            hip_local=hip_local_points,
        )

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self) -> MediaPipePoseEstimator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
