"""Pose estimator tests.

Split by what they need. The clock and the landmark mapping are pure and run
everywhere. The MediaPipe adapter's own behaviour -- that it converts colour
order, that it maps every landmark, that it is deterministic -- needs a real
model, and is marked accordingly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analyzer.contracts.pose import LANDMARK_COUNT
from analyzer.ingestion.reader import VideoFrame
from analyzer.pose.estimator import PoseEstimationError, resolve_model, variant_of
from analyzer.pose.mediapipe_estimator import (
    MediaPipePoseEstimator,
    _MonotonicMilliseconds,
    _to_points,
)
from tests.conftest import CFR_30FPS, requires_ffmpeg, requires_pose_model


class _FakeLandmark:
    """Stands in for a MediaPipe landmark, which is a plain attribute bag."""

    def __init__(self, value: float, *, visibility: float = 0.8, presence: float = 0.9):
        self.x = value
        self.y = value * 2
        self.z = -value
        self.visibility = visibility
        self.presence = presence


class TestMonotonicMilliseconds:
    """MediaPipe rejects a timestamp that does not exceed the previous one.

    At 480 fps frames are 2.08 ms apart, so rounding alone repeats a value and
    the call fails. What matters is that the fix stays inside MediaPipe: no
    measurement is ever taken from this clock.
    """

    def test_converts_seconds_to_milliseconds(self) -> None:
        assert _MonotonicMilliseconds().next(1.5) == 1500

    def test_ordinary_frame_rates_need_no_adjustment(self) -> None:
        clock = _MonotonicMilliseconds()
        values = [clock.next(i / 30) for i in range(60)]
        assert values == sorted(set(values))
        assert clock.nudges == 0

    def test_240_fps_needs_no_adjustment(self) -> None:
        """4.17 ms apart still rounds to distinct milliseconds."""
        clock = _MonotonicMilliseconds()
        values = [clock.next(i / 240) for i in range(240)]
        assert len(set(values)) == len(values)
        assert clock.nudges == 0

    def test_sub_millisecond_spacing_is_forced_to_advance(self) -> None:
        """2000 fps: 0.5 ms apart, so rounding collides and must be nudged."""
        clock = _MonotonicMilliseconds()
        values = [clock.next(i / 2000) for i in range(20)]
        assert values == sorted(set(values)), "must be strictly increasing"
        assert clock.nudges > 0

    def test_identical_timestamps_still_advance(self) -> None:
        clock = _MonotonicMilliseconds()
        assert clock.next(1.0) == 1000
        assert clock.next(1.0) == 1001
        assert clock.next(1.0) == 1002
        assert clock.nudges == 2

    def test_a_timestamp_going_backwards_still_advances(self) -> None:
        """Out-of-order input is a caller error, but must not wedge MediaPipe."""
        clock = _MonotonicMilliseconds()
        clock.next(5.0)
        assert clock.next(1.0) == 5001

    def test_nudges_start_at_zero(self) -> None:
        assert _MonotonicMilliseconds().nudges == 0


class TestLandmarkMapping:
    def test_maps_every_field(self) -> None:
        points = _to_points([_FakeLandmark(0.25)])
        assert points[0].x == pytest.approx(0.25)
        assert points[0].y == pytest.approx(0.5)
        assert points[0].z == pytest.approx(-0.25)
        assert points[0].visibility == pytest.approx(0.8)
        assert points[0].presence == pytest.approx(0.9)

    def test_missing_confidence_is_zero_not_one(self) -> None:
        """Absent confidence must not be read as total confidence."""
        landmark = _FakeLandmark(0.1)
        landmark.visibility = None  # type: ignore[assignment]
        landmark.presence = None  # type: ignore[assignment]

        point = _to_points([landmark])[0]
        assert point.visibility == 0.0
        assert point.presence == 0.0

    def test_preserves_order(self) -> None:
        points = _to_points([_FakeLandmark(i / 100) for i in range(LANDMARK_COUNT)])
        assert [round(p.x * 100) for p in points] == list(range(LANDMARK_COUNT))

    def test_empty_input_yields_nothing(self) -> None:
        assert _to_points([]) == []


class TestResolveModel:
    def test_defaults_to_the_manifest_choice(self) -> None:
        entry, path = resolve_model()
        assert entry.name == "pose_landmarker_full"
        assert path.name == entry.filename

    def test_an_unknown_name_lists_what_is_available(self) -> None:
        with pytest.raises(PoseEstimationError, match="Available:") as excinfo:
            resolve_model("pose_landmarker_enormous")
        assert excinfo.value.remediation is not None

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("pose_landmarker_lite", "lite"),
            ("pose_landmarker_full", "full"),
            ("pose_landmarker_heavy", "heavy"),
        ],
    )
    def test_variant_is_read_from_the_name(self, name: str, expected: str) -> None:
        assert variant_of(name) == expected


@requires_pose_model
class TestMediaPipeAdapter:
    @pytest.fixture
    def estimator(self):  # type: ignore[no-untyped-def]
        instance = MediaPipePoseEstimator("pose_landmarker_lite")
        yield instance
        instance.close()

    def test_reports_the_model_it_loaded(self, estimator: MediaPipePoseEstimator) -> None:
        info = estimator.info
        assert info.name == "pose_landmarker_lite"
        assert info.variant == "lite"
        assert len(info.sha256) == 64
        assert info.delegate == "cpu"

    def test_records_the_thresholds_it_ran_with(self) -> None:
        """A result is only reproducible if the thresholds are known."""
        with MediaPipePoseEstimator(
            "pose_landmarker_lite", min_pose_detection_confidence=0.7
        ) as estimator:
            assert estimator.info.min_pose_detection_confidence == pytest.approx(0.7)

    def test_converts_bgr_to_rgb_before_inference(self, estimator: MediaPipePoseEstimator) -> None:
        """MediaPipe's SRGB means RGB; frames arrive BGR from the decoder.

        Handing it BGR does not fail -- it degrades detection in a way that
        looks like an ordinary bad result -- so the swap is asserted rather than
        assumed.
        """
        received: list[np.ndarray] = []

        class Spy:
            """Replaces the landmarker, so this asserts our conversion, not MediaPipe's."""

            def detect_for_video(self, image, _timestamp_ms):  # type: ignore[no-untyped-def]
                received.append(image.numpy_view().copy())
                return type("R", (), {"pose_landmarks": [], "pose_world_landmarks": []})()

            def close(self) -> None:
                """The fixture closes the estimator, which closes whatever this is."""

        estimator._landmarker = Spy()  # type: ignore[assignment]

        bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        bgr[:, :, 0] = 10  # blue
        bgr[:, :, 2] = 200  # red
        estimator.estimate(VideoFrame(index=0, timestamp_s=0.0, image=bgr))

        assert received[0][0, 0, 0] == 200, "red must land in channel 0 after conversion"
        assert received[0][0, 0, 2] == 10, "blue must land in channel 2 after conversion"

    def test_a_frame_with_no_pose_is_recorded_not_dropped(
        self, estimator: MediaPipePoseEstimator
    ) -> None:
        """The gap is information the filtering layer needs."""
        blank = np.zeros((120, 160, 3), dtype=np.uint8)
        result = estimator.estimate(VideoFrame(index=3, timestamp_s=0.1, image=blank))

        assert result.detected is False
        assert result.frame_index == 3
        assert result.timestamp_s == pytest.approx(0.1)
        assert result.image == []

    def test_estimating_after_close_is_refused(self) -> None:
        estimator = MediaPipePoseEstimator("pose_landmarker_lite")
        estimator.close()
        blank = np.zeros((64, 64, 3), dtype=np.uint8)

        with pytest.raises(PoseEstimationError, match="closed"):
            estimator.estimate(VideoFrame(index=0, timestamp_s=0.0, image=blank))

    def test_close_is_idempotent(self) -> None:
        estimator = MediaPipePoseEstimator("pose_landmarker_lite")
        estimator.close()
        estimator.close()


@requires_pose_model
@requires_ffmpeg
@pytest.mark.slow
class TestDeterminism:
    """The same clip through the same model must give the same landmarks.

    Not a formality: a stateful tracker plus a non-deterministic timestamp would
    produce results that drift between runs, and every measurement downstream
    would inherit that drift with nothing to attribute it to.
    """

    @staticmethod
    def _run(path: Path) -> list[tuple[int, bool, tuple[float, ...]]]:
        from analyzer.pose.extract import extract_poses

        with MediaPipePoseEstimator("pose_landmarker_lite") as estimator:
            sequence = extract_poses(path, estimator)

        return [
            (
                frame.frame_index,
                frame.detected,
                tuple(round(p.x, 9) for p in frame.image),
            )
            for frame in sequence.frames
        ]

    def test_two_runs_agree_exactly(self) -> None:
        assert self._run(CFR_30FPS) == self._run(CFR_30FPS)
