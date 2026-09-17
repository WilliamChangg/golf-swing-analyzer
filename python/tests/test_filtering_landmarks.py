"""Filtering a whole pose sequence.

The layer Phase 4 will consume, so these tests are mostly about the shape and
honesty of what it hands over: which frames carry a usable position, what the
numbers are in, and whether the report explains the difference between the frame
count and the count of frames that survived.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.filtering import (
    ConfidenceGate,
    FilterConfig,
    GapPolicy,
    SignalUnit,
    SmoothingConfig,
    unit_for,
)
from analyzer.contracts.pose import (
    LANDMARK_COUNT,
    Landmark,
    LandmarkPoint,
    LandmarkSpace,
    PoseExtractionStats,
    PoseFrame,
    PoseModelInfo,
    PoseSequence,
)
from analyzer.filtering.landmarks import filter_landmark, filter_sequence, signals_from_series
from analyzer.pose.series import landmark_series
from analyzer.progress import RecordingReporter
from tests.conftest import SQUARE_FRAME

FPS = 120.0


def _points(seed: float, visibility: float = 0.9) -> list[LandmarkPoint]:
    """Landmarks on a smooth circular path, so derivatives have a known character."""
    return [
        LandmarkPoint(
            x=0.5 + 0.2 * np.sin(2 * np.pi * 2.0 * seed),
            y=0.5 + 0.2 * np.cos(2 * np.pi * 2.0 * seed),
            z=0.01 * index,
            visibility=visibility,
            presence=0.99,
        )
        for index in range(LANDMARK_COUNT)
    ]


def _sequence(
    count: int = 240,
    *,
    undetected: range | None = None,
    low_confidence: range | None = None,
    fps: float = FPS,
) -> PoseSequence:
    frames = []
    for index in range(count):
        detected = undetected is None or index not in undetected
        visibility = 0.1 if low_confidence and index in low_confidence else 0.9
        points = _points(index / fps, visibility) if detected else []
        frames.append(
            PoseFrame(
                frame_index=index,
                timestamp_s=index / fps,
                detected=detected,
                image=points,
                hip_local=points,
            )
        )

    found = sum(1 for frame in frames if frame.detected)
    return PoseSequence(
        video_path="/data/swing.mov",
        video_content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED, digest="d" * 64, size_bytes=1
        ),
        geometry=SQUARE_FRAME,
        model=PoseModelInfo(
            name="fake",
            variant="fake",
            precision="float32",
            sha256="0" * 64,
            delegate="cpu",
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ),
        extracted_at=datetime(2026, 9, 16, tzinfo=UTC),
        stats=PoseExtractionStats(
            frames_processed=count,
            frames_detected=found,
            detection_rate=found / count if count else 0.0,
            elapsed_s=1.0,
            ms_per_frame=1.0,
        ),
        frames=frames,
    )


class TestFilterLandmark:
    def test_produces_position_velocity_and_acceleration_per_axis(self) -> None:
        series = landmark_series(_sequence(), Landmark.LEFT_WRIST)
        filtered = filter_landmark(series)

        assert filtered.position.shape == (240, 3)
        assert filtered.velocity.shape == (240, 3)
        assert filtered.acceleration.shape == (240, 3)
        assert np.all(filtered.valid)

    def test_recovers_the_velocity_of_a_known_circular_path(self) -> None:
        """The landmarks trace a circle of radius 0.2 at 2 Hz, so the speed is
        2*pi*2*0.2 everywhere -- a closed-form value to check against."""
        series = landmark_series(_sequence(), Landmark.LEFT_WRIST)
        filtered = filter_landmark(series)

        expected = 2 * np.pi * 2.0 * 0.2
        interior = filtered.speed[20:-20]
        assert np.allclose(interior, expected, rtol=0.02)

    def test_a_frame_needs_all_three_axes_to_count_as_valid(self) -> None:
        """A partial position is not a position: any vector built from one
        would have a silently wrong magnitude."""
        series = landmark_series(_sequence(), Landmark.LEFT_WRIST)
        filtered = filter_landmark(series)
        assert np.array_equal(filtered.valid, np.all(np.isfinite(filtered.position), axis=1))

    def test_the_three_axes_share_gating_and_gap_decisions(self) -> None:
        """Not arranged between the stages -- it follows from x, y and z being
        present or absent together. Worth pinning, because a stage added later
        that does not share it would break the assumption silently.
        """
        series = landmark_series(_sequence(undetected=range(60, 70)), Landmark.LEFT_WRIST)
        filtered = filter_landmark(series)

        finite = np.isfinite(filtered.position)
        assert np.array_equal(finite[:, 0], finite[:, 1])
        assert np.array_equal(finite[:, 0], finite[:, 2])

    def test_reports_units_matching_the_coordinate_space(self) -> None:
        image = filter_landmark(landmark_series(_sequence(), Landmark.LEFT_WRIST))
        assert image.report.position_unit is SignalUnit.NORMALIZED_FRAME
        assert image.report.velocity_unit is SignalUnit.NORMALIZED_FRAME_PER_S

        hip = filter_landmark(
            landmark_series(_sequence(), Landmark.LEFT_WRIST, LandmarkSpace.HIP_LOCAL)
        )
        assert hip.report.position_unit is SignalUnit.APPROX_M
        assert hip.report.acceleration_unit is SignalUnit.APPROX_M_PER_S2

    def test_accounts_for_every_frame_that_did_not_survive(self) -> None:
        """The frame count minus the valid count must be explainable."""
        series = landmark_series(
            _sequence(undetected=range(60, 70), low_confidence=range(100, 106)),
            Landmark.LEFT_WRIST,
        )
        report = filter_landmark(series).report

        assert report.samples == 240
        assert report.never_detected == 10
        assert report.gated_out == 6
        assert report.observed == 240 - 10 - 6
        assert report.valid_samples + report.blocked + report.unsupported == report.samples

    def test_records_the_longest_absence(self) -> None:
        series = landmark_series(_sequence(undetected=range(60, 70)), Landmark.LEFT_WRIST)
        report = filter_landmark(series).report
        # Eleven intervals span the ten missing frames at 120 fps.
        assert report.longest_gap_s == pytest.approx(11 / FPS)

    def test_reports_a_residual_per_axis(self) -> None:
        series = landmark_series(_sequence(), Landmark.LEFT_WRIST)
        report = filter_landmark(series).report
        assert len(report.residual_rms) == 3
        # The synthetic path is noiseless, so the fit should sit on it.
        assert all(value < 1e-4 for value in report.residual_rms)

    def test_carries_the_stage_reports_for_provenance(self) -> None:
        series = landmark_series(_sequence(), Landmark.LEFT_WRIST)
        report = filter_landmark(series).report
        assert [stage.stage for stage in report.stages] == [
            "confidence_gate",
            "gap_policy",
            "local_polynomial",
        ]

    def test_signals_from_series_names_its_channels(self) -> None:
        series = landmark_series(_sequence(), Landmark.LEFT_WRIST)
        signals = signals_from_series(series)
        assert set(signals) == {"x", "y", "z"}
        assert signals["x"].label == "left_wrist.x[image]"


class TestFilterSequence:
    def test_covers_every_landmark(self) -> None:
        result = filter_sequence(_sequence(count=120))
        assert set(result.landmarks) == set(Landmark)
        assert len(result.report.landmarks) == LANDMARK_COUNT

    def test_can_be_restricted_to_the_landmarks_a_caller_needs(self) -> None:
        wanted = (Landmark.LEFT_WRIST, Landmark.RIGHT_WRIST)
        result = filter_sequence(_sequence(count=120), landmarks=wanted)
        assert set(result.landmarks) == set(wanted)

    def test_warns_when_gaps_were_refused_rather_than_filled(self) -> None:
        result = filter_sequence(_sequence(count=240, undetected=range(60, 100)))
        assert any("longer than" in warning for warning in result.report.warnings)

    def test_warns_when_nothing_survived(self) -> None:
        result = filter_sequence(_sequence(count=120, low_confidence=range(0, 120)))
        assert any("No landmark produced" in warning for warning in result.report.warnings)
        assert result.report.mean_valid_fraction == 0.0

    def test_is_quiet_on_a_clean_clip(self) -> None:
        result = filter_sequence(_sequence(count=240))
        assert result.report.warnings == []
        assert result.report.mean_valid_fraction == 1.0

    def test_reports_the_config_it_ran_with(self) -> None:
        """A filtered signal is only interpretable against the policy that made it."""
        config = FilterConfig(gaps=GapPolicy(max_gap_s=0.2))
        result = filter_sequence(_sequence(count=120), config)
        assert result.report.config.gaps.max_gap_s == 0.2

    def test_reports_progress_through_the_landmarks(self) -> None:
        reporter = RecordingReporter()
        filter_sequence(_sequence(count=120), reporter=reporter)

        stages = [event.stage for event in reporter.events]
        assert stages[0] == "starting"
        assert stages[-1] == "done"
        assert reporter.events[-1].current == LANDMARK_COUNT

    def test_an_empty_sequence_produces_no_landmark_data(self) -> None:
        result = filter_sequence(_sequence(count=0))
        assert result.t.size == 0
        assert all(len(entry) == 0 for entry in result.landmarks.values())

    def test_indexing_returns_a_landmark(self) -> None:
        result = filter_sequence(_sequence(count=120), landmarks=(Landmark.NOSE,))
        assert result[Landmark.NOSE].landmark is Landmark.NOSE


class TestLowFrameRate:
    def test_refuses_and_explains_when_the_frame_rate_cannot_support_the_window(self) -> None:
        """The measured cliff, stated rather than silently degraded.

        At 24 fps a 0.10 s window holds three samples, which cannot determine a
        degree-4 polynomial. The right answer is no values plus a reason, not a
        quietly lowered order.
        """
        result = filter_sequence(_sequence(count=120, fps=24.0))

        report = result.report.landmarks[0]
        assert report.valid_samples == 0
        assert report.unsupported == report.samples
        assert any("frame rate" in note for note in report.notes)
        assert any("frame rate" in warning for warning in result.report.warnings)

    def test_a_wider_window_makes_the_same_clip_usable(self) -> None:
        config = FilterConfig(smoothing=SmoothingConfig(window_s=0.30, polyorder=4))
        result = filter_sequence(_sequence(count=120, fps=24.0), config)
        assert result.report.landmarks[0].valid_samples > 0


class TestUnits:
    def test_rejects_a_derivative_order_it_has_no_unit_for(self) -> None:
        with pytest.raises(ValueError, match="derivative order 3"):
            unit_for(LandmarkSpace.IMAGE, 3)

    @pytest.mark.parametrize("space", list(LandmarkSpace))
    def test_every_space_has_a_unit_for_every_order(self, space: LandmarkSpace) -> None:
        assert len({unit_for(space, order) for order in (0, 1, 2)}) == 3


class TestConfigValidation:
    def test_rejects_min_observations_that_cannot_determine_the_fit(self) -> None:
        with pytest.raises(ValueError, match="at least 5"):
            SmoothingConfig(polyorder=4, min_observations=4)

    def test_min_observations_defaults_to_the_mathematical_minimum(self) -> None:
        assert SmoothingConfig(polyorder=4).required_observations == 5

    def test_rejects_a_confidence_threshold_outside_zero_to_one(self) -> None:
        with pytest.raises(ValueError):
            ConfidenceGate(min_visibility=1.5)

    def test_rejects_a_non_positive_gap_limit(self) -> None:
        with pytest.raises(ValueError):
            GapPolicy(max_gap_s=0.0)
