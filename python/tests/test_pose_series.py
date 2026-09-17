"""Landmark series tests.

The transpose from frame-major to landmark-major is where a missing detection
most easily turns into a plausible number. Every test here is really asking the
same question: does absence survive?
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.pose import (
    Landmark,
    LandmarkSpace,
    PoseExtractionStats,
    PoseFrame,
    PoseModelInfo,
    PoseSequence,
)
from analyzer.pose.series import all_series, landmark_series
from tests.conftest import SQUARE_FRAME, landmark_points


def _sequence(detected_flags: list[bool]) -> PoseSequence:
    frames = []
    for index, detected in enumerate(detected_flags):
        points = landmark_points(seed=index / 1000) if detected else []
        frames.append(
            PoseFrame(
                frame_index=index,
                timestamp_s=index / 30,
                detected=detected,
                image=points,
                hip_local=points,
            )
        )

    count = sum(detected_flags)
    return PoseSequence(
        video_path="/data/swing.mov",
        video_content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED, digest="c" * 64, size_bytes=1
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
            frames_processed=len(frames),
            frames_detected=count,
            detection_rate=count / len(frames) if frames else 0.0,
            elapsed_s=1.0,
            ms_per_frame=10.0,
        ),
        frames=frames,
    )


class TestLandmarkSeries:
    def test_length_matches_the_frame_count(self) -> None:
        series = landmark_series(_sequence([True] * 7), Landmark.NOSE)
        assert len(series) == 7

    def test_timestamps_come_from_the_frames(self) -> None:
        series = landmark_series(_sequence([True, True, True]), Landmark.NOSE)
        assert series.timestamps_s == pytest.approx([0.0, 1 / 30, 2 / 30])

    def test_timestamps_are_present_even_where_nothing_was_detected(self) -> None:
        """The timeline is continuous whether or not the subject was found on it."""
        series = landmark_series(_sequence([False, False]), Landmark.NOSE)
        assert series.timestamps_s == pytest.approx([0.0, 1 / 30])

    def test_picks_out_the_requested_landmark(self) -> None:
        """Landmark values encode their index, so the wrong one is detectable."""
        series = landmark_series(_sequence([True]), Landmark.LEFT_WRIST)
        assert series.x[0] == pytest.approx(int(Landmark.LEFT_WRIST) / 100)

    def test_undetected_frames_are_nan_not_zero(self) -> None:
        series = landmark_series(_sequence([True, False, True]), Landmark.NOSE)
        assert not np.isnan(series.x[0])
        assert np.isnan(series.x[1])
        assert not np.isnan(series.x[2])

    def test_every_channel_is_nan_where_undetected(self) -> None:
        series = landmark_series(_sequence([False]), Landmark.NOSE)
        for channel in (series.x, series.y, series.z, series.visibility, series.presence):
            assert np.isnan(channel[0])

    def test_observed_marks_exactly_the_detected_frames(self) -> None:
        series = landmark_series(_sequence([True, False, True, True]), Landmark.NOSE)
        assert series.observed.tolist() == [True, False, True, True]

    def test_observed_fraction_is_measured(self) -> None:
        series = landmark_series(_sequence([True, False, True, False]), Landmark.NOSE)
        assert series.observed_fraction == pytest.approx(0.5)

    def test_observed_fraction_of_an_empty_sequence_is_zero(self) -> None:
        series = landmark_series(_sequence([]), Landmark.NOSE)
        assert series.observed_fraction == 0.0
        assert len(series) == 0

    def test_records_which_landmark_and_space_it_is(self) -> None:
        series = landmark_series(_sequence([True]), Landmark.RIGHT_KNEE, LandmarkSpace.HIP_LOCAL)
        assert series.landmark is Landmark.RIGHT_KNEE
        assert series.space is LandmarkSpace.HIP_LOCAL

    def test_arrays_are_float64_regardless_of_stored_precision(self) -> None:
        """Downstream filtering accumulates; float32 would lose ground doing it."""
        series = landmark_series(_sequence([True]), Landmark.NOSE)
        assert series.x.dtype == np.float64


class TestAllSeries:
    def test_covers_every_landmark(self) -> None:
        series = all_series(_sequence([True, True]))
        assert set(series) == set(Landmark)
        assert len(series) == 33

    def test_each_entry_is_keyed_by_its_own_landmark(self) -> None:
        for landmark, series in all_series(_sequence([True])).items():
            assert series.landmark is landmark

    def test_respects_the_requested_space(self) -> None:
        series = all_series(_sequence([True]), LandmarkSpace.HIP_LOCAL)
        assert all(s.space is LandmarkSpace.HIP_LOCAL for s in series.values())
