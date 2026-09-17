"""Extraction orchestration tests.

All of these run against `FakeEstimator` rather than MediaPipe. That is not a
shortcut: it is the evidence that `PoseEstimator` is a real seam, since the
whole path -- decode, estimate, collect statistics, persist -- runs with the
model replaced and nothing else changed.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from analyzer.contracts.pose import Landmark
from analyzer.ingestion.probe import ProbeError
from analyzer.pose.extract import extract_and_store, extract_poses, output_path_for
from analyzer.pose.series import landmark_series
from analyzer.pose.store import read_sequence
from analyzer.progress import RecordingReporter
from tests.conftest import (
    CFR_30FPS,
    CFR_FRAME_COUNT,
    VFR_30_TO_15FPS,
    FakeEstimator,
    requires_ffmpeg,
)


@requires_ffmpeg
class TestExtractPoses:
    def test_visits_every_frame_once_in_order(self) -> None:
        estimator = FakeEstimator()
        extract_poses(CFR_30FPS, estimator)
        assert estimator.seen == list(range(CFR_FRAME_COUNT))

    def test_produces_one_result_per_frame(self) -> None:
        sequence = extract_poses(CFR_30FPS, FakeEstimator())
        assert len(sequence.frames) == CFR_FRAME_COUNT

    def test_timestamps_come_from_the_container_not_the_index(self) -> None:
        """On a variable-rate clip the two differ, and the container wins."""
        sequence = extract_poses(VFR_30_TO_15FPS, FakeEstimator())
        timestamps = [f.timestamp_s for f in sequence.frames]
        intervals = {round(b - a, 4) for a, b in itertools.pairwise(timestamps)}
        assert len(intervals) > 1, "a variable-rate clip must produce variable timestamps"

    def test_records_the_model_that_produced_it(self) -> None:
        sequence = extract_poses(CFR_30FPS, FakeEstimator(name="pretend"))
        assert sequence.model.name == "pretend"

    def test_records_the_source_video_identity(self) -> None:
        sequence = extract_poses(CFR_30FPS, FakeEstimator())
        assert sequence.video_content_key.size_bytes == CFR_30FPS.stat().st_size
        assert Path(sequence.video_path).name == CFR_30FPS.name

    def test_counts_detections(self) -> None:
        sequence = extract_poses(CFR_30FPS, FakeEstimator(detect_every=2))
        assert sequence.stats.frames_processed == CFR_FRAME_COUNT
        assert sequence.stats.frames_detected == CFR_FRAME_COUNT // 2
        assert sequence.stats.detection_rate == pytest.approx(0.5)

    def test_measures_per_frame_cost(self) -> None:
        sequence = extract_poses(CFR_30FPS, FakeEstimator())
        assert sequence.stats.ms_per_frame > 0
        assert sequence.stats.elapsed_s > 0

    def test_mean_visibility_is_none_when_nothing_was_detected(self) -> None:
        """An average over nothing is not zero; it does not exist."""
        sequence = extract_poses(CFR_30FPS, FakeEstimator(detect_every=0))
        assert sequence.stats.frames_detected == 0
        assert sequence.stats.mean_visibility is None

    def test_mean_visibility_is_measured_when_something_was(self) -> None:
        sequence = extract_poses(CFR_30FPS, FakeEstimator())
        assert sequence.stats.mean_visibility == pytest.approx(0.9)

    def test_an_unreadable_video_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ProbeError):
            extract_poses(tmp_path / "absent.mp4", FakeEstimator())


@requires_ffmpeg
class TestProgress:
    def test_reports_from_start_to_finish(self) -> None:
        reporter = RecordingReporter()
        extract_poses(CFR_30FPS, FakeEstimator(), reporter=reporter)

        stages = [e.stage for e in reporter.events]
        assert stages[0] == "starting"
        assert stages[-1] == "done"
        assert "estimating" in stages

    def test_the_final_update_is_complete(self) -> None:
        """A bar left below 100% on finished work reads as a hang."""
        reporter = RecordingReporter()
        extract_poses(CFR_30FPS, FakeEstimator(), reporter=reporter)

        final = reporter.events[-1]
        assert final.current == CFR_FRAME_COUNT
        assert final.total == CFR_FRAME_COUNT
        assert final.fraction == pytest.approx(1.0)

    def test_progress_advances_monotonically(self) -> None:
        reporter = RecordingReporter()
        extract_poses(CFR_30FPS, FakeEstimator(), reporter=reporter)

        counts = [e.current for e in reporter.events]
        assert counts == sorted(counts)

    def test_the_total_is_known_before_any_frame_is_decoded(self) -> None:
        """It comes from the container index, so the bar is determinate from the start."""
        reporter = RecordingReporter()
        extract_poses(CFR_30FPS, FakeEstimator(), reporter=reporter)
        assert reporter.events[0].total == CFR_FRAME_COUNT

    def test_the_request_id_is_carried(self) -> None:
        reporter = RecordingReporter()
        extract_poses(CFR_30FPS, FakeEstimator(), reporter=reporter, request_id=42)
        assert all(e.request_id == 42 for e in reporter.events)

    def test_progress_is_optional(self) -> None:
        assert extract_poses(CFR_30FPS, FakeEstimator()) is not None


@requires_ffmpeg
class TestExtractAndStore:
    def test_writes_a_readable_file(self, tmp_path: Path) -> None:
        result = extract_and_store(CFR_30FPS, FakeEstimator(), output=tmp_path / "poses.parquet")
        restored = read_sequence(Path(result.output_path))
        assert len(restored.frames) == CFR_FRAME_COUNT

    def test_the_stored_landmarks_survive_the_whole_path(self, tmp_path: Path) -> None:
        """Decode, estimate, persist, reload, transpose -- end to end."""
        result = extract_and_store(CFR_30FPS, FakeEstimator(), output=tmp_path / "poses.parquet")
        series = landmark_series(read_sequence(Path(result.output_path)), Landmark.LEFT_WRIST)

        assert len(series) == CFR_FRAME_COUNT
        assert series.observed_fraction == pytest.approx(1.0)
        assert series.x[0] == pytest.approx(int(Landmark.LEFT_WRIST) / 100, abs=1e-6)

    def test_the_result_excludes_the_landmarks_themselves(self, tmp_path: Path) -> None:
        """Tens of thousands of frames belong in a file, not a JSON-RPC reply."""
        result = extract_and_store(CFR_30FPS, FakeEstimator(), output=tmp_path / "poses.parquet")
        assert "frames" not in result.model_dump()
        assert result.output_path.endswith("poses.parquet")

    def test_warns_when_nothing_was_detected(self, tmp_path: Path) -> None:
        result = extract_and_store(
            CFR_30FPS, FakeEstimator(detect_every=0), output=tmp_path / "poses.parquet"
        )
        assert any("No pose was detected" in w for w in result.warnings)

    def test_warns_when_detection_was_sparse(self, tmp_path: Path) -> None:
        result = extract_and_store(
            CFR_30FPS, FakeEstimator(detect_every=4), output=tmp_path / "poses.parquet"
        )
        assert any("only" in w for w in result.warnings)

    def test_a_fully_detected_clip_produces_no_warnings(self, tmp_path: Path) -> None:
        result = extract_and_store(CFR_30FPS, FakeEstimator(), output=tmp_path / "poses.parquet")
        assert result.warnings == []

    def test_the_default_path_is_keyed_by_video_content(self, isolated_cache: Path) -> None:
        """Re-extracting a clip overwrites its own result; a different clip never collides."""
        sequence = extract_poses(CFR_30FPS, FakeEstimator())
        other = extract_poses(VFR_30_TO_15FPS, FakeEstimator())

        assert output_path_for(sequence) != output_path_for(other)
        assert isolated_cache in output_path_for(sequence).parents

    def test_the_default_path_names_the_model(self, isolated_cache: Path) -> None:
        """Two models over one clip are two results, both worth keeping."""
        sequence = extract_poses(CFR_30FPS, FakeEstimator(name="alpha"))
        assert output_path_for(sequence).name == "alpha.parquet"
