"""Pose persistence tests.

A stored sequence is the handover point between this phase and every later one,
so the question these ask is not "does it write a file" but "does what comes
back mean the same thing as what went in" -- including the parts that are
absent, which is where a round-trip most easily lies.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.pose import (
    LANDMARK_COUNT,
    POSE_SCHEMA_VERSION,
    Landmark,
    LandmarkSpace,
    PoseExtractionStats,
    PoseFrame,
    PoseModelInfo,
    PoseSequence,
)
from analyzer.pose.store import PoseStoreError, read_sequence, write_sequence
from tests.conftest import SQUARE_FRAME, landmark_points


def _model() -> PoseModelInfo:
    return PoseModelInfo(
        name="pose_landmarker_full",
        variant="full",
        precision="float16",
        sha256="a" * 64,
        delegate="cpu",
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.7,
    )


def _sequence(frames: list[PoseFrame]) -> PoseSequence:
    detected = sum(1 for f in frames if f.detected)
    return PoseSequence(
        video_path="/data/raw/swing.mov",
        video_content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED, digest="b" * 64, size_bytes=1234
        ),
        geometry=SQUARE_FRAME,
        model=_model(),
        extracted_at=datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
        stats=PoseExtractionStats(
            frames_processed=len(frames),
            frames_detected=detected,
            detection_rate=detected / len(frames) if frames else 0.0,
            elapsed_s=1.5,
            ms_per_frame=25.0,
            mean_visibility=0.9 if detected else None,
        ),
        frames=frames,
    )


def _detected(index: int) -> PoseFrame:
    points = landmark_points(seed=index / 1000)
    return PoseFrame(
        frame_index=index,
        timestamp_s=index / 30,
        detected=True,
        image=points,
        hip_local=points,
    )


def _missed(index: int) -> PoseFrame:
    return PoseFrame(frame_index=index, timestamp_s=index / 30, detected=False)


def _round_trip(sequence: PoseSequence, tmp_path: Path) -> PoseSequence:
    target = write_sequence(sequence, tmp_path / "poses.parquet", landmark_count=LANDMARK_COUNT)
    return read_sequence(target)


class TestRoundTrip:
    def test_preserves_every_frame_in_order(self, tmp_path: Path) -> None:
        original = _sequence([_detected(i) for i in range(5)])
        restored = _round_trip(original, tmp_path)
        assert [f.frame_index for f in restored.frames] == [0, 1, 2, 3, 4]

    def test_preserves_landmark_values(self, tmp_path: Path) -> None:
        original = _sequence([_detected(7)])
        restored = _round_trip(original, tmp_path)

        before = original.frames[0].image[Landmark.LEFT_WRIST]
        after = restored.frames[0].image[Landmark.LEFT_WRIST]
        # float32 on disk, so exact equality would be testing the wrong thing.
        assert after.x == pytest.approx(before.x, abs=1e-6)
        assert after.y == pytest.approx(before.y, abs=1e-6)
        assert after.z == pytest.approx(before.z, abs=1e-6)
        assert after.visibility == pytest.approx(before.visibility, abs=1e-6)

    def test_preserves_both_coordinate_spaces(self, tmp_path: Path) -> None:
        restored = _round_trip(_sequence([_detected(0)]), tmp_path)
        assert len(restored.frames[0].image) == LANDMARK_COUNT
        assert len(restored.frames[0].hip_local) == LANDMARK_COUNT

    def test_preserves_timestamps(self, tmp_path: Path) -> None:
        original = _sequence([_detected(0), _detected(1), _detected(2)])
        restored = _round_trip(original, tmp_path)
        assert [f.timestamp_s for f in restored.frames] == [f.timestamp_s for f in original.frames]

    def test_preserves_the_model_it_was_produced_with(self, tmp_path: Path) -> None:
        """A result that cannot be traced to an artifact cannot be reproduced."""
        restored = _round_trip(_sequence([_detected(0)]), tmp_path)
        assert restored.model == _model()

    def test_preserves_the_source_video_identity(self, tmp_path: Path) -> None:
        original = _sequence([_detected(0)])
        restored = _round_trip(original, tmp_path)
        assert restored.video_content_key == original.video_content_key
        assert restored.video_path == original.video_path

    def test_preserves_run_statistics(self, tmp_path: Path) -> None:
        original = _sequence([_detected(0), _missed(1)])
        restored = _round_trip(original, tmp_path)
        assert restored.stats == original.stats

    def test_preserves_extraction_time(self, tmp_path: Path) -> None:
        original = _sequence([_detected(0)])
        assert _round_trip(original, tmp_path).extracted_at == original.extracted_at


class TestUndetectedFrames:
    """A gap must survive the round trip as a gap, not as a zero or a deletion."""

    def test_an_undetected_frame_is_kept(self, tmp_path: Path) -> None:
        original = _sequence([_detected(0), _missed(1), _detected(2)])
        restored = _round_trip(original, tmp_path)
        assert [f.frame_index for f in restored.frames] == [0, 1, 2]

    def test_an_undetected_frame_keeps_its_flag_and_timestamp(self, tmp_path: Path) -> None:
        restored = _round_trip(_sequence([_missed(4)]), tmp_path)
        assert restored.frames[0].detected is False
        assert restored.frames[0].timestamp_s == pytest.approx(4 / 30)

    def test_an_undetected_frame_carries_no_landmarks(self, tmp_path: Path) -> None:
        """Not zeros: a landmark at the origin and a missing landmark differ."""
        restored = _round_trip(_sequence([_missed(0)]), tmp_path)
        assert restored.frames[0].image == []
        assert restored.frames[0].hip_local == []

    def test_a_sequence_with_no_detections_at_all_round_trips(self, tmp_path: Path) -> None:
        restored = _round_trip(_sequence([_missed(i) for i in range(3)]), tmp_path)
        assert len(restored.frames) == 3
        assert not any(f.detected for f in restored.frames)


class TestLongFormat:
    def test_writes_one_row_per_frame_space_and_landmark(self, tmp_path: Path) -> None:
        target = write_sequence(
            _sequence([_detected(0), _missed(1)]),
            tmp_path / "p.parquet",
            landmark_count=LANDMARK_COUNT,
        )
        table = pq.read_table(target)
        assert table.num_rows == 2 * 2 * LANDMARK_COUNT

    def test_both_spaces_are_labelled(self, tmp_path: Path) -> None:
        target = write_sequence(
            _sequence([_detected(0)]), tmp_path / "p.parquet", landmark_count=LANDMARK_COUNT
        )
        spaces = set(pq.read_table(target).to_pydict()["space"])
        assert spaces == {LandmarkSpace.IMAGE.value, LandmarkSpace.HIP_LOCAL.value}

    def test_missing_values_are_null_not_zero(self, tmp_path: Path) -> None:
        """Phase 3 needs gaps as NaN so it can refuse to interpolate across them."""
        target = write_sequence(
            _sequence([_missed(0)]), tmp_path / "p.parquet", landmark_count=LANDMARK_COUNT
        )
        assert all(x is None for x in pq.read_table(target).to_pydict()["x"])

    def test_the_file_is_self_describing(self, tmp_path: Path) -> None:
        target = write_sequence(
            _sequence([_detected(0)]), tmp_path / "p.parquet", landmark_count=LANDMARK_COUNT
        )
        metadata = pq.read_table(target).schema.metadata or {}
        assert metadata[b"gsa.schema_version"] == str(POSE_SCHEMA_VERSION).encode()
        assert b"pose_landmarker_full" in metadata[b"gsa.model"]


class TestReadRefuses:
    def test_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(PoseStoreError, match="No pose file"):
            read_sequence(tmp_path / "absent.parquet")

    def test_a_file_that_is_not_parquet(self, tmp_path: Path) -> None:
        target = tmp_path / "junk.parquet"
        target.write_bytes(b"definitely not parquet")
        with pytest.raises(PoseStoreError):
            read_sequence(target)

    def test_a_parquet_file_written_by_something_else(self, tmp_path: Path) -> None:
        """No metadata means no way to know what the numbers mean."""
        import pyarrow as pa

        target = tmp_path / "foreign.parquet"
        pq.write_table(pa.table({"x": [1, 2, 3]}), target)
        with pytest.raises(PoseStoreError, match="metadata key"):
            read_sequence(target)

    def test_a_newer_schema_version(self, tmp_path: Path) -> None:
        """Refused, not coerced: a field may have changed meaning."""
        target = write_sequence(
            _sequence([_detected(0)]), tmp_path / "p.parquet", landmark_count=LANDMARK_COUNT
        )
        table = pq.read_table(target)
        metadata = dict(table.schema.metadata or {})
        metadata[b"gsa.schema_version"] = str(POSE_SCHEMA_VERSION + 1).encode()
        pq.write_table(table.replace_schema_metadata(metadata), target)

        with pytest.raises(PoseStoreError, match="schema version"):
            read_sequence(target)

    def test_a_non_numeric_schema_version(self, tmp_path: Path) -> None:
        target = write_sequence(
            _sequence([_detected(0)]), tmp_path / "p.parquet", landmark_count=LANDMARK_COUNT
        )
        table = pq.read_table(target)
        metadata = dict(table.schema.metadata or {})
        metadata[b"gsa.schema_version"] = b"one"
        pq.write_table(table.replace_schema_metadata(metadata), target)

        with pytest.raises(PoseStoreError, match="non-numeric"):
            read_sequence(target)


class TestAtomicWrite:
    def test_leaves_no_temporary_file_behind(self, tmp_path: Path) -> None:
        write_sequence(
            _sequence([_detected(0)]), tmp_path / "p.parquet", landmark_count=LANDMARK_COUNT
        )
        assert list(tmp_path.glob("*.tmp")) == []

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "p.parquet"
        write_sequence(_sequence([_detected(0)]), target, landmark_count=LANDMARK_COUNT)
        assert target.exists()
