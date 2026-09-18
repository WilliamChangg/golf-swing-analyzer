"""Dispatch table tests.

The dispatch layer is the single place where arbitrary failures become typed RPC
errors. If that normalisation leaks, the desktop app sees a hung worker instead
of an error it can render.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from analyzer import dispatch
from analyzer.contracts.filtering import SequenceFilterReport
from analyzer.contracts.phases import SwingPhases
from analyzer.contracts.pose import PoseExtractionResult
from analyzer.contracts.rpc import EngineError, ErrorCode
from analyzer.contracts.video import VideoMetadata
from analyzer.dispatch import ProbeVideoParams
from analyzer.progress import ProgressReporter, ProgressTracker, RecordingReporter
from tests.conftest import CFR_30FPS, CFR_FRAME_COUNT, SQUARE_FRAME, requires_ffprobe

# Parameters good enough to invoke each registered method once. Maintained by
# hand on purpose: the table is what makes a newly-registered method visible.
_METHOD_PARAMS: dict[str, dict[str, object]] = {
    "doctor": {},
    "probe_video": {"path": str(CFR_30FPS)},
    "extract_poses": {"path": str(CFR_30FPS)},
    "filter_poses": {"path": "poses.parquet"},
    "detect_phases": {"path": "poses.parquet"},
    "compute_metrics": {"path": "poses.parquet"},
    "coach_swing": {"path": "poses.parquet"},
    "sync_clips": {
        "reference": {"path": "a.parquet"},
        "target": {"path": "b.parquet"},
    },
    "create_project": {"name": "Session"},
    "list_projects": {},
    "get_project": {"project_id": 1},
    "delete_project": {"project_id": 1},
    "add_clip": {"project_id": 1, "path": str(CFR_30FPS), "role": "face_on"},
    "remove_clip": {"project_id": 1, "clip_id": 1},
    "relocate_clip": {"project_id": 1, "clip_id": 1, "path": str(CFR_30FPS)},
    "sync_project": {"project_id": 1},
    "calibrate_camera": {"source": "board.mov"},
    "calibrate_stereo": {
        "project_id": 1,
        "reference_source": "a.mov",
        "target_source": "b.mov",
        "reference_role": "face_on",
        "target_role": "down_the_line",
    },
    "get_calibration": {"project_id": 1},
    "clear_calibration": {"project_id": 1},
    "reconstruct": {"project_id": 1},
    "track_club": {"path": str(CFR_30FPS)},
    "detect_ball": {"path": str(CFR_30FPS)},
    "locate_impact": {"path": str(CFR_30FPS)},
}

# Everything that runs in milliseconds. `extract_poses` loads a model and
# decodes a clip, so it is exercised separately under the slow markers.
#
# The project methods qualify because they only touch SQLite -- and they are
# only safe to call here because `conftest.isolated_data` redirects the database
# into `tmp_path`. Without that this table would create projects in the
# developer's real data directory on every run.
_FAST_METHODS = {"doctor", "probe_video", "create_project", "list_projects"}

# `calibrate_camera` and `calibrate_stereo` are absent from the fast set and from
# the slow one: both need board footage, which this repository does not contain,
# and a stub would exercise the dispatch wiring against a fixture rather than
# against the engine. `tests/test_calibration.py` covers the path they call into,
# from rendered board views, which is the stronger test of the two.
#
# `reconstruct` is absent for the same reason and one more: it needs a project
# holding two calibrated clips *and* an alignment between them, which is three
# pieces of state that only exist together on a real session.
# `tests/test_reconstruction.py` drives the engine it calls into directly, from a
# synthetic body whose 3D positions are inputs.


class TestCall:
    def test_unknown_method_raises_method_not_found(self) -> None:
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("no_such_method")
        assert excinfo.value.code == ErrorCode.METHOD_NOT_FOUND

    def test_unknown_method_message_lists_known_methods(self) -> None:
        """The error should be actionable, e.g. after a version skew."""
        with pytest.raises(EngineError, match="doctor"):
            dispatch.call("no_such_method")

    def test_doctor_rejects_unexpected_params(self) -> None:
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("doctor", {"unexpected": 1})
        assert excinfo.value.code == ErrorCode.INVALID_PARAMS

    def test_none_params_is_treated_as_empty(self) -> None:
        assert isinstance(dispatch.call("doctor", None), BaseModel)

    def test_arbitrary_exception_becomes_internal_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A handler blowing up must surface as a structured error, not propagate."""

        def exploding(_params: dict[str, object], _reporter: ProgressReporter) -> BaseModel:
            raise ZeroDivisionError("boom")

        monkeypatch.setitem(dispatch.METHODS, "explode", exploding)

        with pytest.raises(EngineError) as excinfo:
            dispatch.call("explode")
        assert excinfo.value.code == ErrorCode.INTERNAL_ERROR
        assert "ZeroDivisionError" in str(excinfo.value)

    def test_engine_error_from_handler_is_not_rewrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A handler's own typed error must keep its code."""

        def picky(_params: dict[str, object], _reporter: ProgressReporter) -> BaseModel:
            raise EngineError("bad input", code=ErrorCode.UNSUPPORTED_INPUT)

        monkeypatch.setitem(dispatch.METHODS, "picky", picky)

        with pytest.raises(EngineError) as excinfo:
            dispatch.call("picky")
        assert excinfo.value.code == ErrorCode.UNSUPPORTED_INPUT

    def test_every_registered_method_is_covered_by_the_table(self) -> None:
        """A new method with no entry fails here rather than going unexercised.

        Kept separate from the call below so that adding a method is caught even
        when the suite is run without the markers the expensive ones need.
        """
        assert set(_METHOD_PARAMS) == set(dispatch.METHODS), "add the new method to _METHOD_PARAMS"

    @pytest.mark.parametrize("name", sorted(_FAST_METHODS))
    def test_fast_methods_return_a_contract_model(self, name: str) -> None:
        """Guards the invariant the worker relies on when calling model_dump."""
        assert isinstance(dispatch.call(name, _METHOD_PARAMS[name]), BaseModel)

    @pytest.mark.slow
    @pytest.mark.requires_model
    @requires_ffprobe
    def test_extract_poses_returns_a_contract_model(self) -> None:
        """Separated because it loads a model and decodes a clip: seconds, not milliseconds."""
        result = dispatch.call("extract_poses", _METHOD_PARAMS["extract_poses"])
        assert isinstance(result, PoseExtractionResult)
        assert result.stats.frames_processed == CFR_FRAME_COUNT

    def test_progress_is_reported_for_a_long_method(self) -> None:
        """A method that takes seconds must say so while it is taking them."""
        recorder = RecordingReporter()

        def slow(_params: dict[str, object], reporter: ProgressReporter) -> BaseModel:
            ProgressTracker(reporter, task="slow").report("working", 1, 2)
            return ProbeVideoParams(path="x")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setitem(dispatch.METHODS, "slow", slow)
        try:
            dispatch.call("slow", {}, recorder)
        finally:
            monkeypatch.undo()

        assert [e.stage for e in recorder.events] == ["working"]

    def test_methods_that_ignore_progress_still_work(self) -> None:
        """The reporter is always supplied, so a method never has to check for it."""
        assert isinstance(dispatch.call("doctor", {}, RecordingReporter()), BaseModel)


class TestProbeVideoParams:
    """The engine validates its own parameters; the Rust layer forwards them verbatim."""

    def test_missing_path_is_invalid_params(self) -> None:
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("probe_video", {})
        assert excinfo.value.code == ErrorCode.INVALID_PARAMS

    def test_unknown_parameter_is_rejected_rather_than_ignored(self) -> None:
        """A stale or misspelled name must not silently fall back to a default."""
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("probe_video", {"path": str(CFR_30FPS), "refesh": True})
        assert excinfo.value.code == ErrorCode.INVALID_PARAMS

    def test_wrong_type_is_invalid_params(self) -> None:
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("probe_video", {"path": 42})
        assert excinfo.value.code == ErrorCode.INVALID_PARAMS

    def test_an_unusable_file_is_unsupported_input_not_an_internal_error(
        self, tmp_path: Path
    ) -> None:
        """The user picked the wrong file; that is not a crash and must not read as one."""
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("probe_video", {"path": str(tmp_path / "absent.mp4")})
        assert excinfo.value.code == ErrorCode.UNSUPPORTED_INPUT

    def test_the_remedy_is_carried_to_the_caller(self, tmp_path: Path) -> None:
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("probe_video", {"path": str(tmp_path / "absent.mp4")})
        assert (excinfo.value.data or {}).get("remediation")

    @requires_ffprobe
    def test_returns_metadata_for_a_real_file(self) -> None:
        result = dispatch.call("probe_video", {"path": str(CFR_30FPS)})
        assert isinstance(result, VideoMetadata)
        assert result.timing.frame_count == CFR_FRAME_COUNT


class TestFilterPoses:
    """`filter_poses` reads landmarks off disk, so its failure modes are about
    finding them and refusing the ones it cannot interpret."""

    @staticmethod
    def _write_poses(directory: Path, *, fps: float = 120.0, count: int = 180) -> Path:
        from analyzer.contracts.cache import ContentKey, HashAlgorithm
        from analyzer.contracts.pose import (
            LANDMARK_COUNT,
            LandmarkPoint,
            PoseExtractionStats,
            PoseFrame,
            PoseModelInfo,
            PoseSequence,
        )
        from analyzer.pose.store import write_sequence

        frames = []
        for index in range(count):
            points = [
                LandmarkPoint(x=0.5, y=0.5 + index / 1000, z=0.0, visibility=0.9, presence=0.99)
                for _ in range(LANDMARK_COUNT)
            ]
            frames.append(
                PoseFrame(
                    frame_index=index,
                    timestamp_s=index / fps,
                    detected=True,
                    image=points,
                    hip_local=points,
                )
            )

        sequence = PoseSequence(
            video_path="/data/swing.mov",
            video_content_key=ContentKey(
                algorithm=HashAlgorithm.SHA256_SAMPLED, digest="e" * 64, size_bytes=1
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
                frames_detected=count,
                detection_rate=1.0,
                elapsed_s=1.0,
                ms_per_frame=1.0,
            ),
            frames=frames,
        )
        return write_sequence(sequence, directory / "poses.parquet", landmark_count=LANDMARK_COUNT)

    def test_returns_a_contract_model(self, tmp_path: Path) -> None:
        poses = self._write_poses(tmp_path)
        result = dispatch.call("filter_poses", {"path": str(poses)})
        assert isinstance(result, SequenceFilterReport)
        assert result.samples == 180
        assert result.mean_valid_fraction > 0.9

    def test_accepts_a_configuration_override(self, tmp_path: Path) -> None:
        poses = self._write_poses(tmp_path)
        result = dispatch.call(
            "filter_poses",
            {"path": str(poses), "config": {"gaps": {"max_gap_s": 0.25}}},
        )
        assert isinstance(result, SequenceFilterReport)
        assert result.config.gaps.max_gap_s == 0.25

    def test_rejects_an_unknown_parameter_rather_than_ignoring_it(self, tmp_path: Path) -> None:
        poses = self._write_poses(tmp_path)
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("filter_poses", {"path": str(poses), "windo_s": 0.2})
        assert excinfo.value.code == ErrorCode.INVALID_PARAMS

    def test_rejects_a_configuration_that_cannot_determine_a_fit(self, tmp_path: Path) -> None:
        poses = self._write_poses(tmp_path)
        with pytest.raises(EngineError) as excinfo:
            dispatch.call(
                "filter_poses",
                {
                    "path": str(poses),
                    "config": {"smoothing": {"polyorder": 4, "min_observations": 3}},
                },
            )
        assert excinfo.value.code == ErrorCode.INVALID_PARAMS

    def test_a_file_that_is_not_a_pose_file_is_unsupported_input(self, tmp_path: Path) -> None:
        bogus = tmp_path / "not-poses.parquet"
        bogus.write_bytes(b"certainly not parquet")
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("filter_poses", {"path": str(bogus)})
        assert excinfo.value.code == ErrorCode.UNSUPPORTED_INPUT
        assert (excinfo.value.data or {}).get("remediation")

    @requires_ffprobe
    def test_a_video_with_no_extraction_says_which_command_to_run(self) -> None:
        """Given a clip rather than a Parquet file, the error is actionable."""
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("filter_poses", {"path": str(CFR_30FPS), "model": "pose_landmarker_lite"})
        assert excinfo.value.code == ErrorCode.UNSUPPORTED_INPUT
        assert "analyzer extract" in (excinfo.value.data or {}).get("remediation", "")

    def test_reports_progress_through_the_landmarks(self, tmp_path: Path) -> None:
        poses = self._write_poses(tmp_path)
        reporter = RecordingReporter()
        dispatch.call("filter_poses", {"path": str(poses)}, reporter)
        assert [event.task for event in reporter.events] == ["filter_poses"] * len(reporter.events)
        assert reporter.events[-1].stage == "done"


class TestDetectPhases:
    """Detection runs over filtered landmarks, so its dispatch-level failures are
    the same file-resolution ones plus its own refusal to invent a swing."""

    def test_returns_a_contract_model(self, tmp_path: Path) -> None:
        poses = TestFilterPoses._write_poses(tmp_path)
        result = dispatch.call("detect_phases", {"path": str(poses)})
        assert isinstance(result, SwingPhases)
        assert result.frames == 180

    def test_a_clip_without_a_swing_is_a_result_not_an_error(self, tmp_path: Path) -> None:
        """A still subject is an answer the engine can give, not a failure."""
        poses = TestFilterPoses._write_poses(tmp_path)
        result = dispatch.call("detect_phases", {"path": str(poses)})
        assert isinstance(result, SwingPhases)
        assert not result.detected
        assert result.warnings

    def test_accepts_filter_and_phase_configuration(self, tmp_path: Path) -> None:
        poses = TestFilterPoses._write_poses(tmp_path)
        result = dispatch.call(
            "detect_phases",
            {
                "path": str(poses),
                "filter": {"smoothing": {"window_s": 0.2}},
                "phases": {"min_backswing_s": 0.3},
            },
        )
        assert isinstance(result, SwingPhases)
        assert result.config.min_backswing_s == 0.3

    def test_rejects_an_unknown_parameter_rather_than_ignoring_it(self, tmp_path: Path) -> None:
        poses = TestFilterPoses._write_poses(tmp_path)
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("detect_phases", {"path": str(poses), "phase": {}})
        assert excinfo.value.code == ErrorCode.INVALID_PARAMS

    def test_a_file_that_is_not_a_pose_file_is_unsupported_input(self, tmp_path: Path) -> None:
        bogus = tmp_path / "not-poses.parquet"
        bogus.write_bytes(b"certainly not parquet")
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("detect_phases", {"path": str(bogus)})
        assert excinfo.value.code == ErrorCode.UNSUPPORTED_INPUT


class TestTrackClub:
    """The one analysis method that cannot take a pose file instead of a clip."""

    def test_a_pose_file_is_refused_by_name(self, tmp_path: Path) -> None:
        """And the refusal says why, rather than reporting a missing video.

        Every other method above Phase 2 accepts either a Parquet or the clip it
        came from, so a caller reaching for the same habit here gets an error
        that explains the difference: a shaft is found in the pixels, and a
        stored pose sequence does not contain any.
        """
        poses = TestFilterPoses._write_poses(tmp_path)
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("track_club", {"path": str(poses)})
        assert excinfo.value.code == ErrorCode.UNSUPPORTED_INPUT
        assert "pixels" in str(excinfo.value)

    def test_a_video_with_no_extraction_says_which_command_to_run(self) -> None:
        """The hand anchors come from the cache, so the error names the gap."""
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("track_club", {"path": str(CFR_30FPS)})
        assert excinfo.value.code == ErrorCode.UNSUPPORTED_INPUT
        assert "analyzer extract" in str(excinfo.value.data or {})

    def test_rejects_an_unknown_parameter_rather_than_ignoring_it(self) -> None:
        with pytest.raises(EngineError) as excinfo:
            dispatch.call("track_club", {"path": str(CFR_30FPS), "clubs": {}})
        assert excinfo.value.code == ErrorCode.INVALID_PARAMS

    def test_the_configuration_is_reachable_from_the_parameters(self) -> None:
        """A knob that cannot be set over RPC is a knob the desktop app cannot offer."""
        parsed = dispatch.TrackClubParams.model_validate(
            {"path": "clip.mp4", "club": {"min_confidence": 0.8}}
        )
        assert parsed.club.min_confidence == 0.8
