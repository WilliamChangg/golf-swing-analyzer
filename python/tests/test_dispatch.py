"""Dispatch table tests.

The dispatch layer is the single place where arbitrary failures become typed RPC
errors. If that normalisation leaks, the desktop app sees a hung worker instead
of an error it can render.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from analyzer import dispatch
from analyzer.contracts.pose import PoseExtractionResult
from analyzer.contracts.rpc import EngineError, ErrorCode
from analyzer.contracts.video import VideoMetadata
from analyzer.dispatch import ProbeVideoParams
from analyzer.progress import ProgressReporter, ProgressTracker, RecordingReporter
from tests.conftest import CFR_30FPS, CFR_FRAME_COUNT, requires_ffprobe

# Parameters good enough to invoke each registered method once. Maintained by
# hand on purpose: the table is what makes a newly-registered method visible.
_METHOD_PARAMS: dict[str, dict[str, object]] = {
    "doctor": {},
    "probe_video": {"path": str(CFR_30FPS)},
    "extract_poses": {"path": str(CFR_30FPS)},
}

# Everything that runs in milliseconds. `extract_poses` loads a model and
# decodes a clip, so it is exercised separately under the slow markers.
_FAST_METHODS = {"doctor", "probe_video"}


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
