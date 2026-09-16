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
from analyzer.contracts.rpc import EngineError, ErrorCode
from analyzer.contracts.video import VideoMetadata
from tests.conftest import CFR_30FPS, CFR_FRAME_COUNT, requires_ffprobe


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

        def exploding(_params: dict[str, object]) -> BaseModel:
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

        def picky(_params: dict[str, object]) -> BaseModel:
            raise EngineError("bad input", code=ErrorCode.UNSUPPORTED_INPUT)

        monkeypatch.setitem(dispatch.METHODS, "picky", picky)

        with pytest.raises(EngineError) as excinfo:
            dispatch.call("picky")
        assert excinfo.value.code == ErrorCode.UNSUPPORTED_INPUT

    def test_every_registered_method_returns_a_contract_model(self) -> None:
        """Guards the invariant the worker relies on when calling model_dump.

        The parameter table has to be kept up to date by hand, which is the
        point: a new method with no entry fails here rather than going
        unexercised.
        """
        params: dict[str, dict[str, object]] = {
            "doctor": {},
            "probe_video": {"path": str(CFR_30FPS)},
        }
        assert set(params) == set(dispatch.METHODS), "update the table for the new method"

        for name, args in params.items():
            assert isinstance(dispatch.call(name, args), BaseModel), name


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
