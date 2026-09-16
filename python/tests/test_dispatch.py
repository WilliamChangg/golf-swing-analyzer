"""Dispatch table tests.

The dispatch layer is the single place where arbitrary failures become typed RPC
errors. If that normalisation leaks, the desktop app sees a hung worker instead
of an error it can render.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from analyzer import dispatch
from analyzer.contracts.rpc import EngineError, ErrorCode


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
        """Guards the invariant the worker relies on when calling model_dump."""
        for name in dispatch.METHODS:
            assert isinstance(dispatch.call(name), BaseModel), name
