"""Method registry shared by the RPC worker and the CLI.

Both entry points dispatch through this table, so a method is implemented once
and is immediately reachable from the desktop app and from a terminal. That
keeps the engine independently testable and scriptable without involving Rust.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from analyzer.contracts.rpc import EngineError, ErrorCode
from analyzer.environment.doctor import run_doctor
from analyzer.ingestion import ProbeError, probe_video

# A method takes validated params and returns a Pydantic contract model.
Method = Callable[[dict[str, Any]], BaseModel]


def _doctor(params: dict[str, Any]) -> BaseModel:
    """Environment health check. Takes no parameters."""
    if params:
        raise EngineError(
            f"doctor takes no parameters, got: {sorted(params)}",
            code=ErrorCode.INVALID_PARAMS,
        )
    return run_doctor()


class ProbeVideoParams(BaseModel, extra="forbid"):
    """Parameters for `probe_video`.

    `extra="forbid"` so a typo'd or stale parameter name is an error the caller
    sees, rather than silently taking the default.
    """

    path: str = Field(description="Absolute path to the video file.")
    refresh: bool = Field(default=False, description="Re-probe even if a cached result exists.")


def _probe_video(params: dict[str, Any]) -> BaseModel:
    """Read a video's container metadata without decoding it."""
    parsed = ProbeVideoParams.model_validate(params)
    try:
        return probe_video(Path(parsed.path), refresh=parsed.refresh)
    except ProbeError as exc:
        # The user chose a file this system cannot analyse. That is an input
        # problem with a known remedy, not an engine fault, and the two need
        # different treatment in the UI.
        raise EngineError(
            str(exc),
            code=ErrorCode.UNSUPPORTED_INPUT,
            data={"remediation": exc.remediation} if exc.remediation else None,
        ) from exc


METHODS: dict[str, Method] = {
    "doctor": _doctor,
    "probe_video": _probe_video,
}


def call(method: str, params: dict[str, Any] | None = None) -> BaseModel:
    """Invoke a registered method, normalising failures into EngineError.

    Unknown methods and parameter-validation failures are distinguished from
    genuine internal errors so the desktop app can react differently (a version
    mismatch is a different problem from a crash).
    """
    handler = METHODS.get(method)
    if handler is None:
        raise EngineError(
            f"Unknown method '{method}'. Known methods: {', '.join(sorted(METHODS))}",
            code=ErrorCode.METHOD_NOT_FOUND,
        )

    try:
        return handler(params or {})
    except EngineError:
        raise
    except ValidationError as exc:
        raise EngineError(
            f"Invalid parameters for '{method}': {exc}",
            code=ErrorCode.INVALID_PARAMS,
        ) from exc
    except Exception as exc:  # boundary: everything becomes a typed error
        raise EngineError(
            f"'{method}' failed: {type(exc).__name__}: {exc}",
            code=ErrorCode.INTERNAL_ERROR,
        ) from exc
