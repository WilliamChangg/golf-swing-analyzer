"""Method registry shared by the RPC worker and the CLI.

Both entry points dispatch through this table, so a method is implemented once
and is immediately reachable from the desktop app and from a terminal. That
keeps the engine independently testable and scriptable without involving Rust.

Every method is handed a `ProgressReporter`. Short methods ignore it; long ones
report through it and are thereby indifferent to whether the other end is a
JSON-RPC notification, a terminal progress bar, or a list in a test.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.pose import LandmarkSpace
from analyzer.contracts.rpc import EngineError, ErrorCode
from analyzer.environment.doctor import run_doctor
from analyzer.ingestion import ProbeError, probe_video
from analyzer.paths import cache_dir
from analyzer.progress import NullReporter, ProgressReporter

# A method takes validated params and a progress sink, and returns a contract model.
Method = Callable[[dict[str, Any], ProgressReporter], BaseModel]


def _doctor(params: dict[str, Any], _reporter: ProgressReporter) -> BaseModel:
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


def _unsupported_input(exc: Exception, remediation: str | None) -> EngineError:
    """Turn a user-fixable input problem into a typed error the UI can act on.

    Distinguished from an internal error because the two need different
    treatment: a wrong file is something the user corrects, a crash is not.
    """
    return EngineError(
        str(exc),
        code=ErrorCode.UNSUPPORTED_INPUT,
        data={"remediation": remediation} if remediation else None,
    )


def _probe_video(params: dict[str, Any], _reporter: ProgressReporter) -> BaseModel:
    """Read a video's container metadata without decoding it."""
    parsed = ProbeVideoParams.model_validate(params)
    try:
        return probe_video(Path(parsed.path), refresh=parsed.refresh)
    except ProbeError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc


class ExtractPosesParams(BaseModel, extra="forbid"):
    """Parameters for `extract_poses`."""

    path: str = Field(description="Absolute path to the video file.")
    model: str | None = Field(
        default=None,
        description="Manifest model name. None uses the manifest's default_pose_model.",
    )
    output: str | None = Field(
        default=None,
        description="Where to write the Parquet file. None uses the content-keyed cache path.",
    )


def _extract_poses(params: dict[str, Any], reporter: ProgressReporter) -> BaseModel:
    """Run pose estimation over every frame of a clip and store the landmarks."""
    parsed = ExtractPosesParams.model_validate(params)

    # Imported here rather than at module scope: pulling in MediaPipe costs
    # about a second, and the worker should not pay that at spawn for a session
    # that may only ever call `doctor`.
    from analyzer.pose.estimator import PoseEstimationError
    from analyzer.pose.extract import extract_and_store
    from analyzer.pose.mediapipe_estimator import MediaPipePoseEstimator

    try:
        estimator = MediaPipePoseEstimator(parsed.model)
    except PoseEstimationError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc

    try:
        return extract_and_store(
            Path(parsed.path),
            estimator,
            output=Path(parsed.output) if parsed.output else None,
            reporter=reporter,
        )
    except ProbeError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc
    except PoseEstimationError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc
    finally:
        estimator.close()


class FilterPosesParams(BaseModel, extra="forbid"):
    """Parameters for `filter_poses`."""

    path: str = Field(
        description=(
            "A pose Parquet file, or the video it was extracted from -- in which "
            "case the content-keyed cache is consulted for its landmarks."
        )
    )
    model: str | None = Field(
        default=None,
        description="Which model's extraction to filter. Only used when `path` is a video.",
    )
    space: LandmarkSpace = Field(
        default=LandmarkSpace.IMAGE,
        description="Coordinate space to filter. HIP_LOCAL is not calibrated world geometry.",
    )
    config: FilterConfig = Field(
        default_factory=FilterConfig,
        description="Gate, gap and smoothing policy. Defaults are the measured ones.",
    )


def _resolve_pose_file(parsed: FilterPosesParams) -> Path:
    """Find the landmarks to filter, from either a Parquet path or a video path.

    Taking a video and resolving its cache entry is the common case: a caller
    thinks in terms of the clip, not of where extraction happened to put its
    output. When nothing has been extracted the error says which command to run
    rather than reporting a missing file.
    """
    from analyzer.pose.estimator import resolve_model

    candidate = Path(parsed.path)
    if candidate.suffix == ".parquet":
        return candidate

    metadata = probe_video(candidate)
    entry, _ = resolve_model(parsed.model)
    poses = cache_dir() / "poses" / metadata.content_key.as_path_segment() / f"{entry.name}.parquet"
    if not poses.exists():
        raise EngineError(
            f"No extracted poses for {candidate.name} with model '{entry.name}'.",
            code=ErrorCode.UNSUPPORTED_INPUT,
            data={"remediation": f"Run: analyzer extract {parsed.path} --model {entry.name}"},
        )
    return poses


def _filter_poses(params: dict[str, Any], reporter: ProgressReporter) -> BaseModel:
    """Smooth stored landmarks and differentiate them, reporting what was refused."""
    parsed = FilterPosesParams.model_validate(params)

    # Imported here rather than at module scope to keep the worker's spawn cost
    # off a session that only ever calls `doctor`, matching `extract_poses`.
    from analyzer.filtering.landmarks import filter_sequence
    from analyzer.pose.estimator import PoseEstimationError
    from analyzer.pose.store import PoseStoreError, read_sequence

    try:
        poses = _resolve_pose_file(parsed)
        sequence = read_sequence(poses)
    except ProbeError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc
    except PoseEstimationError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc
    except PoseStoreError as exc:
        raise _unsupported_input(exc, "Re-run the extraction for this clip.") from exc

    result = filter_sequence(sequence, parsed.config, space=parsed.space, reporter=reporter)
    return result.report


METHODS: dict[str, Method] = {
    "doctor": _doctor,
    "probe_video": _probe_video,
    "extract_poses": _extract_poses,
    "filter_poses": _filter_poses,
}


def call(
    method: str,
    params: dict[str, Any] | None = None,
    reporter: ProgressReporter | None = None,
) -> BaseModel:
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
        return handler(params or {}, reporter or NullReporter())
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
