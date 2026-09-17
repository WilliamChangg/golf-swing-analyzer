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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.metrics import MetricConfig
from analyzer.contracts.phases import PhaseConfig
from analyzer.contracts.pose import LandmarkSpace
from analyzer.contracts.projects import CameraRole
from analyzer.contracts.rpc import EngineError, ErrorCode
from analyzer.contracts.sync import SyncConfig
from analyzer.environment.doctor import run_doctor
from analyzer.ingestion import ProbeError, probe_video
from analyzer.paths import cache_dir
from analyzer.progress import NullReporter, ProgressReporter

if TYPE_CHECKING:  # imports that would pull numpy and scipy in at worker spawn
    from analyzer.contracts.projects import Project, ProjectClip
    from analyzer.sync import SyncInput

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
        default=LandmarkSpace.FRAME_WIDTHS,
        description=(
            "Reference frame to filter in. FRAME_WIDTHS is isotropic and is what "
            "every measurement is taken in; IMAGE is the estimator's own output, "
            "anisotropic and y-down, useful for drawing. Neither is metric."
        ),
    )
    config: FilterConfig = Field(
        default_factory=FilterConfig,
        description="Gate, gap and smoothing policy. Defaults are the measured ones.",
    )
    slow_motion_factor: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "How many times slower than real time the clip plays. 1 for an "
            "ordinary recording, 8 for eight-times slow motion. Timestamps are "
            "divided by it, which is all it takes to put every duration, speed "
            "and filter window on a real clock. Nothing in a conformed file "
            "records this, so it is supplied rather than measured."
        ),
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

    result = filter_sequence(
        sequence,
        parsed.config,
        space=parsed.space,
        slow_motion_factor=parsed.slow_motion_factor,
        reporter=reporter,
    )
    return result.report


class DetectPhasesParams(BaseModel, extra="forbid"):
    """Parameters for `detect_phases`."""

    path: str = Field(
        description=(
            "A pose Parquet file, or the video it was extracted from -- in which "
            "case the content-keyed cache is consulted for its landmarks."
        )
    )
    model: str | None = Field(
        default=None,
        description="Which model's extraction to use. Only used when `path` is a video.",
    )
    filter: FilterConfig = Field(
        default_factory=FilterConfig,
        description="Smoothing policy. Detection reads the window from it when scoring events.",
    )
    phases: PhaseConfig = Field(
        default_factory=PhaseConfig,
        description="Structural bounds a motion must satisfy to be reported as a swing.",
    )
    slow_motion_factor: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "How many times slower than real time the clip plays. 1 for an "
            "ordinary recording, 8 for eight-times slow motion. Timestamps are "
            "divided by it, which is all it takes to put every duration, speed "
            "and filter window on a real clock. Nothing in a conformed file "
            "records this, so it is supplied rather than measured."
        ),
    )


def _detect_phases(params: dict[str, Any], reporter: ProgressReporter) -> BaseModel:
    """Locate the swing events in a clip's stored landmarks."""
    parsed = DetectPhasesParams.model_validate(params)

    from analyzer.filtering.landmarks import filter_sequence
    from analyzer.phases import SignalError, detect_phases
    from analyzer.pose.estimator import PoseEstimationError
    from analyzer.pose.store import PoseStoreError, read_sequence

    try:
        poses = _resolve_pose_file(
            FilterPosesParams(path=parsed.path, model=parsed.model, config=parsed.filter)
        )
        sequence = read_sequence(poses)
    except ProbeError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc
    except PoseEstimationError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc
    except PoseStoreError as exc:
        raise _unsupported_input(exc, "Re-run the extraction for this clip.") from exc

    # Detection always reads FRAME_WIDTHS: it is the only frame here that is
    # isotropic and upward-positive, and both matter to rules about how far and
    # how high the hands went.
    filtered = filter_sequence(
        sequence,
        parsed.filter,
        space=LandmarkSpace.FRAME_WIDTHS,
        slow_motion_factor=parsed.slow_motion_factor,
        reporter=reporter,
    )
    try:
        return detect_phases(filtered, parsed.phases)
    except SignalError as exc:
        raise _unsupported_input(exc, None) from exc


class ComputeMetricsParams(BaseModel, extra="forbid"):
    """Parameters for `compute_metrics`."""

    path: str = Field(
        description=(
            "A pose Parquet file, or the video it was extracted from -- in which "
            "case the content-keyed cache is consulted for its landmarks."
        )
    )
    model: str | None = Field(
        default=None,
        description="Which model's extraction to use. Only used when `path` is a video.",
    )
    filter: FilterConfig = Field(
        default_factory=FilterConfig,
        description="Smoothing policy, applied before phases are detected and metrics measured.",
    )
    phases: PhaseConfig = Field(
        default_factory=PhaseConfig,
        description="Structural bounds a motion must satisfy to be reported as a swing.",
    )
    metrics: MetricConfig = Field(
        default_factory=MetricConfig,
        description="When a measurement is too ill-conditioned to report at all.",
    )
    slow_motion_factor: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "How many times slower than real time the clip plays. 1 for an "
            "ordinary recording, 8 for eight-times slow motion. Timestamps are "
            "divided by it, which is all it takes to put every duration, speed "
            "and filter window on a real clock. Nothing in a conformed file "
            "records this, so it is supplied rather than measured."
        ),
    )


def _compute_metrics(params: dict[str, Any], reporter: ProgressReporter) -> BaseModel:
    """Measure the biomechanics metrics for a clip's stored landmarks.

    Runs the whole chain rather than taking a detection as input: filtering is
    milliseconds and detection is cheaper still, so recomputing them here costs
    nothing measurable and removes the possibility of metrics being measured
    against a detection produced under a different filter configuration.
    """
    parsed = ComputeMetricsParams.model_validate(params)

    from analyzer.biomechanics import BodyError, compute_metrics
    from analyzer.filtering.landmarks import filter_sequence
    from analyzer.phases import SignalError, detect_phases
    from analyzer.pose.estimator import PoseEstimationError
    from analyzer.pose.store import PoseStoreError, read_sequence

    try:
        poses = _resolve_pose_file(
            FilterPosesParams(path=parsed.path, model=parsed.model, config=parsed.filter)
        )
        sequence = read_sequence(poses)
    except ProbeError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc
    except PoseEstimationError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc
    except PoseStoreError as exc:
        raise _unsupported_input(exc, "Re-run the extraction for this clip.") from exc

    filtered = filter_sequence(
        sequence,
        parsed.filter,
        space=LandmarkSpace.FRAME_WIDTHS,
        slow_motion_factor=parsed.slow_motion_factor,
        reporter=reporter,
    )
    try:
        detected = detect_phases(filtered, parsed.phases)
        return compute_metrics(filtered, detected, parsed.metrics)
    except (SignalError, BodyError) as exc:
        raise _unsupported_input(exc, None) from exc


class SyncClipParams(BaseModel, extra="forbid"):
    """One side of a synchronisation request."""

    path: str = Field(
        description=(
            "A pose Parquet file, or the video it was extracted from -- in which "
            "case the content-keyed cache is consulted for its landmarks."
        )
    )
    model: str | None = Field(
        default=None,
        description="Which model's extraction to use. Only used when `path` is a video.",
    )
    slow_motion_factor: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "How many times slower than real time this clip plays. Supplied per "
            "clip, not per request: two cameras in one session routinely differ, "
            "and a swing filmed at 240 fps beside one at 30 is the ordinary case."
        ),
    )


class ManualAnchorParams(BaseModel, extra="forbid"):
    """One instant a person identified in both clips.

    Frames rather than seconds, because a frame is what a person picks out of a
    scrubber and seconds are what only the engine can compute -- on variable-rate
    footage `frame / fps` is not when the frame was taken.
    """

    label: str = Field(description="What this instant is. Shown back to the reader.")
    reference_frame: int = Field(ge=0)
    target_frame: int = Field(ge=0)


class SyncClipsParams(BaseModel, extra="forbid"):
    """Parameters for `sync_clips`."""

    reference: SyncClipParams
    target: SyncClipParams
    filter: FilterConfig = Field(
        default_factory=FilterConfig,
        description=(
            "Smoothing policy, applied to **both** clips. Shared rather than "
            "per clip on purpose: a wider window flattens and slightly shifts "
            "the speed features both methods key on, so smoothing two clips "
            "differently would bias the alignment by an amount nothing "
            "measures. The consequence is that the coarser clip sets the window "
            "for the pair, and a 30 fps camera cannot support the 0.10 s default."
        ),
    )
    phases: PhaseConfig = Field(
        default_factory=PhaseConfig,
        description="Structural bounds a motion must satisfy to be reported as a swing.",
    )
    sync: SyncConfig = Field(
        default_factory=SyncConfig, description="Which method, and when to refuse."
    )
    anchors: list[ManualAnchorParams] = Field(
        default_factory=list,
        description=(
            "Instants a person identified in both clips. Supplying any switches "
            "the method to `manual`: a person who has looked at both frames is "
            "giving the answer, not an estimate for the engine to arbitrate."
        ),
    )


def _sync_input(
    clip: SyncClipParams,
    filter_config: FilterConfig,
    phase_config: PhaseConfig,
    reporter: ProgressReporter,
) -> SyncInput:
    """Run one clip through to signals and events, ready to be aligned.

    Recomputed rather than taken as input for the same reason `compute_metrics`
    does it: filtering is milliseconds, and accepting a pre-computed detection
    would allow two clips to be aligned whose events were found under different
    filter settings -- which is a difference that would surface as a sync error
    with no way to attribute it.
    """
    from analyzer.filtering.landmarks import filter_sequence
    from analyzer.phases import SignalError, detect_phases
    from analyzer.phases.signals import swing_signals
    from analyzer.pose.estimator import PoseEstimationError
    from analyzer.pose.store import PoseStoreError, read_sequence
    from analyzer.sync import SyncInput

    try:
        poses = _resolve_pose_file(
            FilterPosesParams(path=clip.path, model=clip.model, config=filter_config)
        )
        sequence = read_sequence(poses)
    except ProbeError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc
    except PoseEstimationError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc
    except PoseStoreError as exc:
        raise _unsupported_input(exc, "Re-run the extraction for this clip.") from exc

    filtered = filter_sequence(
        sequence,
        filter_config,
        space=LandmarkSpace.FRAME_WIDTHS,
        slow_motion_factor=clip.slow_motion_factor,
        reporter=reporter,
    )
    try:
        # Signals and events are both needed: correlation reads the first,
        # anchors the second, and a clip with no swing in it still contributes
        # the first.
        return SyncInput(
            path=Path(clip.path),
            signals=swing_signals(filtered),
            phases=detect_phases(filtered, phase_config),
            notes=tuple(filtered.report.warnings),
        )
    except SignalError as exc:
        raise _unsupported_input(exc, None) from exc


def _sync_clips(params: dict[str, Any], reporter: ProgressReporter) -> BaseModel:
    """Relate two clips' clocks, reporting how well the relation is determined."""
    parsed = SyncClipsParams.model_validate(params)

    from analyzer.sync import ManualPick, align
    from analyzer.sync.anchors import AnchorError

    reference = _sync_input(parsed.reference, parsed.filter, parsed.phases, reporter)
    target = _sync_input(parsed.target, parsed.filter, parsed.phases, reporter)

    picks = [
        ManualPick(
            label=anchor.label,
            reference_frame=anchor.reference_frame,
            target_frame=anchor.target_frame,
        )
        for anchor in parsed.anchors
    ]
    try:
        return align(reference, target, parsed.sync, picks=picks)
    except AnchorError as exc:
        raise _unsupported_input(exc, "Pick a frame that exists in both clips.") from exc
    except ValueError as exc:
        raise _unsupported_input(exc, None) from exc


class CreateProjectParams(BaseModel, extra="forbid"):
    """Parameters for `create_project`."""

    name: str = Field(description="What this session is. Not unique; the id is the identity.")
    notes: str = Field(default="", description="Free text. Never interpreted.")


class ProjectParams(BaseModel, extra="forbid"):
    """Parameters for the methods that address one project by id."""

    project_id: int


class AddClipParams(BaseModel, extra="forbid"):
    """Parameters for `add_clip`."""

    project_id: int
    path: str = Field(description="Absolute path to the video file.")
    role: CameraRole = Field(
        description=(
            "Where the camera was, as declared. Phase 6 measures the view from "
            "the footage and may disagree, which is a disagreement worth being "
            "able to state -- and cannot be stated unless this was recorded."
        )
    )
    slow_motion_factor: float = Field(default=1.0, gt=0.0)
    label: str = Field(default="", description="Free text. Never interpreted.")


class SyncProjectParams(BaseModel, extra="forbid"):
    """Parameters for `sync_project`."""

    project_id: int
    reference_clip_id: int | None = Field(
        default=None,
        description=(
            "Which clip is the reference. None picks the face-on clip where "
            "there is one, and the lowest id otherwise."
        ),
    )
    target_clip_id: int | None = Field(
        default=None, description="Which clip to align to the reference. None picks the other one."
    )
    model: str | None = None
    filter: FilterConfig = Field(default_factory=FilterConfig)
    phases: PhaseConfig = Field(default_factory=PhaseConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    anchors: list[ManualAnchorParams] = Field(default_factory=list)
    save: bool = Field(
        default=True, description="Store the result on the project, replacing any previous one."
    )


def _project_error(exc: Exception) -> EngineError:
    """Normalise a store failure, keeping whatever remedy it carried."""
    return _unsupported_input(exc, getattr(exc, "remediation", None))


def _create_project(params: dict[str, Any], _reporter: ProgressReporter) -> BaseModel:
    """Create an empty project."""
    parsed = CreateProjectParams.model_validate(params)

    from analyzer.projects import ProjectError, ProjectStore

    try:
        with ProjectStore() as store:
            return store.create_project(parsed.name, notes=parsed.notes)
    except ProjectError as exc:
        raise _project_error(exc) from exc


def _list_projects(params: dict[str, Any], _reporter: ProgressReporter) -> BaseModel:
    """Every project, with its clips. Stored alignments are omitted; see `get_project`."""
    if params:
        raise EngineError(
            f"list_projects takes no parameters, got: {sorted(params)}",
            code=ErrorCode.INVALID_PARAMS,
        )

    from analyzer.projects import ProjectError, ProjectStore

    try:
        with ProjectStore() as store:
            return store.list_projects()
    except ProjectError as exc:
        raise _project_error(exc) from exc


def _get_project(params: dict[str, Any], _reporter: ProgressReporter) -> BaseModel:
    """One project, with its clips and its stored alignments."""
    parsed = ProjectParams.model_validate(params)

    from analyzer.projects import ProjectError, ProjectStore

    try:
        with ProjectStore() as store:
            return store.get_project(parsed.project_id)
    except ProjectError as exc:
        raise _project_error(exc) from exc


def _add_clip(params: dict[str, Any], _reporter: ProgressReporter) -> BaseModel:
    """Attach a video to a project, identifying it by content rather than by path."""
    parsed = AddClipParams.model_validate(params)

    from analyzer.projects import ProjectError, ProjectStore

    try:
        metadata = probe_video(Path(parsed.path))
    except ProbeError as exc:
        raise _unsupported_input(exc, exc.remediation) from exc

    try:
        with ProjectStore() as store:
            return store.add_clip(
                parsed.project_id,
                path=Path(parsed.path),
                content_key=metadata.content_key,
                role=parsed.role,
                slow_motion_factor=parsed.slow_motion_factor,
                label=parsed.label,
            )
    except ProjectError as exc:
        raise _project_error(exc) from exc


class ClipParams(BaseModel, extra="forbid"):
    """Parameters for the methods that address one clip of one project."""

    project_id: int
    clip_id: int


class RelocateClipParams(BaseModel, extra="forbid"):
    """Parameters for `relocate_clip`."""

    project_id: int
    clip_id: int
    path: str = Field(description="Where the file is now.")


def _delete_project(params: dict[str, Any], _reporter: ProgressReporter) -> BaseModel:
    """Delete a project, its clips and its alignments. The video files are untouched."""
    parsed = ProjectParams.model_validate(params)

    from analyzer.projects import ProjectError, ProjectStore

    try:
        with ProjectStore() as store:
            store.delete_project(parsed.project_id)
            # The remaining projects rather than nothing: every other method
            # here returns the state that resulted, and a caller that has just
            # removed something is usually about to re-render the list.
            return store.list_projects()
    except ProjectError as exc:
        raise _project_error(exc) from exc


def _remove_clip(params: dict[str, Any], _reporter: ProgressReporter) -> BaseModel:
    """Detach a clip from a project. Its stored alignments go with it."""
    parsed = ClipParams.model_validate(params)

    from analyzer.projects import ProjectError, ProjectStore

    try:
        with ProjectStore() as store:
            return store.remove_clip(parsed.project_id, parsed.clip_id)
    except ProjectError as exc:
        raise _project_error(exc) from exc


def _relocate_clip(params: dict[str, Any], _reporter: ProgressReporter) -> BaseModel:
    """Point a clip at a moved file, keeping its recorded content key.

    The key is deliberately not recomputed: a file whose bytes changed is a
    different clip and should be added as one, and silently accepting new
    content under an existing id would leave every stored alignment describing
    footage that is no longer there.
    """
    parsed = RelocateClipParams.model_validate(params)

    from analyzer.projects import ProjectError, ProjectStore

    try:
        with ProjectStore() as store:
            return store.set_clip_path(parsed.project_id, parsed.clip_id, Path(parsed.path))
    except ProjectError as exc:
        raise _project_error(exc) from exc


def _sync_project(params: dict[str, Any], reporter: ProgressReporter) -> BaseModel:
    """Align two of a project's clips, and store the result on the project."""
    parsed = SyncProjectParams.model_validate(params)

    from analyzer.contracts.sync import SyncModel
    from analyzer.projects import ProjectError, ProjectStore

    try:
        with ProjectStore() as store:
            project = store.get_project(parsed.project_id)
            reference, target = _choose_pair(
                project, parsed.reference_clip_id, parsed.target_clip_id
            )

            result = _sync_clips(
                {
                    "reference": {
                        "path": reference.path,
                        "model": parsed.model,
                        "slow_motion_factor": reference.slow_motion_factor,
                    },
                    "target": {
                        "path": target.path,
                        "model": parsed.model,
                        "slow_motion_factor": target.slow_motion_factor,
                    },
                    "filter": parsed.filter.model_dump(mode="json"),
                    "phases": parsed.phases.model_dump(mode="json"),
                    "sync": parsed.sync.model_dump(mode="json"),
                    "anchors": [anchor.model_dump(mode="json") for anchor in parsed.anchors],
                },
                reporter,
            )
            assert isinstance(result, SyncModel)  # noqa: S101 - narrows the dispatch return type

            if parsed.save:
                store.save_sync(
                    parsed.project_id,
                    reference_clip_id=reference.id,
                    target_clip_id=target.id,
                    model=result,
                )
            return result
    except ProjectError as exc:
        raise _project_error(exc) from exc


def _choose_pair(
    project: Project, reference_id: int | None, target_id: int | None
) -> tuple[ProjectClip, ProjectClip]:
    """Resolve which two clips to align, defaulting to the face-on one as reference.

    Face-on by default because it is the view that sees the whole sweep of the
    hands, which is the signal both methods key on; a down-the-line camera loses
    much of it to foreshortening. Nothing breaks if the other order is chosen --
    the map is invertible -- but the default should be the one more likely to
    produce a peak worth trusting.

    Anything ambiguous is an error rather than a choice. A project with three
    clips and neither end named has three possible pairs, and picking one would
    make the answer depend on insertion order.
    """
    from analyzer.projects import ProjectError

    if len(project.clips) < 2:
        raise ProjectError(
            f"Project {project.id} has {len(project.clips)} clip(s); synchronisation needs two.",
            remediation="Add a second clip with `analyzer project add`.",
        )
    if reference_id is not None and reference_id == target_id:
        raise ProjectError("A clip cannot be synchronised against itself.")

    def resolve(clip_id: int) -> ProjectClip:
        found = project.clip_by_id(clip_id)
        if found is None:
            raise ProjectError(
                f"Project {project.id} has no clip with id {clip_id}.",
                remediation=f"Run `analyzer project show {project.id}` to see its clips.",
            )
        return found

    def only_other(exclude: int) -> ProjectClip:
        remaining = [clip for clip in project.clips if clip.id != exclude]
        if len(remaining) > 1:
            raise ProjectError(
                f"Project {project.id} has {len(project.clips)} clips, so which pair to align "
                "is ambiguous.",
                remediation="Name both ends with --reference-clip and --target-clip.",
            )
        return remaining[0]

    if reference_id is not None:
        reference = resolve(reference_id)
        return reference, (
            resolve(target_id) if target_id is not None else only_other(reference.id)
        )

    if target_id is not None:
        target = resolve(target_id)
        return only_other(target.id), target

    face_on = project.clip(CameraRole.FACE_ON)
    reference = face_on if face_on is not None else project.clips[0]
    return reference, only_other(reference.id)


METHODS: dict[str, Method] = {
    "doctor": _doctor,
    "probe_video": _probe_video,
    "extract_poses": _extract_poses,
    "filter_poses": _filter_poses,
    "detect_phases": _detect_phases,
    "compute_metrics": _compute_metrics,
    "sync_clips": _sync_clips,
    "create_project": _create_project,
    "list_projects": _list_projects,
    "get_project": _get_project,
    "delete_project": _delete_project,
    "add_clip": _add_clip,
    "remove_clip": _remove_clip,
    "relocate_clip": _relocate_clip,
    "sync_project": _sync_project,
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
