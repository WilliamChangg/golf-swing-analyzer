"""The pose estimation seam.

`PoseEstimator` is what every later phase depends on. MediaPipe is one
implementation of it and its types stop at the adapter: no `NormalizedLandmark`,
no `mp.Image`, no `PoseLandmarkerResult` reaches a caller. That is what makes
the model replaceable -- by a different MediaPipe variant, by a learned model
trained in Phase 12, or by a fake in a test -- without touching anything that
consumes poses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from analyzer.contracts.pose import PoseFrame, PoseModelInfo
from analyzer.environment.models import ManifestError, ModelEntry, load_manifest
from analyzer.hashing import sha256_file
from analyzer.ingestion.reader import VideoFrame
from analyzer.paths import models_dir


class PoseEstimationError(RuntimeError):
    """A pose estimator could not be created or could not run.

    Carries a remediation for the same reason `ProbeError` does: a missing model
    file is something the user fixes, not something the engine recovers from.
    """

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


class PoseEstimator(Protocol):
    """Turns video frames into landmarks.

    Implementations are stateful: `estimate` must be called with frames in
    increasing timestamp order, because a tracker carries state between frames.
    """

    @property
    def info(self) -> PoseModelInfo: ...

    def estimate(self, frame: VideoFrame) -> PoseFrame:
        """Landmarks for one frame, or a `PoseFrame` with `detected=False`."""
        ...

    def close(self) -> None: ...


def resolve_model(name: str | None = None) -> tuple[ModelEntry, Path]:
    """Find a pose model by manifest name, defaulting to the manifest's own choice.

    Resolved through the manifest rather than by globbing the models directory,
    so that what gets loaded is the artifact the manifest pins and the sha256
    recorded alongside results means something.
    """
    try:
        manifest = load_manifest()
    except ManifestError as exc:
        raise PoseEstimationError(
            str(exc), remediation="Restore models/manifest.json from version control."
        ) from exc

    wanted = name or manifest.default_pose_model
    entry = manifest.get(wanted)
    if entry is None:
        available = ", ".join(m.name for m in manifest.models)
        raise PoseEstimationError(
            f"No model named '{wanted}' in the manifest. Available: {available}.",
            remediation="Check the spelling, or add the model to models/manifest.json.",
        )

    path = models_dir() / entry.filename
    if not path.exists():
        raise PoseEstimationError(
            f"Model '{entry.name}' is not downloaded: {path} does not exist.",
            remediation=f"Run `python scripts/download_models.py --only {entry.name}`.",
        )

    return entry, path


def model_info(
    entry: ModelEntry,
    path: Path,
    *,
    delegate: str,
    min_pose_detection_confidence: float,
    min_pose_presence_confidence: float,
    min_tracking_confidence: float,
) -> PoseModelInfo:
    """Describe the model that was actually loaded.

    The digest is taken from the file on disk rather than copied from the
    manifest: the manifest says what *should* be there, and the point of
    recording it on a result is to say what *was*.
    """
    return PoseModelInfo(
        name=entry.name,
        variant=variant_of(entry.name),
        precision=entry.precision,
        sha256=sha256_file(path),
        delegate=delegate,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def variant_of(name: str) -> str:
    """'pose_landmarker_full' -> 'full'."""
    return name.rsplit("_", 1)[-1]
