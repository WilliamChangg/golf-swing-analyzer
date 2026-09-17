"""Verifying that pose inference actually runs, rather than that it imports.

Phase 0's health check reported `mediapipe` as OK on the strength of an
`import mediapipe` succeeding. It did, and pose inference was nevertheless
completely broken: MediaPipe 1.0.1 aborts the process when the pose landmarker
graph opens on macOS arm64. The report was a green light for a pipeline that
could not run a single frame.

That is the precise failure this project exists to avoid, so the check now runs
inference instead of inferring from an import. It costs about a second, which is
the honest price of the claim; see ADR-0008.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from analyzer.contracts.health import ComponentStatus, HealthStatus
from analyzer.environment import pose_smoke
from analyzer.environment.models import ManifestError, load_manifest
from analyzer.paths import models_dir

COMPONENT_NAME = "pose_runtime"

# Generous next to the ~0.8s a working run takes. A graph that has wedged rather
# than aborted must still end the probe, because a health check that hangs is
# worse than one that reports a failure.
_TIMEOUT_S = 60

_BROKEN_BUILD_REMEDY = (
    "This MediaPipe build cannot run pose inference on this platform. Pin "
    "`mediapipe==1.0.0` in python/pyproject.toml and run `uv sync --project python`. "
    "See docs/decisions/ADR-0008-mediapipe-1.0.0.md"
)


def _smallest_available_model() -> Path | None:
    """Pick the cheapest pose model that is actually on disk.

    The smoke test only asks whether the graph opens, and every variant uses the
    same graph, so there is no reason to pay for the largest one.
    """
    try:
        manifest = load_manifest()
    except ManifestError:
        return None

    candidates = [
        models_dir() / entry.filename
        for entry in manifest.models
        if entry.kind == "pose_landmarker"
    ]
    present = [path for path in candidates if path.exists()]
    if not present:
        return None
    return min(present, key=lambda path: path.stat().st_size)


def probe_pose_runtime(model_path: Path | None = None) -> ComponentStatus:
    """Run one pose inference in a child process and report what happened."""
    target = model_path or _smallest_available_model()
    if target is None:
        # Not an error in itself -- the model probe already reports the missing
        # file. What is reported here is that the claim went unverified, which
        # is different from the claim being false.
        return ComponentStatus(
            name=COMPONENT_NAME,
            status=HealthStatus.DEGRADED,
            detail=("Pose inference was not verified: no pose model is present to test with."),
            remediation="Run `python scripts/download_models.py`, then re-run this check.",
        )

    try:
        proc = subprocess.run(  # noqa: S603
            # By file path rather than `-m`: running it as a module would import
            # the analyzer package, dragging OpenCV and the ingestion layer into
            # a child that needs neither.
            [sys.executable, str(Path(pose_smoke.__file__)), str(target)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ComponentStatus(
            name=COMPONENT_NAME,
            status=HealthStatus.ERROR,
            detail=f"Pose inference did not finish within {_TIMEOUT_S}s and was killed.",
            remediation=_BROKEN_BUILD_REMEDY,
        )
    except OSError as exc:
        return ComponentStatus(
            name=COMPONENT_NAME,
            status=HealthStatus.ERROR,
            detail=f"Could not start the pose smoke test: {exc}",
            remediation="Run `uv sync --project python`.",
        )

    if proc.returncode == pose_smoke.EXIT_OK and pose_smoke.SUCCESS_SENTINEL in proc.stdout:
        elapsed = proc.stdout.split(pose_smoke.SUCCESS_SENTINEL, 1)[1].strip()
        return ComponentStatus(
            name=COMPONENT_NAME,
            status=HealthStatus.OK,
            detail=f"Ran one pose inference on {target.name} in {elapsed} ms.",
        )

    # The last line, not the first: a child that dies noisily prints its warnings
    # before the thing that actually killed it.
    lines = proc.stderr.strip().splitlines()
    last = lines[-1].strip() if lines else ""

    if proc.returncode == pose_smoke.EXIT_MODEL_MISSING:
        return ComponentStatus(
            name=COMPONENT_NAME,
            status=HealthStatus.MISSING,
            detail=f"Pose inference was not verified: {target} does not exist.",
            remediation="Run `python scripts/download_models.py`, then re-run this check.",
        )

    if proc.returncode == pose_smoke.EXIT_INFERENCE_FAILED:
        # Catchable, so the interpreter survived: the model or the options are
        # wrong, rather than the MediaPipe build being unable to run at all.
        return ComponentStatus(
            name=COMPONENT_NAME,
            status=HealthStatus.ERROR,
            detail=f"MediaPipe refused to run pose inference on {target.name}: {last}",
            remediation=(
                "The model file is probably corrupt. Re-download it with "
                "`python scripts/download_models.py --force`."
            ),
        )

    # A negative return code is death by signal, which is what an abort looks
    # like from out here, and is the signature of a MediaPipe build whose graph
    # cannot open on this platform.
    cause = (
        f"was killed by signal {-proc.returncode}"
        if proc.returncode < 0
        else f"exited with code {proc.returncode}"
    )
    detail = f"Pose inference failed: the child process {cause}."
    if last:
        detail += f" Last output: {last}"

    return ComponentStatus(
        name=COMPONENT_NAME,
        status=HealthStatus.ERROR,
        detail=detail,
        remediation=_BROKEN_BUILD_REMEDY,
    )
