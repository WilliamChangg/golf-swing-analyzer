"""Inventory and atomic installation of manifest-pinned models.

An update installs the version pinned by this build. Upstream 'latest' never
changes the manifest implicitly. Failed transfers leave the installed file intact.
"""

from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path
from typing import Literal

from analyzer.contracts.models import ManagedModel, ModelInventory
from analyzer.environment.hardware import mediapipe_delegate
from analyzer.environment.models import ModelEntry, load_manifest
from analyzer.hashing import sha256_file
from analyzer.paths import models_dir
from analyzer.progress import NullReporter, ProgressReporter, ProgressTracker, ThrottledReporter


class ModelInstallError(RuntimeError):
    """The requested artifact could not be installed and verified."""


def describe_model(entry: ModelEntry, directory: Path) -> ManagedModel:
    target = directory / entry.filename
    digest = sha256_file(target) if target.is_file() else None
    matches = digest == entry.sha256 and target.stat().st_size == entry.size_bytes
    state: Literal["verified", "missing", "mismatch"] = (
        "verified" if matches else "missing" if digest is None else "mismatch"
    )
    return ManagedModel(
        name=entry.name,
        version=entry.sha256,
        backend="mediapipe_tasks",
        device=mediapipe_delegate(),
        input_requirements=[
            "One person per frame",
            "RGB image (BGR video frames converted by the adapter)",
            "Video frames in increasing timestamp order",
        ],
        size_bytes=entry.size_bytes,
        required=entry.required,
        state=state,
        installed_sha256=digest,
        detail={
            "verified": "Size and SHA-256 match the pinned version. Inference is checked separately by Re-check.",
            "missing": "Not installed.",
            "mismatch": "Installed bytes differ from this build's pinned version. Update to restore it.",
        }[state],
    )


def inventory() -> ModelInventory:
    manifest = load_manifest()
    return ModelInventory(
        default_pose_model=manifest.default_pose_model,
        models=[describe_model(entry, models_dir()) for entry in manifest.models],
    )


def download_artifact(
    entry: ModelEntry,
    target: Path,
    *,
    reporter: ProgressReporter | None = None,
    verify: bool = True,
) -> None:
    """Stage, verify, then replace. `verify=False` is CLI-only explicit re-pinning."""
    if not entry.url.startswith("https://"):
        raise ModelInstallError("Model downloads require an HTTPS manifest URL.")
    target.parent.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(ThrottledReporter(reporter or NullReporter()), task="install_model")
    progress.report("download", 0, entry.size_bytes, entry.name)
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as handle:
            partial = Path(handle.name)
            with urllib.request.urlopen(entry.url, timeout=20) as response:  # noqa: S310 - HTTPS manifest URL
                if not response.geturl().startswith("https://"):
                    raise ModelInstallError("Model download redirected away from HTTPS.")
                received = 0
                while chunk := response.read(256 * 1024):
                    received += len(chunk)
                    if verify and received > entry.size_bytes:
                        raise ModelInstallError("Download exceeds the pinned model size.")
                    handle.write(chunk)
                    progress.report("download", received, entry.size_bytes, entry.name)
        progress.report("verify", 0, 1, entry.name)
        if verify and (
            partial.stat().st_size != entry.size_bytes or sha256_file(partial) != entry.sha256
        ):
            raise ModelInstallError(
                "Downloaded size or SHA-256 differs from the pinned version; installed file preserved. "
                "Upstream may have changed. Update the application manifest deliberately."
            )
        partial.replace(target)
        progress.report("verify", 1, 1, entry.name)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def install_model(name: str, reporter: ProgressReporter) -> ModelInventory:
    manifest = load_manifest()
    entry = manifest.get(name)
    if entry is None:
        raise ModelInstallError(f"Unknown model '{name}'.")
    download_artifact(entry, models_dir() / entry.filename, reporter=reporter)
    return inventory()
