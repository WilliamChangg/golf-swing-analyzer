"""Model manifest loading and on-disk verification.

Model weights are not committed to git (they are tens of megabytes and are
redistributable artifacts, not source). The manifest -- which *is* committed --
pins the exact artifact each measurement in this repository was taken against,
by sha256.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyzer.contracts.health import ComponentStatus, HealthStatus
from analyzer.paths import model_manifest_path, models_dir

_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ModelEntry:
    """One model as described by models/manifest.json."""

    name: str
    kind: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    precision: str
    required: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelEntry:
        return cls(
            name=str(raw["name"]),
            kind=str(raw["kind"]),
            filename=str(raw["filename"]),
            url=str(raw["url"]),
            sha256=str(raw["sha256"]),
            size_bytes=int(raw["size_bytes"]),
            precision=str(raw.get("precision", "unknown")),
            required=bool(raw.get("required", False)),
        )


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    default_pose_model: str
    models: tuple[ModelEntry, ...]

    def get(self, name: str) -> ModelEntry | None:
        return next((m for m in self.models if m.name == name), None)


class ManifestError(RuntimeError):
    """The manifest is absent or malformed."""


def load_manifest(path: Path | None = None) -> ModelManifest:
    manifest_path = path or model_manifest_path()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"Model manifest not found at {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Model manifest at {manifest_path} is not valid JSON: {exc}") from exc

    try:
        return ModelManifest(
            schema_version=int(raw["schema_version"]),
            default_pose_model=str(raw["default_pose_model"]),
            models=tuple(ModelEntry.from_dict(m) for m in raw["models"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"Model manifest at {manifest_path} is malformed: {exc}") from exc


def sha256_file(path: Path) -> str:
    """Stream a file through sha256 rather than reading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _download_hint(entry: ModelEntry) -> str:
    return f"Run `python scripts/download_models.py --only {entry.name}` to fetch it."


def probe_model(entry: ModelEntry, directory: Path | None = None) -> ComponentStatus:
    """Check one model's presence, size, and hash against the manifest."""
    target = (directory or models_dir()) / entry.filename
    label = f"model:{entry.name}"

    if not target.exists():
        # An optional model that is simply not downloaded is not a problem.
        status = HealthStatus.MISSING if entry.required else HealthStatus.DEGRADED
        detail = f"{entry.filename} is not present in {target.parent}."
        if not entry.required:
            detail += " This model is optional."
        return ComponentStatus(
            name=label,
            status=status,
            detail=detail,
            remediation=_download_hint(entry),
        )

    actual_size = target.stat().st_size
    if actual_size != entry.size_bytes:
        return ComponentStatus(
            name=label,
            status=HealthStatus.ERROR,
            detail=(
                f"{entry.filename} is {actual_size} bytes but the manifest pins "
                f"{entry.size_bytes}. The file is truncated or was replaced."
            ),
            remediation=_download_hint(entry),
        )

    actual_hash = sha256_file(target)
    if actual_hash != entry.sha256:
        # Upstream publishes to a 'latest' channel, so a changed hash most likely
        # means a new upstream release rather than tampering. Either way the
        # artifact is no longer the one this project's numbers were measured
        # against, and saying so is the honest report.
        return ComponentStatus(
            name=label,
            status=HealthStatus.DEGRADED,
            detail=(
                f"{entry.filename} loads, but its sha256 ({actual_hash[:16]}...) differs "
                f"from the pinned artifact ({entry.sha256[:16]}...). Recorded benchmarks "
                "do not necessarily apply to this file."
            ),
            version=actual_hash[:16],
            remediation=(
                "Re-download to restore the pinned artifact, or update models/manifest.json "
                "deliberately and re-run the benchmarks."
            ),
        )

    return ComponentStatus(
        name=label,
        status=HealthStatus.OK,
        detail=f"{entry.filename} present and sha256 matches the manifest.",
        version=entry.sha256[:16],
    )


def probe_models(directory: Path | None = None) -> list[ComponentStatus]:
    """Probe every model in the manifest.

    A missing or malformed manifest is reported as a single ERROR component
    rather than raising, so the rest of the health report still renders.
    """
    try:
        manifest = load_manifest()
    except ManifestError as exc:
        return [
            ComponentStatus(
                name="model_manifest",
                status=HealthStatus.ERROR,
                detail=str(exc),
                remediation="Restore models/manifest.json from version control.",
            )
        ]

    return [probe_model(entry, directory) for entry in manifest.models]
