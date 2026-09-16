"""Model manifest and on-disk verification tests.

The failure modes that matter here are the quiet ones: a truncated download or a
silently-replaced artifact must not be reported as a healthy model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analyzer.contracts.health import HealthStatus
from analyzer.environment.models import (
    ManifestError,
    ModelEntry,
    load_manifest,
    probe_model,
    probe_models,
    sha256_file,
)
from analyzer.paths import ENV_MODELS_DIR, model_manifest_path

CONTENT = b"fake model bytes"
CONTENT_SHA = hashlib.sha256(CONTENT).hexdigest()


def _entry(
    *, required: bool = True, sha: str = CONTENT_SHA, size: int = len(CONTENT)
) -> ModelEntry:
    return ModelEntry(
        name="test_model",
        kind="pose_landmarker",
        filename="test_model.task",
        url="https://example.invalid/test_model.task",
        sha256=sha,
        size_bytes=size,
        precision="float16",
        required=required,
    )


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    (tmp_path / "test_model.task").write_bytes(CONTENT)
    return tmp_path


class TestSha256File:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        target.write_bytes(CONTENT)
        assert sha256_file(target) == CONTENT_SHA

    def test_handles_file_larger_than_chunk(self, tmp_path: Path) -> None:
        """The reader streams in 1 MiB chunks; verify multi-chunk files hash correctly."""
        payload = b"x" * (3 * 1024 * 1024 + 7)
        target = tmp_path / "big.bin"
        target.write_bytes(payload)
        assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


class TestProbeModel:
    def test_ok_when_size_and_hash_match(self, model_dir: Path) -> None:
        result = probe_model(_entry(), model_dir)
        assert result.status is HealthStatus.OK

    def test_missing_required_model_is_missing(self, tmp_path: Path) -> None:
        result = probe_model(_entry(required=True), tmp_path)
        assert result.status is HealthStatus.MISSING
        assert result.remediation is not None

    def test_missing_optional_model_is_only_degraded(self, tmp_path: Path) -> None:
        result = probe_model(_entry(required=False), tmp_path)
        assert result.status is HealthStatus.DEGRADED
        assert "optional" in result.detail

    def test_truncated_file_is_error(self, model_dir: Path) -> None:
        """A wrong size means a broken download; that is an error, not a warning."""
        result = probe_model(_entry(size=len(CONTENT) + 100), model_dir)
        assert result.status is HealthStatus.ERROR
        assert "truncated" in result.detail

    def test_hash_mismatch_is_degraded_and_says_so(self, model_dir: Path) -> None:
        """Upstream publishes to a 'latest' channel, so a changed hash is
        plausible without tampering -- but the file is no longer the artifact
        the recorded measurements were taken against, and the report says that.
        """
        result = probe_model(_entry(sha="0" * 64), model_dir)
        assert result.status is HealthStatus.DEGRADED
        assert "differs from the pinned artifact" in result.detail


class TestLoadManifest:
    def test_loads_the_real_committed_manifest(self) -> None:
        manifest = load_manifest()
        assert manifest.schema_version == 1
        assert manifest.get(manifest.default_pose_model) is not None

    def test_every_manifest_entry_has_a_64_char_sha(self) -> None:
        for entry in load_manifest().models:
            assert len(entry.sha256) == 64, entry.name
            assert entry.size_bytes > 0, entry.name

    def test_exactly_one_required_pose_model(self) -> None:
        required = [m for m in load_manifest().models if m.required]
        assert len(required) == 1

    def test_missing_file_raises_manifest_error(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="not found"):
            load_manifest(tmp_path / "absent.json")

    def test_malformed_json_raises_manifest_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "manifest.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ManifestError, match="not valid JSON"):
            load_manifest(bad)

    def test_missing_key_raises_manifest_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "manifest.json"
        bad.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        with pytest.raises(ManifestError, match="malformed"):
            load_manifest(bad)


class TestProbeModels:
    def test_reports_error_component_when_manifest_is_unreadable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken manifest must not abort the whole health report."""
        monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path))
        assert model_manifest_path() == tmp_path / "manifest.json"

        results = probe_models(tmp_path)
        assert len(results) == 1
        assert results[0].name == "model_manifest"
        assert results[0].status is HealthStatus.ERROR
