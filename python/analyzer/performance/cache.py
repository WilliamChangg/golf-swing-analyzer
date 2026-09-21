"""Content- and configuration-addressed cache for small analysis reports.

Pose Parquet is already a content-addressed cache. This cache deliberately
holds only compact JSON result contracts (phases, metrics, coaching), not frame
arrays: it makes repeated analysis clicks cheap without turning cache
invalidation into a second persistence format for every numerical intermediate.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from analyzer.contracts.cache import ContentKey
from analyzer.paths import cache_dir

_CACHE_VERSION = 1
_Model = TypeVar("_Model", bound=BaseModel)
_DISABLE_CACHE_ENV = "GSA_DISABLE_ANALYSIS_CACHE"


def config_digest(config: Any) -> str:
    """Hash canonical JSON, so dict order cannot cause a cache miss."""
    if isinstance(config, BaseModel):
        config = config.model_dump(mode="json")
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AnalysisCache:
    """Best-effort persistent cache for validated Pydantic result models.

    A corrupt, stale, or unwritable entry is a miss. Cache health must never
    decide whether an analysis can run.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory or (cache_dir() / "analysis-results")

    def _path(self, content: ContentKey, namespace: str, config: Any) -> Path:
        if not namespace.replace("_", "").isalnum():
            raise ValueError(f"Invalid cache namespace {namespace!r}.")
        return (
            self._directory
            / content.as_path_segment()
            / f"{namespace}-{config_digest(config)[:24]}.json"
        )

    def load(
        self, content: ContentKey, namespace: str, config: Any, model: type[_Model]
    ) -> _Model | None:
        if os.environ.get(_DISABLE_CACHE_ENV) == "1":
            return None
        path = self._path(content, namespace, config)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            if (
                raw.get("cache_version") != _CACHE_VERSION
                or raw.get("content_key") != content.model_dump(mode="json")
                or raw.get("namespace") != namespace
                or raw.get("config_digest") != config_digest(config)
            ):
                return None
            return model.model_validate(raw["result"])
        except (KeyError, OSError, TypeError, ValueError):
            return None

    def store(self, content: ContentKey, namespace: str, config: Any, result: BaseModel) -> None:
        if os.environ.get(_DISABLE_CACHE_ENV) == "1":
            return
        path = self._path(content, namespace, config)
        payload = {
            "cache_version": _CACHE_VERSION,
            "content_key": content.model_dump(mode="json"),
            "namespace": namespace,
            "config_digest": config_digest(config),
            "result": result.model_dump(mode="json"),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            return
