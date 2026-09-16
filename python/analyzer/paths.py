"""Filesystem layout resolution.

Resolving these in one place keeps path assumptions from spreading through the
codebase, and gives tests a single seam to redirect (via the environment
variables below) instead of monkeypatching call sites.
"""

from __future__ import annotations

import os
from pathlib import Path

# Marker files that identify the repository root when walking upwards.
_ROOT_MARKERS = ("models/manifest.json", ".git")

ENV_REPO_ROOT = "GSA_REPO_ROOT"
ENV_MODELS_DIR = "GSA_MODELS_DIR"


def repo_root() -> Path:
    """Locate the repository root.

    Honours ``GSA_REPO_ROOT`` first so a packaged build can point at a different
    layout without code changes. Otherwise walks up from this file looking for a
    marker, and falls back to the known source layout (``<root>/python/analyzer``).
    """
    override = os.environ.get(ENV_REPO_ROOT)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate

    # <root>/python/analyzer/paths.py -> parents[2] is <root>
    return here.parents[2]


def models_dir() -> Path:
    override = os.environ.get(ENV_MODELS_DIR)
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "models"


def model_manifest_path() -> Path:
    return models_dir() / "manifest.json"
