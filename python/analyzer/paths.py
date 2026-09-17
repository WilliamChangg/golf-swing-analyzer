"""Filesystem layout resolution.

Resolving these in one place keeps path assumptions from spreading through the
codebase, and gives tests a single seam to redirect (via the environment
variables below) instead of monkeypatching call sites.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Marker files that identify the repository root when walking upwards.
_ROOT_MARKERS = ("models/manifest.json", ".git")

ENV_REPO_ROOT = "GSA_REPO_ROOT"
ENV_MODELS_DIR = "GSA_MODELS_DIR"
ENV_CACHE_DIR = "GSA_CACHE_DIR"
ENV_DATA_DIR = "GSA_DATA_DIR"

_APP_DIR_NAME = "golf-swing-analyzer"


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


def cache_dir() -> Path:
    """Directory for derived artifacts that can be recomputed from source video.

    Deliberately not inside ``data/``: that holds the user's own footage, and
    filling it with derived files makes it harder to see what is irreplaceable.
    The platform cache location is also the one backup tools already know to
    skip, which is the correct treatment for something regenerable.
    """
    override = os.environ.get(ENV_CACHE_DIR)
    if override:
        return Path(override).expanduser().resolve()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / _APP_DIR_NAME

    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / _APP_DIR_NAME


def data_dir() -> Path:
    """Directory for state that cannot be recomputed from source video.

    The counterpart to ``cache_dir()``, and the distinction is the point: a
    project records which two clips are the same swing, which is a decision
    rather than a derivation. Putting it in the cache location would file
    irreplaceable state under the one directory every backup tool is told to
    skip, and the first time a user cleared their caches the projects would go
    with them.
    """
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override).expanduser().resolve()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_DIR_NAME

    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / _APP_DIR_NAME


def projects_database_path() -> Path:
    """The SQLite file holding projects, clips and stored syncs."""
    return data_dir() / "projects.db"
