"""Projects: the clips of one swing, and how their clocks relate.

The engine's only durable, user-owned state. See `store.py` for why it is SQLite
under `data_dir()` rather than JSON under `cache_dir()`.
"""

from analyzer.projects.store import ProjectError, ProjectStore

__all__ = ["ProjectError", "ProjectStore"]
