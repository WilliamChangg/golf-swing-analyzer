"""Typed contracts for the project model.

A project is the first mutable, user-owned state in this engine. Everything
before it is either input the user already has (video files) or output that can
be thrown away and recomputed (pose Parquet, metadata cache). A project is
neither: it records a decision -- *these two clips are the same swing, filmed
from these two positions* -- which no amount of computation can recover once it
is lost.

That difference decides where it lives. Derived artifacts go under `cache_dir()`,
which backup tools skip on purpose. Projects go under `data_dir()`, which they do
not, and in SQLite rather than a directory of JSON files because two clips, a
sync and later a calibration form a graph with referential integrity worth
enforcing rather than re-checking at every read.

## Clips are identified twice

Every clip carries both a `path` and a `content_key`. The path is how the file is
opened and is the thing that breaks -- footage gets moved, renamed, and copied
between machines. The content key survives all of that and is what every derived
artifact is already filed under, so a clip whose path has gone stale can still be
matched against its own extracted poses. A project therefore reports a missing
file as a state rather than failing to load, because the project is still correct
and only one of its references is not.

## The pair, and why it is not a list of two

`CameraRole` is on the clip rather than implied by position, because "the first
one" is not a fact about a recording and "the face-on one" is. Phase 6 measures
the view from the footage and can disagree with what the user declared, which is
a useful disagreement to be able to state; it cannot state it if the declared
role was never recorded.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from analyzer.contracts.cache import ContentKey
from analyzer.contracts.calibration import CameraRig
from analyzer.contracts.camera import CameraRole
from analyzer.contracts.sync import SyncModel

__all__ = [
    "PROJECTS_SCHEMA_VERSION",
    "CameraRole",
    "Project",
    "ProjectClip",
    "ProjectList",
    "ProjectSync",
]

# Bump on any change to the stored shape. The store compares this against
# SQLite's `user_version` and refuses a database it does not understand, rather
# than reading old rows into new fields.
PROJECTS_SCHEMA_VERSION = 2
# 1 -> 2: Phase 8 added the `rigs` table, holding one `CameraRig` per project.
# Migrated rather than refused, because a project is the one thing in this engine
# that cannot be recomputed from the files.


class ProjectClip(BaseModel):
    """One video belonging to a project."""

    id: int
    role: CameraRole
    path: str
    name: str = Field(description="Basename, for display.")
    content_key: ContentKey
    slow_motion_factor: float = Field(
        gt=0.0,
        description=(
            "How many times slower than real time this clip plays, as supplied "
            "when it was added. Stored per clip because two cameras in one "
            "session routinely differ -- one phone at 240 fps and one at 30."
        ),
    )
    added_at: datetime
    exists: bool = Field(
        description=(
            "Whether a file is still at `path`. False does not invalidate the "
            "project: the content key still identifies the clip and its cached "
            "extraction, and the fix is to point the project at the moved file."
        )
    )
    label: str = Field(default="", description="Free text the user attached. Never interpreted.")


class ProjectSync(BaseModel):
    """A stored alignment between two of a project's clips.

    The clip ids are here as well as inside the model because the model
    identifies its clips by path, and a project's clips can be re-pointed at
    moved files. The ids are what survive that.
    """

    reference_clip_id: int
    target_clip_id: int
    model: SyncModel
    updated_at: datetime


class Project(BaseModel):
    """A session: the clips of one swing, and how their clocks relate.

    `clips` is a list rather than a fixed pair. Two is what Phases 7 to 9 need
    and what the capture protocol asks for, but nothing here is made harder by
    allowing three, and a fixed pair would have to be widened later by changing
    the stored shape rather than by adding a row.
    """

    schema_version: int = PROJECTS_SCHEMA_VERSION
    id: int
    name: str
    notes: str = ""
    created_at: datetime
    clips: list[ProjectClip] = Field(default_factory=list)
    syncs: list[ProjectSync] = Field(default_factory=list)
    rig: CameraRig | None = Field(
        default=None,
        description=(
            "What is known about this project's cameras: their intrinsics, and "
            "their relative pose where it has been measured. None until a "
            "calibration is stored, which is the ordinary state -- an "
            "uncalibrated project is fully supported and simply makes no "
            "metric-scale claims.\n\n"
            "One rig per project rather than one per clip, because extrinsics "
            "relate two cameras and belong to neither. It is read whole and "
            "never queried by part, which is why it is stored as a document."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Facts about reading this project back. A stored sync written under "
            "a schema version this build does not understand is reported here "
            "and left out of `syncs`, rather than making the whole project "
            "unopenable over one stale row."
        ),
    )

    def clip(self, role: CameraRole) -> ProjectClip | None:
        """The first clip with a given role, or None. Roles are not unique by construction."""
        return next((entry for entry in self.clips if entry.role is role), None)

    def clip_by_id(self, clip_id: int) -> ProjectClip | None:
        return next((entry for entry in self.clips if entry.id == clip_id), None)


class ProjectList(BaseModel):
    """Every project, with its clips but without its syncs.

    Syncs are omitted here on purpose: a listing is for choosing which project to
    open, and a stored `SyncModel` carries anchors, residuals and a correlation
    report, which would make the list of a dozen projects an order of magnitude
    larger than the thing being chosen from.
    """

    schema_version: int = PROJECTS_SCHEMA_VERSION
    projects: list[Project] = Field(default_factory=list)
    database_path: str = Field(
        description="Where these were read from. Useful when one is missing."
    )
