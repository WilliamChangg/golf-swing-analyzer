"""SQLite-backed storage for projects, their clips and their alignments.

The first state in this engine that cannot be recomputed. Everything else is
either input the user already has or a derived artifact under `cache_dir()` that
exists to save time; a project records a *decision* -- these two clips are one
swing, filmed from here and from there -- and nothing can recover that from the
files once it is lost. It therefore lives under `data_dir()`, which backup tools
do not skip, and in a database rather than a directory of JSON.

## Why SQLite rather than files

Three reasons, in descending order of how much they matter.

Referential integrity is enforced rather than re-checked: a sync cannot name a
clip that is not in its project, because a foreign key says so. Deleting a
project takes its clips and syncs with it, in one statement, and cannot half
fail. And a write is atomic against a reader in another process -- the desktop
app and a terminal running the CLI are two processes, and a directory of JSON
files gives a reader no way to tell a complete write from a partial one.

## Why the sync is stored as JSON inside a row

A `SyncModel` carries anchors, residuals, a correlation report and a decomposed
confidence -- a nested document, always read whole and never queried by part.
Normalising it would produce four tables to keep in step with one Pydantic model,
and the model is already the authoritative definition of the shape. So the
contract is the schema, and its own `schema_version` rides in the stored text.

The column beside it repeats that version so a stale row can be *found* without
being parsed, which is what makes the refusal below cheap and total: a sync
written by a future build is skipped and reported, never half-read into today's
fields.

## Identity

A clip is identified by content, not by path. `UNIQUE(project_id, content_key)`
means the same footage cannot enter one project twice under two names, and a file
that has been moved still matches its own cached extraction. The path is kept
because it is how the file is opened, and `ProjectClip.exists` reports whether it
still resolves -- a project with a moved clip is correct and incomplete, which is
a different thing from corrupt.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from pydantic import ValidationError

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.calibration import CALIBRATION_SCHEMA_VERSION, CameraRig
from analyzer.contracts.projects import (
    PROJECTS_SCHEMA_VERSION,
    CameraRole,
    Project,
    ProjectClip,
    ProjectList,
    ProjectSync,
)
from analyzer.contracts.sync import SYNC_SCHEMA_VERSION, SyncModel
from analyzer.paths import projects_database_path

_SCHEMA = """
CREATE TABLE projects (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    notes       TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL
);

CREATE TABLE clips (
    id                  INTEGER PRIMARY KEY,
    project_id          INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    role                TEXT    NOT NULL,
    path                TEXT    NOT NULL,
    content_key         TEXT    NOT NULL,
    content_algorithm   TEXT    NOT NULL,
    content_size_bytes  INTEGER NOT NULL,
    slow_motion_factor  REAL    NOT NULL DEFAULT 1.0,
    label               TEXT    NOT NULL DEFAULT '',
    added_at            TEXT    NOT NULL,
    UNIQUE(project_id, content_key)
);

CREATE TABLE syncs (
    id                  INTEGER PRIMARY KEY,
    project_id          INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    reference_clip_id   INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    target_clip_id      INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    sync_schema_version INTEGER NOT NULL,
    model_json          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    UNIQUE(project_id, reference_clip_id, target_clip_id)
);

CREATE INDEX clips_by_project ON clips(project_id);
CREATE INDEX syncs_by_project ON syncs(project_id);
"""

# Added in Phase 8, and separated from `_SCHEMA` so that creating a fresh
# database and upgrading an existing one run exactly the same statement rather
# than two that have to be kept saying the same thing.
_RIGS_TABLE = """
CREATE TABLE rigs (
    project_id                 INTEGER PRIMARY KEY
                               REFERENCES projects(id) ON DELETE CASCADE,
    calibration_schema_version INTEGER NOT NULL,
    model_json                 TEXT    NOT NULL,
    updated_at                 TEXT    NOT NULL
);
"""


class ProjectError(ValueError):
    """A project operation could not be carried out.

    Carries a remediation where one exists, matching `ProbeError`, so the same
    message and the same suggested fix reach the CLI and the desktop app.
    """

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


def _now() -> datetime:
    return datetime.now(UTC)


class ProjectStore:
    """Open connection to the projects database.

    Use as a context manager. The connection is opened lazily on first use so
    constructing a store against a path that does not exist yet -- which is every
    first run -- does not create a file until something is written or read.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or projects_database_path()
        self._connection: sqlite3.Connection | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> ProjectStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        # Off by default in SQLite, and per connection rather than per database.
        # Without it every REFERENCES clause above is documentation.
        connection.execute("PRAGMA foreign_keys = ON")
        # A reader in the desktop app and a writer in a terminal are two
        # processes; WAL is what stops one blocking the other.
        connection.execute("PRAGMA journal_mode = WAL")

        self._migrate(connection)
        self._connection = connection
        return connection

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Create the schema, or bring an older one forward, or refuse a newer one.

        Upgrades run in order and each one is additive, which is what makes them
        safe to apply to a database someone is relying on. A project is the only
        state in this engine that cannot be recomputed from the files, so an
        older database is migrated rather than refused -- refusing it would be
        asking the user to throw away the one thing they cannot get back.

        A database from a *newer* build is still refused, by name. That
        direction cannot be handled: this build does not know what changed, and
        reading tomorrow's rows into today's fields is how a stale assumption
        becomes a silent wrong answer.
        """
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])

        if version == 0:
            with connection:
                connection.executescript(_SCHEMA)
                connection.executescript(_RIGS_TABLE)
                connection.execute(f"PRAGMA user_version = {PROJECTS_SCHEMA_VERSION}")
            return

        if version > PROJECTS_SCHEMA_VERSION:
            raise ProjectError(
                f"{self.path} was written with project schema version {version}; this build "
                f"understands version {PROJECTS_SCHEMA_VERSION}.",
                remediation=(
                    "Use the version of the app that wrote it, or move the file aside to "
                    "start a fresh database."
                ),
            )

        if version < PROJECTS_SCHEMA_VERSION:
            with connection:
                if version < 2:
                    connection.executescript(_RIGS_TABLE)
                connection.execute(f"PRAGMA user_version = {PROJECTS_SCHEMA_VERSION}")

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        with connection:
            yield connection

    # -- projects ----------------------------------------------------------

    def create_project(self, name: str, *, notes: str = "") -> Project:
        """Create an empty project. Names are not unique; the id is the identity."""
        cleaned = name.strip()
        if not cleaned:
            raise ProjectError(
                "A project needs a name.",
                remediation="Pass a name, for example the session date and the player.",
            )

        created = _now()
        with self._write() as connection:
            cursor = connection.execute(
                "INSERT INTO projects (name, notes, created_at) VALUES (?, ?, ?)",
                (cleaned, notes, created.isoformat()),
            )
        return Project(id=int(cursor.lastrowid or 0), name=cleaned, notes=notes, created_at=created)

    def delete_project(self, project_id: int) -> None:
        """Delete a project, its clips and its syncs. The video files are untouched."""
        with self._write() as connection:
            cursor = connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        if cursor.rowcount == 0:
            raise ProjectError(f"No project with id {project_id}.")

    def get_project(self, project_id: int) -> Project:
        connection = self._connect()
        row = connection.execute(
            "SELECT id, name, notes, created_at FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row is None:
            raise ProjectError(
                f"No project with id {project_id}.",
                remediation="Run `analyzer project list` to see which projects exist.",
            )
        return self._hydrate(connection, row, with_syncs=True)

    def list_projects(self) -> ProjectList:
        connection = self._connect()
        rows = connection.execute(
            "SELECT id, name, notes, created_at FROM projects ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return ProjectList(
            projects=[self._hydrate(connection, row, with_syncs=False) for row in rows],
            database_path=str(self.path),
        )

    def _hydrate(
        self, connection: sqlite3.Connection, row: sqlite3.Row, *, with_syncs: bool
    ) -> Project:
        clips = [
            self._clip(entry)
            for entry in connection.execute(
                "SELECT * FROM clips WHERE project_id = ? ORDER BY id", (row["id"],)
            )
        ]

        syncs: list[ProjectSync] = []
        rig: CameraRig | None = None
        warnings: list[str] = []
        if with_syncs:
            syncs, warnings = self._syncs(connection, int(row["id"]))
            rig, rig_warnings = self._rig(connection, int(row["id"]))
            warnings.extend(rig_warnings)

        return Project(
            id=int(row["id"]),
            name=str(row["name"]),
            notes=str(row["notes"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            clips=clips,
            syncs=syncs,
            rig=rig,
            warnings=warnings,
        )

    @staticmethod
    def _clip(row: sqlite3.Row) -> ProjectClip:
        path = Path(str(row["path"]))
        return ProjectClip(
            id=int(row["id"]),
            role=CameraRole(str(row["role"])),
            path=str(path),
            name=path.name,
            content_key=ContentKey(
                algorithm=HashAlgorithm(str(row["content_algorithm"])),
                digest=str(row["content_key"]),
                size_bytes=int(row["content_size_bytes"]),
            ),
            slow_motion_factor=float(row["slow_motion_factor"]),
            added_at=datetime.fromisoformat(str(row["added_at"])),
            exists=path.is_file(),
            label=str(row["label"]),
        )

    @staticmethod
    def _syncs(
        connection: sqlite3.Connection, project_id: int
    ) -> tuple[list[ProjectSync], list[str]]:
        """Read the stored alignments, skipping any this build cannot read.

        The version check happens against the column rather than against the
        parsed document, so a row from a future build never reaches
        `model_validate_json` -- where it would either fail with a message about
        a field name or, worse, succeed by ignoring fields that changed meaning.
        """
        stored: list[ProjectSync] = []
        warnings: list[str] = []

        for row in connection.execute(
            "SELECT * FROM syncs WHERE project_id = ? ORDER BY id", (project_id,)
        ):
            version = int(row["sync_schema_version"])
            if version != SYNC_SCHEMA_VERSION:
                warnings.append(
                    f"The stored alignment between clips {row['reference_clip_id']} and "
                    f"{row['target_clip_id']} was written under sync schema version "
                    f"{version}; this build reads version {SYNC_SCHEMA_VERSION}. It was left "
                    "out. Re-run the synchronisation to replace it."
                )
                continue
            try:
                model = SyncModel.model_validate_json(str(row["model_json"]))
            except ValidationError as exc:
                warnings.append(
                    f"The stored alignment between clips {row['reference_clip_id']} and "
                    f"{row['target_clip_id']} did not parse and was left out: {exc.error_count()} "
                    "field(s) did not match the contract. Re-run the synchronisation."
                )
                continue
            stored.append(
                ProjectSync(
                    reference_clip_id=int(row["reference_clip_id"]),
                    target_clip_id=int(row["target_clip_id"]),
                    model=model,
                    updated_at=datetime.fromisoformat(str(row["updated_at"])),
                )
            )

        return stored, warnings

    # -- clips -------------------------------------------------------------

    def add_clip(
        self,
        project_id: int,
        *,
        path: Path,
        content_key: ContentKey,
        role: CameraRole,
        slow_motion_factor: float = 1.0,
        label: str = "",
    ) -> Project:
        """Attach a clip to a project. The content key comes from probing the file.

        Takes the key rather than deriving it so this layer never opens a video:
        probing is ingestion's job, it is the thing that can fail in interesting
        ways, and its failures already carry remediations this store would only
        be able to flatten.
        """
        if slow_motion_factor <= 0:
            raise ProjectError("A slow-motion factor must be greater than zero.")

        try:
            with self._write() as connection:
                connection.execute(
                    """
                    INSERT INTO clips (
                        project_id, role, path, content_key, content_algorithm,
                        content_size_bytes, slow_motion_factor, label, added_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        role.value,
                        str(path),
                        content_key.digest,
                        content_key.algorithm.value,
                        content_key.size_bytes,
                        slow_motion_factor,
                        label,
                        _now().isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            # Two distinct causes reach here and they need different answers.
            if "FOREIGN KEY" in str(exc):
                raise ProjectError(
                    f"No project with id {project_id}.",
                    remediation="Run `analyzer project list` to see which projects exist.",
                ) from exc
            raise ProjectError(
                f"{path.name} is already in project {project_id}. Clips are identified by "
                "content, so the same footage cannot be added twice even under another name.",
                remediation="Remove the existing clip first if you meant to change its role.",
            ) from exc

        return self.get_project(project_id)

    def remove_clip(self, project_id: int, clip_id: int) -> Project:
        """Detach a clip. Its syncs go with it, by foreign key."""
        with self._write() as connection:
            cursor = connection.execute(
                "DELETE FROM clips WHERE id = ? AND project_id = ?", (clip_id, project_id)
            )
        if cursor.rowcount == 0:
            raise ProjectError(f"Project {project_id} has no clip with id {clip_id}.")
        return self.get_project(project_id)

    def set_clip_path(self, project_id: int, clip_id: int, path: Path) -> Project:
        """Re-point a clip at a moved file, keeping its recorded content key.

        The key is deliberately *not* recomputed. A clip whose content changed is
        a different clip and should be added as one; silently accepting new bytes
        under an existing id would leave every stored alignment describing
        footage that is no longer there.
        """
        with self._write() as connection:
            cursor = connection.execute(
                "UPDATE clips SET path = ? WHERE id = ? AND project_id = ?",
                (str(path), clip_id, project_id),
            )
        if cursor.rowcount == 0:
            raise ProjectError(f"Project {project_id} has no clip with id {clip_id}.")
        return self.get_project(project_id)

    # -- syncs -------------------------------------------------------------

    def save_sync(
        self,
        project_id: int,
        *,
        reference_clip_id: int,
        target_clip_id: int,
        model: SyncModel,
    ) -> Project:
        """Store an alignment, replacing any previous one for the same ordered pair.

        Ordered: a sync from A to B is stored separately from one from B to A.
        They carry the same information and each is invertible, but keeping the
        stored direction explicit means nothing has to invert a map to read it,
        and inverting an affine map with an uncertainty is exactly the kind of
        arithmetic that acquires a sign error.
        """
        if reference_clip_id == target_clip_id:
            raise ProjectError("A clip cannot be synchronised against itself.")

        project = self.get_project(project_id)
        for clip_id in (reference_clip_id, target_clip_id):
            if project.clip_by_id(clip_id) is None:
                raise ProjectError(f"Project {project_id} has no clip with id {clip_id}.")

        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO syncs (
                    project_id, reference_clip_id, target_clip_id,
                    sync_schema_version, model_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, reference_clip_id, target_clip_id)
                DO UPDATE SET
                    sync_schema_version = excluded.sync_schema_version,
                    model_json = excluded.model_json,
                    updated_at = excluded.updated_at
                """,
                (
                    project_id,
                    reference_clip_id,
                    target_clip_id,
                    model.schema_version,
                    json.dumps(model.model_dump(mode="json")),
                    _now().isoformat(),
                ),
            )
        return self.get_project(project_id)

    # -- calibration -------------------------------------------------------

    @staticmethod
    def _rig(connection: sqlite3.Connection, project_id: int) -> tuple[CameraRig | None, list[str]]:
        """Read the stored rig, skipping it if this build cannot read it.

        Same shape as `_syncs`, and for the same reason: the version is checked
        against the column rather than against the parsed document, so a rig
        written by a future build never reaches `model_validate_json`, where it
        would either fail on a field name or succeed by ignoring a field whose
        meaning changed.

        A stale rig costs the project its calibration and nothing else. That is
        the right blast radius -- an uncalibrated project is a supported state,
        and losing the metric-scale claims is much better than making them from
        a document this build is guessing at.
        """
        row = connection.execute(
            "SELECT * FROM rigs WHERE project_id = ?", (project_id,)
        ).fetchone()
        if row is None:
            return None, []

        version = int(row["calibration_schema_version"])
        if version != CALIBRATION_SCHEMA_VERSION:
            return None, [
                f"The stored camera calibration was written under calibration schema "
                f"version {version}; this build reads version "
                f"{CALIBRATION_SCHEMA_VERSION}. It was left out, so this project is "
                "treated as uncalibrated. Re-run the calibration to replace it."
            ]

        try:
            return CameraRig.model_validate_json(str(row["model_json"])), []
        except ValidationError as exc:
            return None, [
                f"The stored camera calibration did not parse and was left out: "
                f"{exc.error_count()} field(s) did not match the contract. This project is "
                "treated as uncalibrated. Re-run the calibration."
            ]

    def save_rig(self, project_id: int, rig: CameraRig) -> Project:
        """Store what is known about this project's cameras, replacing any previous rig.

        Written whole rather than merged, because the parts are not independent:
        `CameraRig.status` is `STEREO` only when both intrinsics and the
        extrinsics between them agree about which cameras they describe, and
        updating one camera's intrinsics without reconsidering the extrinsics
        fitted against the old ones would leave a rig whose three pieces
        describe two different setups. Callers that mean to change one camera
        read the rig, change it, and write it back -- which makes the question
        "is the stereo pose still valid?" one they have to answer.
        """
        self.get_project(project_id)

        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO rigs (
                    project_id, calibration_schema_version, model_json, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    calibration_schema_version = excluded.calibration_schema_version,
                    model_json = excluded.model_json,
                    updated_at = excluded.updated_at
                """,
                (
                    project_id,
                    rig.schema_version,
                    json.dumps(rig.model_dump(mode="json")),
                    _now().isoformat(),
                ),
            )
        return self.get_project(project_id)

    def clear_rig(self, project_id: int) -> Project:
        """Forget this project's calibration. The footage is untouched."""
        with self._write() as connection:
            connection.execute("DELETE FROM rigs WHERE project_id = ?", (project_id,))
        return self.get_project(project_id)
