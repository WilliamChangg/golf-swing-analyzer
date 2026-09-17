"""Project store tests.

The store holds the only state in this engine that cannot be recomputed, so the
tests are weighted towards the ways that state can be silently lost or corrupted
rather than towards the happy path: a database from a version this build does
not understand, a stored alignment whose contract has moved on, a clip whose
file has been moved out from under it, and the same footage added twice under
two names.

Every test runs against a database in `tmp_path`. None of them touch the real
one under `data_dir()`, which is the point of `ProjectStore` taking a path.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.projects import PROJECTS_SCHEMA_VERSION, CameraRole
from analyzer.contracts.sync import (
    SYNC_SCHEMA_VERSION,
    ClipRef,
    SyncConfig,
    SyncMethod,
    SyncModel,
    TimeMap,
)
from analyzer.paths import ENV_DATA_DIR, data_dir, projects_database_path
from analyzer.projects import ProjectError, ProjectStore


def key(digest: str = "a", size: int = 1024) -> ContentKey:
    return ContentKey(algorithm=HashAlgorithm.SHA256_SAMPLED, digest=digest * 64, size_bytes=size)


def clip_ref(name: str) -> ClipRef:
    return ClipRef(
        path=f"/data/{name}",
        name=name,
        frames=120,
        start_s=0.0,
        duration_s=4.0,
        median_interval_s=1 / 30,
        slow_motion_factor=1.0,
    )


def sync_model(offset_s: float = 0.25) -> SyncModel:
    return SyncModel(
        aligned=True,
        method=SyncMethod.EVENTS,
        reference=clip_ref("face-on.mov"),
        target=clip_ref("dtl.mov"),
        time_map=TimeMap(
            offset_s=offset_s,
            rate=1.0,
            rate_estimated=False,
            pivot_s=1.5,
            offset_uncertainty_s=0.004,
            support_start_s=0.5,
            support_end_s=2.5,
        ),
        config=SyncConfig(),
    )


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    with ProjectStore(tmp_path / "projects.db") as opened:
        yield opened


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "face-on.mov"
    path.write_bytes(b"not really a video")
    return path


class TestLocation:
    """Projects are not derived data and must not live where derived data lives."""

    def test_the_database_is_not_under_the_cache(self, monkeypatch) -> None:
        """Checked against the real defaults, with the suite's redirects removed.

        The autouse fixtures point both directories inside one `tmp_path`, which
        would make this pass whatever the platform layout said.
        """
        from analyzer.paths import ENV_CACHE_DIR, cache_dir

        monkeypatch.delenv(ENV_CACHE_DIR, raising=False)
        monkeypatch.delenv(ENV_DATA_DIR, raising=False)

        assert cache_dir() not in projects_database_path().parents
        assert cache_dir() != projects_database_path().parent

    def test_the_data_directory_is_overridable(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "elsewhere"))
        assert data_dir() == (tmp_path / "elsewhere").resolve()
        assert projects_database_path().parent == (tmp_path / "elsewhere").resolve()

    def test_no_file_is_created_until_the_store_is_used(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "projects.db"
        ProjectStore(path)
        assert not path.exists()


class TestProjects:
    def test_a_created_project_has_an_id_and_no_clips(self, store: ProjectStore) -> None:
        project = store.create_project("Tuesday range session")

        assert project.id > 0
        assert project.name == "Tuesday range session"
        assert project.clips == []
        assert project.syncs == []

    def test_projects_round_trip(self, store: ProjectStore) -> None:
        created = store.create_project("Session", notes="wedge work")
        read = store.get_project(created.id)

        assert read.name == created.name
        assert read.notes == "wedge work"
        assert read.created_at == created.created_at

    def test_listing_is_newest_first(self, store: ProjectStore) -> None:
        first = store.create_project("First")
        second = store.create_project("Second")

        listed = store.list_projects()
        assert [entry.id for entry in listed.projects][:2] == [second.id, first.id]
        assert listed.database_path == str(store.path)

    def test_an_unnamed_project_is_refused_with_a_remedy(self, store: ProjectStore) -> None:
        with pytest.raises(ProjectError) as caught:
            store.create_project("   ")
        assert caught.value.remediation is not None

    def test_an_unknown_id_names_the_command_that_lists_them(self, store: ProjectStore) -> None:
        with pytest.raises(ProjectError, match="No project with id 404"):
            store.get_project(404)

    def test_deleting_takes_the_clips_with_it(self, store: ProjectStore, video: Path) -> None:
        project = store.create_project("Session")
        store.add_clip(project.id, path=video, content_key=key(), role=CameraRole.FACE_ON)

        store.delete_project(project.id)

        with pytest.raises(ProjectError):
            store.get_project(project.id)
        # The cascade is the database's job, and it only runs with foreign keys on.
        remaining = store._connect().execute("SELECT COUNT(*) FROM clips").fetchone()[0]
        assert remaining == 0

    def test_the_video_file_is_never_touched(self, store: ProjectStore, video: Path) -> None:
        project = store.create_project("Session")
        store.add_clip(project.id, path=video, content_key=key(), role=CameraRole.FACE_ON)
        store.delete_project(project.id)

        assert video.exists(), "deleting a project must not delete the footage"


class TestClips:
    def test_a_clip_carries_its_role_and_factor(self, store: ProjectStore, video: Path) -> None:
        project = store.create_project("Session")
        updated = store.add_clip(
            project.id,
            path=video,
            content_key=key(),
            role=CameraRole.FACE_ON,
            slow_motion_factor=8.0,
            label="iPhone, tripod",
        )

        clip = updated.clips[0]
        assert clip.role is CameraRole.FACE_ON
        assert clip.slow_motion_factor == 8.0
        assert clip.label == "iPhone, tripod"
        assert clip.name == "face-on.mov"

    def test_a_clip_is_identified_by_content_not_by_path(
        self, store: ProjectStore, video: Path, tmp_path: Path
    ) -> None:
        """The same footage under two names is the same clip, and is refused."""
        project = store.create_project("Session")
        store.add_clip(project.id, path=video, content_key=key(), role=CameraRole.FACE_ON)

        copy = tmp_path / "copy-of-face-on.mov"
        copy.write_bytes(video.read_bytes())

        with pytest.raises(ProjectError, match="already in project"):
            store.add_clip(project.id, path=copy, content_key=key(), role=CameraRole.DOWN_THE_LINE)

    def test_different_footage_may_share_a_role(
        self, store: ProjectStore, video: Path, tmp_path: Path
    ) -> None:
        """Roles are declarations, not a unique index. Two face-on angles are allowed."""
        project = store.create_project("Session")
        other = tmp_path / "second.mov"
        other.write_bytes(b"different")

        store.add_clip(project.id, path=video, content_key=key("a"), role=CameraRole.FACE_ON)
        updated = store.add_clip(
            project.id, path=other, content_key=key("b"), role=CameraRole.FACE_ON
        )

        assert len(updated.clips) == 2

    def test_adding_to_a_missing_project_is_refused(self, store: ProjectStore, video: Path) -> None:
        with pytest.raises(ProjectError, match="No project with id"):
            store.add_clip(999, path=video, content_key=key(), role=CameraRole.FACE_ON)

    def test_a_missing_file_is_a_state_not_a_failure(
        self, store: ProjectStore, video: Path
    ) -> None:
        project = store.create_project("Session")
        store.add_clip(project.id, path=video, content_key=key(), role=CameraRole.FACE_ON)
        video.unlink()

        read = store.get_project(project.id)
        assert read.clips[0].exists is False
        assert read.clips[0].content_key.digest == key().digest

    def test_repointing_keeps_the_recorded_content_key(
        self, store: ProjectStore, video: Path, tmp_path: Path
    ) -> None:
        """A clip whose bytes changed is a different clip, not a moved one."""
        project = store.create_project("Session")
        added = store.add_clip(project.id, path=video, content_key=key(), role=CameraRole.FACE_ON)
        moved = tmp_path / "archive" / "face-on.mov"
        moved.parent.mkdir()
        video.rename(moved)

        updated = store.set_clip_path(project.id, added.clips[0].id, moved)

        assert updated.clips[0].path == str(moved)
        assert updated.clips[0].exists is True
        assert updated.clips[0].content_key.digest == key().digest

    def test_removing_a_clip_that_is_not_there_is_refused(self, store: ProjectStore) -> None:
        project = store.create_project("Session")
        with pytest.raises(ProjectError, match="no clip with id"):
            store.remove_clip(project.id, 77)

    def test_clip_lookup_by_role_and_id(self, store: ProjectStore, video: Path) -> None:
        project = store.create_project("Session")
        updated = store.add_clip(
            project.id, path=video, content_key=key(), role=CameraRole.DOWN_THE_LINE
        )

        assert updated.clip(CameraRole.DOWN_THE_LINE) is not None
        assert updated.clip(CameraRole.FACE_ON) is None
        assert updated.clip_by_id(updated.clips[0].id) is not None
        assert updated.clip_by_id(-1) is None


class TestSyncs:
    @pytest.fixture
    def pair(self, store: ProjectStore, tmp_path: Path) -> tuple[int, int, int]:
        project = store.create_project("Session")
        for name, digest, role in (
            ("face-on.mov", "a", CameraRole.FACE_ON),
            ("dtl.mov", "b", CameraRole.DOWN_THE_LINE),
        ):
            path = tmp_path / name
            path.write_bytes(name.encode())
            store.add_clip(project.id, path=path, content_key=key(digest), role=role)

        clips = store.get_project(project.id).clips
        return project.id, clips[0].id, clips[1].id

    def test_a_stored_alignment_round_trips(
        self, store: ProjectStore, pair: tuple[int, int, int]
    ) -> None:
        project_id, reference, target = pair
        store.save_sync(
            project_id, reference_clip_id=reference, target_clip_id=target, model=sync_model()
        )

        stored = store.get_project(project_id).syncs
        assert len(stored) == 1
        assert stored[0].reference_clip_id == reference
        assert stored[0].model.time_map is not None
        assert stored[0].model.time_map.offset_s == pytest.approx(0.25)

    def test_saving_again_replaces_rather_than_accumulates(
        self, store: ProjectStore, pair: tuple[int, int, int]
    ) -> None:
        project_id, reference, target = pair
        for offset in (0.25, 0.40):
            store.save_sync(
                project_id,
                reference_clip_id=reference,
                target_clip_id=target,
                model=sync_model(offset),
            )

        stored = store.get_project(project_id).syncs
        assert len(stored) == 1
        assert stored[0].model.time_map is not None
        assert stored[0].model.time_map.offset_s == pytest.approx(0.40)

    def test_the_two_directions_are_stored_separately(
        self, store: ProjectStore, pair: tuple[int, int, int]
    ) -> None:
        project_id, reference, target = pair
        store.save_sync(
            project_id, reference_clip_id=reference, target_clip_id=target, model=sync_model()
        )
        updated = store.save_sync(
            project_id, reference_clip_id=target, target_clip_id=reference, model=sync_model(-0.25)
        )

        assert len(updated.syncs) == 2

    def test_a_clip_cannot_be_aligned_against_itself(
        self, store: ProjectStore, pair: tuple[int, int, int]
    ) -> None:
        project_id, reference, _ = pair
        with pytest.raises(ProjectError, match="against itself"):
            store.save_sync(
                project_id,
                reference_clip_id=reference,
                target_clip_id=reference,
                model=sync_model(),
            )

    def test_a_sync_naming_a_foreign_clip_is_refused(
        self, store: ProjectStore, pair: tuple[int, int, int]
    ) -> None:
        project_id, reference, _ = pair
        with pytest.raises(ProjectError, match="no clip with id"):
            store.save_sync(
                project_id, reference_clip_id=reference, target_clip_id=9999, model=sync_model()
            )

    def test_removing_a_clip_takes_its_alignments(
        self, store: ProjectStore, pair: tuple[int, int, int]
    ) -> None:
        project_id, reference, target = pair
        store.save_sync(
            project_id, reference_clip_id=reference, target_clip_id=target, model=sync_model()
        )

        updated = store.remove_clip(project_id, target)
        assert updated.syncs == []

    def test_listing_omits_syncs(self, store: ProjectStore, pair: tuple[int, int, int]) -> None:
        """A listing is for choosing what to open, not for carrying every residual."""
        project_id, reference, target = pair
        store.save_sync(
            project_id, reference_clip_id=reference, target_clip_id=target, model=sync_model()
        )

        listed = next(entry for entry in store.list_projects().projects if entry.id == project_id)
        assert listed.syncs == []
        assert listed.clips != []

    def test_a_refusal_round_trips_as_a_refusal(
        self, store: ProjectStore, pair: tuple[int, int, int]
    ) -> None:
        """`aligned: false` has to survive storage, or a refusal reads as an alignment at zero."""
        project_id, reference, target = pair
        refused = SyncModel(
            aligned=False,
            reference=clip_ref("face-on.mov"),
            target=clip_ref("dtl.mov"),
            config=SyncConfig(),
            refusal="The anchors cannot be reconciled.",
        )
        store.save_sync(
            project_id, reference_clip_id=reference, target_clip_id=target, model=refused
        )

        stored = store.get_project(project_id).syncs[0].model
        assert stored.aligned is False
        assert stored.time_map is None
        assert stored.refusal == "The anchors cannot be reconciled."


class TestSchemaRefusals:
    """A version this build does not understand is refused by name, never guessed at."""

    def test_a_future_database_is_refused_with_a_remedy(self, tmp_path: Path) -> None:
        path = tmp_path / "projects.db"
        with ProjectStore(path) as store:
            store.create_project("Session")

        connection = sqlite3.connect(path)
        connection.execute(f"PRAGMA user_version = {PROJECTS_SCHEMA_VERSION + 1}")
        connection.commit()
        connection.close()

        with (
            pytest.raises(ProjectError, match="project schema version") as caught,
            ProjectStore(path) as store,
        ):
            store.list_projects()
        assert caught.value.remediation is not None

    def test_a_fresh_database_stamps_its_version(self, tmp_path: Path) -> None:
        path = tmp_path / "projects.db"
        with ProjectStore(path) as store:
            store.create_project("Session")

        connection = sqlite3.connect(path)
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        connection.close()
        assert version == PROJECTS_SCHEMA_VERSION

    def test_a_stale_sync_is_skipped_and_reported(
        self, store: ProjectStore, tmp_path: Path
    ) -> None:
        """One unreadable alignment must not make the whole project unopenable."""
        project = store.create_project("Session")
        for name, digest, role in (
            ("a.mov", "a", CameraRole.FACE_ON),
            ("b.mov", "b", CameraRole.DOWN_THE_LINE),
        ):
            path = tmp_path / name
            path.write_bytes(name.encode())
            store.add_clip(project.id, path=path, content_key=key(digest), role=role)

        clips = store.get_project(project.id).clips
        store._connect().execute(
            """
            INSERT INTO syncs (project_id, reference_clip_id, target_clip_id,
                               sync_schema_version, model_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                project.id,
                clips[0].id,
                clips[1].id,
                SYNC_SCHEMA_VERSION + 1,
                json.dumps({"whatever": "the next version looks like"}),
                "2026-09-17T00:00:00+00:00",
            ),
        )
        store._connect().commit()

        read = store.get_project(project.id)
        assert read.syncs == []
        assert any("sync schema version" in note for note in read.warnings)
        assert read.clips != [], "the project itself is still perfectly readable"

    def test_an_unparseable_sync_is_skipped_and_reported(
        self, store: ProjectStore, tmp_path: Path
    ) -> None:
        project = store.create_project("Session")
        for name, digest, role in (
            ("a.mov", "a", CameraRole.FACE_ON),
            ("b.mov", "b", CameraRole.DOWN_THE_LINE),
        ):
            path = tmp_path / name
            path.write_bytes(name.encode())
            store.add_clip(project.id, path=path, content_key=key(digest), role=role)

        clips = store.get_project(project.id).clips
        store._connect().execute(
            """
            INSERT INTO syncs (project_id, reference_clip_id, target_clip_id,
                               sync_schema_version, model_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                project.id,
                clips[0].id,
                clips[1].id,
                SYNC_SCHEMA_VERSION,
                json.dumps({"aligned": "not a boolean"}),
                "2026-09-17T00:00:00+00:00",
            ),
        )
        store._connect().commit()

        read = store.get_project(project.id)
        assert read.syncs == []
        assert any("did not parse" in note for note in read.warnings)


class TestForeignKeys:
    def test_foreign_keys_are_on(self, store: ProjectStore) -> None:
        """Off by default in SQLite, and per connection. Without it the cascades are fiction."""
        enabled = store._connect().execute("PRAGMA foreign_keys").fetchone()[0]
        assert enabled == 1
