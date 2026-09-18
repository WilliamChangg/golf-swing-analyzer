"""The label schema and the store, which are the only record of human judgement.

Everything a model later claims rests on these files, and nothing downstream can
tell that a label was wrong. So the tests here are about what the schema
*refuses* far more than about what it accepts.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.labels import (
    LABEL_SCHEMA_VERSION,
    ClipLabel,
    EventLabel,
    LabelProvenance,
    LabelSet,
)
from analyzer.contracts.phases import SwingEvent
from analyzer.ml.labels import (
    LabelStoreError,
    label_filename,
    load_label_set,
    read_label,
    write_label,
)


def make_label(
    *,
    player: str = "p1",
    session: str = "s1",
    swing: str = "w1",
    frames: tuple[int, int, int, int] = (10, 40, 55, 80),
    fps: float = 60.0,
    is_swing: bool = True,
    uncertainty: int = 1,
    digest: str | None = None,
) -> ClipLabel:
    events = (
        [
            EventLabel(event=event, frame_index=frame, uncertainty_frames=uncertainty)
            for event, frame in zip(
                (SwingEvent.TAKEAWAY, SwingEvent.TOP, SwingEvent.IMPACT, SwingEvent.FINISH),
                frames,
                strict=True,
            )
        ]
        if is_swing
        else []
    )
    return ClipLabel(
        clip_path=f"/clips/{player}-{swing}.mov",
        content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED,
            digest=digest or f"{player}{session}{swing}".ljust(64, "0"),
            size_bytes=1024,
        ),
        player_id=player,
        session_id=session,
        swing_id=swing,
        provenance=LabelProvenance.HUMAN,
        labeller="wc",
        labelled_at=datetime(2026, 9, 17, tzinfo=UTC),
        is_swing=is_swing,
        events=events,
        frames=120,
        fps=fps,
    )


def test_events_must_occur_in_their_definitional_order() -> None:
    """Takeaway, top, impact, finish is what the events *are*, not a golf norm."""
    with pytest.raises(ValidationError, match="strictly increasing"):
        make_label(frames=(10, 55, 40, 80))


def test_an_event_cannot_be_marked_twice() -> None:
    with pytest.raises(ValidationError, match="may be marked once"):
        ClipLabel(
            clip_path="/clips/a.mov",
            player_id="p",
            session_id="s",
            swing_id="w",
            provenance=LabelProvenance.HUMAN,
            labeller="wc",
            labelled_at=datetime(2026, 9, 17, tzinfo=UTC),
            is_swing=True,
            events=[
                EventLabel(event=SwingEvent.TOP, frame_index=10, uncertainty_frames=0),
                EventLabel(event=SwingEvent.TOP, frame_index=20, uncertainty_frames=0),
            ],
            frames=120,
            fps=60.0,
        )


def test_a_mark_past_the_end_of_the_clip_is_refused() -> None:
    with pytest.raises(ValidationError, match="past the end"):
        make_label(frames=(10, 40, 55, 500))


def test_a_clip_with_no_swing_carries_no_events() -> None:
    """The two statements cannot both be made; one of them is a mis-key."""
    with pytest.raises(ValidationError, match="no swing carries no events"):
        ClipLabel(
            clip_path="/clips/a.mov",
            player_id="p",
            session_id="s",
            swing_id="w",
            provenance=LabelProvenance.HUMAN,
            labeller="wc",
            labelled_at=datetime(2026, 9, 17, tzinfo=UTC),
            is_swing=False,
            events=[EventLabel(event=SwingEvent.TOP, frame_index=10, uncertainty_frames=0)],
            frames=120,
            fps=60.0,
        )


def test_group_keys_have_no_defaults() -> None:
    """A split cannot invent who was swinging, so the schema will not let it try."""
    with pytest.raises(ValidationError):
        ClipLabel(
            clip_path="/clips/a.mov",
            session_id="s",
            swing_id="w",
            provenance=LabelProvenance.HUMAN,
            labeller="wc",
            labelled_at=datetime(2026, 9, 17, tzinfo=UTC),
            is_swing=False,
            frames=10,
            fps=30.0,
        )  # type: ignore[call-arg]


def test_uncertainty_converts_to_real_seconds_through_the_slow_motion_factor() -> None:
    """Three frames means 100 ms on a phone and 1.6 ms on slow-motion footage."""
    phone = make_label(fps=30.0, uncertainty=3)
    assert phone.uncertainty_s(SwingEvent.TOP) == pytest.approx(0.1)

    slow = phone.model_copy(update={"fps": 240.0, "slow_motion_factor": 8.0})
    assert slow.uncertainty_s(SwingEvent.TOP) == pytest.approx(3 / 1920)
    assert slow.real_seconds_per_frame == pytest.approx(1 / 1920)


def test_an_unmarked_event_has_no_uncertainty_rather_than_zero() -> None:
    label = make_label(is_swing=False)
    assert label.uncertainty_s(SwingEvent.TOP) is None


def test_a_set_is_synthetic_if_any_clip_is() -> None:
    """A mixed set can claim no more than its weakest member."""
    human = make_label()
    generated = make_label(player="p2", digest="b" * 64).model_copy(
        update={"provenance": LabelProvenance.SYNTHETIC}
    )
    assert LabelSet(clips=[human]).provenance is LabelProvenance.HUMAN
    assert LabelSet(clips=[human, generated]).provenance is LabelProvenance.SYNTHETIC
    assert LabelSet(clips=[]).provenance is LabelProvenance.SYNTHETIC


def test_the_digest_does_not_depend_on_traversal_order() -> None:
    """Two machines walking a directory differently describe the same set."""
    first = make_label(player="a", digest="a" * 64)
    second = make_label(player="b", digest="b" * 64)
    assert LabelSet(clips=[first, second]).digest() == LabelSet(clips=[second, first]).digest()


def test_the_digest_changes_when_a_mark_moves_by_one_frame() -> None:
    before = LabelSet(clips=[make_label()]).digest()
    after = LabelSet(clips=[make_label(frames=(10, 41, 55, 80))]).digest()
    assert before != after


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    label = make_label()
    path = write_label(label, tmp_path)
    assert path.name == label_filename(label)
    assert read_label(path) == label


def test_ids_that_would_escape_the_directory_are_made_safe(tmp_path: Path) -> None:
    """Ids are typed by a person, so they can contain anything at all."""
    label = make_label(player="../../etc", session="a b", swing="c/d")
    path = write_label(label, tmp_path)
    assert path.parent == tmp_path
    assert read_label(path).player_id == "../../etc"


def test_a_file_from_a_future_schema_is_refused_rather_than_interpreted(tmp_path: Path) -> None:
    path = write_label(make_label(), tmp_path)
    raw = json.loads(path.read_text())
    raw["schema_version"] = LABEL_SCHEMA_VERSION + 1
    path.write_text(json.dumps(raw))

    with pytest.raises(LabelStoreError, match="schema version"):
        read_label(path)


def test_unparseable_json_names_the_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(LabelStoreError, match=re.escape("broken.json")):
        read_label(broken)


def test_loading_refuses_a_set_containing_an_unreadable_file(tmp_path: Path) -> None:
    """Silently dropping it would shrink the population a split is made from."""
    write_label(make_label(), tmp_path)
    (tmp_path / "broken.json").write_text("{")

    with pytest.raises(LabelStoreError):
        load_label_set(tmp_path)

    assert len(load_label_set(tmp_path, strict=False).clips) == 1


def test_the_same_footage_labelled_twice_under_two_ids_is_caught(tmp_path: Path) -> None:
    """A file copied into two session folders is the leak no id comparison sees."""
    shared = "c" * 64
    write_label(make_label(player="p1", digest=shared), tmp_path)
    write_label(make_label(player="p2", digest=shared), tmp_path)

    with pytest.raises(LabelStoreError, match=re.escape("same file contents")):
        load_label_set(tmp_path)


def test_an_absent_directory_is_an_empty_set_rather_than_an_error(tmp_path: Path) -> None:
    assert load_label_set(tmp_path / "nothing").clips == []
