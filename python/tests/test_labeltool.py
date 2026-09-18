"""The labelling tool's state machine and its overlay.

The window itself has no test -- it needs a display -- which is exactly why
everything that can corrupt a label lives outside it. These tests are the reason
that split exists.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from analyzer.contracts.phases import SwingEvent
from analyzer.ml.labeltool import (
    DEFAULT_UNCERTAINTY_FRAMES,
    MARK_COLOURS,
    MAX_UNCERTAINTY_FRAMES,
    MIN_WIDTH,
    LabelSession,
    LabelToolError,
    _wrap,
    apply_key,
    key_name,
    render,
)


def session(**overrides: object) -> LabelSession:
    defaults: dict[str, object] = {
        "clip_path": Path("/clips/a.mov"),
        "frames": 120,
        "fps": 60.0,
        "player_id": "p1",
        "session_id": "s1",
        "swing_id": "w1",
        "labeller": "wc",
    }
    defaults.update(overrides)
    return LabelSession(**defaults)  # type: ignore[arg-type]


def mark_a_full_swing(state: LabelSession) -> None:
    for event, frame in (
        (SwingEvent.TAKEAWAY, 10),
        (SwingEvent.TOP, 40),
        (SwingEvent.IMPACT, 55),
        (SwingEvent.FINISH, 80),
    ):
        state.seek(frame)
        state.mark(event)


def test_stepping_is_clamped_rather_than_an_error() -> None:
    """Holding a key down at the end of a clip is not a mistake."""
    state = session()
    state.step(-5)
    assert state.cursor == 0
    state.seek(10_000)
    assert state.cursor == 119


def test_a_mark_starts_with_a_bracket_rather_than_a_certainty() -> None:
    state = session()
    state.seek(42)
    state.mark(SwingEvent.TOP)
    assert state.marks[SwingEvent.TOP].frame_index == 42
    assert state.marks[SwingEvent.TOP].uncertainty_frames == DEFAULT_UNCERTAINTY_FRAMES


def test_re_marking_an_event_keeps_the_bracket_the_labeller_chose() -> None:
    """Nudging a mark by a frame is not a statement about how sure they were."""
    state = session()
    state.seek(42)
    state.mark(SwingEvent.TOP, uncertainty_frames=5)
    state.seek(43)
    state.mark(SwingEvent.TOP)
    assert state.marks[SwingEvent.TOP].frame_index == 43
    assert state.marks[SwingEvent.TOP].uncertainty_frames == 5


def test_widening_adjusts_the_mark_nearest_the_cursor() -> None:
    """A labeller looking again at the top expects to widen the top."""
    state = session()
    mark_a_full_swing(state)
    state.seek(41)
    state.widen(3)
    assert state.marks[SwingEvent.TOP].uncertainty_frames == DEFAULT_UNCERTAINTY_FRAMES + 3
    assert state.marks[SwingEvent.IMPACT].uncertainty_frames == DEFAULT_UNCERTAINTY_FRAMES


def test_a_bracket_cannot_go_negative_or_unbounded() -> None:
    state = session()
    state.seek(10)
    state.mark(SwingEvent.TOP)
    for _ in range(5):
        state.widen(-1)
    assert state.marks[SwingEvent.TOP].uncertainty_frames == 0
    for _ in range(MAX_UNCERTAINTY_FRAMES + 5):
        state.widen(1)
    assert state.marks[SwingEvent.TOP].uncertainty_frames == MAX_UNCERTAINTY_FRAMES


def test_the_bracket_is_shown_in_milliseconds_as_well_as_frames() -> None:
    state = session(fps=30.0)
    state.seek(10)
    state.mark(SwingEvent.TOP, uncertainty_frames=3)
    assert state.uncertainty_ms(state.marks[SwingEvent.TOP]) == pytest.approx(100.0)


def test_declaring_no_swing_discards_the_marks() -> None:
    """A saved label saying both things at once is a file with two answers in it."""
    state = session()
    mark_a_full_swing(state)
    state.set_not_a_swing()
    assert state.marks == {}
    assert state.is_swing is False
    assert state.blocking_reason() is None


def test_marking_after_declaring_no_swing_takes_the_declaration_back() -> None:
    state = session()
    state.set_not_a_swing()
    state.seek(10)
    state.mark(SwingEvent.TAKEAWAY)
    assert state.is_swing is True
    assert SwingEvent.TAKEAWAY in state.marks


def test_undo_restores_the_previous_state_including_the_swing_flag() -> None:
    state = session()
    mark_a_full_swing(state)
    state.set_not_a_swing()
    state.undo()
    assert state.is_swing is True
    assert len(state.marks) == 4


def test_undo_with_nothing_to_undo_says_so_rather_than_failing() -> None:
    state = session()
    state.undo()
    assert "Nothing to undo" in state.status


def test_a_half_marked_swing_cannot_be_saved() -> None:
    """It would read downstream as events that were not visible."""
    state = session()
    state.seek(10)
    state.mark(SwingEvent.TAKEAWAY)
    state.seek(55)
    state.mark(SwingEvent.IMPACT)

    assert "Still to mark" in (state.blocking_reason() or "")
    with pytest.raises(LabelToolError, match="Still to mark"):
        state.to_label()


def test_marks_in_the_wrong_order_block_the_save() -> None:
    state = session()
    mark_a_full_swing(state)
    state.seek(70)
    state.mark(SwingEvent.TOP)
    assert state.out_of_order()
    with pytest.raises(LabelToolError, match="out of order"):
        state.to_label()


def test_a_complete_session_produces_a_valid_human_label() -> None:
    state = session()
    mark_a_full_swing(state)
    label = state.to_label(now=datetime(2026, 9, 17, tzinfo=UTC))

    assert label.is_swing
    assert [entry.frame_index for entry in label.events] == [10, 40, 55, 80]
    assert label.provenance.value == "human"
    assert label.player_id == "p1"


def test_a_clip_with_no_swing_saves_as_a_negative_example() -> None:
    state = session()
    state.set_not_a_swing()
    label = state.to_label()
    assert label.is_swing is False
    assert label.events == []


def test_a_clip_with_no_frames_or_no_rate_is_refused() -> None:
    with pytest.raises(LabelToolError, match="no frames"):
        session(frames=0)
    with pytest.raises(LabelToolError, match="no frame rate"):
        session(fps=0.0)


def test_digits_mark_and_shifted_digits_clear() -> None:
    state = session()
    state.seek(20)
    apply_key(state, "2")
    assert state.marks[SwingEvent.TOP].frame_index == 20
    apply_key(state, "@")
    assert SwingEvent.TOP not in state.marks


def test_unbound_keys_report_rather_than_doing_something() -> None:
    state = session()
    mark_a_full_swing(state)
    before = dict(state.marks)
    apply_key(state, "z")
    assert state.marks == before
    assert "Nothing is bound" in state.status


def test_key_codes_translate_to_names_and_a_timeout_is_not_a_key() -> None:
    assert key_name(ord("1")) == "1"
    assert key_name(63235) == "right"
    assert key_name(-1) is None


def test_the_overlay_adds_a_timeline_and_a_panel_to_the_frame() -> None:
    """`render` returns the image the window shows, so a test sees what a
    labeller sees rather than a description of it."""
    state = session()
    mark_a_full_swing(state)
    frame = np.full((200, 700, 3), 40, dtype=np.uint8)

    canvas = render(state, frame)
    assert canvas.shape[1] == 700
    assert canvas.shape[0] > 200
    # The video region is untouched and the strip below it is not empty.
    assert np.array_equal(canvas[:200], frame)
    assert canvas[200:].any()


def test_a_portrait_clip_gets_a_panel_wide_enough_to_read() -> None:
    """Golf footage is a phone held upright, which is where a frame-width panel
    collides its own text."""
    state = session()
    mark_a_full_swing(state)
    frame = np.full((520, 292, 3), 40, dtype=np.uint8)

    canvas = render(state, frame)
    assert canvas.shape[1] == MIN_WIDTH
    # The clip is centred rather than stretched, so nothing in it is distorted.
    inset = (MIN_WIDTH - 292) // 2
    assert np.array_equal(canvas[:520, inset : inset + 292], frame)
    assert not canvas[:520, :inset].any()


def test_the_panel_grows_to_fit_what_it_has_to_say() -> None:
    """A fixed height drew a two-line refusal over the key help."""
    ready = session()
    mark_a_full_swing(ready)
    blocked = session()
    blocked.seek(10)
    blocked.mark(SwingEvent.TAKEAWAY)

    frame = np.zeros((120, 700, 3), dtype=np.uint8)
    assert render(blocked, frame).shape[0] > render(ready, frame).shape[0]


def test_a_refusal_is_wrapped_on_words_rather_than_truncated() -> None:
    """A message cut mid-word reads as a rendering glitch, not an instruction."""
    text = "Still to mark: finish. A clip is either a swing with all four events."
    lines = _wrap(text, 300, 0.42)
    assert len(lines) > 1
    assert " ".join(lines) == text


def test_the_overlay_refuses_a_greyscale_frame() -> None:
    state = session()
    with pytest.raises(LabelToolError, match="colour frame"):
        render(state, np.zeros((100, 100), dtype=np.uint8))


def test_a_wider_bracket_draws_a_wider_band() -> None:
    """The bracket is the thing being decided, so it has to be visible."""
    narrow = session()
    narrow.seek(60)
    narrow.mark(SwingEvent.TOP, uncertainty_frames=1)
    wide = session()
    wide.seek(60)
    wide.mark(SwingEvent.TOP, uncertainty_frames=20)

    frame = np.zeros((40, 700, 3), dtype=np.uint8)
    colour = np.array(MARK_COLOURS[SwingEvent.TOP], dtype=np.uint8)
    banded = [
        int(np.count_nonzero(np.all(render(state, frame)[40:66] == colour, axis=2)))
        for state in (narrow, wide)
    ]
    assert banded[1] > banded[0]
