"""The labelling tool: a state machine, a renderer, and a window around them.

Phase 12's first sub-item, and it is first on purpose. Building the model before
the tool is how a project ends up training on whatever labels were convenient to
produce; building the tool first forces the question of what a label *is* to be
answered while it is still cheap to change the answer.

## Why the state machine is separate from the window

Everything a labeller does -- stepping, marking, widening a bracket, undoing,
declaring that a clip has no swing in it -- happens in `LabelSession`, which
knows nothing about OpenCV, keyboards or pixels. `render` turns a session and a
decoded frame into an image. `run_window` is the loop that reads keys and calls
the other two, and it is deliberately the only part that cannot be tested
without a display.

That split is not tidiness. A labelling tool is the one piece of software in this
project whose bugs are unrecoverable: a mis-keyed mark becomes a label, the label
becomes training data, and no later phase can tell that the number was wrong.
The parts that can silently corrupt a label are the parts that are tested.

## What the tool refuses

It will not save a partially-marked swing. A clip is either *not a swing* -- and
then it carries no events and is a genuine negative example, which a labelled
set needs -- or it is a swing with all four events marked. A file with a takeaway
and an impact and nothing in between reads downstream as a clip where the top was
not visible, which is a different statement from a labeller having been
interrupted.

## The bracket

Every mark carries `uncertainty_frames`, and the tool starts one at
`DEFAULT_UNCERTAINTY_FRAMES` and shows it in the overlay next to the mark. The
tool cannot know how sure somebody is; what it can do is refuse to let the number
be invisible. Widen it with `[` and `]`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.cache import ContentKey
from analyzer.contracts.labels import (
    EVENT_ORDER,
    LABEL_SCHEMA_VERSION,
    ClipLabel,
    EventLabel,
    LabelProvenance,
)
from analyzer.contracts.phases import SwingEvent

# Bumped when the tool's behaviour changes in a way that could change what a
# labeller produced. Stored on every label it writes.
TOOL_VERSION = 1

# What a fresh mark starts at. One frame, not zero: a labeller who has just
# stepped to a frame and pressed a key has not established that the event could
# not be the frame either side, and starting at zero would record a certainty
# nobody asserted. It is a starting point to be adjusted, and the overlay shows
# it on every mark so it cannot be left at the default unseen.
DEFAULT_UNCERTAINTY_FRAMES = 1
MAX_UNCERTAINTY_FRAMES = 60

# Keys. Digits mark events in swing order, which is also the order they are
# listed in the overlay, so the mapping needs no memorising.
EVENT_KEYS: dict[str, SwingEvent] = {
    "1": SwingEvent.TAKEAWAY,
    "2": SwingEvent.TOP,
    "3": SwingEvent.IMPACT,
    "4": SwingEvent.FINISH,
}

_JUMP_FRAMES = 10


class LabelToolError(RuntimeError):
    """The tool cannot do what was asked, and saying so is the right outcome."""


@dataclass(frozen=True)
class Mark:
    """One event placed at one frame, with the labeller's own bracket."""

    event: SwingEvent
    frame_index: int
    uncertainty_frames: int = DEFAULT_UNCERTAINTY_FRAMES
    note: str = ""


@dataclass
class LabelSession:
    """The whole of the labeller's state for one clip.

    `history` holds previous states of `marks` and `is_swing` so an accidental
    keypress can be taken back. It holds states rather than inverse operations:
    an undo stack of operations has to get every inverse right, and the one that
    is got wrong is the one that silently leaves a mark behind.
    """

    clip_path: Path
    frames: int
    fps: float
    player_id: str
    session_id: str
    swing_id: str
    labeller: str
    slow_motion_factor: float = 1.0
    content_key: ContentKey | None = None
    notes: str = ""

    cursor: int = 0
    is_swing: bool = True
    marks: dict[SwingEvent, Mark] = field(default_factory=dict)
    history: list[tuple[dict[SwingEvent, Mark], bool]] = field(default_factory=list)
    status: str = "Step with the arrow keys; 1-4 mark the events; s saves."
    dirty: bool = False
    finished: bool = False

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise LabelToolError("A clip with no frames cannot be labelled.")
        if self.fps <= 0:
            raise LabelToolError("A clip with no frame rate cannot be labelled.")
        self.cursor = max(0, min(self.cursor, self.frames - 1))

    # -- time ---------------------------------------------------------------

    @property
    def timestamp_s(self) -> float:
        """Where the cursor is, on the clip's own clock."""
        return self.cursor / self.fps

    @property
    def real_timestamp_s(self) -> float:
        """Where the cursor is in **real** time, slow motion divided out."""
        return self.timestamp_s / self.slow_motion_factor

    def uncertainty_ms(self, mark: Mark) -> float:
        """A bracket in real milliseconds. What the frame count actually means.

        Shown next to every mark because the two units diverge sharply on exactly
        the footage worth capturing: three frames is 100 ms on a phone at 30 fps
        and 1.6 ms on an eight-times slow-motion clip at 240.
        """
        return 1000.0 * mark.uncertainty_frames / (self.fps * self.slow_motion_factor)

    # -- navigation ---------------------------------------------------------

    def seek(self, frame_index: int) -> None:
        """Move the cursor, clamped to the clip. Never an error: a labeller
        holding a key down at the end of a clip has not done anything wrong."""
        self.cursor = max(0, min(int(frame_index), self.frames - 1))

    def step(self, delta: int) -> None:
        self.seek(self.cursor + delta)

    def seek_mark(self, event: SwingEvent) -> None:
        mark = self.marks.get(event)
        if mark is None:
            self.status = f"{event.value} is not marked yet."
            return
        self.seek(mark.frame_index)
        self.status = f"At the {event.value} mark."

    # -- marking ------------------------------------------------------------

    def _checkpoint(self) -> None:
        self.history.append((dict(self.marks), self.is_swing))
        self.dirty = True

    def mark(self, event: SwingEvent, uncertainty_frames: int | None = None) -> None:
        """Place an event at the cursor, replacing any previous mark for it.

        Marking on a clip declared not-a-swing turns that declaration off rather
        than refusing: pressing 1 while looking at a takeaway is an unambiguous
        statement that there is a swing here, and making the labeller undo a
        keypress first would be pedantry.
        """
        self._checkpoint()
        previous = self.marks.get(event)
        resolved = (
            uncertainty_frames
            if uncertainty_frames is not None
            else previous.uncertainty_frames
            if previous is not None
            else DEFAULT_UNCERTAINTY_FRAMES
        )
        self.marks[event] = Mark(
            event=event,
            frame_index=self.cursor,
            uncertainty_frames=max(0, min(int(resolved), MAX_UNCERTAINTY_FRAMES)),
        )
        if not self.is_swing:
            self.is_swing = True
            self.status = f"{event.value} at frame {self.cursor}; this is a swing again."
            return
        self.status = f"{event.value} at frame {self.cursor} (+-{resolved} frames)."

    def clear(self, event: SwingEvent) -> None:
        if event not in self.marks:
            self.status = f"{event.value} was not marked."
            return
        self._checkpoint()
        del self.marks[event]
        self.status = f"{event.value} cleared."

    def widen(self, delta: int) -> None:
        """Adjust the bracket on the mark nearest the cursor.

        Nearest rather than last-touched: a labeller stepping back to look again
        at the top expects `]` to widen the top, not whichever mark they happened
        to place most recently.
        """
        nearest = self.nearest_mark()
        if nearest is None:
            self.status = "Nothing is marked, so there is no bracket to widen."
            return
        self._checkpoint()
        widened = max(0, min(nearest.uncertainty_frames + int(delta), MAX_UNCERTAINTY_FRAMES))
        self.marks[nearest.event] = replace(nearest, uncertainty_frames=widened)
        self.status = (
            f"{nearest.event.value} +-{widened} frames "
            f"({self.uncertainty_ms(self.marks[nearest.event]):.0f} ms)."
        )

    def nearest_mark(self) -> Mark | None:
        if not self.marks:
            return None
        return min(self.marks.values(), key=lambda mark: abs(mark.frame_index - self.cursor))

    def set_not_a_swing(self) -> None:
        """Declare the clip a negative example, discarding any marks.

        Discarding rather than keeping them hidden: a saved label carrying marks
        it says are not there is a file with two answers in it, and the schema
        rejects it anyway.
        """
        self._checkpoint()
        self.marks.clear()
        self.is_swing = False
        self.status = "Marked as containing no swing. Any marks were discarded."

    def undo(self) -> None:
        if not self.history:
            self.status = "Nothing to undo."
            return
        self.marks, self.is_swing = self.history.pop()
        self.status = "Undone."

    # -- validity -----------------------------------------------------------

    def missing(self) -> tuple[SwingEvent, ...]:
        """Events a swing still needs. Empty for a clip declared not-a-swing."""
        if not self.is_swing:
            return ()
        return tuple(event for event in EVENT_ORDER if event not in self.marks)

    def out_of_order(self) -> bool:
        """Whether the marks present violate the definitional ordering."""
        placed = [self.marks[event].frame_index for event in EVENT_ORDER if event in self.marks]
        return any(later <= earlier for earlier, later in pairwise(placed))

    def blocking_reason(self) -> str | None:
        """Why this session cannot be saved yet, or None when it can."""
        if self.out_of_order():
            return (
                "The marks are out of order: takeaway, top, impact and finish must "
                "occur in that order. Clear the one that is wrong with shift+its digit."
            )
        missing = self.missing()
        if missing:
            names = ", ".join(event.value for event in missing)
            return (
                f"Still to mark: {names}. A clip is either a swing with all four "
                "events or, with n, a clip containing no swing -- a half-marked "
                "swing would read downstream as events that were not visible."
            )
        return None

    def to_label(self, *, now: datetime | None = None) -> ClipLabel:
        """The label this session represents. Raises rather than saving a partial one."""
        reason = self.blocking_reason()
        if reason is not None:
            raise LabelToolError(reason)
        return ClipLabel(
            schema_version=LABEL_SCHEMA_VERSION,
            clip_path=str(self.clip_path),
            content_key=self.content_key,
            player_id=self.player_id,
            session_id=self.session_id,
            swing_id=self.swing_id,
            provenance=LabelProvenance.HUMAN,
            labeller=self.labeller,
            labelled_at=now or datetime.now(UTC),
            tool_version=TOOL_VERSION,
            is_swing=self.is_swing,
            events=[
                EventLabel(
                    event=mark.event,
                    frame_index=mark.frame_index,
                    uncertainty_frames=mark.uncertainty_frames,
                    note=mark.note,
                )
                for mark in sorted(
                    self.marks.values(), key=lambda mark: EVENT_ORDER.index(mark.event)
                )
            ],
            frames=self.frames,
            fps=self.fps,
            slow_motion_factor=self.slow_motion_factor,
            notes=self.notes,
        )


def apply_key(session: LabelSession, key: str) -> None:
    """Drive a session from one keypress.

    A pure function of a printable key name, which is what makes the whole
    interaction testable: `run_window` translates OpenCV's integer key codes into
    these names and does nothing else with them.
    """
    if key in EVENT_KEYS:
        session.mark(EVENT_KEYS[key])
        return
    if key in {"!", "@", "#", "$"}:
        # shift+digit on a US layout. Clearing is deliberately awkward to reach.
        session.clear(EVENT_KEYS["1234"["!@#$".index(key)]])
        return

    match key:
        case "left":
            session.step(-1)
        case "right":
            session.step(1)
        case "down":
            session.step(-_JUMP_FRAMES)
        case "up":
            session.step(_JUMP_FRAMES)
        case "home":
            session.seek(0)
        case "end":
            session.seek(session.frames - 1)
        case "[":
            session.widen(-1)
        case "]":
            session.widen(1)
        case "n":
            session.set_not_a_swing()
        case "u":
            session.undo()
        case "g":
            nearest = session.nearest_mark()
            if nearest is not None:
                session.seek_mark(nearest.event)
        case "q":
            session.finished = True
        case _:
            session.status = f"Nothing is bound to {key!r}."


# -- rendering --------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TIMELINE_H = 26
_LINE_H = 21
_MARGIN = 12

# The panel gets its own minimum width, and the video is centred inside it when
# the clip is narrower. Golf footage is overwhelmingly vertical -- a phone held
# upright is the whole reference set here -- so the naive "panel as wide as the
# frame" layout collides its own text on exactly the common case. Letterboxing a
# portrait clip costs nothing; a bracket the labeller cannot read costs a label.
MIN_WIDTH = 640
_WHITE = (255, 255, 255)
_DIM = (170, 170, 170)
_WARN = (80, 190, 250)
_OK = (120, 230, 140)
MARK_COLOURS: dict[SwingEvent, tuple[int, int, int]] = {
    SwingEvent.TAKEAWAY: (240, 190, 90),
    SwingEvent.TOP: (140, 220, 250),
    SwingEvent.IMPACT: (110, 120, 250),
    SwingEvent.FINISH: (180, 160, 240),
}


def _timeline(session: LabelSession, width: int) -> NDArray[np.uint8]:
    """The strip showing where the marks are, and where the cursor is.

    The bracket is drawn as a band rather than the mark as a line, because the
    bracket is the thing being decided. A labeller looking at a three-frame band
    and a one-frame band can see at a glance which of their own marks they were
    unsure about; four identical ticks would hide exactly that.
    """
    strip = np.zeros((_TIMELINE_H, width, 3), dtype=np.uint8)
    strip[:] = (28, 28, 30)
    span = max(session.frames - 1, 1)

    def to_x(frame: int) -> int:
        return round(frame / span * (width - 1))

    for mark in session.marks.values():
        colour = MARK_COLOURS[mark.event]
        left = to_x(max(0, mark.frame_index - mark.uncertainty_frames))
        right = to_x(min(session.frames - 1, mark.frame_index + mark.uncertainty_frames))
        cv2.rectangle(strip, (left, 6), (max(right, left + 1), _TIMELINE_H - 7), colour, -1)
        centre = to_x(mark.frame_index)
        cv2.line(strip, (centre, 2), (centre, _TIMELINE_H - 3), _WHITE, 1)

    cursor = to_x(session.cursor)
    cv2.line(strip, (cursor, 0), (cursor, _TIMELINE_H - 1), (60, 255, 255), 2)
    return strip


def _wrap(text: str, width: int, scale: float) -> list[str]:
    """Break text into lines that fit, on word boundaries.

    Truncating instead was the first version, and it produced a panel that said
    "Still to mark: finish. A clip is eit" -- which reads as a rendering glitch
    rather than as the instruction it is. The one thing the panel exists to do is
    tell a labeller why the clip will not save.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and cv2.getTextSize(candidate, _FONT, scale, 1)[0][0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _panel_lines(
    session: LabelSession, width: int
) -> list[tuple[str, tuple[int, int, int], float]]:
    """Every line of the panel, in order, with its colour and scale.

    Built as data before anything is drawn, so the panel can be sized to its
    contents. The earlier version guessed a fixed height and then drew a variable
    number of lines into it, which overlapped the key help on any clip whose
    refusal ran to two lines.
    """
    body = width - 2 * _MARGIN
    lines: list[tuple[str, tuple[int, int, int], float]] = [
        (
            f"{session.player_id}/{session.session_id}/{session.swing_id}"
            f"   frame {session.cursor + 1}/{session.frames}"
            f"   {session.real_timestamp_s:.3f}s",
            _WHITE,
            0.52,
        )
    ]

    if session.is_swing:
        for event in EVENT_ORDER:
            mark = session.marks.get(event)
            text = (
                f"{event.value:<9} frame {mark.frame_index:<6} "
                f"+-{mark.uncertainty_frames} frames / {session.uncertainty_ms(mark):.0f} ms"
                if mark
                else f"{event.value:<9} -"
            )
            lines.append((text, MARK_COLOURS[event] if mark else _DIM, 0.45))
    else:
        lines.append(("NO SWING IN THIS CLIP -- saved as a negative example", _WARN, 0.45))

    blocking = session.blocking_reason()
    colour = _WARN if blocking else _OK
    for line in _wrap(blocking or "Ready to save.", body, 0.42):
        lines.append((line, colour, 0.42))
    for line in _wrap(session.status, body, 0.42):
        lines.append((line, _DIM, 0.42))
    lines.append(("", _DIM, 0.42))
    for line in _wrap(
        "<- -> step   up/down jump 10   1-4 mark   shift+1-4 clear   [ ] bracket   "
        "g go to mark   n no swing   u undo   s save   q quit",
        body,
        0.40,
    ):
        lines.append((line, _DIM, 0.40))
    return lines


def render(session: LabelSession, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Compose the labelling view: the frame, the timeline, and the state.

    Returns a new image and mutates nothing, so it can be called in a test and
    asserted on. `run_window` shows what this returns and adds nothing of its
    own -- there is no state visible on screen that a test cannot also see.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise LabelToolError(f"Expected a colour frame, got an array of shape {frame.shape}.")

    height, frame_width = frame.shape[:2]
    width = max(frame_width, MIN_WIDTH)
    lines = _panel_lines(session, width)
    panel_height = _LINE_H * len(lines) + 2 * _MARGIN

    canvas = np.zeros((height + _TIMELINE_H + panel_height, width, 3), dtype=np.uint8)
    inset = (width - frame_width) // 2
    canvas[:height, inset : inset + frame_width] = frame
    canvas[height : height + _TIMELINE_H] = _timeline(session, width)
    canvas[height + _TIMELINE_H :] = (18, 18, 20)

    baseline = height + _TIMELINE_H + _MARGIN + 4
    for index, (text, colour, scale) in enumerate(lines):
        if not text:
            continue
        cv2.putText(
            canvas,
            text,
            (_MARGIN, baseline + index * _LINE_H),
            _FONT,
            scale,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


# -- the window -------------------------------------------------------------

# OpenCV returns platform-specific codes for the non-printable keys. These are
# the macOS/Qt values; the arrow keys also arrive as the second set on some
# builds, so both are mapped rather than one being assumed.
_KEY_CODES: dict[int, str] = {
    81: "left",
    82: "up",
    83: "right",
    84: "down",
    2: "home",
    3: "end",
    63234: "left",
    63232: "up",
    63235: "right",
    63233: "down",
    13: "save",
    27: "q",
}


def key_name(code: int) -> str | None:
    """Translate an OpenCV key code into the names `apply_key` understands.

    None for "nothing was pressed", which is what `waitKey` returns on a timeout
    and is not an error.
    """
    if code in (-1, 255):
        return None
    if code in _KEY_CODES:
        return _KEY_CODES[code]
    if 32 <= code < 127:
        return chr(code)
    return None


def run_window(
    session: LabelSession,
    frames: Sequence[NDArray[np.uint8]] | Any,
    *,
    window: str = "label swing",
    on_save: Any = None,
) -> LabelSession:  # pragma: no cover - requires a display
    """Show the clip and drive the session from the keyboard.

    The one part of this module that cannot run without a display, and therefore
    the one part with no test. It is kept to a loop with no decisions in it: read
    a frame, render, wait for a key, hand the key to `apply_key`. Everything a
    key does lives in the state machine above, where it is tested.

    `frames` is anything indexable by frame number -- a `FrameSource` is wrapped
    by the CLI so this module does not depend on the ingestion layer.
    """
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while not session.finished:
            image = frames[session.cursor]
            cv2.imshow(window, render(session, image))
            name = key_name(cv2.waitKey(20))
            if name is None:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            if name in {"save", "s"}:
                reason = session.blocking_reason()
                if reason is not None:
                    session.status = reason
                    continue
                if on_save is not None:
                    on_save(session.to_label())
                session.dirty = False
                session.status = "Saved."
                continue
            apply_key(session, name)
    finally:
        cv2.destroyWindow(window)
    return session
