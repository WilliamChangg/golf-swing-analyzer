"""Typed contracts for hand-labelled swings.

The first thing in this engine that records a **human judgement** rather than a
measurement. Everything below reads pixels; a label is somebody saying where the
top of the backswing was, and the schema exists to keep that statement honest
about three things it would otherwise lose.

**Who said it, and when.** A label is attributable. Two labellers disagree about
the top by more than the frame rate resolves, and a set built from both without
recording which is which cannot be audited afterwards.

**How sure they were.** `EventLabel.uncertainty_frames` is required, not
defaulted. A takeaway on a sharp 240 fps clip is placeable to a frame; impact on
a 30 fps clip where the club crosses the ball inside one exposure is not, and a
labeller who marks both without saying so has produced two numbers that look
identical and mean different things. This field is what later lets an evaluation
refuse to credit a model with precision its labels never had.

**Whose swing it is.** `player_id` and `session_id` are required and have no
default, because the split that uses them is the only thing standing between
this project and the most common way a golf model reports a number it has not
earned. Swings from one session are near-duplicates -- same player, same club,
same camera, same light, minutes apart -- so a swing in training and its
neighbour in test is the model being asked a question it has already been given
the answer to. The identity cannot be inferred from a file path, so it is not
inferred: it is entered, or there is no label.

**Provenance is part of the schema.** A synthetic swing whose events are known by
construction is a legitimate input to every piece of machinery here and is not a
legitimate input to an accuracy claim, and `LabelProvenance` is what lets the
evaluation layer tell them apart without trusting a caller to remember.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from itertools import pairwise

from pydantic import BaseModel, Field, model_validator

from analyzer.contracts.cache import ContentKey
from analyzer.contracts.phases import SwingEvent

# Bump on any change that alters what a stored label means. A stored set carries
# this, and a reader that does not understand the version refuses the file
# rather than interpreting fields it may be reading differently.
LABEL_SCHEMA_VERSION = 1

# The order the four events must occur in. Not a golf norm -- it is the
# definition of the events themselves, and a label that violates it is a typo
# rather than an unusual swing.
EVENT_ORDER: tuple[SwingEvent, ...] = (
    SwingEvent.TAKEAWAY,
    SwingEvent.TOP,
    SwingEvent.IMPACT,
    SwingEvent.FINISH,
)


class LabelProvenance(StrEnum):
    """Where a label's frame numbers came from.

    HUMAN     - somebody stepped the clip and marked the frames. The only kind
                an accuracy claim may be computed from.
    SYNTHETIC - the events are inputs to a generator that drew the motion, so
                they are exact and describe a fixture rather than a golfer. Every
                report derived from these is marked, and the evaluation layer
                refuses to permit a claim from them.

    There is deliberately no third value for "produced by the rule-based
    detector". Training on a detector's own output and then comparing the model
    against that detector measures imitation, and would answer Phase 12's central
    question with a number that cannot mean anything. See
    `docs/decisions/ADR-0016-labels-groups-and-the-noise-floor.md`.
    """

    HUMAN = "human"
    SYNTHETIC = "synthetic"


class EventLabel(BaseModel):
    """One instant a labeller marked, and how tightly they could place it."""

    event: SwingEvent
    frame_index: int = Field(ge=0, description="Frame in the source clip, not a real-time index.")
    uncertainty_frames: int = Field(
        ge=0,
        description=(
            "Half-width of the bracket the labeller would accept, in frames of "
            "this clip. 0 means they could name the single frame. Required: a "
            "label without it is a number whose precision nobody stated."
        ),
    )
    note: str = Field(
        default="",
        description="Why this was hard, in the labeller's words. Empty when it was not.",
    )


class ClipLabel(BaseModel):
    """Everything one person decided about one clip.

    `is_swing` is false for a clip that contains no swing, and then `events` is
    empty. Those clips are not filler. The rule-based detector's most valuable
    behaviour is refusing a clip that has no swing in it, a learned detector has
    no such behaviour unless it is trained to, and a set made only of swings
    cannot measure either. A labelled set with no negatives can score a detector
    that answers every clip.
    """

    schema_version: int = LABEL_SCHEMA_VERSION
    clip_path: str = Field(
        description="Where the clip was when it was labelled. A hint, not an id."
    )
    content_key: ContentKey | None = Field(
        default=None,
        description=(
            "Identity of the clip's bytes, which is the real id: a renamed clip "
            "keeps its labels and a replaced one loses them. None for a generated "
            "fixture, which has no file."
        ),
    )

    player_id: str = Field(
        min_length=1,
        description=(
            "Who is swinging. The split's grouping key. No default, because a "
            "wrong guess here is invisible and inflates every number downstream."
        ),
    )
    session_id: str = Field(
        min_length=1,
        description=(
            "One capture sitting: same place, same camera, same session of "
            "hitting. Swings inside one are near-duplicates."
        ),
    )
    swing_id: str = Field(min_length=1, description="Distinguishes clips within a session.")

    provenance: LabelProvenance
    labeller: str = Field(min_length=1, description="Who marked it. An id, not a display name.")
    labelled_at: datetime
    tool_version: int = Field(
        default=LABEL_SCHEMA_VERSION,
        description="Which build of the labelling tool produced this.",
    )

    is_swing: bool
    events: list[EventLabel] = Field(default_factory=list)

    frames: int = Field(gt=0, description="Frames in the clip the labeller stepped through.")
    fps: float = Field(gt=0.0, description="Frame rate of the file, before any slow-motion factor.")
    slow_motion_factor: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "How many times slower than real time the clip plays. Carried because "
            "an uncertainty in frames only becomes an uncertainty in real "
            "milliseconds once this is known, and nothing in the file records it."
        ),
    )
    notes: str = Field(default="", description="Anything about the clip as a whole.")

    @model_validator(mode="after")
    def _check(self) -> ClipLabel:
        if not self.is_swing:
            if self.events:
                raise ValueError(
                    "A clip labelled as containing no swing carries no events; this one "
                    f"has {len(self.events)}. Either it is a swing or the marks are stray."
                )
            return self

        seen = [entry.event for entry in self.events]
        if len(set(seen)) != len(seen):
            raise ValueError(f"Each event may be marked once; got {[e.value for e in seen]}.")

        ordered = [entry for entry in self.events if entry.event in EVENT_ORDER]
        ordered.sort(key=lambda entry: EVENT_ORDER.index(entry.event))
        frames = [entry.frame_index for entry in ordered]
        if any(later <= earlier for earlier, later in pairwise(frames)):
            marks = ", ".join(f"{e.event.value}={e.frame_index}" for e in ordered)
            raise ValueError(
                "Swing events must be strictly increasing in frame index -- takeaway, "
                f"top, impact, finish -- and these are not: {marks}. That ordering is "
                "the definition of the events, so this is a mis-keyed mark rather than "
                "an unusual swing."
            )
        for entry in self.events:
            if entry.frame_index >= self.frames:
                raise ValueError(
                    f"{entry.event.value} is marked at frame {entry.frame_index}, past the "
                    f"end of a {self.frames}-frame clip."
                )
        return self

    @property
    def group_key(self) -> str:
        """The value a split must never divide. See `analyzer.ml.splits`."""
        return self.player_id

    @property
    def session_key(self) -> str:
        """Player and session together, which is what a near-duplicate shares."""
        return f"{self.player_id}/{self.session_id}"

    @property
    def clip_key(self) -> str:
        """Identity of this labelled clip within a set."""
        return f"{self.player_id}/{self.session_id}/{self.swing_id}"

    def event(self, which: SwingEvent) -> EventLabel | None:
        """The named mark, or None if this clip has none."""
        return next((entry for entry in self.events if entry.event is which), None)

    @property
    def real_seconds_per_frame(self) -> float:
        """How much real time one frame of this clip covers.

        `1 / (fps * slow_motion_factor)`, matching `pose.series.landmark_series`,
        which divides the clip's own timestamps by the factor. An eight-times
        slow-motion clip at 240 fps advances real time by 1/1920 s per frame,
        which is what makes a one-frame bracket on such a clip worth so much more
        than a one-frame bracket on a phone's 30 fps.
        """
        return 1.0 / (self.fps * self.slow_motion_factor)

    def uncertainty_s(self, which: SwingEvent) -> float | None:
        """A mark's bracket in **real** seconds, or None if it was not marked.

        The slow-motion factor is applied here rather than left to the caller for
        the same reason it is applied below the filter in Phase 3: exactly once,
        in the place that knows about it, so nothing above can forget.
        """
        entry = self.event(which)
        if entry is None:
            return None
        return entry.uncertainty_frames / (self.fps * self.slow_motion_factor)


class LabelSet(BaseModel):
    """A collection of labelled clips, and what it is made of.

    The counts are computed rather than stored so they cannot disagree with the
    clips. They are the numbers that decide whether anything downstream is
    allowed to run: a set with one player has no held-out player, and a set with
    one session per player has no way to tell a model that generalises from one
    that memorised a camera angle.
    """

    schema_version: int = LABEL_SCHEMA_VERSION
    clips: list[ClipLabel] = Field(default_factory=list)

    @property
    def players(self) -> tuple[str, ...]:
        return tuple(sorted({clip.player_id for clip in self.clips}))

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted({clip.session_key for clip in self.clips}))

    @property
    def labellers(self) -> tuple[str, ...]:
        return tuple(sorted({clip.labeller for clip in self.clips}))

    @property
    def provenance(self) -> LabelProvenance:
        """SYNTHETIC if any clip is, because a mixed set can claim no more than
        its weakest member. A set with no clips is synthetic by the same logic:
        nothing in it was seen by anyone."""
        if all(clip.provenance is LabelProvenance.HUMAN for clip in self.clips) and self.clips:
            return LabelProvenance.HUMAN
        return LabelProvenance.SYNTHETIC

    @property
    def swings(self) -> int:
        return sum(1 for clip in self.clips if clip.is_swing)

    @property
    def non_swings(self) -> int:
        return sum(1 for clip in self.clips if not clip.is_swing)

    def digest(self) -> str:
        """A content hash over every label in the set, order-independent.

        Recorded on datasets, splits and trained models, so a model can say which
        labels it was trained against and a reader can tell that two reports
        describing "the labelled set" describe the same one. Order-independent
        because the set is a set: relabelling nothing but the traversal order of
        a directory must not produce a different id.
        """
        parts = sorted(
            json.dumps(clip.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            for clip in self.clips
        )
        digest = hashlib.sha256()
        for part in parts:
            digest.update(part.encode())
            digest.update(b"\n")
        return digest.hexdigest()
