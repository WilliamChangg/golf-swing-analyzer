"""Typed contracts for swing phase detection.

The first golf-specific layer. Everything below this in the engine is general
computer vision; from here up, the code knows what a golf swing is.

Four events divide a swing into four phases:

    ADDRESS  |  BACKSWING  |  DOWNSWING  |  FOLLOW_THROUGH
             ^             ^             ^                ^
          takeaway        top         impact           finish

`ADDRESS` has no event of its own because it has no interesting instant: it is
whatever precedes the takeaway, and its last frame *is* the frame before the
takeaway. Inventing an "address event" would produce two events one frame apart
and imply a precision the signal does not carry.

**Impact is a kinematic estimate, not an observation.** Nothing here sees the
ball or the club; the estimate comes from the hand's own motion, and the
methodology is recorded on the event so a later reader knows what produced it.
Phase 10 (club tracking) and Phase 11 (ball detection) provide independent
evidence, at which point these estimates can be checked rather than trusted.

**Confidence is computed and decomposed.** Each event carries three factors
along with their product, because a single number hides which of them was weak —
and they mean different things. A confident event needs a clear signal, a
visible landmark, *and* a frame rate able to resolve it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# Bump on any change that alters what a stored or reported detection means.
PHASES_SCHEMA_VERSION = 1


class SwingEvent(StrEnum):
    """The instants that divide a swing.

    TAKEAWAY - the hands begin moving away from the ball.
    TOP      - the backswing reverses; the hands are momentarily near rest.
    IMPACT   - estimated from hand kinematics, not observed. See module docstring.
    FINISH   - the hands come back to rest after the follow-through.
    """

    TAKEAWAY = "takeaway"
    TOP = "top"
    IMPACT = "impact"
    FINISH = "finish"


class SwingPhase(StrEnum):
    """The intervals between events."""

    ADDRESS = "address"
    BACKSWING = "backswing"
    DOWNSWING = "downswing"
    FOLLOW_THROUGH = "follow_through"


class HandSource(StrEnum):
    """Which landmark(s) the hand trajectory was taken from.

    Chosen once for a clip rather than per frame. Switching between a two-wrist
    midpoint and a single wrist partway through would move the tracked point by
    half the distance between the hands, and the differentiator would read that
    step as a velocity spike -- in the middle of a swing, plausibly at the very
    instant being measured.
    """

    MIDPOINT = "midpoint"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"


class EventConfidence(BaseModel):
    """Why an event is or is not trustworthy, decomposed.

    `overall` is the product of the three factors: an event needs all of them,
    and a product says so where an average would let a strong factor cover for a
    fatal one. The factors are reported individually because "0.3" is not
    actionable while "the landmark was barely visible" is.
    """

    overall: float = Field(ge=0.0, le=1.0, description="Product of the three factors below.")
    margin: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How clearly the signal singles this instant out -- peak prominence for "
            "IMPACT, depth of the speed minimum for TOP, crossing sharpness for "
            "TAKEAWAY and FINISH."
        ),
    )
    visibility: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean reported visibility of the hand landmark around this event.",
    )
    resolution: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Whether the capture and smoothing can resolve an event of this "
            "duration at all. Falls towards zero as the smoothing window "
            "approaches the length of the phase being measured."
        ),
    )


class DetectedEvent(BaseModel):
    """One located instant, with how it was found and what corroborates it."""

    event: SwingEvent
    frame_index: int
    timestamp_s: float
    confidence: EventConfidence
    methodology: str = Field(
        description="The rule that produced this frame, in words. Never omitted."
    )
    corroboration_frame: int | None = Field(
        default=None,
        description=(
            "Frame an independent signal put this event at, where a second signal "
            "exists. None means no corroboration was available -- which is not the "
            "same as the two agreeing."
        ),
    )
    corroboration_delta_s: float | None = Field(
        default=None,
        description="Seconds between this event and its corroborating estimate.",
    )


class DetectedPhase(BaseModel):
    """One interval between events.

    `start_frame` is inclusive and `end_frame` exclusive, so consecutive phases
    tile the clip without overlapping and `end_frame - start_frame` is the frame
    count. `duration_s` is measured from timestamps rather than derived from the
    frame count, because on variable-rate footage those disagree.
    """

    phase: SwingPhase
    start_frame: int
    end_frame: int = Field(description="Exclusive.")
    start_s: float
    end_s: float
    duration_s: float
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The weaker of the confidences of the two events bounding this phase.",
    )


class HandSignalInfo(BaseModel):
    """What the hand trajectory was built from, and how well it stands out.

    `travel_ratio` is the measurement that decides whether there is a swing here
    at all: how far the hands ranged, measured in the subject's own torso
    lengths. Judging against the body rather than against the frame makes it
    independent of where the camera was put — the same swing filmed twice as far
    away halves every distance in frame, torso included, and leaves the ratio
    unchanged.
    """

    source: HandSource
    valid_frames: int
    total_frames: int
    peak_speed: float = Field(description="In the filtered signal's velocity unit.")
    travel: float = Field(
        description="Diagonal of the box the hands stayed inside, in frame widths."
    )
    torso_length: float = Field(
        description="Median shoulder-midpoint to hip-midpoint distance, in frame widths."
    )
    travel_ratio: float = Field(description="travel / torso_length.")


class PhaseConfig(BaseModel, extra="forbid"):
    """Policy for phase detection.

    These are **structural sanity bounds, not golf norms.** Nothing here claims
    that a backswing lasts a particular time; the durations exist to reject
    motion that cannot be a swing at all, and are set loose enough that a slow
    practice swing and a fast one both pass. Numbers tight enough to encode what
    a swing "should" look like would need a labelled set, which is Phase 12.
    """

    moving_fraction: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of peak hand speed above which the hands count as moving. "
            "Expressed as a fraction because absolute speed has no fixed meaning "
            "in normalised frame coordinates."
        ),
    )
    min_travel_ratio: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "The hands must range at least this many torso lengths before a swing "
            "is reported. Half a torso is a long way below a full swing, which "
            "sweeps them through more than one; the bound is there to exclude a "
            "subject standing still, not to describe what a swing looks like."
        ),
    )
    min_phase_travel_ratio: float = Field(
        default=0.15,
        gt=0.0,
        description=(
            "The backswing and the downswing must each move the hands at least "
            "this many torso lengths. Without it a clip that is still except for "
            "one brief twitch yields a top on whichever still frame noise made "
            "highest, and a long backswing containing no movement at all."
        ),
    )
    min_backswing_s: float = Field(
        default=0.20, gt=0.0, description="Below this the motion is not treated as a backswing."
    )
    min_downswing_s: float = Field(
        default=0.06, gt=0.0, description="Below this the motion is not treated as a downswing."
    )
    max_downswing_s: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Above this the motion is not treated as a downswing. A club falling "
            "under gravity alone covers the distance far quicker than a second, so "
            "a slower descent is someone lowering the club rather than swinging it."
        ),
    )
    max_backswing_s: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "How far back from the top to look for the takeaway. A search bound, "
            "not a gate: a backswing longer than this is not refused, it is "
            "simply not searched for past here, because on an untrimmed clip the "
            "last still stretch before the swing can be a minute of someone "
            "standing about. Loose on purpose -- the slowest genuine backswing "
            "in the reference footage is 2.4 s, and a waggle before it is part "
            "of the swing rather than a second event."
        ),
    )
    top_search_s: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "How far back from the fastest frame to look for the top. Deliberately "
            "far looser than `max_downswing_s`, which would seem the natural bound "
            "and is the wrong one: on a conformed slow-motion clip every duration "
            "arrives stretched by the playback factor, and a search cut at one "
            "second would measure a truncated downswing and understate the factor "
            "the clip needs -- breaking the one diagnostic that would have told the "
            "caller what was wrong. Three seconds holds an eight-times slowed "
            "downswing whole, while still excluding the unrelated motion that makes "
            "an untrimmed clip report a descent lasting ten seconds."
        ),
    )
    transition_search_s: float = Field(
        default=0.25,
        gt=0.0,
        description=(
            "How far either side of the highest hand position to look for the "
            "speed minimum that marks the top."
        ),
    )
    confidence_window_s: float = Field(
        default=0.10,
        gt=0.0,
        description="Span around an event over which visibility and margin are measured.",
    )
    min_still_s: float = Field(
        default=0.10,
        gt=0.0,
        description=(
            "How long the hands must stay below the moving threshold before that "
            "counts as being still. Without a minimum, the momentary pause at the "
            "top of the backswing reads as the hands having stopped, and the "
            "takeaway is found there instead of at address."
        ),
    )
    min_tracking_gap_s: float = Field(
        default=0.15,
        gt=0.0,
        description=(
            "A stretch in which the estimator saw no hand is reported once it lasts "
            "this long, because by then it could be concealing the fastest part of a "
            "swing. Shorter stretches are ordinary and reporting them would be noise."
        ),
    )


class SwingPhases(BaseModel):
    """Everything detection concluded about one clip.

    `detected` is false whenever no swing was found, and then `events` and
    `phases` are empty rather than populated with guesses. A caller that checks
    only for events therefore cannot mistake an undetected swing for one at
    frame zero.

    Durations are here; ratios are not. Tempo -- the backswing-to-downswing
    ratio -- is a biomechanics metric and belongs to Phase 5, which is also
    where it acquires a unit, a confidence and a methodology. Computing it here
    would put the same number in two places with two different provenances.
    """

    schema_version: int = PHASES_SCHEMA_VERSION
    detected: bool
    events: list[DetectedEvent] = Field(default_factory=list)
    phases: list[DetectedPhase] = Field(default_factory=list)
    hand: HandSignalInfo
    frames: int
    slow_motion_factor: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "How many times slower than real time the clip plays, as supplied by "
            "the caller. 1.0 is an ordinary recording. Timestamps here are **real "
            "seconds**, already divided by it, so they no longer index into the "
            "video file -- frame numbers do. Nothing in a conformed slow-motion "
            "clip records this, so it cannot be measured and is not guessed."
        ),
    )
    config: PhaseConfig
    warnings: list[str] = Field(default_factory=list)

    def event(self, which: SwingEvent) -> DetectedEvent | None:
        """The named event, or None if it was not located."""
        return next((entry for entry in self.events if entry.event is which), None)

    def phase_at(self, frame_index: int) -> SwingPhase | None:
        """Which phase a frame falls in, or None if it falls outside every phase."""
        for interval in self.phases:
            if interval.start_frame <= frame_index < interval.end_frame:
                return interval.phase
        return None
