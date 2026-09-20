"""Typed contracts for comparing two swings.

Everything below this layer measures **one** recording. This is the first layer
whose output is a statement about a *pair*, and almost everything difficult about
it comes from that: two recordings differ for reasons that have nothing to do
with the two swings, and a difference is only worth showing once those have been
excluded or bounded.

## There is no score, and there is no "better"

The same absence Phase 13 built into `Finding`, for the same reason and one
more. A number ranking two swings against each other would need a scale relating
degrees of shoulder turn to seconds of tempo, and nobody has measured one; worse,
it would have to know which direction of each quantity is desirable, and this
system has no such knowledge and no source that supplies one. Every rotation
threshold Phase 13 catalogued turned out to be a convention, a population
summary, or a number with no published protocol. None of them is a target.

So a difference here carries a **direction** -- higher or lower -- and nothing
else. `Direction` has two members rather than three: there is deliberately no
"the same". Two values closer together than the pair can resolve is not evidence
that the swings agree, it is the absence of evidence that they differ, and those
are reported differently. The first would be a measurement; the second is
`DifferenceRefusal.UNRESOLVED`, and it sits in the refusal list where the reader
can see what would have to change to resolve it.

## Why a normalised clock is needed, and what it destroys

Two swings take different amounts of time, so a trajectory plotted against
seconds cannot be laid over another one. `PhaseClock` maps a clip's own clock
onto **swing position**: 0 at the takeaway, 1 at the top, 2 at impact, 3 at the
finish, linear in time within each phase.

The map is exact at the four knots by construction, which is the point and also
the cost. **Every timing difference between the two swings is destroyed by the
normalisation**, because the normalisation is what forces the four instants to
coincide. Nothing in an overlaid trajectory can say that one player reached the
top later; that comparison lives in the metric differences, where a duration is
compared as a duration with a bracket of its own. The two halves of this report
answer different questions on purpose, and the clock's own knots are carried so
that what was divided out is still visible.

## Why the horizontal axis carries an uncertainty

The four knots are located to a frame, so the position of any *real* instant on
the normalised axis is known only as well as the events bounding it. A
difference between two curves at a given position can therefore be entirely
temporal: the same curve, sampled at two slightly different places. Each sample
carries a `bracket` which is that ambiguity propagated through the local slope,
and a difference inside it is not reported as a difference.

**That bracket is a floor, and a weak one near the ends.** It is built from one
frame interval per event, which is the same convention `coaching.bracket` uses
for a duration. Phase 7 measured what landmark noise actually does to these
instants: at 120 fps the top moved 8 ms and the takeaway moved 542 ms, which is
sixty-five frames. Nothing in a single clip measures that, so nothing here
claims it -- but a reader should know that the takeaway end of every plot in this
report is the least trustworthy part of it, and that the refusals below are a
lower bound on the refusals warranted.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.cache import ContentKey
from analyzer.contracts.calibration import CalibrationStatus
from analyzer.contracts.metrics import (
    BodySide,
    CameraView,
    MetricBasis,
    MetricGroup,
    MetricName,
    MetricUnit,
)
from analyzer.contracts.phases import SwingEvent

# Bump on any change that alters what a reported difference means.
COMPARISON_SCHEMA_VERSION = 1

SWING_POSITION: dict[SwingEvent, float] = {
    SwingEvent.TAKEAWAY: 0.0,
    SwingEvent.TOP: 1.0,
    SwingEvent.IMPACT: 2.0,
    SwingEvent.FINISH: 3.0,
}
"""Where each event sits on the normalised axis.

Integers, one per phase boundary, so a position reads as "one and a half phases
in" rather than as a percentage whose landmarks a reader has to look up. The
address phase has no position: it is bounded by an event on one side only, and a
normalisation needs two knots to interpolate between.
"""


class Direction(StrEnum):
    """Which way a difference went. Two members, and both are deliberate.

    HIGHER  - the target's value is larger than the reference's.
    LOWER   - the target's value is smaller.

    There is no third member. "The same" is not a verdict this system can reach:
    two values closer together than the pair can resolve produce
    `DifferenceRefusal.UNRESOLVED`, which says the recordings could not tell them
    apart rather than that the swings agreed. And there is no "better": see the
    module docstring.
    """

    HIGHER = "higher"
    LOWER = "lower"


class DifferenceRefusal(StrEnum):
    """Why a quantity present in both clips was not compared.

    Each is a different thing to do about it, which is why they are a typed set
    rather than a sentence -- the same argument `FindingRefusal` makes one layer
    across.

    NO_SWING
        One of the clips contains no detected swing, so it has no metrics and no
        clock to normalise onto.
    MISSING
        The quantity was produced for one clip and not the other. Usually a fact
        about the recording that did not produce it, and its own `MetricSet`
        says which.
    VIEW_MISMATCH
        The two clips were filmed from different camera positions. A projected
        angle is a fact about a camera position as much as about a body, so the
        two numbers are not measurements of the same quantity. Durations survive
        this; a stopwatch does not care where the camera stood.
    CAMERA_MOVED
        Both clips were filmed from the same *class* of position and not from the
        same position. Measured from the address shoulder span, which is what the
        view detector reads and what every foreshortening rotation is referred
        to. See `CameraAgreement`.
    CALIBRATION_MISMATCH
        One clip's landmarks had a lens correction applied and the other's did
        not. Phase 8 measured that correction moving peak hand speed by 3.0% and
        a rotation by up to 1.2 degrees on real footage, which is a systematic
        difference between the two recordings and not between the two swings.
    SUPPLIED_TIMEBASE
        At least one clip carries a slow-motion factor, which is supplied rather
        than measured. A duration from such a clip is a measured number
        multiplied by a guess, and a difference of two of them is a difference of
        two guesses. A **ratio** of two durations from one clip divides it out
        and survives, which is the same gate Phase 13 applies.
    LOW_CONFIDENCE
        One of the two measurements does not support a conclusion on its own, so
        a difference of the two supports less.
    NO_BRACKET
        Neither clip's measurement carries an uncertainty, and the quantity is
        not one whose bracket can be derived from the clock. There is then no
        distance a difference could be required to clear, and one invented here
        would decide the answer.
    UNRESOLVED
        The difference is real arithmetic and smaller than what the two
        recordings can resolve. The most common outcome, and the one worth
        reading: it names a difference that a faster camera or a steadier
        landmark would settle.
    """

    NO_SWING = "no_swing"
    MISSING = "missing"
    VIEW_MISMATCH = "view_mismatch"
    CAMERA_MOVED = "camera_moved"
    CALIBRATION_MISMATCH = "calibration_mismatch"
    SUPPLIED_TIMEBASE = "supplied_timebase"
    LOW_CONFIDENCE = "low_confidence"
    NO_BRACKET = "no_bracket"
    UNRESOLVED = "unresolved"


class TrajectoryChannel(StrEnum):
    """A scalar signal through the swing, comparable once both clips are normalised.

    These are the signals Phase 4 reads the swing's events off, carried up rather
    than recomputed, so that a curve in this report and the timeline beside it
    cannot come from two different readings of the same clip.

    HAND_SPEED and HAND_HEIGHT are divided by each subject's own torso length, so
    they survive a change of framing and are dimensionless ratios rather than
    speeds and distances. SHOULDER_ANGLE and HIP_ANGLE are projected into the
    image plane and are comparable only between clips filmed from the same place.
    """

    HAND_SPEED = "hand_speed"
    HAND_HEIGHT = "hand_height"
    SHOULDER_ANGLE = "shoulder_angle"
    HIP_ANGLE = "hip_angle"


class ClockKnot(BaseModel):
    """One event, and how well the clip places it.

    `ambiguity_s` is the half-width of where the true instant could be, in real
    seconds. It starts at one frame interval -- the clip's own clock resolution,
    taken from the events rather than from a declared frame rate, for the reasons
    `coaching.bracket` gives -- and is widened where an independent estimate of
    the same instant disagrees by more. Phase 4 corroborates impact with the low
    point of the hand arc; where two estimates of one instant sit 40 ms apart,
    40 ms is the honest ambiguity and one frame is not.
    """

    event: SwingEvent
    position: float = Field(description="Where this event sits on the normalised axis.")
    frame_index: int
    timestamp_s: float = Field(
        description=(
            "Real-clock instant, after any slow-motion factor has been divided "
            "out. On a slowed clip this does not index the video file."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Phase 4's own, carried through.")
    ambiguity_s: float = Field(
        ge=0.0, description="Half-width of where this instant could be, in real seconds."
    )
    ambiguity_source: str = Field(description="What set the ambiguity, in words.")


class PhaseClock(BaseModel):
    """The map from one clip's clock onto swing position.

    Piecewise linear, with a knot at each of the four events. Linear *in time*
    within each phase rather than in frame number, because the footage may be
    variable-rate and because a slow-motion factor has already been divided out
    of the timestamps.

    A clip missing any of the four events has no clock: three knots cannot
    normalise the phase they do not bound, and extrapolating one would invent the
    instant that decides where every sample lands.
    """

    knots: list[ClockKnot] = Field(
        description="Four, in swing order, or empty when the clip could not be normalised."
    )
    usable: bool = Field(description="Whether all four events were located.")
    frame_interval_s: float | None = Field(
        default=None,
        description=(
            "The clip's clock resolution in real seconds, as its own events imply "
            "it. The floor under every ambiguity above."
        ),
    )
    slow_motion_factor: float = Field(
        default=1.0,
        description=(
            "The factor the timestamps were divided by, as supplied. Carried "
            "because a duration read off this clock is a measured number "
            "multiplied by it."
        ),
    )
    methodology: str = Field(description="How the map was built, in words. Never omitted.")

    def knot(self, event: SwingEvent) -> ClockKnot | None:
        """The named knot, or None when this clip did not locate that event."""
        return next((entry for entry in self.knots if entry.event is event), None)


class ClipSummary(BaseModel):
    """What one side of the comparison is, and what it is entitled to claim.

    Every field here is something that can differ between two recordings without
    either swing differing, which is why they are on the report rather than left
    to the caller to remember: a reader looking at a difference needs to be able
    to see that one clip was undistorted and the other was not.
    """

    path: str
    content_key: ContentKey = Field(
        description=(
            "Identifies the recording by content. Two comparisons of the same "
            "pair are the same comparison; two clips with the same filename are "
            "not necessarily the same clip."
        )
    )
    frames: int
    view: CameraView
    lead_side: BodySide | None = None
    calibration: CalibrationStatus
    slow_motion_factor: float
    torso_length: float | None = Field(
        default=None,
        description=(
            "Median shoulder-midpoint to hip-midpoint distance, in frame widths. "
            "Null when the clip never tracked a torso, which is what a clip with "
            "no detected swing looks like. "
            "Every length and speed in this report is divided by it, so it is "
            "reported rather than hidden -- it is the one number that decides "
            "whether two clips of differently-framed subjects are comparable at "
            "all."
        ),
    )
    clock: PhaseClock


class CameraAgreement(BaseModel):
    """Whether the two clips were filmed from the same place, and how closely.

    **The measurement this phase exists for.** Every projected quantity is a fact
    about a camera position as much as about a body, and the view detector's
    three labels are far too coarse to decide whether two clips share one: a
    camera can move twenty degrees round a player and stay comfortably inside
    `face_on`.

    The evidence is the projected shoulder span at address, in torso lengths --
    the same number `ViewEstimate` decides the view from, and the same number
    every foreshortening rotation is referred to. `openness` divides it by the
    widest the line was ever seen in that clip, which makes it the cosine of how
    far off broadside the shoulders were at address; the arccosine of that is an
    estimate of the camera's azimuth.

    **It is an estimate with two real weaknesses, stated rather than buried.** It
    assumes both clips show the same body, because a wider-shouldered player
    projects a wider line from the same place; and it depends on each clip
    containing a frame where the shoulders come square, which Phase 5 recorded
    landmark noise overstating by 12% on the reference footage. It is used to
    refuse and to widen a bracket, never to correct a value.
    """

    reference_span: float | None = Field(
        default=None,
        description=(
            "Projected shoulder span at address, in torso lengths. Null when that "
            "clip produced no view estimate, which is what a clip with no detected "
            "swing does -- there is then no address phase to measure it over."
        ),
    )
    target_span: float | None = None
    span_disagreement: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Absolute difference of the two spans, divided by their mean. Null "
            "when either span is, and **not** a large number standing in for "
            "unknown: the gate below reads this, and an infinity here would refuse "
            "with a reason that sounded like a measurement."
        ),
    )
    reference_openness: float | None = None
    target_openness: float | None = None
    reference_azimuth_deg: float | None = Field(
        default=None,
        description=(
            "How far off broadside this camera stood, in degrees, from the "
            "arccosine of `openness`. Null when the clip supplied no usable "
            "openness -- which is not a claim that the camera was square.\n\n"
            "**Reported and bracketed with, never gated on.** The arccosine is "
            "flat near broadside, so it turns a one-percent error in `openness` "
            "into eight degrees of azimuth; a gate on this number would refuse "
            "every pair ever filmed. The gate is `span_disagreement`, which is "
            "the measurement rather than a function of it."
        ),
    )
    target_azimuth_deg: float | None = None
    azimuth_separation_deg: float | None = Field(
        default=None,
        description=(
            "The furthest apart the two cameras can have been, in degrees: the "
            "**sum** of the two azimuths, not their difference. Both are unsigned, "
            "because foreshortening is identical from either side of broadside and "
            "nothing here separates them -- so two cameras each five degrees off "
            "may have been in the same place or ten degrees apart, and a bracket "
            "has to assume the second."
        ),
    )
    consistent: bool = Field(
        description=(
            "Whether the two cameras are close enough together for a projected "
            "quantity to be compared at all. Decided by the view class and by "
            "`span_disagreement`; false refuses every projected, image-plane and "
            "foreshortened difference by name."
        )
    )
    methodology: str


class BracketTerm(BaseModel):
    """One named contribution to what a difference had to clear.

    Itemised rather than summed into a single number, because the three sources
    call for completely different responses: a clock term is fixed by a faster
    camera, a measurement term by a steadier landmark, and a camera term by
    putting the tripod back where it was.
    """

    source: str = Field(description="What produced this term, in a few words.")
    value: float = Field(ge=0.0, description="Its size, in the metric's own unit.")
    reason: str


class MetricDifference(BaseModel):
    """One quantity measured in both clips, and how far apart the two came out.

    Carries no severity, no score and no judgement of which value is preferable.
    See the module docstring; a test asserts the absence of those field names.
    """

    name: MetricName
    label: str = Field(description="The metric's own display label, including its anchor.")
    group: MetricGroup
    unit: MetricUnit
    basis: MetricBasis
    event: SwingEvent | None = None
    reference_value: float
    target_value: float
    difference: float = Field(description="target minus reference, in the metric's own unit.")
    direction: Direction
    bracket: float = Field(
        ge=0.0,
        description=(
            "What the difference had to clear, in the metric's unit. The **sum** "
            "of the terms below rather than their quadrature, which is where this "
            "parts company with `coaching.combined_bracket`: two anchors in one "
            "clip share a systematic error that largely cancels in their "
            "difference, and two separate recordings share nothing. A sum of "
            "bounds is a bound; a quadrature of bounds is not."
        ),
    )
    terms: list[BracketTerm] = Field(description="Where the bracket came from, itemised.")
    margin: float = Field(
        gt=0.0,
        description=(
            "How far the difference sits outside the bracket, in the metric's "
            "unit. Always positive: a difference that does not clear its bracket "
            "is refused as `unresolved` rather than reported with a margin of "
            "zero."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="The weaker of the two measurements' confidences."
    )
    reference_frames: list[int] = Field(
        description="Frames the reference clip's value was measured on. Indices into its video."
    )
    target_frames: list[int] = Field(description="The same, for the target clip's video.")
    interpretation: str = Field(
        description="What the difference means anatomically from this view, in words."
    )
    methodology: str = Field(description="How both numbers were produced. Never omitted.")


class RefusedDifference(BaseModel):
    """A quantity that could have been compared and deliberately was not.

    Reported rather than omitted, for the reason `RefusedMetric` and
    `RefusedFinding` are: a quantity that did not differ and a quantity that
    could not be compared look identical in a list of differences, and they mean
    opposite things.
    """

    name: MetricName
    label: str
    event: SwingEvent | None = None
    refusal: DifferenceRefusal
    reason: str
    reference_value: float | None = Field(
        default=None,
        description=(
            "The arithmetic, where both values exist. Present on an `unresolved` "
            "refusal precisely because the numbers are not secret -- what is "
            "refused is the claim that they differ, not the values themselves."
        ),
    )
    target_value: float | None = None
    bracket: float | None = None


class ChannelSample(BaseModel):
    """Both clips' value at one position on the normalised axis, and their difference.

    One object rather than parallel arrays, so that "null exactly where nothing
    is supported" survives into TypeScript instead of becoming a check every
    consumer writes separately. Same argument Phase 15 made for a scene point.
    """

    position: float
    reference: float | None = Field(description="Null where the reference clip supports no value.")
    target: float | None = None
    difference: float | None = Field(
        default=None, description="target minus reference. Null unless both sides have a value."
    )
    bracket: float | None = Field(
        default=None,
        description=(
            "What a difference here would have to clear: each clip's event "
            "ambiguity, carried onto this axis and multiplied by the local slope "
            "of its own curve, then summed. Large where the curve is steep, which "
            "is exactly where two curves most easily appear to differ."
        ),
    )
    resolved: bool = Field(
        default=False,
        description="Whether `difference` clears `bracket`. False where either is null.",
    )


class TrajectoryComparison(BaseModel):
    """One signal from both clips, on one axis, with the difference bracketed."""

    channel: TrajectoryChannel
    label: str
    unit: MetricUnit
    basis: MetricBasis
    samples: list[ChannelSample] = Field(default_factory=list)
    resolved_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the sampled positions where the two curves differ by "
            "more than the pair can resolve. A count of where they differ, and "
            "deliberately not a distance between them: a single number summarising "
            "how far apart two swings are is the score this report does not have."
        ),
    )
    largest_difference: float | None = Field(
        default=None, description="Biggest resolved difference, in the channel's unit."
    )
    largest_at: float | None = Field(default=None, description="Where on the axis that sat.")
    refusal: DifferenceRefusal | None = Field(
        default=None, description="Set when this channel was not comparable at all."
    )
    reason: str = Field(default="", description="Why, when `refusal` is set.")


class PathSample(BaseModel):
    """Where each clip's hands were at one position on the normalised axis.

    In torso lengths from that clip's own address hand position, so two subjects
    of different sizes filmed from different distances are on one scale. `x` is
    along the image's horizontal and `y` is upward -- an image direction, not an
    anatomical one, because a down-the-line camera may be in front of the player
    or behind them and nothing here separates those.
    """

    position: float
    reference_x: float | None = None
    reference_y: float | None = None
    target_x: float | None = None
    target_y: float | None = None


class HandPathOverlay(BaseModel):
    """Both clips' hand arcs, resampled onto the shared axis.

    The one part of this report that is a picture rather than a number, and it
    carries no bracket: a bracket on a two-dimensional path would be an ellipse
    per sample, and nothing here has measured the second axis of one. It is drawn
    as evidence to look at, alongside numbers that are bracketed, and the
    contract says so rather than leaving a reader to assume the arcs are as well
    determined as the curves beside them.
    """

    samples: list[PathSample] = Field(default_factory=list)
    unit: MetricUnit = MetricUnit.TORSO_LENGTHS
    comparable: bool = Field(
        description="False when the two clips' views differ, which makes the arcs two shapes."
    )
    reason: str = Field(default="")


class ComparisonConfig(BaseModel, extra="forbid"):
    """Policy for the comparison layer.

    **Structural bounds, not golf norms**, as everywhere else here. Nothing in
    this file says what a good swing looks like, and the two camera tolerances
    are the only numbers with a measurement behind them rather than a statement
    of policy.
    """

    samples: int = Field(
        default=121,
        ge=8,
        le=2001,
        description=(
            "Points on the normalised axis, spanning takeaway to finish. 121 puts "
            "one every fortieth of a phase, which is finer than the frame rate of "
            "any clip this project has and coarse enough that the payload stays "
            "small. Nothing is interpolated beyond the clip's own samples: a "
            "position whose neighbours are unsupported comes back null."
        ),
    )
    min_confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence below which a measurement does not support a comparison. "
            "Stated policy, and deliberately the same number `CoachingConfig` "
            "uses: a measurement good enough to compare against a published band "
            "is good enough to compare against another measurement."
        ),
    )
    max_span_disagreement: float = Field(
        default=0.10,
        ge=0.0,
        description=(
            "How far apart the two clips' address shoulder spans may sit, as a "
            "fraction of their mean, before the cameras are judged to have moved "
            "and every projected difference is refused.\n\n"
            "**The one threshold here with a measurement behind it.**"
            " `scripts/benchmark_compare.py --sweep camera` films one unchanged "
            "swing from a series of azimuths at the landmark noise Phase 3 "
            "measured on real footage, and reports both what the span does and "
            "what the metrics do. The gate is on the span rather than on the "
            "azimuth derived from it because the span is the measurement: the "
            "arccosine that turns it into an angle is flat near broadside and "
            "amplifies a percent into eight degrees."
        ),
    )


class SwingComparison(BaseModel):
    """Everything this system concluded about two recordings, side by side.

    `computed` is false when either clip produced no swing, and then every list
    is empty rather than populated from whichever half worked.

    **Two recordings, and not necessarily two swings by one player.** Nothing
    here can tell whether the same person made both, and nothing here asks: the
    lengths are in each subject's own torso lengths and the durations are on each
    clip's own clock, which is what makes a comparison arithmetically meaningful
    across subjects and framings. What it does not make it is a comparison of
    like with like -- two people are not two attempts -- and that judgement
    belongs to whoever chose the two files.
    """

    schema_version: int = COMPARISON_SCHEMA_VERSION
    computed: bool
    reference: ClipSummary
    target: ClipSummary
    camera: CameraAgreement
    differences: list[MetricDifference] = Field(default_factory=list)
    refused: list[RefusedDifference] = Field(default_factory=list)
    trajectories: list[TrajectoryComparison] = Field(default_factory=list)
    hand_path: HandPathOverlay | None = None
    metrics_considered: int = Field(
        default=0, description="Quantities examined, whatever came out of them."
    )
    config: ComparisonConfig = Field(default_factory=ComparisonConfig)
    warnings: list[str] = Field(default_factory=list)

    def difference(
        self, name: MetricName, event: SwingEvent | None = None
    ) -> MetricDifference | None:
        """The named difference, or None when that quantity was not compared."""
        return next(
            (entry for entry in self.differences if entry.name is name and entry.event is event),
            None,
        )

    def refusal(
        self, name: MetricName, event: SwingEvent | None = None
    ) -> RefusedDifference | None:
        """The named refusal, or None when that quantity was not refused."""
        return next(
            (entry for entry in self.refused if entry.name is name and entry.event is event),
            None,
        )

    def trajectory(self, channel: TrajectoryChannel) -> TrajectoryComparison | None:
        """The named channel, or None when it was not built."""
        return next((entry for entry in self.trajectories if entry.channel is channel), None)
