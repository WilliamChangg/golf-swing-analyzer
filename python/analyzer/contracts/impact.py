"""One instant, four ways of finding it, and the rule for reporting one number.

By Phase 11 this engine can locate impact four different ways, built in three
different phases out of three different kinds of evidence. That is a good problem
and it is also exactly the failure `contracts/phases.py` warned about when it
kept tempo out of Phase 4 and `contracts/club.py` repeated when it refused to let
the club-head estimate overwrite the kinematic one: **reporting one instant from
several places with several provenances is how a reader ends up with several
numbers and no way to tell which they are reading.**

So this module exists to produce one number with one named provenance and one
stated error bar, and to keep every other number visible beside it rather than
absorbed into it.

## The four, and what each actually measures

    ball departure   the ball stopped being visible      an observation
    club head        the head reached its lowest point   a measurement, of impact
    hand low         the hands reached their lowest      a proxy, well correlated
    hand speed       the hands reached peak speed        a proxy, biased early

Only the first observes the event. The second measures the club at a moment
defined geometrically rather than by contact, and refuses on most real footage
because it needs the head visible through the blurriest frames in the clip. The
last two read the *hands*, which are not what strikes the ball -- they reach peak
speed before the head arrives, which Phase 4's own methodology string says in
words and which nothing until now could put a number on.

## Why they are not averaged

Averaging is the obvious fusion and it is wrong here, for a reason that is about
the estimators rather than about arithmetic. Three of the four are **biased**,
and in a known direction; the fourth is not. Averaging an unbiased observation
with a biased proxy moves the answer away from the truth by a fraction of the
bias, and it does so while producing a number whose provenance is "a bit of each",
which no reader can reason about and no later phase can correct.

The rule instead is **precedence, with the disagreements measured and kept**:

1.  The best available source supplies the answer, where "best" is a fixed order
    with a reason attached to each step rather than a comparison of confidences.
    An observation beats a measurement of a correlated quantity; a measurement of
    a correlated quantity beats a proxy; and a proxy with no known bias beats one
    with a known bias.
2.  Every other source that answered is carried as an `ImpactCandidate` with its
    delta to the chosen one. Those deltas are not diagnostics -- they are
    **measurements of the other methods' bias**, and they are the only way this
    project will ever learn how early the hand-speed peak runs on real footage.
3.  A disagreement too large to be a bias is a warning rather than a silent
    preference. If two sources sit half a downswing apart, one of them is not
    measuring impact, and a report that quietly picked the higher-precedence one
    would be hiding the most useful thing it knows.

## The error bar is a bracket, not a deviation

`uncertainty_s` on a ball departure is one frame interval and it is a **bound**:
the ball was present at one end and absent at the other, so impact is inside,
full stop. On the kinematic sources it is the width of the smoothing window the
clip's frame rate forced, and that is a much weaker statement -- a scale on which
the peak could have moved, not an interval it is inside.

The two are not comparable and the field documentation says which is which,
because a consumer that treated them alike would read a 24 fps clip's +-80 ms
kinematic scale and a 240 fps clip's +-4 ms bracket as the same kind of claim.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# Bump on any change that alters what a fused impact means.
IMPACT_SCHEMA_VERSION = 1


class ImpactSource(StrEnum):
    """Where an impact estimate came from.

    Listed in precedence order, which is also the order of the module docstring's
    table. The order is a property of what each one measures and not of how
    confident any particular clip's version of it happens to be -- a confident
    proxy is still a proxy.

    BALL_DEPARTURE
        Phase 11. The last frame carrying a ball and the first carrying none.
        The only one of the four that observes the event.
    CLUB_HEAD
        Phase 10. The lowest point the club head reached after the top, over the
        frames whose edge evidence ran to the end of the club. Measures the club
        at impact rather than impact itself, and refuses whenever the downswing
        was not more than half tracked -- which is most real footage.
    HAND_LOW
        Phase 4's existing corroboration signal: the lowest point the hands
        reached after the top. A proxy, and empirically the better of the two
        hand-derived ones on slow-motion footage.
    HAND_SPEED
        Phase 4's primary estimate: peak hand speed. A proxy with a **known sign
        of bias** -- the hands peak before the club head arrives at the ball, so
        it runs early.
    """

    BALL_DEPARTURE = "ball_departure"
    CLUB_HEAD = "club_head"
    HAND_LOW = "hand_low"
    HAND_SPEED = "hand_speed"


class ImpactCandidate(BaseModel):
    """One source's answer, and how far it sits from the one that was reported.

    Kept for every source that answered, including the one that won -- which then
    carries a `delta_s` of zero. Dropping the winner from the list would make the
    list mean "the ones that lost", and a reader comparing methods wants the full
    set with the chosen one marked.
    """

    source: ImpactSource
    frame_index: int
    timestamp_s: float
    uncertainty_s: float | None = Field(
        default=None,
        description=(
            "The error bar on this source, in real seconds. **Read "
            "`FusedImpact.uncertainty_is_bracket` before comparing two of these**: "
            "on a ball departure it is an interval impact is inside, and on the "
            "kinematic sources it is the scale of the smoothing that located the "
            "peak, which is not the same kind of claim."
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="What the producing phase thought of its own answer, where it said.",
    )
    methodology: str = Field(description="The rule that produced this frame, in words.")
    delta_s: float | None = Field(
        default=None,
        description=(
            "Seconds from the reported instant to this one. Positive means this "
            "source is **later**. For HAND_SPEED a negative value is the expected "
            "case and its magnitude is a measurement of how early the hands peak."
        ),
    )
    delta_frames: int | None = Field(
        default=None,
        description=(
            "The same difference in frames, which is the form a reader checks "
            "against a video scrubber. Carried alongside rather than derived, "
            "because on variable-rate footage the two do not convert."
        ),
    )


class FusedImpact(BaseModel):
    """One impact instant, with one provenance and one error bar.

    What a coaching rule, a comparison view or an export should read. The
    individual phases keep their own estimates -- `SwingPhases` still reports the
    hand-speed peak and `ClubTrackingReport` still reports the club-head minimum,
    both unchanged -- and this is the single place that says which of them to
    believe for this clip and by how much the others disagreed.

    **`observed` is the field that changes what may be claimed downstream.** When
    it is true the instant rests on a frame in which the ball was there and a
    frame in which it was not; when it is false, every number here is derived from
    the motion of the player's hands or from where a club head was seen, and a
    finding phrased as "at impact" is really "at the estimated impact".
    """

    schema_version: int = IMPACT_SCHEMA_VERSION
    frame_index: int
    timestamp_s: float
    source: ImpactSource = Field(description="Which of the four supplied the reported instant.")
    observed: bool = Field(
        description=(
            "True only for BALL_DEPARTURE. The distinction `contracts/phases.py` "
            "has carried since Phase 4 -- a kinematic estimate is not an "
            "observation -- reduced to a single field a consumer can gate on."
        )
    )
    uncertainty_s: float | None = Field(
        default=None, description="The error bar on the reported instant, in real seconds."
    )
    uncertainty_is_bracket: bool = Field(
        default=False,
        description=(
            "True when `uncertainty_s` is an interval the instant is **inside** -- "
            "which is the case for a ball departure and for nothing else. False "
            "means it is a scale on which the estimate could have moved. See the "
            "module docstring; the two are not comparable and conflating them "
            "would read a 24 fps clip's smoothing width as a guarantee."
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The producing phase's own confidence in the reported instant.",
    )
    methodology: str = Field(description="The rule that produced the reported instant, in words.")

    candidates: list[ImpactCandidate] = Field(
        default_factory=list,
        description=(
            "Every source that answered, the chosen one included, each with its "
            "delta. **The deltas are the measurement**, not a diagnostic: they are "
            "how the bias of the kinematic estimates gets quantified, on this clip "
            "and eventually across a set of them."
        ),
    )
    agreement_s: float | None = Field(
        default=None,
        description=(
            "Largest absolute delta among the sources that answered, or None when "
            "only one did. The spread of the methods on this clip."
        ),
    )
    warnings: list[str] = Field(default_factory=list)

    def candidate(self, source: ImpactSource) -> ImpactCandidate | None:
        """One source's answer, or None if that source did not produce one."""
        return next((entry for entry in self.candidates if entry.source is source), None)
