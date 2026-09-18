"""Typed contracts for ball detection.

The first thing this engine measures by **watching something stop existing**.

Every layer below measures a presence. A landmark is somewhere, a shaft points
somewhere, a reconstructed joint sits at some depth, and the measurement is a
reading taken off a thing that is in the picture. A golf ball is in the picture
for hundreds of frames doing nothing at all, and then it is not, and the single
instant this phase exists to locate is the boundary between those two states.

## Why the absence is the measurement, and not a failure to detect it

A ball in flight is the obvious thing to track and it is the wrong thing to ask
for. It leaves the face at around 70 m/s, so at a 1/60 s exposure it smears
across roughly a metre of the picture and, at any consumer frame rate, it is
several frames past the edge of the frame before it has been drawn sharply once.
Phase 10 met the same wall with the club head and named it: blur is a cliff, not
a decline, and past it lowering a threshold recovers nothing.

The ball at rest has the opposite character. It is still, round, high-contrast
and in the same place for the whole of address, the backswing and the downswing
-- hundreds of frames of unambiguous evidence for one small object -- and the
event being timed is not something it *does* but the frame after the last one it
appears in. So this phase measures the ball where the ball is easy, and reads
impact off the edge of that interval.

**The consequence is an error bar of exactly one frame interval, whatever the
footage.** Nothing here estimates a peak, fits a curve or smooths a signal, so
nothing here has a resolution that degrades. Impact happened after the last frame
carrying a ball and not after the first frame without one, and those are one
frame apart by construction. At 30 fps that is 33 ms and at 240 fps it is 4 ms,
and in both cases it is a **bound** rather than a standard deviation.

Set that against `DetectedEvent`'s kinematic impact, whose uncertainty is the
width of the smoothing window that the clip's own frame rate forced -- which
`EventConfidence.resolution` reports, and which on a 24 fps clip is about as long
as the downswing being measured. This is the first impact estimate in the project
that is an observation.

## What it cannot claim, and the one that sounds fatal

**The ball is not observed being struck. It is observed stopping being visible.**
Those are the same instant for a struck ball and they are not the same statement,
and the difference is where every failure of this phase lives.

The one that sounds fatal first: *at impact the club head is at the ball, so a
ball hidden behind the club head and a ball that has left look identical.* True,
and it does not matter for this measurement -- both happen at impact, and the
frame is the same frame. It would matter if the club covered the ball
appreciably *before* contact, which it can on a down-the-line view where the
shaft crosses the line of sight, and that is worth at most a frame or two.
`DepartureConfidence.abruptness` is what notices: a ball being progressively
covered dims for several frames before it goes, and a struck ball reads at full
strength in the last frame it exists.

The ones that are real:

    a practice swing       nothing departs, and nothing is reported
    a ball that rolls off  it departs, in the wrong phase, and is refused
    a ball hit at the lens the departure is real and impact is right anyway
    a body in the way      the ball comes back, so it was occluded, not struck

Only the second and fourth need a check, and both are checked: the departure must
land in the downswing, and it must be **permanent**. A ball that reappears was
never gone.

## The identification is circular, and the circle is closed deliberately

The ball is picked out of the candidates partly by the fact that it leaves --
nothing else in a golf frame is a small still round bright thing that vanishes
once and never returns. Scoring the confidence of the departure on the departure
would then be an argument with itself.

So the two confidences are separate objects and they are computed from disjoint
evidence. `BallConfidence` scores **one frame's observation** from the image at
that frame and the frames around it: how far the ball stood out, whether anything
else stood out as far, whether it stayed put. `DepartureConfidence` scores **the
instant** from what happened either side of it: how well established the ball was
before, how abruptly it went, whether it stayed gone. Nothing in the first reads
the departure, and nothing in the second reads whether the tracker liked its
candidate.

## The three factors, a third time

    contrast   how far the ball stands out from its surround   the fit
    margin     whether anything else in the region stands out  a check
    stillness  whether it stayed where a teed ball stays       a check

The shape Phases 8, 9 and 10 each arrived at, and it recurs for the reason it
recurred there rather than out of consistency: **the number the detector reports
about its own fit is blind to the failure that matters.** A white shoe, a range
ball three feet away, a sprinkler head, a bare patch of mat and a daisy all have
excellent contrast. What separates the ball from them is not how well it was
seen. It is that nothing else in the region was seen as well (`margin`), that it
did not move (`stillness`), and that it left (`DepartureConfidence`).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.phases import SwingPhase
from analyzer.contracts.pose import FrameGeometry, LandmarkSpace

# Bump on any change that alters what a reported ball track means.
BALL_SCHEMA_VERSION = 1


class BallRefusal(StrEnum):
    """Why one frame carries no ball.

    Counted separately per reason, as `ShaftRefusal` is and for the same reason:
    each names a different thing to change about the capture, and a single "not
    detected" tally would say that something went wrong and nothing about what.

    `GONE` is the one that is not a failure. It is what every frame after a
    located departure carries, and separating it from `NO_CANDIDATE` is the
    difference between "the ball has been struck" and "the ball should be here
    and is not" -- which are the same picture and opposite facts.

    NO_ANCHOR
        The pose layer produced no hand position at this instant, so there was
        nowhere to put the search region. Nothing about the ball.
    OUT_OF_FRAME
        The search region fell wholly or mostly outside the picture. The ball is
        below the hands by about a club length, so this is what a clip framed
        tight on the upper body produces, and it is a framing fix rather than a
        detection one.
    NO_CANDIDATE
        The region held nothing of a ball's size that was round enough and stood
        out enough from its surround. Before the ball is established this is a
        contrast or framing problem; after it, and before a departure is located,
        it is a frame in which the ball was momentarily lost.
    AMBIGUOUS
        Two or more candidates were equally plausible. A second ball on the mat,
        a white tee marker, a bright reflection. `data/README.md` asks for one
        ball in the hitting area for exactly this reason.
    MOVED
        The best candidate sat further from the established position than a teed
        ball sits from itself. Whatever it is, it is not the ball this clip
        established -- most often a shoe or a glove entering the region.
    LOW_CONFIDENCE
        A candidate survived every structural check and the product of its three
        factors still sat below the bound.
    GONE
        The ball departed before this frame. **Not a failure**: the measured
        outcome of the clip, carried per frame so that a reader and an overlay
        can see where the ball stopped being expected.
    """

    NO_ANCHOR = "no_anchor"
    OUT_OF_FRAME = "out_of_frame"
    NO_CANDIDATE = "no_candidate"
    AMBIGUOUS = "ambiguous"
    MOVED = "moved"
    LOW_CONFIDENCE = "low_confidence"
    GONE = "gone"


class BallDetectorInfo(BaseModel):
    """Which detector produced a track, pinned so results stay attributable.

    `ClubDetectorInfo`'s counterpart and deliberately identical in shape: a
    classical detector has no weights to digest and no training set to name, but
    it does have a method and a library version, and both are the measurement
    instrument. A result from a different OpenCV is a different result.
    """

    name: str = Field(description="Implementation name, e.g. 'contrast_blob'.")
    method: str = Field(
        description="The pipeline in words, e.g. 'roi -> threshold -> components -> geometry'."
    )
    opencv_version: str = Field(description="The version of OpenCV that ran, as measured.")


class BallConfidence(BaseModel):
    """Why one frame's ball is or is not trustworthy, decomposed.

    `overall` is the **product** of the three, as in `ShaftConfidence` and
    `EventConfidence`, because a ball observation needs all three and a product
    says so where an average would let a strong factor cover for a fatal one.

    Note again which of the three the image at this frame supplies. `contrast`
    does, and it is the one a blob detector naturally reports. The other two come
    from what the candidate was compared against -- everything else in the region,
    and where the ball was in the frames around this one -- and they are the two
    that catch the failures contrast cannot see. See the module docstring.
    """

    overall: float = Field(ge=0.0, le=1.0, description="Product of the three factors below.")
    contrast: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How far the candidate's interior stands away from the ring of "
            "background just outside it, as a fraction of the range the region "
            "spans. Measured against the **local** surround rather than against a "
            "fixed intensity, so a white ball on a dark mat and a dark ball on a "
            "bright one score alike -- the sign of the step is not read."
        ),
    )
    margin: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How far the accepted candidate's contrast stands above that of the "
            "best other candidate **sitting in the same place** -- within the drift "
            "radius of the established position -- as a fraction of its own. 1.0 "
            "when nothing else was there, which is the ordinary case.\n\n"
            "Co-located rather than region-wide, and the distinction matters. A "
            "white shoe on the far side of the region is not a competitor at this "
            "frame: the choice was made on position, and the shoe scores nothing on "
            "`stillness`. What this catches is the competitor position cannot "
            "separate -- a ball against a mat seam, a ball and its own shadow, a "
            "ball on a white tee. **Whether the clip identified the right object in "
            "the first place is a different question with a different answer**, and "
            "it is `EstablishedBall.margin`."
        ),
    )
    stillness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How close this frame's candidate sits to the position established "
            "across the clip, scored against `BallConfig.max_drift_torso`. 1.0 on "
            "the established position, falling to zero at the bound.\n\n"
            "A teed ball does not move at all, so this is a much sharper check "
            "than `ShaftConfidence.continuity` -- the club's equivalent has to "
            "allow for a club that is genuinely turning, and this allows only for "
            "the detector's own noise."
        ),
    )


class BallObservation(BaseModel):
    """The ball measured in one frame: a position, a size, and what it beat.

    Coordinates are **FRAME_WIDTHS** -- isotropic, y increasing upward, both axes
    divided by the frame width -- like every measured quantity above Phase 3.
    Detection runs in pixels, and the conversion is a scale and a y flip applied
    once, on the way out of the detector.

    There is no velocity here and there is deliberately no field for one. A teed
    ball has no velocity, and the frames in which it does have one are the frames
    in which it cannot be seen. See the module docstring.
    """

    x: float = Field(description="Ball centre in frame widths.")
    y: float
    radius: float = Field(gt=0.0, description="Apparent radius in frame widths.")
    radius_torso: float = Field(
        gt=0.0,
        description=(
            "The same radius in the subject's own torso lengths, which is the "
            "form that survives a change of camera distance. A golf ball is 42.7 "
            "mm and an adult's shoulder-to-hip span about 450 mm, so a ball is "
            "close to 0.047 torso lengths across and this should read near 0.024."
        ),
    )
    confidence: BallConfidence
    runner_up_distance_torso: float | None = Field(
        default=None,
        description=(
            "How far the best **rejected** candidate sat from this one, in torso "
            "lengths, where there was one. The evidence behind `confidence.margin`: "
            "a reader who wants to know whether the detector was choosing between "
            "a ball and a shoe can see how far apart they were."
        ),
    )
    candidates: int = Field(
        ge=1, description="Regions that satisfied the size and shape bounds at this frame."
    )


class BallFrame(BaseModel):
    """What was concluded about the ball in one video frame.

    A frame with no ball is kept rather than dropped, which is `PoseFrame`'s and
    `ClubFrame`'s decision and made for the same two reasons: the absence is
    itself information -- here it is *the* information -- and dropping the frame
    would break the correspondence between frame index and position in the
    sequence.

    Exactly one of `ball` and `refusal` is set.
    """

    frame_index: int
    timestamp_s: float = Field(
        description=(
            "Elapsed real seconds from the first frame, already divided by any "
            "slow-motion factor, as everywhere above Phase 3."
        )
    )
    detected: bool
    ball: BallObservation | None = Field(
        default=None, description="The measurement, when there is one."
    )
    refusal: BallRefusal | None = Field(
        default=None, description="Why there is none. Set exactly when `detected` is false."
    )
    phase: SwingPhase | None = Field(
        default=None,
        description="Which swing phase this frame falls in, where a swing was detected.",
    )


class EstablishedBall(BaseModel):
    """The ball this clip settled on, and how much evidence stands behind it.

    **The identification, separated from the measurement.** Before any instant can
    be read off a departure there has to be an agreed answer to "which of the
    bright round things in this picture is the ball", and that answer is made once
    for a clip from the frames where it is easiest -- the ones before the club
    arrives anywhere near it.

    The same argument Phase 10 makes about seeding a track at the strongest
    evidence and growing outward, applied to a different question. There the
    evidence was best at address because the club is nearly still; here it is best
    at address because the ball is still, unoccluded, and sitting there for longer
    than any other event in the clip lasts.
    """

    x: float = Field(description="Agreed ball position in frame widths.")
    y: float
    radius: float = Field(gt=0.0)
    radius_torso: float = Field(gt=0.0)
    frames: int = Field(
        ge=1,
        description=(
            "Frames that agreed on this position. The weight behind the "
            "identification: one bright blob in one frame is not a ball, and a "
            "hundred frames of the same blob in the same place is hard to be."
        ),
    )
    searched_frames: int = Field(
        ge=1, description="Frames the identification was made from, agreeing or not."
    )
    spread_torso: float = Field(
        ge=0.0,
        description=(
            "How far the agreeing observations scattered about the agreed "
            "position, in torso lengths. A teed ball scatters by the detector's "
            "own noise and nothing else, so a large value here means the "
            "identification settled on something that moves."
        ),
    )
    median_contrast: float = Field(ge=0.0, le=1.0)
    margin: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How far this position stood above the next-best candidate **the "
            "choice was actually made over**, as a fraction of its own score. "
            "**The check that catches the failure this phase most needs "
            "catching**, and the one no single frame can make: reporting an "
            "instant read off the wrong object.\n\n"
            "Where something departed, the pool is the other things that also "
            "departed, and 1.0 means the ball was the only object in the region "
            "that left -- which is the ordinary result, and a strong statement. "
            "Where nothing departed, the pool is every stationary candidate, "
            "because then nothing has singled any of them out.\n\n"
            "The pool matters. A tee marker beside the ball for a whole clip is "
            "agreed on by exactly as many frames as the ball, so scoring against "
            "every stationary candidate would report a coin toss on any clip with "
            "a second white object in it -- and a margin that says 'ambiguous' "
            "everywhere is not read anywhere, least of all in the case it exists "
            "for: **two objects that both leave**, which one view cannot resolve "
            "and which scores near zero here.\n\n"
            "Distinct from `BallConfidence.margin`, which asks a much narrower "
            "question about one frame. A clip can score 1.0 on every frame's margin "
            "and near zero here, and that combination is exactly the dangerous one: "
            "an unambiguous track of an object that may not be the ball."
        ),
    )
    runner_up_distance_torso: float | None = Field(
        default=None,
        description=(
            "How far that next-best candidate sat from this one, in torso lengths, "
            "where there was one. The evidence behind `margin`, carried so it can "
            "be audited rather than trusted -- a reader can see whether the "
            "competitor was a shadow a ball's width away or a shoe half a body away."
        ),
    )
    departed: bool = Field(
        default=False,
        description=(
            "Whether this position is the one that emptied, which is how it was "
            "chosen among the stationary candidates when more than one qualified. "
            "False means no candidate departed and this is simply the "
            "best-established of them -- the practice-swing case."
        ),
    )
    methodology: str


class DepartureConfidence(BaseModel):
    """Why a located departure is or is not trustworthy, decomposed.

    Deliberately a **different object** from `BallConfidence` and computed from
    disjoint evidence; see the module docstring on why the identification would
    otherwise be arguing with itself.

    `overall` is the product, for the third time in this file and the same reason
    each time: an instant read off a disappearance needs a ball that was really
    there, a disappearance that was really abrupt, and an absence that really
    lasted. Any one of the three at zero makes the other two worthless.
    """

    overall: float = Field(ge=0.0, le=1.0, description="Product of the three factors below.")
    establishment: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the frames before the departure that carried the ball. "
            "**Was there a ball there at all**: a clip that saw it in six frames "
            "out of two hundred has found something that flickers, and a "
            "disappearance is what a flicker looks like from one side."
        ),
    )
    abruptness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Strength of the last few observations before the departure against "
            "the ball's own established strength. 1.0 means it was seen as well in "
            "its final frame as it ever was, which is what a struck ball does. "
            "**The factor that separates a strike from an occlusion**: something "
            "moving in front of the ball dims it for several frames first, and a "
            "club head does exactly that on a down-the-line view."
        ),
    )
    permanence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Frames after the departure in which **the image offered nothing** at "
            "the established position, as a fraction of the frames "
            "`BallConfig.permanence_window_s` asks for. **The only factor that "
            "separates a strike from an occlusion**, because nothing at the "
            "instant itself can: a ball hidden behind a club head and a ball that "
            "has gone are the same picture.\n\n"
            "Measured on the detector's candidates rather than on the frames the "
            "tracker accepted, and the difference is the case this factor exists "
            "for. A ball being progressively covered dims until it falls under the "
            "confidence bound, which ends the run early; every frame after that is "
            "then one the tracker rejected, so an acceptance-based score would read "
            "1.0 while the ball was still plainly in the picture. The dim ball is "
            "still a candidate, so reading candidates notices and reading verdicts "
            "does not.\n\n"
            "It falls for three different reasons and that is deliberate rather "
            "than a conflation: the ball came back, so it was occluded; the ball "
            "faded rather than left, so the instant runs early; or the clip ended "
            "before the absence could be verified, so the check was never made. All "
            "three mean the same thing about how much this instant is worth, and "
            "the warnings on the report name them."
        ),
    )


class BallDeparture(BaseModel):
    """The instant the ball stopped being visible, and what brackets it.

    **The whole output of this phase.** Impact is not `last_seen_frame` and it is
    not `first_absent_frame`; it is somewhere in the interval between them, and
    the interval is one frame long. Both ends are carried rather than a single
    number plus an error, because the bound is a genuine bracket and not a
    symmetric uncertainty about a midpoint: the ball was definitely present at one
    end and definitely absent at the other.

    `frame_index` and `timestamp_s` name `first_absent_frame`, which is the
    earliest frame at which impact has certainly happened. Choosing the later end
    is a convention and is stated on the field, because the alternative convention
    differs by a frame and nothing in the picture prefers either.
    """

    frame_index: int = Field(
        description=(
            "`first_absent_frame`. The earliest frame by which impact has "
            "certainly occurred, which is the conservative end of the bracket and "
            "the one every consumer of this contract reads."
        )
    )
    timestamp_s: float
    last_seen_frame: int = Field(description="Final frame carrying a ball. Impact is after this.")
    last_seen_s: float
    first_absent_frame: int = Field(
        description="First frame carrying none. Impact is at or before this."
    )
    first_absent_s: float
    interval_s: float = Field(
        gt=0.0,
        description=(
            "`first_absent_s - last_seen_s`: **the uncertainty of this estimate, "
            "as a bound rather than a deviation.** One frame interval, whatever "
            "the footage, because nothing here fits or smooths anything. Compare "
            "against the kinematic estimate's, which is the width of the smoothing "
            "window the clip's frame rate forced."
        ),
    )
    phase: SwingPhase | None = Field(
        default=None,
        description=(
            "Which swing phase the departure landed in. **The gate**, applied to "
            "this phase's own derived quantity the way Phase 10 gates its "
            "club-head impact on downswing coverage: a ball that leaves during the "
            "backswing left for some reason other than being hit."
        ),
    )
    confidence: DepartureConfidence
    methodology: str
    kinematic_frame: int | None = Field(
        default=None, description="Where Phase 4 put impact, for comparison. None if it found none."
    )
    delta_s: float | None = Field(
        default=None,
        description=(
            "Seconds from the kinematic estimate to this one. **Positive means the "
            "hands peaked early**, which is the direction the physics predicts and "
            "which Phase 4's own methodology string already warns about. This is "
            "the number that turns that warning into a measurement."
        ),
    )


class BallQuality(BaseModel):
    """What the track is worth, in the senses that differ.

    Shaped like `ClubTrackingQuality`, `ReconstructionQuality` and
    `CalibrationQuality`, because the lesson was the same a fourth time: report
    the fit, the check and the capture separately, and label each with the
    question it answers.
    """

    median_contrast: float | None = None
    median_margin: float | None = Field(
        default=None,
        description="Median margin over the best rejected candidate. The check contrast cannot make.",
    )
    median_stillness: float | None = None
    median_confidence: float | None = None

    radius_px: float | None = Field(
        default=None,
        description=(
            "Median apparent radius in pixels. **The capture number.** A ball "
            "under about three pixels of radius has no shape left to be round, and "
            "no threshold recovers one -- it is a framing and resolution fact, the "
            "way `ClubTrackingQuality.max_blur_px` is a shutter fact."
        ),
    )
    spread_torso: float | None = Field(
        default=None,
        description="How far the observations scattered about the established position.",
    )
    drift_torso: float | None = Field(
        default=None,
        description=(
            "How far the ball's image position travelled between the first and "
            "last frame it was seen in, in torso lengths. **A measurement of the "
            "camera, not of the ball.** A teed ball does not move; if its picture "
            "does, the picture moved -- handheld shake, digital stabilisation, or a "
            "slow pan.\n\n"
            "Reported because the alternative is for it to act silently. The "
            "per-frame drift check follows the ball rather than holding a fixed "
            "position, so a drifting camera costs nothing until it exceeds "
            "`BallConfig.max_drift_torso` between two consecutive sightings -- but a "
            "reader comparing two clips should know which one was filmed off a "
            "tripod. Measured at 0.02 torso lengths on the reference footage in "
            "this repository, which is a phone held in a hand."
        ),
    )

    methodology: str


class BallTrackingReport(BaseModel):
    """Everything the engine concluded about the ball in one clip.

    The per-frame observations are here for `ClubTrackingReport`'s reasons: one
    small object per frame is kilobytes where a pose sequence is megabytes, and a
    detector is checked by drawing it on the frames it came from.

    `detected` is false whenever no frame produced a ball, and then `quality` and
    `established` are None rather than zeroed -- "nothing was found" and "a ball
    was found in none of the frames" are the same sentence, and neither of them is
    a measurement of zero.

    **`detected` and `departure` are independent.** A clip can establish a ball
    perfectly and locate no departure, which is what a practice swing, a clip that
    ends before impact, and a clip whose ball rolls away all produce; and that is
    the correct output for all three rather than a failure.
    """

    schema_version: int = BALL_SCHEMA_VERSION
    detected: bool
    video_path: str
    detector: BallDetectorInfo
    space: LandmarkSpace = Field(
        default=LandmarkSpace.FRAME_WIDTHS,
        description=(
            "The frame the observations are in: isotropic, y up, both axes in "
            "frame widths. **Not metric** -- the ball's position in one view is a "
            "point in a picture, and where it sat in the world needs two "
            "calibrated views."
        ),
    )
    geometry: FrameGeometry = Field(
        description="Displayed pixel dimensions, carried so the report is self-contained for drawing."
    )
    torso_length: float = Field(
        description="The subject's shoulder-to-hip span in frame widths. The scale every bound is in."
    )
    slow_motion_factor: float = Field(
        default=1.0,
        description="The clip's factor. Timestamps here are real seconds, as everywhere above Phase 3.",
    )

    frame_count: int
    observed_frames: int
    pre_departure_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the frames **before the departure** that carried a ball. "
            "**The number to read**, and the reason a clip-wide rate is not "
            "reported as the headline: after a located departure the ball is "
            "correctly absent, so every frame of the follow-through would count "
            "against a coverage figure that spanned the clip. Where no departure "
            "was located this spans every frame."
        ),
    )

    established: EstablishedBall | None = None
    departure: BallDeparture | None = Field(
        default=None,
        description=(
            "The instant the ball stopped being visible, where one was located and "
            "survived the phase gate. **Absent is a normal outcome**: a practice "
            "swing has no departure to find."
        ),
    )
    quality: BallQuality | None = None
    frames: list[BallFrame] = Field(
        default_factory=list, description="One entry per video frame, observed or refused."
    )
    refusals: dict[BallRefusal, int] = Field(
        default_factory=dict, description="Frames carrying no ball, counted by reason."
    )
    tracked_at: datetime | None = None
    elapsed_s: float = 0.0
    ms_per_frame: float = 0.0

    config: BallConfig
    refusal: str | None = Field(
        default=None,
        description="Why nothing was found, in words. Set exactly when `detected` is false.",
    )
    warnings: list[str] = Field(default_factory=list)

    def frame(self, frame_index: int) -> BallFrame | None:
        """The entry for one frame, or None if the clip does not contain it."""
        if 0 <= frame_index < len(self.frames):
            entry = self.frames[frame_index]
            if entry.frame_index == frame_index:
                return entry
        return next((item for item in self.frames if item.frame_index == frame_index), None)


class BallConfig(BaseModel, extra="forbid"):
    """Policy for ball detection.

    Two kinds of number live here and the field documentation says which is which,
    as `PhaseConfig` and `ClubConfig` do. **Structural bounds** describe what a
    teed golf ball physically is -- how big against a body, how far from the hands,
    how little it moves -- and follow from published dimensions rather than from
    tuning. **Measured defaults** come from `scripts/benchmark_ball.py`, which
    renders a ball of known size that vanishes at a known frame and sweeps each
    threshold against the located instant.

    Nothing here is tuned against hand-labelled ball positions on real footage,
    because none exist. That is Phase 12's labelling tool, and until it exists the
    honest statement is that the synthetic sweep establishes the *shape* of each
    relationship and where on the curve to refuse is a judgement about what a
    wrong answer costs.
    """

    search_radius_torso: float = Field(
        default=1.5,
        gt=0.0,
        description=(
            "Radius of the region searched around the **ankle midpoint** at "
            "address, in torso lengths. **Structural**: a teed ball rests on the "
            "ground within about a torso length of the feet -- forward of centre "
            "for a driver, back of centre for a wedge, and never far from the "
            "player's stance -- so this leaves half a torso of margin either side.\n\n"
            "Anchored at the feet rather than at the hands, and the radius follows "
            "from that. 'Within a club length of the hands' is also true and is a "
            "much weaker statement: it is a disc of about 2.5 torso lengths whose "
            "centre is chest-high, which on a vertical phone clip contains the sky. "
            "Larger is not free and the cost is not only work -- the region is what "
            "the search runs over, so it sets how much of the picture is allowed to "
            "compete with the ball for the identification, and a cloud between two "
            "branches competes very well."
        ),
    )
    ball_radius_torso: float = Field(
        default=0.047,
        gt=0.0,
        description=(
            "Expected apparent radius of the ball, in torso lengths. "
            "**Structural, and derived rather than tuned**: a golf ball is 42.7 mm "
            "across and an adult's shoulder-to-hip span about 450 mm, so the ball "
            "is 0.095 torso lengths across and 0.047 in radius. It is the centre "
            "of a tolerance band, not a threshold; see `radius_tolerance`."
        ),
    )
    radius_tolerance: float = Field(
        default=0.6,
        gt=0.0,
        lt=1.0,
        description=(
            "Fractional band either side of `ball_radius_torso` a candidate's "
            "radius may fall in. **Deliberately wide.** The ball sits nearer the "
            "camera than the torso on a face-on view and further on some "
            "down-the-line ones, so perspective alone moves its apparent size by "
            "tens of per cent, and a tee raises it further. The band exists to "
            "reject a shoe and a bag, not to measure a ball."
        ),
    )
    min_radius_px: float = Field(
        default=2.5,
        gt=0.0,
        description=(
            "Smallest radius that can be assessed for roundness at all. "
            "**A capture bound, not a policy.** Below about three pixels a disc "
            "has no shape left -- circularity of a five-pixel blob is dominated by "
            "which pixels the rasteriser happened to light -- so a clip framed so "
            "wide that the ball is this small is refused rather than guessed at."
        ),
    )
    min_circularity: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "How round a region must be, as 4*pi*area / perimeter^2, which is 1.0 "
            "for a disc. **A sanity filter, not the discriminator**, and the "
            "difference is a correction the reference footage forced.\n\n"
            "On the synthetic fixture a ball scores 0.89 and everything else falls "
            "below 0.4, which invites a bound around 0.65 and a belief that shape "
            "identifies a golf ball. It does not, at the size a golf ball actually "
            "occupies. Measured on `data/face-on/rory_face_on.mp4`, where the ball "
            "is 4.4 px in radius on a 576-wide phone clip: the ball scores **0.59** "
            "-- a compressed, slightly blocky disc has a ragged boundary, and "
            "circularity divides by the square of a perimeter that rasterisation "
            "and compression both inflate -- while grass texture and compression "
            "blocks in the same size band score anywhere from 0.11 to 0.71. At that "
            "resolution shape does not separate them at all, and a bound set from "
            "the fixture rejected the only real ball this project has.\n\n"
            "So the bound is set where it still rejects what is unambiguously not a "
            "ball -- a mat edge, a shaft, a turf line, all of which score under 0.3 "
            "-- and the identification is left to the thing that can actually make "
            "it: a position that holds a candidate for hundreds of consecutive "
            "frames and then stops. `BallQuality.radius_px` is what says whether "
            "shape had anything to work with."
        ),
    )
    max_drift_torso: float = Field(
        default=0.10,
        gt=0.0,
        description=(
            "How far an observation may sit from the established position and "
            "still be the ball, in torso lengths. **Structural**: a teed ball does "
            "not move at all, so this is a bound on the detector's own noise plus "
            "the wobble of a ball on a tee in wind, and it is about two ball "
            "diameters. It is also the scale `BallConfidence.stillness` decays "
            "against, so it sets how quickly that factor falls rather than only "
            "where it cuts off."
        ),
    )
    min_contrast: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        description=(
            "How far a candidate's interior must stand from the ring just outside "
            "it, as a fraction of the region's own intensity range. Measured: a "
            "ball on grass or a mat sits above 0.4 and falls through 0.15 as it "
            "goes into shadow, and a floor rather than a ranking quantity for the "
            "same reason `ClubConfig.min_support` is -- the brightest thing in a "
            "golf frame is frequently not the ball."
        ),
    )
    min_margin: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description=(
            "How far the winner must stand above the best rejected candidate "
            "before the choice counts as made. Below this the frame is refused as "
            "`AMBIGUOUS` rather than resolved by a coin toss the confidence would "
            "then describe as a measurement."
        ),
    )
    min_confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "Product of the three factors below which a frame emits nothing. Set "
            "below `ClubConfig.min_confidence` deliberately: a missed ball frame "
            "before the departure costs a little establishment, while a missed "
            "club frame costs a hole in a trajectory, so the asymmetry runs the "
            "other way here."
        ),
    )
    min_establishment_frames: int = Field(
        default=8,
        ge=2,
        description=(
            "How many agreeing frames are needed before a position is called the "
            "ball. The equivalent of `ClubConfig.min_track_frames` and set far "
            "higher for a reason that is about the object rather than the "
            "detector: a club is genuinely hard to see for most of a swing, and a "
            "teed ball is in plain view for every frame before impact. Needing "
            "only a few would accept a flicker, and a flicker followed by nothing "
            "is precisely what this phase reports as an impact."
        ),
    )
    min_establishment_coverage: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of the searched frames that must agree on the position. "
            "Counts as well as an absolute frame count because the two fail "
            "differently: a long clip reaches eight agreeing frames by accident, "
            "and a short one cannot reach them at all."
        ),
    )
    permanence_window_s: float = Field(
        default=0.30,
        gt=0.0,
        description=(
            "How long after a candidate departure the ball must stay away, in real "
            "seconds. **The check that separates a strike from an occlusion**, and "
            "the only one that can: nothing at the instant itself distinguishes a "
            "ball that left from a ball something moved in front of. Long enough "
            "to outlast a club head and a pair of hands crossing the region, short "
            "enough to be measurable on a clip that ends shortly after impact."
        ),
    )
    abruptness_frames: int = Field(
        default=3,
        ge=1,
        description=(
            "How many frames before a departure are scored for abruptness. Short, "
            "because the thing being looked for is a dimming that lasts a handful "
            "of frames; a long window averages it away against the hundreds of "
            "healthy frames behind it."
        ),
    )
    max_candidates: int = Field(
        default=32,
        ge=2,
        description=(
            "Regions kept from one frame before geometry is applied. A bound on "
            "work rather than a policy: a cluttered region can return hundreds, "
            "and the ones past this are the smallest, which the size filter would "
            "reject anyway."
        ),
    )


# Pydantic resolves the forward reference from `BallTrackingReport` to the config
# declared below it. Declared in this order because a reader meets the result
# first and the policy second, which is the order they matter in -- the same
# arrangement `contracts/club.py` and `contracts/reconstruction.py` use.
BallTrackingReport.model_rebuild()
