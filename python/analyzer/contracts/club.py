"""Typed contracts for club tracking.

The first thing in this engine that measures an **object the player is holding**
rather than the player. Everything up to here has read a body: landmarks a model
was trained to find, carrying the model's own confidences. A golf shaft has no
model, no landmark index and no reported visibility. What it has is edges, and
the whole of this phase is about what edges can and cannot be asked.

## What is measured, and what is inferred

A shaft in one frame is a **ray from the hands**: an origin the pose layer
already supplies, a direction the image determines well, and a length the image
frequently does not determine at all. Those three are not equally trustworthy and
this module keeps them apart by name.

That split is Phase 8's, one layer up and in a different medium. `apply.bearings`
returns unit vectors and there is deliberately no function there that turns one
into a 3D point, because a calibrated pixel is a *direction* and how far along it
anything sat is what the projection destroyed. Here a detected segment is a
*direction* and how far along it the club ends is what motion blur destroyed.
`ShaftObservation.reaches_head` is the flag that says whether the second question
was answered, and the club-head position is absent when it was not.

**Two different things shorten a segment and one view cannot separate them.**
The evidence stops early when the club head is smeared past recognition, and it
stops early when the club is pointing towards the camera and is genuinely short
in the picture -- the same foreshortening Phase 5 measures rotation from. Nothing
in one clip distinguishes those, so `reaches_head` is a statement about the
evidence and never about the club. Two calibrated views do separate them, which
is `analyzer.reconstruction`, and a shaft reconstructed from two tracked views is
the version of this measurement that would carry a length in metres.

## The direction is a direction, not an orientation

Phase 4 folds a shoulder line onto (-90, 90] because a shoulder line has an
orientation and no direction: nothing distinguishes its two ends, so an unfolded
angle flips by 180 degrees every time the line crosses level.

A shaft is the opposite case and must not be folded. Its two ends *are*
distinguishable, because one of them is the grip and the pose layer says where
the hands are. So `angle_deg` is a full direction on (-180, 180], and the
consequence is the trap one layer along: a swing carries the shaft through more
than a full turn, so an angle series has to be **unwrapped** before any rate
taken from it means anything. `analyzer.club.track` does that once, and
`angular_rate_deg_s` is the result.

## The three numbers, and the question each answers

The shape Phase 8 and Phase 9 both arrived at, because the failure is the same
shape again: the number that scores the fit is blind to the failure that matters,
and it is blind in a nameable direction.

    support        how much edge evidence backs this line        the fit
    margin         whether anything else in frame fits as well   a check
    phase coverage whether the frames that matter have any       the gate

**Support is not the quality, and through the downswing it is actively
misleading.** A probabilistic Hough transform scores a line by how many edge
pixels vote for it, so it prefers whatever in the picture is longest, straightest
and sharpest. Through the downswing the club is the fastest thing in the frame
and therefore the *blurriest*, while the door frame behind the player is
perfectly still and perfectly sharp. The background wins on support exactly where
the club matters most, and a detector confident in proportion to support would be
most confident when it was most wrong. `margin` is what notices: it is the best
candidate's score against the best *rejected* candidate's, so a shaft with a
convincing rival scores low however strong its own edges are.
`ShaftObservation.runner_up_angle_deg` carries what it was choosing between, so
the number can be audited rather than trusted.

**Coverage is the gate, and the aggregate rate is the thing not to report.**
Detection is easy at address and at the top, where the club is nearly still, and
hardest through the downswing, where it is moving fastest. So a clip-wide
detection rate is dominated by the frames nobody wants to measure. A clip tracked
in 80% of frames can contain none of the downswing at all, and that is not an
unusual case -- it is the ordinary one at 30 fps. `ClubTrackingReport` therefore
reports coverage **per phase** and the report's own verdict rests on the
downswing's, not on the mean.

## Refusal is the normal outcome, not the failure case

The exit criterion for this phase is that low confidence emits nothing, and on
consumer footage that applies to a large fraction of the frames that matter. The
per-frame result is therefore modelled as "an observation or a named reason there
is none", and a `ClubFrame` is kept for every frame either way -- the same
decision `PoseFrame` makes, for the same reason: a gap is information, and
dropping the frame would break the correspondence between frame index and
position in the sequence.

The reasons are separate rather than a single `detected=False`, because they have
completely different fixes. No edges is a lighting problem, no candidate is a
framing or a blur problem, ambiguity is a background problem, discontinuity means
the track was following something that is not a club, and low confidence means
all three were marginal at once.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.phases import SwingPhase
from analyzer.contracts.pose import FrameGeometry, LandmarkSpace

# Bump on any change that alters what a reported club track means.
CLUB_SCHEMA_VERSION = 1


class ShaftRefusal(StrEnum):
    """Why one frame carries no shaft.

    Counted separately per reason, because each names a different thing to
    change about the capture or the configuration. A single "not detected" tally
    would say that something went wrong and nothing about what.

    NO_GRIP
        The pose layer produced no hand position at this instant, so there was no
        anchor to search from. Nothing about the club: the wrists were occluded,
        gated out for low confidence, inside a refused gap, or outside the span
        the filter could support.
    NO_EDGES
        The region of interest held too few edge pixels to run a line search on.
        Low contrast between club and background, underexposure, or a shaft
        entirely outside the searched region.
    NO_CANDIDATE
        Line segments were found and none of them could be a shaft: none began
        near the hands, or none had a plausible length. This is what a blurred
        club looks like -- the smear has no edge straight enough to vote for a
        line.
    AMBIGUOUS
        Two or more candidates disagreed about where the club points and had
        comparable support. Almost always a straight background feature: a door
        frame, a fence post, a window mullion. `data/README.md` asks for an
        uncluttered background for exactly this reason.
    DISCONTINUOUS
        The best candidate would require the shaft to have rotated faster than a
        club can rotate. Whatever it is, it is not the same object the previous
        frames were following.
    LOW_CONFIDENCE
        A candidate survived every structural check and the product of its three
        factors still sat below the bound. The frames where support, margin and
        continuity are each individually tolerable and jointly not.
    ISOLATED
        A candidate was accepted and then discarded because too few frames around
        it agreed. Nothing was wrong with the frame itself, which is what makes
        this a different fact from `LOW_CONFIDENCE`: one line near the hands is
        not evidence that anything was *followed*, and a single frame cannot be
        told apart from a lucky background match by any amount of looking at it.
    """

    NO_GRIP = "no_grip"
    NO_EDGES = "no_edges"
    NO_CANDIDATE = "no_candidate"
    AMBIGUOUS = "ambiguous"
    DISCONTINUOUS = "discontinuous"
    LOW_CONFIDENCE = "low_confidence"
    ISOLATED = "isolated"


class ClubDetectorInfo(BaseModel):
    """Which detector produced a track, pinned so results stay attributable.

    `PoseModelInfo`'s counterpart, and it records less because there is less to
    record: a classical detector has no weights to digest and no training set to
    name. What it does have is a method and a library version, and both matter --
    OpenCV's Hough and Canny implementations are the measurement instrument here,
    and a result produced by a different one is a different result.

    Phase 12 may add a learned shaft detector. When it does, `method` is what
    distinguishes the two in a stored report, and the absence of a `sha256` here
    is itself informative: a classical detector is reproducible from its
    configuration, which the report already carries.
    """

    name: str = Field(description="Implementation name, e.g. 'hough_shaft'.")
    method: str = Field(
        description="The pipeline in words, e.g. 'roi -> canny -> hough_p -> geometry'."
    )
    opencv_version: str = Field(description="The version of OpenCV that ran, as measured.")


class ShaftConfidence(BaseModel):
    """Why one frame's shaft is or is not trustworthy, decomposed.

    Shaped like `EventConfidence` and for the same reason: `overall` is the
    **product** of the three, because a shaft needs all of them and a product
    says so where an average would let a strong factor cover for a fatal one. The
    factors are reported individually because "0.3" is not actionable and "there
    is something else in this frame that fits just as well" is.

    Note which of the three the image supplies. `support` comes from the edge
    evidence, and it is the one a line detector naturally reports. The other two
    come from what the frame was compared *against* -- everything else in it, and
    the frames either side -- and they are the two that catch the failures
    support cannot see. See the module docstring.
    """

    overall: float = Field(ge=0.0, le=1.0, description="Product of the three factors below.")
    support: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the fitted segment that sits on an edge pixel. How much "
            "of the line the image actually drew, as opposed to how much of it "
            "the transform was willing to extrapolate across gaps."
        ),
    )
    margin: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How far the accepted candidate stands above the best rejected one, "
            "as a fraction of its own score. 1.0 means nothing else in the frame "
            "was a plausible shaft; 0.0 means something else was exactly as "
            "plausible. **The factor that catches a background line**, which "
            "support does not and cannot."
        ),
    )
    continuity: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well this frame's direction agrees with the direction predicted "
            "from its tracked neighbours, scored against the angular rate a club "
            "can actually reach. 1.0 where the prediction lands on it; falls to "
            "zero at `ClubConfig.max_angular_rate_deg_s` of disagreement.\n\n"
            "1.0 where there is no tracked neighbour to predict from, which is "
            "**absence of evidence against** rather than agreement. "
            "`ClubTrackingReport.unanchored_frames` counts those, so the "
            "distinction is visible rather than folded into the score."
        ),
    )


class ShaftObservation(BaseModel):
    """The shaft measured in one frame: an origin, a direction, and maybe a length.

    Coordinates are **FRAME_WIDTHS** -- isotropic, y increasing upward, both axes
    divided by the frame width -- like every measured quantity above Phase 3.
    Detection itself runs in pixels, which is already isotropic; the conversion is
    a scale and a y flip applied once, in `analyzer.club.geometry`.

    `grip_x` and `grip_y` are **not measured here.** They are the hand position
    the pose layer produced, carried onto the observation so that a reader can
    see what the search was anchored to and an overlay can draw the segment
    without re-deriving it. A shaft that the detector found to be somewhere other
    than the hands would not have been accepted, so the grip is a constraint that
    was applied rather than a result that was obtained.
    """

    grip_x: float = Field(description="Hand anchor the search started from, in frame widths.")
    grip_y: float
    tip_x: float = Field(
        description=(
            "Far end of the **supported evidence**, in frame widths. The club head "
            "only when `reaches_head` is true; otherwise the point at which the "
            "image stopped drawing a line, which is nearer."
        )
    )
    tip_y: float
    angle_deg: float = Field(
        description=(
            "Direction from grip to tip, in degrees on (-180, 180]. Zero points "
            "along +x, ninety points up. A **direction, not an orientation** -- see "
            "the module docstring on why this is not folded the way a shoulder "
            "line is."
        )
    )
    length: float = Field(
        gt=0.0, description="Distance from grip to tip, in frame widths. Projected, not true."
    )
    length_torso: float = Field(
        gt=0.0,
        description=(
            "The same length in the subject's own torso lengths, which is the form "
            "that survives a change of camera distance. Compared against the "
            "clip's own longest observation to decide `reaches_head`."
        ),
    )
    reaches_head: bool = Field(
        description=(
            "Whether the evidence ran to the end of the club, making `tip` the "
            "club head. False means the segment stopped early, which happens for "
            "two reasons one view cannot separate: the head was smeared by motion "
            "blur, or the club was pointing at the camera. See the module "
            "docstring."
        )
    )
    confidence: ShaftConfidence
    runner_up_angle_deg: float | None = Field(
        default=None,
        description=(
            "Direction of the best candidate that was **rejected**, where there "
            "was one. The evidence behind `confidence.margin`: a reader who wants "
            "to know whether the tracker was choosing between a club and a door "
            "frame can see the other option rather than infer it from a score."
        ),
    )
    candidates: int = Field(
        ge=1, description="Line segments that satisfied the geometry at this frame."
    )
    angular_rate_deg_s: float | None = Field(
        default=None,
        description=(
            "Rate of change of `angle_deg` against its tracked neighbours, from "
            "the **unwrapped** angle series -- a swing carries the shaft through "
            "more than a full turn, and differencing a wrapped angle would report "
            "a 360-degree jump somewhere in the middle of every clip. None at an "
            "observation with no tracked neighbour."
        ),
    )
    tip_speed_px_s: float | None = Field(
        default=None,
        description=(
            "How fast the tip moved across the image, in pixels per second, "
            "against its tracked neighbours. **The quantity that decides whether "
            "this frame could have been sharp**: multiplied by the exposure time "
            "it is the length of the smear. Nothing in a video file records the "
            "exposure time, so the smear cannot be computed -- but the exposure "
            "cannot exceed the frame interval, so this times the interval is an "
            "upper bound on it, and a faster shutter reduces it proportionally."
        ),
    )


class ClubFrame(BaseModel):
    """What was concluded about the club in one video frame.

    A frame with no shaft is kept rather than dropped, which is `PoseFrame`'s
    decision and made for the same two reasons: the absence is itself information
    that a later layer needs in order to know it is looking at a gap, and
    dropping the frame would break the correspondence between frame index and
    position in the sequence.

    Exactly one of `shaft` and `refusal` is set. A consumer that checks only for
    `shaft` therefore cannot mistake a refused frame for a tracked one at the
    origin.
    """

    frame_index: int
    timestamp_s: float = Field(
        description=(
            "Elapsed real seconds from the first frame, already divided by any "
            "slow-motion factor, as everywhere above Phase 3."
        )
    )
    detected: bool
    shaft: ShaftObservation | None = Field(
        default=None, description="The measurement, when there is one."
    )
    refusal: ShaftRefusal | None = Field(
        default=None, description="Why there is none. Set exactly when `detected` is false."
    )
    phase: SwingPhase | None = Field(
        default=None,
        description=(
            "Which swing phase this frame falls in, where a swing was detected. "
            "Carried per frame because the coverage that gates this phase is "
            "per phase, and because an overlay reads it."
        ),
    )


class PhaseCoverage(BaseModel):
    """How much of one swing phase carries a tracked shaft.

    **The gate.** See the module docstring: detection is easiest where the club
    is slowest, so a clip-wide rate is an average over the frames that matter
    least. A downswing coverage of zero on a clip tracked in four frames out of
    five is the ordinary result at consumer frame rates, and it is the result
    that decides whether anything here can be used.
    """

    phase: SwingPhase
    frames: int
    tracked: int
    coverage: float = Field(ge=0.0, le=1.0)
    median_confidence: float | None = Field(
        default=None, description="Median overall confidence over the tracked frames of this phase."
    )
    median_tip_speed_px_s: float | None = Field(
        default=None,
        description=(
            "Median image speed of the tip through this phase. The number that "
            "explains the coverage: it is an order of magnitude larger through "
            "the downswing than anywhere else."
        ),
    )


class ClubImpactEstimate(BaseModel):
    """Impact located from the club rather than from the hands.

    Phase 4 estimates impact as the peak of hand speed and records, in its own
    contract, that this is a kinematic estimate rather than an observation:
    nothing there sees the ball or the club, and the hands reach their peak
    slightly *before* the club reaches the ball. `docs/architecture.md` promised
    that Phases 10 and 11 would replace that corroboration with real evidence.

    This is half of that. It is the lowest point the **club head** reaches after
    the top, which is a different signal from the lowest point the hands reach,
    and it is emitted only from frames where the head was actually observed --
    which is the hard requirement, because those are the blurriest frames in the
    clip.

    **It does not move Phase 4's answer**, and that is deliberate. Reporting one
    instant from two methods in two places with two provenances is how a reader
    ends up with two numbers and no way to tell which they are reading; the same
    reasoning kept tempo out of Phase 4 and "transition to impact" out of Phase 5.
    Fusing the two estimates is Phase 11's job, once the ball supplies the third
    piece of evidence that can arbitrate between them.
    """

    frame_index: int
    timestamp_s: float
    methodology: str = Field(description="The rule that produced this frame, in words.")
    head_frames_searched: int = Field(
        description=(
            "Frames after the top in which the club head was observed at all. The "
            "support behind this estimate: a handful of frames either side of a "
            "minimum locates it much less well than a dense arc does."
        )
    )
    kinematic_frame: int | None = Field(
        default=None, description="Where Phase 4 put impact, for comparison. None if it found none."
    )
    delta_s: float | None = Field(
        default=None,
        description=(
            "Seconds from the kinematic estimate to this one. Positive means the "
            "club-head estimate is later, which is the direction the physics "
            "predicts -- the hands peak before the head arrives."
        ),
    )


class ClubTrackingQuality(BaseModel):
    """What the track is worth, in the senses that differ.

    Shaped like `ReconstructionQuality` and `CalibrationQuality`, because the
    lesson turned out to be the same one a third time: report the fit, the check
    and the capture separately, and label each with the question it answers.
    """

    median_support: float | None = None
    median_margin: float | None = Field(
        default=None,
        description="Median margin over the best rejected candidate. The check support cannot make.",
    )
    median_continuity: float | None = None
    median_confidence: float | None = None

    median_angular_rate_deg_s: float | None = Field(
        default=None,
        description="Median |rate of change of the shaft direction| over tracked frames.",
    )
    max_angular_rate_deg_s: float | None = Field(
        default=None,
        description=(
            "The fastest the shaft was seen to turn. Compare against "
            "`ClubConfig.max_angular_rate_deg_s`: a track that repeatedly "
            "approaches the bound is a track being held together by it."
        ),
    )

    median_tip_speed_px_s: float | None = None
    max_tip_speed_px_s: float | None = Field(
        default=None,
        description=(
            "The fastest the tip crossed the image. **The capture number.** "
            "Multiplied by the exposure time it is the length of the smear on the "
            "club head, and the exposure cannot exceed the frame interval -- so "
            "`max_blur_px` below is what this clip's shutter would have to beat."
        ),
    )
    max_blur_px: float | None = Field(
        default=None,
        description=(
            "`max_tip_speed_px_s` times the median frame interval: the smear the "
            "club head would carry at a 360-degree shutter, which is the worst any "
            "camera does. A shutter n times faster than the frame interval divides "
            "it by n. Reported because nothing in a video file records the "
            "exposure, so this is the most that can be said about blur from the "
            "file alone -- and it is enough to explain a refused downswing."
        ),
    )

    median_length_torso: float | None = None
    max_length_torso: float | None = Field(
        default=None,
        description=(
            "The longest shaft seen, in torso lengths. The clip's own estimate of "
            "how long the club is in the picture, and the scale `reaches_head` is "
            "judged against. Not the club's length: it is the longest *projection* "
            "of it, which is the true length only when the shaft happened to lie "
            "square to the camera."
        ),
    )
    head_frames: int = Field(
        default=0, description="Frames whose evidence ran to the end of the club."
    )
    head_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="`head_frames` over tracked frames. How often a club-head position exists at all.",
    )

    methodology: str


class ClubTrackingReport(BaseModel):
    """Everything the engine concluded about the club in one clip.

    **The per-frame observations are here**, which is a departure from
    `PoseExtractionResult` and `ReconstructionReport`, both of which deliberately
    leave their geometry out of the response and hand back a path instead. Two
    things make this the different case. A club track is *one* object per frame
    rather than thirty-three landmarks in two spaces, so a swing's worth of it is
    kilobytes where a pose sequence is megabytes. And a detector is checked by
    drawing it on the frames it came from -- that is what `scripts/overlay_club.py`
    exists for -- so a report that omitted the geometry would force every consumer
    to re-run the detection in order to see it.

    `tracked` is false whenever no frame produced a shaft, and then `quality` is
    None rather than zeroed. Zero coverage and "nothing was attempted" are
    different facts, and a consumer reading a zero as a measurement would be
    reading a refusal as a result.
    """

    schema_version: int = CLUB_SCHEMA_VERSION
    tracked: bool
    video_path: str
    detector: ClubDetectorInfo
    space: LandmarkSpace = Field(
        default=LandmarkSpace.FRAME_WIDTHS,
        description=(
            "The frame the observations are in: isotropic, y up, both axes in "
            "frame widths. **Not metric** -- a club tracked in one view is a line "
            "in a picture, and its length in metres needs two calibrated views."
        ),
    )
    geometry: FrameGeometry = Field(
        description=(
            "Displayed pixel dimensions, carried so the report is self-contained "
            "for drawing: an overlay converts frame widths to pixels with it and "
            "does not have to re-probe a file that may have moved."
        )
    )
    torso_length: float = Field(
        description=(
            "The subject's shoulder-to-hip span in frame widths, as Phase 4 "
            "measures it. The scale every length here was judged against."
        )
    )
    slow_motion_factor: float = Field(
        default=1.0,
        description="The clip's factor. Timestamps here are real seconds, as everywhere above Phase 3.",
    )

    frame_count: int
    tracked_frames: int
    coverage: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Tracked frames over all frames. **The number not to read on its own.** "
            "It is dominated by address and the follow-through, where the club is "
            "nearly still and detection is easy, and says almost nothing about the "
            "downswing. `phase_coverage` is what the verdict rests on."
        ),
    )
    unanchored_frames: int = Field(
        default=0,
        description=(
            "Tracked frames that had no tracked neighbour to predict them, so "
            "their continuity factor is 1.0 by absence of evidence rather than by "
            "agreement. See `ShaftConfidence.continuity`."
        ),
    )

    phase_coverage: list[PhaseCoverage] = Field(
        default_factory=list,
        description="Per swing phase. Empty when no swing was detected in the clip.",
    )
    quality: ClubTrackingQuality | None = None
    impact: ClubImpactEstimate | None = Field(
        default=None,
        description=(
            "Impact located from the club head, where enough of it was observed "
            "after the top. Independent of Phase 4's kinematic estimate and "
            "deliberately does not replace it."
        ),
    )
    frames: list[ClubFrame] = Field(
        default_factory=list, description="One entry per video frame, tracked or refused."
    )
    refusals: dict[ShaftRefusal, int] = Field(
        default_factory=dict, description="Frames carrying no shaft, counted by reason."
    )
    tracked_at: datetime | None = None
    elapsed_s: float = 0.0
    ms_per_frame: float = 0.0

    config: ClubConfig
    refusal: str | None = Field(
        default=None,
        description="Why nothing was tracked, in words. Set exactly when `tracked` is false.",
    )
    warnings: list[str] = Field(default_factory=list)

    def frame(self, frame_index: int) -> ClubFrame | None:
        """The entry for one frame, or None if the clip does not contain it."""
        if 0 <= frame_index < len(self.frames):
            entry = self.frames[frame_index]
            if entry.frame_index == frame_index:
                return entry
        return next((item for item in self.frames if item.frame_index == frame_index), None)

    def coverage_of(self, phase: SwingPhase) -> PhaseCoverage | None:
        """Coverage of one named phase, or None if the clip has no such phase."""
        return next((entry for entry in self.phase_coverage if entry.phase is phase), None)


class ClubConfig(BaseModel, extra="forbid"):
    """Policy for club tracking.

    Two kinds of number live here and the field documentation says which is
    which, as `PhaseConfig` does. **Structural bounds** describe what a golf club
    physically is or is not -- how long, how fast it can turn -- and are set loose
    enough that no real swing approaches them; they exist to reject a background
    line, not to describe a club. **Measured defaults** come from
    `scripts/benchmark_club.py`, which renders a shaft at known angles under
    known blur and clutter and sweeps each threshold against detection rate and
    angular error.

    Nothing here is tuned against hand-labelled shaft positions on real footage,
    because none exist. That is Phase 12's labelling tool, and until it exists the
    honest statement is that the synthetic sweep establishes the *shape* of each
    relationship and where on the curve to refuse is a judgement about what a
    wrong answer costs.
    """

    roi_radius_torso: float = Field(
        default=3.2,
        gt=0.0,
        description=(
            "Radius of the searched region around the hands, in torso lengths. "
            "**Structural.** A driver shaft is about 1.15 m and the "
            "shoulder-to-hip span of an adult is about 0.45 m, so a club is "
            "roughly 2.5 torso lengths and this leaves most of a torso of margin. "
            "Larger is not free: the region is what the line search runs over, so "
            "it sets both the cost and how much background is allowed to compete."
        ),
    )
    canny_sigma: float = Field(
        default=0.33,
        gt=0.0,
        lt=1.0,
        description=(
            "Half-width of the Canny hysteresis band, as a fraction either side of "
            "the region's median intensity. Thresholds are derived from the region "
            "rather than fixed, because a fixed pair encodes one exposure and one "
            "background: the same numbers that find a black shaft on grass find "
            "nothing at all against a bright sky."
        ),
    )
    hough_threshold: int = Field(
        default=18,
        ge=1,
        description=(
            "Votes a line needs in the probabilistic Hough accumulator. Measured: "
            "below about 12 the background supplies candidates faster than the "
            "margin check can reject them, and above about 30 a blurred shaft "
            "stops voting at all."
        ),
    )
    max_line_gap_torso: float = Field(
        default=0.06,
        ge=0.0,
        description=(
            "How much of a line may be missing before the transform treats it as "
            "two lines, in torso lengths. **Deliberately small.** A generous gap "
            "is the obvious way to bridge the break a motion-blurred club head "
            "leaves in its own edges, and it is the wrong fix: it also bridges the "
            "gap between a shaft and a fence post behind it, merging two objects "
            "into one long confident line. The break is real information and the "
            "support score is where it belongs."
        ),
    )
    min_shaft_length_torso: float = Field(
        default=0.8,
        gt=0.0,
        description=(
            "Shortest segment that may be a shaft, in torso lengths. "
            "**Structural**, and deliberately well below a club: a shaft seen "
            "nearly end-on is genuinely short in the picture, and refusing those "
            "would discard the frames at the top of the backswing on every "
            "down-the-line clip."
        ),
    )
    max_shaft_length_torso: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "Longest segment that may be a shaft, in torso lengths. "
            "**Structural.** Past this it is a wall, a fence or a horizon."
        ),
    )
    max_grip_distance_torso: float = Field(
        default=0.45,
        gt=0.0,
        description=(
            "How far a candidate's near end may sit from the hand anchor, in torso "
            "lengths. **The constraint that does most of the work**: a club is "
            "held, so the one thing known for certain about the shaft is that it "
            "starts at the hands, and almost every straight background feature "
            "fails this before anything else is asked of it. About a hand's width "
            "plus the pose layer's own error on a wrist."
        ),
    )
    max_angular_rate_deg_s: float = Field(
        default=4000.0,
        gt=0.0,
        description=(
            "Fastest the shaft may turn between two tracked frames, in degrees per "
            "second. **Structural, and derived rather than tuned**: a tour driver "
            "head reaches about 50 m/s while the grip travels about 8, so the "
            "shaft's tip moves about 42 m/s about a 1.1 m lever -- some 38 rad/s, "
            "or 2,200 deg/s. Double that is past anything a human produces and "
            "still rejects the jump to a background line, which is typically a "
            "quarter-turn in one frame.\n\n"
            "It is also the scale `ShaftConfidence.continuity` is measured "
            "against, so it sets how quickly that factor decays rather than only "
            "where it cuts off."
        ),
    )
    max_prediction_window_s: float = Field(
        default=0.10,
        gt=0.0,
        description=(
            "How far in time a tracked neighbour may be and still predict this "
            "frame's direction. Past it the prediction is extrapolating across a "
            "gap long enough to contain most of a downswing, and a search narrowed "
            "by that prediction would be narrowed towards a guess."
        ),
    )
    min_support: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the segment that must sit on an edge pixel. Measured: the "
            "true shaft stays above 0.8 while it is sharp and falls through 0.5 as "
            "the smear sets in, and background lines sit near 1.0 -- which is why "
            "this is a floor on plausibility and **not** the quantity ranked on."
        ),
    )
    min_margin: float = Field(
        default=0.15,
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
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Product of the three factors below which the frame emits nothing. "
            "**This is the phase's exit criterion in one number.** It is a stated "
            "policy: 0.30 is roughly 'two factors good and the third middling', "
            "and the cost of setting it wrong is asymmetric -- a refused frame "
            "leaves a visible gap in a track, and an accepted wrong frame puts a "
            "confident shaft angle on a door frame."
        ),
    )
    min_head_length_fraction: float = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of the clip's longest observed shaft that this frame's must "
            "reach before its far end is called the club head. Measured against "
            "the clip's own maximum rather than an assumed club length, because "
            "the projection of a club varies with where it points and no clip "
            "records what club was used."
        ),
    )
    min_track_frames: int = Field(
        default=3,
        ge=1,
        description=(
            "Shortest run of consecutive tracked frames that counts as a track. "
            "Isolated single frames are discarded even when they pass every other "
            "check: one line that happens to sit near the hands is not evidence "
            "that anything was followed, and the tracker has no way to tell such a "
            "frame from a lucky background match."
        ),
    )
    max_candidates: int = Field(
        default=64,
        ge=2,
        description=(
            "Line segments kept from one frame's Hough transform before geometry "
            "is applied. A bound on work rather than a policy: a cluttered frame "
            "can return hundreds, and the ones past this are the shortest, which "
            "the length filter would reject anyway."
        ),
    )


# Pydantic resolves the forward reference from `ClubTrackingReport` to the config
# declared below it. Declared in this order because a reader meets the result
# first and the policy second, which is the order they matter in -- the same
# arrangement `contracts/reconstruction.py` uses.
ClubTrackingReport.model_rebuild()
