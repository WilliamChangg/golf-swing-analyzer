"""Typed contracts for the biomechanics engine.

A `Metric` is the unit of output of this system. Everything a coach, a report or
a later language model ever sees about a swing arrives as one of these, and the
shape of the model is chosen so that a number cannot travel without the things
that make it interpretable:

* **a unit**, named rather than implied;
* **a basis**, saying what *kind* of claim the number is -- and in particular
  whether it is a measurement of the world or a measurement of a projection of
  the world;
* **an anchor**, the swing event or phase it belongs to;
* **the frames it came from**, so it can be checked against the video;
* **a decomposed confidence**; and
* **a methodology**, in words, that is never omitted.

## Why `basis` exists

Everything measured in Phase 5 comes from a single uncalibrated camera. That
camera flattens three dimensions into two, and no amount of care in the
arithmetic afterwards puts the third one back. A shoulder line that appears to
turn 80 degrees may have turned 80 degrees, or it may have turned 90 and been
seen slightly off-axis; nothing here can tell those apart, and a system that
reported "80 degrees of shoulder turn" without qualification would be claiming
it could.

So the qualification is a typed field rather than a sentence in a document.
`TEMPORAL` metrics -- durations and their ratios -- are the only ones in this
phase unaffected by the missing dimension, because a stopwatch does not care
where the camera stood. Everything else is explicitly a statement about the
image plane. Phases 8 and 9 add calibration and triangulation, and only then can
a metric claim to be about the body rather than about its picture.

## What `confidence` is, and is not

The three factors answer **how well this method determined this quantity as
defined**, not how close the quantity is to an anatomical truth. That second
question is what `basis` is for, and keeping it out of the number is deliberate:
a single scalar that mixed "the landmark was blurry" with "a camera cannot see
rotation" would be uninterpretable, and the fix for those two is not the same.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.calibration import CalibrationStatus
from analyzer.contracts.phases import SwingEvent, SwingPhase
from analyzer.contracts.pose import FrameGeometry

# Bump on any change that alters what a reported metric means.
METRICS_SCHEMA_VERSION = 1


class MetricUnit(StrEnum):
    """What a metric's value is in.

    `TORSO_LENGTHS` is the length unit throughout, rather than frame widths or
    pixels. A distance in frame widths halves when the camera is moved twice as
    far away; the same distance in the subject's own torso lengths does not. It
    is not a metric unit and does not pretend to be one -- it is a ratio of two
    measured image distances, which is exactly what survives an unknown camera.
    """

    DEGREES = "degrees"
    SECONDS = "seconds"
    RATIO = "ratio"
    TORSO_LENGTHS = "torso_lengths"
    TORSO_LENGTHS_PER_S = "torso_lengths_per_s"
    METRES_PER_S = "metres_per_s"
    """Genuinely metres, and the only unit here that is a physical length.

    It exists only for metrics whose `basis` is `SPATIAL`, which means they were
    triangulated from two calibrated views; the scale descends from a printed
    board square measured with a ruler. `TORSO_LENGTHS_PER_S` sitting beside it
    is the same quantity from one camera, and the two must never be compared:
    one is a ratio that survives an unknown camera and the other is a speed."""


class MetricGroup(StrEnum):
    """Which family a metric belongs to. Presentation, not semantics."""

    POSTURE = "posture"
    ROTATION = "rotation"
    ARMS = "arms"
    TIMING = "timing"


class MetricBasis(StrEnum):
    """What kind of claim a metric's value is.

    TEMPORAL
        A duration, or a ratio of durations. Depends on the clock and nothing
        else, so it is unaffected by where the camera stood. The only basis in
        this phase that measures the body rather than a picture of it.

    IMAGE_PLANE
        A distance or a speed between two points as they appear in the frame,
        divided by the subject's torso length. Exact as a statement about the
        image; it under-reports any motion that ran towards or away from the
        camera, which it cannot see at all.

    PROJECTED_ANGLE
        The angle between two segments as they appear in the frame. Equal to the
        real joint angle only when both segments lie in the image plane, and
        smaller than it otherwise. Never a 3D joint angle.

    FORESHORTENED_ANGLE
        A rotation about the vertical inferred from how much a body segment
        shortened in the image: a shoulder line seen at 60% of its full width
        has turned about 53 degrees away from the camera. It assumes the segment
        was square to the camera at its reference frame, and it is blind to
        direction -- a turn and its mirror image shorten identically -- so the
        value is a magnitude.

    SPATIAL
        A length, angle or speed between points **reconstructed in three
        dimensions** from two calibrated views of the same instant. The only
        basis here that is a statement about the body rather than about a
        picture of it, and the reason every other one is named the way it is.

        It is also the only basis whose meaning does not depend on the camera
        view: a distance between two 3D points is the same distance from
        anywhere, so a `SPATIAL` metric carries one anatomical reading instead
        of one per view. What it does depend on is the *reconstruction*, whose
        own error is reported per point and propagated into the metric's
        `uncertainty` -- a 3D number is not automatically a better number, it is
        a differently-conditioned one, and the conditioning is the ray
        convergence angle rather than the camera position.
    """

    TEMPORAL = "temporal"
    IMAGE_PLANE = "image_plane"
    PROJECTED_ANGLE = "projected_angle"
    FORESHORTENED_ANGLE = "foreshortened_angle"
    SPATIAL = "spatial"


class MetricName(StrEnum):
    """Every quantity this engine knows how to measure.

    Named per *quantity*, not per quantity-and-instant: `SPINE_TILT` at the top
    and at impact are the same measurement made twice, and they carry the same
    unit, basis and methodology. Which instant a particular value belongs to is
    on the metric's `event` field, so the set of names does not grow by
    multiplication every time another anchor becomes interesting.
    """

    # --- posture ---
    SPINE_TILT = "spine_tilt"
    LEFT_KNEE_FLEX = "left_knee_flex"
    RIGHT_KNEE_FLEX = "right_knee_flex"
    HIP_SWAY = "hip_sway"
    HEAD_SWAY = "head_sway"
    HEAD_LIFT = "head_lift"

    # --- rotation ---
    SHOULDER_TURN = "shoulder_turn"
    PELVIS_TURN = "pelvis_turn"
    X_FACTOR = "x_factor"
    SHOULDER_TILT = "shoulder_tilt"
    PELVIS_TILT = "pelvis_tilt"

    # --- hands and arms ---
    HAND_DEPTH = "hand_depth"
    HAND_PATH_LENGTH = "hand_path_length"
    PEAK_HAND_SPEED = "peak_hand_speed"
    LEAD_ARM_ANGLE = "lead_arm_angle"
    TRAIL_ARM_ANGLE = "trail_arm_angle"

    # --- reconstructed in three dimensions (needs a stereo calibration) ---
    #
    # Named apart from their projected namesakes rather than replacing them,
    # because they are different claims about different things: `SHOULDER_TURN`
    # is a rotation inferred from how much a line shortened in one picture, and
    # `SHOULDER_TURN_3D` is the angle between two measured directions in space.
    # They will disagree, the disagreement is informative, and a single name
    # whose meaning depended on whether a rig happened to be calibrated would
    # make two numbers that cannot be compared look like one that can.
    SHOULDER_TURN_3D = "shoulder_turn_3d"
    PELVIS_TURN_3D = "pelvis_turn_3d"
    X_FACTOR_3D = "x_factor_3d"
    LEAD_ARM_ANGLE_3D = "lead_arm_angle_3d"
    TRAIL_ARM_ANGLE_3D = "trail_arm_angle_3d"
    PEAK_HAND_SPEED_3D = "peak_hand_speed_3d"

    # --- timing ---
    BACKSWING_DURATION = "backswing_duration"
    DOWNSWING_DURATION = "downswing_duration"
    FOLLOW_THROUGH_DURATION = "follow_through_duration"
    TAKEAWAY_TO_IMPACT = "takeaway_to_impact"
    TEMPO_RATIO = "tempo_ratio"


class BodySide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class CameraView(StrEnum):
    """Where the camera stood relative to the player.

    The single fact that decides what a projected measurement *means*. The same
    spine tilt is lateral side bend seen face-on and forward posture angle seen
    down the line; the same hip displacement is a slide towards the target in one
    and a move towards the ball in the other. Nothing in the arithmetic
    distinguishes them, so the view is measured and carried on every metric, and
    the anatomical reading is stated per view rather than assumed.

    FACE_ON
        Perpendicular to the target line, looking at the player. The shoulder
        line lies across the frame, so rotation is measurable by foreshortening
        and left and right are distinguishable.
    DOWN_THE_LINE
        Along the target line. The shoulder line points towards the camera and
        collapses, so rotation about the spine is not recoverable, and the
        frame's horizontal axis runs towards and away from the ball.

        **Which end of the target line is not determined.** A camera behind the
        player and one in front of them foreshorten the shoulder line
        identically, and nothing else here separates them. The measurable
        consequences are the same either way, which is why one label covers
        both; what it costs is the sign of anything measured along the frame's
        horizontal axis, so those quantities are reported as image directions
        rather than as "towards the player" or "away from them". The reference
        clip `data/amateur/dtl/iron_dtl.mp4` is filmed from in front, despite its name.
    UNKNOWN
        Oblique, or too little of the body tracked at address to tell. Not a
        failure: an oblique camera genuinely supports some measurements and not
        others, and saying so beats picking the nearer label.
    """

    FACE_ON = "face_on"
    DOWN_THE_LINE = "down_the_line"
    UNKNOWN = "unknown"


class ViewEstimate(BaseModel):
    """Which view a clip was shot from, and the measurements behind the verdict.

    Decided from the **projected width of the shoulder line at address**, in
    torso lengths. Seen face-on the shoulders lie broadside and span most of a
    torso length or more; seen down the line they point at the camera and
    collapse to almost nothing. On the reference clips the two are 0.83 and 0.10,
    which is a margin of eight times rather than a close call.

    `openness` corroborates it from an independent direction, the way Phase 4
    corroborates impact: the address span divided by the widest the line was ever
    seen in the clip is the cosine of how far off broadside it was at address. A
    swing turns the shoulders through about a right angle, so both views contain
    a frame where the line is nearly square, and the two signals should agree.
    They are kept separate because the second depends on the clip containing that
    frame and the first does not.
    """

    view: CameraView
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How far clear of ambiguity the measurement sits, as a fraction of "
            "the band between the two thresholds. 1.0 is a full band clear."
        ),
    )
    shoulder_span_ratio: float = Field(
        description="Projected shoulder width at address, in torso lengths. The verdict."
    )
    hip_span_ratio: float = Field(
        description="Projected hip width at address, in torso lengths. Reported, not used."
    )
    openness: float = Field(
        description=(
            "Address shoulder span divided by the widest in the clip: the cosine "
            "of how far off broadside the shoulders were at address. Corroboration."
        )
    )
    frames: list[int] = Field(
        default_factory=list, description="The address frames the spans were measured over."
    )
    methodology: str


class MetricConfidence(BaseModel):
    """Why a metric is or is not trustworthy, decomposed.

    Three factors and their product, matching how Phase 4 reports an event's
    confidence -- for the same reason. A single number says a metric is worth
    0.3 without saying whether the fix is a better camera angle, a brighter
    room, or a faster shutter.
    """

    overall: float = Field(ge=0.0, le=1.0, description="Product of the three factors below.")
    observation: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Mean reported visibility of the landmarks this metric used, over the "
            "frames it used them on."
        ),
    )
    anchor: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence of the swing event or phase this metric is measured at, "
            "propagated from phase detection. A perfectly measured angle at an "
            "instant that is not really the top of the backswing is not a "
            "measurement of anything."
        ),
    )
    method: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well the method itself pins the quantity down, measured rather "
            "than assumed. For a duration, how finely the frame rate divides it. "
            "For a projected angle, the fraction of the segment lying in the image "
            "plane -- a segment seen nearly end-on subtends a noisy angle. For a "
            "foreshortened rotation, the sine of the angle found, which is how "
            "sharply the arccos converts a length into an angle there."
        ),
    )


class Metric(BaseModel):
    """One measured quantity, with everything needed to interpret it."""

    name: MetricName
    group: MetricGroup
    label: str = Field(description="Human-readable name, including the instant it was measured at.")
    value: float
    unit: MetricUnit
    basis: MetricBasis
    event: SwingEvent | None = Field(
        default=None, description="The instant this was measured at, for metrics taken at one."
    )
    phase: SwingPhase | None = Field(
        default=None, description="The interval this was measured over, for metrics spanning one."
    )
    source_frames: list[int] = Field(
        description=(
            "Every frame whose landmarks entered this value, so it can be checked "
            "against the video rather than taken on trust."
        )
    )
    view: CameraView = Field(
        description=(
            "The camera view this was measured in. Carried on every metric "
            "because a projected quantity means different things from different "
            "places, and two clips of the same swing must never be compared "
            "across views as though the numbers described the same thing."
        )
    )
    interpretation: str = Field(
        description=(
            "What the number means anatomically from this view, in words. "
            "Separate from `methodology`, which says how it was computed: the "
            "arithmetic is the same from every camera position and the meaning "
            "is not."
        )
    )
    uncertainty: float | None = Field(
        default=None,
        description=(
            "Measured uncertainty in the value, in the metric's own unit, where "
            "the method admits one. None means it was not quantified -- which is "
            "not a claim that the value is exact.\n\n"
            "Reported separately rather than folded into `confidence` on purpose. "
            "The three confidence factors are dimensionless and multiply; an "
            "uncertainty has a unit and a meaning, and 'shoulder turn 53 degrees, "
            "give or take 6' tells a reader something no score between 0 and 1 "
            "can. Mapping it into one would need a scale nobody has measured."
        ),
    )
    confidence: MetricConfidence
    methodology: str = Field(description="How this number was produced, in words. Never omitted.")


class RefusedMetric(BaseModel):
    """A metric that was asked for and deliberately not produced.

    Reported rather than silently omitted. An absent metric and a metric that
    could not honestly be computed look identical in a list of results, and they
    call for different responses: one is a gap in the engine, the other is a fact
    about the recording that the person holding the camera can fix.
    """

    name: MetricName
    event: SwingEvent | None = None
    reason: str


class LeadSide(BaseModel):
    """Which side of the body leads the swing, and how that was decided.

    Needed because "lead arm" and "trail arm" are not "left arm" and "right
    arm": they swap between a right- and a left-handed player, and nothing in a
    pose sequence declares which is which.

    It is inferred from a measurement rather than configured: at the top of the
    backswing the hands sit over the **trail** shoulder, for either handedness
    and from either side of the player. Projecting the hand position onto the
    shoulder line therefore names the trail side directly, and the size of that
    projection says how confidently.
    """

    side: BodySide | None = Field(
        description="The leading side, or None when the evidence was too weak to name one."
    )
    margin: float = Field(
        description=(
            "How far the hands sat towards one shoulder at the top, as a fraction "
            "of half the shoulder span. 1.0 means directly over a shoulder, 0.0 "
            "means over the middle of the chest and therefore undecidable."
        )
    )
    shoulder_span_ratio: float = Field(
        description=(
            "Projected shoulder span at the top, in torso lengths. A down-the-line "
            "camera sees the shoulders nearly end-on, which makes this small and "
            "the side undecidable -- correctly, because that view genuinely does "
            "not show it."
        )
    )
    methodology: str


class RotationReference(BaseModel):
    """The baseline a foreshortening rotation is measured against, and its check.

    A rotation here is a comparison of a body line's projected length against its
    length when square to the camera. That baseline is taken **at address**,
    because that is where the method's premise says the player is square: they
    are set up to the ball with the camera in front of them.

    Taking it instead as the widest view anywhere in the clip was tried and is
    worse, for a reason the reference footage demonstrated. A maximum over a
    whole clip is a maximum over that clip's landmark noise too, so it selects
    the single frame where the estimator most overstated the span -- on the
    face-on reference clip, a frame just past impact where the shoulders read 12%
    wider than the player can physically be. Every rotation in the clip is then
    measured against an error.

    The premise is still checked, by the fields below: if some frame mid-swing
    projects substantially wider than address did, the player was not square
    there, and this is not a face-on recording.
    """

    landmarks: str = Field(description="Which segment: 'shoulders' or 'hips'.")
    span: float = Field(
        description="Median projected length over the address phase, in torso lengths."
    )
    frames: list[int] = Field(description="The address frames the baseline was taken over.")
    widest_span: float = Field(
        description="Largest projected length anywhere in the clip, in torso lengths."
    )
    widest_frame: int
    excess: float = Field(
        description=(
            "widest_span / span. At most slightly above 1 on a face-on recording, "
            "where address is the squarest the line ever is. Far above it when the "
            "line is seen end-on at address and opens up through the swing, which "
            "is what a down-the-line camera sees."
        )
    )
    square_at_address: bool = Field(
        description="Whether `excess` is within the configured tolerance."
    )


class MetricConfig(BaseModel, extra="forbid"):
    """Policy for the biomechanics engine.

    As in Phase 4, these are **structural bounds, not golf norms.** Nothing here
    states what a good swing looks like. They exist to decide when a measurement
    is too ill-conditioned to report at all, and the thresholds are stated policy
    rather than measured optima, because no labelled set exists until Phase 12.
    """

    min_shoulder_span_ratio: float = Field(
        default=0.35,
        gt=0.0,
        description=(
            "Projected shoulder span at the top, in torso lengths, below which the "
            "shoulder line is too foreshortened to read a direction from -- which "
            "is what naming the lead side depends on. Measured at the top rather "
            "than at address, and so a different question from the one the view "
            "thresholds answer: a down-the-line clip has the shoulders nearly "
            "square at the top even though they were end-on at address."
        ),
    )
    min_lead_side_margin: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "How far towards one shoulder the hands must sit at the top before the "
            "lead side is named. Below it, no side is reported and the metrics that "
            "need one are refused rather than assigned to a coin flip."
        ),
    )
    face_on_span_ratio: float = Field(
        default=0.55,
        gt=0.0,
        description=(
            "Projected shoulder width at address, in torso lengths, at or above "
            "which the camera is treated as face-on. The basis is anatomical "
            "rather than measured here: an adult's shoulder width is a fairly "
            "stable multiple of the distance from their shoulders to their hips, "
            "so a shoulder line spanning more than half of it cannot be pointing "
            "at the camera. The reference face-on clip measures 0.83."
        ),
    )
    down_the_line_span_ratio: float = Field(
        default=0.30,
        gt=0.0,
        description=(
            "Projected shoulder width at address, in torso lengths, at or below "
            "which the camera is treated as down-the-line. The reference "
            "down-the-line clip measures 0.10. Between this and "
            "`face_on_span_ratio` the view is reported as unknown rather than "
            "rounded to the nearer label -- an oblique camera is a real thing to "
            "have recorded, and it supports some measurements and not others."
        ),
    )
    stability_window_s: float = Field(
        default=0.06,
        gt=0.0,
        description=(
            "Half-width, in **real** seconds, of the window a rotation's stability "
            "is measured over. The body turns smoothly, so a shoulder line that "
            "jumps about within a window this short is the estimator guessing an "
            "occluded landmark rather than the player moving. Real seconds, so a "
            "slow-motion clip supplies proportionally more frames rather than a "
            "proportionally longer look at the swing."
        ),
    )
    min_stability_frames: int = Field(
        default=5,
        ge=3,
        description=(
            "Frames the stability window must contain before an uncertainty is "
            "quantified at all. A 30 fps recording supplies three, which is too "
            "few to separate a jumping landmark from a turning body -- so the "
            "uncertainty is reported as unknown rather than as a small number "
            "computed from almost nothing."
        ),
    )
    max_rotation_uncertainty_deg: float = Field(
        default=15.0,
        gt=0.0,
        description=(
            "Uncertainty above which a foreshortening rotation is refused rather "
            "than reported. Stated policy, not a measured optimum: it is roughly "
            "the point past which the number stops distinguishing a full turn "
            "from a half one, which is the distinction it exists to make."
        ),
    )
    max_reference_excess: float = Field(
        default=1.25,
        gt=1.0,
        description=(
            "How much wider than its address baseline a body line may project "
            "somewhere in the clip before the recording is judged not to be "
            "face-on and its rotations refused. Above 1 because landmark noise "
            "alone moves the maximum a little past address -- it reaches 1.12 on "
            "the face-on reference clip. A down-the-line recording, where the "
            "shoulders start end-on and open through the backswing, exceeds it by "
            "a wide margin rather than a marginal one."
        ),
    )
    min_torso_length: float = Field(
        default=1e-6,
        gt=0.0,
        description=(
            "Torso length, in frame widths, below which the subject is too small "
            "or too poorly tracked to divide by. A guard against producing an "
            "enormous ratio from a torso that collapsed to a point."
        ),
    )


class MetricSet(BaseModel):
    """Everything the biomechanics engine concluded about one clip.

    `computed` is false when no swing was detected, and then `metrics` is empty
    rather than populated from arbitrary frames. Metrics are anchored to swing
    events; without events there is nothing to anchor them to, and measuring a
    "spine tilt at the top" on a clip with no top would be inventing the instant
    and then measuring it carefully.
    """

    schema_version: int = METRICS_SCHEMA_VERSION
    computed: bool
    metrics: list[Metric] = Field(default_factory=list)
    refused: list[RefusedMetric] = Field(default_factory=list)
    view: ViewEstimate | None = None
    lead_side: LeadSide | None = None
    references: list[RotationReference] = Field(default_factory=list)
    torso_length: float = Field(
        description="Median shoulder-midpoint to hip-midpoint distance, in frame widths."
    )
    geometry: FrameGeometry
    frames: int
    calibration: CalibrationStatus = Field(
        default=CalibrationStatus.NONE,
        description=(
            "What was known about this camera's geometry when these metrics were "
            "computed, and therefore what they are entitled to claim.\n\n"
            "`none` means the landmarks still carry the lens's distortion, which "
            "displaces a point near the frame edge by tens of pixels on an "
            "ordinary phone and biases every angle and distance measured from it. "
            "`intrinsics` means the lens was measured and removed, so the values "
            "below are cleaner statements about the image plane -- **and are "
            "still statements about the image plane**, because one camera cannot "
            "see depth however well it is calibrated. `stereo` is what Phase 9 "
            "triangulates with; no metric in this set requires it yet, and the "
            "gate that enforces that is in `compute.py`."
        ),
    )
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
    config: MetricConfig
    warnings: list[str] = Field(default_factory=list)

    def get(self, name: MetricName, event: SwingEvent | None = None) -> Metric | None:
        """The named metric at the given instant, or None if it was not produced."""
        return next(
            (m for m in self.metrics if m.name is name and m.event is event),
            None,
        )

    def by_group(self, group: MetricGroup) -> list[Metric]:
        """Every metric in one family, in the order they were computed."""
        return [m for m in self.metrics if m.group is group]
