"""Typed contracts for multi-view 3D reconstruction.

Two calibrated cameras, two rays, one point. This is the layer every earlier
phase has been qualifying its output against: Phase 5 labelled its angles
`PROJECTED_ANGLE` because one camera cannot see the third dimension, Phase 6
tagged every metric with the view it was taken from because a projection means
different things from different places, and Phase 8 returned unit `bearings` and
deliberately offered no function that turns one into a point. Here two of those
bearings meet, and the result is metres.

## The one thing this phase is built around

**The reprojection residual is blind to half of the error that matters**, and it
is blind to it in a specific, describable direction. That is the same shape as
Phase 8's finding and it is not the same fact, so it is worth stating precisely.

A point detected in camera 1 defines a ray. Every 3D point on that ray projects
into camera 2 along a single line -- the **epipolar line**. Now suppose camera
2's detection of the same joint is displaced from the truth by some small vector,
which it always is. Split that displacement into two components:

* **Across the epipolar line.** No 3D point on camera 1's ray can project there.
  The two rays are skew, the optimiser splits the difference, and the miss
  appears in the residual. This component is *visible*.
* **Along the epipolar line.** There is a 3D point on camera 1's ray that
  projects exactly there -- a point further along the ray, or nearer. The fit is
  perfect. The answer is wrong. This component is *invisible*, and it is the one
  that moves the reconstruction in depth.

So a reconstruction can have a quarter-pixel residual and be centimetres out, and
nothing in the residual says so. For the camera pair this system is built for --
one face-on, one down-the-line, roughly a right angle apart -- the epipolar lines
run nearly horizontally in both images, and nothing about a pose estimator makes
its horizontal error smaller than its vertical one. Roughly half the error, by
variance, lands in the half the residual cannot see.

`scripts/benchmark_reconstruct.py --sweep epipolar` measures exactly this: it
displaces one view's landmarks along the epipolar direction and watches the
residual stay flat while the true error grows without bound.

## So what does catch it

Two things, and they answer different questions, in the same way Phase 8's three
numbers did.

**Ray convergence angle -- the gate.** How much 3D error a given along-epipolar
displacement produces depends only on the angle between the two rays where they
meet. Depth error is roughly `sigma_px * Z / (f * sin(theta))`: at 90 degrees a
pixel of invisible error costs about a pixel's worth of depth, and at 5 degrees
it costs eleven times that. The angle is a property of **where the cameras were
put**, it is computable per point and per frame without any ground truth, and it
is what decides whether the capture could have determined the answer at all. It
is Phase 9's `CoverageReport`: the number that describes the capture rather than
the fit.

**Bone-length consistency -- the independent check.** A point sliding along its
ray changes its distance to its neighbours, so a displacement invisible to the
residual is fully visible in the length of the bones that end at it. And a bone
does not change length during a swing, which makes this a check that needs no
ground truth and no anatomical table: the *variation* of a reconstructed bone
length across a clip is error, whatever its absolute value is. Left-right
symmetry is a second such check, free for the same reason.

Neither of these makes the residual useless. It catches the failures it can see
-- a mis-paired instant, a landmark the two cameras disagree about across the
epipolar line, a stereo calibration that has drifted -- and those are real. It is
simply not the quality of the reconstruction, and this module reports all three
so no reader has to infer which question was answered.

## The Phase 8 capture instruction has no analogue here

Phase 8 established that two unsynchronised cameras can be stereo-calibrated
because the pairing error is `sync_error * image_speed`, and a board held still
drives the second factor to zero. The whole cost of not having a genlock
disappears if nothing moves.

**Nothing in a golf swing is still.** Hands reach twenty torso lengths per
second, which on a 1080p frame is thousands of pixels per second; at 30 fps a
half-frame pairing error is tens of pixels of displacement -- two orders of
magnitude past the sub-pixel residuals the calibration was gated on. Pairing
nearest frames, which is what Phase 8 does, is therefore not available.

So the target camera's trajectory is **resampled onto the reference camera's
clock** instead, through the time map, using the position *and the velocity* the
Phase 3 filter already fitted. What survives is the error in the map itself,
which Phase 7 reports and which is converted here into pixels the same way Phase
8 converted it: multiplied by the landmark's own measured image speed.
`PairingSummary` carries it, and it is directly comparable with the reprojection
error it would otherwise be mistaken for.

## What comes out is CAMERA, not WORLD

Triangulation produces metres in the **reference camera's** frame. That is
`LandmarkSpace.CAMERA`, and it is a genuine metric-scale 3D measurement: every
length, angle and speed between two reconstructed points is invariant to the
frame they are expressed in, so bone lengths, true joint angles and speeds in
metres per second are all available from it.

`WORLD` is not, and this phase does not produce it. A scene-fixed frame needs two
directions nothing here measures: **which way is up**, and **which way the target
line runs**. Neither is recoverable from a stereo pair alone -- the cameras do
not know their own attitude, and a body does not declare a target line. Both are
obtainable from a capture that puts the calibration board flat on the ground in
the hitting area, which the capture protocol does not currently ask for. That is
a change to `data/README.md` and a later phase, not an arithmetic problem, and
naming it is better than rotating into a frame whose axes were assumed.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.calibration import CalibrationStatus
from analyzer.contracts.camera import CameraRole
from analyzer.contracts.pose import LandmarkSpace

# Bump on any change that alters what a reported reconstruction means.
RECONSTRUCTION_SCHEMA_VERSION = 1


class RefusalReason(StrEnum):
    """Why one landmark at one frame produced no point.

    Counted per landmark rather than summarised into a single "failed" tally,
    because the four have completely different fixes: re-film with the cameras
    further apart, re-film with the subject unoccluded, re-synchronise, or
    re-calibrate. A single number would tell a reader that something went wrong
    and nothing about which.

    NOT_SEEN
        One or both cameras had no usable position for this landmark at this
        instant -- never detected, gated out for low confidence, inside a refused
        gap, or outside the window the filter could support.
    OUTSIDE_OVERLAP
        The reference instant maps outside the target clip's own timestamps, so
        resampling would be extrapolation. The clips overlap for less than the
        whole of the reference.
    ILL_CONDITIONED
        The two rays meet at too shallow an angle for their intersection to mean
        anything. See the module docstring: this is the gate.
    REPROJECTION
        The fitted point lands too far from where one of the cameras saw it. The
        two views are not looking at the same thing -- a mis-paired instant, a
        landmark one camera has confused, or extrinsics that have moved.
    UNCERTAIN
        The point was determined, and not well enough to be worth reporting: the
        propagated positional uncertainty is past the bound. Distinct from
        ILL_CONDITIONED because the causes differ -- that one is the camera
        placement, this one can also be a subject who was simply a long way off,
        since uncertainty grows with the square of distance while the convergence
        angle only shrinks.
    """

    NOT_SEEN = "not_seen"
    OUTSIDE_OVERLAP = "outside_overlap"
    ILL_CONDITIONED = "ill_conditioned"
    REPROJECTION = "reprojection"
    UNCERTAIN = "uncertain"


class PairingSummary(BaseModel):
    """How the two cameras' instants were brought together, and what it cost.

    The Phase 8 thread, continued into a scene where its capture instruction is
    unavailable. See the module docstring: a board can be held still and a swing
    cannot, so the target clip is resampled onto the reference clock rather than
    paired frame-to-frame, and what remains is the time map's own uncertainty.

    That is reported here in **pixels**, by multiplying it by the landmark's
    measured image speed, for the same reason `PairingReport` does: it is then
    the same unit as the reprojection error it contaminates, and the two can be
    compared instead of one being mistaken for the other.
    """

    method: str = Field(
        description=(
            "How the target clip's positions were obtained at reference instants: "
            "'resampled' (Hermite interpolation of the filter's own fitted "
            "position and velocity) or 'nearest_frame'."
        )
    )
    resampled_frames: int = Field(description="Reference frames a target position was produced at.")
    outside_overlap: int = Field(
        description="Reference frames whose mapped instant fell outside the target clip."
    )
    median_interval_ms: float = Field(
        description="The target clip's own frame interval. What nearest-frame pairing would round to."
    )
    median_landmark_speed_px_s: float = Field(
        description=(
            "Median image speed of the tracked landmarks in the target view, over "
            "the reconstructed frames. The factor that turns a timing error into a "
            "displacement -- and the one a still board sets to zero and a swing "
            "cannot."
        )
    )
    max_landmark_speed_px_s: float = Field(
        description="The fastest any tracked landmark moved in the target image. Usually a hand."
    )
    sync_uncertainty_ms: float | None = Field(
        default=None,
        description=(
            "The time map's own uncertainty over the reconstructed span, in "
            "milliseconds, as Phase 7 reports it. None when the map carried none."
        ),
    )
    median_pairing_error_px: float | None = Field(
        default=None,
        description=(
            "Sync uncertainty times landmark image speed, at the median. The "
            "displacement attributable to not knowing exactly when each frame was "
            "taken. Comparable with `median_reprojection_px` because it is the "
            "same unit and it adds to it."
        ),
    )
    max_pairing_error_px: float | None = Field(
        default=None,
        description=(
            "The same quantity at the fastest landmark. This is the number that "
            "makes nearest-frame pairing unusable on a swing: it is tens of pixels "
            "at consumer frame rates, against sub-pixel calibration residuals."
        ),
    )
    nearest_frame_error_px: float | None = Field(
        default=None,
        description=(
            "What pairing to the nearest target frame *would* have cost at the "
            "**fastest** landmark: a quarter of a frame interval -- the mean "
            "magnitude of a rounding uniform over half of one -- times that "
            "speed. Quoted at the fastest rather than the median because the "
            "median is dominated by the parts of a body that barely move, and "
            "every metric worth computing is anchored to the parts that do. "
            "Reported rather than assumed, because it is the measurement that "
            "decided this layer resamples."
        ),
    )


class BoneConsistency(BaseModel):
    """One skeletal segment's reconstructed length across the clip.

    **The check the reprojection residual cannot perform.** A landmark displaced
    along its epipolar line reprojects perfectly and sits at the wrong depth; the
    bones ending at it are then the wrong length, and they are a different wrong
    length in every frame, because the displacement varies. So the *variation* is
    the error signal, and it needs no ground truth -- a humerus does not change
    length during a swing.

    `median_length_m` is reported too, and it is this engine's **first metric
    statement about a body**. It is not an anatomical bone length: MediaPipe's
    landmarks are the model's estimate of a joint's image position, not a joint
    centre, and the distance between two of them is not a distance between two
    bones. What it is good for is a sanity check a person can make against a
    tape measure, and that is how it should be read.
    """

    name: str = Field(description="The segment, as 'left_shoulder-left_elbow'.")
    frames: int = Field(description="Frames in which both ends were reconstructed.")
    median_length_m: float
    variation: float = Field(
        ge=0.0,
        description=(
            "Robust coefficient of variation: the interquartile range over the "
            "median. Robust rather than a standard deviation because a handful of "
            "badly reconstructed frames should not be allowed to describe the "
            "clip, and because a clip with any of them still has a usable middle."
        ),
    )
    spread_m: float = Field(
        ge=0.0, description="Interquartile range of the length, in metres. The same fact, unscaled."
    )
    stable: bool = Field(
        description="Whether `variation` sits inside `ReconstructionConfig.max_bone_variation`."
    )


class SymmetryCheck(BaseModel):
    """How far a left segment's reconstructed length sits from its right twin.

    A second check that needs no ground truth, for a second reason: a person's
    left and right upper arms are the same length to within a per cent or two,
    and the two are reconstructed from completely different landmarks with
    completely different occlusion histories. A large disagreement is evidence
    about the reconstruction rather than about the body.

    **It is corroboration, not a stronger instrument, and the design here
    expected otherwise.** The argument for it was that a landmark with a
    *consistent* depth bias -- one camera placing a hip a few centimetres deep
    throughout, which occlusion does routinely -- would give a stable wrong
    length in every frame and sail through the variation check. Measured, it does
    not: a swing rotates the body, so a displacement that is constant in the
    camera's frame is not constant relative to the bone, and the length moves.
    Displacing one elbow 5 cm along the optical axis on the synthetic body makes
    the forearm's variation 11% -- past the bound -- while the left-right
    disagreement is 2.6%, well inside it. Variation gets there first.

    So what this adds is **localisation**, which is worth its two lines: it
    compares two chains reconstructed from different landmarks, so a large
    disagreement says one *side* is systematically worse, which a per-segment
    variation does not say. The test asserting this is
    `test_a_depth_bias_reaches_variation_before_symmetry`, and it is written to
    fail if anyone restores the stronger claim.
    """

    segment: str = Field(description="The pair, as 'shoulder-elbow'.")
    left_m: float
    right_m: float
    disagreement: float = Field(ge=0.0, description="|left - right| divided by their mean.")
    agrees: bool = Field(
        description="Whether the disagreement sits inside `ReconstructionConfig.max_asymmetry`."
    )


class LandmarkReconstruction(BaseModel):
    """One landmark's reconstruction across the clip, and where it failed.

    Per landmark rather than clip-wide, because the failures are per landmark:
    down-the-line footage hides one wrist behind the other for half a swing, and
    a clip-wide coverage figure would report that as a uniform 80% rather than as
    one landmark at 40% and the rest at 95%. The first reading suggests a
    marginal capture; the second names the landmark to distrust.
    """

    landmark: int = Field(description="The `Landmark` enum value.")
    name: str
    frames: int = Field(description="Reference frames attempted.")
    reconstructed: int
    coverage: float = Field(ge=0.0, le=1.0)
    median_convergence_deg: float | None = Field(
        default=None,
        description="Median angle between the two rays. The conditioning of the intersection.",
    )
    median_reprojection_px: float | None = None
    median_uncertainty_m: float | None = Field(
        default=None,
        description=(
            "Median propagated positional uncertainty, in metres, in the worst "
            "direction -- which is essentially always the depth direction. See "
            "`ReconstructionQuality.methodology`."
        ),
    )
    refused: dict[RefusalReason, int] = Field(
        default_factory=dict, description="Frames refused, counted by reason."
    )


class ReconstructionQuality(BaseModel):
    """What the reconstruction is worth, in the three senses that differ.

    Deliberately shaped like `CalibrationQuality`, because the lesson is the
    same one a layer up. The residual is the fit, the convergence angle is the
    capture, and the bones are the independent check -- and a tool that reported
    only the first would be reporting the number that a badly-conditioned capture
    makes *better*, since two nearly-parallel rays can be made to agree by moving
    a point a long way in depth.
    """

    points_attempted: int = Field(description="Landmark-frames both cameras were asked about.")
    points_reconstructed: int
    coverage: float = Field(ge=0.0, le=1.0)

    median_reprojection_px: float | None = Field(
        default=None,
        description=(
            "Median distance, over both views, between a reconstructed point's "
            "projection and where that camera saw the landmark. **How well the "
            "two views agree in the direction they can disagree in** -- the "
            "component of correspondence error across the epipolar line. Blind to "
            "the component along it, which is the one that moves the point in "
            "depth."
        ),
    )
    max_reprojection_px: float | None = None
    median_convergence_deg: float | None = Field(
        default=None,
        description=(
            "Median angle between the two rays at their meeting point. **The "
            "gate.** Depth error scales as 1/sin of it, so this is what decides "
            "whether the capture could determine a position, independently of how "
            "well the fit agrees with itself."
        ),
    )
    min_convergence_deg: float | None = None
    median_uncertainty_m: float | None = Field(
        default=None,
        description=(
            "Median propagated positional uncertainty in the worst direction, in "
            "metres. Propagated from a per-view pixel sigma through the "
            "triangulation's own Jacobian, so it carries the convergence angle "
            "automatically: the same pixel noise at 10 degrees of convergence "
            "gives six times the uncertainty it gives at 90."
        ),
    )
    p95_uncertainty_m: float | None = None

    bones: list[BoneConsistency] = Field(default_factory=list)
    symmetry: list[SymmetryCheck] = Field(default_factory=list)
    worst_bone_variation: float | None = Field(
        default=None,
        description=(
            "The least stable segment's coefficient of variation. The headline of "
            "the check the residual cannot make."
        ),
    )
    pixel_sigma_px: float = Field(
        description=(
            "The per-view landmark uncertainty the propagation used, in pixels. "
            "**Measured from this clip**, as the filter's own residual RMS -- how "
            "far the raw detections sat from the fitted trajectory -- rather than "
            "assumed. It is an upper bound on the fitted position's own error, "
            "because smoothing averages several samples, and is used as one."
        )
    )
    methodology: str


class ReconstructionReport(BaseModel):
    """Everything the engine concluded about reconstructing one pair of clips.

    The landmarks themselves are **not** here, for the reason
    `PoseExtractionResult` leaves them out: a swing at 240 fps is tens of
    thousands of frames of 33 points, which belongs in an array in the engine or
    a columnar file on disk rather than in a JSON-RPC response. What crosses the
    boundary is what a reader needs in order to decide whether to believe it.

    `reconstructed` is false whenever no point could be produced, and then the
    quality fields are None rather than zero -- zero coverage and "no
    reconstruction attempted" are different facts, and a consumer reading a zero
    as a measurement would be reading a refusal as a result.
    """

    schema_version: int = RECONSTRUCTION_SCHEMA_VERSION
    reconstructed: bool
    space: LandmarkSpace = Field(
        default=LandmarkSpace.CAMERA,
        description=(
            "The frame the points are in: metres, centred on the reference "
            "camera. **Not WORLD.** A scene-fixed frame needs a gravity direction "
            "and a target line, and a stereo pair supplies neither; see the module "
            "docstring for what would."
        ),
    )
    reference_role: CameraRole
    target_role: CameraRole
    reference_name: str = Field(description="Basename of the reference clip, for display.")
    target_name: str

    frames: int = Field(description="Reference-clip frames the reconstruction spans.")
    reconstructed_frames: int = Field(
        description="Frames carrying at least one reconstructed landmark."
    )
    slow_motion_factor: float = Field(
        default=1.0,
        description=(
            "The reference clip's factor. Timestamps here are real seconds, as "
            "everywhere else above Phase 3."
        ),
    )

    calibration: CalibrationStatus = Field(
        description=(
            "What was known about the two cameras' geometry. Always `stereo` for "
            "a reconstruction that happened; carried so a stored result cannot be "
            "read without it."
        )
    )
    baseline_m: float = Field(description="Distance between the two optical centres, from the rig.")
    convergence_deg: float = Field(
        description=(
            "Angle between the two optical axes, from the rig. Distinct from the "
            "per-point ray convergence in `ReconstructionQuality`: this is where "
            "the cameras pointed, that is how well two particular rays met."
        )
    )

    quality: ReconstructionQuality | None = None
    pairing: PairingSummary | None = None
    landmarks: list[LandmarkReconstruction] = Field(default_factory=list)
    reconstructed_at: datetime | None = None

    config: ReconstructionConfig
    refusal: str | None = Field(
        default=None,
        description="Why nothing was produced, in words. Set exactly when `reconstructed` is false.",
    )
    warnings: list[str] = Field(default_factory=list)


class ReconstructionConfig(BaseModel, extra="forbid"):
    """Policy for reconstruction.

    As everywhere else in this engine, **stated policy informed by measurement**
    rather than measured optima. `scripts/benchmark_reconstruct.py` establishes
    the shape of each relationship -- what error a given convergence angle
    produces, how a residual responds to an epipolar displacement and how it
    fails to -- and where on that curve to refuse is a judgement about what a
    wrong answer costs. Where a number came from a sweep, the field says so.
    """

    min_convergence_deg: float = Field(
        default=15.0,
        gt=0.0,
        description=(
            "Angle between the two rays below which a point is refused rather "
            "than reported. **The gate**, and the only bound here that describes "
            "the capture rather than the fit.\n\n"
            "Measured (`--sweep convergence`): depth error scales as 1/sin, so 15 "
            "degrees costs 3.9x the error of a right-angled pair and 5 degrees "
            "costs 11.5x. Fifteen is where a centimetre-scale reconstruction "
            "becomes a several-centimetre one, which is roughly where a joint "
            "angle stops being worth reporting. Two cameras placed as the capture "
            "protocol asks -- one face-on, one down-the-line -- converge at nearly "
            "90 degrees and are nowhere near this."
        ),
    )
    max_reprojection_px: float = Field(
        default=6.0,
        gt=0.0,
        description=(
            "How far a reconstructed point may land from where a camera saw it "
            "before that point is refused. Loose on purpose, and loose for a "
            "different reason from the calibration bound it resembles: a "
            "calibration residual is a static board fitted by a rigid model, and "
            "this is a pose estimator's guess at a joint on a moving body seen "
            "from two sides. Several pixels of honest disagreement is ordinary.\n\n"
            "What it catches is gross failure -- a mis-paired instant, one camera "
            "tracking the wrong wrist. It is **not** a quality measure, and the "
            "module docstring is about why not."
        ),
    )
    max_uncertainty_m: float | None = Field(
        default=0.15,
        description=(
            "Propagated positional uncertainty above which a point is refused. "
            "Fifteen centimetres is most of the width of a pelvis, which is "
            "roughly the point past which a reconstructed joint stops "
            "distinguishing the poses it exists to distinguish. None disables the "
            "bound and reports the uncertainty without acting on it."
        ),
    )
    min_visibility: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Reported visibility, in **either** view, below which the landmark is "
            "not triangulated at that instant. Either rather than both: a point "
            "reconstructed from one confident view and one guess is a point "
            "located by the guess, since it is the disagreement between the two "
            "that fixes the depth."
        ),
    )
    max_bone_variation: float = Field(
        default=0.10,
        gt=0.0,
        description=(
            "Robust coefficient of variation above which a reconstructed segment "
            "is reported as unstable. Ten per cent of a forearm is about three "
            "centimetres, which is the scale at which the reconstruction stops "
            "supporting the joint angles it exists to produce.\n\n"
            "**Not tighter, because not every segment here is a bone.** "
            "`POSE_CONNECTIONS` includes shoulder-to-hip on each side, and a "
            "torso genuinely changes that distance as it twists -- the synthetic "
            "body, whose limb lengths are exact by construction, varies those two "
            "by 3% across a swing purely from the shoulders and pelvis turning "
            "different amounts. A bound tight enough for a humerus would report "
            "that anatomy as an error in every clip."
        ),
    )
    max_asymmetry: float = Field(
        default=0.15,
        gt=0.0,
        description=(
            "How far a left segment's median length may sit from its right twin's, "
            "as a fraction of their mean, before the pair is reported as "
            "disagreeing. Wider than `max_bone_variation` because real people are "
            "slightly asymmetric and because one side is systematically the more "
            "occluded in any single capture."
        ),
    )
    pixel_sigma_px: float | None = Field(
        default=None,
        description=(
            "Per-view landmark uncertainty used to propagate a positional "
            "uncertainty. None measures it from the clip, as the filter's own "
            "residual RMS -- which is what the detections actually did, rather "
            "than a number chosen to make the output look good. Supplying one is "
            "for a sensitivity check."
        ),
    )
    resample: bool = Field(
        default=True,
        description=(
            "Whether to resample the target clip onto the reference clock rather "
            "than pairing each reference frame with the target frame nearest it "
            "in time.\n\n"
            "**Measured, and it is not a close call.** Phase 8 pairs nearest "
            "frames and gets away with it because a calibration board can be held "
            "still, which drives the `time_error x image_speed` product to zero. "
            "A swing cannot be held still: at 30 fps a quarter-frame pairing "
            "error against hands moving several thousand pixels per second is "
            "tens of pixels of displacement, two orders of magnitude past the "
            "sub-pixel residuals the calibration itself was gated on. "
            "`scripts/benchmark_reconstruct.py --sweep pairing` has both. False "
            "is there so that comparison can be run, not as an option worth "
            "choosing."
        ),
    )
    refine: bool = Field(
        default=True,
        description=(
            "Whether to refine the linear triangulation by minimising reprojection "
            "error in both views. The linear solution minimises an algebraic "
            "quantity with no geometric meaning and is biased when the two views "
            "are at very different distances; the refinement is a few Gauss-Newton "
            "steps from it and is what makes the residual a reprojection error "
            "rather than a proxy for one. Off is for measuring what it bought."
        ),
    )
    max_refine_iterations: int = Field(
        default=10,
        ge=1,
        description=(
            "Gauss-Newton steps. From a linear seed this converges in two or "
            "three; the bound exists so a pathological point cannot spend the "
            "clip's budget rather than because ten are expected to be used."
        ),
    )


# Pydantic resolves the forward reference from `ReconstructionReport` to the
# config declared below it. Declared in this order because a reader meets the
# report first and the policy second, which is the order they matter in.
ReconstructionReport.model_rebuild()
