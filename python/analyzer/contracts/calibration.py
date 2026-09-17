"""Typed contracts for camera calibration.

A calibration is the first thing in this engine that turns a picture back into a
statement about space. Everything below it measures the image: a distance in
frame widths, an angle between two projected segments, a rotation inferred from
how much a body line shortened. Those are exact statements about a photograph and
approximate statements about a body, and no amount of care in the arithmetic
closes that gap -- only knowing where the light went does.

Calibration is what supplies that. It answers, for every pixel, *which direction
through space did this come from*, and it does so by measuring a known object
rather than by trusting the file: focal lengths, the optical centre, and the lens
distortion that bends a straight line into an arc near the edge of the frame.

## The one thing calibration does not do

It does not make one camera see in three dimensions. A calibrated pixel is a
**ray**, not a point -- it names a direction and says nothing about how far along
that direction the thing sat. Two rays from two calibrated cameras intersect, and
that intersection is a point; one ray does not.

This is why `CalibrationStatus` has three values rather than two. A system with
`INTRINSICS` can remove the lens distortion from a landmark, which genuinely
improves every projected measurement above it, and it still cannot make a single
metric-scale 3D claim. Collapsing "calibrated" into one boolean would let a
consumer read a corrected 2D measurement as a 3D one, and the correction is
exactly what makes it look trustworthy enough to misread.

## RMS reprojection error is not the quality of a calibration

This is the trap the phase is built around, and it is not obvious, because RMS
reprojection error is the number every calibration tool prints and the number
everyone quotes.

It measures **how well the model fits the data it was given**. It says nothing
about whether that data determined the model. Fifteen views of a board held
square-on at arm's length, all near the middle of the frame, will fit a pinhole
model beautifully -- an RMS of a fifth of a pixel is easy -- and will leave the
focal length essentially unmeasured, because at fixed distance and fixed
orientation a longer lens further away and a shorter lens nearer produce very
nearly the same picture. The residual is small because the model has nothing to
disagree with, not because it is right.

So three separate numbers are reported, and they answer three different
questions:

    rms_reprojection_px   how well the model fits the data          (the fit)
    focal_uncertainty_px  what the fit says about its own spread    (a check)
    CoverageReport        what the views could possibly determine   (the gate)

## Neither of the first two catches a degenerate capture

The design here assumed the middle one would. `cv2.calibrateCameraExtended`
returns a standard deviation for every intrinsic, propagated from the residuals
through the fit's Jacobian, and a near-degenerate problem ought to show up as an
uncertainty that explodes. **Measured, it does the opposite**, and
`scripts/benchmark_calibration.py --sweep tilt` is where that was found:

    tilt spread     RMS px     reported sigma fx     true fx error
    +-0 deg          0.220           0.012%              6.33%
    +-2 deg          0.210           0.012%             10.04%
    +-20 deg         0.245           0.656%              0.10%
    +-35 deg         0.218           0.314%              0.05%

The residual is flat across the whole range, and the reported uncertainty is
**thirty times smaller** exactly where the answer is worst. The reason is that
the degeneracy is not left unresolved: the distortion coefficients absorb it.
The fit finds a combination of focal length and distortion that reproduces those
centred, square-on views almost perfectly, and that combination is tightly
determined *by those views* and wrong everywhere else. A covariance computed
from the same views cannot see outside them.

So the parameter uncertainty is reported and is a useful check on the fit's own
conditioning, and it is **not** what `usable` rests on. `CoverageReport` is,
because it is the only one of the three that describes the *capture* rather than
the fit -- and a degenerate capture is what all of this is trying to catch.

## What coverage has to contain

Three things, because three different degeneracies are possible and each is
invisible in the residual.

**Area.** Distortion is a function of radius from the optical centre and is
almost nothing in the middle of the frame. A board that never leaves the centre
leaves `k1` fitted to noise, and the fitted value is then applied out at the
corners where it does all its work.

**Tilt.** A board held parallel to the sensor at several distances does not
separate focal length from distance: scaling both leaves the image identical.
Tilting the board breaks that, because perspective foreshortening across a
tilted plane depends on the focal length in a way that a pure scale does not.
This is the degeneracy that makes a careful-looking calibration wrong, and
`tilt_range_deg` is what detects it.

**Scale.** Views at one distance measure the lens at one working distance. The
spread is reported for the same reason as the other two: as evidence about what
the data could constrain.

## The stereo pair, and the sync error that reaches it in pixels

Stereo extrinsics need the two cameras to see the board **at the same instant**,
and two consumer cameras do not share a clock -- which is the whole of Phase 7.
The naive consequence would be that stereo calibration needs a genlock.

It does not, and the reason is worth stating because it turns into a capture
instruction. What matters is not that the frames were simultaneous but that
*nothing moved between them*. Hold the board still for a second at each position
and a pairing error of 30 ms costs nothing, because the board is in the same
place 30 ms later. Wave it around and the same 30 ms is a real displacement.

So the pairing error is not assumed away and it is not assumed fatal: it is
**measured, in pixels**, by multiplying the time map's own uncertainty at that
instant (Phase 7 reports it) by the board's observed image speed there. That
number is a length in the same unit as the reprojection error it contaminates,
so the two are directly comparable, and `PairingReport` carries it. A pair whose
sync cost exceeds `CalibrationConfig.max_pairing_error_px` is dropped before the
fit rather than degrading it invisibly.

## A calibration belongs to a camera *and a setting*

It is not a property of a phone. It is a property of a phone at one zoom, one
lens, one capture resolution -- and the most common way to get a wrong answer
from a right calibration is to apply it to footage shot differently.

Most of that cannot be detected from a video file. Frame size can, so it is
carried on `CameraIntrinsics` and checked by `applies_to`, and a mismatch is a
refusal rather than a rescale: a 4K calibration scaled to 1080p is right only if
the sensor was cropped the way the guess assumes, and nothing in the file says
whether it was.

The rest is a capture-protocol problem, stated in `data/README.md`, and the
contract carries `notes` so the setting can at least be recorded by the person
who knows it.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from analyzer.contracts.camera import CameraRole

# Bump on any change that alters what a stored or reported calibration means.
# The projects store keys on this: a row written under a version it does not
# understand is refused rather than parsed hopefully.
CALIBRATION_SCHEMA_VERSION = 1


class BoardFamily(StrEnum):
    """Which ArUco dictionary the board's markers are drawn from.

    Carried rather than fixed because the board is a physical object that the
    user printed, and a detector looking for the wrong dictionary finds nothing
    at all -- a failure that looks exactly like bad lighting.

    The default is `DICT_5X5_100`: 5x5 markers stay legible at the size a
    Charuco square allows on an A4 sheet, and 100 of them is far more than a
    board this size consumes, so ids never wrap.
    """

    DICT_4X4_50 = "DICT_4X4_50"
    DICT_4X4_100 = "DICT_4X4_100"
    DICT_5X5_100 = "DICT_5X5_100"
    DICT_5X5_250 = "DICT_5X5_250"
    DICT_6X6_250 = "DICT_6X6_250"


class BoardSpec(BaseModel):
    """The physical board, as printed.

    Charuco rather than a plain chessboard, for two reasons that both matter on
    footage taken by one person with no assistant. A chessboard must be seen
    **whole** or it yields nothing, so every view where a corner leaves the
    frame is wasted -- which is exactly the view that carries the distortion
    information. And its corners are interchangeable, so a rotationally
    symmetric view is ambiguous. Charuco's markers identify each corner
    individually, so a partial view contributes the corners it does show and the
    correspondence is never in doubt.

    `square_length_m` is what puts a metre into the system. Every metric-scale
    claim Phase 9 makes descends from this one measured number, so it is the
    length to measure on the printed sheet with a ruler rather than to take from
    the file that was sent to the printer: printers scale to fit, and a page
    scaled to 96% makes every future distance wrong by 4% with nothing
    anywhere looking amiss.
    """

    squares_x: int = Field(ge=2, description="Chessboard squares across.")
    squares_y: int = Field(ge=2, description="Chessboard squares down.")
    square_length_m: float = Field(
        gt=0.0,
        description=(
            "Side of one chessboard square, in metres, **measured on the printed "
            "sheet**. The scale of every metric claim descends from this number."
        ),
    )
    marker_length_m: float = Field(
        gt=0.0,
        description=(
            "Side of one ArUco marker, in metres. Must be smaller than the "
            "square it sits in; around 0.75 of it leaves a white margin that the "
            "detector needs to find the marker's border."
        ),
    )
    family: BoardFamily = BoardFamily.DICT_5X5_100
    legacy_pattern: bool = Field(
        default=False,
        description=(
            "Whether the board was generated by OpenCV before 4.6, which laid "
            "the markers out differently. A board printed from an old generator "
            "and detected as a new one produces corners in the wrong places "
            "rather than no corners at all, which is the worse failure. False "
            "for any board this build generated."
        ),
    )

    @property
    def interior_corners(self) -> int:
        """Chessboard corners the board can supply. The most any one view can give."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def width_m(self) -> float:
        return self.squares_x * self.square_length_m

    @property
    def height_m(self) -> float:
        return self.squares_y * self.square_length_m


class DistortionModel(StrEnum):
    """Which lens-distortion coefficients were fitted.

    Named and chosen rather than defaulted, because a coefficient that is not
    determined by the data does not sit harmlessly near zero -- it takes a value
    that cancels some of the residual in the region the board happened to cover
    and then diverges outside it, which is the region a swing is filmed in.

    PINHOLE
        No distortion. Correct only for a lens that has none, which no phone
        wide-angle has; useful as a null model and as the thing a test can
        recover exactly.
    RADIAL_TANGENTIAL_4
        k1, k2, p1, p2. Two radial terms and the tangential pair that models a
        lens not quite parallel to the sensor. The default here, on the
        measurement in `scripts/benchmark_calibration.py`.
    RADIAL_TANGENTIAL_5
        The above plus k3. OpenCV's own default, and a third radial term is
        genuinely needed by a fisheye or an action camera -- but it is the term
        that most readily absorbs noise when the board never reaches the corners,
        and it does its damage exactly there.
    """

    PINHOLE = "pinhole"
    RADIAL_TANGENTIAL_4 = "radial_tangential_4"
    RADIAL_TANGENTIAL_5 = "radial_tangential_5"

    @property
    def coefficients(self) -> int:
        return {
            DistortionModel.PINHOLE: 0,
            DistortionModel.RADIAL_TANGENTIAL_4: 4,
            DistortionModel.RADIAL_TANGENTIAL_5: 5,
        }[self]


class CalibrationStatus(StrEnum):
    """What this build may claim about the geometry of a recording.

    The honesty gate of Phases 8 and 9, as a type. It is ordered, and
    `at_least` is how a consumer asks the question it actually has -- "may I
    triangulate" -- rather than enumerating the values that permit it and
    forgetting one when a fourth is added.

    NONE
        No calibration. Every measurement is a statement about the image plane,
        which is what Phases 5 and 6 produce and label as such. The lens's
        distortion is present in every landmark and unmeasured.
    INTRINSICS
        One camera's focal lengths, optical centre and distortion are measured.
        Landmarks can be undistorted, so projected measurements improve, and a
        pixel becomes a known direction. **Still no depth, so still no
        metric-scale 3D claim.**
    STEREO
        Both cameras are calibrated and their relative pose is measured, so two
        views of one instant give an intersection rather than two directions.
        This is what Phase 9 triangulates with.
    """

    NONE = "none"
    INTRINSICS = "intrinsics"
    STEREO = "stereo"

    @property
    def rank(self) -> int:
        return {
            CalibrationStatus.NONE: 0,
            CalibrationStatus.INTRINSICS: 1,
            CalibrationStatus.STEREO: 2,
        }[self]

    def at_least(self, required: CalibrationStatus) -> bool:
        """Whether this status permits what `required` demands."""
        return self.rank >= required.rank


class CoverageReport(BaseModel):
    """What the board views actually sampled, and therefore what they could fix.

    The diagnostic half of a calibration. Each field detects a different
    degeneracy, and every one of them is invisible in the reprojection residual
    -- a degenerate set fits *better*, because the model has less to disagree
    with. See the module docstring for why each matters.
    """

    views: int = Field(description="Board views that entered the fit.")
    corners: int = Field(description="Chessboard corners summed over those views.")
    image_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the frame, on a coarse grid, containing at least one "
            "detected corner. The blunt measure of whether the lens was sampled "
            "where it bends."
        ),
    )
    edge_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of all detected corners lying in the outer fifth of the "
            "frame radius. Distortion is a function of radius and is nearly "
            "nothing at the centre, so this is the share of the evidence that "
            "carries any information about it at all."
        ),
    )
    tilt_range_deg: float = Field(
        ge=0.0,
        description=(
            "Spread between the least and most tilted board view, in degrees "
            "off square to the camera. **The field that catches the worst "
            "case**: a board always held parallel to the sensor cannot separate "
            "focal length from distance, and produces an excellent residual "
            "around a focal length that is simply not determined."
        ),
    )
    scale_range: float = Field(
        ge=1.0,
        description=(
            "Largest apparent board size over smallest, across views. 1.0 means "
            "every view was taken at one distance."
        ),
    )
    methodology: str


class CameraIntrinsics(BaseModel):
    """One camera's interior geometry, measured.

    Stated as four scalars rather than a 3x3 matrix because that is what was
    fitted: the skew term of a general K is zero for any sensor manufactured in
    the last forty years, and carrying a full matrix would invite a reader to
    believe this build had measured it. `matrix()` assembles the conventional
    form for the arithmetic that wants one.

    Pixel coordinates here are **displayed** pixels, after Phase 1's rotation has
    been applied -- the same frame the landmarks are normalised against. A
    calibration measured on the stored orientation and applied to the displayed
    one has fx and fy interchanged, which is a plausible-looking calibration that
    is wrong everywhere.
    """

    fx: float = Field(gt=0.0, description="Focal length in pixels, horizontal.")
    fy: float = Field(gt=0.0, description="Focal length in pixels, vertical.")
    cx: float = Field(description="Optical centre, horizontal, in pixels from the left.")
    cy: float = Field(description="Optical centre, vertical, in pixels from the top.")
    distortion: list[float] = Field(
        description=(
            "Distortion coefficients in OpenCV's order (k1, k2, p1, p2[, k3]), "
            "as many as `model` declares."
        )
    )
    model: DistortionModel
    image_width: int = Field(gt=0, description="Displayed frame width this was measured at.")
    image_height: int = Field(gt=0, description="Displayed frame height this was measured at.")

    fx_uncertainty: float | None = Field(
        default=None,
        description=(
            "Standard deviation of `fx`, in pixels, propagated from the fit's "
            "residuals through its Jacobian.\n\n"
            "**It does not catch a degenerate capture, which is what the design "
            "here expected of it.** Measured, it is smallest exactly where the "
            "answer is worst -- 0.012% on a capture whose focal length is wrong "
            "by 6%, against 0.31% on one that is right to 0.05% -- because the "
            "distortion coefficients absorb the degeneracy and leave a tightly "
            "determined wrong answer. A covariance computed from one set of "
            "views cannot see outside them. Reported because it is a real "
            "statement about the fit's conditioning, and not what `usable` rests "
            "on; `CoverageReport` is. None where the fit did not report one."
        ),
    )
    fy_uncertainty: float | None = None
    cx_uncertainty: float | None = None
    cy_uncertainty: float | None = None

    def matrix(self) -> NDArray[np.float64]:
        """The conventional 3x3 camera matrix, with zero skew."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def distortion_vector(self) -> NDArray[np.float64]:
        """Distortion coefficients as OpenCV wants them: a row, possibly empty."""
        return np.array(self.distortion, dtype=np.float64).reshape(1, -1)

    @property
    def horizontal_fov_deg(self) -> float:
        """Horizontal field of view, in degrees.

        Carried because it is the one intrinsic a person can check against the
        world without any tooling. A phone's main camera sees roughly 65-70
        degrees across, an ultra-wide over 100, a long lens under 30. A
        calibration reporting 110 degrees for footage that plainly is not
        ultra-wide is wrong, and that is visible here and nowhere else in the
        numbers.
        """
        return math.degrees(2.0 * math.atan(self.image_width / (2.0 * self.fx)))

    @property
    def vertical_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.image_height / (2.0 * self.fy)))

    @property
    def focal_disagreement(self) -> float:
        """|fx - fy| / mean, a unitless check on the fit.

        Square pixels are universal in consumer sensors, so fx and fy describe
        one physical focal length in two units that ought to be the same. A
        large disagreement is evidence of a bad fit or of anamorphic
        preprocessing, not of an unusual sensor.
        """
        mean = 0.5 * (self.fx + self.fy)
        return abs(self.fx - self.fy) / mean if mean > 0 else float("inf")

    def applies_to(self, width: int, height: int) -> bool:
        """Whether this calibration may be used on a frame of this size.

        Exact equality, deliberately. Rescaling intrinsics to another resolution
        is right only when the sensor was scaled rather than cropped, and a
        video file does not record which happened -- so the guess would be
        invisible and wrong about a third of the time.
        """
        return self.image_width == width and self.image_height == height


class BoardObservation(BaseModel):
    """One frame in which the board was found.

    Kept per frame rather than summarised because 8.4 draws them: the review UI
    shows where in the image each view landed, which is how a person sees that
    their board never reached a corner -- a fact no single number communicates
    as quickly as the picture of it does.
    """

    frame: int
    corners: int = Field(description="Chessboard corners identified in this view.")
    reprojection_rms_px: float | None = Field(
        default=None,
        description=(
            "This view's own RMS residual after the fit. None before a fit, and "
            "for a view the fit excluded."
        ),
    )
    tilt_deg: float | None = Field(
        default=None, description="How far off square to the camera this view sat."
    )
    distance_m: float | None = Field(
        default=None,
        description=(
            "Board distance along the optical axis. Metric, because the board's "
            "own square size supplies the scale -- the first genuinely metric "
            "length in this engine."
        ),
    )
    centroid_x: float = Field(description="Mean corner position, in pixels.")
    centroid_y: float
    used: bool = Field(
        default=True, description="Whether this view entered the fit, or was dropped."
    )
    dropped_reason: str | None = None


class DetectionReport(BaseModel):
    """What board detection found across a source, and what it discarded.

    Every count is here rather than only the survivors, because the ratios are
    the diagnostic. Frames scanned against frames with a board found says
    whether the board was visible; found against used says whether the views
    were good enough; and a large gap in either has a different fix -- reshoot
    with the board in frame, or reshoot with it closer and steadier.
    """

    frames_scanned: int
    frames_with_board: int
    views_used: int
    corners_total: int
    board: BoardSpec
    observations: list[BoardObservation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CalibrationQuality(BaseModel):
    """How good a calibration is, in the three senses that differ.

    See the module docstring. The short version: `rms_reprojection_px` is the
    fit, the parameter uncertainties on `CameraIntrinsics` are the answer, and
    `coverage` is the cause. A tool that reports only the first is reporting the
    one of the three that a degenerate capture makes *better*.
    """

    rms_reprojection_px: float = Field(
        ge=0.0,
        description=(
            "Root-mean-square distance between each detected corner and where "
            "the fitted model puts it. How well the model fits the data it was "
            "given -- necessary, and on its own not evidence that the data "
            "determined the model."
        ),
    )
    max_reprojection_px: float = Field(
        ge=0.0,
        description=(
            "Worst single corner residual. Separates a uniformly mediocre fit "
            "from a good one with a misdetected view in it, which call for "
            "different responses."
        ),
    )
    per_view_rms_px: list[float] = Field(
        default_factory=list,
        description="Residual per board view, in the order the views were used. Drawn by 8.4.",
    )
    coverage: CoverageReport
    degrees_of_freedom: int = Field(
        description=(
            "Corner observations times two, minus the parameters fitted. The "
            "same guard `SyncQuality.degrees_of_freedom` provides one layer "
            "down: a residual from a fit with nothing spare is not evidence."
        )
    )


class CameraCalibration(BaseModel):
    """One camera's calibration: what was measured, from what, and how well.

    `usable` and `refusal` are the gate. A calibration that exists is not a
    calibration that may be used, and the difference is decided here rather than
    by each consumer, so a consumer that forgets to check the coverage cannot
    silently use a set of views that never determined a focal length.
    """

    schema_version: int = CALIBRATION_SCHEMA_VERSION
    role: CameraRole
    intrinsics: CameraIntrinsics
    quality: CalibrationQuality
    detection: DetectionReport
    calibrated_at: datetime
    source: str = Field(description="What the board views came from, for provenance.")
    notes: str = Field(
        default="",
        description=(
            "Free text for the capture setting a video file cannot record: which "
            "lens, what zoom, whether stabilisation was off. The failure this "
            "field exists for is a correct calibration applied to footage shot "
            "at a different zoom, which nothing can detect."
        ),
    )
    usable: bool = Field(
        description="Whether this calibration passed the bounds in `CalibrationConfig`."
    )
    refusal: str | None = Field(
        default=None, description="Why it did not, in words. Set exactly when `usable` is false."
    )
    warnings: list[str] = Field(default_factory=list)


class PairingReport(BaseModel):
    """How the two cameras' board views were paired, and what that cost.

    The measurement that lets an unsynchronised pair be stereo-calibrated at
    all. `worst_pairing_error_px` is the time map's uncertainty converted into
    the unit it contaminates, by multiplying it by the board's observed image
    speed: a still board makes it nearly zero however badly the clocks are
    known, and a moving one makes it large however well they are. See the module
    docstring.
    """

    pairs: int = Field(description="Frame pairs that entered the stereo fit.")
    candidates: int = Field(description="Pairs considered before the bounds below dropped any.")
    method: str = Field(description="How frames were matched: 'time_map' or 'explicit'.")
    median_time_error_ms: float | None = Field(
        default=None,
        description=(
            "Median distance between the two frames of a pair once mapped onto "
            "one clock. None for explicit pairs, where the caller asserted "
            "simultaneity and nothing here can check it."
        ),
    )
    worst_pairing_error_px: float | None = Field(
        default=None,
        description=(
            "Largest board displacement, in pixels, attributable to the pairing "
            "being imperfect: the sync uncertainty there times the board's image "
            "speed there. Directly comparable with the reprojection error, "
            "because it is the same unit and it adds to it."
        ),
    )
    median_board_speed_px_s: float | None = Field(
        default=None,
        description=(
            "How fast the board was moving in the image across the used pairs. "
            "The number the capture instruction acts on: hold it still and this "
            "goes to zero, and with it the whole cost of not having a genlock."
        ),
    )
    dropped: list[str] = Field(
        default_factory=list, description="Why each rejected candidate was rejected."
    )


class StereoCalibration(BaseModel):
    """Where the second camera stands relative to the first, measured.

    Stated as the rotation and translation taking a point from the reference
    camera's frame into the target's. The baseline is reported separately
    because it is the one number in here a person can check with a tape measure,
    and a stereo calibration whose baseline is wrong by a factor of two is
    usually a board whose `square_length_m` was taken from the file rather than
    the print.
    """

    schema_version: int = CALIBRATION_SCHEMA_VERSION
    reference_role: CameraRole
    target_role: CameraRole
    rotation: list[list[float]] = Field(description="3x3 rotation, reference frame to target.")
    translation_m: list[float] = Field(description="3-vector, in metres, in the target's frame.")
    baseline_m: float = Field(
        gt=0.0, description="Distance between the two optical centres. Checkable with a tape."
    )
    convergence_deg: float = Field(
        description=(
            "Angle between the two optical axes. Near zero is a parallel rig; a "
            "face-on and a down-the-line camera are near 90, which is the best "
            "possible geometry for triangulation and the worst for finding "
            "anything visible in both."
        )
    )
    quality: CalibrationQuality
    pairing: PairingReport
    calibrated_at: datetime
    usable: bool
    refusal: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def rotation_matrix(self) -> NDArray[np.float64]:
        return np.array(self.rotation, dtype=np.float64)

    def translation_vector(self) -> NDArray[np.float64]:
        return np.array(self.translation_m, dtype=np.float64)


class CalibrationConfig(BaseModel, extra="forbid"):
    """Policy for calibration.

    The bounds are **stated policy informed by measurement**, not measured
    optima: `scripts/benchmark_calibration.py` establishes the shape of each
    relationship -- that focal uncertainty explodes as tilt range collapses,
    for instance -- and where on that curve to refuse is a judgement about what
    a wrong answer costs downstream. Where a number came from a sweep, the field
    says so.
    """

    distortion_model: DistortionModel = Field(
        default=DistortionModel.RADIAL_TANGENTIAL_4,
        description=(
            "Which coefficients to fit. Four rather than OpenCV's five, on the "
            "measurement: with the board reaching the frame edge the fifth term "
            "buys almost nothing, and without it the fifth term is fitted to "
            "noise and does its damage at the corners where the swing is. "
            "`scripts/benchmark_calibration.py --sweep distortion` has the "
            "numbers, and a fisheye lens is the case for overriding this."
        ),
    )
    min_views: int = Field(
        default=8,
        ge=3,
        description=(
            "Board views required before a calibration is attempted. Three is "
            "the algebraic minimum for a planar target and is nowhere near "
            "enough in practice; the usual advice of 15-20 is about coverage "
            "rather than count, which `CoverageReport` measures directly. This "
            "is a floor below which the coverage numbers themselves stop meaning "
            "much."
        ),
    )
    min_corners_per_view: int = Field(
        default=6,
        ge=4,
        description=(
            "Corners a view must contribute to be used. Four is the minimum for "
            "a pose; six leaves something over, and a view showing fewer is "
            "usually the board leaving the frame rather than a useful oblique."
        ),
    )
    min_image_fraction: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the frame the board must have visited. Below it the "
            "lens is unmeasured over most of its area, and the distortion "
            "coefficients are extrapolating into the region the subject is "
            "filmed in."
        ),
    )
    min_tilt_range_deg: float = Field(
        default=20.0,
        ge=0.0,
        description=(
            "Spread of board tilts required. **The bound that catches the "
            "failure neither the residual nor the reported uncertainty can "
            "see.** With every view square to the sensor, focal length and "
            "distance are very nearly interchangeable, and the fit is free to "
            "pick badly while fitting beautifully.\n\n"
            "Measured (`--sweep tilt`): at 0-2 degrees of spread the focal "
            "length comes out 6-10% wrong at an RMS of 0.21 px; at 20 degrees "
            "it is 0.10% wrong at an RMS of 0.25 px. Twenty is where the error "
            "collapses, and it is far more tilt than anyone holding a board "
            "produces by accident, which is the point."
        ),
    )
    max_focal_uncertainty_ratio: float = Field(
        default=0.02,
        gt=0.0,
        description=(
            "Standard deviation of the fitted focal length, as a fraction of it, "
            "above which the calibration is refused. Two percent of focal length "
            "is roughly two percent of every triangulated depth, which is a "
            "centimetre at half a metre and compounds through Phase 9.\n\n"
            "A backstop rather than the main instrument: it catches a fit that "
            "is visibly ill-conditioned and, as `fx_uncertainty` documents, "
            "misses the degenerate capture entirely. The coverage bounds are "
            "what catch that, and they are checked first."
        ),
    )
    max_rms_reprojection_px: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Residual above which the fit is refused outright. A pixel is a loose "
            "bound -- a good calibration lands well under half of one -- and it "
            "is deliberately loose, because this bound catches gross failure "
            "(a mis-specified board, a wrong dictionary) rather than mediocrity, "
            "which the uncertainty bound above is the instrument for."
        ),
    )
    max_focal_disagreement: float = Field(
        default=0.05,
        gt=0.0,
        description=(
            "How far fx and fy may differ, as a fraction of their mean. Consumer "
            "sensors have square pixels, so a real disagreement is evidence of a "
            "bad fit rather than of an unusual camera."
        ),
    )
    max_pairing_error_px: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Board displacement attributable to imperfect pairing, above which a "
            "stereo pair is dropped. One pixel, so that the sync error a pair "
            "contributes stays below the reprojection error it would otherwise "
            "be mistaken for. Holding the board still is what keeps pairs inside "
            "this; nothing else has to."
        ),
    )
    max_pair_time_error_ms: float = Field(
        default=50.0,
        gt=0.0,
        description=(
            "How far apart, on one clock, two frames of a pair may sit. A "
            "backstop for the bound above rather than the main instrument: a "
            "perfectly still board would satisfy the pixel test at any time "
            "error, and at some separation the assumption that nothing else in "
            "the scene moved stops being reasonable."
        ),
    )
    min_stereo_pairs: int = Field(
        default=6,
        ge=3,
        description=(
            "Paired board views required for stereo extrinsics. Fewer resolves "
            "the pose but leaves nothing over to check it with, and a stereo "
            "calibration with no spare degrees of freedom has a residual of zero "
            "by construction."
        ),
    )


class CameraRig(BaseModel):
    """Everything known about the geometry of one project's cameras.

    Stored whole, one per project, for the reason `ProjectSync` is stored whole:
    it is a nested document read entire and never queried by part, and the
    Pydantic model is already the authoritative definition of its shape.

    Keeping both cameras and their relation in one document rather than three
    rows is what makes `status` computable without a join, and `status` is the
    thing every consumer actually asks for.
    """

    schema_version: int = CALIBRATION_SCHEMA_VERSION
    cameras: dict[CameraRole, CameraCalibration] = Field(default_factory=dict)
    stereo: StereoCalibration | None = None
    config: CalibrationConfig = Field(default_factory=CalibrationConfig)

    def camera(self, role: CameraRole) -> CameraCalibration | None:
        """The calibration for one role, whether or not it is usable."""
        return self.cameras.get(role)

    def usable_camera(self, role: CameraRole) -> CameraCalibration | None:
        """The calibration for one role, only if it passed its bounds."""
        entry = self.cameras.get(role)
        return entry if entry is not None and entry.usable else None

    def status(self, role: CameraRole | None = None) -> CalibrationStatus:
        """What may be claimed, for one camera or for the rig as a whole.

        A rig is `STEREO` only when both named cameras are usable *and* their
        relative pose is, because triangulation needs all three and a partial
        set cannot do it at all. Asked about one role, the answer stops at
        `INTRINSICS`: one calibrated camera is one calibrated camera whatever
        the other one is doing.
        """
        if role is not None:
            return (
                CalibrationStatus.INTRINSICS
                if self.usable_camera(role) is not None
                else CalibrationStatus.NONE
            )

        if self.stereo is not None and self.stereo.usable:
            ends = (self.stereo.reference_role, self.stereo.target_role)
            if all(self.usable_camera(end) is not None for end in ends):
                return CalibrationStatus.STEREO

        if any(entry.usable for entry in self.cameras.values()):
            return CalibrationStatus.INTRINSICS
        return CalibrationStatus.NONE

    def status_for_frame(self, role: CameraRole, width: int, height: int) -> CalibrationStatus:
        """What may be claimed about a clip of this size from this camera.

        The check that stops a correct calibration being applied to the wrong
        footage. A calibration measured at one frame size does not describe
        another, and this is where that becomes a refusal rather than a silent
        rescale.
        """
        entry = self.usable_camera(role)
        if entry is None or not entry.intrinsics.applies_to(width, height):
            return CalibrationStatus.NONE
        return self.status(role)
