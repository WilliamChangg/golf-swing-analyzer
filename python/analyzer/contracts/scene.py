"""Typed contracts for a reconstruction that is going to be looked at.

Phase 9 produced metres and reported what they were worth. It deliberately drew
nothing -- "a reconstruction is checked by its own bones and by the numbers
`analyzer reconstruct` prints", and the viewport "should render something this
phase has already validated rather than being the thing that validates it". This
module is the other half of that sentence: the same measurement, shaped for a
picture.

## A drawing is made from a viewpoint, and that is the whole problem

`PoseOverlay` had no such difficulty. A landmark on a frame is drawn in exactly
one place, because the camera that took the frame is the camera that has to be
drawn from, and the engine could therefore hand the UI finished coordinates and
keep every convention to itself.

Three dimensions removes that. **There is no canonical view of a reconstructed
swing**, so the viewpoint becomes a choice a person makes with a mouse, sixty
times a second, and the projection has to live where the mouse is. What the
engine can still do -- and what this contract exists to do -- is hand over the
things the projection must not be allowed to invent:

* **Metres, exactly as reconstructed.** `LandmarkSpace.CAMERA`, centred on the
  reference camera, not rescaled into any renderer's box. A viewport that wants a
  unit cube may compute one; nothing here has pre-divided by a number the
  consumer would then have to un-divide to state a length.
* **The cameras themselves**, with the intrinsics they actually had. This is the
  check that makes the rest trustworthy: place the viewport at the reference
  camera, project with its `fx, fy, cx, cy`, and the result must land on the
  pixels `PoseOverlay` draws, because those are two renderings of one
  measurement. `tests/test_scene.py` asserts exactly that, and it is the reason a
  hand-written projection is safe to ship.
* **The uncertainty as an ellipsoid, not a radius.** See below.

## The number this phase is built around

A 3D viewport draws a joint as a dot, and a dot has no direction. Everything the
reconstruction does not know about that joint therefore lands in one of two
places: spread across the picture where a reader can see it, or pointing straight
down the line of sight, where it hides behind the dot and the picture looks
exactly as confident as a perfect one would.

Which of those happens depends only on where the viewer put the camera -- and the
**default** camera is the one almost nobody moves. So a viewport that opens
looking from the reference camera is a viewport that opens with its errors
pointing away from the reader, and every honest number Phase 9 computed is on
screen as a body that looks solidly located.

`PointUncertainty` carries the whole covariance rather than
`ReconstructionQuality.median_uncertainty_m`'s single worst-direction figure,
because the single figure cannot be drawn: it is a radius, a radius is a sphere,
and the thing being described is not a sphere. How far from one it is depends on
the capture, and it is exactly the capture failure that is hardest to see:

    convergence 90 deg  ->  nearly isotropic; every viewpoint shows the error
    convergence 15 deg  ->  a long axis along the shared depth direction, which
                            is roughly where the reference camera is looking

so the flattery is worst precisely where the reconstruction is worst.
`reconstruction.triangulate.visible_uncertainty_fraction` turns a viewpoint into
the fraction of that ellipsoid it shows, `scripts/benchmark_viewport.py` measures
it against convergence angle, and a viewport is expected to report it rather than
leave a reader to assume a dot means a known position.

## What is indexed by what

`frames` carry the **reference clip's** frame indices and its real-clock
timestamps, because that is what the reconstruction is parameterised in and what
every swing event was located in. A consumer holding a frame number from Phase 4
indexes straight in.

It is also why `reference_content_key` is here. The scene is scrubbed against a
video element playing a file, and nothing about a 3D scene makes it obvious which
file that should be -- so the identity of the clip the frame numbers belong to
travels with them, and a viewport handed a different clip can refuse instead of
showing a body from one recording against the pixels of another.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from analyzer.contracts.cache import ContentKey
from analyzer.contracts.calibration import CalibrationStatus
from analyzer.contracts.camera import CameraRole
from analyzer.contracts.pose import Landmark, LandmarkSpace
from analyzer.contracts.reconstruction import ReconstructionReport, RefusalReason

# Bump on any change that alters what a drawn scene means.
SCENE_SCHEMA_VERSION = 1


class Vec3(BaseModel):
    """A point or a direction in `LandmarkSpace.CAMERA` metres.

    Named components rather than a three-element tuple, which JSON Schema would
    express as `prefixItems` and TypeScript would receive as a positional array.
    A consumer assembling a basis from `[0]`, `[1]`, `[2]` can transpose it
    silently; one reading `.x` cannot. This engine has already paid for a
    transposed convention once -- `tests/synthetic_body3d.look_at` built two
    cameras upside down and mirrored, and triangulation could not see it.
    """

    x: float
    y: float
    z: float


class SceneCameraKind(StrEnum):
    """What a camera in the scene is.

    REFERENCE
        The camera the coordinates are centred on. It sits at the origin looking
        down +z by construction, and its intrinsics are what make the
        "look from here" view reproducible against `PoseOverlay`.
    TARGET
        The second view, at its measured position and attitude in the reference
        camera's frame. Drawn because **where these two stood is the single
        largest thing determining what the reconstruction is worth**, and a
        reader who can see the angle between them can see the conditioning.
    """

    REFERENCE = "reference"
    TARGET = "target"


class SceneCamera(BaseModel):
    """One of the two real cameras, placed in the scene it produced.

    The attitude is three named unit vectors rather than a rotation matrix, for
    `Vec3`'s reason one level up: a matrix has a convention, a convention can be
    transposed, and a transposed one here produces a plausible picture of a rig
    that never existed. `forward` is the optical axis, pointing the way the camera
    looks; `up` points along the image's -y, since image y increases downward.
    """

    kind: SceneCameraKind
    role: CameraRole = Field(description="The camera position declared for this clip.")
    name: str = Field(description="Basename of the clip this camera filmed, for display.")

    position: Vec3 = Field(description="Optical centre, in scene metres. Zero for the reference.")
    forward: Vec3 = Field(description="Unit optical axis. `(0, 0, 1)` for the reference.")
    up: Vec3 = Field(description="Unit vector along the image's -y, so up on screen.")
    right: Vec3 = Field(description="Unit vector along the image's +x.")

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    horizontal_fov_deg: float = Field(
        description=(
            "Field of view implied by `fx` and the image width. Carried so a "
            "viewport can adopt this camera's framing without re-deriving an "
            "arctangent, and so a frustum can be drawn at the right shape."
        )
    )


class PointUncertainty(BaseModel):
    """Where a reconstructed joint could be, as the ellipsoid it really is.

    The six unique elements of the `3x3` covariance in scene metres squared, plus
    the worst-direction standard deviation `ReconstructionQuality` already
    reports, so the two cannot drift apart in a consumer that needs both.

    **Six numbers rather than a radius** because the ellipsoid is not a sphere and
    the departure from one is the finding. A stereo pair localises a point well
    across both rays and poorly along their bisector; at the right-angled capture
    the protocol asks for those are comparable, and as the two cameras move
    together the bisector axis grows without bound while the other two do not.
    Drawing `sigma_m` as a circle would report the worst axis in every direction,
    which overstates two of them and -- worse -- makes every viewpoint look
    equally informative.

    A consumer rebuilds the matrix as

        [[xx, xy, xz],
         [xy, yy, yz],
         [xz, yz, zz]]

    and projects it with `A S A'` for the `2x3` Jacobian `A` of its own
    projection, which is the ellipse to draw. It is a first-order statement, like
    every propagated uncertainty in this engine.
    """

    sigma_m: float = Field(
        ge=0.0,
        description=(
            "Standard deviation along the worst-determined direction, in metres. "
            "The square root of the covariance's largest eigenvalue, and the same "
            "quantity `LandmarkReconstruction.median_uncertainty_m` summarises."
        ),
    )
    xx: float
    yy: float
    zz: float
    xy: float
    xz: float
    yz: float


class ScenePoint(BaseModel):
    """One landmark at one instant, in metres or refused.

    `position` and `uncertainty` are null exactly when `refused` is set, and
    `refused` is set exactly when they are null -- the same invariant
    `OverlayPoint` holds between `state` and its coordinates, for the same reason:
    JSON has no NaN, so an absence that is real has to be carried in the type
    where a consumer must handle it.

    **`refused` is the reason, not a flag.** Phase 9 counts its five refusal
    reasons per landmark rather than as one "failed" tally because the fixes are
    unrelated -- move a camera, re-film the occlusion, re-synchronise, or accept
    that the subject was too far away. A viewport that draws a hole in a skeleton
    should be able to say which of those it is looking at, and a boolean could
    not.
    """

    landmark: Landmark
    position: Vec3 | None = Field(description="Scene metres. Null exactly when `refused` is set.")
    refused: RefusalReason | None = Field(
        default=None,
        description="Why this landmark produced no point here. Null when it did produce one.",
    )
    uncertainty: PointUncertainty | None = None
    convergence_deg: float | None = Field(
        default=None,
        description=(
            "Angle the two rays met at, at this point. The conditioning of this "
            "particular intersection, which varies across a frame -- a hand out "
            "in front of the body converges differently from a hip."
        ),
    )
    reprojection_px: float | None = Field(
        default=None,
        description=(
            "RMS distance over both views between this point's projection and "
            "where each camera saw the landmark. **Not the accuracy**, for the "
            "reason `analyzer/contracts/reconstruction.py` opens with."
        ),
    )
    visibility: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "The weaker of the two views' reported visibilities. The estimator's "
            "own opinion of itself, shown and never trusted -- this project has "
            "measured it reporting 1.00 across frames whose implied shoulder turn "
            "varied by 27 degrees."
        ),
    )


class SceneFrame(BaseModel):
    """Everything drawn at one instant of the swing."""

    frame_index: int = Field(description="The **reference clip's** frame number.")
    timestamp_s: float = Field(
        description=(
            "Real-clock seconds from the reference clip's first frame, after any "
            "slow-motion factor has been divided out. Not where the frame sits in "
            "the video file, which is what a player seeks by -- see `SeekIndex`."
        )
    )
    points: list[ScenePoint] = Field(
        description="One entry per landmark asked for, in ascending landmark order."
    )
    reconstructed: int = Field(
        ge=0,
        description=(
            "How many of them produced a position. Zero is a real frame with "
            "nothing in it, and a viewport should draw nothing rather than hold "
            "the previous pose -- which would put a body on screen in a position "
            "it was never measured in."
        ),
    )


class SceneTrajectory(BaseModel):
    """One landmark's path through the clip, ready to be stroked as a polyline.

    Sent as its own series rather than left to a consumer to gather out of
    `frames`, because the gathering has a trap in it: a refused instant is a
    **break** in the path, not a point to interpolate across. A consumer that
    filtered the nulls out and joined what was left would draw a straight line
    through the gap -- exactly the failure `OverlayState.BLOCKED` exists to
    prevent one dimension down, where Phase 4 found the wrists untracked for
    1.92 s of a real clip precisely when they were moving fastest.
    """

    landmark: Landmark
    name: str
    points: list[Vec3 | None] = Field(
        description=(
            "One entry per frame in the scene's range, in order. Null where the "
            "landmark was refused -- the polyline breaks there and resumes after."
        )
    )
    reconstructed: int = Field(ge=0, description="How many of them are non-null.")


class ReconstructionScene(BaseModel):
    """A reconstructed swing, everything needed to draw it, and what it is worth.

    A frame range rather than a whole clip, for `PoseOverlay`'s reason and more
    strongly: a scene point carries a position, a covariance and three diagnostics
    where an overlay point carried two coordinates, so the same number of frames
    is several times the payload. `MAX_SCENE_FRAMES` in `analyzer/scene.py` is the
    bound, and a request past it is refused by name rather than truncated.

    `report` is embedded whole rather than fetched separately. The viewport has to
    state the calibration status and the conditioning next to the picture --
    otherwise the picture is a claim with its qualifications on another screen --
    and a second call would let the two disagree about which reconstruction is on
    screen.
    """

    schema_version: int = SCENE_SCHEMA_VERSION
    space: LandmarkSpace = Field(
        default=LandmarkSpace.CAMERA,
        description=(
            "Metres, centred on the reference camera. **Not WORLD**: there is no "
            "measured gravity direction and no target line, so nothing here may "
            "be drawn as a ground plane or labelled 'towards the target'. A "
            "viewport that draws a horizon is drawing an assumption."
        ),
    )

    reference_content_key: ContentKey = Field(
        description=(
            "Identity of the clip whose frame numbers `frames` carry. A viewport "
            "scrubbing against a video element compares this with the clip it is "
            "playing, and refuses rather than pairing one recording's pixels with "
            "another's reconstruction."
        )
    )
    reference_name: str
    target_name: str
    reference_role: CameraRole
    target_role: CameraRole

    start_frame: int
    end_frame: int = Field(description="Exclusive, as every frame range in this project is.")
    frames: list[SceneFrame]
    landmarks: list[Landmark] = Field(
        description="Which landmarks each frame carries, in the order they appear."
    )
    connections: list[tuple[Landmark, Landmark]] = Field(
        description=(
            "Skeleton edges to draw, from `POSE_CONNECTIONS` -- the same tuple "
            "Phase 9's bone-length checks read and `PoseOverlay` draws, so the app "
            "cannot grow a second opinion about human anatomy. Edges whose "
            "endpoints were not both requested are omitted."
        )
    )
    trajectories: list[SceneTrajectory] = Field(default_factory=list)
    cameras: list[SceneCamera] = Field(
        description="The reference camera first, then the target. Both are real placements."
    )

    calibration: CalibrationStatus = Field(
        description=(
            "What was known about the two cameras' geometry. Always `stereo` for a "
            "scene that exists, and carried so a viewport states it rather than "
            "letting three dimensions on screen imply it."
        )
    )
    centroid: Vec3 = Field(
        description=(
            "Median position of every reconstructed point in the range. What a "
            "viewport orbits around -- **measured from the body rather than "
            "chosen**, so a subject filmed from four metres and one from two both "
            "open framed."
        )
    )
    radius_m: float = Field(
        gt=0.0,
        description=(
            "Distance from `centroid` containing 95% of the reconstructed points. "
            "The scale to frame by, robust so that one badly triangulated hand "
            "cannot push the whole swing into the distance."
        ),
    )
    slow_motion_factor: float = Field(
        description="The factor the reference clip's timestamps were divided by."
    )
    pixel_sigma_px: float = Field(
        description=(
            "The per-view landmark scatter every covariance here was propagated "
            "from, in pixels. Measured from the clip where the filter's residual "
            "supplied one; `ReconstructionQuality.methodology` says which."
        )
    )

    report: ReconstructionReport
    warnings: list[str] = Field(default_factory=list)
