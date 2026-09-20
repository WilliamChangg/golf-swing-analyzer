"""Turn a reconstruction into something that can be drawn from any direction.

The three-dimensional counterpart of `analyzer/overlay.py`, and the differences
between the two are the interesting part.

`build_overlay` finishes the job: it converts filtered landmarks all the way into
the coordinates they are painted in, because a frame has exactly one camera and
the engine may as well own the arithmetic. `build_scene` cannot finish the job and
does not pretend to. There is no canonical view of a reconstructed swing, so the
last step -- the projection -- belongs to whatever is holding the mouse, and this
module's work is to hand that over everything it must not invent:

    metres            exactly as triangulated, in the reference camera's frame
    the cameras       where they really stood, with the focal lengths they had
    the covariance    the whole ellipsoid, not its longest radius
    the refusals      per point, by reason, so a hole in a skeleton can explain
                      itself

The covariance is the one that earns this module its existence.
`analyzer/contracts/scene.py` has the argument in full: a dot on a screen carries
no direction, so a viewpoint decides whether a joint's uncertainty is spread
across the picture or hidden behind the joint -- and the viewpoint almost nobody
changes is the default one. A `median_uncertainty_m` cannot express that. Six
numbers per point can.

## Nothing here is cached

Building a scene is a projection-free rearrangement of arrays the reconstruction
already produced, plus one eigendecomposition per point. That is milliseconds
against the seconds `extract_poses` costs and the tens of milliseconds
`reconstruct_pair` costs, so a cache would add a schema to version and invalidate
for no measurable gain -- the same measured reason Phase 3 gives and every layer
above it has inherited.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from analyzer.contracts.calibration import CalibrationStatus
from analyzer.contracts.pose import POSE_CONNECTIONS, Landmark
from analyzer.contracts.reconstruction import RefusalReason
from analyzer.contracts.scene import (
    PointUncertainty,
    ReconstructionScene,
    SceneCamera,
    SceneCameraKind,
    SceneFrame,
    ScenePoint,
    SceneTrajectory,
    Vec3,
)
from analyzer.reconstruction.triangulate import uncertainty_covariance

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from analyzer.contracts.cache import ContentKey
    from analyzer.contracts.calibration import CameraIntrinsics
    from analyzer.contracts.camera import CameraRole
    from analyzer.reconstruction.reconstruct import ReconstructedSequence
    from analyzer.reconstruction.triangulate import StereoGeometry

# The largest range one call will serialise.
#
# Lower than `MAX_OVERLAY_FRAMES` on measurement rather than on symmetry: a scene
# point carries a position, six covariance elements and three diagnostics against
# an overlay point's two coordinates, so the same frame is roughly four times the
# bytes. `scripts/benchmark_viewport.py --sweep payload` reports what a frame
# actually costs; this bound is where a whole-clip request stays inside a
# response a local pipe moves and a WebView parses without a visible pause.
MAX_SCENE_FRAMES = 600

# Whose path is stroked through the scene unless a caller says otherwise.
#
# The wrists, because the hands are what every metric in this engine is anchored
# to and what a person watching a swing is actually following. Not the club: a
# shaft is tracked in one image (Phase 10) and never triangulated, so a 3D club
# path would be a line this pipeline has never computed.
DEFAULT_TRAJECTORIES: tuple[Landmark, ...] = (Landmark.LEFT_WRIST, Landmark.RIGHT_WRIST)


class SceneError(ValueError):
    """A scene was asked for that cannot be produced as asked."""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


def _vec(values: NDArray[np.float64]) -> Vec3:
    return Vec3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _fov_deg(intrinsics: CameraIntrinsics) -> float:
    """Horizontal field of view implied by the focal length and the image width."""
    return float(2.0 * np.degrees(np.arctan2(intrinsics.image_width / 2.0, intrinsics.fx)))


def _cameras(
    geometry: StereoGeometry,
    *,
    reference_role: CameraRole,
    target_role: CameraRole,
    reference_name: str,
    target_name: str,
) -> list[SceneCamera]:
    """Both cameras, placed in the frame the points are already in.

    The reference camera is at the origin looking down +z, which is not a
    convention chosen here but the one `StereoGeometry` is built around -- it is
    what makes a triangulated point a `CAMERA` coordinate by construction.

    The target's axes come out of `rotation`, which carries a point **from** the
    reference frame **into** the target's. A vector that reads `e_i` in the target
    frame therefore reads `R' e_i` in the reference frame, which is row `i` of
    `R`. `up` negates the second of those because image y increases downward, so
    a camera's own +y axis points at the bottom of its picture.
    """
    rotation = geometry.rotation
    return [
        SceneCamera(
            kind=SceneCameraKind.REFERENCE,
            role=reference_role,
            name=reference_name,
            position=Vec3(x=0.0, y=0.0, z=0.0),
            forward=Vec3(x=0.0, y=0.0, z=1.0),
            up=Vec3(x=0.0, y=-1.0, z=0.0),
            right=Vec3(x=1.0, y=0.0, z=0.0),
            fx=geometry.reference.fx,
            fy=geometry.reference.fy,
            cx=geometry.reference.cx,
            cy=geometry.reference.cy,
            image_width=geometry.reference.image_width,
            image_height=geometry.reference.image_height,
            horizontal_fov_deg=_fov_deg(geometry.reference),
        ),
        SceneCamera(
            kind=SceneCameraKind.TARGET,
            role=target_role,
            name=target_name,
            position=_vec(geometry.target_centre),
            forward=_vec(rotation[2]),
            up=_vec(-rotation[1]),
            right=_vec(rotation[0]),
            fx=geometry.target.fx,
            fy=geometry.target.fy,
            cx=geometry.target.cx,
            cy=geometry.target.cy,
            image_width=geometry.target.image_width,
            image_height=geometry.target.image_height,
            horizontal_fov_deg=_fov_deg(geometry.target),
        ),
    ]


def _framing(points: NDArray[np.float64], valid: NDArray[np.bool_]) -> tuple[Vec3, float]:
    """Where to orbit and how far back, measured from the body rather than chosen.

    The median position and the 95th percentile of the distance from it. Both
    robust, and for a reason this project has met before: a maximum over a clip is
    a maximum over its noise, so one badly conditioned hand at the edge of the
    overlap would otherwise set the framing for the whole swing and open the
    viewport looking at a speck.

    Falls back to a metre at the origin when nothing was reconstructed, which is
    a scene with no body in it -- the viewport still has to have somewhere to
    stand, and `radius_m` is constrained positive so it cannot be zero.
    """
    kept = points[valid]
    if kept.size == 0:
        return Vec3(x=0.0, y=0.0, z=0.0), 1.0

    centre = np.median(kept, axis=0)
    spread = float(np.percentile(np.linalg.norm(kept - centre, axis=1), 95.0))
    return _vec(centre), max(spread, 1e-3)


def _uncertainty(covariance: NDArray[np.float64], sigma_m: float) -> PointUncertainty | None:
    """One point's ellipsoid, with the worst-direction sigma the sequence measured.

    `sigma_m` is taken from `ReconstructedSequence.uncertainty_m` rather than
    recomputed from `covariance`, so the number a viewport prints is the same
    number the gate refused on and the report summarised. They are the same
    quantity by construction -- `sqrt` of this matrix's largest eigenvalue -- and
    `test_scene.py` asserts it, because "by construction" across two functions is
    a claim and not a guarantee.
    """
    if not np.isfinite(covariance).all() or not np.isfinite(sigma_m):
        return None
    return PointUncertainty(
        sigma_m=float(max(sigma_m, 0.0)),
        xx=float(covariance[0, 0]),
        yy=float(covariance[1, 1]),
        zz=float(covariance[2, 2]),
        xy=float(covariance[0, 1]),
        xz=float(covariance[0, 2]),
        yz=float(covariance[1, 2]),
    )


def _refusal(reason: np.str_) -> RefusalReason:
    """The reason a point produced nothing, as the gates recorded it.

    An empty or unrecognised string becomes `NOT_SEEN`, which is the weakest of
    the five and the only one that claims nothing about the capture: a viewport
    drawing a hole should not be told the cameras were badly placed on the
    strength of a reason nothing recorded. A produced point never reaches here.
    """
    try:
        return RefusalReason(str(reason))
    except ValueError:
        return RefusalReason.NOT_SEEN


def build_scene(
    reconstruction: ReconstructedSequence,
    *,
    reference_content_key: ContentKey,
    start_frame: int = 0,
    end_frame: int | None = None,
    trajectories: tuple[Landmark, ...] | None = None,
) -> ReconstructionScene:
    """A reconstructed swing over `[start_frame, end_frame)`, ready to be drawn.

    Frame indices are the **reference clip's**, and both ends are clamped to the
    reconstruction rather than refused, for `build_overlay`'s reason: a viewport
    asking for a window around the last frame is asking a reasonable question.

    The covariances are recomputed here from the same points, the same geometry
    and the same measured pixel sigma the reconstruction used, rather than being
    carried on `ReconstructedSequence`. That keeps a `(frames, landmarks, 3, 3)`
    array out of every reconstruction that is never drawn -- nine times the
    points themselves -- and it is not a second opinion: it is one call to the
    one function, over the range that was asked for.
    """
    total = len(reconstruction)
    start = max(0, min(start_frame, total))
    end = max(start, min(end_frame if end_frame is not None else total, total))
    span = end - start

    if span > MAX_SCENE_FRAMES:
        raise SceneError(
            f"A scene of {span} frames was asked for; the limit is {MAX_SCENE_FRAMES}.",
            remediation=(
                f"Ask for at most {MAX_SCENE_FRAMES} frames at a time. A viewport draws one "
                "frame; the range exists to avoid a request per frame, not to move the clip."
            ),
        )

    report = reconstruction.report
    sigma_px = report.quality.pixel_sigma_px if report.quality is not None else 0.0

    selected = reconstruction.landmarks
    width = len(selected)
    points = reconstruction.points[start:end]
    valid = reconstruction.valid[start:end]

    # One flattened call, as the reconstruction itself does: an eigendecomposition
    # of a 3x3 is cheap and a Python loop around thirty-three of them per frame is
    # not, and numpy batches the whole range in one pass either way.
    covariance = uncertainty_covariance(
        points.reshape(-1, 3), reconstruction.geometry, sigma_px
    ).reshape(span, width, 3, 3)

    frames: list[SceneFrame] = []
    for offset in range(span):
        index = start + offset
        entries: list[ScenePoint] = []
        for column, landmark in enumerate(selected):
            drawn = bool(valid[offset, column])
            entries.append(
                ScenePoint(
                    landmark=landmark,
                    position=_vec(points[offset, column]) if drawn else None,
                    refused=(None if drawn else _refusal(reconstruction.refusal[index, column])),
                    uncertainty=(
                        _uncertainty(
                            covariance[offset, column],
                            float(reconstruction.uncertainty_m[index, column]),
                        )
                        if drawn
                        else None
                    ),
                    convergence_deg=(
                        _finite(reconstruction.convergence_deg[index, column]) if drawn else None
                    ),
                    reprojection_px=(
                        _finite(reconstruction.reprojection_px[index, column]) if drawn else None
                    ),
                    visibility=_visibility(reconstruction.visibility[index, column]),
                )
            )
        frames.append(
            SceneFrame(
                frame_index=index,
                timestamp_s=float(reconstruction.t[index]),
                points=entries,
                reconstructed=int(np.count_nonzero(valid[offset])),
            )
        )

    wanted = tuple(trajectories) if trajectories is not None else DEFAULT_TRAJECTORIES
    index_of = reconstruction.index
    paths: list[SceneTrajectory] = []
    for landmark in wanted:
        track = index_of.get(landmark)
        if track is None:
            continue
        series = [
            _vec(points[offset, track]) if bool(valid[offset, track]) else None
            for offset in range(span)
        ]
        paths.append(
            SceneTrajectory(
                landmark=landmark,
                name=landmark.name.lower(),
                points=series,
                reconstructed=sum(1 for entry in series if entry is not None),
            )
        )

    centroid, radius = _framing(points.reshape(-1, 3), valid.reshape(-1))
    requested = set(selected)

    return ReconstructionScene(
        reference_content_key=reference_content_key,
        reference_name=report.reference_name,
        target_name=report.target_name,
        reference_role=report.reference_role,
        target_role=report.target_role,
        start_frame=start,
        end_frame=end,
        frames=frames,
        landmarks=list(selected),
        connections=[pair for pair in POSE_CONNECTIONS if set(pair) <= requested],
        trajectories=paths,
        cameras=_cameras(
            reconstruction.geometry,
            reference_role=report.reference_role,
            target_role=report.target_role,
            reference_name=report.reference_name,
            target_name=report.target_name,
        ),
        calibration=CalibrationStatus.STEREO,
        centroid=centroid,
        radius_m=radius,
        slow_motion_factor=reconstruction.slow_motion_factor,
        pixel_sigma_px=sigma_px,
        report=report,
    )


def _finite(value: np.float64) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _visibility(reported: np.float64) -> float:
    """The estimator's reported visibility as a number the contract can carry.

    NaN becomes zero, matching `analyzer/overlay.py`: a frame with no detection
    is a frame where the landmark was not seen, so zero is the reading rather
    than a substitution, and it always arrives beside a `refused` that says the
    same thing a second way.
    """
    value = float(reported)
    return 0.0 if not np.isfinite(value) else float(np.clip(value, 0.0, 1.0))
