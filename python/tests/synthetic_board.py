"""A camera with known intrinsics, photographing a board at known poses.

Ground truth for Phase 8, in the only sense available without an optical bench:
the camera is an input. Every intrinsic the fit is asked to recover is a number
put in here, so "the calibration is accurate" becomes a comparison rather than a
judgement, and the benchmark can sweep the capture geometry and watch what the
error does.

## It renders an image rather than projecting points

The cheap version of this file would project the board's corners through the
camera model and hand the resulting pixel coordinates to the calibrator. That
would test the fit and nothing else, and it would test it against the very model
it is inverting -- a closed loop that cannot fail for any reason a real capture
fails for.

So instead a real image is rendered and the **real detector** runs on it. The
rendering is an inverse map, which is what makes it exact: for every output
pixel, undistort it into a ray, intersect the ray with the board's plane, and
read the board's printed texture there. Forward-projecting the texture would
leave holes and need interpolation to fill them; going backwards, every output
pixel has exactly one source and the distortion is applied at full resolution
rather than approximated by warping a mesh.

What this still does not contain is everything that makes real calibration
footage hard: motion blur, rolling shutter, defocus, JPEG ringing on the marker
borders, and a printed sheet that is not quite flat. The corner detections here
are therefore better than a real capture's, and the errors measured against this
are a floor rather than an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.calibration.board import board_texture, build, object_points
from analyzer.contracts.calibration import BoardSpec, CameraIntrinsics, DistortionModel

# A board that fits on A4 with a sensible margin: 7x5 squares of 35 mm is
# 245x175 mm. 24 interior corners is plenty for a single view to be
# well-determined.
DEFAULT_BOARD = BoardSpec(
    squares_x=7,
    squares_y=5,
    square_length_m=0.035,
    marker_length_m=0.026,
)


def default_camera(
    width: int = 1920,
    height: int = 1080,
    *,
    fx: float = 1400.0,
    fy: float | None = None,
    distortion: tuple[float, ...] = (-0.28, 0.12, 0.0006, -0.0004),
) -> CameraIntrinsics:
    """A plausible phone camera: about 69 degrees across, with barrel distortion.

    The defaults are chosen to be *typical rather than convenient*. fx = 1400 px
    on a 1920-wide frame is a 69 degree horizontal field of view, which is what a
    phone's main camera sees; k1 = -0.28 is ordinary barrel distortion for such a
    lens, and it displaces a corner pixel by tens of pixels, which is the whole
    reason undistortion is worth doing.
    """
    return CameraIntrinsics(
        fx=fx,
        fy=fy if fy is not None else fx,
        cx=width / 2.0 - 3.0,
        cy=height / 2.0 + 4.0,
        distortion=list(distortion),
        model=(
            DistortionModel.RADIAL_TANGENTIAL_4
            if len(distortion) == 4
            else DistortionModel.RADIAL_TANGENTIAL_5
        ),
        image_width=width,
        image_height=height,
    )


@dataclass(frozen=True)
class BoardPose:
    """Where the board sat, relative to the camera.

    `rotation_deg` is applied as a rotation vector in the camera's frame and
    `distance_m` places the board along the optical axis. `offset_m` slides it
    sideways and vertically, which is how a view is put in the corner of the
    frame -- the views that carry the distortion information.
    """

    distance_m: float = 0.6
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_m: tuple[float, float] = (0.0, 0.0)

    def rvec_tvec(self, spec: BoardSpec) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """OpenCV's rotation vector and translation for this pose.

        The board is centred on its own middle before the pose is applied, so a
        rotation tilts it about its centre rather than swinging it away about a
        corner. Without that, "tilt by 30 degrees" would also translate the board
        by most of its own width and change the framing at the same time.
        """
        rvec = np.radians(np.array(self.rotation_deg, dtype=np.float64)).reshape(3, 1)
        rotation, _ = cv2.Rodrigues(rvec)
        centre = np.array([spec.width_m / 2.0, spec.height_m / 2.0, 0.0], dtype=np.float64)
        tvec = (
            np.array([self.offset_m[0], self.offset_m[1], self.distance_m], dtype=np.float64)
            - rotation @ centre
        ).reshape(3, 1)
        return rvec, tvec


class SyntheticCamera:
    """Renders views of a board through a known camera model."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        spec: BoardSpec = DEFAULT_BOARD,
        *,
        texture_px: int = 1400,
        background: int = 110,
        supersample: int = 1,
    ) -> None:
        self.intrinsics = intrinsics or default_camera()
        self.spec = spec
        self.background = background
        self.supersample = max(1, int(supersample))
        # The printed sheet, inside a white margin. The margin is not cosmetic:
        # OpenCV refuses to render some board sizes without one, and a real
        # printed board has one anyway -- the outermost markers need white
        # around them to be found at all. It is tracked in pixels so the texture
        # lookup can subtract it and stay a pure scale over the board proper.
        self._margin_px = 24
        self.texture, self._board_px = board_texture(
            spec,
            board_px=(
                int(texture_px),
                round(texture_px * spec.height_m / spec.width_m),
            ),
            margin_px=self._margin_px,
        )

    def render(self, pose: BoardPose) -> NDArray[np.uint8]:
        """One image of the board at a pose, with distortion applied exactly.

        The inverse map: every output pixel is undistorted into a ray, the ray is
        intersected with the board's plane, and the board's texture is sampled
        there. Pixels whose ray misses the board, or hits the plane behind the
        camera, take the background.

        `supersample` renders at a multiple of the output size and averages
        down, which is what a real sensor does to the light falling on a pixel.
        It defaults to 1, **on the measurement, and the measurement is a small
        instance of this phase's whole point**: supersampling by 2 lowers the
        fitted reprojection residual from 0.236 px to 0.168 px and moves the
        recovered focal length *further* from the truth, from -0.02% to -0.15%.
        A better residual, a worse calibration. It costs 3.5x the render time
        for that, so the fixture stays at 1 and the option remains for anyone
        who wants to check that a result is not an artefact of aliasing.
        """
        return self.render_pose(*pose.rvec_tvec(self.spec))

    def render_pose(
        self, rvec: NDArray[np.float64], tvec: NDArray[np.float64]
    ) -> NDArray[np.uint8]:
        """Render the board at an explicit OpenCV pose.

        The form the stereo rig needs: the second camera's view of a board is
        the first camera's pose for it, carried through the rig's relative pose,
        and that composition does not go back through `BoardPose`'s convenience
        parameterisation.
        """
        output = self._render_at(rvec, tvec, self.supersample)
        if self.supersample == 1:
            return output
        return np.asarray(
            cv2.resize(
                output,
                (self.intrinsics.image_width, self.intrinsics.image_height),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.uint8,
        )

    def _render_at(
        self, rvec: NDArray[np.float64], tvec: NDArray[np.float64], factor: int
    ) -> NDArray[np.uint8]:
        """The inverse map, at `factor` times the camera's own resolution.

        Scaling the intrinsics rather than the output image is what keeps the
        supersampled render a render of the *same camera*: fx, fy, cx and cy all
        scale with the sampling grid, so a pixel at 3x sits exactly where a third
        of a pixel sits at 1x. Distortion coefficients are in normalised
        coordinates and do not scale, which is the reason this works at all.
        """
        width = self.intrinsics.image_width * factor
        height = self.intrinsics.image_height * factor
        matrix = self.intrinsics.matrix()
        # Scaling a camera is not multiplying every entry by the factor. A pixel
        # centre at integer u covers [u - 0.5, u + 0.5], so preserving the
        # fraction of the frame a point sits at gives u' = f*u + (f - 1)/2: the
        # focal lengths scale and the principal point scales *and shifts*.
        # Without the half-pixel term the render is displaced by (f-1)/2f of an
        # output pixel, which is a systematic bias in every corner it produces.
        matrix[0, 0] *= factor
        matrix[1, 1] *= factor
        matrix[0, 2] = matrix[0, 2] * factor + (factor - 1) / 2.0
        matrix[1, 2] = matrix[1, 2] * factor + (factor - 1) / 2.0
        distortion = self.intrinsics.distortion_vector()
        rotation, _ = cv2.Rodrigues(rvec)
        translation = tvec.reshape(3)

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)

        # Distorted pixel -> normalised ray on the z = 1 plane.
        normalised = cv2.undistortPoints(pixels, matrix, distortion).reshape(-1, 2)
        rays = np.concatenate(
            (normalised, np.ones((normalised.shape[0], 1), dtype=np.float64)), axis=1
        )

        # Intersect each ray with the board plane: n . (s*d - t) = 0.
        normal = rotation @ np.array([0.0, 0.0, 1.0])
        denominator = rays @ normal
        numerator = float(normal @ translation)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = numerator / denominator
        points = rays * scale[:, None]

        # Camera frame -> board frame, and then to texture pixels. The margin
        # offsets the board rectangle inside the texture.
        board_points = (points - translation) @ rotation
        map_x = self._margin_px + (board_points[:, 0] / self.spec.width_m) * self._board_px[0]
        map_y = self._margin_px + (board_points[:, 1] / self.spec.height_m) * self._board_px[1]

        valid = np.isfinite(map_x) & np.isfinite(map_y) & (scale > 0)
        map_x = np.where(valid, map_x, -1.0).astype(np.float32).reshape(height, width)
        map_y = np.where(valid, map_y, -1.0).astype(np.float32).reshape(height, width)

        return cv2.remap(
            self.texture,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.background,
        )

    def project_corners(self, pose: BoardPose) -> NDArray[np.float64]:
        """Where the board's corners land in the image, from the model directly.

        The check on the renderer: a detector run over `render(pose)` should
        find corners at these positions, and a test asserts it does. If the
        texture mapping had a transposed or flipped axis the rendering would
        still look like a board and the two would disagree, which is exactly the
        kind of error a rendered fixture can otherwise hide.
        """
        return self.project_corners_pose(*pose.rvec_tvec(self.spec))

    def project_corners_pose(
        self, rvec: NDArray[np.float64], tvec: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Where the board's corners land, for an explicit pose."""
        projected, _ = cv2.projectPoints(
            object_points(self.spec),
            rvec,
            tvec,
            self.intrinsics.matrix(),
            self.intrinsics.distortion_vector(),
        )
        return np.asarray(projected, dtype=np.float64).reshape(-1, 2)

    def board_object(self) -> cv2.aruco.CharucoBoard:
        return build(self.spec)


def poses_at(
    targets: list[tuple[float, float]],
    intrinsics: CameraIntrinsics,
    *,
    tilt_deg: float = 35.0,
    distances_m: tuple[float, float] = (0.45, 0.95),
    seed: int = 7,
) -> list[BoardPose]:
    """Poses that put the board at named places in the frame.

    `targets` are normalised image positions, (0, 0) being the centre and
    (-1, -1) the top-left corner. Aiming in image space rather than in metres is
    what makes a capture reproducible across camera models: a 30 mm sideways
    offset fills the frame on a long lens and is invisible on a wide one, and
    the quantity the calibration cares about is where the board landed on the
    *sensor*.

    The metric offset needed to hit a target scales with distance, which is why
    it is computed per pose rather than drawn from a fixed range.
    """
    rng = np.random.default_rng(seed)
    half_width = intrinsics.image_width / 2.0
    half_height = intrinsics.image_height / 2.0
    poses: list[BoardPose] = []

    for index, (target_x, target_y) in enumerate(targets):
        distance = float(
            rng.uniform(*distances_m) if distances_m[0] < distances_m[1] else distances_m[0]
        )
        poses.append(
            BoardPose(
                distance_m=distance,
                rotation_deg=(
                    float(rng.uniform(-tilt_deg, tilt_deg)),
                    float(rng.uniform(-tilt_deg, tilt_deg)),
                    float(rng.uniform(-25.0, 25.0)),
                ),
                offset_m=(
                    target_x * half_width * distance / intrinsics.fx,
                    target_y * half_height * distance / intrinsics.fy,
                ),
            )
        )
        del index
    return poses


# Where a good capture puts the board: the centre, the edges and the corners.
# The corners are the ones that matter and the ones people skip, because the
# board half leaves the frame there -- which is exactly what Charuco tolerates
# and a plain chessboard does not.
SPREAD_TARGETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (-0.75, -0.65),
    (0.75, -0.65),
    (-0.75, 0.65),
    (0.75, 0.65),
    (0.0, -0.7),
    (0.0, 0.7),
    (-0.8, 0.0),
    (0.8, 0.0),
    (-0.4, -0.35),
    (0.4, -0.35),
    (-0.4, 0.35),
    (0.4, 0.35),
    (0.0, 0.0),
)


def well_spread_poses(
    count: int = 14, *, seed: int = 7, intrinsics: CameraIntrinsics | None = None
) -> list[BoardPose]:
    """A capture that determines a calibration: varied place, tilt and distance.

    What the capture protocol asks a person to do, expressed as poses. Three
    things are deliberate and each breaks a different degeneracy: the **tilt**
    separates focal length from distance, the **corner** placements are the only
    views that say anything about distortion, and the **distance** spread stops
    the fit describing the lens at one working distance only.
    """
    camera = intrinsics or default_camera()
    targets = [SPREAD_TARGETS[index % len(SPREAD_TARGETS)] for index in range(count)]
    return poses_at(targets, camera, tilt_deg=35.0, distances_m=(0.45, 0.95), seed=seed)


def degenerate_poses(count: int = 14, *, seed: int = 7) -> list[BoardPose]:
    """A capture that does not: square to the camera, centred, one distance.

    The failure this phase exists to detect. Every view is a good view -- the
    board is sharp, fully visible and detects perfectly -- and the set as a whole
    determines almost nothing, because a board held square at one distance cannot
    separate focal length from distance. The reprojection error is *better* here
    than for `well_spread_poses`, which is the measurement that justifies
    `CoverageReport`.
    """
    rng = np.random.default_rng(seed)
    return [
        BoardPose(
            distance_m=0.6 + 0.01 * float(rng.uniform(-1.0, 1.0)),
            rotation_deg=(
                1.5 * float(rng.uniform(-1.0, 1.0)),
                1.5 * float(rng.uniform(-1.0, 1.0)),
                float(rng.uniform(-8.0, 8.0)),
            ),
            offset_m=(
                0.02 * float(rng.uniform(-1.0, 1.0)),
                0.02 * float(rng.uniform(-1.0, 1.0)),
            ),
        )
        for _ in range(count)
    ]


@dataclass(frozen=True)
class StereoRig:
    """Two synthetic cameras with a known relative pose.

    The ground truth for stereo extrinsics: `rotation` and `translation_m` are
    what `calibrate_stereo` is asked to recover, and they are inputs here.

    The convention matches the one the contract uses, because the alternative is
    a transpose that produces a plausible rig pointing the wrong way: a point in
    the reference camera's frame becomes `rotation @ point + translation_m` in
    the target's.
    """

    reference: SyntheticCamera
    target: SyntheticCamera
    rotation: NDArray[np.float64]
    translation_m: NDArray[np.float64]

    def render_pair(self, pose: BoardPose) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Both cameras' views of one board pose, stated in the reference frame."""
        rvec, tvec = pose.rvec_tvec(self.reference.spec)
        board_rotation, _ = cv2.Rodrigues(rvec)

        target_rotation = self.rotation @ board_rotation
        target_translation = self.rotation @ tvec.reshape(3) + self.translation_m

        return (
            self.reference.render_pose(rvec, tvec),
            self.target.render_pose(
                cv2.Rodrigues(target_rotation)[0], target_translation.reshape(3, 1)
            ),
        )


def stereo_rig(
    *,
    baseline_m: float = 0.9,
    working_distance_m: float = 1.3,
    supersample: int = 1,
) -> StereoRig:
    """A two-camera rig, both cameras aimed at one working volume.

    Parameterised by where the cameras are rather than by the rotation between
    them, because that is the thing a person sets up: the target camera stands
    `baseline_m` to the reference's right and both are turned to look at a point
    `working_distance_m` down the reference's optical axis. The convergence
    angle follows from those two, rather than being a third number free to
    disagree with them.

    Getting this backwards produces a rig whose cameras point away from each
    other, which renders a board only one of them can see -- and the symptom is
    an empty detection rather than an error.
    """
    angle = float(np.arctan2(baseline_m, working_distance_m))
    # Yaw about the vertical: the target camera turns left to face the volume.
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    centre = np.array([baseline_m, 0.0, 0.0], dtype=np.float64)
    # X_target = R (X_reference - C), so the stored translation is -R C.
    translation = -rotation @ centre

    return StereoRig(
        reference=SyntheticCamera(supersample=supersample),
        target=SyntheticCamera(
            default_camera(fx=1500.0, distortion=(-0.22, 0.08, 0.0002, 0.0003)),
            supersample=supersample,
        ),
        rotation=rotation,
        translation_m=translation,
    )
