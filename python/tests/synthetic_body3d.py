"""A swing that exists in three dimensions, filmed by two cameras that exist.

Ground truth for Phase 9, in the only sense available without a motion-capture
lab: **the body is an input**. Every 3D point the reconstruction is asked to
recover is a number put in here, so "the reconstruction is accurate to 8 mm" is a
subtraction rather than a judgement.

Three things about it are load-bearing, and each exists because the reconstruction
has a check that would otherwise have nothing to check against.

**Bone lengths are exact by construction.** The arms are placed by two-link
inverse kinematics from a hand position, not by interpolating joint positions, so
the upper arm is `UPPER_ARM_M` long in every frame to floating-point precision.
That is what makes `skeleton.bone_consistency` measurable: any variation the
reconstruction reports is error it introduced, because there is none in the
input.

**The two cameras see one swing.** `tests/synthetic.py` gives one swing sampled
by two clocks and one projection; this gives one swing sampled by two clocks *and
two viewpoints*, which is the thing Phase 7 recorded as absent from this project
-- "the simulated cameras see the same projection of the same body, and two real
cameras do not". These two do not. They see a face-on and a down-the-line view of
one 3D motion, at different frame rates, with different lenses.

**The landmarks are distorted.** Points are projected through each camera's real
distortion coefficients, so the pipeline's undistortion is exercised rather than
bypassed. A reconstruction run without a calibration on this fixture is wrong by
the amount the lens bends the frame, which is the measurement Phase 8 promised
and could not make on a body.

## What it does not contain

Everything that makes real footage hard. The landmarks are the model's ideal:
there is no pose estimator here, so there is no motion blur at the bottom of the
downswing, no occluded hip that MediaPipe guesses at, no frame where the wrists
swap identity. Noise is added as an explicit, controllable displacement, which is
not the same thing as an estimator's error -- that has structure, and this does
not. **Errors measured against this fixture are a floor.**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.calibration import (
    BoardSpec,
    CalibrationQuality,
    CameraCalibration,
    CameraIntrinsics,
    CoverageReport,
    DetectionReport,
    DistortionModel,
    PairingReport,
    StereoCalibration,
)
from analyzer.contracts.calibration import CameraRig as RigContract
from analyzer.contracts.camera import CameraRole
from analyzer.contracts.pose import (
    LANDMARK_COUNT,
    FrameGeometry,
    Landmark,
    LandmarkPoint,
    PoseExtractionStats,
    PoseFrame,
    PoseModelInfo,
    PoseSequence,
)

# --- the body, in metres --------------------------------------------------
#
# A 1.78 m player, with segment lengths in the usual anthropometric proportions.
# They are exact inputs rather than estimates of anybody: what matters is that
# they do not change from frame to frame, which is the property the bone-length
# check exists to verify.

HIP_HEIGHT_M = 0.95
SHOULDER_HEIGHT_M = 1.44
SHOULDER_HALF_WIDTH_M = 0.20
HIP_HALF_WIDTH_M = 0.13
UPPER_ARM_M = 0.32
FOREARM_M = 0.29
THIGH_M = 0.42
SHIN_M = 0.42
HEEL_M = 0.07
FOOT_M = 0.18
HEAD_UP_M = 0.24
STANCE_HALF_M = 0.22

# Forward lean of the spine at address, towards the ball, in degrees.
POSTURE_DEG = 32.0

# --- the swing ------------------------------------------------------------
#
# The same four instants `tests/synthetic.py` is built around, so a clip produced
# here is detected as a swing by the same rules and its events land where that
# fixture's do.

TAKEAWAY_S = 0.50
TOP_S = 1.30
IMPACT_S = 1.70
FINISH_S = TOP_S + 2 * (IMPACT_S - TOP_S)
DURATION_S = 2.60

# How far the shoulders and the pelvis turn at the top, in degrees. A tour
# professional's numbers, so the X-factor the reconstruction recovers is a
# realistic one rather than a right angle chosen for numerical convenience.
SHOULDER_TURN_DEG = 92.0
PELVIS_TURN_DEG = 45.0

# The hands' arc: its radius, and how far down the spine its centre sits below
# the shoulders. Chosen so the hands stay **inside both arms' reach** through the
# whole swing -- they range 0.39 m to 0.59 m from a shoulder against a 0.61 m
# limit, which is a lead arm between 79 and 152 degrees of interior angle. That
# margin is load-bearing rather than cosmetic: a hand placed beyond reach would
# make `_two_link` clamp, and a clamped chain has the wrong forearm length, so
# the bone-consistency check would be measuring this file instead of the
# reconstruction.
HAND_RADIUS_M = 0.46
HAND_CENTRE_DROP_M = 0.16
SWING_PLANE_DEG = 58.0
"""Inclination of the hand plane from horizontal. A real swing plane, so the
hands travel in a circle that is neither in nor perpendicular to any camera's
image plane -- which is what makes the reconstruction's depth component
non-trivial in both views at once."""


def _rotation_about(axis: NDArray[np.float64], angle_rad: float) -> NDArray[np.float64]:
    """Rodrigues rotation about a unit axis. OpenCV's, so there is one convention."""
    unit = axis / np.linalg.norm(axis)
    return np.asarray(cv2.Rodrigues((unit * angle_rad).reshape(3, 1))[0], dtype=np.float64)


def turn_angles(t: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shoulder and pelvis rotation about the spine, in radians, against time.

    Eased in and out of the top, and through impact, so the angular velocity is
    continuous -- a body that changes direction instantaneously would give the
    filter a discontinuity to smooth, and the reconstruction would then be
    measured against a trajectory the filter could not follow.
    """
    fraction = np.zeros_like(t)

    rising = (t >= TAKEAWAY_S) & (t < TOP_S)
    u = (t[rising] - TAKEAWAY_S) / (TOP_S - TAKEAWAY_S)
    fraction[rising] = np.sin(np.pi / 2 * u) ** 2

    sweeping = (t >= TOP_S) & (t <= FINISH_S)
    u = (t[sweeping] - TOP_S) / (FINISH_S - TOP_S)
    fraction[sweeping] = np.cos(np.pi * u)

    fraction[t > FINISH_S] = -1.0

    return (
        np.radians(SHOULDER_TURN_DEG) * fraction,
        np.radians(PELVIS_TURN_DEG) * fraction,
    )


def _two_link(
    origin: NDArray[np.float64],
    end: NDArray[np.float64],
    upper: float,
    lower: float,
    hint: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The middle joint of a two-link chain with **exact** segment lengths.

    Given the two ends and the two lengths, the elbow lies on a circle; `hint`
    picks the point on it by naming which side the joint bends towards. This is
    what makes the fixture's bone lengths exact rather than approximately right:
    interpolating an elbow position between a shoulder and a wrist would give
    segment lengths that drift by centimetres through the swing, and the bone
    consistency check would then be measuring the fixture.
    """
    delta = end - origin
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        return origin + np.array([0.0, -upper, 0.0])

    # A chain cannot reach further than its own length; the caller's arc is sized
    # not to, and this clamps rather than producing a complex number if it does.
    reach = min(distance, upper + lower - 1e-6)
    direction = delta / distance

    along = (upper * upper - lower * lower + reach * reach) / (2.0 * reach)
    height = float(np.sqrt(max(upper * upper - along * along, 0.0)))

    perpendicular = hint - direction * float(hint @ direction)
    norm = float(np.linalg.norm(perpendicular))
    if norm <= 1e-9:  # pragma: no cover - the hint is chosen never to be parallel
        perpendicular = np.array([0.0, -1.0, 0.0]) - direction * float(-direction[1])
        norm = float(np.linalg.norm(perpendicular))
    perpendicular = perpendicular / norm

    return origin + direction * along + perpendicular * height


def body_at(instant: float) -> NDArray[np.float64]:
    """Every landmark's 3D position at one instant, `(33, 3)` in metres.

    World frame: **+x along the target line, +y up, +z out towards where a
    face-on camera stands.** The player addresses the ball at the origin, leaning
    forward over it by `POSTURE_DEG`, shoulders and hips square to the target
    line. The spine axis runs from the hip centre to the shoulder centre and is
    what both rotations turn about -- which is the definition the 3D metrics use,
    so the fixture and the measurement agree about what a shoulder turn is.
    """
    times = np.array([instant], dtype=np.float64)
    shoulder_turn, pelvis_turn = (value[0] for value in turn_angles(times))

    lean = np.radians(POSTURE_DEG)
    # The spine tilts forward about the target line (+x), so its top moves
    # towards the ball. Length is preserved: the hips stay put and the shoulders
    # swing forward and down.
    spine_length = SHOULDER_HEIGHT_M - HIP_HEIGHT_M
    hip_centre = np.array([0.0, HIP_HEIGHT_M, 0.0])
    spine_axis = np.array([0.0, np.cos(lean), np.sin(lean)])
    shoulder_centre = hip_centre + spine_length * spine_axis

    shoulder_rotation = _rotation_about(spine_axis, shoulder_turn)
    pelvis_rotation = _rotation_about(spine_axis, pelvis_turn)

    # Square to the target line at address, then turned about the spine.
    across = np.array([1.0, 0.0, 0.0])
    shoulder_arm = shoulder_rotation @ (SHOULDER_HALF_WIDTH_M * across)
    hip_arm = pelvis_rotation @ (HIP_HALF_WIDTH_M * across)

    left_shoulder = shoulder_centre - shoulder_arm
    right_shoulder = shoulder_centre + shoulder_arm
    left_hip = hip_centre - hip_arm
    right_hip = hip_centre + hip_arm

    hands = _hand_position(instant, shoulder_centre, spine_axis)

    # The two hands sit a little apart along the grip, so the wrists are distinct
    # points and Phase 4's hand-source choice is exercised. The offset runs along
    # the shoulder line, so it turns with the body rather than staying fixed in
    # the world -- a grip that stayed pointing at +z while the player turned
    # would stretch the forearm through the swing, and the bone-length check
    # would then be measuring the fixture.
    grip = 0.03 * _unit(shoulder_arm)
    left_wrist = hands - grip
    right_wrist = hands + grip

    # The elbows bend away from the chest, which the spine axis names without any
    # reference to a camera or to the world's own up. Solved **to each wrist
    # separately**, so that shoulder-elbow and elbow-wrist are exactly their
    # declared lengths in every frame: an elbow solved to the midpoint between
    # the hands would leave each forearm varying by centimetres as the arms
    # rotate, which is precisely the error the reconstruction is being measured
    # for introducing.
    left_elbow = _two_link(left_shoulder, left_wrist, UPPER_ARM_M, FOREARM_M, -spine_axis)
    right_elbow = _two_link(right_shoulder, right_wrist, UPPER_ARM_M, FOREARM_M, -spine_axis)

    points = np.zeros((LANDMARK_COUNT, 3), dtype=np.float64)

    def place(landmark: Landmark, value: NDArray[np.float64]) -> None:
        points[int(landmark)] = value

    place(Landmark.LEFT_SHOULDER, left_shoulder)
    place(Landmark.RIGHT_SHOULDER, right_shoulder)
    place(Landmark.LEFT_HIP, left_hip)
    place(Landmark.RIGHT_HIP, right_hip)
    place(Landmark.LEFT_ELBOW, left_elbow)
    place(Landmark.RIGHT_ELBOW, right_elbow)
    place(Landmark.LEFT_WRIST, left_wrist)
    place(Landmark.RIGHT_WRIST, right_wrist)
    for landmark, offset in (
        (Landmark.LEFT_PINKY, np.array([0.0, -0.05, -0.04])),
        (Landmark.LEFT_INDEX, np.array([0.02, -0.06, -0.02])),
        (Landmark.LEFT_THUMB, np.array([-0.02, -0.03, -0.03])),
        (Landmark.RIGHT_PINKY, np.array([0.0, -0.05, 0.04])),
        (Landmark.RIGHT_INDEX, np.array([0.02, -0.06, 0.05])),
        (Landmark.RIGHT_THUMB, np.array([-0.02, -0.03, 0.03])),
    ):
        place(landmark, hands + offset)

    # The head rides on the shoulder centre, tilted with the spine. Static
    # relative to the torso, which is what "keep your head still" describes and
    # is close enough for a fixture whose face landmarks nothing measures.
    head = shoulder_centre + HEAD_UP_M * spine_axis
    place(Landmark.NOSE, head + np.array([0.0, -0.02, 0.08]))
    for landmark, offset in (
        (Landmark.LEFT_EYE_INNER, np.array([-0.02, 0.02, 0.07])),
        (Landmark.LEFT_EYE, np.array([-0.03, 0.02, 0.07])),
        (Landmark.LEFT_EYE_OUTER, np.array([-0.04, 0.02, 0.06])),
        (Landmark.RIGHT_EYE_INNER, np.array([0.02, 0.02, 0.07])),
        (Landmark.RIGHT_EYE, np.array([0.03, 0.02, 0.07])),
        (Landmark.RIGHT_EYE_OUTER, np.array([0.04, 0.02, 0.06])),
        (Landmark.LEFT_EAR, np.array([-0.08, 0.01, 0.0])),
        (Landmark.RIGHT_EAR, np.array([0.08, 0.01, 0.0])),
        (Landmark.MOUTH_LEFT, np.array([-0.03, -0.05, 0.07])),
        (Landmark.MOUTH_RIGHT, np.array([0.03, -0.05, 0.07])),
    ):
        place(landmark, head + offset)

    # Legs, static, with exact lengths. A real swing moves them; this fixture
    # keeps them still so that a bone-length failure in a leg is unambiguously
    # the reconstruction's and never the fixture's.
    for side, hip, sign in (
        ("left", left_hip, -1.0),
        ("right", right_hip, 1.0),
    ):
        ankle_x = sign * STANCE_HALF_M
        knee = np.array([(hip[0] + ankle_x) / 2.0, hip[1] - THIGH_M * 0.82, hip[2] + 0.06])
        knee = hip + THIGH_M * _unit(knee - hip)
        ankle = knee + SHIN_M * _unit(np.array([ankle_x, 0.02, -0.02]) - knee)
        heel = ankle + HEEL_M * _unit(np.array([0.0, -0.4, -1.0]))
        toe = heel + FOOT_M * _unit(np.array([0.0, -0.1, 1.0]))
        place(Landmark[f"{side.upper()}_KNEE"], knee)
        place(Landmark[f"{side.upper()}_ANKLE"], ankle)
        place(Landmark[f"{side.upper()}_HEEL"], heel)
        place(Landmark[f"{side.upper()}_FOOT_INDEX"], toe)

    return points


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _hand_position(
    instant: float, shoulder_centre: NDArray[np.float64], spine_axis: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Where the hands are, on an inclined circular arc about the shoulder centre.

    The arc is tilted out of every camera's image plane by `SWING_PLANE_DEG`, so
    the hands move in depth as well as across the frame in both views at once.
    That is deliberate: an arc lying in one camera's image plane would let that
    camera alone determine the whole trajectory, and the reconstruction would
    score well without the second view contributing anything.
    """
    times = np.array([instant], dtype=np.float64)
    angle = _arc_angle(times)[0]

    plane = np.radians(SWING_PLANE_DEG)
    # Two orthogonal directions spanning the swing plane: one along the target
    # line, one tilted up out of the ground towards the player.
    across = np.array([1.0, 0.0, 0.0])
    up_plane = _unit(np.array([0.0, np.sin(plane), -np.cos(plane)]))
    centre = shoulder_centre - HAND_CENTRE_DROP_M * spine_axis

    return centre + HAND_RADIUS_M * (np.sin(angle) * across - np.cos(angle) * up_plane)


def _arc_angle(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Angle of the hands along their arc, zero at the bottom.

    The same shape `tests/synthetic.py` uses, for the same reason: speed is zero
    at the takeaway and again at the top, and maximum where the angle crosses
    zero -- the bottom of the arc, which is where impact is defined to be.
    """
    angle = np.zeros_like(t)

    rising = (t >= TAKEAWAY_S) & (t < TOP_S)
    u = (t[rising] - TAKEAWAY_S) / (TOP_S - TAKEAWAY_S)
    angle[rising] = 2.2 * np.sin(np.pi / 2 * u) ** 2

    sweeping = (t >= TOP_S) & (t <= FINISH_S)
    u = (t[sweeping] - TOP_S) / (FINISH_S - TOP_S)
    angle[sweeping] = 2.2 * np.cos(np.pi * u)

    angle[t > FINISH_S] = -2.2
    return angle


# --- the cameras ----------------------------------------------------------


@dataclass(frozen=True)
class SyntheticView:
    """One camera: where it stands, what it sees through, and what it records.

    `rotation` and `translation` take a **world** point into this camera's frame,
    which is OpenCV's convention and the one `cv2.projectPoints` consumes. The
    reconstruction's own frame is the reference camera's, so the ground truth it
    is compared against has to be carried into that frame -- which
    `SwingRig.truth_in_reference` does, once, rather than at every call site.
    """

    role: CameraRole
    intrinsics: CameraIntrinsics
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    fps: float
    world_start_s: float = 0.0
    slow_motion_factor: float = 1.0

    @property
    def geometry(self) -> FrameGeometry:
        return FrameGeometry(width=self.intrinsics.image_width, height=self.intrinsics.image_height)

    def times(self, duration_s: float = DURATION_S) -> NDArray[np.float64]:
        """This camera's own clock, starting at zero as a container reports it."""
        return np.arange(0.0, duration_s, 1.0 / self.fps)

    def world_times(self, duration_s: float = DURATION_S) -> NDArray[np.float64]:
        """The swing's own clock at each of this camera's frames."""
        return self.world_start_s + self.times(duration_s)

    def project(self, points_world: NDArray[np.float64]) -> NDArray[np.float64]:
        """World points to pixels, **with** this camera's distortion applied.

        Distortion is applied rather than skipped so that the pipeline's
        undistortion is exercised on a body rather than only on a board. A
        reconstruction run on this fixture without a calibration is then wrong by
        exactly the amount the lens bends the frame, which is a measurement the
        board could not make.
        """
        flat = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        projected, _ = cv2.projectPoints(
            flat,
            cv2.Rodrigues(self.rotation)[0],
            self.translation.reshape(3, 1),
            self.intrinsics.matrix(),
            self.intrinsics.distortion_vector(),
        )
        return np.asarray(projected, dtype=np.float64).reshape(*points_world.shape[:-1], 2)

    def to_camera(self, points_world: NDArray[np.float64]) -> NDArray[np.float64]:
        """World points in this camera's frame, in metres."""
        flat = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        moved = flat @ self.rotation.T + self.translation
        return moved.reshape(points_world.shape)


def look_at(
    position: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    up: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """World-to-camera rotation and translation for a camera aimed at a point.

    Built from where the camera *is* rather than from Euler angles, because that
    is what a person sets up and what the capture protocol describes. The camera
    looks down its own +z, with +x to the right of the image and +y downward,
    which is OpenCV's image convention.

    **The cross-product order is the thing to get right, and it is not the
    obvious one.** `cross(forward, world_up)` is the image's rightward direction;
    `cross(world_up, forward)` is its mirror, and a rig built from it is
    perfectly self-consistent -- both cameras are wrong the same way, so
    triangulation still recovers the body exactly and the reconstruction error
    measures as zero. What it produces is footage that is upside down and
    mirrored, which nothing in this phase notices and which Phase 4 notices
    immediately: the hands reach their *lowest* point at the top of the
    backswing, and the swing is refused. That is how this was found.
    """
    world_up = np.array([0.0, 1.0, 0.0]) if up is None else up
    forward = _unit(target - position)
    right = _unit(np.cross(forward, world_up))
    down = np.cross(forward, right)

    rotation = np.stack((right, down, forward), axis=0)
    return rotation, -rotation @ position


@dataclass(frozen=True)
class SwingRig:
    """Two cameras, one swing, and the ground truth relating them."""

    reference: SyntheticView
    target: SyntheticView

    @property
    def relative_rotation(self) -> NDArray[np.float64]:
        """Rotation carrying a point from the reference camera's frame to the target's."""
        return self.target.rotation @ self.reference.rotation.T

    @property
    def relative_translation(self) -> NDArray[np.float64]:
        return self.target.translation - self.relative_rotation @ self.reference.translation

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.relative_translation))

    @property
    def convergence_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.relative_rotation[2, 2], -1.0, 1.0))))

    def truth_in_reference(self, times_s: NDArray[np.float64]) -> NDArray[np.float64]:
        """The body's true 3D positions in the reference camera's frame.

        `times_s` are on the **reference clip's own clock**, which is what the
        reconstruction indexes by, so a caller comparing the two arrays does not
        have to reason about clock offsets a second time.
        """
        world = np.stack(
            [body_at(self.reference.world_start_s + float(t)) for t in times_s], axis=0
        )
        return self.reference.to_camera(world)

    def time_offset_s(self) -> float:
        """How far ahead of the reference clock the target clock runs.

        A camera that started rolling earlier in the swing has a clock that reads
        further along it, so the offset is the difference of the two start
        instants with the sign that makes `to_target` carry a reference time onto
        the target's clip clock.
        """
        return self.reference.world_start_s - self.target.world_start_s


def default_rig(
    *,
    reference_fps: float = 120.0,
    target_fps: float = 60.0,
    distance_m: float = 3.4,
    target_offset_s: float = -0.3571,
    convergence_deg: float = 90.0,
) -> SwingRig:
    """The capture the protocol asks for: one face-on camera, one down the line.

    The face-on camera stands out along +z looking back at the player; the
    down-the-line camera stands along the target line at `convergence_deg` round
    from it. Ninety degrees is the default because that is where two cameras
    determine a point best -- and because it is what a person filming a golf
    swing naturally does, which is the happy case worth measuring first. The
    parameter exists so `--sweep convergence` can close it down and watch the
    error grow.

    The two cameras are given different focal lengths, different distortion and
    different frame rates, because two phones are never the same phone and a
    fixture where they are would hide any place the code assumed they were.

    **`distance_m` is set by how much of the frame the subject fills**, not by
    what looks like a plausible tripod position. Landmark noise is quoted in
    pixels and its consequences scale with the subject's size in pixels, so a
    fixture that frames the player small turns an ordinary noise level into a
    stress test: at 4.2 m the torso spans 0.072 frame widths and Phase 4 refuses
    to call the clip a swing at the 2.7 px scatter Phase 3 measured on real
    footage, while at 3.4 m it spans 0.088 and detection holds. The nearer
    framing is also what `data/README.md` asks for.

    **`target_offset_s` is deliberately not a whole number of frames.** At an
    offset of exactly 0.35 s, every 120 fps reference instant maps onto an exact
    60 fps target frame, so pairing nearest frames is *exact* and the sweep that
    exists to measure what that costs measures zero. The default is offset by a
    fraction of a frame so the two clocks are related the way two hand-started
    cameras really are.
    """
    centre = np.array([0.0, 1.15, 0.0])

    face_on_position = np.array([0.0, 1.35, distance_m])
    angle = np.radians(convergence_deg)
    target_position = np.array(
        [
            -distance_m * np.sin(angle),
            1.35,
            distance_m * np.cos(angle),
        ]
    )

    face_on_rotation, face_on_translation = look_at(face_on_position, centre)
    target_rotation, target_translation = look_at(target_position, centre)

    return SwingRig(
        reference=SyntheticView(
            role=CameraRole.FACE_ON,
            intrinsics=_camera(1920, 1080, fx=1400.0, distortion=(-0.28, 0.12, 0.0006, -0.0004)),
            rotation=face_on_rotation,
            translation=face_on_translation,
            fps=reference_fps,
        ),
        target=SyntheticView(
            role=CameraRole.DOWN_THE_LINE,
            intrinsics=_camera(1920, 1080, fx=1500.0, distortion=(-0.22, 0.08, 0.0002, 0.0003)),
            rotation=target_rotation,
            translation=target_translation,
            fps=target_fps,
            world_start_s=target_offset_s,
        ),
    )


def _camera(
    width: int,
    height: int,
    *,
    fx: float,
    distortion: tuple[float, ...],
) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=fx,
        fy=fx,
        cx=width / 2.0 - 3.0,
        cy=height / 2.0 + 4.0,
        distortion=list(distortion),
        model=DistortionModel.RADIAL_TANGENTIAL_4,
        image_width=width,
        image_height=height,
    )


# --- turning the projections into what the pipeline consumes --------------


def pose_sequence_for(
    view: SyntheticView,
    *,
    duration_s: float = DURATION_S,
    noise_px: float = 0.0,
    epipolar_noise_px: float = 0.0,
    epipolar_towards: SyntheticView | None = None,
    visibility: float = 0.97,
    seed: int = 11,
    path: str | None = None,
) -> PoseSequence:
    """One camera's recording of the swing, as a stored pose sequence would be.

    `noise_px` is isotropic landmark scatter, which is what a pose estimator
    produces to first order. `epipolar_noise_px` is the interesting one: it
    displaces every landmark **along the epipolar line** towards
    `epipolar_towards`, which is the component a reprojection residual cannot
    see. Sweeping it is how `scripts/benchmark_reconstruct.py` demonstrates that
    the residual is not the quality of a reconstruction.
    """
    rng = np.random.default_rng(seed)
    clock = view.times(duration_s)
    world = np.stack([body_at(view.world_start_s + float(t)) for t in clock], axis=0)
    pixels = view.project(world)

    if epipolar_noise_px > 0.0 and epipolar_towards is not None:
        pixels = pixels + epipolar_noise_px * _epipolar_directions(
            view, world, epipolar_towards, rng
        )
    if noise_px > 0.0:
        pixels = pixels + rng.normal(0.0, noise_px, size=pixels.shape)

    geometry = view.geometry
    normalised = pixels / np.array([geometry.width, geometry.height], dtype=np.float64)

    frames: list[PoseFrame] = []
    for index in range(clock.size):
        points = [
            LandmarkPoint(
                x=float(normalised[index, landmark, 0]),
                y=float(normalised[index, landmark, 1]),
                # The z channel is MediaPipe's own single-camera depth guess, and
                # nothing in this phase reads it. Zero rather than the true depth,
                # so a reconstruction that accidentally used it would be wrong
                # rather than accidentally right.
                z=0.0,
                visibility=visibility,
                presence=0.99,
            )
            for landmark in range(LANDMARK_COUNT)
        ]
        frames.append(
            PoseFrame(
                frame_index=index,
                timestamp_s=float(clock[index]) * view.slow_motion_factor,
                detected=True,
                image=points,
                hip_local=points,
            )
        )

    return PoseSequence(
        video_path=path or f"/data/synthetic_{view.role.value}.mov",
        video_content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED,
            digest=f"{abs(hash(view.role.value)) % 16:x}" * 64,
            size_bytes=1,
        ),
        geometry=geometry,
        model=PoseModelInfo(
            name="synthetic",
            variant="synthetic",
            precision="float64",
            sha256="0" * 64,
            delegate="cpu",
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ),
        extracted_at=datetime(2026, 9, 17, tzinfo=UTC),
        stats=PoseExtractionStats(
            frames_processed=int(clock.size),
            frames_detected=int(clock.size),
            detection_rate=1.0,
            elapsed_s=1.0,
            ms_per_frame=1.0,
            mean_visibility=visibility,
        ),
        frames=frames,
    )


def _epipolar_directions(
    view: SyntheticView,
    world: NDArray[np.float64],
    other: SyntheticView,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Unit image directions along which `view` cannot contradict `other`.

    The epipolar line of a point, in `view`, is where every 3D point on `other`'s
    ray through that landmark projects. So the direction is found by moving the
    point a little along that ray and seeing which way its image goes: a
    displacement in that direction is exactly a displacement that some 3D point
    explains perfectly, and therefore one no reprojection error can object to.

    Signed randomly per landmark-frame, so the sweep adds scatter along the blind
    direction rather than a coherent push that would simply translate the whole
    reconstruction.
    """
    in_other = other.to_camera(world)
    nudged_other = in_other * 1.001
    # Back to the world, then into this view: the same 3D point, moved along the
    # other camera's line of sight.
    nudged_world = (nudged_other.reshape(-1, 3) - other.translation) @ other.rotation
    nudged_world = nudged_world.reshape(world.shape)

    here = view.project(world)
    moved = view.project(nudged_world)
    direction = moved - here

    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    unit = np.divide(direction, norms, out=np.zeros_like(direction), where=norms > 1e-9)
    signs = rng.choice((-1.0, 1.0), size=(*world.shape[:-1], 1))
    return unit * signs


def camera_rig(rig: SwingRig) -> RigContract:
    """A `CameraRig` describing the fixture exactly, marked usable.

    The quality numbers are what this rig genuinely is rather than plausible
    filler: the geometry is exact, so the reprojection residual of the
    calibration that produced it is zero, and the coverage is stated as the
    synthetic capture it stands in for. `source` says 'synthetic' in both
    calibrations so a rig that escaped a test into a report cannot be mistaken
    for a measurement of a real camera.
    """
    return RigContract(
        cameras={view.role: _calibration_for(view) for view in (rig.reference, rig.target)},
        stereo=StereoCalibration(
            reference_role=rig.reference.role,
            target_role=rig.target.role,
            rotation=[[float(value) for value in row] for row in rig.relative_rotation],
            translation_m=[float(value) for value in rig.relative_translation],
            baseline_m=rig.baseline_m,
            convergence_deg=rig.convergence_deg,
            quality=_quality(),
            pairing=PairingReport(pairs=8, candidates=8, method="explicit"),
            calibrated_at=datetime(2026, 9, 17, tzinfo=UTC),
            usable=True,
        ),
    )


def _calibration_for(view: SyntheticView) -> CameraCalibration:
    spec = BoardSpec(squares_x=7, squares_y=5, square_length_m=0.035, marker_length_m=0.026)
    return CameraCalibration(
        role=view.role,
        intrinsics=view.intrinsics,
        quality=_quality(),
        detection=DetectionReport(
            frames_scanned=0,
            frames_with_board=0,
            views_used=14,
            corners_total=14 * spec.interior_corners,
            board=spec,
        ),
        calibrated_at=datetime(2026, 9, 17, tzinfo=UTC),
        source="synthetic: the intrinsics are inputs, not a fit",
        usable=True,
    )


def _quality() -> CalibrationQuality:
    return CalibrationQuality(
        rms_reprojection_px=0.0,
        max_reprojection_px=0.0,
        coverage=CoverageReport(
            views=14,
            corners=14 * 24,
            image_fraction=0.9,
            edge_fraction=0.3,
            tilt_range_deg=40.0,
            scale_range=2.0,
            methodology="synthetic rig: the camera is an input, so there is no fit to score",
        ),
        degrees_of_freedom=14 * 24 * 2 - 9,
    )
