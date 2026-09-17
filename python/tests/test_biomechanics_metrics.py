"""Biomechanics engine tests, against a body built from known angles.

The fixture is written in **plane coordinates** -- x to the right, y upwards,
both in frame widths -- and converted backwards into the normalised IMAGE
coordinates a pose estimator would have produced, on a deliberately non-square
1080x1920 frame. So the numbers asserted below are the numbers the body was
built with, and the whole chain has to undo the aspect ratio and the y flip to
get back to them. A fixture written directly in image coordinates could not tell
a correct engine from one that skipped both.

Angles are chosen so their expected values are facts rather than agreements: a
knee built at 25 degrees of flex must read 25, an arm whose elbow sits on the
line between shoulder and wrist must read 180, and one whose elbow sits on the
circle with that line as diameter must read 90 by Thales' theorem.

Turn schedules are held **flat** across the instants they are measured at. A
local polynomial fit reproduces a constant exactly, so the recovered shoulder
turn at the top is the constructed one to floating-point precision rather than
to whatever the smoothing of a curved signal happens to leave.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from analyzer.biomechanics import body_from, compute_metrics
from analyzer.biomechanics.registry import REGISTRY
from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.metrics import (
    BodySide,
    MetricBasis,
    MetricConfig,
    MetricName,
    MetricUnit,
)
from analyzer.contracts.phases import SwingEvent, SwingPhase
from analyzer.contracts.pose import (
    LANDMARK_COUNT,
    FrameGeometry,
    Landmark,
    LandmarkPoint,
    LandmarkSpace,
    PoseExtractionStats,
    PoseFrame,
    PoseModelInfo,
    PoseSequence,
)
from analyzer.filtering.landmarks import filter_sequence
from analyzer.phases import detect_phases

FPS = 120.0
DURATION_S = 2.60
TAKEAWAY_S, TOP_S, IMPACT_S = 0.50, 1.30, 1.70
FINISH_S = TOP_S + 2 * (IMPACT_S - TOP_S)

# Portrait, so the aspect correction is 1.7778 rather than 1 and a pipeline that
# skipped it is off by a factor no tolerance would absorb.
GEOMETRY = FrameGeometry(width=1080, height=1920)

# --- the body, in frame widths -------------------------------------------
HIP_MID = (0.50, 0.55)
TORSO = 0.30
SPINE_TILT_DEG = 8.0
SHOULDER_SPAN = 0.36
HIP_SPAN = 0.24
SHOULDER_TILT_DEG = 12.0
HIP_TILT_DEG = -6.0
THIGH, SHIN = 0.22, 0.20
LEFT_KNEE_FLEX_DEG, RIGHT_KNEE_FLEX_DEG = 25.0, 15.0
HEAD_ABOVE_SHOULDERS = 0.12

# --- what moves, in torso lengths ----------------------------------------
SHOULDER_TURN_TOP, SHOULDER_TURN_IMPACT = 40.0, 15.0
PELVIS_TURN_TOP, PELVIS_TURN_IMPACT = 20.0, 10.0
HIP_SWAY_TOP, HIP_SWAY_IMPACT = -0.15, 0.05
HEAD_EXTRA_TOP, HEAD_EXTRA_IMPACT = -0.10, -0.04
HEAD_LIFT_TOP, HEAD_LIFT_IMPACT = 0.08, -0.06

# --- the hand arc, which is what phase detection reads --------------------
ARC_RADIUS = 0.25
ARC_LOW = 0.50
ARC_TOP_ANGLE = 2.2

X_FACTOR_TOP = SHOULDER_TURN_TOP - PELVIS_TURN_TOP


def _schedule(t: np.ndarray, at_top: float, at_impact: float) -> np.ndarray:
    """A quantity that is zero at address and flat across the top and impact.

    The flats are what make the expected values exact. A local polynomial fit
    reproduces a constant exactly, so a value measured anywhere inside one comes
    back unchanged by the smoothing; a peak at a corner would be rounded off and
    the test would be asserting against the filter rather than the fixture.
    """
    knots_t = [
        0.0,
        TAKEAWAY_S,
        TOP_S - 0.15,
        TOP_S + 0.15,
        IMPACT_S - 0.10,
        IMPACT_S + 0.10,
        DURATION_S,
    ]
    knots_v = [0.0, 0.0, at_top, at_top, at_impact, at_impact, at_impact]
    return np.interp(t, knots_t, knots_v)


def _arc_angle(t: np.ndarray) -> np.ndarray:
    """The hands' angle along their arc, zero at the bottom. As Phase 4's fixture."""
    angle = np.zeros_like(t)

    rising = (t >= TAKEAWAY_S) & (t < TOP_S)
    u = (t[rising] - TAKEAWAY_S) / (TOP_S - TAKEAWAY_S)
    angle[rising] = ARC_TOP_ANGLE * np.sin(np.pi / 2 * u) ** 2

    sweeping = (t >= TOP_S) & (t <= FINISH_S)
    u = (t[sweeping] - TOP_S) / (FINISH_S - TOP_S)
    angle[sweeping] = ARC_TOP_ANGLE * np.cos(np.pi * u)

    angle[t > FINISH_S] = -ARC_TOP_ANGLE
    return angle


def _to_image(plane: np.ndarray, geometry: FrameGeometry) -> tuple[float, float]:
    """Invert `plane_coordinates`: frame widths and y-up back to normalised image."""
    return float(plane[0]), float(1.0 - plane[1] / geometry.aspect_ratio)


def _unit(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    return np.array([math.cos(radians), math.sin(radians)])


def _elbow(shoulder: np.ndarray, wrist: np.ndarray, *, straight: bool) -> np.ndarray:
    """An elbow giving an exactly known shoulder-elbow-wrist angle.

    On the segment: the three points are collinear and the angle is 180 degrees,
    a straight arm. On the circle with that segment as its diameter: the angle
    is 90 degrees by Thales' theorem, whatever the shoulder and wrist are doing.
    """
    centre = (shoulder + wrist) / 2.0
    if straight:
        return centre
    chord = wrist - shoulder
    perpendicular = np.array([-chord[1], chord[0]])
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    return centre + perpendicular * (np.linalg.norm(chord) / 2.0)


def _swing_frames(
    *,
    mirrored: bool = False,
    shoulder_turn_top: float = SHOULDER_TURN_TOP,
    pelvis_turn_top: float = PELVIS_TURN_TOP,
    shoulder_span: float = SHOULDER_SPAN,
    address_turn: float = 0.0,
) -> tuple[np.ndarray, list[dict[Landmark, np.ndarray]]]:
    """The whole body, per frame, in plane coordinates.

    `mirrored` flips the swing left to right, which is what separates a
    right-handed player from a left-handed one and is the only thing the lead
    side inference has to go on. `address_turn` starts the body already turned
    away from the camera, which is what a down-the-line recording looks like.
    """
    t = np.arange(0.0, DURATION_S, 1.0 / FPS)
    side = -1.0 if not mirrored else 1.0

    shoulder_turn = address_turn + _schedule(t, shoulder_turn_top, SHOULDER_TURN_IMPACT)
    pelvis_turn = address_turn + _schedule(t, pelvis_turn_top, PELVIS_TURN_IMPACT)
    sway = _schedule(t, HIP_SWAY_TOP, HIP_SWAY_IMPACT) * TORSO * side
    head_extra = _schedule(t, HEAD_EXTRA_TOP, HEAD_EXTRA_IMPACT) * TORSO * side
    head_lift = _schedule(t, HEAD_LIFT_TOP, HEAD_LIFT_IMPACT) * TORSO
    arc = _arc_angle(t) * side

    shoulder_direction = _unit(SHOULDER_TILT_DEG * side)
    hip_direction = _unit(HIP_TILT_DEG * side)
    spine = np.array(
        [
            math.sin(math.radians(SPINE_TILT_DEG * side)),
            math.cos(math.radians(SPINE_TILT_DEG * side)),
        ]
    )

    frames: list[dict[Landmark, np.ndarray]] = []
    for index in range(t.size):
        hip_mid = np.array([HIP_MID[0] + sway[index], HIP_MID[1]])
        shoulder_mid = hip_mid + TORSO * spine

        half_shoulder = shoulder_span * math.cos(math.radians(shoulder_turn[index])) / 2.0
        half_hip = HIP_SPAN * math.cos(math.radians(pelvis_turn[index])) / 2.0

        # The player faces the camera, so their left is on the +x side of the
        # frame -- the arrangement the tilt sign convention has to cope with.
        left_shoulder = shoulder_mid + half_shoulder * shoulder_direction
        right_shoulder = shoulder_mid - half_shoulder * shoulder_direction
        left_hip = hip_mid + half_hip * hip_direction
        right_hip = hip_mid - half_hip * hip_direction

        hand = np.array(
            [
                HIP_MID[0] + sway[index] + ARC_RADIUS * math.sin(arc[index]),
                ARC_LOW + ARC_RADIUS * (1.0 - math.cos(arc[index])),
            ]
        )
        left_wrist = hand + np.array([0.008, 0.0])
        right_wrist = hand - np.array([0.008, 0.0])

        # The left arm is built straight and the right bent to a right angle, so
        # whichever the engine calls the lead arm has a value known in advance.
        pose = {
            Landmark.NOSE: shoulder_mid
            + np.array([head_extra[index], HEAD_ABOVE_SHOULDERS + head_lift[index]]),
            Landmark.LEFT_SHOULDER: left_shoulder,
            Landmark.RIGHT_SHOULDER: right_shoulder,
            Landmark.LEFT_WRIST: left_wrist,
            Landmark.RIGHT_WRIST: right_wrist,
            Landmark.LEFT_ELBOW: _elbow(left_shoulder, left_wrist, straight=True),
            Landmark.RIGHT_ELBOW: _elbow(right_shoulder, right_wrist, straight=False),
            Landmark.LEFT_HIP: left_hip,
            Landmark.RIGHT_HIP: right_hip,
        }
        for hip, knee_name, ankle_name, flex in (
            (left_hip, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE, LEFT_KNEE_FLEX_DEG),
            (right_hip, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE, RIGHT_KNEE_FLEX_DEG),
        ):
            knee = hip + np.array([0.0, -THIGH])
            radians = math.radians(flex)
            pose[knee_name] = knee
            pose[ankle_name] = knee + SHIN * np.array([math.sin(radians), -math.cos(radians)])

        frames.append(pose)

    return t, frames


def _sequence(
    t: np.ndarray,
    frames: list[dict[Landmark, np.ndarray]],
    *,
    visibility: float = 0.95,
    hidden: tuple[Landmark, ...] = (),
    geometry: FrameGeometry = GEOMETRY,
) -> PoseSequence:
    """Wrap plane-coordinate poses into a `PoseSequence` in normalised IMAGE space."""
    pose_frames = []
    for index in range(t.size):
        points = []
        for landmark in range(LANDMARK_COUNT):
            place = frames[index].get(Landmark(landmark))
            if place is None:
                # Untracked landmarks the metric layer never reads. Parked on the
                # hip so they are inside the frame and plainly not a body part.
                place = frames[index][Landmark.LEFT_HIP]
            x, y = _to_image(place, geometry)
            seen = 0.05 if Landmark(landmark) in hidden else visibility
            points.append(LandmarkPoint(x=x, y=y, z=0.0, visibility=seen, presence=0.99))
        pose_frames.append(
            PoseFrame(
                frame_index=index,
                timestamp_s=float(t[index]),
                detected=True,
                image=points,
                hip_local=points,
            )
        )

    return PoseSequence(
        video_path="/data/synthetic.mov",
        video_content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED, digest="a" * 64, size_bytes=1
        ),
        geometry=geometry,
        model=PoseModelInfo(
            name="fake",
            variant="fake",
            precision="float32",
            sha256="0" * 64,
            delegate="cpu",
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ),
        extracted_at=datetime(2026, 9, 16, tzinfo=UTC),
        stats=PoseExtractionStats(
            frames_processed=t.size,
            frames_detected=t.size,
            detection_rate=1.0,
            elapsed_s=1.0,
            ms_per_frame=1.0,
        ),
        frames=pose_frames,
    )


def _measure(sequence: PoseSequence, config: MetricConfig | None = None):
    filtered = filter_sequence(sequence, FilterConfig())
    return compute_metrics(filtered, detect_phases(filtered), config)


def _swing(*, geometry: FrameGeometry = GEOMETRY, **kwargs):
    t, frames = _swing_frames(**kwargs)
    return _sequence(t, frames, geometry=geometry)


@pytest.fixture(scope="module")
def result():
    """The reference synthetic swing, measured once for the whole module."""
    return _measure(_swing())


def _value(result, name: MetricName, event: SwingEvent | None) -> float:
    found = result.get(name, event)
    assert found is not None, f"{name.value} at {event} was not produced"
    return found.value


class TestTheEngineRuns:
    def test_a_detected_swing_produces_metrics(self, result) -> None:
        assert result.computed
        assert result.metrics

    def test_the_torso_scale_is_the_one_the_body_was_built_with(self, result) -> None:
        assert result.torso_length == pytest.approx(TORSO, abs=0.002)

    def test_the_frame_geometry_is_carried_through_to_the_result(self, result) -> None:
        assert result.geometry.width == GEOMETRY.width
        assert result.geometry.height == GEOMETRY.height


class TestPosture:
    """5.2 -- angles the fixture was built from, recovered."""

    def test_spine_tilt_is_the_constructed_lean(self, result) -> None:
        # Negative because the unmirrored fixture is a right-handed player, whose
        # spine the constructor leans towards the frame's left.
        assert _value(result, MetricName.SPINE_TILT, None) == pytest.approx(
            -SPINE_TILT_DEG, abs=0.5
        )

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (MetricName.LEFT_KNEE_FLEX, LEFT_KNEE_FLEX_DEG),
            (MetricName.RIGHT_KNEE_FLEX, RIGHT_KNEE_FLEX_DEG),
        ],
    )
    def test_knee_flex_is_the_constructed_bend(self, result, name, expected) -> None:
        assert _value(result, name, SwingEvent.TOP) == pytest.approx(expected, abs=0.5)

    def test_a_straight_leg_would_read_zero(self) -> None:
        """The subtraction from 180: larger means more bent, not less."""
        t, frames = _swing_frames()
        for pose in frames:
            knee = pose[Landmark.LEFT_KNEE]
            pose[Landmark.LEFT_ANKLE] = knee + np.array([0.0, -SHIN])
        measured = _measure(_sequence(t, frames))
        assert _value(measured, MetricName.LEFT_KNEE_FLEX, SwingEvent.TOP) == pytest.approx(
            0.0, abs=0.5
        )

    def test_hip_sway_is_measured_from_the_address_pose(self, result) -> None:
        # The fixture sways towards the frame's left on an unmirrored swing.
        assert _value(result, MetricName.HIP_SWAY, SwingEvent.TOP) == pytest.approx(
            -HIP_SWAY_TOP, abs=0.02
        )

    def test_head_sway_includes_the_body_it_sits_on(self, result) -> None:
        """The head moves with the hips plus whatever it does of its own."""
        expected = -(HIP_SWAY_TOP + HEAD_EXTRA_TOP)
        assert _value(result, MetricName.HEAD_SWAY, SwingEvent.TOP) == pytest.approx(
            expected, abs=0.02
        )

    def test_head_lift_is_positive_upwards(self, result) -> None:
        assert _value(result, MetricName.HEAD_LIFT, SwingEvent.TOP) == pytest.approx(
            HEAD_LIFT_TOP, abs=0.02
        )
        assert _value(result, MetricName.HEAD_LIFT, SwingEvent.IMPACT) == pytest.approx(
            HEAD_LIFT_IMPACT, abs=0.02
        )

    def test_displacements_are_in_torso_lengths_not_frame_widths(self, result) -> None:
        """Which is what makes them comparable between recordings."""
        lift = result.get(MetricName.HEAD_LIFT, SwingEvent.TOP)
        assert lift.unit is MetricUnit.TORSO_LENGTHS


class TestRotation:
    """5.3 -- foreshortening, and what it can and cannot say."""

    def test_shoulder_turn_recovers_the_constructed_angle(self, result) -> None:
        assert _value(result, MetricName.SHOULDER_TURN, SwingEvent.TOP) == pytest.approx(
            SHOULDER_TURN_TOP, abs=1.0
        )

    def test_pelvis_turn_recovers_the_constructed_angle(self, result) -> None:
        assert _value(result, MetricName.PELVIS_TURN, SwingEvent.TOP) == pytest.approx(
            PELVIS_TURN_TOP, abs=1.0
        )

    def test_x_factor_is_the_separation_between_them(self, result) -> None:
        assert _value(result, MetricName.X_FACTOR, SwingEvent.TOP) == pytest.approx(
            X_FACTOR_TOP, abs=1.5
        )

    def test_turn_is_blind_to_direction(self) -> None:
        """A turn and its mirror image shorten the shoulder line identically.

        Not a defect to be fixed at this layer -- it is what a cosine does -- but
        a property the value has to be read with, which is why it is a magnitude
        and why the basis says so.
        """
        mirrored = _measure(_swing(mirrored=True))
        straight = _measure(_swing())
        assert _value(mirrored, MetricName.SHOULDER_TURN, SwingEvent.TOP) == pytest.approx(
            _value(straight, MetricName.SHOULDER_TURN, SwingEvent.TOP), abs=1.0
        )

    def test_turn_is_not_reported_at_address(self, result) -> None:
        """Address is the baseline, so the value there is zero by construction."""
        assert result.get(MetricName.SHOULDER_TURN, None) is None

    def test_the_baseline_is_taken_at_address_and_reported(self, result) -> None:
        shoulders = next(r for r in result.references if r.landmarks == "shoulders")
        assert shoulders.span == pytest.approx(SHOULDER_SPAN / TORSO, abs=0.02)
        assert shoulders.square_at_address
        assert shoulders.frames

    def test_shoulder_tilt_is_read_by_position_in_the_frame(self, result) -> None:
        """Positive means the shoulder further right in the frame is higher.

        The fixture puts the player's left shoulder on the +x side, tilted up,
        so an unmirrored swing reads negative: the constructor leans the whole
        body the other way for a right-handed player.
        """
        tilt = _value(result, MetricName.SHOULDER_TILT, None)
        assert tilt == pytest.approx(-SHOULDER_TILT_DEG, abs=1.0)

    def test_pelvis_tilt_carries_its_own_sign(self, result) -> None:
        """Mirrored like every other angle, because the fixture mirrors the player."""
        assert _value(result, MetricName.PELVIS_TILT, None) == pytest.approx(-HIP_TILT_DEG, abs=1.0)


class TestRotationRefusal:
    """The down-the-line case, which is the reason the check exists."""

    @staticmethod
    def _down_the_line():
        # Shoulders start nearly end-on and open up through the backswing, which
        # is exactly what a camera down the target line sees.
        return _measure(_swing(address_turn=82.0, shoulder_turn_top=-82.0, pelvis_turn_top=-70.0))

    def test_turn_is_refused_when_the_body_was_not_square_at_address(self) -> None:
        result = self._down_the_line()
        assert result.get(MetricName.SHOULDER_TURN, SwingEvent.TOP) is None
        assert any(entry.name is MetricName.SHOULDER_TURN for entry in result.refused)

    def test_the_refusal_says_the_recording_is_not_face_on(self) -> None:
        result = self._down_the_line()
        assert any("face-on" in warning for warning in result.warnings)

    def test_x_factor_goes_with_the_turns_it_is_built_from(self) -> None:
        result = self._down_the_line()
        assert result.get(MetricName.X_FACTOR, SwingEvent.TOP) is None

    def test_the_reference_records_how_far_off_the_baseline_was(self) -> None:
        result = self._down_the_line()
        shoulders = next(r for r in result.references if r.landmarks == "shoulders")
        assert not shoulders.square_at_address
        assert shoulders.excess > 1.25

    def test_tilts_survive_a_view_that_refuses_turns(self) -> None:
        """A projected tilt needs no square baseline, so it is still available."""
        result = self._down_the_line()
        assert result.get(MetricName.SHOULDER_TILT, SwingEvent.TOP) is not None


class TestArms:
    """5.4 -- including working out which arm leads."""

    def test_the_lead_side_is_inferred_from_where_the_hands_go(self, result) -> None:
        """The hands sit over the trail shoulder at the top, so lead is the other."""
        assert result.lead_side is not None
        assert result.lead_side.side is BodySide.LEFT

    def test_mirroring_the_swing_swaps_the_lead_side(self) -> None:
        """The whole point: handedness is measured, not assumed."""
        mirrored = _measure(_swing(mirrored=True))
        assert mirrored.lead_side.side is BodySide.RIGHT

    def test_the_lead_arm_is_the_one_built_straight(self, result) -> None:
        """Left arm built collinear, so 180 degrees, and left is the lead here."""
        assert _value(result, MetricName.LEAD_ARM_ANGLE, SwingEvent.TOP) == pytest.approx(
            180.0, abs=1.0
        )

    def test_the_trail_arm_is_the_one_built_bent(self, result) -> None:
        """Right elbow on the circle with shoulder-wrist as diameter: 90 by Thales."""
        assert _value(result, MetricName.TRAIL_ARM_ANGLE, SwingEvent.TOP) == pytest.approx(
            90.0, abs=4.0
        )

    def test_the_arms_swap_with_the_lead_side(self) -> None:
        mirrored = _measure(_swing(mirrored=True))
        # The left arm is still the straight one, and is now the trail arm.
        assert _value(mirrored, MetricName.TRAIL_ARM_ANGLE, SwingEvent.TOP) == pytest.approx(
            180.0, abs=1.0
        )

    def test_arm_angles_are_refused_when_the_side_cannot_be_named(self) -> None:
        """Rather than assuming the player is right-handed."""
        t, frames = _swing_frames()
        for pose in frames:
            # Shoulders collapsed onto one point: the line no longer shows a side.
            mid = (pose[Landmark.LEFT_SHOULDER] + pose[Landmark.RIGHT_SHOULDER]) / 2.0
            pose[Landmark.LEFT_SHOULDER] = mid + np.array([0.0005, 0.0])
            pose[Landmark.RIGHT_SHOULDER] = mid - np.array([0.0005, 0.0])
        measured = _measure(_sequence(t, frames))

        assert measured.lead_side.side is None
        assert measured.get(MetricName.LEAD_ARM_ANGLE, SwingEvent.TOP) is None
        assert any(entry.name is MetricName.LEAD_ARM_ANGLE for entry in measured.refused)

    def test_hand_path_is_summed_along_the_arc_not_across_its_chord(self, result) -> None:
        """An arc reported as its chord is a quiet, consistent underestimate."""
        path = _value(result, MetricName.HAND_PATH_LENGTH, None)
        chord = (2 * ARC_RADIUS * math.sin(ARC_TOP_ANGLE / 2)) / TORSO
        assert path > chord * 1.1

    def test_peak_hand_speed_is_in_torso_lengths_per_second(self, result) -> None:
        peak = result.get(MetricName.PEAK_HAND_SPEED, None)
        assert peak is not None
        assert peak.unit is MetricUnit.TORSO_LENGTHS_PER_S
        assert peak.value > 0


class TestTiming:
    """5.5 -- the only metrics here a camera position cannot spoil."""

    def test_backswing_duration_matches_the_fixture(self, result) -> None:
        assert _value(result, MetricName.BACKSWING_DURATION, None) == pytest.approx(
            TOP_S - TAKEAWAY_S, abs=0.15
        )

    def test_downswing_duration_matches_the_fixture(self, result) -> None:
        assert _value(result, MetricName.DOWNSWING_DURATION, None) == pytest.approx(
            IMPACT_S - TOP_S, abs=0.10
        )

    def test_tempo_is_the_ratio_of_the_two(self, result) -> None:
        backswing = _value(result, MetricName.BACKSWING_DURATION, None)
        downswing = _value(result, MetricName.DOWNSWING_DURATION, None)
        assert _value(result, MetricName.TEMPO_RATIO, None) == pytest.approx(
            backswing / downswing, rel=1e-9
        )

    def test_tempo_is_unitless(self, result) -> None:
        assert result.get(MetricName.TEMPO_RATIO, None).unit is MetricUnit.RATIO

    def test_timing_is_the_only_basis_that_is_not_a_projection(self, result) -> None:
        for metric in result.metrics:
            assert (metric.basis is MetricBasis.TEMPORAL) == (
                metric.unit in (MetricUnit.SECONDS, MetricUnit.RATIO)
            )

    @staticmethod
    def _downswing_at(fps: float):
        """The downswing duration measured from the same swing sampled at `fps`."""
        t = np.arange(0.0, DURATION_S, 1.0 / fps)
        _, frames = _swing_frames()
        indices = np.clip((t * FPS).astype(int), 0, len(frames) - 1)
        measured = _measure(_sequence(t, [frames[index] for index in indices]))
        found = measured.get(MetricName.DOWNSWING_DURATION, None)
        assert found is not None, f"no downswing duration at {fps} fps"
        return found

    def test_a_duration_resolves_better_at_a_higher_frame_rate(self) -> None:
        """One frame of uncertainty is a bigger fraction of a shorter phase.

        60 rather than 30 fps as the slow end, because at the filter's shipped
        defaults a 30 fps clip yields no trajectory at all -- the limitation
        Phase 3 measured, not something this layer can compensate for.
        """
        slow, fast = self._downswing_at(60.0), self._downswing_at(240.0)
        assert fast.confidence.method > slow.confidence.method


class TestConfidence:
    """5.6 -- propagation, and what each factor means."""

    def test_overall_is_the_product_of_its_factors(self, result) -> None:
        for metric in result.metrics:
            factors = metric.confidence
            assert factors.overall == pytest.approx(
                factors.observation * factors.anchor * factors.method, abs=1e-9
            )

    def test_the_anchor_factor_is_the_events_own_confidence(self, result) -> None:
        """The propagation: a metric is never worth more than the instant it claims."""
        filtered = filter_sequence(_swing(), FilterConfig())
        phases = detect_phases(filtered)
        top = phases.event(SwingEvent.TOP)

        at_top = [m for m in result.metrics if m.event is SwingEvent.TOP]
        assert at_top
        for metric in at_top:
            assert metric.confidence.anchor == pytest.approx(top.confidence.overall, abs=1e-9)

    def test_a_poorly_seen_landmark_lowers_observation_without_moving_the_value(self) -> None:
        t, frames = _swing_frames()
        clear = _measure(_sequence(t, frames))
        dim = _measure(_sequence(t, frames, visibility=0.55))

        for name in (MetricName.LEFT_KNEE_FLEX,):
            bright = clear.get(name, SwingEvent.TOP)
            faded = dim.get(name, SwingEvent.TOP)
            assert faded.value == pytest.approx(bright.value, abs=0.5)
            assert faded.confidence.observation < bright.confidence.observation

    def test_a_foreshortened_segment_scores_a_lower_method_factor(self) -> None:
        """An angle read across a segment seen nearly end-on is noisier."""
        square = _measure(_swing())
        oblique = _measure(_swing(shoulder_span=SHOULDER_SPAN * 0.4))

        assert (
            oblique.get(MetricName.SHOULDER_TILT, SwingEvent.TOP).confidence.method
            <= square.get(MetricName.SHOULDER_TILT, SwingEvent.TOP).confidence.method
        )

    def test_a_small_turn_is_less_sharply_resolved_than_a_large_one(self) -> None:
        """arccos conditioning: d(theta)/d(span) goes as 1/sin(theta)."""
        big = _measure(_swing())
        small = _measure(_swing(shoulder_turn_top=8.0))
        assert (
            small.get(MetricName.SHOULDER_TURN, SwingEvent.TOP).confidence.method
            < big.get(MetricName.SHOULDER_TURN, SwingEvent.TOP).confidence.method
        )


class TestRefusal:
    """What produces nothing, and whether it says why."""

    def test_no_swing_means_no_metrics(self) -> None:
        t = np.arange(0.0, 2.0, 1.0 / FPS)
        _, frames = _swing_frames()
        still = [frames[0]] * t.size
        measured = _measure(_sequence(t, still))

        assert not measured.computed
        assert measured.metrics == []
        assert any("No swing was detected" in warning for warning in measured.warnings)

    def test_a_refusal_names_every_metric_it_could_not_produce(self) -> None:
        t = np.arange(0.0, 2.0, 1.0 / FPS)
        _, frames = _swing_frames()
        measured = _measure(_sequence(t, [frames[0]] * t.size))
        assert {entry.name for entry in measured.refused} == set(REGISTRY)

    def test_a_gated_out_landmark_refuses_the_metrics_that_need_it(self) -> None:
        """Rather than measuring an angle from a position nobody observed."""
        t, frames = _swing_frames()
        measured = _measure(_sequence(t, frames, hidden=(Landmark.LEFT_ANKLE,)))

        assert measured.get(MetricName.LEFT_KNEE_FLEX, SwingEvent.TOP) is None
        assert any(entry.name is MetricName.LEFT_KNEE_FLEX for entry in measured.refused)
        # And the ones that do not need it are unaffected.
        assert measured.get(MetricName.RIGHT_KNEE_FLEX, SwingEvent.TOP) is not None

    def test_hip_local_space_is_refused(self) -> None:
        """Displacements there are measured from an origin that moves with the body."""
        from analyzer.biomechanics import BodyError

        filtered = filter_sequence(_swing(), FilterConfig(), space=LandmarkSpace.HIP_LOCAL)
        with pytest.raises(BodyError, match="HIP_LOCAL"):
            body_from(filtered)


class TestRegistry:
    """5.6 -- the declaration is the single source of a metric's identity."""

    def test_every_name_has_a_definition(self) -> None:
        assert set(REGISTRY) == set(MetricName)

    def test_every_emitted_metric_matches_its_declaration(self, result) -> None:
        """A unit stated next to a computation drifts; one looked up cannot."""
        for metric in result.metrics:
            declared = REGISTRY[metric.name]
            assert metric.unit is declared.unit
            assert metric.basis is declared.basis
            assert metric.group is declared.group

    def test_every_metric_carries_a_methodology(self, result) -> None:
        assert all(metric.methodology for metric in result.metrics)

    def test_every_metric_reports_the_frames_it_came_from(self, result) -> None:
        for metric in result.metrics:
            assert metric.source_frames
            assert all(0 <= frame < result.frames for frame in metric.source_frames)

    def test_an_address_metric_reduces_over_the_whole_address_phase(self, result) -> None:
        """The subject is still there, so the median beats any single frame."""
        spine = result.get(MetricName.SPINE_TILT, None)
        assert spine.phase is SwingPhase.ADDRESS
        assert len(spine.source_frames) > 1

    def test_an_event_metric_reads_exactly_one_frame(self, result) -> None:
        """The top is an extreme; averaging around it reports something smaller."""
        spine = result.get(MetricName.SPINE_TILT, SwingEvent.TOP)
        assert len(spine.source_frames) == 1

    def test_labels_name_the_instant_except_where_the_metric_already_does(self, result) -> None:
        assert result.get(MetricName.SPINE_TILT, SwingEvent.TOP).label.endswith("at the top")
        assert result.get(MetricName.TEMPO_RATIO, None).label == "Tempo ratio"


class TestAspectCorrection:
    """The defect carried in from Phase 4, and the evidence it is gone.

    The body is built in frame widths, so the same body written into a portrait
    frame and into its transpose is the same body. Without the correction the
    two disagree by tens of degrees, because a vertical distance counts for
    0.5625 of a horizontal one on a 16:9 frame and for 1.7778 of one on its
    transpose -- a factor of 3.16 between the two readings of one shape.
    """

    LANDSCAPE = FrameGeometry(width=1920, height=1080)

    def test_angles_do_not_depend_on_the_shape_of_the_frame(self) -> None:
        portrait = _measure(_swing())
        landscape = _measure(_swing(geometry=self.LANDSCAPE))

        for name in (MetricName.SPINE_TILT, MetricName.SHOULDER_TILT):
            assert _value(landscape, name, None) == pytest.approx(
                _value(portrait, name, None), abs=1.0
            )

    def test_the_torso_scale_does_not_depend_on_the_shape_of_the_frame(self) -> None:
        portrait = _measure(_swing())
        landscape = _measure(_swing(geometry=self.LANDSCAPE))
        assert landscape.torso_length == pytest.approx(portrait.torso_length, abs=0.005)

    def test_a_turn_does_not_depend_on_the_shape_of_the_frame(self) -> None:
        """Foreshortening is a ratio of two spans, so it survives either way --
        but only once both spans are measured in the same isotropic unit."""
        portrait = _measure(_swing())
        landscape = _measure(_swing(geometry=self.LANDSCAPE))
        assert _value(landscape, MetricName.SHOULDER_TURN, SwingEvent.TOP) == pytest.approx(
            _value(portrait, MetricName.SHOULDER_TURN, SwingEvent.TOP), abs=1.0
        )
