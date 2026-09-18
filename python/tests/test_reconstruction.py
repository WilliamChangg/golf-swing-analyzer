"""Reconstruction, against a body whose 3D positions are known by construction.

The fixture in `tests/synthetic_body3d.py` is the ground truth: every point the
reconstruction is asked to recover is a number put into it, so accuracy here is a
subtraction. What it does not contain is a pose estimator, so there is no motion
blur, no occluded hip and no frame where the wrists swap -- the errors measured
here are a **floor**.

The tests that matter most are the ones asserting a **negative**, as in Phase 8:

* that displacing a landmark *along the epipolar line* leaves the reprojection
  error flat while the true 3D error grows without bound, so a reader who gates
  on the residual is gating on the wrong number;
* that the bone-length check sees exactly that displacement, which is why it is
  reported beside the residual rather than instead of it;
* that a shallow ray convergence angle is refused even though its residual is
  fine, because the residual is fine *because* the geometry is degenerate.

Those three are the phase's whole argument, and they fail if anyone later decides
the reprojection error is the quality of a reconstruction.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from analyzer.biomechanics import compute_metrics
from analyzer.contracts.calibration import CalibrationStatus, CameraRig
from analyzer.contracts.camera import CameraRole
from analyzer.contracts.filtering import FilterConfig, SignalUnit, unit_for
from analyzer.contracts.metrics import MetricBasis, MetricName
from analyzer.contracts.phases import SwingEvent
from analyzer.contracts.pose import POSE_CONNECTIONS, Landmark, LandmarkSpace
from analyzer.contracts.reconstruction import ReconstructionConfig, RefusalReason
from analyzer.contracts.sync import TimeMap
from analyzer.coordinates import CoordinateError, require_reachable
from analyzer.filtering.landmarks import FilteredSequence, filter_sequence
from analyzer.phases import detect_phases
from analyzer.reconstruction import (
    ReconstructionError,
    StereoGeometry,
    convergence_angles,
    measured_pixel_sigma,
    positional_uncertainty,
    reconstruct_pair,
    reprojection_errors,
    require_stereo_rig,
    skeleton,
    triangulate,
    velocity_from_image,
)
from analyzer.reconstruction import pairing as pairing_module
from analyzer.reconstruction.triangulate import TriangulationError, triangulate_linear
from tests.synthetic_body3d import (
    DURATION_S,
    PELVIS_TURN_DEG,
    SHOULDER_TURN_DEG,
    TOP_S,
    SwingRig,
    body_at,
    camera_rig,
    default_rig,
    pose_sequence_for,
)

# --- helpers ---------------------------------------------------------------


def _filtered(rig: SwingRig, view_name: str, **kwargs: object) -> FilteredSequence:
    view = getattr(rig, view_name)
    contract = camera_rig(rig)
    camera = contract.cameras[view.role]
    sequence = pose_sequence_for(view, **kwargs)  # type: ignore[arg-type]
    return filter_sequence(sequence, FilterConfig(), intrinsics=camera.intrinsics)


def _time_map(rig: SwingRig, uncertainty_s: float = 0.001) -> TimeMap:
    return TimeMap(
        offset_s=rig.time_offset_s(),
        rate=1.0,
        rate_estimated=False,
        pivot_s=TOP_S,
        offset_uncertainty_s=uncertainty_s,
        support_start_s=0.5,
        support_end_s=2.0,
    )


def _reconstruct(rig: SwingRig, *, config: ReconstructionConfig | None = None, **noise: object):
    reference = _filtered(rig, "reference", **noise)
    target = _filtered(rig, "target", seed=99, **noise)
    return reconstruct_pair(
        reference,
        target,
        camera_rig(rig),
        _time_map(rig),
        reference_role=rig.reference.role,
        target_role=rig.target.role,
        config=config,
    )


def _errors_against_truth(
    rig: SwingRig, result: object, landmark: Landmark | None = None
) -> np.ndarray:
    """Distance from the reconstructed point to the true one, in metres.

    `landmark` narrows to one, which matters more than it looks: most of the
    fixture's body is deliberately static, so a statistic over all 33 landmarks
    is dominated by parts that do not move and is blind to every error that
    scales with speed. The hands are where those live.
    """
    truth = rig.reference.to_camera(
        np.stack([body_at(rig.reference.world_start_s + float(t)) for t in result.t])  # type: ignore[attr-defined]
    )
    distance = np.linalg.norm(result.points - truth, axis=2)  # type: ignore[attr-defined]
    if landmark is not None:
        distance = distance[:, int(landmark)]
    return distance[np.isfinite(distance)]


@pytest.fixture(scope="module")
def rig() -> SwingRig:
    return default_rig()


@pytest.fixture(scope="module")
def clean(rig: SwingRig):
    return _reconstruct(rig)


# --- the fixture has to be ground truth before anything is measured against it


def test_limb_lengths_are_constant_through_the_synthetic_swing() -> None:
    """The bone check has nothing to check unless the fixture's bones are rigid.

    Solved by inverse kinematics rather than interpolated, so the arm segments
    are exact. Torso segments are not asserted here: a shoulder-to-hip distance
    genuinely changes as the shoulders and pelvis turn different amounts, which
    is anatomy rather than fixture error and is why `max_bone_variation` is not
    tighter than it is.
    """
    limbs = [
        pair
        for pair in POSE_CONNECTIONS
        if not {"SHOULDER", "HIP"} <= {end.name.split("_")[-1] for end in pair}
    ]
    samples = np.stack([body_at(float(t)) for t in np.linspace(0.0, DURATION_S, 60)])

    for first, second in limbs:
        lengths = np.linalg.norm(samples[:, first] - samples[:, second], axis=1)
        assert np.ptp(lengths) < 1e-9, f"{first.name}-{second.name} changes length in the fixture"


def test_left_and_right_limbs_are_the_same_length() -> None:
    """Otherwise the symmetry check would be measuring the fixture's asymmetry."""
    body = body_at(0.9)
    pairs = (
        (
            Landmark.LEFT_SHOULDER,
            Landmark.LEFT_ELBOW,
            Landmark.RIGHT_SHOULDER,
            Landmark.RIGHT_ELBOW,
        ),
        (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
        (Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    )
    for a, b, c, d in pairs:
        left = np.linalg.norm(body[a] - body[b])
        right = np.linalg.norm(body[c] - body[d])
        assert left == pytest.approx(right, abs=1e-9)


def test_the_cameras_are_the_right_way_up(rig: SwingRig) -> None:
    """A camera built with the cross products transposed is self-consistent and wrong.

    Both cameras being upside down leaves triangulation exact -- the reference
    frame is just rotated -- so nothing in this phase notices. What notices is
    Phase 4: the hands reach their lowest point at the top of the backswing and
    the swing is refused. This asserts the convention directly rather than
    waiting for that symptom.
    """
    for view in (rig.reference, rig.target):
        low = view.project(np.array([[0.0, 0.5, 0.0]]))
        high = view.project(np.array([[0.0, 1.5, 0.0]]))
        assert high[0, 1] < low[0, 1], f"{view.role.value} sees the world upside down"


def test_the_whole_body_stays_inside_both_frames(rig: SwingRig) -> None:
    for view in (rig.reference, rig.target):
        for instant in np.linspace(0.0, DURATION_S, 40):
            pixels = view.project(body_at(view.world_start_s + float(instant)))
            assert np.all(pixels[:, 0] > 0) and np.all(pixels[:, 0] < view.intrinsics.image_width)
            assert np.all(pixels[:, 1] > 0) and np.all(pixels[:, 1] < view.intrinsics.image_height)


# --- triangulation numerics ------------------------------------------------


def test_a_projected_point_triangulates_back_to_itself(rig: SwingRig) -> None:
    geometry = StereoGeometry(
        reference=rig.reference.intrinsics,
        target=rig.target.intrinsics,
        rotation=rig.relative_rotation,
        translation=rig.relative_translation,
    )
    points = rig.reference.to_camera(body_at(0.9))
    # Projected without distortion, because the pipeline undistorts below the
    # filter and the model this layer inverts is the pinhole one.
    first, second = geometry.project(points)

    recovered = triangulate(first, second, geometry)
    assert np.allclose(recovered, points, atol=1e-9)


def test_an_unseen_landmark_stays_unseen(rig: SwingRig) -> None:
    geometry = StereoGeometry(
        reference=rig.reference.intrinsics,
        target=rig.target.intrinsics,
        rotation=rig.relative_rotation,
        translation=rig.relative_translation,
    )
    points = rig.reference.to_camera(body_at(0.9))
    first, second = geometry.project(points)
    first[3] = np.nan

    recovered = triangulate(first, second, geometry)
    assert np.all(np.isnan(recovered[3]))
    assert np.all(np.isfinite(recovered[4]))


def test_mismatched_point_counts_are_refused(rig: SwingRig) -> None:
    geometry = StereoGeometry(
        reference=rig.reference.intrinsics,
        target=rig.target.intrinsics,
        rotation=rig.relative_rotation,
        translation=rig.relative_translation,
    )
    with pytest.raises(TriangulationError, match="pairs them one for one"):
        triangulate_linear(np.zeros((4, 2)), np.zeros((3, 2)), geometry)


def test_refinement_beats_the_linear_solution_under_noise(rig: SwingRig) -> None:
    """The DLT minimises an algebraic quantity; the refinement minimises the real one."""
    geometry = StereoGeometry(
        reference=rig.reference.intrinsics,
        target=rig.target.intrinsics,
        rotation=rig.relative_rotation,
        translation=rig.relative_translation,
    )
    generator = np.random.default_rng(3)
    truth = rig.reference.to_camera(
        np.stack([body_at(float(t)) for t in np.linspace(0.4, 2.2, 30)])
    ).reshape(-1, 3)
    first, second = geometry.project(truth)
    first = first + generator.normal(0.0, 2.0, first.shape)
    second = second + generator.normal(0.0, 2.0, second.shape)

    linear = triangulate(first, second, geometry, refine_points=False)
    refined = triangulate(first, second, geometry, refine_points=True)

    assert np.median(reprojection_errors(refined, first, second, geometry)) < np.median(
        reprojection_errors(linear, first, second, geometry)
    )


def test_convergence_is_the_angle_between_the_rays(rig: SwingRig) -> None:
    """A right-angled rig meets at a right angle where it is aimed."""
    geometry = StereoGeometry(
        reference=rig.reference.intrinsics,
        target=rig.target.intrinsics,
        rotation=rig.relative_rotation,
        translation=rig.relative_translation,
    )
    aim = rig.reference.to_camera(np.array([[0.0, 1.15, 0.0]]))
    assert convergence_angles(aim, geometry)[0] == pytest.approx(90.0, abs=1.0)


def test_uncertainty_grows_as_the_rays_close_up() -> None:
    """The 1/sin law, measured rather than asserted in prose.

    Halving the sine of the convergence angle should roughly double the
    positional uncertainty, and this is the relationship the whole gate rests
    on. It is checked across a wide sweep rather than at one pair of angles, so
    a change that broke the scaling could not pass by being right somewhere.
    """
    ratios = []
    for angle in (90.0, 30.0, 15.0):
        rig = default_rig(convergence_deg=angle)
        geometry = StereoGeometry(
            reference=rig.reference.intrinsics,
            target=rig.target.intrinsics,
            rotation=rig.relative_rotation,
            translation=rig.relative_translation,
        )
        aim = rig.reference.to_camera(np.array([[0.0, 1.15, 0.0]]))
        spread = positional_uncertainty(aim, geometry, sigma_px=1.0)[0]
        ratios.append(spread * np.sin(np.radians(convergence_angles(aim, geometry)[0])))

    # sigma / sin(theta) is the model, so multiplying back by sin should leave a
    # constant. Loose, because the exact covariance also carries the two focal
    # lengths and the distances, which do move a little across the sweep.
    assert max(ratios) / min(ratios) < 1.6


def test_velocity_comes_from_the_fit_and_matches_the_truth(rig: SwingRig, clean) -> None:
    """A reconstructed speed is the two views' fitted image velocities, carried through."""
    truth = rig.reference.to_camera(
        np.stack([body_at(rig.reference.world_start_s + float(t)) for t in clean.t])
    )
    column = int(Landmark.LEFT_WRIST)
    differenced = np.linalg.norm(np.gradient(truth[:, column, :], clean.t, axis=0), axis=1)
    reconstructed = clean.speed(Landmark.LEFT_WRIST)

    usable = np.isfinite(reconstructed)
    assert np.median(np.abs(reconstructed[usable] - differenced[usable])) < 0.01
    assert np.nanmax(reconstructed) == pytest.approx(differenced.max(), rel=0.02)


def test_velocity_is_nan_where_the_point_is(rig: SwingRig) -> None:
    geometry = StereoGeometry(
        reference=rig.reference.intrinsics,
        target=rig.target.intrinsics,
        rotation=rig.relative_rotation,
        translation=rig.relative_translation,
    )
    points = np.array([[0.0, 0.0, 3.0], [np.nan, np.nan, np.nan]])
    velocity = velocity_from_image(points, np.zeros((2, 2)), np.zeros((2, 2)), geometry)
    assert np.all(np.isfinite(velocity[0]))
    assert np.all(np.isnan(velocity[1]))


# --- accuracy on the synthetic truth ---------------------------------------


def test_a_clean_pair_reconstructs_to_under_a_millimetre(rig: SwingRig, clean) -> None:
    errors = _errors_against_truth(rig, clean)
    assert np.median(errors) < 1e-3
    assert clean.report.reconstructed
    assert clean.space is LandmarkSpace.CAMERA


def test_landmark_noise_produces_proportionate_error(rig: SwingRig) -> None:
    """Two pixels of scatter should cost millimetres, not centimetres, at 90 degrees."""
    noisy = _reconstruct(rig, noise_px=2.0)
    errors = _errors_against_truth(rig, noisy)
    assert 1e-3 < np.median(errors) < 0.02


# --- the phase's argument: the residual is blind along the epipolar line ----


def test_epipolar_displacement_moves_the_answer_and_not_the_residual(rig: SwingRig) -> None:
    """**The central negative.** A landmark displaced along the epipolar line
    reprojects perfectly and sits at the wrong depth.

    The reprojection error stays flat across a sweep over which the true error
    grows by an order of magnitude. Anyone gating a reconstruction on its
    residual is gating on a number that cannot see the failure.
    """
    residuals: list[float] = []
    errors: list[float] = []

    for displacement in (0.0, 2.0, 6.0):
        reference = _filtered(rig, "reference")
        target = _filtered(
            rig,
            "target",
            epipolar_noise_px=displacement,
            epipolar_towards=rig.reference,
        )
        result = reconstruct_pair(
            reference,
            target,
            camera_rig(rig),
            _time_map(rig),
            reference_role=rig.reference.role,
            target_role=rig.target.role,
            # The bounds would refuse the worst points, and refusing them is what
            # hides the effect: this test is about what the residual reports for
            # the points it accepts.
            config=ReconstructionConfig(max_uncertainty_m=None, max_reprojection_px=1e9),
        )
        residuals.append(result.report.quality.median_reprojection_px)
        errors.append(float(np.median(_errors_against_truth(rig, result))))

    assert errors[2] > 10 * max(errors[0], 1e-4), "epipolar noise should move the answer"
    assert residuals[2] < 1.0, "and should barely register in the residual"
    assert residuals[2] < 0.25 * errors[2] * 1000, (
        "the residual in pixels must not track the error in millimetres -- if it "
        "does, this test has stopped measuring the thing it exists for"
    )


def test_isotropic_noise_does_register_in_the_residual(rig: SwingRig) -> None:
    """The residual is not useless: it sees the component it can see.

    The complement of the test above, and the reason the residual is still
    reported and still gates. Noise across the epipolar line has no 3D
    explanation, so it lands in the reprojection error.
    """
    quiet = _reconstruct(rig, noise_px=0.0)
    loud = _reconstruct(rig, noise_px=3.0)
    assert loud.report.quality.median_reprojection_px > 5 * max(
        quiet.report.quality.median_reprojection_px, 1e-6
    )


def test_bone_length_sees_what_the_residual_cannot(rig: SwingRig) -> None:
    """The independent check, on exactly the displacement the residual misses."""
    reference = _filtered(rig, "reference")
    target = _filtered(rig, "target", epipolar_noise_px=6.0, epipolar_towards=rig.reference)
    result = reconstruct_pair(
        reference,
        target,
        camera_rig(rig),
        _time_map(rig),
        reference_role=rig.reference.role,
        target_role=rig.target.role,
        config=ReconstructionConfig(max_uncertainty_m=None, max_reprojection_px=1e9),
    )

    assert result.report.quality.median_reprojection_px < 1.0
    assert result.report.quality.worst_bone_variation > 0.10
    assert any(not entry.stable for entry in result.report.quality.bones)
    assert any("does not change length" in note for note in result.report.warnings)


def test_a_depth_bias_reaches_variation_before_symmetry(rig: SwingRig) -> None:
    """The measurement that corrected this module's own design.

    Symmetry was built expecting to catch what variation could not: a landmark
    placed consistently deep was supposed to give a *stably* wrong length. On a
    rotating body it does not -- a displacement constant in the camera's frame is
    not constant relative to the bone -- so variation gets there first, and by a
    long way. Asserted in that order so that restoring the stronger claim fails
    here.
    """
    frames = 40
    times = np.linspace(0.6, 2.0, frames)
    truth = rig.reference.to_camera(np.stack([body_at(float(t)) for t in times]))

    biased = truth.copy()
    column = int(Landmark.LEFT_ELBOW)
    biased[:, column, 2] += 0.05  # five centimetres along the optical axis, every frame

    valid = np.ones((frames, len(Landmark)), dtype=bool)
    config = ReconstructionConfig()
    bones = skeleton.bone_consistency(biased, valid, tuple(Landmark), config)
    by_name = {entry.name: entry for entry in bones}
    checks = {entry.segment: entry for entry in skeleton.symmetry(bones, config)}

    assert not by_name["left_elbow-left_wrist"].stable, "variation sees a depth bias"
    assert checks["elbow-wrist"].agrees, "and symmetry, on the same bias, does not yet"
    assert checks["elbow-wrist"].disagreement < by_name["left_elbow-left_wrist"].variation, (
        "variation is the more sensitive instrument here, which is not what was expected"
    )


def test_symmetry_localises_a_one_sided_reconstruction(rig: SwingRig) -> None:
    """What symmetry does add: it names the side, which a per-segment number cannot."""
    frames = 40
    times = np.linspace(0.6, 2.0, frames)
    truth = rig.reference.to_camera(np.stack([body_at(float(t)) for t in times]))

    # Only the elbow. Displacing a whole chain by one vector leaves every bone
    # inside it exactly the right length, which is a fact about rigid motion and
    # not a reconstruction this check could ever see.
    biased = truth.copy()
    biased[:, int(Landmark.LEFT_ELBOW), 2] += 0.15

    valid = np.ones((frames, len(Landmark)), dtype=bool)
    config = ReconstructionConfig()
    bones = skeleton.bone_consistency(biased, valid, tuple(Landmark), config)
    checks = {entry.segment: entry for entry in skeleton.symmetry(bones, config)}

    assert not checks["shoulder-elbow"].agrees
    assert checks["shoulder-elbow"].left_m > checks["shoulder-elbow"].right_m, (
        "and it names which side, which a per-segment variation does not"
    )


def test_shallow_convergence_is_refused_although_its_residual_is_fine() -> None:
    """The gate is the geometry, and it is checked before the residual.

    Two nearly-parallel rays agree with each other beautifully, because a point
    slid a long way in depth still reprojects onto both. Refusing on the residual
    would accept this capture; refusing on the angle does not.
    """
    rig = default_rig(convergence_deg=6.0)
    result = _reconstruct(rig, config=ReconstructionConfig(min_convergence_deg=15.0))

    refused = [
        entry.refused.get(RefusalReason.ILL_CONDITIONED, 0) for entry in result.report.landmarks
    ]
    assert sum(refused) > 0, "a six-degree pair should be refused as ill-conditioned"

    permissive = _reconstruct(
        rig,
        config=ReconstructionConfig(
            min_convergence_deg=0.1, max_uncertainty_m=None, max_reprojection_px=1e9
        ),
    )
    assert permissive.report.quality.median_reprojection_px < 1.0, (
        "and its residual is fine, which is the point"
    )


# --- pairing ---------------------------------------------------------------


def test_resampling_reproduces_a_sample_exactly(rig: SwingRig) -> None:
    track = pairing_module.track_from(_filtered(rig, "target"))
    sampled = pairing_module.resample(track, track.t[5:10])

    assert np.allclose(sampled.pixels[0], track.pixels[5], atol=1e-9, equal_nan=True)
    assert np.allclose(sampled.velocity[0], track.velocity[5], atol=1e-6, equal_nan=True)


def test_resampling_refuses_outside_the_clip(rig: SwingRig) -> None:
    track = pairing_module.track_from(_filtered(rig, "target"))
    outside = np.array([track.t[0] - 1.0, track.t[-1] + 1.0])
    sampled = pairing_module.resample(track, outside)

    assert not np.any(sampled.inside)
    assert not np.any(sampled.valid)


def test_resampling_beats_nearest_frame_on_a_moving_landmark(rig: SwingRig) -> None:
    """The measurement that decided the default.

    Phase 8 pairs nearest frames and gets away with it because a board can be
    held still. A hand cannot, and the cost is the frame interval times its image
    speed.
    """
    track = pairing_module.track_from(_filtered(rig, "target"))
    # Half-way between samples, which is where nearest-frame pairing is worst.
    queries = (track.t[:-1] + track.t[1:]) / 2.0
    exact = pairing_module.resample(track, queries)
    rounded = pairing_module.nearest(track, queries)

    column = track.landmarks.index(Landmark.LEFT_WRIST)
    both = exact.valid[:, column] & rounded.valid[:, column]
    gap = np.linalg.norm(exact.pixels[both, column] - rounded.pixels[both, column], axis=1)
    assert np.max(gap) > 5.0, "nearest-frame pairing costs pixels on a swinging hand"


def test_nearest_frame_pairing_costs_real_accuracy(rig: SwingRig) -> None:
    """Measured at a **hand**, because that is the only place the cost lives.

    The error from pairing the wrong instant is the time error times the
    landmark's speed, so on a body that is mostly standing still it is nothing
    almost everywhere. A statistic over all 33 landmarks reports that the
    difference is negligible, which is true of the fixture's shins and false of
    the thing every metric is measured from.
    """
    resampled = _reconstruct(rig)
    rounded = _reconstruct(rig, config=ReconstructionConfig(resample=False))

    exact = _errors_against_truth(rig, resampled, Landmark.LEFT_WRIST)
    approximate = _errors_against_truth(rig, rounded, Landmark.LEFT_WRIST)

    # Absolute bounds rather than a ratio, because the magnitudes are the fact:
    # on this rig the hand is ~18 mm out when paired to a neighbouring frame and
    # under a fifth of a millimetre when the trajectory is resampled -- and that
    # residual is the cubic interpolant's own error at the sharpest part of the
    # arc, not the pairing's.
    assert np.max(approximate) > 0.010, "a hand paired to the wrong frame is centimetres out"
    assert np.max(exact) < 0.001, "and a resampled one is under a millimetre"
    assert rounded.report.pairing.method == "nearest_frame"

    # And over the whole body it looks like almost nothing, which is why the
    # summary quotes the fastest landmark rather than the median one.
    assert np.median(_errors_against_truth(rig, rounded)) < 1e-5


def test_the_sync_cost_is_reported_in_pixels(rig: SwingRig, clean) -> None:
    """Phase 8's conversion, on a subject that cannot be held still."""
    pairing = clean.report.pairing
    assert pairing.max_pairing_error_px == pytest.approx(
        (pairing.sync_uncertainty_ms / 1000.0) * pairing.max_landmark_speed_px_s, rel=1e-6
    )
    assert pairing.nearest_frame_error_px > pairing.max_pairing_error_px


def test_pixels_and_frame_widths_round_trip(rig: SwingRig) -> None:
    filtered = _filtered(rig, "reference")
    track = pairing_module.track_from(filtered)
    entry = filtered[Landmark.NOSE]
    column = track.landmarks.index(Landmark.NOSE)

    width = float(filtered.geometry.width)
    height = float(filtered.geometry.height)
    back_x = track.pixels[:, column, 0] / width
    back_y = (height - track.pixels[:, column, 1]) / width

    assert np.allclose(back_x, entry.position[:, 0], equal_nan=True)
    assert np.allclose(back_y, entry.position[:, 1], equal_nan=True)


# --- gates and refusals ----------------------------------------------------


def test_an_uncalibrated_project_is_refused_by_name(rig: SwingRig) -> None:
    with pytest.raises(ReconstructionError, match="relative pose measured"):
        reconstruct_pair(
            _filtered(rig, "reference"),
            _filtered(rig, "target"),
            CameraRig(),
            _time_map(rig),
            reference_role=rig.reference.role,
            target_role=rig.target.role,
        )


def test_the_rig_can_be_refused_without_looking_at_any_footage(rig: SwingRig) -> None:
    """The check that lets the dispatcher refuse before it does any work.

    Without it, a reconstruction extracts poses for two clips and fits an
    alignment between them before discovering the cameras were never calibrated,
    and then reports the last thing that went wrong -- "no extracted poses for
    this clip" -- when the real blocker was knowable from a single database read.
    This asserts the cheap check needs nothing but the rig.
    """
    with pytest.raises(ReconstructionError, match="this project has 'none'"):
        require_stereo_rig(CameraRig(), rig.reference.role, rig.target.role)

    with pytest.raises(ReconstructionError, match="this project has 'none'"):
        require_stereo_rig(None, rig.reference.role, rig.target.role)

    # And it passes on a rig that is genuinely stereo, so it cannot refuse the
    # good case cheaply and leave the real work unreachable.
    require_stereo_rig(camera_rig(rig), rig.reference.role, rig.target.role)


def test_a_rig_measured_for_another_pair_of_roles_is_refused(rig: SwingRig) -> None:
    with pytest.raises(ReconstructionError, match="and this reconstruction is between"):
        require_stereo_rig(camera_rig(rig), rig.reference.role, CameraRole.OTHER)


def test_an_unaligned_pair_is_refused_by_name(rig: SwingRig) -> None:
    with pytest.raises(ReconstructionError, match="clocks have not been related"):
        reconstruct_pair(
            _filtered(rig, "reference"),
            _filtered(rig, "target"),
            camera_rig(rig),
            None,
            reference_role=rig.reference.role,
            target_role=rig.target.role,
        )


def test_a_calibration_for_another_frame_size_is_refused(rig: SwingRig) -> None:
    contract = camera_rig(rig)
    camera = contract.cameras[rig.reference.role]
    contract.cameras[rig.reference.role] = camera.model_copy(
        update={"intrinsics": camera.intrinsics.model_copy(update={"image_width": 1280})}
    )

    with pytest.raises(ReconstructionError, match="was measured on 1280x"):
        reconstruct_pair(
            _filtered(rig, "reference"),
            _filtered(rig, "target"),
            contract,
            _time_map(rig),
            reference_role=rig.reference.role,
            target_role=rig.target.role,
        )


def test_frames_outside_the_overlap_are_counted_as_such(rig: SwingRig, clean) -> None:
    """The target camera stops rolling first, and those frames are named, not lost."""
    outside = sum(
        entry.refused.get(RefusalReason.OUTSIDE_OVERLAP, 0) for entry in clean.report.landmarks
    )
    assert outside > 0
    assert clean.report.pairing.outside_overlap > 0


def test_a_hidden_landmark_is_refused_as_unseen(rig: SwingRig) -> None:
    reference = _filtered(rig, "reference", visibility=0.1)
    target = _filtered(rig, "target")
    result = reconstruct_pair(
        reference,
        target,
        camera_rig(rig),
        _time_map(rig),
        reference_role=rig.reference.role,
        target_role=rig.target.role,
    )
    assert not result.report.reconstructed
    assert result.report.refusal is not None


def test_the_uncertainty_bound_refuses_rather_than_reporting(rig: SwingRig) -> None:
    result = _reconstruct(
        rig, noise_px=2.0, config=ReconstructionConfig(pixel_sigma_px=40.0, max_uncertainty_m=0.05)
    )
    refused = sum(
        entry.refused.get(RefusalReason.UNCERTAIN, 0) for entry in result.report.landmarks
    )
    assert refused > 0


def test_camera_coordinates_cannot_be_read_from_one_clip() -> None:
    """CAMERA is produced by this build and is still not a conversion.

    Reading it off a stored sequence would mean inventing the depth the
    projection destroyed, so `landmark_series` refuses it -- and the message
    names triangulation rather than a phase number.
    """
    with pytest.raises(CoordinateError, match="triangulated from two calibrated views"):
        require_reachable(LandmarkSpace.CAMERA)


def test_world_coordinates_name_what_they_are_missing() -> None:
    with pytest.raises(CoordinateError, match="gravity direction and a target line"):
        require_reachable(LandmarkSpace.WORLD)


def test_camera_space_has_a_real_metre_unit() -> None:
    assert unit_for(LandmarkSpace.CAMERA, 0) is SignalUnit.M
    assert unit_for(LandmarkSpace.CAMERA, 1) is SignalUnit.M_PER_S
    assert unit_for(LandmarkSpace.HIP_LOCAL, 0) is SignalUnit.APPROX_M


# --- pixel sigma -----------------------------------------------------------


def test_pixel_sigma_is_measured_from_the_clip(rig: SwingRig) -> None:
    quiet, measured_quiet = measured_pixel_sigma(_filtered(rig, "reference"), tuple(Landmark))
    loud, measured_loud = measured_pixel_sigma(
        _filtered(rig, "reference", noise_px=3.0), tuple(Landmark)
    )
    assert measured_quiet and measured_loud
    assert loud > 5 * quiet


def test_a_zero_residual_falls_back_rather_than_claiming_perfection(rig: SwingRig) -> None:
    """A residual of exactly zero is what an exactly-determined fit produces.

    It is not evidence that the landmarks were perfect, so it is not used as
    one -- the fallback is the scatter Phase 3 measured on real footage.
    """
    filtered = _filtered(rig, "reference")
    for entry in filtered.landmarks.values():
        entry.report.residual_rms[:] = [0.0, 0.0, 0.0]

    sigma, measured = measured_pixel_sigma(filtered, tuple(Landmark))
    assert not measured
    assert sigma == pytest.approx(0.0014 * filtered.geometry.width)


# --- the time map inverts --------------------------------------------------


def test_a_time_map_inverts(rig: SwingRig) -> None:
    forward = TimeMap(
        offset_s=0.37,
        rate=1.012,
        rate_estimated=True,
        pivot_s=1.4,
        offset_uncertainty_s=0.004,
        rate_uncertainty=0.003,
        support_start_s=0.5,
        support_end_s=2.1,
    )
    backward = forward.inverse()

    for instant in (0.0, 0.8, 1.4, 2.6):
        assert backward.to_target(forward.to_target(instant)) == pytest.approx(instant, abs=1e-12)

    # The uncertainty at one instant is the same fact from either side.
    assert backward.uncertainty_at(forward.to_target(1.4)) == pytest.approx(
        forward.uncertainty_at(1.4), rel=0.05
    )


# --- metrics: the gate Phase 8 built now blocks something -------------------


@pytest.fixture(scope="module")
def measured(rig: SwingRig, clean):
    filtered = _filtered(rig, "reference")
    phases = detect_phases(filtered)
    assert phases.detected, "the synthetic swing must be detected for the metrics to anchor"
    return (
        filtered,
        phases,
        compute_metrics(filtered, phases, None, CalibrationStatus.STEREO, clean),
    )


def test_spatial_metrics_recover_the_true_rotations(measured) -> None:
    """The reason a second camera is worth owning, measured against known angles."""
    _, _, result = measured

    shoulder = result.get(MetricName.SHOULDER_TURN_3D, SwingEvent.TOP)
    pelvis = result.get(MetricName.PELVIS_TURN_3D, SwingEvent.TOP)
    x_factor = result.get(MetricName.X_FACTOR_3D, SwingEvent.TOP)

    assert shoulder is not None and pelvis is not None and x_factor is not None
    assert shoulder.value == pytest.approx(SHOULDER_TURN_DEG, abs=2.0)
    assert pelvis.value == pytest.approx(PELVIS_TURN_DEG, abs=2.0)
    assert x_factor.value == pytest.approx(SHOULDER_TURN_DEG - PELVIS_TURN_DEG, abs=3.0)
    assert x_factor.basis is MetricBasis.SPATIAL


def test_a_spatial_metric_carries_an_uncertainty_in_degrees(measured) -> None:
    _, _, result = measured
    shoulder = result.get(MetricName.SHOULDER_TURN_3D, SwingEvent.TOP)
    assert shoulder.uncertainty is not None
    assert 0.0 < shoulder.uncertainty < 5.0
    # And the error really does sit inside a few of those.
    assert abs(shoulder.value - SHOULDER_TURN_DEG) < 5 * shoulder.uncertainty


def test_the_method_factor_is_the_measured_conditioning(measured) -> None:
    """Not a chosen constant: the sine of the ray convergence angle where measured."""
    _, _, result = measured
    shoulder = result.get(MetricName.SHOULDER_TURN_3D, SwingEvent.TOP)
    assert shoulder.confidence.method > 0.9, "a right-angled pair is well conditioned"

    narrow_rig = default_rig(convergence_deg=25.0)
    narrow = _reconstruct(narrow_rig, config=ReconstructionConfig(min_convergence_deg=5.0))
    filtered = _filtered(narrow_rig, "reference")
    phases = detect_phases(filtered)
    narrow_metrics = compute_metrics(filtered, phases, None, CalibrationStatus.STEREO, narrow)
    narrow_shoulder = narrow_metrics.get(MetricName.SHOULDER_TURN_3D, SwingEvent.TOP)
    assert narrow_shoulder is not None
    assert narrow_shoulder.confidence.method < shoulder.confidence.method


def test_peak_hand_speed_is_in_metres_per_second(measured) -> None:
    _, _, result = measured
    speed = next(
        metric for metric in result.metrics if metric.name is MetricName.PEAK_HAND_SPEED_3D
    )
    assert speed.unit.value == "metres_per_s"
    assert 1.0 < speed.value < 20.0


def test_without_a_reconstruction_the_spatial_metrics_are_refused(measured) -> None:
    filtered, phases, _ = measured
    result = compute_metrics(filtered, phases, None, CalibrationStatus.STEREO, None)

    assert not [m for m in result.metrics if m.basis is MetricBasis.SPATIAL]
    reasons = {entry.name: entry.reason for entry in result.refused}
    assert "No 3D reconstruction was supplied" in reasons[MetricName.X_FACTOR_3D]


def test_an_uncalibrated_project_refuses_them_naming_the_calibration(measured) -> None:
    """The gate Phase 8 built, enforced and tested against nothing, now blocks something."""
    filtered, phases, _ = measured
    result = compute_metrics(filtered, phases, None, CalibrationStatus.NONE, None)

    reasons = {entry.name: entry.reason for entry in result.refused}
    assert "needs a stereo calibration" in reasons[MetricName.X_FACTOR_3D]
    assert "this project has none" in reasons[MetricName.X_FACTOR_3D]


def test_intrinsics_alone_are_not_enough(measured) -> None:
    filtered, phases, _ = measured
    result = compute_metrics(filtered, phases, None, CalibrationStatus.INTRINSICS, None)
    reasons = {entry.name: entry.reason for entry in result.refused}
    assert "this project has intrinsics" in reasons[MetricName.SHOULDER_TURN_3D]


def test_a_view_that_refuses_the_projected_rotation_still_supports_the_3d_one(
    rig: SwingRig,
) -> None:
    """The difference a second camera makes, stated as a test.

    A down-the-line clip is refused `SHOULDER_TURN` because that view does not
    contain it. With a calibrated partner, `SHOULDER_TURN_3D` is available from
    the very same footage -- which is the whole argument for the two names being
    different names.
    """
    reference = _filtered(rig, "target")  # the down-the-line camera, as the reference
    target = _filtered(rig, "reference")
    reconstruction = reconstruct_pair(
        reference,
        target,
        camera_rig(rig),
        _time_map(rig).inverse(),
        reference_role=rig.target.role,
        target_role=rig.reference.role,
    )
    phases = detect_phases(reference)
    if not phases.detected:
        pytest.skip("the down-the-line view does not detect a swing on this fixture")

    result = compute_metrics(reference, phases, None, CalibrationStatus.STEREO, reconstruction)
    assert result.view.view.value in {"down_the_line", "unknown"}
    assert result.get(MetricName.SHOULDER_TURN_3D, SwingEvent.TOP) is not None


def test_every_spatial_metric_declares_the_calibration_it_needs() -> None:
    """Exhaustive, so a metric added later cannot forget to declare one."""
    from analyzer.biomechanics.registry import REGISTRY

    for name, entry in REGISTRY.items():
        if entry.basis is MetricBasis.SPATIAL:
            assert entry.requires is CalibrationStatus.STEREO, name
            assert not entry.refused_in, f"{name} is reconstructed, so no camera view refuses it"
            assert entry.meaning, f"{name} must state what it means, and it means one thing"


def test_reconstruction_config_is_carried_on_the_report(rig: SwingRig, clean) -> None:
    assert clean.report.config.min_convergence_deg == ReconstructionConfig().min_convergence_deg
    assert clean.report.calibration is CalibrationStatus.STEREO
    assert clean.report.baseline_m == pytest.approx(rig.baseline_m, rel=1e-9)


def test_the_report_states_that_this_is_not_a_world_frame(clean) -> None:
    assert clean.report.space is LandmarkSpace.CAMERA
    assert any("not a scene-fixed frame" in note for note in clean.report.warnings)


def test_dataclass_fields_stay_immutable(clean) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        clean.t = np.zeros(3)  # type: ignore[misc]
