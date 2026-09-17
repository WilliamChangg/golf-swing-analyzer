"""Calibration, against a camera whose parameters are known by construction.

The board views are rendered by `tests/synthetic_board.py` and detected by the
real detector, so these exercise the whole path -- render, detect, select, fit,
judge -- rather than the arithmetic alone. What they do not exercise is
everything that makes real footage hard (blur, defocus, a sheet that is not
flat), so the tolerances here are tight in a way a real capture will not match.

The tests that matter most are the ones asserting a **negative**: that a
degenerate capture fits just as well as a good one and is refused anyway. Those
are the phase's whole argument, and they fail if anyone later decides to gate on
the reprojection error alone.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from analyzer.biomechanics.compute import _apply_calibration_gate
from analyzer.biomechanics.registry import REGISTRY
from analyzer.calibration.apply import (
    CalibrationMismatchError,
    bearings,
    distortion_displacement,
    require_applicable,
    undistort_normalized,
    undistort_pixels,
)
from analyzer.calibration.board import (
    BoardError,
    board_texture,
    build,
    generate_board_image,
    object_points,
    validate,
)
from analyzer.calibration.detect import (
    detect_in_directory,
    detect_in_image,
    select_distinct,
)
from analyzer.calibration.intrinsics import CalibrationError, calibrate_intrinsics, coverage
from analyzer.calibration.stereo import (
    StereoError,
    board_speed,
    calibrate_stereo,
    pair_by_time_map,
    pair_explicitly,
)
from analyzer.contracts.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    BoardFamily,
    BoardSpec,
    CalibrationConfig,
    CalibrationStatus,
    CameraIntrinsics,
    CameraRig,
    DistortionModel,
)
from analyzer.contracts.camera import CameraRole
from analyzer.contracts.pose import FrameGeometry
from analyzer.contracts.sync import TimeMap
from analyzer.projects.store import _SCHEMA, ProjectError, ProjectStore
from tests.synthetic_board import (
    DEFAULT_BOARD,
    BoardPose,
    SyntheticCamera,
    default_camera,
    degenerate_poses,
    stereo_rig,
    well_spread_poses,
)

# Rendering is the expensive part of these tests, so the two capture sets that
# most of them need are built once for the module.


@pytest.fixture(scope="module")
def camera() -> SyntheticCamera:
    return SyntheticCamera()


def _views(camera: SyntheticCamera, poses: list[BoardPose]) -> list:
    found = []
    for index, pose in enumerate(poses):
        view = detect_in_image(camera.render(pose), camera.spec, frame=index)
        if view is not None:
            found.append(view)
    return found


@pytest.fixture(scope="module")
def spread_views(camera: SyntheticCamera) -> list:
    return _views(camera, well_spread_poses(14))


@pytest.fixture(scope="module")
def degenerate_views(camera: SyntheticCamera) -> list:
    return _views(camera, degenerate_poses(14))


# --- the board -----------------------------------------------------------


def test_marker_larger_than_square_is_refused() -> None:
    spec = BoardSpec(squares_x=5, squares_y=4, square_length_m=0.02, marker_length_m=0.02)
    with pytest.raises(BoardError, match="cannot fit inside"):
        validate(spec)


def test_board_too_large_for_its_dictionary_is_refused() -> None:
    # A 12x10 board needs 60 markers; DICT_4X4_50 has 50.
    spec = BoardSpec(
        squares_x=12,
        squares_y=10,
        square_length_m=0.03,
        marker_length_m=0.022,
        family=BoardFamily.DICT_4X4_50,
    )
    with pytest.raises(BoardError, match="needs 60 markers"):
        validate(spec)


def test_object_points_are_planar_and_metric() -> None:
    corners = object_points(DEFAULT_BOARD)
    assert corners.shape == (DEFAULT_BOARD.interior_corners, 3)
    # The board is a plane. Everything downstream assumes it.
    assert np.allclose(corners[:, 2], 0.0)
    # And it is in metres: the corner span is one square short of the full board.
    span = corners[:, 0].max() - corners[:, 0].min()
    assert span == pytest.approx(DEFAULT_BOARD.width_m - 2 * DEFAULT_BOARD.square_length_m)


def test_board_texture_reports_the_size_it_actually_rendered() -> None:
    # 1400x1000 is a size OpenCV refuses outright; the retry must report what it
    # settled on, because the caller maps metres onto those pixels.
    image, actual = board_texture(DEFAULT_BOARD, board_px=(1400, 1000), margin_px=24)
    assert actual[0] >= 1400
    assert image.shape == (actual[1] + 48, actual[0] + 48)


def test_generated_board_detects_as_itself() -> None:
    """The generator and the detector must describe one object."""
    image = generate_board_image(DEFAULT_BOARD, pixels_per_metre=3000.0)
    view = detect_in_image(image, DEFAULT_BOARD)
    assert view is not None
    assert len(view) == DEFAULT_BOARD.interior_corners


def test_legacy_pattern_flag_changes_the_board() -> None:
    """A pre-4.6 board is a different object, and the flag must reach OpenCV."""
    modern = build(DEFAULT_BOARD)
    legacy = build(DEFAULT_BOARD.model_copy(update={"legacy_pattern": True}))
    assert modern.getLegacyPattern() is False
    assert legacy.getLegacyPattern() is True


def test_board_needs_a_margin() -> None:
    with pytest.raises(BoardError, match="white margin"):
        board_texture(DEFAULT_BOARD, board_px=(700, 500), margin_px=0)


# --- the fixture's own check ---------------------------------------------


def test_rendered_corners_match_the_analytic_projection(camera: SyntheticCamera) -> None:
    """The renderer and the camera model must agree, or every test below is vacuous.

    A transposed or flipped texture mapping still renders something that looks
    like a board, and the calibration fitted from it would still converge -- on
    the wrong answer. This is the check that catches that.
    """
    pose = BoardPose(0.6, (25.0, -15.0, 8.0), (0.05, -0.02))
    view = detect_in_image(camera.render(pose), camera.spec)
    assert view is not None

    expected = camera.project_corners(pose)
    identifiers = view.corner_ids.ravel()
    error = np.linalg.norm(view.corners_px.reshape(-1, 2) - expected[identifiers], axis=1)
    assert float(error.mean()) < 0.6


# --- detection -----------------------------------------------------------


def test_no_board_is_not_a_detection(camera: SyntheticCamera) -> None:
    blank = np.full((1080, 1920), 128, dtype=np.uint8)
    assert detect_in_image(blank, camera.spec) is None


def test_colour_and_grayscale_agree(camera: SyntheticCamera) -> None:
    gray = camera.render(BoardPose(0.6, (15.0, 10.0, 0.0), (0.0, 0.0)))
    colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    from_gray = detect_in_image(gray, camera.spec)
    from_colour = detect_in_image(colour, camera.spec)
    assert from_gray is not None and from_colour is not None
    assert np.allclose(from_gray.corners_px, from_colour.corners_px)


def test_selection_drops_repeats_and_keeps_movement(camera: SyntheticCamera) -> None:
    still = detect_in_image(
        camera.render(BoardPose(0.6, (10.0, 0.0, 0.0), (0.0, 0.0))), camera.spec
    )
    moved = detect_in_image(
        camera.render(BoardPose(0.6, (10.0, 0.0, 0.0), (0.12, 0.05))), camera.spec
    )
    nearer = detect_in_image(
        camera.render(BoardPose(0.35, (10.0, 0.0, 0.0), (0.0, 0.0))), camera.spec
    )
    assert still is not None and moved is not None and nearer is not None

    # The same view twice is one view.
    assert len(select_distinct([still, still, still])) == 1
    # Moved across the frame is a new view.
    assert len(select_distinct([still, moved])) == 2
    # And so is the same place at a different distance, which distance alone
    # would reject -- that is what the scale term is for.
    assert len(select_distinct([still, nearer])) == 2


def test_selection_respects_its_limit(camera: SyntheticCamera) -> None:
    views = _views(camera, well_spread_poses(14))
    assert len(select_distinct(views, limit=5)) == 5


def test_empty_directory_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No images"):
        detect_in_directory(tmp_path, DEFAULT_BOARD)


def test_directory_detection_keeps_every_still(tmp_path: Path, camera: SyntheticCamera) -> None:
    """A directory is a set someone assembled, so selection is off by default."""
    for index, pose in enumerate(well_spread_poses(6)):
        cv2.imwrite(str(tmp_path / f"{index:02d}.png"), camera.render(pose))

    views, report = detect_in_directory(tmp_path, camera.spec)
    assert report.frames_scanned == 6
    assert len(views) == report.views_used == report.frames_with_board


# --- intrinsics ----------------------------------------------------------


def test_well_spread_capture_recovers_the_camera(
    camera: SyntheticCamera, spread_views: list
) -> None:
    result = calibrate_intrinsics(spread_views, camera.spec, role=CameraRole.FACE_ON, source="test")
    truth = camera.intrinsics
    found = result.intrinsics

    assert result.usable
    assert found.fx == pytest.approx(truth.fx, rel=0.01)
    assert found.fy == pytest.approx(truth.fy, rel=0.01)
    assert found.cx == pytest.approx(truth.cx, abs=12.0)
    assert found.cy == pytest.approx(truth.cy, abs=12.0)
    assert found.distortion[0] == pytest.approx(truth.distortion[0], abs=0.02)
    assert result.quality.rms_reprojection_px < 0.5


def test_a_degenerate_capture_fits_as_well_and_is_refused_anyway(
    camera: SyntheticCamera, spread_views: list, degenerate_views: list
) -> None:
    """The phase's central claim, as an assertion.

    A board held square to the camera at one distance cannot separate focal
    length from distance. The fit is just as good -- better, usually -- and the
    focal length is wrong by tens of percent. If anyone ever gates on the
    reprojection error alone, this test is what fails.
    """
    good = calibrate_intrinsics(spread_views, camera.spec, role=CameraRole.FACE_ON, source="good")
    bad = calibrate_intrinsics(degenerate_views, camera.spec, role=CameraRole.FACE_ON, source="bad")

    # The residual cannot tell them apart.
    assert bad.quality.rms_reprojection_px < good.quality.rms_reprojection_px * 1.5

    # The answer is wrong by a wide margin.
    truth = camera.intrinsics.fx
    assert abs(bad.intrinsics.fx - truth) / truth > 0.10

    # And it is refused, naming the tilt rather than the residual.
    assert not bad.usable
    assert bad.refusal is not None
    assert "tilt" in bad.refusal.lower() or "orientation" in bad.refusal.lower()


def test_reported_parameter_uncertainty_does_not_catch_it(
    camera: SyntheticCamera, degenerate_views: list
) -> None:
    """The second half of the same finding, and the more surprising one.

    OpenCV's own standard deviation for the focal length is *smaller* on the
    degenerate capture than on the good one, because the distortion coefficients
    absorb the degeneracy and leave a tightly determined wrong answer. So the
    uncertainty is reported but is not what the gate can rest on -- coverage is.
    """
    bad = calibrate_intrinsics(degenerate_views, camera.spec, role=CameraRole.FACE_ON, source="bad")
    assert bad.intrinsics.fx_uncertainty is not None
    ratio = bad.intrinsics.fx_uncertainty / bad.intrinsics.fx
    assert ratio < CalibrationConfig().max_focal_uncertainty_ratio


def test_coverage_separates_the_two_captures(
    camera: SyntheticCamera, spread_views: list, degenerate_views: list
) -> None:
    good = calibrate_intrinsics(
        spread_views, camera.spec, role=CameraRole.FACE_ON, source="good"
    ).quality.coverage
    bad = calibrate_intrinsics(
        degenerate_views, camera.spec, role=CameraRole.FACE_ON, source="bad"
    ).quality.coverage

    assert good.tilt_range_deg > 20.0 > bad.tilt_range_deg
    assert good.image_fraction > 2 * bad.image_fraction
    assert good.scale_range > bad.scale_range
    assert good.edge_fraction > bad.edge_fraction


def test_too_few_views_is_an_error_not_a_bad_calibration(
    camera: SyntheticCamera, spread_views: list
) -> None:
    with pytest.raises(CalibrationError, match="are required"):
        calibrate_intrinsics(spread_views[:3], camera.spec, role=CameraRole.FACE_ON, source="test")


def test_mixed_frame_sizes_are_refused(camera: SyntheticCamera, spread_views: list) -> None:
    """Intrinsics are in pixels, so one set cannot describe two frame sizes."""
    mixed = list(spread_views)
    mixed[0] = dataclasses.replace(mixed[0], image_size=(1280, 720))
    with pytest.raises(CalibrationError, match="different sizes"):
        calibrate_intrinsics(mixed, camera.spec, role=CameraRole.FACE_ON, source="test")


@pytest.mark.parametrize(
    ("model", "count"),
    [
        (DistortionModel.PINHOLE, 0),
        (DistortionModel.RADIAL_TANGENTIAL_4, 4),
        (DistortionModel.RADIAL_TANGENTIAL_5, 5),
    ],
)
def test_distortion_model_controls_what_is_fitted(
    camera: SyntheticCamera, spread_views: list, model: DistortionModel, count: int
) -> None:
    result = calibrate_intrinsics(
        spread_views,
        camera.spec,
        role=CameraRole.FACE_ON,
        source="test",
        config=CalibrationConfig(distortion_model=model),
    )
    assert len(result.intrinsics.distortion) == count
    assert result.intrinsics.model is model


def test_a_pinhole_camera_is_recovered_exactly() -> None:
    """With no lens to measure, the fit should return what went in."""
    truth = default_camera(distortion=(0.0, 0.0, 0.0, 0.0))
    camera = SyntheticCamera(truth)
    views = _views(camera, well_spread_poses(14))

    result = calibrate_intrinsics(
        views,
        camera.spec,
        role=CameraRole.FACE_ON,
        source="test",
        config=CalibrationConfig(distortion_model=DistortionModel.PINHOLE),
    )
    assert result.intrinsics.fx == pytest.approx(truth.fx, rel=0.005)
    assert result.intrinsics.cx == pytest.approx(truth.cx, abs=5.0)


def test_coverage_of_nothing_is_not_an_error() -> None:
    report = coverage([], (1920, 1080), [])
    assert report.views == 0
    assert report.tilt_range_deg == 0.0
    assert report.scale_range == 1.0


def test_observations_carry_per_view_results(camera: SyntheticCamera, spread_views: list) -> None:
    """8.4 draws these, so they have to survive the fit."""
    result = calibrate_intrinsics(spread_views, camera.spec, role=CameraRole.FACE_ON, source="test")
    used = [entry for entry in result.detection.observations if entry.used]
    assert len(used) == len(spread_views)
    assert all(entry.reprojection_rms_px is not None for entry in used)
    assert all(entry.tilt_deg is not None for entry in used)
    # Distances are metric, from the board's own square size.
    assert all(0.2 < (entry.distance_m or 0.0) < 2.0 for entry in used)


# --- applying a calibration ----------------------------------------------


def test_undistortion_inverts_the_lens(camera: SyntheticCamera) -> None:
    """Undistorting a detected corner must land it where a pinhole would see it."""
    pose = BoardPose(0.6, (20.0, -10.0, 0.0), (0.10, 0.04))
    view = detect_in_image(camera.render(pose), camera.spec)
    assert view is not None

    truth = camera.intrinsics
    pinhole = truth.model_copy(update={"distortion": [0.0, 0.0, 0.0, 0.0]})
    rvec, tvec = pose.rvec_tvec(camera.spec)
    ideal, _ = cv2.projectPoints(
        object_points(camera.spec), rvec, tvec, pinhole.matrix(), pinhole.distortion_vector()
    )

    corrected = undistort_pixels(view.corners_px.reshape(-1, 2), truth)
    expected = ideal.reshape(-1, 2)[view.corner_ids.ravel()]
    assert float(np.linalg.norm(corrected - expected, axis=1).mean()) < 1.0


def test_undistortion_keeps_absence_absent() -> None:
    intrinsics = default_camera()
    points = np.array([[100.0, 200.0], [np.nan, np.nan], [1800.0, 1000.0]])
    result = undistort_pixels(points, intrinsics)
    assert np.isnan(result[1]).all()
    assert np.isfinite(result[0]).all()
    assert np.isfinite(result[2]).all()


def test_undistortion_leaves_the_z_channel_alone() -> None:
    """MediaPipe's z is not a measurement and must not acquire geometry here."""
    intrinsics = default_camera()
    geometry = FrameGeometry(width=1920, height=1080)
    points = np.array([[0.9, 0.9, 0.42], [0.5, 0.5, -0.1]])
    result = undistort_normalized(points, intrinsics, geometry)
    assert np.allclose(result[:, 2], points[:, 2])
    # And the corner really did move.
    assert not np.allclose(result[0, :2], points[0, :2])


def test_a_calibration_is_refused_on_a_different_frame_size() -> None:
    intrinsics = default_camera(1920, 1080)
    require_applicable(intrinsics, FrameGeometry(width=1920, height=1080))
    with pytest.raises(CalibrationMismatchError, match="1920x1080"):
        require_applicable(intrinsics, FrameGeometry(width=1280, height=720))


def test_bearings_are_unit_vectors_pointing_forward() -> None:
    intrinsics = default_camera()
    points = np.array([[intrinsics.cx, intrinsics.cy], [100.0, 100.0], [1800.0, 1000.0]])
    rays = bearings(points, intrinsics)

    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0)
    # Everything in front of the camera has a positive z component.
    assert np.all(rays[:, 2] > 0)
    # The principal point looks straight down the optical axis.
    assert rays[0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_distortion_displacement_grows_with_the_lens() -> None:
    mild = distortion_displacement(default_camera(distortion=(-0.02, 0.0, 0.0, 0.0)))
    strong = distortion_displacement(default_camera(distortion=(-0.42, 0.22, 0.0, 0.0)))
    assert strong[0] > mild[0] > 0.0
    assert strong[1] >= strong[0]


# --- the status type, and the rig ----------------------------------------


def test_status_is_ordered() -> None:
    assert CalibrationStatus.STEREO.at_least(CalibrationStatus.INTRINSICS)
    assert CalibrationStatus.STEREO.at_least(CalibrationStatus.NONE)
    assert not CalibrationStatus.INTRINSICS.at_least(CalibrationStatus.STEREO)
    assert CalibrationStatus.NONE.at_least(CalibrationStatus.NONE)


def _usable_calibration(camera: SyntheticCamera, views: list, role: CameraRole):
    return calibrate_intrinsics(views, camera.spec, role=role, source="test")


def test_rig_status_needs_every_piece(camera: SyntheticCamera, spread_views: list) -> None:
    calibration = _usable_calibration(camera, spread_views, CameraRole.FACE_ON)
    assert CameraRig().status() is CalibrationStatus.NONE

    rig = CameraRig(cameras={CameraRole.FACE_ON: calibration})
    assert rig.status() is CalibrationStatus.INTRINSICS
    assert rig.status(CameraRole.FACE_ON) is CalibrationStatus.INTRINSICS
    # A camera that is not in the rig is not calibrated.
    assert rig.status(CameraRole.DOWN_THE_LINE) is CalibrationStatus.NONE


def test_an_unusable_calibration_does_not_count(
    camera: SyntheticCamera, degenerate_views: list
) -> None:
    bad = _usable_calibration(camera, degenerate_views, CameraRole.FACE_ON)
    assert not bad.usable
    rig = CameraRig(cameras={CameraRole.FACE_ON: bad})
    assert rig.status() is CalibrationStatus.NONE
    assert rig.usable_camera(CameraRole.FACE_ON) is None
    # It is still readable, because the numbers and the refusal are both useful.
    assert rig.camera(CameraRole.FACE_ON) is not None


def test_a_calibration_does_not_apply_to_another_frame_size(
    camera: SyntheticCamera, spread_views: list
) -> None:
    rig = CameraRig(
        cameras={CameraRole.FACE_ON: _usable_calibration(camera, spread_views, CameraRole.FACE_ON)}
    )
    assert rig.status_for_frame(CameraRole.FACE_ON, 1920, 1080) is CalibrationStatus.INTRINSICS
    assert rig.status_for_frame(CameraRole.FACE_ON, 1280, 720) is CalibrationStatus.NONE


def test_intrinsics_report_a_checkable_field_of_view() -> None:
    """The one intrinsic a person can sanity-check without any tooling."""
    phone = CameraIntrinsics(
        fx=1400.0,
        fy=1400.0,
        cx=960.0,
        cy=540.0,
        distortion=[],
        model=DistortionModel.PINHOLE,
        image_width=1920,
        image_height=1080,
    )
    assert 60.0 < phone.horizontal_fov_deg < 75.0
    assert phone.focal_disagreement == 0.0


# --- the metric-layer gate (8.5) -----------------------------------------


def test_every_shipped_metric_is_measurable_without_a_calibration() -> None:
    """Phase 5 and 6 measure the image plane, which an uncalibrated camera gives.

    If this ever fails it means a metric was added that claims three dimensions,
    and the question to ask is whether it should exist rather than whether the
    gate should be relaxed.
    """
    assert all(entry.requires is CalibrationStatus.NONE for entry in REGISTRY.values())


def test_the_gate_blocks_nothing_at_present() -> None:
    produced, refused = _apply_calibration_gate([], [], CalibrationStatus.NONE)
    assert produced == []
    assert refused == []


def test_the_gate_refuses_a_metric_that_needs_more_than_is_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate is enforced now, against a registry entry that demands stereo.

    No shipped metric requires one, so this stands in for Phase 9's. It is the
    test that makes the gate real rather than decorative.
    """
    from analyzer.biomechanics import compute
    from analyzer.contracts.metrics import MetricName

    target = MetricName.SHOULDER_TURN
    patched = dict(REGISTRY)
    patched[target] = dataclasses.replace(REGISTRY[target], requires=CalibrationStatus.STEREO)
    monkeypatch.setattr(compute, "REGISTRY", patched)
    monkeypatch.setattr("analyzer.biomechanics.registry.REGISTRY", patched)

    produced, refused = compute._apply_calibration_gate([], [], CalibrationStatus.INTRINSICS)
    assert [entry.name for entry in refused] == [target]
    assert "stereo" in refused[0].reason

    # And it passes once the rig supplies what the metric asked for.
    produced, refused = compute._apply_calibration_gate([], [], CalibrationStatus.STEREO)
    assert refused == []
    assert produced == []


def test_the_calibration_state_is_always_stated() -> None:
    """A reader seeing no refusals must not conclude the geometry was known."""
    from analyzer.biomechanics.compute import _calibration_warnings

    for status in CalibrationStatus:
        warnings = _calibration_warnings(status)
        assert len(warnings) == 1
        assert warnings[0]
    assert "not calibrated" in _calibration_warnings(CalibrationStatus.NONE)[0]
    assert "image plane" in _calibration_warnings(CalibrationStatus.INTRINSICS)[0]


# --- stereo ---------------------------------------------------------------


def _stereo_capture(
    rig, stations: int = 8, hold_s: float = 1.0, offset_s: float = 0.4, seed: int = 5
):
    """Render both cameras' views of a board moved between stations."""
    rng = np.random.default_rng(seed)
    reference_views, target_views = [], []
    clock = 0.0
    for station in range(stations):
        pose = BoardPose(
            distance_m=float(rng.uniform(1.0, 1.6)),
            rotation_deg=(
                float(rng.uniform(-30, 30)),
                float(rng.uniform(-30, 30)),
                float(rng.uniform(-20, 20)),
            ),
            offset_m=(float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.08, 0.08))),
        )
        left, right = rig.render_pair(pose)
        reference = detect_in_image(left, rig.reference.spec, frame=station, timestamp_s=clock)
        target = detect_in_image(
            right, rig.target.spec, frame=station, timestamp_s=clock + offset_s
        )
        if reference is not None:
            reference_views.append(reference)
        if target is not None:
            target_views.append(target)
        clock += hold_s
    return reference_views, target_views, offset_s, clock


@pytest.fixture(scope="module")
def rig():
    return stereo_rig()


@pytest.fixture(scope="module")
def stereo_views(rig):
    return _stereo_capture(rig)


def _time_map(offset_s: float, span_s: float, uncertainty_s: float) -> TimeMap:
    return TimeMap(
        offset_s=offset_s,
        rate=1.0,
        rate_estimated=False,
        pivot_s=span_s / 2,
        offset_uncertainty_s=uncertainty_s,
        support_start_s=0.0,
        support_end_s=span_s,
    )


def test_stereo_recovers_the_rig(rig, stereo_views) -> None:
    reference_views, target_views, offset_s, span = stereo_views
    pairs, dropped = pair_by_time_map(
        reference_views, target_views, _time_map(offset_s, span, 0.0), CalibrationConfig()
    )
    assert len(pairs) >= CalibrationConfig().min_stereo_pairs

    result = calibrate_stereo(
        pairs,
        rig.reference.spec,
        rig.reference.intrinsics,
        rig.target.intrinsics,
        reference_role=CameraRole.FACE_ON,
        target_role=CameraRole.DOWN_THE_LINE,
        dropped=dropped,
    )

    truth_baseline = float(np.linalg.norm(rig.translation_m))
    assert result.usable
    assert result.baseline_m == pytest.approx(truth_baseline, rel=0.02)

    rotation_error = np.degrees(
        np.arccos(np.clip((np.trace(result.rotation_matrix() @ rig.rotation.T) - 1) / 2, -1, 1))
    )
    assert float(rotation_error) < 1.0
    assert result.convergence_deg == pytest.approx(
        float(np.degrees(np.arccos(rig.rotation[2, 2]))), abs=1.0
    )


def test_sync_uncertainty_becomes_a_pixel_cost(rig) -> None:
    """The mechanism that lets two unsynchronised cameras be calibrated at all.

    A board moved quickly between frames makes the pairing expensive; the same
    clock uncertainty over a still board costs nothing. The pairs that survive
    are what differs, and nothing about the images does.
    """
    config = CalibrationConfig()

    still_reference, still_target, offset, span = _stereo_capture(rig, hold_s=2.0)
    still_pairs, _ = pair_by_time_map(
        still_reference, still_target, _time_map(offset, span, 0.02), config
    )

    waved_reference, waved_target, offset, span = _stereo_capture(rig, hold_s=0.03)
    waved_pairs, waved_dropped = pair_by_time_map(
        waved_reference, waved_target, _time_map(offset, span, 0.02), config
    )

    assert len(still_pairs) > len(waved_pairs)
    assert any("still" in entry for entry in waved_dropped)
    if still_pairs:
        assert max(pair.pairing_error_px for pair in still_pairs) < config.max_pairing_error_px


def test_pairs_too_far_apart_in_time_are_dropped(rig, stereo_views) -> None:
    reference_views, target_views, offset_s, span = stereo_views
    # Wrong by 0.4 s, against stations 1 s apart: the nearest target view is
    # still the right one, and it is 400 ms away, past the 50 ms bound.
    pairs, dropped = pair_by_time_map(
        reference_views, target_views, _time_map(offset_s + 0.4, span, 0.0), CalibrationConfig()
    )
    assert pairs == []
    assert dropped
    assert all("ms away" in entry for entry in dropped)


def test_an_aliased_pairing_is_caught_by_the_residual_not_by_the_clock(rig, stereo_views) -> None:
    """The one failure the pairing's own numbers cannot see, and what does see it.

    Board stations a second apart, with an offset wrong by exactly a second,
    pair every reference view with its *neighbour*: each pair is simultaneous to
    the millisecond and shows the board in two different places. No time-based
    check can detect that -- the clocks agree perfectly.

    What detects it is the fit. A point located by one camera then lands nowhere
    near where the other camera sees it, so the residual explodes and the
    calibration is refused. That is the argument for gating stereo on the
    reprojection error even though gating *intrinsics* on it is exactly what
    this phase says not to do: here it is measuring a correspondence rather than
    a model's fit to its own data.
    """
    reference_views, target_views, offset_s, span = stereo_views
    pairs, dropped = pair_by_time_map(
        reference_views, target_views, _time_map(offset_s + 1.0, span, 0.0), CalibrationConfig()
    )
    # The mis-pairing goes undetected here: the clocks agree.
    assert len(pairs) >= CalibrationConfig().min_stereo_pairs
    assert all(pair.time_error_s < 0.001 for pair in pairs)

    result = calibrate_stereo(
        pairs,
        rig.reference.spec,
        rig.reference.intrinsics,
        rig.target.intrinsics,
        reference_role=CameraRole.FACE_ON,
        target_role=CameraRole.DOWN_THE_LINE,
        dropped=dropped,
    )
    assert not result.usable
    assert result.refusal is not None
    assert "same moment" in result.refusal


def test_too_few_pairs_is_an_error(rig, stereo_views) -> None:
    reference_views, target_views, offset_s, span = stereo_views
    pairs, _ = pair_by_time_map(
        reference_views, target_views, _time_map(offset_s, span, 0.0), CalibrationConfig()
    )
    with pytest.raises(StereoError, match="are required"):
        calibrate_stereo(
            pairs[:2],
            rig.reference.spec,
            rig.reference.intrinsics,
            rig.target.intrinsics,
            reference_role=CameraRole.FACE_ON,
            target_role=CameraRole.DOWN_THE_LINE,
        )


def test_explicit_pairing_does_not_claim_a_measured_time_error(rig, stereo_views) -> None:
    """A caller asserting simultaneity is not a measurement of it."""
    reference_views, target_views, _, _ = stereo_views
    frame_pairs = [
        (view.frame, view.frame)
        for view in reference_views
        if view.frame in {other.frame for other in target_views}
    ]
    pairs, dropped = pair_explicitly(
        reference_views, target_views, frame_pairs, CalibrationConfig()
    )
    assert len(pairs) >= CalibrationConfig().min_stereo_pairs
    assert dropped == []

    result = calibrate_stereo(
        pairs,
        rig.reference.spec,
        rig.reference.intrinsics,
        rig.target.intrinsics,
        reference_role=CameraRole.FACE_ON,
        target_role=CameraRole.DOWN_THE_LINE,
        method="explicit",
    )
    # None, not zero: zero would read as perfect synchronisation having been
    # observed, and nothing here observed anything.
    assert result.pairing.median_time_error_ms is None
    assert result.pairing.worst_pairing_error_px is None


def test_explicit_pairing_names_a_missing_frame(rig, stereo_views) -> None:
    reference_views, target_views, _, _ = stereo_views
    pairs, dropped = pair_explicitly(reference_views, target_views, [(999, 0)], CalibrationConfig())
    assert pairs == []
    assert "reference frame 999" in dropped[0]


def test_board_speed_of_a_single_view_is_zero(rig, stereo_views) -> None:
    """Honest rather than convenient: one view is no evidence of stillness."""
    reference_views, _, _, _ = stereo_views
    assert board_speed(reference_views[:1], 0) == 0.0


# --- storage --------------------------------------------------------------


def test_a_rig_round_trips(tmp_path: Path, camera: SyntheticCamera, spread_views: list) -> None:
    store = ProjectStore(tmp_path / "projects.db")
    project = store.create_project("calibrated")
    assert project.rig is None

    rig = CameraRig(
        cameras={CameraRole.FACE_ON: _usable_calibration(camera, spread_views, CameraRole.FACE_ON)}
    )
    saved = store.save_rig(project.id, rig)
    assert saved.rig is not None
    assert saved.rig.status() is CalibrationStatus.INTRINSICS

    reopened = ProjectStore(tmp_path / "projects.db").get_project(project.id)
    assert reopened.rig is not None
    assert reopened.rig.cameras[CameraRole.FACE_ON].intrinsics.fx == pytest.approx(
        rig.cameras[CameraRole.FACE_ON].intrinsics.fx
    )


def test_clearing_a_rig_leaves_the_project(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.db")
    project = store.create_project("t")
    store.save_rig(project.id, CameraRig())
    cleared = store.clear_rig(project.id)
    assert cleared.rig is None
    assert cleared.name == "t"


def test_a_rig_from_a_future_build_is_skipped_not_guessed_at(tmp_path: Path) -> None:
    """A stale rig costs the calibration and nothing else."""
    path = tmp_path / "projects.db"
    store = ProjectStore(path)
    project = store.create_project("t")
    store.save_rig(project.id, CameraRig())

    connection = store._connect()
    with connection:
        connection.execute(
            "UPDATE rigs SET calibration_schema_version = ?",
            (CALIBRATION_SCHEMA_VERSION + 1,),
        )

    reopened = ProjectStore(path).get_project(project.id)
    assert reopened.rig is None
    assert any("calibration schema version" in warning for warning in reopened.warnings)


def test_an_unparseable_rig_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "projects.db"
    store = ProjectStore(path)
    project = store.create_project("t")
    store.save_rig(project.id, CameraRig())

    connection = store._connect()
    with connection:
        connection.execute("UPDATE rigs SET model_json = ?", ('{"cameras": "not a mapping"}',))

    reopened = ProjectStore(path).get_project(project.id)
    assert reopened.rig is None
    assert any("did not parse" in warning for warning in reopened.warnings)


def test_a_version_one_database_is_migrated_rather_than_refused(tmp_path: Path) -> None:
    """A project cannot be recomputed from the files, so it is brought forward."""
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.executescript(_SCHEMA)
    connection.execute("PRAGMA user_version = 1")
    connection.execute(
        "INSERT INTO projects (name, notes, created_at) VALUES (?, '', ?)",
        ("legacy", "2026-01-01T00:00:00+00:00"),
    )
    connection.commit()
    connection.close()

    store = ProjectStore(path)
    listed = store.list_projects()
    assert [entry.name for entry in listed.projects] == ["legacy"]

    # And the new table exists, so the migrated database is fully usable.
    saved = store.save_rig(listed.projects[0].id, CameraRig())
    assert saved.rig is not None


def test_a_database_from_a_future_build_is_still_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    connection = sqlite3.connect(path)
    connection.executescript(_SCHEMA)
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()

    with pytest.raises(ProjectError, match="schema version 99"):
        ProjectStore(path).list_projects()


# --- undistortion reaches the pipeline ------------------------------------


def _pose_sequence(x: float, y: float, *, width: int = 1920, height: int = 1080):
    """A one-frame sequence with every landmark at one normalised IMAGE position."""
    from datetime import UTC, datetime

    from analyzer.contracts.cache import ContentKey, HashAlgorithm
    from analyzer.contracts.pose import (
        FrameGeometry as Geometry,
    )
    from analyzer.contracts.pose import (
        Landmark,
        LandmarkPoint,
        PoseExtractionStats,
        PoseFrame,
        PoseModelInfo,
        PoseSequence,
    )

    points = [
        LandmarkPoint(x=x, y=y, z=0.0, visibility=1.0, presence=1.0) for _ in range(len(Landmark))
    ]
    return PoseSequence(
        video_path="/data/swing.mov",
        video_content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED, digest="c" * 64, size_bytes=1
        ),
        geometry=Geometry(width=width, height=height),
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
        extracted_at=datetime(2026, 9, 17, tzinfo=UTC),
        stats=PoseExtractionStats(
            frames_processed=1,
            frames_detected=1,
            detection_rate=1.0,
            elapsed_s=0.0,
            ms_per_frame=0.0,
        ),
        frames=[
            PoseFrame(frame_index=0, timestamp_s=0.0, detected=True, image=points, hip_local=points)
        ],
    )


def test_a_calibration_moves_the_landmarks_below_the_filter() -> None:
    """Undistortion enters at the series, which is below the filter for a reason.

    The filter is linear, so a correction applied to positions before fitting
    emerges correctly in the velocity and acceleration. Applied afterwards it
    would need a second correction kept in step by hand, which is the failure
    `coordinates.py` already avoids for the aspect ratio and the y flip.
    """
    from analyzer.contracts.pose import Landmark
    from analyzer.pose.series import landmark_series

    # Near the frame corner, where a lens does its work.
    sequence = _pose_sequence(0.94, 0.90)
    intrinsics = default_camera()

    plain = landmark_series(sequence, Landmark.NOSE)
    corrected = landmark_series(sequence, Landmark.NOSE, intrinsics=intrinsics)

    moved = float(np.hypot(corrected.x[0] - plain.x[0], corrected.y[0] - plain.y[0]))
    assert moved > 0.01

    # A landmark at the optical centre barely moves, because distortion is
    # radial and is nearly nothing there.
    centred = _pose_sequence(intrinsics.cx / 1920, intrinsics.cy / 1080)
    plain_centre = landmark_series(centred, Landmark.NOSE)
    fixed_centre = landmark_series(centred, Landmark.NOSE, intrinsics=intrinsics)
    assert abs(float(fixed_centre.x[0] - plain_centre.x[0])) < 1e-4


def test_undistortion_is_refused_on_a_frame_of_another_size() -> None:
    from analyzer.contracts.pose import Landmark
    from analyzer.pose.series import landmark_series

    sequence = _pose_sequence(0.5, 0.5, width=1280, height=720)
    with pytest.raises(CalibrationMismatchError):
        landmark_series(sequence, Landmark.NOSE, intrinsics=default_camera(1920, 1080))


def test_hip_local_coordinates_carry_no_lens() -> None:
    """MediaPipe's body-centred guess is not an image, so it has no distortion."""
    from analyzer.contracts.pose import Landmark, LandmarkSpace
    from analyzer.pose.series import landmark_series

    sequence = _pose_sequence(0.5, 0.5)
    with pytest.raises(ValueError, match="cannot be applied"):
        landmark_series(
            sequence, Landmark.NOSE, LandmarkSpace.HIP_LOCAL, intrinsics=default_camera()
        )


def test_frame_widths_conversion_happens_after_undistortion() -> None:
    """Order matters: distortion is defined in pixels, not in frame widths.

    Undistorting after the aspect correction would apply a radial model in a
    frame whose y axis has been scaled and flipped, which is a different and
    wrong correction that still produces plausible coordinates.
    """
    intrinsics = default_camera()
    geometry = FrameGeometry(width=1920, height=1080)

    corner = np.array([[0.95, 0.92, 0.0]])
    undistorted = undistort_normalized(corner, intrinsics, geometry)

    # In pixels the correction is radial about the principal point, so a corner
    # point moves outward along that radius.
    before = corner[0, :2] * np.array([1920, 1080])
    after = undistorted[0, :2] * np.array([1920, 1080])
    centre = np.array([intrinsics.cx, intrinsics.cy])
    assert np.linalg.norm(after - centre) > np.linalg.norm(before - centre)


# --- the calibration reaching a real measurement --------------------------


def test_metrics_report_the_calibration_state_they_were_measured_under() -> None:
    """A reader must never have to infer this from the absence of refusals."""
    from analyzer.contracts.metrics import MetricSet

    assert MetricSet.model_fields["calibration"].default is CalibrationStatus.NONE


def test_compute_metrics_without_a_project_is_uncalibrated() -> None:
    from analyzer.dispatch import _calibration_for

    intrinsics, status = _calibration_for(None, object())
    assert intrinsics is None
    assert status is CalibrationStatus.NONE


def test_a_clip_outside_the_project_is_refused_rather_than_measured(
    camera: SyntheticCamera, spread_views: list
) -> None:
    """Without a role there is no way to know which camera filmed it.

    Guessing would mean applying one camera's lens to another camera's footage,
    which is a wrong answer that looks exactly like a right one.
    """
    from types import SimpleNamespace

    from analyzer.contracts.cache import ContentKey, HashAlgorithm
    from analyzer.contracts.rpc import EngineError
    from analyzer.dispatch import _calibration_for
    from analyzer.projects.store import ProjectStore

    # The default path, which `conftest.isolated_data` redirects into tmp_path --
    # and which is what `_calibration_for` opens.
    store = ProjectStore()
    project = store.create_project("session")
    store.save_rig(
        project.id,
        CameraRig(
            cameras={
                CameraRole.FACE_ON: _usable_calibration(camera, spread_views, CameraRole.FACE_ON)
            }
        ),
    )
    store.close()

    sequence = SimpleNamespace(
        video_content_key=ContentKey(
            algorithm=HashAlgorithm.SHA256_SAMPLED, digest="f" * 64, size_bytes=1
        ),
        geometry=FrameGeometry(width=1920, height=1080),
    )

    with pytest.raises(EngineError, match="not a clip of project"):
        _calibration_for(project.id, sequence)
