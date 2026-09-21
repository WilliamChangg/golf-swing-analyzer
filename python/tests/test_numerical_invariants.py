"""Numerical invariants across clocks, image shapes, and coordinate scales."""

from __future__ import annotations

import numpy as np
import pytest

from analyzer.biomechanics.geometry import distance, interior_angle_deg, project_onto
from analyzer.contracts.pose import FrameGeometry
from analyzer.contracts.sync import AnchorSource, SyncAnchor, SyncConfig
from analyzer.coordinates import image_to_frame_widths
from analyzer.filtering.localpoly import local_polynomial_fit
from analyzer.reconstruction.triangulate import StereoGeometry, triangulate_linear
from analyzer.sync.timemap import fit_time_map
from tests.synthetic_board import default_camera
from tests.synthetic_signals import cubic_motion, irregular_clock

INTRINSICS = default_camera(distortion=(0, 0, 0, 0))


@pytest.mark.parametrize("angle", [-137, -45, 0, 73, 180])
@pytest.mark.parametrize("scale", [0.001, 1, 1000])
def test_geometry_survives_similarity_transforms(angle: float, scale: float) -> None:
    radians = np.radians(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    points = np.array([[0.0, 3.0], [0.0, 0.0], [4.0, 0.0]])
    a, vertex, b = scale * points @ rotation.T + [23.0, -71.0]
    assert interior_angle_deg(a, vertex, b) == pytest.approx(90, abs=1e-8)
    assert distance(a, b) == pytest.approx(5 * scale)
    assert project_onto(a - vertex, b - vertex) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize(
    "geometry", [FrameGeometry(width=1080, height=1920), FrameGeometry(width=1920, height=1080)]
)
def test_pixel_triangle_preserves_angles_through_aspect_correction(geometry: FrameGeometry) -> None:
    pixels = np.array([[300.0, 600.0], [300.0, 300.0], [700.0, 300.0]])
    corrected = image_to_frame_widths(pixels / [geometry.width, geometry.height], geometry)
    assert interior_angle_deg(*corrected) == pytest.approx(90)
    assert distance(corrected[0], corrected[2]) * geometry.width == pytest.approx(500)


@pytest.mark.parametrize("clock_scale,origin", [(1.0, 0.0), (0.25, 120.0), (4.0, 3600.0)])
def test_vfr_derivatives_obey_clock_units(clock_scale: float, origin: float) -> None:
    t = irregular_clock()
    truth = cubic_motion(t)
    fit = local_polynomial_fit(
        origin + clock_scale * t,
        truth[0],
        np.ones(t.size),
        window_s=0.5 * clock_scale,
        polyorder=3,
        min_observations=4,
    )
    # Require every sample, so an implementation returning only NaNs cannot pass.
    for order, actual in enumerate((fit.value, fit.velocity, fit.acceleration)):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, truth[order] / clock_scale**order, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("baseline", [0.1, 0.5, 2.0])
def test_rectified_stereo_recovers_depth_from_analytical_disparity(baseline: float) -> None:
    geometry = StereoGeometry(INTRINSICS, INTRINSICS, np.eye(3), np.array([-baseline, 0.0, 0.0]))
    truth = np.array([[0.2, -0.1, 2.0], [-0.4, 0.3, 4.0], [0.0, 0.0, 8.0]])
    # Construct pixels directly from the pinhole equations, independently of
    # StereoGeometry.project. The right camera sits baseline metres to the right.
    left = truth[:, :2] / truth[:, 2, None] * [INTRINSICS.fx, INTRINSICS.fy] + [
        INTRINSICS.cx,
        INTRINSICS.cy,
    ]
    right = left.copy()
    right[:, 0] -= INTRINSICS.fx * baseline / truth[:, 2]
    np.testing.assert_allclose(triangulate_linear(left, right, geometry), truth, atol=1e-10)


@pytest.mark.parametrize("origin", [0.0, 3600.0])
@pytest.mark.parametrize("rate", [0.97, 1.03])
def test_alignment_recovers_affine_clocks_and_reverses_direction(
    origin: float, rate: float
) -> None:
    stamps = origin + np.array([0, 2, 5, 9, 12], dtype=float)
    targets = rate * (stamps - origin) + origin + 0.4
    anchors = [
        SyncAnchor(
            label=str(i),
            source=AnchorSource.MANUAL,
            reference_frame=i,
            target_frame=i,
            reference_s=float(t),
            target_s=float(targets[i]),
            confidence=1.0,
        )
        for i, t in enumerate(stamps)
    ]
    mapping, residuals, _ = fit_time_map(anchors, floor_s=0.001, config=SyncConfig())
    assert mapping.rate_estimated
    assert mapping.rate == pytest.approx(rate)
    assert max(abs(r.residual_ms) for r in residuals) < 1e-8
    for reference, target in zip(stamps, targets, strict=True):
        assert mapping.to_target(float(reference)) == pytest.approx(target)
        assert mapping.inverse().to_target(float(target)) == pytest.approx(reference)
