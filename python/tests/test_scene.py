"""What gets handed to a viewport, and what it is not allowed to invent.

A scene is the same reconstruction Phase 9 already measured against a body that
is an input, so these tests are not about whether triangulation works. They are
about the three ways a drawing can quietly stop describing the measurement:

* by **projecting differently** from the engine, so the picture and the video
  frame it is checked against disagree with nobody able to say which is wrong;
* by **rounding an absence into a position**, so a hole in the skeleton is filled
  with a guess and a refusal reads as a measurement;
* by **flattering the viewpoint**, so a badly conditioned capture draws as
  confidently as a good one.

The third is the phase's own argument and it has its own section below.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.pose import POSE_CONNECTIONS, Landmark, LandmarkSpace
from analyzer.contracts.reconstruction import ReconstructionConfig, RefusalReason
from analyzer.contracts.scene import SceneCameraKind
from analyzer.contracts.sync import TimeMap
from analyzer.filtering.landmarks import filter_sequence
from analyzer.reconstruction import ReconstructedSequence, reconstruct_pair
from analyzer.reconstruction.triangulate import (
    positional_uncertainty,
    uncertainty_covariance,
    visible_uncertainty_fraction,
)
from analyzer.scene import MAX_SCENE_FRAMES, SceneError, build_scene
from tests.synthetic_body3d import (
    TOP_S,
    SwingRig,
    camera_rig,
    default_rig,
    pose_sequence_for,
)

# No file backs a synthetic reconstruction, and this key says so rather than
# looking like a digest of something.
SYNTHETIC_KEY = ContentKey(algorithm=HashAlgorithm.SHA256, digest="0" * 64, size_bytes=0)


def _filtered(rig: SwingRig, view_name: str, **kwargs: object):  # type: ignore[no-untyped-def]
    view = getattr(rig, view_name)
    contract = camera_rig(rig)
    camera = contract.cameras[view.role]
    sequence = pose_sequence_for(view, **kwargs)  # type: ignore[arg-type]
    return filter_sequence(sequence, FilterConfig(), intrinsics=camera.intrinsics)


def _time_map(rig: SwingRig) -> TimeMap:
    return TimeMap(
        offset_s=rig.time_offset_s(),
        rate=1.0,
        rate_estimated=False,
        pivot_s=TOP_S,
        offset_uncertainty_s=0.001,
        support_start_s=0.5,
        support_end_s=2.0,
    )


def _reconstruct(
    rig: SwingRig, *, config: ReconstructionConfig | None = None, **noise: object
) -> ReconstructedSequence:
    return reconstruct_pair(
        _filtered(rig, "reference", **noise),
        _filtered(rig, "target", seed=99, **noise),
        camera_rig(rig),
        _time_map(rig),
        reference_role=rig.reference.role,
        target_role=rig.target.role,
        config=config,
    )


@pytest.fixture(scope="module")
def rig() -> SwingRig:
    return default_rig()


@pytest.fixture(scope="module")
def reconstruction(rig: SwingRig) -> ReconstructedSequence:
    return _reconstruct(rig, noise_px=2.7)


@pytest.fixture(scope="module")
def scene(reconstruction: ReconstructedSequence):  # type: ignore[no-untyped-def]
    return build_scene(
        reconstruction, reference_content_key=SYNTHETIC_KEY, start_frame=0, end_frame=60
    )


# --- the shape of what is handed over --------------------------------------


def test_the_scene_is_metres_in_the_reference_camera_and_says_so(scene) -> None:  # type: ignore[no-untyped-def]
    assert scene.space is LandmarkSpace.CAMERA
    assert scene.reference_content_key == SYNTHETIC_KEY
    assert scene.start_frame == 0
    assert scene.end_frame == 60
    assert len(scene.frames) == 60


def test_frame_indices_are_the_reference_clip_s_own(scene) -> None:  # type: ignore[no-untyped-def]
    """A consumer holding a frame number from Phase 4 must be able to index in.

    Positional indexing would work for a scene that always started at zero and
    break silently for one that did not, which is exactly the window a viewport
    asks for when it pages.
    """
    assert [frame.frame_index for frame in scene.frames] == list(range(60))


def test_a_window_carries_its_own_indices(reconstruction) -> None:  # type: ignore[no-untyped-def]
    window = build_scene(
        reconstruction, reference_content_key=SYNTHETIC_KEY, start_frame=20, end_frame=30
    )
    assert [frame.frame_index for frame in window.frames] == list(range(20, 30))
    assert window.start_frame == 20


def test_a_range_past_the_limit_is_refused_by_name() -> None:
    """Refused rather than truncated, for `build_overlay`'s reason.

    A viewport handed the first six hundred frames of a longer scene would scrub
    off the end into nothing, with no indication that it had been given less than
    it asked for.

    The fixture is filmed at 240 fps so that it is genuinely longer than the
    bound -- the clamp runs first, so a request past the *end of a short clip* is
    a reasonable question that gets a short answer, and only a clip that really
    holds more frames than the limit can reach the refusal.
    """
    long_clip = _reconstruct(default_rig(reference_fps=240.0), noise_px=2.7)
    assert len(long_clip) > MAX_SCENE_FRAMES

    with pytest.raises(SceneError) as caught:
        build_scene(long_clip, reference_content_key=SYNTHETIC_KEY, start_frame=0, end_frame=None)
    assert str(MAX_SCENE_FRAMES) in str(caught.value)
    assert caught.value.remediation is not None

    # And the window that fits is produced without complaint.
    window = build_scene(long_clip, reference_content_key=SYNTHETIC_KEY, end_frame=MAX_SCENE_FRAMES)
    assert len(window.frames) == MAX_SCENE_FRAMES


def test_ends_are_clamped_rather_than_refused(reconstruction) -> None:  # type: ignore[no-untyped-def]
    total = len(reconstruction)
    window = build_scene(
        reconstruction,
        reference_content_key=SYNTHETIC_KEY,
        start_frame=total - 3,
        end_frame=total + 50,
    )
    assert window.end_frame == total
    assert len(window.frames) == 3


def test_connections_are_the_project_s_one_opinion_about_anatomy(scene) -> None:  # type: ignore[no-untyped-def]
    """The same tuple the overlay draws and the bone checks measure."""
    assert set(map(tuple, scene.connections)) == {tuple(pair) for pair in POSE_CONNECTIONS}


def test_trajectories_default_to_the_wrists_and_break_at_refusals(scene) -> None:  # type: ignore[no-untyped-def]
    assert [entry.landmark for entry in scene.trajectories] == [
        Landmark.LEFT_WRIST,
        Landmark.RIGHT_WRIST,
    ]
    for entry in scene.trajectories:
        assert len(entry.points) == len(scene.frames)
        assert entry.reconstructed == sum(1 for point in entry.points if point is not None)


def test_an_empty_trajectory_request_asks_for_none(reconstruction) -> None:  # type: ignore[no-untyped-def]
    empty = build_scene(
        reconstruction, reference_content_key=SYNTHETIC_KEY, end_frame=10, trajectories=()
    )
    assert empty.trajectories == []


def test_framing_is_measured_from_the_body(scene) -> None:  # type: ignore[no-untyped-def]
    """The viewport's orbit centre and distance are measurements, not constants.

    A subject filmed from four metres and one filmed from two must both open
    framed, which a hard-coded camera distance cannot do.
    """
    positions = np.array(
        [
            [point.position.x, point.position.y, point.position.z]
            for frame in scene.frames
            for point in frame.points
            if point.position is not None
        ]
    )
    centre = np.array([scene.centroid.x, scene.centroid.y, scene.centroid.z])
    assert np.allclose(centre, np.median(positions, axis=0))

    distances = np.linalg.norm(positions - centre, axis=1)
    assert scene.radius_m == pytest.approx(float(np.percentile(distances, 95.0)))
    # A 1.78 m body seen whole: not a speck, and not the whole room.
    assert 0.5 < scene.radius_m < 2.0


# --- the cameras, which are what makes the projection checkable -------------


def test_the_reference_camera_is_the_origin_looking_down_z(scene) -> None:  # type: ignore[no-untyped-def]
    reference = scene.cameras[0]
    assert reference.kind is SceneCameraKind.REFERENCE
    assert (reference.position.x, reference.position.y, reference.position.z) == (0.0, 0.0, 0.0)
    assert (reference.forward.x, reference.forward.y, reference.forward.z) == (0.0, 0.0, 1.0)
    # Image y increases downward, so the camera's own up is -y.
    assert (reference.up.x, reference.up.y, reference.up.z) == (0.0, -1.0, 0.0)


def test_the_target_camera_sits_where_the_rig_says_it_does(scene, reconstruction, rig) -> None:  # type: ignore[no-untyped-def]
    target = scene.cameras[1]
    assert target.kind is SceneCameraKind.TARGET
    placed = np.array([target.position.x, target.position.y, target.position.z])
    assert np.allclose(placed, reconstruction.geometry.target_centre)

    # The rig puts the two cameras 90 degrees apart, which is what the capture
    # protocol asks for and what the scene must show a reader.
    reference = scene.cameras[0]
    axes = np.array(
        [
            [reference.forward.x, reference.forward.y, reference.forward.z],
            [target.forward.x, target.forward.y, target.forward.z],
        ]
    )
    between = np.degrees(np.arccos(np.clip(float(axes[0] @ axes[1]), -1.0, 1.0)))
    assert between == pytest.approx(rig.convergence_deg, abs=1.0)


def test_the_camera_bases_are_orthonormal_and_right_handed(scene) -> None:  # type: ignore[no-untyped-def]
    """The check `look_at` failed silently in Phase 9, asserted directly.

    A basis built with the cross product the wrong way round produces a camera
    that is upside down and mirrored. Triangulation cannot see it -- the
    reference frame is merely rotated -- and a picture drawn from it is a
    convincing image of a body that never existed.
    """
    for camera in scene.cameras:
        right = np.array([camera.right.x, camera.right.y, camera.right.z])
        up = np.array([camera.up.x, camera.up.y, camera.up.z])
        forward = np.array([camera.forward.x, camera.forward.y, camera.forward.z])
        for axis in (right, up, forward):
            assert np.linalg.norm(axis) == pytest.approx(1.0)
        assert right @ up == pytest.approx(0.0, abs=1e-9)
        assert right @ forward == pytest.approx(0.0, abs=1e-9)
        assert up @ forward == pytest.approx(0.0, abs=1e-9)
        # Screen right, screen up, and the way the camera looks: a left-handed
        # triple, because the image's y runs downward.
        assert np.cross(right, up) @ forward == pytest.approx(-1.0)


def test_the_reference_view_reproduces_the_pixels_the_camera_saw(scene, reconstruction) -> None:  # type: ignore[no-untyped-def]
    """**The claim that makes a hand-written viewport projection safe to ship.**

    Place a camera at the scene's reference camera, project with the intrinsics
    the scene carries, and every point must land where `StereoGeometry` puts it --
    which is where that camera saw the landmark, and where `PoseOverlay` draws it.
    A viewport and a video frame are then two renderings of one measurement, and
    a disagreement between them is a real disagreement rather than a rendering
    artefact.

    The same assertion is made against the same numbers in TypeScript, through
    `projection-truth.json`, so both implementations are pinned to this.
    """
    camera = scene.cameras[0]
    points = np.array(
        [
            [point.position.x, point.position.y, point.position.z]
            for frame in scene.frames
            for point in frame.points
            if point.position is not None
        ]
    )
    expected, _ = reconstruction.geometry.project(points)

    # Written out rather than calling `project`, so the two are not one mistake.
    basis = np.array(
        [
            [camera.right.x, camera.right.y, camera.right.z],
            [-camera.up.x, -camera.up.y, -camera.up.z],
            [camera.forward.x, camera.forward.y, camera.forward.z],
        ]
    )
    local = points @ basis.T - basis @ np.array(
        [camera.position.x, camera.position.y, camera.position.z]
    )
    drawn = np.stack(
        (
            camera.fx * local[:, 0] / local[:, 2] + camera.cx,
            camera.fy * local[:, 1] / local[:, 2] + camera.cy,
        ),
        axis=-1,
    )
    assert np.allclose(drawn, expected, atol=1e-9)


def test_the_field_of_view_matches_the_focal_length(scene) -> None:  # type: ignore[no-untyped-def]
    for camera in scene.cameras:
        expected = 2.0 * np.degrees(np.arctan2(camera.image_width / 2.0, camera.fx))
        assert camera.horizontal_fov_deg == pytest.approx(expected)


# --- absence stays absent ---------------------------------------------------


@pytest.fixture(scope="module")
def refusing_scene():  # type: ignore[no-untyped-def]
    """A capture so shallow that the convergence gate rejects most of it.

    Six degrees between the cameras: the fix is to move one of them, and the
    point of this fixture is that the scene has to say so rather than drawing a
    body with holes in it for no stated reason.
    """
    shallow = _reconstruct(default_rig(convergence_deg=6.0), noise_px=2.7)
    return build_scene(shallow, reference_content_key=SYNTHETIC_KEY, end_frame=40)


def test_a_refused_point_carries_no_position_and_a_reason(scene, refusing_scene) -> None:  # type: ignore[no-untyped-def]
    """The invariant the type exists to hold, checked on every point of both scenes."""
    seen_refusal = False
    for subject in (scene, refusing_scene):
        for frame in subject.frames:
            for point in frame.points:
                if point.refused is None:
                    assert point.position is not None
                    assert point.uncertainty is not None
                else:
                    seen_refusal = True
                    assert point.position is None
                    assert point.uncertainty is None
                    assert point.convergence_deg is None
                    assert point.reprojection_px is None
    assert seen_refusal, "the shallow fixture is expected to refuse something"


def test_the_refusal_reason_is_the_one_the_gate_recorded(refusing_scene) -> None:  # type: ignore[no-untyped-def]
    """Not a boolean, and not the last gate that happened to run.

    Phase 9 counts five reasons because their fixes are unrelated -- move a
    camera, re-film the occlusion, re-synchronise, accept the distance. A
    viewport drawing a hole has to be able to say which, and the answer for this
    capture is the camera placement rather than anything about the footage.
    """
    reasons = {
        point.refused
        for frame in refusing_scene.frames
        for point in frame.points
        if point.refused is not None
    }
    assert RefusalReason.ILL_CONDITIONED in reasons


def test_a_frame_with_nothing_in_it_is_a_frame_with_nothing_in_it(reconstruction) -> None:  # type: ignore[no-untyped-def]
    """`reconstructed` counts this frame, never the last good one.

    A viewport that held the previous pose across an empty frame would put a body
    on screen in a position it was never measured in -- the three-dimensional
    version of an overlay interpolating across a blocked gap, which Phase 4 found
    covering 1.92 s of a real clip exactly where the wrists moved fastest.
    """
    scene = build_scene(reconstruction, reference_content_key=SYNTHETIC_KEY, end_frame=60)
    for frame in scene.frames:
        produced = sum(1 for point in frame.points if point.position is not None)
        assert frame.reconstructed == produced


def test_nothing_reconstructed_still_produces_a_scene(rig) -> None:  # type: ignore[no-untyped-def]
    """An impossible bound refuses every point, and the scene says so honestly."""
    nothing = _reconstruct(rig, config=ReconstructionConfig(max_uncertainty_m=1e-9), noise_px=2.7)
    scene = build_scene(nothing, reference_content_key=SYNTHETIC_KEY, end_frame=20)
    assert all(frame.reconstructed == 0 for frame in scene.frames)
    assert all(point.position is None for frame in scene.frames for point in frame.points)
    # Still somewhere to stand, and a radius that cannot be zero.
    assert scene.radius_m > 0.0
    assert scene.report.quality is None or scene.report.quality.points_reconstructed == 0


# --- the uncertainty, which is what the phase is about ----------------------


def test_the_ellipsoid_and_the_reported_sigma_are_the_same_quantity(scene, reconstruction) -> None:  # type: ignore[no-untyped-def]
    """`sigma_m` must be the square root of the covariance's largest eigenvalue.

    They come from two functions -- `positional_uncertainty` gated the point and
    `uncertainty_covariance` shaped it -- and "the same by construction" across
    two functions is a claim, not a guarantee. This is the guarantee.
    """
    checked = 0
    for frame in scene.frames[:10]:
        for point in frame.points:
            if point.uncertainty is None:
                continue
            matrix = np.array(
                [
                    [point.uncertainty.xx, point.uncertainty.xy, point.uncertainty.xz],
                    [point.uncertainty.xy, point.uncertainty.yy, point.uncertainty.yz],
                    [point.uncertainty.xz, point.uncertainty.yz, point.uncertainty.zz],
                ]
            )
            largest = float(np.sqrt(np.linalg.eigvalsh(matrix)[-1]))
            assert point.uncertainty.sigma_m == pytest.approx(largest, rel=1e-9)
            checked += 1
    assert checked > 100


def test_the_covariance_is_a_covariance(scene) -> None:  # type: ignore[no-untyped-def]
    """Symmetric by shape, and positive semi-definite because it is a variance."""
    for frame in scene.frames[:5]:
        for point in frame.points:
            if point.uncertainty is None:
                continue
            matrix = np.array(
                [
                    [point.uncertainty.xx, point.uncertainty.xy, point.uncertainty.xz],
                    [point.uncertainty.xy, point.uncertainty.yy, point.uncertainty.yz],
                    [point.uncertainty.xz, point.uncertainty.yz, point.uncertainty.zz],
                ]
            )
            assert np.all(np.linalg.eigvalsh(matrix) >= -1e-18)


def test_the_covariance_agrees_with_the_scalar_phase_9_already_reported(reconstruction) -> None:  # type: ignore[no-untyped-def]
    """The anti-drift pin between the cheap path and the full one.

    `positional_uncertainty` computes only the smallest eigenvalue of `J'J` and
    is what every gate and report in Phase 9 uses. `uncertainty_covariance` builds
    the whole inverse. They must not disagree.
    """
    points = reconstruction.points.reshape(-1, 3)
    valid = reconstruction.valid.reshape(-1)
    sigma_px = reconstruction.report.quality.pixel_sigma_px
    kept = points[valid][:500]

    scalar = positional_uncertainty(kept, reconstruction.geometry, sigma_px)
    matrices = uncertainty_covariance(kept, reconstruction.geometry, sigma_px)
    largest = np.sqrt(np.linalg.eigvalsh(matrices)[:, -1])
    assert np.allclose(scalar, largest, rtol=1e-9)


def test_a_viewpoint_along_the_worst_axis_sees_the_least(reconstruction) -> None:  # type: ignore[no-untyped-def]
    """The fraction is a property of the view, and it is bounded by construction.

    Looking straight down a point's worst-determined axis must show less than
    looking across it, and no viewpoint can show more than all of it.
    """
    points = reconstruction.points.reshape(-1, 3)
    valid = reconstruction.valid.reshape(-1)
    sigma_px = reconstruction.report.quality.pixel_sigma_px
    kept = points[valid][:200]
    matrices = uncertainty_covariance(kept, reconstruction.geometry, sigma_px)

    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    worst_axis = eigenvectors[:, :, -1]
    best_axis = eigenvectors[:, :, 0]

    along_worst = visible_uncertainty_fraction(matrices, worst_axis)
    along_best = visible_uncertainty_fraction(matrices, best_axis)

    assert np.all(along_worst <= along_best + 1e-9)
    assert np.all((along_worst >= 0.0) & (along_worst <= 1.0))
    # Looking across the worst axis shows all of it, by definition.
    assert np.allclose(along_best, 1.0, atol=1e-9)
    # And a point whose uncertainty is genuinely anisotropic hides most of it.
    anisotropy = eigenvalues[:, -1] / np.maximum(eigenvalues[:, 0], 1e-30)
    assert np.all(along_worst[anisotropy > 4.0] < 0.55)


def test_the_default_viewpoint_flatters_a_shallow_capture(rig) -> None:  # type: ignore[no-untyped-def]
    """**Phase 15's finding, as an assertion.**

    Bring the two cameras together and the reconstruction gets several times
    worse. Seen from the reference camera -- where a viewport opens unless
    somebody moves it -- it does not look worse at all, because the direction
    that grew is the direction that camera is looking along.

    `scripts/benchmark_viewport.py --sweep convergence` measures the whole curve;
    this pins its two ends so the claim cannot rot.
    """

    def seen_from_the_reference(separation: float) -> tuple[float, float]:
        result = _reconstruct(default_rig(convergence_deg=separation), noise_px=2.7)
        sigma_px = result.report.quality.pixel_sigma_px
        points = result.points.reshape(-1, 3)
        kept = points[result.valid.reshape(-1)]
        matrices = uncertainty_covariance(kept, result.geometry, sigma_px)
        worst = np.sqrt(np.linalg.eigvalsh(matrices)[:, -1])
        # The reference camera is the origin, so the direction to each point is
        # the point.
        fraction = visible_uncertainty_fraction(matrices, kept)
        finite = np.isfinite(worst) & np.isfinite(fraction)
        return float(np.median(worst[finite])), float(np.median(worst[finite] * fraction[finite]))

    wide_true, wide_seen = seen_from_the_reference(90.0)
    narrow_true, narrow_seen = seen_from_the_reference(15.0)

    # The reconstruction really is several times worse.
    assert narrow_true > 3.0 * wide_true
    # And the default viewpoint shows essentially the same smear either way.
    assert narrow_seen == pytest.approx(wide_seen, rel=0.25)
