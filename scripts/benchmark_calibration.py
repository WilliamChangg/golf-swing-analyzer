#!/usr/bin/env python3
"""Measure what a calibration is worth, against a camera whose parameters are known.

**This is ground truth in a stronger sense than Phase 7's benchmark, and a weaker
one than an optical bench.** The camera is an input: every intrinsic the fit is
asked to recover is a number put into the renderer, so the error is a subtraction
rather than a judgement. What the renderer does not contain is everything that
makes real calibration footage hard -- motion blur, rolling shutter, defocus,
JPEG ringing on the marker borders, and a printed sheet that is not quite flat.
The detections here are therefore cleaner than any real capture's, and the errors
below are a **floor** rather than an estimate of what a phone will achieve.

The question the sweeps exist to answer is not "how accurate is calibration". It
is *whether the numbers a calibration reports about itself can tell you when it
has gone wrong* -- because the one that everybody quotes cannot:

    tilt        does the residual notice a capture that determines nothing?
    area        what does never leaving the middle of the frame cost?
    model       is OpenCV's fifth distortion coefficient worth fitting?
    views       how many board views are actually needed?
    stereo      what does pairing two unsynchronised cameras cost, in pixels?

Usage:
    uv run --project python python scripts/benchmark_calibration.py
    uv run --project python python scripts/benchmark_calibration.py --repeats 3
    uv run --project python python scripts/benchmark_calibration.py --sweep tilt
    uv run --project python python scripts/benchmark_calibration.py --real <dir-or-video>
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.calibration.apply import distortion_displacement  # noqa: E402
from analyzer.calibration.detect import (  # noqa: E402
    BoardView,
    detect_in_directory,
    detect_in_image,
    detect_in_video,
)
from analyzer.calibration.intrinsics import calibrate_intrinsics  # noqa: E402
from analyzer.calibration.stereo import (  # noqa: E402
    calibrate_stereo,
    pair_by_time_map,
)
from analyzer.contracts.calibration import (  # noqa: E402
    CalibrationConfig,
    CameraCalibration,
    CameraIntrinsics,
    DistortionModel,
)
from analyzer.contracts.camera import CameraRole  # noqa: E402
from analyzer.contracts.sync import TimeMap  # noqa: E402
from tests.synthetic_board import (  # noqa: E402
    DEFAULT_BOARD,
    SPREAD_TARGETS,
    BoardPose,
    SyntheticCamera,
    default_camera,
    poses_at,
    stereo_rig,
    well_spread_poses,
)


@dataclass(frozen=True)
class Outcome:
    """One calibration, scored against the camera that produced its views."""

    calibration: CameraCalibration | None
    fx_error_pct: float
    k1_error: float
    principal_error_px: float
    edge_error_px: float
    """How far the fitted model misplaces a point at the frame edge.

    The quantity that actually matters downstream, and the one no calibration
    tool reports: a landmark out where the lens bends is displaced by the
    *difference* between the true distortion and the fitted one, and that is
    what every angle and distance above inherits.
    """

    @property
    def rms(self) -> float:
        return self.calibration.quality.rms_reprojection_px if self.calibration else float("nan")

    @property
    def sigma_fx_pct(self) -> float:
        if self.calibration is None:
            return float("nan")
        found = self.calibration.intrinsics
        if found.fx_uncertainty is None:
            return float("nan")
        return 100.0 * found.fx_uncertainty / found.fx


def render_views(
    camera: SyntheticCamera, poses: list[BoardPose], *, quiet: bool = True
) -> list[BoardView]:
    """Photograph the board at each pose and detect it."""
    views: list[BoardView] = []
    for index, pose in enumerate(poses):
        view = detect_in_image(camera.render(pose), camera.spec, frame=index)
        if view is not None:
            views.append(view)
    if not quiet:
        print(f"    {len(views)}/{len(poses)} views detected")
    return views


def edge_displacement_error(fitted: CameraIntrinsics, truth: CameraIntrinsics) -> float:
    """Worst disagreement between two camera models, in pixels, over the frame.

    Both models are used to undistort the same grid of pixels; the error is how
    far apart the two answers land. This is the honest summary of "how wrong is
    this calibration", because it combines the focal length, the principal point
    and every distortion coefficient into the one quantity that reaches a
    measurement -- and it weights them the way the image does, which is heavily
    towards the edge.
    """
    from analyzer.calibration.apply import undistort_pixels

    width, height = truth.image_width, truth.image_height
    xs = np.linspace(0, width - 1, 20)
    ys = np.linspace(0, height - 1, 20)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return float(np.max(np.linalg.norm(undistort_pixels(grid, fitted) - undistort_pixels(grid, truth), axis=1)))


def score(
    views: list[BoardView],
    camera: SyntheticCamera,
    config: CalibrationConfig | None = None,
) -> Outcome:
    """Fit a calibration from these views and compare it with the truth."""
    truth = camera.intrinsics
    try:
        calibration = calibrate_intrinsics(
            views,
            camera.spec,
            role=CameraRole.FACE_ON,
            source="synthetic",
            config=config,
        )
    except ValueError:
        return Outcome(None, float("nan"), float("nan"), float("nan"), float("nan"))

    found = calibration.intrinsics
    fitted_k1 = found.distortion[0] if found.distortion else 0.0
    return Outcome(
        calibration=calibration,
        fx_error_pct=100.0 * (found.fx - truth.fx) / truth.fx,
        k1_error=fitted_k1 - truth.distortion[0],
        principal_error_px=float(np.hypot(found.cx - truth.cx, found.cy - truth.cy)),
        edge_error_px=edge_displacement_error(found, truth),
    )


def _row(label: str, outcomes: list[Outcome], extra: str = "") -> None:
    """One line of a sweep table: medians over the seeds."""
    usable = [entry for entry in outcomes if entry.calibration is not None]
    if not usable:
        print(f"  {label:<18} {'refused':>9}")
        return

    def median(values: list[float]) -> float:
        finite = [value for value in values if np.isfinite(value)]
        return float(np.median(finite)) if finite else float("nan")

    accepted = sum(1 for entry in usable if entry.calibration and entry.calibration.usable)
    print(
        f"  {label:<18} "
        f"{median([e.rms for e in usable]):>7.3f} "
        f"{median([e.sigma_fx_pct for e in usable]):>9.3f} "
        f"{abs(median([e.fx_error_pct for e in usable])):>9.2f} "
        f"{abs(median([e.k1_error for e in usable])):>8.3f} "
        f"{median([e.edge_error_px for e in usable]):>9.1f} "
        f"{accepted}/{len(usable):<5} {extra}"
    )


_HEADER = (
    f"  {'':<18} {'RMS px':>7} {'sigma fx%':>9} {'fx err%':>9} {'k1 err':>8} "
    f"{'edge px':>9} {'usable':<6}"
)


def sweep_tilt(repeats: int) -> None:
    """The headline: what a board held square to the camera does.

    Focal length and distance are very nearly interchangeable when every view is
    parallel to the sensor -- a longer lens further away makes the same picture
    -- so a capture with no tilt cannot determine either. The reprojection error
    does not notice, because the model still fits those views.
    """
    print("\nTilt spread. Board held at N degrees of tilt, same positions otherwise.")
    print(_HEADER)

    camera = SyntheticCamera()
    for tilt in (0.0, 2.0, 5.0, 10.0, 20.0, 35.0):
        outcomes = []
        for seed in range(repeats):
            poses = poses_at(
                list(SPREAD_TARGETS),
                camera.intrinsics,
                tilt_deg=tilt,
                distances_m=(0.45, 0.95),
                seed=seed + 1,
            )
            outcomes.append(score(render_views(camera, poses), camera))
        _row(f"tilt +-{tilt:.0f} deg", outcomes)

    print(
        "\n  The residual is flat and the focal length is not. That is the whole\n"
        "  argument for CoverageReport: the number every tool prints cannot see\n"
        "  the failure, and on these runs the reported parameter uncertainty is\n"
        "  *smallest* exactly where the answer is worst."
    )


def sweep_area(repeats: int) -> None:
    """What never leaving the middle of the frame costs.

    Distortion is a function of radius and is almost nothing at the centre, so a
    centred capture leaves the coefficients fitted to noise -- and then applies
    them out at the edge, where they do all their work and where a swing is.
    """
    print("\nFrame coverage. Board confined to the central fraction of the frame.")
    print(_HEADER)

    camera = SyntheticCamera()
    for reach in (0.1, 0.25, 0.5, 0.8):
        outcomes = []
        for seed in range(repeats):
            targets = [(x * reach, y * reach) for x, y in SPREAD_TARGETS]
            poses = poses_at(
                targets, camera.intrinsics, tilt_deg=35.0, distances_m=(0.45, 0.95), seed=seed + 1
            )
            outcomes.append(score(render_views(camera, poses), camera))
        _row(f"reach {reach:.0%}", outcomes)


def sweep_model(repeats: int) -> None:
    """Whether OpenCV's fifth distortion coefficient earns its place.

    The board is rendered through a lens with four coefficients, so fitting five
    is fitting one term to noise. The question is what that costs, and where.
    """
    print("\nDistortion model, on a four-coefficient lens, with a good capture.")
    print(_HEADER)

    camera = SyntheticCamera()
    for model in DistortionModel:
        outcomes = []
        for seed in range(repeats):
            poses = well_spread_poses(14, seed=seed + 1)
            outcomes.append(
                score(
                    render_views(camera, poses),
                    camera,
                    CalibrationConfig(distortion_model=model),
                )
            )
        _row(model.value, outcomes)

    print("\n  And the same models on a capture that never reaches the frame edge:")
    print(_HEADER)
    for model in DistortionModel:
        outcomes = []
        for seed in range(repeats):
            targets = [(x * 0.3, y * 0.3) for x, y in SPREAD_TARGETS]
            poses = poses_at(
                targets, camera.intrinsics, tilt_deg=35.0, distances_m=(0.45, 0.95), seed=seed + 1
            )
            outcomes.append(
                score(
                    render_views(camera, poses),
                    camera,
                    CalibrationConfig(distortion_model=model),
                )
            )
        _row(model.value, outcomes)


def sweep_views(repeats: int) -> None:
    """How many board views are actually needed, given they are well spread."""
    print("\nView count, all well spread.")
    print(_HEADER)

    camera = SyntheticCamera()
    for count in (4, 6, 8, 10, 14, 20):
        outcomes = []
        for seed in range(repeats):
            poses = well_spread_poses(count, seed=seed + 1)
            outcomes.append(score(render_views(camera, poses), camera))
        _row(f"{count} views", outcomes)


def sweep_undistortion() -> None:
    """How far the lens moves a landmark, for a few plausible cameras.

    The number that says whether Phase 8 is worth anything to a single-camera
    user. Undistortion is a correction applied to every landmark before any
    measurement, and its size is a property of the lens rather than of this
    software.
    """
    print("\nWhat the lens does to a landmark, before any calibration removes it.")
    print(f"  {'lens':<26} {'edge px':>9} {'worst px':>9} {'h-fov':>7}")

    lenses = [
        ("phone main (k1 -0.28)", default_camera(fx=1400.0, distortion=(-0.28, 0.12, 0.0, 0.0))),
        ("phone wide (k1 -0.42)", default_camera(fx=900.0, distortion=(-0.42, 0.22, 0.0, 0.0))),
        ("mild (k1 -0.10)", default_camera(fx=1800.0, distortion=(-0.10, 0.03, 0.0, 0.0))),
        ("long lens (k1 -0.02)", default_camera(fx=3000.0, distortion=(-0.02, 0.0, 0.0, 0.0))),
    ]
    for label, lens in lenses:
        edge, worst = distortion_displacement(lens)
        print(f"  {label:<26} {edge:>9.1f} {worst:>9.1f} {lens.horizontal_fov_deg:>6.1f}d")

    print(
        "\n  On a 1920x1080 frame. A landmark that far out of place is inherited by\n"
        "  every angle, distance and speed measured above it, and nothing in those\n"
        "  numbers shows that it happened."
    )


def _station_poses(seed: int, stations: int = 8) -> list[BoardPose]:
    """Where the board is held, one pose per station."""
    rng = np.random.default_rng(seed)
    return [
        BoardPose(
            distance_m=float(rng.uniform(1.0, 1.6)),
            rotation_deg=(
                float(rng.uniform(-30, 30)),
                float(rng.uniform(-30, 30)),
                float(rng.uniform(-20, 20)),
            ),
            offset_m=(float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.08, 0.08))),
        )
        for _ in range(stations)
    ]


def _lerp(a: BoardPose, b: BoardPose, t: float) -> BoardPose:
    """A pose partway between two stations, for the frames spent in transit."""
    return BoardPose(
        distance_m=a.distance_m + t * (b.distance_m - a.distance_m),
        rotation_deg=tuple(  # type: ignore[arg-type]
            first + t * (second - first)
            for first, second in zip(a.rotation_deg, b.rotation_deg, strict=True)
        ),
        offset_m=tuple(  # type: ignore[arg-type]
            first + t * (second - first)
            for first, second in zip(a.offset_m, b.offset_m, strict=True)
        ),
    )


def _board_sequence(
    stations: list[BoardPose], dwell_frames: int, transit_frames: int
) -> list[BoardPose]:
    """The board's pose in every recorded frame, dwelling and then moving.

    This is what makes the sweep mean anything. A benchmark that renders one
    frame per station cannot distinguish a board held patiently from one waved
    about, because both produce exactly the same set of images -- the difference
    lives entirely in the frames *between* them, which is where the board's
    image speed is measured.
    """
    frames: list[BoardPose] = []
    for index, station in enumerate(stations):
        frames.extend([station] * dwell_frames)
        if index + 1 < len(stations):
            nxt = stations[index + 1]
            frames.extend(
                _lerp(station, nxt, (step + 1) / (transit_frames + 1))
                for step in range(transit_frames)
            )
    return frames


def sweep_stereo(repeats: int) -> None:
    """What pairing two unsynchronised cameras costs, as a function of board speed.

    Both cameras record continuously at 30 fps while the board is carried between
    eight stations. `dwell` is how many frames it rests at each one: rest long
    enough and its image speed at the paired frames is nearly zero, so an
    imperfect pairing costs nothing, however badly the two clocks are known.

    This is the evidence for the capture instruction, which is simply *hold it
    still* -- and for why that instruction is worth giving, since a waved board
    produces images that look exactly as good.
    """
    print("\nStereo extrinsics. Both cameras at 30 fps, clocks related to 12 ms.")
    print(
        f"  {'board':<18} {'speed px/s':>10} {'pair err px':>11} {'pairs':>6} "
        f"{'RMS px':>7} {'baseline err%':>13} {'rot err deg':>11}"
    )

    rig = stereo_rig()
    truth_baseline = float(np.linalg.norm(rig.translation_m))
    truth_rotation = rig.rotation
    config = CalibrationConfig()
    interval_s = 1.0 / 30.0
    offset_s = 0.427

    # Swept finely around the interesting region, because the useful answer is
    # not "hold it still" but *how long* -- and it turns out to be far shorter
    # than the instruction anyone would write by hand.
    for label, dwell in (
        ("held 1 s", 30),
        ("held 0.2 s", 6),
        ("held 3 frames", 3),
        ("held 2 frames", 2),
        ("never stops", 1),
    ):
        baselines: list[float] = []
        rotations: list[float] = []
        speeds: list[float] = []
        pair_errors: list[float] = []
        pair_counts: list[int] = []
        rms_values: list[float] = []

        for seed in range(repeats):
            sequence = _board_sequence(_station_poses(seed + 1), dwell, transit_frames=4)
            reference_views: list[BoardView] = []
            target_views: list[BoardView] = []

            # A static board in front of a static camera produces the same
            # frame every time, so a dwell is rendered once and reused. That is
            # not an approximation -- it is what "held still" means -- and it is
            # what keeps a 30-frame dwell from costing thirty renders.
            rendered: dict[BoardPose, tuple[Any, Any]] = {}
            for index, pose in enumerate(sequence):
                if pose not in rendered:
                    rendered[pose] = rig.render_pair(pose)
                left, right = rendered[pose]
                clock = index * interval_s
                reference = detect_in_image(
                    left, rig.reference.spec, frame=index, timestamp_s=clock
                )
                target = detect_in_image(
                    right, rig.target.spec, frame=index, timestamp_s=clock + offset_s
                )
                if reference is not None:
                    reference_views.append(reference)
                if target is not None:
                    target_views.append(target)

            if len(reference_views) < 3 or len(target_views) < 3:
                continue

            span = len(sequence) * interval_s
            time_map = TimeMap(
                offset_s=offset_s,
                rate=1.0,
                rate_estimated=False,
                pivot_s=span / 2,
                offset_uncertainty_s=0.012,
                support_start_s=0.0,
                support_end_s=span,
            )
            pairs, dropped = pair_by_time_map(reference_views, target_views, time_map, config)
            pair_counts.append(len(pairs))
            speeds.extend(pair.board_speed_px_s for pair in pairs)
            pair_errors.extend(pair.pairing_error_px for pair in pairs)

            if len(pairs) < config.min_stereo_pairs:
                continue

            found = calibrate_stereo(
                pairs,
                rig.reference.spec,
                rig.reference.intrinsics,
                rig.target.intrinsics,
                reference_role=CameraRole.FACE_ON,
                target_role=CameraRole.DOWN_THE_LINE,
                dropped=dropped,
                candidates=len(reference_views),
                config=config,
            )
            baselines.append(100.0 * (found.baseline_m - truth_baseline) / truth_baseline)
            rotations.append(
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                (np.trace(found.rotation_matrix() @ truth_rotation.T) - 1) / 2,
                                -1,
                                1,
                            )
                        )
                    )
                )
            )
            rms_values.append(found.quality.rms_reprojection_px)

        def show(values: list[float], width: int, places: int) -> str:
            if not values:
                return f"{'-':>{width}}"
            return f"{float(np.median(values)):>{width}.{places}f}"

        print(
            f"  {label:<18} {show(speeds, 10, 1)} {show(pair_errors, 11, 2)} "
            f"{show([float(count) for count in pair_counts], 6, 0)} {show(rms_values, 7, 3)} "
            f"{show([abs(value) for value in baselines], 13, 3)} {show(rotations, 11, 3)}"
        )

    print(
        "\n  A still board makes an unsynchronised pair as good as a genlocked one:\n"
        "  the clock error is multiplied by a speed of nearly zero. A waved one is\n"
        "  refused rather than fitted badly, which is why the pair count collapses\n"
        "  before any error does. The images are equally sharp in every row."
    )


def measure_cost() -> None:
    """What detection and fitting cost, on this machine."""
    print("\nCost.")
    camera = SyntheticCamera()
    poses = well_spread_poses(14)

    images = [camera.render(pose) for pose in poses]

    started = time.perf_counter()
    views = [detect_in_image(image, camera.spec, frame=index) for index, image in enumerate(images)]
    detect_s = time.perf_counter() - started
    found = [view for view in views if view is not None]

    started = time.perf_counter()
    calibrate_intrinsics(found, camera.spec, role=CameraRole.FACE_ON, source="cost")
    fit_s = time.perf_counter() - started

    print(f"  detect board, 1920x1080      {1000 * detect_s / len(images):8.1f} ms/frame")
    print(f"  fit intrinsics, {len(found):2d} views     {1000 * fit_s:8.1f} ms")


def real_capture(path: Path, board_squares: tuple[int, int], square_mm: float) -> int:
    """Run the real pipeline over real calibration footage, and report what it says.

    No ground truth is available for a real camera here, so nothing is scored.
    What this establishes is whether the detector finds the board at all, what
    coverage the capture achieved, and whether the result passes its own bounds
    -- which is the question a person actually has after filming a board.
    """
    spec = DEFAULT_BOARD.model_copy(
        update={
            "squares_x": board_squares[0],
            "squares_y": board_squares[1],
            "square_length_m": square_mm / 1000.0,
            "marker_length_m": 0.75 * square_mm / 1000.0,
        }
    )
    print(f"\nReal capture: {path}")
    print(f"  board {spec.squares_x}x{spec.squares_y}, {square_mm:.0f} mm squares")

    if path.is_dir():
        views, report = detect_in_directory(path, spec)
    else:
        views, report = detect_in_video(path, spec, stride=5)

    print(f"  scanned {report.frames_scanned}, board found in {report.frames_with_board}, "
          f"{report.views_used} distinct views kept")
    for warning in report.warnings:
        print(f"  ! {warning}")

    if not views:
        print("  No board detected. Check the square count and the dictionary.")
        return 1

    try:
        calibration = calibrate_intrinsics(
            views, spec, role=CameraRole.OTHER, source=str(path), detection=report
        )
    except ValueError as exc:
        print(f"  Refused: {exc}")
        return 1

    found = calibration.intrinsics
    quality = calibration.quality
    print(f"  fx {found.fx:.1f}  fy {found.fy:.1f}  cx {found.cx:.1f}  cy {found.cy:.1f}")
    print(f"  horizontal field of view {found.horizontal_fov_deg:.1f} deg  "
          f"(check this against the lens)")
    print(f"  distortion {['%+.4f' % value for value in found.distortion]}")
    print(f"  RMS {quality.rms_reprojection_px:.3f} px   max {quality.max_reprojection_px:.3f} px")
    if found.fx_uncertainty is not None:
        print(f"  focal uncertainty {found.fx_uncertainty:.2f} px "
              f"({100 * found.fx_uncertainty / found.fx:.2f}%)")
    print(f"  coverage: area {quality.coverage.image_fraction:.0%}  "
          f"edge {quality.coverage.edge_fraction:.0%}  "
          f"tilt {quality.coverage.tilt_range_deg:.0f} deg  "
          f"scale {quality.coverage.scale_range:.2f}x")
    edge, worst = distortion_displacement(found)
    print(f"  this lens moves an edge pixel by {edge:.1f} px (worst {worst:.1f} px)")
    print(f"  usable: {calibration.usable}")
    if calibration.refusal:
        print(f"  {calibration.refusal}")
    for warning in calibration.warnings:
        print(f"  ! {warning}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Seeds per measurement.")
    parser.add_argument(
        "--sweep",
        choices=("tilt", "area", "model", "views", "undistortion", "stereo", "cost", "all"),
        default="all",
    )
    parser.add_argument("--real", type=Path, default=None, help="A directory of stills, or a video.")
    parser.add_argument("--squares", default="7x5", help="Board squares, as WxH, for --real.")
    parser.add_argument("--square-mm", type=float, default=35.0, help="Square size for --real.")
    arguments = parser.parse_args()

    if arguments.real is not None:
        width, _, height = arguments.squares.partition("x")
        return real_capture(arguments.real, (int(width), int(height)), arguments.square_mm)

    truth = default_camera()
    print(f"Synthetic camera: fx={truth.fx:.0f} fy={truth.fy:.0f} "
          f"cx={truth.cx:.1f} cy={truth.cy:.1f} "
          f"distortion={['%+.3f' % value for value in truth.distortion]}")
    print(f"Board: {DEFAULT_BOARD.squares_x}x{DEFAULT_BOARD.squares_y}, "
          f"{DEFAULT_BOARD.square_length_m * 1000:.0f} mm squares, "
          f"{DEFAULT_BOARD.interior_corners} corners")
    print("Not a real camera: rendered board views, no blur, no defocus. A floor, not an estimate.")

    chosen = arguments.sweep
    if chosen in ("tilt", "all"):
        sweep_tilt(arguments.repeats)
    if chosen in ("area", "all"):
        sweep_area(arguments.repeats)
    if chosen in ("model", "all"):
        sweep_model(arguments.repeats)
    if chosen in ("views", "all"):
        sweep_views(arguments.repeats)
    if chosen in ("undistortion", "all"):
        sweep_undistortion()
    if chosen in ("stereo", "all"):
        sweep_stereo(arguments.repeats)
    if chosen in ("cost", "all"):
        measure_cost()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
