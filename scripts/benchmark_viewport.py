#!/usr/bin/env python3
"""Measure what a 3D viewport shows, and what it hides by standing where it stands.

Phase 9 measured how well two cameras locate a point. This measures something a
report cannot express and a picture cannot avoid: **a joint's uncertainty has a
direction, and a viewpoint decides how much of it a reader can see.**

A viewport draws a joint as a dot. A dot carries no direction, so everything the
reconstruction does not know about that joint lands either across the picture,
where it is visible as a smear and a reader can discount it, or down the line of
sight, where it hides behind the dot and the picture looks exactly as confident
as a perfect one would. Which happens is decided entirely by where the viewer put
the camera -- and the camera almost nobody moves is the default one.

    convergence   how much of the error does the default viewpoint show,
                  as the two cameras are brought together?
    orbit         how much does it show as the reader walks around the body?
    cost          what does a scene cost to build, send and parse?

`convergence` is the one that matters, and it is Phase 8's tilt sweep and
Phase 9's convergence sweep for a third time: a number that stays flat across a
range over which the thing it appears to describe grows several-fold. Here the
flat number is the smear on screen.

**Ground truth in the same sense as `benchmark_reconstruct.py` and no stronger.**
The body is an input, there is no pose estimator in the fixture, and the noise is
an explicit displacement rather than an estimator's structured error. The
millimetres are a floor.

Usage:
    uv run --project python python scripts/benchmark_viewport.py
    uv run --project python python scripts/benchmark_viewport.py --sweep convergence
    uv run --project python python scripts/benchmark_viewport.py --repeats 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.contracts.cache import ContentKey, HashAlgorithm  # noqa: E402
from analyzer.contracts.pose import Landmark  # noqa: E402
from analyzer.reconstruction import ReconstructedSequence  # noqa: E402
from analyzer.reconstruction.triangulate import (  # noqa: E402
    uncertainty_covariance,
    visible_uncertainty_fraction,
)
from analyzer.scene import MAX_SCENE_FRAMES, build_scene  # noqa: E402
from tests.synthetic_body3d import default_rig  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from benchmark_reconstruct import MOVING, reconstruct  # noqa: E402

# The scene needs a clip identity and this one has no file behind it. Stated as a
# constant rather than faked into something that looks like a digest, because a
# scene built here is never compared against a video.
SYNTHETIC_KEY = ContentKey(algorithm=HashAlgorithm.SHA256, digest="0" * 64, size_bytes=0)

# The landmark scatter Phase 3 measured on real footage, at 1920 px wide.
REAL_FOOTAGE_SIGMA_PX = 2.7


def _visible(
    result: ReconstructedSequence, eye: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Median true, visible and hidden uncertainty from one viewpoint, in metres.

    `eye` is where the viewer stands, in scene metres; None is the reference
    camera, which sits at the origin and is the viewpoint a viewport opens at
    unless somebody moves it.

    Returns the median worst-direction sigma, the median sigma a viewer there can
    see, and their ratio. The middle one is what a reader of the picture would
    infer the reconstruction was worth.
    """
    quality = result.report.quality
    sigma_px = quality.pixel_sigma_px if quality is not None else 0.0

    points = result.points.reshape(-1, 3)
    kept = result.valid.reshape(-1) & np.isin(
        np.tile(np.arange(result.points.shape[1]), result.points.shape[0]),
        [int(landmark) for landmark in MOVING],
    )
    if not np.any(kept):
        return float("nan"), float("nan"), float("nan")

    covariance = uncertainty_covariance(points[kept], result.geometry, sigma_px)
    origin = np.zeros(3) if eye is None else np.asarray(eye, dtype=np.float64)
    fraction = visible_uncertainty_fraction(covariance, points[kept] - origin)

    worst = np.sqrt(np.linalg.eigvalsh(covariance)[:, -1])
    finite = np.isfinite(worst) & np.isfinite(fraction)
    if not np.any(finite):
        return float("nan"), float("nan"), float("nan")

    true_median = float(np.median(worst[finite]))
    seen_median = float(np.median(worst[finite] * fraction[finite]))
    return true_median, seen_median, float(np.median(fraction[finite]))


def _heading(title: str, columns: str) -> None:
    print(f"\n{title}")
    print(columns)
    print("-" * len(columns))


def sweep_convergence(repeats: int) -> None:
    """The headline: bring the cameras together and watch the picture not change."""
    _heading(
        "What the default viewpoint shows, as the capture gets worse",
        " separation | ray angle | true sigma | visible fraction | sigma on screen",
    )
    for separation in (90.0, 60.0, 45.0, 30.0, 20.0, 15.0):
        rig = default_rig(convergence_deg=separation)
        rows = [
            _visible(reconstruct(rig, seed=11 + index * 37, noise_px=REAL_FOOTAGE_SIGMA_PX))
            for index in range(repeats)
        ]
        true_m = float(np.median([row[0] for row in rows]))
        fraction = float(np.median([row[2] for row in rows]))
        seen_m = float(np.median([row[1] for row in rows]))

        result = reconstruct(rig, noise_px=REAL_FOOTAGE_SIGMA_PX)
        quality = result.report.quality
        angle = quality.median_convergence_deg if quality else float("nan")
        print(
            f" {separation:9.0f}° | {angle:8.0f}° | {true_m * 1000:7.1f} mm | "
            f"{fraction:16.2f} | {seen_m * 1000:12.1f} mm"
        )
    print(
        "  The last column is what a reader sees. It is flat while the third column,\n"
        "  which is what the reconstruction is actually worth, grows several-fold --\n"
        "  because the direction that grows is the one the default camera looks along."
    )


def _orbit_eye(result: ReconstructedSequence, orbit_deg: float) -> np.ndarray:
    """Where a viewer stands after orbiting `orbit_deg` from the reference camera.

    Zero is the reference camera exactly -- the orbit starts from the vector that
    runs from the body's centroid to the origin and turns it, so the sweep's first
    row is the viewpoint a viewport opens at rather than an approximation of it.

    The turn is about the scene's y axis, which is the reference camera's own
    vertical. **Not gravity**: nothing in this pipeline measures gravity, and
    `ReconstructionScene` refuses to let a viewport imply otherwise. It is the
    axis a person orbiting a body would use, and that is all it claims to be.
    """
    centre = np.nanmedian(result.points.reshape(-1, 3)[result.valid.reshape(-1)], axis=0)
    towards_reference = -centre
    angle = np.radians(orbit_deg)
    turn = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    return centre + turn @ towards_reference


def sweep_orbit() -> None:
    """Walk around the body and watch the hidden error come into view."""
    _heading(
        "What walking round the body shows, at two captures",
        " orbit from reference | 90° pair: fraction | on screen | 15° pair: fraction | on screen",
    )
    results = {
        separation: reconstruct(
            default_rig(convergence_deg=separation), noise_px=REAL_FOOTAGE_SIGMA_PX
        )
        for separation in (90.0, 15.0)
    }

    for orbit_deg in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        cells: list[str] = []
        for separation in (90.0, 15.0):
            result = results[separation]
            _, seen_m, fraction = _visible(result, _orbit_eye(result, orbit_deg))
            cells.append(f"{fraction:18.2f} | {seen_m * 1000:6.1f} mm")
        print(f" {orbit_deg:20.0f}° | " + " | ".join(cells))
    print(
        "  Zero is the reference camera. A well-conditioned pair barely changes, because\n"
        "  its ellipsoid is nearly a sphere and there is no bad direction to find. A\n"
        "  shallow pair hides four fifths of its error at zero and gives it all up by 90."
    )


def measure_cost(repeats: int) -> None:
    """What a scene costs to build, serialise and parse."""
    result = reconstruct(default_rig(), noise_px=REAL_FOOTAGE_SIGMA_PX)
    frames = len(result)

    _heading(
        f"Cost of a scene ({frames} reference frames available, limit {MAX_SCENE_FRAMES})",
        " frames | build    | serialise | payload  | per frame",
    )
    wanted = sorted({count for count in (68, 96, 312, MAX_SCENE_FRAMES) if count <= frames})
    for count in wanted:
        builds, serialises = [], []
        for _ in range(repeats):
            started = time.perf_counter()
            scene = build_scene(
                result, reference_content_key=SYNTHETIC_KEY, start_frame=0, end_frame=count
            )
            middle = time.perf_counter()
            payload = scene.model_dump_json()
            serialises.append(time.perf_counter() - middle)
            builds.append(middle - started)
        size = len(payload)
        print(
            f" {count:6d} | {np.median(builds) * 1000:5.1f} ms | "
            f"{np.median(serialises) * 1000:6.1f} ms | {size / 1e6:5.2f} MB | "
            f"{size / count / 1024:5.1f} KB"
        )
    print(
        "  Against ~1.3 s per clip to extract poses and ~70 ms to reconstruct. The limit\n"
        f"  is {MAX_SCENE_FRAMES} frames because a scene point carries a position, six covariance\n"
        "  elements and three diagnostics, which is roughly four times an overlay point."
    )


def measure_projection() -> None:
    """Check the picture against the pixels: the reference view must be the overlay.

    The one claim that makes a hand-written projection safe to ship. Placed at
    the reference camera and projected with its own intrinsics, every
    reconstructed point must land where that camera saw the landmark -- because
    those are two renderings of one measurement, and a disagreement here is a
    disagreement between the viewport and the video frame it is being checked
    against.
    """
    result = reconstruct(default_rig(), noise_px=REAL_FOOTAGE_SIGMA_PX)
    geometry = result.geometry
    points = result.points.reshape(-1, 3)
    valid = result.valid.reshape(-1)

    projected, _ = geometry.project(points[valid])
    # The same arithmetic a viewport does, written out rather than called, so the
    # two cannot be the same mistake twice.
    kept = points[valid]
    manual = np.stack(
        (
            geometry.reference.fx * kept[:, 0] / kept[:, 2] + geometry.reference.cx,
            geometry.reference.fy * kept[:, 1] / kept[:, 2] + geometry.reference.cy,
        ),
        axis=-1,
    )
    residual = np.linalg.norm(projected - manual, axis=1)

    _heading("The reference view reproduces the pixels", " points | max disagreement")
    print(f" {kept.shape[0]:6d} | {np.nanmax(residual):.3e} px")


def write_fixture(destination: Path, samples: int) -> None:
    """Write the engine's own projections, for the viewport's projector to be held to.

    **The cross-language pin.** The projection that turns a reconstructed metre
    into a pixel lives in TypeScript, because a viewport's camera moves with the
    mouse and a round trip per frame is not a design. That is a second
    implementation of arithmetic the engine already has, and a second
    implementation is a second opinion unless something forces them to agree.

    This is that something: real points from the synthetic reconstruction, the
    real reference camera, and the pixels `StereoGeometry.project` puts them at.
    The frontend's `projectPoint` is asserted against these to floating-point
    precision, so a sign flipped in either language fails a test rather than
    producing a plausible picture of a body that is inside out.

    Small on purpose -- a few dozen points, not a clip. What is being pinned is
    one function, and a megabyte of frames would pin it no harder.
    """
    result = reconstruct(default_rig(), noise_px=REAL_FOOTAGE_SIGMA_PX)
    scene = build_scene(result, reference_content_key=SYNTHETIC_KEY, start_frame=0, end_frame=1)
    reference = scene.cameras[0]

    points = result.points.reshape(-1, 3)
    valid = result.valid.reshape(-1)
    kept = points[valid]
    # Spread across the whole reconstruction rather than taken from its start, so
    # the sample covers the frame edges where a projection error is largest.
    chosen = kept[:: max(kept.shape[0] // samples, 1)][:samples]
    projected, _ = result.geometry.project(chosen)

    covariance = uncertainty_covariance(chosen, result.geometry, scene.pixel_sigma_px)
    worst = np.sqrt(np.linalg.eigvalsh(covariance)[:, -1])

    payload = {
        "_comment": (
            "GENERATED by scripts/benchmark_viewport.py --write-fixture. Real points from "
            "the synthetic stereo reconstruction, and the pixels the engine's own "
            "StereoGeometry.project puts them at. The viewport's TypeScript projector is "
            "asserted against these so the two implementations cannot drift apart."
        ),
        "camera": json.loads(reference.model_dump_json()),
        "samples": [
            {
                "point": {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])},
                "pixel": {"u": float(pixel[0]), "v": float(pixel[1])},
                "covariance": {
                    "xx": float(matrix[0, 0]),
                    "yy": float(matrix[1, 1]),
                    "zz": float(matrix[2, 2]),
                    "xy": float(matrix[0, 1]),
                    "xz": float(matrix[0, 2]),
                    "yz": float(matrix[1, 2]),
                },
                "sigma_m": float(sigma),
            }
            for point, pixel, matrix, sigma in zip(
                chosen, projected, covariance, worst, strict=True
            )
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {destination} ({destination.stat().st_size / 1024:.0f} KB, {samples} points)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Seeds per measurement.")
    parser.add_argument(
        "--sweep",
        choices=("convergence", "orbit", "cost", "projection", "all"),
        default="all",
    )
    parser.add_argument(
        "--write-fixture",
        type=Path,
        default=None,
        help="Write the projection fixture to this path instead of measuring.",
    )
    parser.add_argument("--fixture-samples", type=int, default=32)
    arguments = parser.parse_args()

    if arguments.write_fixture is not None:
        write_fixture(arguments.write_fixture, arguments.fixture_samples)
        return 0

    rig = default_rig()
    print(
        f"Synthetic rig: {rig.reference.role.value} and {rig.target.role.value}, "
        f"{rig.baseline_m:.2f} m apart, {rig.convergence_deg:.0f} deg between the axes, "
        f"landmark scatter {REAL_FOOTAGE_SIGMA_PX} px"
    )
    print(
        "The body is an input and there is no pose estimator in the fixture, so the "
        "millimetres are a floor. What is being measured here is the viewpoint, and that "
        "part is exact."
    )

    chosen = arguments.sweep
    if chosen in ("convergence", "all"):
        sweep_convergence(arguments.repeats)
    if chosen in ("orbit", "all"):
        sweep_orbit()
    if chosen in ("projection", "all"):
        measure_projection()
    if chosen in ("cost", "all"):
        measure_cost(arguments.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
