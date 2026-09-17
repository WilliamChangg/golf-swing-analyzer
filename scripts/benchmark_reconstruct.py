#!/usr/bin/env python3
"""Measure what a 3D reconstruction is worth, against a body that is an input.

**Ground truth in the same sense as Phase 8's benchmark, and no stronger.** The
swing is a function in `tests/synthetic_body3d.py`: every 3D point the
reconstruction is asked to recover is a number put in there, so the error is a
subtraction rather than a judgement. What the fixture does not contain is a pose
estimator -- no motion blur at the bottom of the downswing, no hip the model
guesses at through an occlusion, no frame where the two wrists swap identity.
Noise is added as an explicit displacement, which has none of the structure a
real estimator's error has. **The errors below are a floor.**

The question these sweeps exist to answer is not "how accurate is
triangulation". It is *whether the number a reconstruction reports about itself
can tell you when it has gone wrong* -- because the obvious one cannot:

    noise         what does landmark scatter cost, in millimetres?
    epipolar      does the reprojection error notice a point at the wrong depth?
    convergence   what does camera placement cost, and where does it stop paying?
    pairing       what does resampling buy over pairing nearest frames?
    refine        what does the non-linear step buy over the linear solution?
    metrics       how well do the 3D rotations recover angles that are known?
    cost          how long does any of this take?

`epipolar` is the one that matters. It sweeps a displacement the residual is
blind to by construction and watches the residual stay flat while the true error
grows by an order of magnitude.

Usage:
    uv run --project python python scripts/benchmark_reconstruct.py
    uv run --project python python scripts/benchmark_reconstruct.py --sweep epipolar
    uv run --project python python scripts/benchmark_reconstruct.py --repeats 5
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

from analyzer.biomechanics import compute_metrics  # noqa: E402
from analyzer.contracts.calibration import CalibrationStatus  # noqa: E402
from analyzer.contracts.filtering import FilterConfig  # noqa: E402
from analyzer.contracts.metrics import MetricName  # noqa: E402
from analyzer.contracts.phases import SwingEvent  # noqa: E402
from analyzer.contracts.pose import Landmark  # noqa: E402
from analyzer.contracts.reconstruction import ReconstructionConfig  # noqa: E402
from analyzer.contracts.sync import TimeMap  # noqa: E402
from analyzer.filtering.landmarks import filter_sequence  # noqa: E402
from analyzer.phases import detect_phases  # noqa: E402
from analyzer.reconstruction import ReconstructedSequence, reconstruct_pair  # noqa: E402
from tests.synthetic_body3d import (  # noqa: E402
    PELVIS_TURN_DEG,
    SHOULDER_TURN_DEG,
    TOP_S,
    SwingRig,
    body_at,
    camera_rig,
    default_rig,
    pose_sequence_for,
)

# Landmarks the summary statistics are taken over. The fixture's legs and face
# are deliberately static, so a statistic over all 33 is dominated by parts that
# do not move and is blind to every error that scales with speed -- which is the
# whole of the pairing term. These are the ones every metric is measured from.
MOVING = (
    Landmark.LEFT_SHOULDER,
    Landmark.RIGHT_SHOULDER,
    Landmark.LEFT_ELBOW,
    Landmark.RIGHT_ELBOW,
    Landmark.LEFT_WRIST,
    Landmark.RIGHT_WRIST,
    Landmark.LEFT_HIP,
    Landmark.RIGHT_HIP,
)


@dataclass(frozen=True)
class Outcome:
    """One reconstruction, scored against the body that produced it."""

    median_error_m: float
    p95_error_m: float
    hand_error_m: float
    reprojection_px: float
    convergence_deg: float
    uncertainty_m: float
    bone_variation: float
    coverage: float

    @staticmethod
    def of(rig: SwingRig, result: ReconstructedSequence) -> Outcome:
        truth = rig.reference.to_camera(
            np.stack([body_at(rig.reference.world_start_s + float(t)) for t in result.t])
        )
        distance = np.linalg.norm(result.points - truth, axis=2)
        moving = distance[:, [int(landmark) for landmark in MOVING]]
        hands = distance[:, [int(Landmark.LEFT_WRIST), int(Landmark.RIGHT_WRIST)]]

        quality = result.report.quality
        return Outcome(
            median_error_m=_finite_stat(moving, np.median),
            p95_error_m=_finite_percentile(moving, 95.0),
            hand_error_m=_finite_percentile(hands, 95.0),
            reprojection_px=_or_nan(quality.median_reprojection_px if quality else None),
            convergence_deg=_or_nan(quality.median_convergence_deg if quality else None),
            uncertainty_m=_or_nan(quality.median_uncertainty_m if quality else None),
            bone_variation=_or_nan(quality.worst_bone_variation if quality else None),
            coverage=_or_nan(quality.coverage if quality else None),
        )


def _finite_stat(values: np.ndarray, reducer) -> float:  # type: ignore[no-untyped-def]
    finite = values[np.isfinite(values)]
    return float(reducer(finite)) if finite.size else float("nan")


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _or_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _median_outcome(outcomes: list[Outcome]) -> Outcome:
    """The median of each field independently, which is what a sweep row reports."""

    def pick(name: str) -> float:
        values = [getattr(entry, name) for entry in outcomes]
        finite = [value for value in values if np.isfinite(value)]
        return float(np.median(finite)) if finite else float("nan")

    return Outcome(*(pick(field) for field in Outcome.__annotations__))


def reconstruct(
    rig: SwingRig,
    *,
    seed: int = 11,
    noise_px: float = 0.0,
    epipolar_noise_px: float = 0.0,
    config: ReconstructionConfig | None = None,
    sync_uncertainty_s: float = 0.001,
) -> ReconstructedSequence:
    """One reconstruction of the synthetic swing, end to end through the real pipeline."""
    contract = camera_rig(rig)
    reference = filter_sequence(
        pose_sequence_for(rig.reference, noise_px=noise_px, seed=seed),
        FilterConfig(),
        intrinsics=contract.cameras[rig.reference.role].intrinsics,
    )
    target = filter_sequence(
        pose_sequence_for(
            rig.target,
            noise_px=noise_px,
            epipolar_noise_px=epipolar_noise_px,
            epipolar_towards=rig.reference,
            seed=seed + 500,
        ),
        FilterConfig(),
        intrinsics=contract.cameras[rig.target.role].intrinsics,
    )
    time_map = TimeMap(
        offset_s=rig.time_offset_s(),
        rate=1.0,
        rate_estimated=False,
        pivot_s=TOP_S,
        offset_uncertainty_s=sync_uncertainty_s,
        support_start_s=0.5,
        support_end_s=2.0,
    )
    return reconstruct_pair(
        reference,
        target,
        contract,
        time_map,
        reference_role=rig.reference.role,
        target_role=rig.target.role,
        config=config,
    )


def _heading(title: str, columns: str) -> None:
    print(f"\n{title}")
    print(columns)
    print("-" * len(columns))


def _mm(value: float) -> str:
    return "     -" if not np.isfinite(value) else f"{value * 1000:6.1f}"


def sweep_noise(repeats: int) -> None:
    """What landmark scatter costs, in millimetres.

    The headline accuracy table. Phase 3 measured 0.0014 frame widths of scatter
    on real footage, which on a 1920-wide frame is 2.7 px, so that row is the one
    to read for what this pipeline would do on a perfect capture of a real swing.
    """
    _heading(
        "Reconstruction error against isotropic landmark noise",
        " sigma px |  median |    p95  |  hands  | reproj px | unc mm | bone var",
    )
    for sigma in (0.0, 0.5, 1.0, 2.7, 5.0, 10.0):
        outcomes = [
            Outcome.of(default_rig(), reconstruct(default_rig(), noise_px=sigma, seed=11 + seed))
            for seed in range(repeats)
        ]
        row = _median_outcome(outcomes)
        print(
            f" {sigma:8.1f} | {_mm(row.median_error_m)}  | {_mm(row.p95_error_m)}  | "
            f"{_mm(row.hand_error_m)}  | {row.reprojection_px:9.2f} | "
            f"{_mm(row.uncertainty_m)} | {row.bone_variation:7.1%}"
        )
    print("  millimetres, over the eight landmarks that move; hands is their 95th percentile")
    print("  2.7 px is the scatter Phase 3 measured on real footage, at 1920 px wide")


def sweep_epipolar(repeats: int) -> None:
    """**The sweep this phase is built around.**

    A landmark displaced along the epipolar line slides the reconstructed point
    up or down its own ray, where a perfect fit is always available. So the
    residual cannot see it, and the table shows the residual essentially flat
    while the true error grows by more than an order of magnitude.

    Read it beside `sweep_noise`: the same number of pixels of *isotropic* noise
    moves the residual proportionally, because half of that noise has no 3D
    explanation. It is only the along-epipolar half that is invisible.
    """
    _heading(
        "Displacement along the epipolar line: the error the residual cannot see",
        " along px |  median |    p95  | reproj px | bone var | unc mm",
    )
    permissive = ReconstructionConfig(max_uncertainty_m=None, max_reprojection_px=1e9)
    for displacement in (0.0, 1.0, 2.0, 4.0, 8.0):
        outcomes = [
            Outcome.of(
                default_rig(),
                reconstruct(
                    default_rig(),
                    epipolar_noise_px=displacement,
                    config=permissive,
                    seed=11 + seed,
                ),
            )
            for seed in range(repeats)
        ]
        row = _median_outcome(outcomes)
        print(
            f" {displacement:8.1f} | {_mm(row.median_error_m)}  | {_mm(row.p95_error_m)}  | "
            f"{row.reprojection_px:9.2f} | {row.bone_variation:7.1%}  | {_mm(row.uncertainty_m)}"
        )
    print("  the residual is flat and the answer is not; bone variation is what notices")

    _heading(
        "The same pixel counts, isotropic, for comparison",
        " sigma px |  median |    p95  | reproj px | bone var",
    )
    for sigma in (0.0, 1.0, 2.0, 4.0, 8.0):
        outcomes = [
            Outcome.of(
                default_rig(),
                reconstruct(default_rig(), noise_px=sigma, config=permissive, seed=11 + seed),
            )
            for seed in range(repeats)
        ]
        row = _median_outcome(outcomes)
        print(
            f" {sigma:8.1f} | {_mm(row.median_error_m)}  | {_mm(row.p95_error_m)}  | "
            f"{row.reprojection_px:9.2f} | {row.bone_variation:7.1%}"
        )


def sweep_convergence(repeats: int) -> None:
    """What camera placement costs. The gate, as a curve.

    Depth uncertainty scales as `1 / sin(theta)` where theta is the angle between
    the two rays, so this is the number that decides whether a capture could have
    determined a position at all -- and it is a property of where the tripods
    went, not of anything the software can fix afterwards.
    """
    _heading(
        "Camera separation: the geometry that decides the answer",
        " sep deg | rays deg |  median |    p95  | reproj px | unc mm | 1/sin",
    )
    permissive = ReconstructionConfig(
        min_convergence_deg=0.5, max_uncertainty_m=None, max_reprojection_px=1e9
    )
    for separation in (90.0, 60.0, 45.0, 30.0, 15.0, 8.0):
        outcomes = []
        for seed in range(repeats):
            rig = default_rig(convergence_deg=separation)
            outcomes.append(
                Outcome.of(rig, reconstruct(rig, noise_px=2.7, config=permissive, seed=11 + seed))
            )
        row = _median_outcome(outcomes)
        amplification = 1.0 / max(np.sin(np.radians(row.convergence_deg)), 1e-6)
        print(
            f" {separation:7.0f} | {row.convergence_deg:8.0f} | {_mm(row.median_error_m)}  | "
            f"{_mm(row.p95_error_m)}  | {row.reprojection_px:9.2f} | "
            f"{_mm(row.uncertainty_m)} | {amplification:5.1f}x"
        )
    print("  at 2.7 px of landmark noise; 'rays deg' is the measured per-point convergence")
    print("  the residual barely moves across the whole sweep, which is the reason for the gate")


def sweep_pairing(repeats: int) -> None:
    """What resampling buys over pairing nearest frames.

    Phase 8 pairs the board views nearest in time and gets away with it, because
    a board can be held still and the cost is `time_error x image_speed`. Nothing
    in a swing is still, so this is where that instruction runs out.
    """
    _heading(
        "Resampling against nearest-frame pairing, at the target's frame rate",
        " tgt fps | method   |  hands p95 | median | worst nearest-frame px",
    )
    for fps in (240.0, 120.0, 60.0, 30.0):
        refused: str | None = None
        for resample in (True, False):
            outcomes = []
            worst_px = float("nan")
            for seed in range(repeats):
                rig = default_rig(target_fps=fps)
                result = reconstruct(
                    rig,
                    noise_px=0.0,
                    config=ReconstructionConfig(resample=resample),
                    seed=11 + seed,
                )
                if not result.report.reconstructed:
                    refused = result.report.refusal
                    continue
                outcomes.append(Outcome.of(rig, result))
                worst_px = _or_nan(result.report.pairing.nearest_frame_error_px)
            if not outcomes:
                continue
            row = _median_outcome(outcomes)
            label = "resample" if resample else "nearest "
            print(
                f" {fps:7.0f} | {label} | {_mm(row.hand_error_m)}     | "
                f"{_mm(row.median_error_m)} | {worst_px:6.1f}"
            )
        if refused is not None:
            # A 30 fps clip cannot support the 0.10 s window at degree 4, which is
            # Phase 3's frame-rate floor and not this layer's refusal. Printed
            # rather than left as a blank row, because "no number" and "this
            # capture cannot be analysed at all" are different answers.
            print(f" {fps:7.0f} | refused  | the target clip produced no filtered trajectory")
    print("  no landmark noise, so every millimetre here is the pairing and nothing else")
    print("  'worst nearest-frame px' is a quarter of a frame interval at the fastest landmark")

    _heading(
        "And what the alignment's own uncertainty costs, resampled",
        " sync ms |  hands p95 | median | pairing px",
    )
    for uncertainty_ms in (0.0, 2.0, 5.0, 15.0):
        outcomes = []
        pairing_px = float("nan")
        for seed in range(repeats):
            rig = default_rig()
            result = reconstruct(
                rig, noise_px=0.0, sync_uncertainty_s=uncertainty_ms / 1000.0, seed=11 + seed
            )
            outcomes.append(Outcome.of(rig, result))
            pairing_px = _or_nan(result.report.pairing.max_pairing_error_px)
        row = _median_outcome(outcomes)
        print(
            f" {uncertainty_ms:7.1f} | {_mm(row.hand_error_m)}     | "
            f"{_mm(row.median_error_m)} | {pairing_px:9.2f}"
        )
    print("  the uncertainty is reported, not applied: it does not move the answer, it bounds it")


def sweep_refine(repeats: int) -> None:
    """What the non-linear step buys over the linear solution."""
    _heading(
        "Linear triangulation against reprojection-refined",
        " sigma px | method   |  median |    p95  | reproj px",
    )
    for sigma in (0.0, 2.7, 8.0):
        for refine in (False, True):
            outcomes = [
                Outcome.of(
                    default_rig(),
                    reconstruct(
                        default_rig(),
                        noise_px=sigma,
                        config=ReconstructionConfig(refine=refine, max_uncertainty_m=None),
                        seed=11 + seed,
                    ),
                )
                for seed in range(repeats)
            ]
            row = _median_outcome(outcomes)
            label = "refined " if refine else "linear  "
            print(
                f" {sigma:8.1f} | {label} | {_mm(row.median_error_m)}  | "
                f"{_mm(row.p95_error_m)}  | {row.reprojection_px:9.2f}"
            )


def sweep_metrics(repeats: int) -> None:
    """How well the 3D rotations recover angles that are known by construction.

    The fixture turns the shoulders and the pelvis by declared amounts about a
    declared axis, so the X-factor at the top has a true value -- which is the
    one thing no single-camera measurement in this project has ever had.
    """
    _heading(
        "3D rotations against the angles the fixture was built with",
        " sigma px | shoulder | pelvis | X-factor | reported +/- | worst err",
    )
    truth = {
        MetricName.SHOULDER_TURN_3D: SHOULDER_TURN_DEG,
        MetricName.PELVIS_TURN_3D: PELVIS_TURN_DEG,
        MetricName.X_FACTOR_3D: SHOULDER_TURN_DEG - PELVIS_TURN_DEG,
    }

    for sigma in (0.0, 1.0, 2.7, 5.0):
        rows: dict[MetricName, list[float]] = {name: [] for name in truth}
        spreads: list[float] = []
        undetected = 0
        for seed in range(repeats):
            rig = default_rig()
            contract = camera_rig(rig)
            reference = filter_sequence(
                pose_sequence_for(rig.reference, noise_px=sigma, seed=11 + seed),
                FilterConfig(),
                intrinsics=contract.cameras[rig.reference.role].intrinsics,
            )
            result = reconstruct(rig, noise_px=sigma, seed=11 + seed)
            phases = detect_phases(reference)
            if not phases.detected:
                undetected += 1
                continue
            measured = compute_metrics(reference, phases, None, CalibrationStatus.STEREO, result)
            for name in truth:
                metric = measured.get(name, SwingEvent.TOP)
                if metric is not None:
                    rows[name].append(metric.value)
                    if name is MetricName.X_FACTOR_3D and metric.uncertainty is not None:
                        spreads.append(metric.uncertainty)

        def summarise(name: MetricName) -> tuple[float, float]:
            values = rows[name]
            if not values:
                return float("nan"), float("nan")
            return float(np.median(values)), float(np.max(np.abs(np.array(values) - truth[name])))

        if undetected == repeats:
            # Worth a line rather than a row of blanks: it is the finding. The
            # reconstruction is fine at this noise level -- `sweep_noise` puts it
            # at a few millimetres -- and Phase 4 declines to call the clip a
            # swing, so there is no instant to anchor a rotation to. On noisy
            # footage the limit on a 3D metric is the phase detector, not the
            # triangulation.
            print(
                f" {sigma:8.1f} |  no swing detected in any seed -- Phase 4 refuses the clip, "
                "and the reconstruction is unaffected"
            )
            continue

        shoulder, shoulder_err = summarise(MetricName.SHOULDER_TURN_3D)
        pelvis, pelvis_err = summarise(MetricName.PELVIS_TURN_3D)
        factor, factor_err = summarise(MetricName.X_FACTOR_3D)
        spread = float(np.median(spreads)) if spreads else float("nan")
        print(
            f" {sigma:8.1f} | {shoulder:8.1f} | {pelvis:6.1f} | {factor:8.1f} | "
            f"{spread:11.1f} | {max(shoulder_err, pelvis_err, factor_err):8.1f}"
        )
    print(
        f"  truth: shoulder {SHOULDER_TURN_DEG:.0f}, pelvis {PELVIS_TURN_DEG:.0f}, "
        f"X-factor {SHOULDER_TURN_DEG - PELVIS_TURN_DEG:.0f} degrees"
    )
    print("  'reported +/-' is the X-factor's own propagated uncertainty, for comparison")


def measure_cost() -> None:
    """Where the time goes, against the pose extraction it sits on top of."""
    rig = default_rig()
    contract = camera_rig(rig)

    started = time.perf_counter()
    sequences = [
        pose_sequence_for(rig.reference, noise_px=2.7),
        pose_sequence_for(rig.target, noise_px=2.7, seed=99),
    ]
    build = time.perf_counter() - started

    started = time.perf_counter()
    filtered = [
        filter_sequence(
            sequences[0], FilterConfig(), intrinsics=contract.cameras[rig.reference.role].intrinsics
        ),
        filter_sequence(
            sequences[1], FilterConfig(), intrinsics=contract.cameras[rig.target.role].intrinsics
        ),
    ]
    filtering = time.perf_counter() - started

    time_map = TimeMap(
        offset_s=rig.time_offset_s(),
        rate=1.0,
        rate_estimated=False,
        pivot_s=TOP_S,
        offset_uncertainty_s=0.001,
        support_start_s=0.5,
        support_end_s=2.0,
    )

    timings: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        result = reconstruct_pair(
            filtered[0],
            filtered[1],
            contract,
            time_map,
            reference_role=rig.reference.role,
            target_role=rig.target.role,
        )
        timings.append(time.perf_counter() - started)

    frames = len(result)
    points = result.report.quality.points_reconstructed if result.report.quality else 0
    _heading("Cost", " stage                         | median   | note")
    print(f" build the fixture (2 clips)    | {build * 1000:6.1f} ms | not part of the pipeline")
    print(f" filter both clips              | {filtering * 1000:6.1f} ms | 33 landmarks x 3 axes x 2")
    print(
        f" reconstruct                    | {np.median(timings) * 1000:6.1f} ms | "
        f"{frames} frames, {points} points"
    )
    print("  against roughly 1.3 s per clip to extract poses in the first place")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Seeds per measurement.")
    parser.add_argument(
        "--sweep",
        choices=("noise", "epipolar", "convergence", "pairing", "refine", "metrics", "cost", "all"),
        default="all",
    )
    arguments = parser.parse_args()

    rig = default_rig()
    print(
        f"Synthetic rig: {rig.reference.role.value} at fx={rig.reference.intrinsics.fx:.0f} and "
        f"{rig.target.role.value} at fx={rig.target.intrinsics.fx:.0f}, "
        f"{rig.baseline_m:.2f} m apart, {rig.convergence_deg:.0f} deg between the axes"
    )
    print(
        f"Clocks: {rig.reference.fps:.0f} fps and {rig.target.fps:.0f} fps, "
        f"{rig.time_offset_s() * 1000:+.0f} ms apart"
    )
    print(
        "Not a real capture: the body is an input and there is no pose estimator, so there is "
        "no blur, no occlusion and no mis-tracked wrist. A floor, not an estimate."
    )

    chosen = arguments.sweep
    if chosen in ("noise", "all"):
        sweep_noise(arguments.repeats)
    if chosen in ("epipolar", "all"):
        sweep_epipolar(arguments.repeats)
    if chosen in ("convergence", "all"):
        sweep_convergence(arguments.repeats)
    if chosen in ("pairing", "all"):
        sweep_pairing(arguments.repeats)
    if chosen in ("refine", "all"):
        sweep_refine(arguments.repeats)
    if chosen in ("metrics", "all"):
        sweep_metrics(arguments.repeats)
    if chosen in ("cost", "all"):
        measure_cost()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
