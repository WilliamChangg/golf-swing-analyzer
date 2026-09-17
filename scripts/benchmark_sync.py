#!/usr/bin/env python3
"""Measure how accurately two clips can be aligned, against known offsets.

**This is not a measurement against ground truth, and the difference matters.**
Ground truth for synchronisation would be two cameras that genuinely filmed one
swing at once, with the offset between their clocks established by some means
outside this system -- a clapperboard, a flash, a genlock. No such recording
exists in this project, so what is measured here is one synthetic swing sampled
twice by two simulated cameras, where the offset is an input.

That is a weaker claim, and it is weak in a specific direction worth naming. It
establishes that the arithmetic, the anchor pairing and the correlation recover
an offset that was put in. It cannot establish that a *real* pair recovers one,
because the synthetic cameras differ only in their clocks: they see the same
projection of the same body, and two real cameras see two different projections
whose speed signals are not the same shape. The `--real` mode below runs the
pipeline on whatever footage is to hand and reports what it says, without
pretending the answer is checkable.

Five things are swept, each of which decides something:

    offsets       does a known offset come back, and to within what?
    frame rates   what does pairing a slow camera with a fast one cost?
    noise         how does landmark noise propagate into the alignment?
    factor error  can two cameras measure a wrong slow-motion factor?
    methods       how much worse is correlation alone than anchored events?

Usage:
    uv run --project python python scripts/benchmark_sync.py
    uv run --project python python scripts/benchmark_sync.py --repeats 5
    uv run --project python python scripts/benchmark_sync.py --real <ref.mp4> <target.mp4>
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.contracts.filtering import FilterConfig, SmoothingConfig  # noqa: E402
from analyzer.contracts.sync import SyncConfig, SyncMethod, SyncModel  # noqa: E402
from analyzer.filtering.landmarks import filter_sequence  # noqa: E402
from analyzer.ingestion.probe import probe  # noqa: E402
from analyzer.paths import cache_dir  # noqa: E402
from analyzer.phases import detect_phases, swing_signals  # noqa: E402
from analyzer.pose.estimator import resolve_model  # noqa: E402
from analyzer.pose.store import read_sequence  # noqa: E402
from analyzer.sync import SyncInput, align, quantisation_floor_s  # noqa: E402
from tests.synthetic import DURATION_S, hand_path, pose_sequence  # noqa: E402

# Landmark noise measured from real footage in Phase 3, in frame widths. Applied
# to the wrists here so the alignment is judged on a trajectory as noisy as the
# ones it will actually see, rather than on an analytically clean one.
MEASURED_SIGMA = 0.0014


def camera(
    offset_s: float,
    fps: float,
    *,
    duration_s: float = DURATION_S,
    time_scale: float = 1.0,
    sigma: float = 0.0,
    seed: int = 0,
    config: FilterConfig | None = None,
    name: str = "camera.mov",
) -> SyncInput:
    """One simulated camera's view of the shared swing.

    `offset_s` is how far ahead of the reference this camera's clock reads, and
    is the quantity `align` is asked to recover. Noise is added per camera with
    its own seed, because two cameras make independent estimation errors and
    correlated noise would make the alignment look better than it is.
    """
    clock = np.arange(0.0, duration_s, 1.0 / fps)
    x, y = hand_path(-offset_s + clock / time_scale)

    if sigma > 0.0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0.0, sigma, x.shape)
        y = y + rng.normal(0.0, sigma, y.shape)

    filtered = filter_sequence(pose_sequence(x, y, clock), config or FilterConfig())
    return SyncInput(
        path=Path(name),
        signals=swing_signals(filtered),
        phases=detect_phases(filtered),
        notes=tuple(filtered.report.warnings),
    )


def error_ms(result: SyncModel, truth_s: float) -> float | None:
    if result.time_map is None:
        return None
    return (result.time_map.offset_s - truth_s) * 1000.0


def _cell(value: float | None, width: int = 9) -> str:
    return f"{'refused':>{width}}" if value is None else f"{value:>{width}.1f}"


def sweep_offsets(repeats: int) -> None:
    print("\n## Known offsets, matched cameras at 120 fps")
    print("Error in the recovered offset, in milliseconds.\n")

    floor = quantisation_floor_s(1 / 120, 1 / 120) * 1000
    header = f"{'truth (ms)':>11}{'clean':>10}{'noisy':>10}{'conf':>8}{'residual':>11}"
    print(header)
    print("-" * len(header))

    for truth in (-400, -250, -100, 0, 100, 250, 400, 500):
        reference = camera(0.0, 120.0, name="reference.mov")
        clean = align(reference, camera(truth / 1000, 120.0, name="target.mov"))

        noisy_errors = []
        noisy: SyncModel | None = None
        for seed in range(repeats):
            noisy = align(
                camera(0.0, 120.0, sigma=MEASURED_SIGMA, seed=seed, name="reference.mov"),
                camera(
                    truth / 1000, 120.0, sigma=MEASURED_SIGMA, seed=seed + 1000, name="target.mov"
                ),
            )
            found = error_ms(noisy, truth / 1000)
            if found is not None:
                noisy_errors.append(abs(found))

        residual = (
            noisy.quality.residual_rms_ms
            if noisy is not None and noisy.quality is not None
            else None
        )
        confidence = (
            noisy.confidence.overall if noisy is not None and noisy.confidence is not None else 0.0
        )
        print(
            f"{truth:>11}"
            f"{_cell(error_ms(clean, truth / 1000), 10)}"
            f"{_cell(statistics.median(noisy_errors) if noisy_errors else None, 10)}"
            f"{confidence:>8.2f}"
            f"{_cell(residual, 11)}"
        )

    print(
        f"\nThe frame-rate floor for this pair is {floor:.1f} ms: the standard deviation of "
        "two frame quantisations added in quadrature, and the best any method can do.\n"
        f"Noisy columns are the median absolute error over {repeats} seeds, at the "
        f"sigma = {MEASURED_SIGMA} frame widths Phase 3 measured from real footage."
    )


def sweep_frame_rates() -> None:
    print("\n## Mismatched frame rates")
    print("Both clips share one smoothing window, set by whichever is coarser.\n")

    header = (
        f"{'reference':>11}{'target':>9}{'window':>9}{'floor (ms)':>12}"
        f"{'error (ms)':>12}{'conf':>7}"
    )
    print(header)
    print("-" * len(header))

    for reference_fps, target_fps, window in (
        (240.0, 240.0, 0.10),
        (120.0, 120.0, 0.10),
        (120.0, 60.0, 0.10),
        (120.0, 30.0, 0.25),
        (60.0, 30.0, 0.25),
        (30.0, 30.0, 0.25),
        (240.0, 30.0, 0.25),
    ):
        config = FilterConfig(smoothing=SmoothingConfig(window_s=window))
        truth = 0.35
        result = align(
            camera(0.0, reference_fps, config=config, sigma=MEASURED_SIGMA, seed=1),
            camera(truth, target_fps, config=config, sigma=MEASURED_SIGMA, seed=2),
        )
        floor = quantisation_floor_s(1 / reference_fps, 1 / target_fps) * 1000
        confidence = result.confidence.overall if result.confidence else 0.0
        print(
            f"{reference_fps:>10.0f}{'':>1}{target_fps:>8.0f}{'':>1}{window:>8.2f}{'s':>1}"
            f"{floor:>12.1f}{_cell(error_ms(result, truth), 12)}{confidence:>7.2f}"
        )

    print(
        "\nUpgrading one camera of a 30 fps pair is worth at most a factor of sqrt(2) on "
        "the floor: it removes one of two equal quantisation terms and the other remains."
    )


def sweep_factor_error() -> None:
    print("\n## A wrong slow-motion factor, measured as a clock rate")
    print("What one camera cannot recover about itself, two can recover about each other.\n")

    header = f"{'factor error':>13}{'true rate':>11}{'fitted':>10}{'estimated':>11}{'conf':>7}"
    print(header)
    print("-" * len(header))

    for error in (1.00, 1.01, 1.02, 1.05, 1.10, 1.20, 1.40):
        result = align(
            camera(0.0, 120.0, sigma=MEASURED_SIGMA, seed=1),
            camera(0.0, 120.0, time_scale=1 / error, sigma=MEASURED_SIGMA, seed=2),
        )
        mapping = result.time_map
        confidence = result.confidence.overall if result.confidence else 0.0
        fitted = "-" if mapping is None else f"{mapping.rate:.4f}"
        estimated = "yes" if mapping is not None and mapping.rate_estimated else "no"
        print(
            f"{(error - 1) * 100:>12.0f}%{1 / error:>11.4f}{fitted:>10}"
            f"{estimated:>11}{confidence:>7.2f}"
        )

    print(
        "\n`estimated: no` means the fit found a rate within two standard errors of 1.0 and "
        "dropped it, because keeping an insignificant rate costs accuracy away from the "
        "anchors and buys nothing between them."
    )


def sweep_methods(repeats: int) -> None:
    print("\n## Events against correlation, on the same pairs")
    print("The two are independent estimates and are never averaged.\n")

    header = f"{'truth (ms)':>11}{'events':>10}{'correlation':>13}{'peak':>8}{'rival':>8}"
    print(header)
    print("-" * len(header))

    for truth in (-250, 0, 250, 400):
        events: list[float] = []
        correlations: list[float] = []
        peak = rival = 0.0
        for seed in range(repeats):
            reference = camera(0.0, 120.0, sigma=MEASURED_SIGMA, seed=seed)
            target = camera(truth / 1000, 120.0, sigma=MEASURED_SIGMA, seed=seed + 1000)

            by_events = align(reference, target, SyncConfig(method=SyncMethod.EVENTS))
            by_correlation = align(
                reference, target, SyncConfig(method=SyncMethod.CORRELATION)
            )
            for result, into in ((by_events, events), (by_correlation, correlations)):
                found = error_ms(result, truth / 1000)
                if found is not None:
                    into.append(abs(found))
            if by_correlation.correlation is not None:
                peak = by_correlation.correlation.peak_correlation
                rival = by_correlation.correlation.rival_correlation or 0.0

        print(
            f"{truth:>11}"
            f"{_cell(statistics.median(events) if events else None, 10)}"
            f"{_cell(statistics.median(correlations) if correlations else None, 13)}"
            f"{peak:>8.3f}{rival:>8.3f}"
        )

    print(
        "\nThe rival is the best correlation at a lag far enough away to be a different "
        "alignment. A swing has two speed humps, so one always exists; how far the peak "
        "sits above it is what a correlation-only alignment rests on."
    )


def sweep_partial_overlap() -> None:
    print("\n## Partial overlap: a camera that started late")
    print("The second camera misses the start of the swing, so one of its events is spurious.\n")

    header = f"{'starts at':>10}{'truth (ms)':>12}{'error (ms)':>12}{'anchors':>9}{'conf':>7}"
    print(header)
    print("-" * len(header))

    for start in (0.0, 0.4, 0.8, 1.0, 1.4):
        truth = -start
        result = align(
            camera(0.0, 120.0, sigma=MEASURED_SIGMA, seed=1),
            camera(
                truth,
                120.0,
                duration_s=DURATION_S - start,
                sigma=MEASURED_SIGMA,
                seed=2,
                name="late.mov",
            ),
        )
        confidence = result.confidence.overall if result.confidence else 0.0
        print(
            f"{start:>9.1f}{'s':>1}{truth * 1000:>12.0f}"
            f"{_cell(error_ms(result, truth), 12)}{len(result.anchors):>9}{confidence:>7.2f}"
        )

    print(
        "\nAn anchor whose implied offset disagrees with the median of the others is dropped "
        "before fitting. Where that leaves no majority to believe, nothing is dropped and "
        "the residual reports the disagreement instead."
    )


def measure_cost(repeats: int) -> None:
    print("\n## Cost")

    reference = camera(0.0, 120.0, sigma=MEASURED_SIGMA, seed=1)
    target = camera(0.35, 120.0, sigma=MEASURED_SIGMA, seed=2)

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        align(reference, target)
        timings.append((time.perf_counter() - started) * 1000)

    print(
        f"\nAligning two {len(reference.signals.t)}-frame clips: "
        f"{statistics.median(timings):.1f} ms (median of {repeats}), on signals already "
        "filtered. Nothing is cached, for the same reason Phase 3 caches nothing."
    )


def resolve_poses(path: Path, model: str | None) -> Path:
    if path.suffix == ".parquet":
        return path
    metadata = probe(path).metadata
    entry, _ = resolve_model(model)
    poses = cache_dir() / "poses" / metadata.content_key.as_path_segment() / f"{entry.name}.parquet"
    if not poses.exists():
        raise SystemExit(f"No extracted poses for {path.name}. Run: analyzer extract {path}")
    return poses


def real_pair(
    reference_path: Path,
    target_path: Path,
    *,
    factors: tuple[float, float],
    window: float,
    model: str | None,
) -> None:
    """Run the aligner on real footage and report what it says, checking nothing.

    There is no truth to compare against here. What this is for is the other
    half of the claim: that the reported confidence collapses when the input
    does not support an alignment, which real footage can demonstrate and
    synthetic footage cannot.
    """
    print("\n## Real footage")
    print("No ground truth exists for this pair. The residual is the only evidence.\n")

    config = FilterConfig(smoothing=SmoothingConfig(window_s=window))
    inputs = []
    for path, factor in ((reference_path, factors[0]), (target_path, factors[1])):
        sequence = read_sequence(resolve_poses(path, model))
        filtered = filter_sequence(sequence, config, slow_motion_factor=factor)
        inputs.append(
            SyncInput(
                path=path,
                signals=swing_signals(filtered),
                phases=detect_phases(filtered),
                notes=tuple(filtered.report.warnings),
            )
        )

    result = align(inputs[0], inputs[1])
    mapping, quality, confidence = result.time_map, result.quality, result.confidence

    print(f"aligned:     {result.aligned}  ({result.method.value if result.method else '-'})")
    if mapping is not None:
        uncertainty = (mapping.offset_uncertainty_s or 0.0) * 1000
        print(f"offset:      {mapping.offset_s * 1000:+.1f} +/-{uncertainty:.1f} ms")
        print(f"clock rate:  {mapping.rate:.5f} (estimated: {mapping.rate_estimated})")
    if quality is not None:
        residual = (
            "not measurable"
            if quality.residual_rms_ms is None
            else f"{quality.residual_rms_ms:.1f} ms rms "
            f"({quality.residual_rms_ms / max(quality.quantisation_floor_ms, 1e-9):.1f}x floor)"
        )
        print(f"floor:       {quality.quantisation_floor_ms:.1f} ms")
        print(f"residual:    {residual}")
    if confidence is not None:
        print(
            f"confidence:  {confidence.overall:.2f}  "
            f"(agreement {confidence.agreement:.2f}, anchors {confidence.anchors:.2f}, "
            f"stability {confidence.stability:.2f})"
        )
    if result.correlation is not None:
        print(
            f"correlation: peak {result.correlation.peak_correlation:.3f} at "
            f"{result.correlation.peak_offset_s * 1000:+.1f} ms"
        )
    for entry in result.residuals:
        print(f"  {entry.label:<10} {entry.residual_ms:+8.1f} ms")
    if result.refusal:
        print(f"\nREFUSED: {result.refusal}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9, help="Seeds per noisy measurement.")
    parser.add_argument(
        "--real", nargs=2, type=Path, metavar=("REFERENCE", "TARGET"), help="Two real clips."
    )
    parser.add_argument(
        "--slow-motion",
        nargs=2,
        type=float,
        default=(1.0, 1.0),
        metavar=("REFERENCE", "TARGET"),
        help="Slow-motion factor for each real clip.",
    )
    parser.add_argument("--window", type=float, default=0.10, help="Shared smoothing window.")
    parser.add_argument("--model", default=None, help="Which extraction to read.")
    args = parser.parse_args()

    if args.real:
        real_pair(
            args.real[0],
            args.real[1],
            factors=tuple(args.slow_motion),
            window=args.window,
            model=args.model,
        )
        return 0

    print("Synchronisation, measured against known offsets.")
    print("Not ground truth: two simulated cameras, one synthetic swing. See the docstring.")

    sweep_offsets(args.repeats)
    sweep_frame_rates()
    sweep_partial_overlap()
    sweep_factor_error()
    sweep_methods(args.repeats)
    measure_cost(args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
