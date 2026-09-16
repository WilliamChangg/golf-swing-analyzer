#!/usr/bin/env python3
"""Measure the temporal filter: accuracy against analytical trajectories, and cost.

Filtering is the first stage whose output cannot be checked by looking at it. A
smoothed trajectory looks correct whatever it does to the underlying motion, and
a velocity curve looks even more convincing. So the only way to know whether the
filter works is to run it over motion whose true velocity and acceleration are
known in closed form, and measure the error.

That is what this script does, and it is also how the defaults in
`contracts/filtering.py` were chosen. The sweep is run at several frame rates
against three analytical trajectories, at a noise level measured from real
footage rather than assumed, and the configuration with the lowest velocity
error is reported. **The recommendation is conditional on those trajectories**:
they are models of swing-like motion, not a golf swing, and no claim is made
that they match one. What they do give is an exact ground truth, which real
footage cannot until Phase 12 provides labelled landmarks.

Sections:
  noise      residual of the fit on real pose data -- an upper bound on landmark noise
  sweep      window_s x polyorder against analytical trajectories, per frame rate
  nonuniform cost of assuming uniform sampling when the sampling is not uniform
  solver     normal equations vs pseudo-inverse: agreement and speed
  throughput ms to filter a clip, by size

Usage:
    uv run --project python python scripts/benchmark_filter.py
    uv run --project python python scripts/benchmark_filter.py --poses <pose.parquet>
    uv run --project python python scripts/benchmark_filter.py --only sweep --markdown
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.contracts.filtering import (  # noqa: E402
    FilterConfig,
    SmoothingConfig,
)
from analyzer.contracts.pose import Landmark, LandmarkSpace  # noqa: E402
from analyzer.filtering.landmarks import filter_sequence  # noqa: E402
from analyzer.filtering.localpoly import local_polynomial_fit  # noqa: E402
from analyzer.pose.series import landmark_series  # noqa: E402
from analyzer.pose.store import read_sequence  # noqa: E402

SECTIONS = ("noise", "sweep", "nonuniform", "solver", "throughput")

# Capture protocol asks for >= 120 fps, 240 preferred (data/README.md). 30 and 60
# are measured anyway, because footage arrives at whatever it arrives at and the
# honest thing is to know what the filter does there rather than to assume.
FRAME_RATES = (30, 60, 120, 240)

WINDOWS_S = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30)
POLYORDERS = (2, 3, 4)

# Landmarks the noise measurement uses: the ones every later phase depends on,
# and the ones that actually move. A nose residual says little about whether a
# wrist trajectory can be differentiated.
NOISE_LANDMARKS = (
    Landmark.LEFT_WRIST,
    Landmark.RIGHT_WRIST,
    Landmark.LEFT_SHOULDER,
    Landmark.RIGHT_SHOULDER,
    Landmark.LEFT_HIP,
    Landmark.RIGHT_HIP,
)

SWING_DURATION_S = 1.6


# --------------------------------------------------------------------------
# Analytical trajectories: exact position, velocity and acceleration.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectory:
    """Motion with closed-form derivatives, so error can be measured rather than eyeballed."""

    name: str
    note: str

    def position(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def velocity(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def acceleration(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError


@dataclass(frozen=True)
class Sinusoid(Trajectory):
    """Pure tone. The textbook case, and the one where any error is unambiguous."""

    frequency_hz: float = 2.0
    amplitude: float = 0.2

    def position(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.amplitude * np.sin(2 * np.pi * self.frequency_hz * t)

    def velocity(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        w = 2 * np.pi * self.frequency_hz
        return self.amplitude * w * np.cos(w * t)

    def acceleration(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        w = 2 * np.pi * self.frequency_hz
        return -self.amplitude * w * w * np.sin(w * t)


@dataclass(frozen=True)
class Chirp(Trajectory):
    """Frequency rising with time, as a swing's motion does from address to impact."""

    f0_hz: float = 0.5
    f1_hz: float = 6.0
    duration_s: float = SWING_DURATION_S
    amplitude: float = 0.2

    @property
    def _rate(self) -> float:
        return (self.f1_hz - self.f0_hz) / self.duration_s

    def _phase(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2 * np.pi * (self.f0_hz * t + 0.5 * self._rate * t * t)

    def _omega(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2 * np.pi * (self.f0_hz + self._rate * t)

    def position(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.amplitude * np.sin(self._phase(t))

    def velocity(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.amplitude * self._omega(t) * np.cos(self._phase(t))

    def acceleration(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        omega, phase = self._omega(t), self._phase(t)
        return self.amplitude * (
            2 * np.pi * self._rate * np.cos(phase) - omega * omega * np.sin(phase)
        )


@dataclass(frozen=True)
class ErfPulse(Trajectory):
    """A monotone transition whose velocity is a single pulse.

    The closest of the three to a downswing: position moves once, from one side
    to the other; speed rises to a peak and falls away; acceleration changes sign
    at the peak. `width_s` sets how sharp the pulse is, and 0.08 s puts most of
    the motion inside about a quarter of a second.
    """

    centre_s: float = 0.8
    width_s: float = 0.08
    amplitude: float = 0.3

    def position(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        from scipy.special import erf

        return self.amplitude * erf((t - self.centre_s) / self.width_s)

    def velocity(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        u = (t - self.centre_s) / self.width_s
        return self.amplitude * 2.0 / (self.width_s * np.sqrt(np.pi)) * np.exp(-u * u)

    def acceleration(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        u = (t - self.centre_s) / self.width_s
        return -2.0 * u / self.width_s * self.velocity(t)


def trajectories() -> tuple[Trajectory, ...]:
    return (
        Sinusoid(name="sine-2hz", note="pure tone at 2 Hz", frequency_hz=2.0),
        Chirp(name="chirp-0.5-6hz", note="0.5 Hz rising to 6 Hz"),
        ErfPulse(name="erf-pulse-80ms", note="single velocity pulse, 80 ms wide"),
    )


# --------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------


def _rms(error: NDArray[np.float64]) -> float:
    finite = error[np.isfinite(error)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def _interior(count: int, fraction: float = 0.1) -> slice:
    """Drop the outer tenth at each end.

    Edge windows are one-sided, so their error is genuinely larger and genuinely
    different. Including them would let an edge policy dominate a comparison
    that is about the interior kernel.
    """
    margin = max(1, int(count * fraction))
    return slice(margin, count - margin)


@dataclass(frozen=True)
class Accuracy:
    position_rms: float
    velocity_rms: float
    acceleration_rms: float
    valid_fraction: float
    peak_time_error_s: float
    peak_speed_error: float


def measure_accuracy(
    trajectory: Trajectory,
    fps: float,
    noise: float,
    window_s: float,
    polyorder: int,
    *,
    seed: int = 0,
    duration_s: float = SWING_DURATION_S,
) -> Accuracy:
    """Filter a noisy sample of a known trajectory and compare against the truth.

    Two kinds of error are reported, because they answer different questions and
    can disagree. RMS says how close the whole curve is. The peak errors say
    whether the *fastest instant* is where and what it really was -- which is
    what Phase 4 asks of this layer when it looks for impact, and which a wide
    smoothing window degrades by flattening the peak without moving RMS much.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / fps)
    truth = trajectory.position(t)
    observed = truth + noise * rng.standard_normal(t.size)

    fit = local_polynomial_fit(
        t,
        observed,
        np.ones_like(t),
        window_s=window_s,
        polyorder=polyorder,
        min_observations=polyorder + 1,
    )

    inner = _interior(t.size)
    true_velocity = trajectory.velocity(t)

    peak_time_error = float("nan")
    peak_speed_error = float("nan")
    fitted_speed = np.abs(fit.velocity[inner])
    if np.any(np.isfinite(fitted_speed)):
        true_speed = np.abs(true_velocity[inner])
        times = t[inner]
        found = int(np.nanargmax(fitted_speed))
        actual = int(np.argmax(true_speed))
        peak_time_error = abs(float(times[found] - times[actual]))
        if true_speed[actual]:
            peak_speed_error = float(
                (fitted_speed[found] - true_speed[actual]) / true_speed[actual]
            )

    return Accuracy(
        position_rms=_rms(fit.value[inner] - truth[inner]),
        velocity_rms=_rms(fit.velocity[inner] - true_velocity[inner]),
        acceleration_rms=_rms(fit.acceleration[inner] - trajectory.acceleration(t)[inner]),
        valid_fraction=float(np.count_nonzero(fit.supported) / t.size),
        peak_time_error_s=peak_time_error,
        peak_speed_error=peak_speed_error,
    )


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def section_noise(poses: Path | None) -> float | None:
    """Residual of the fit on real pose data: an upper bound on landmark noise.

    An upper bound rather than the noise itself, because the residual also
    contains whatever real motion the local polynomial could not follow. A
    narrower window reduces the model error and raises the variance, so the
    figures are reported across several windows and the smallest is the tightest
    bound available without ground truth.
    """
    print("\n== noise floor on real pose data ==")
    if poses is None:
        print("  (skipped: pass --poses <pose.parquet> to measure)")
        return None

    sequence = read_sequence(poses)
    timestamps = np.array([frame.timestamp_s for frame in sequence.frames], dtype=np.float64)
    interval = float(np.median(np.diff(timestamps))) if timestamps.size > 1 else float("nan")
    fps = 1.0 / interval if interval else float("nan")
    print(
        f"  {Path(sequence.video_path).name}: {len(sequence.frames)} frames, "
        f"measured {fps:.2f} fps"
    )

    # Windows sized in samples rather than seconds. A clip's frame rate is
    # whatever it is, and a window fixed in seconds either has too few samples to
    # fit (nothing measured) or far more on a fast clip than a slow one (not
    # comparable). Sample counts keep the fit's support identical across clips,
    # which is what makes the residuals comparable.
    bounds: list[float] = []
    for span in (5, 7, 9):
        window_s = (span - 0.5) * interval
        residuals: list[float] = []
        for landmark in NOISE_LANDMARKS:
            series = landmark_series(sequence, landmark, LandmarkSpace.IMAGE)
            for channel in (series.x, series.y):
                fit = local_polynomial_fit(
                    series.timestamps_s,
                    channel,
                    np.ones_like(channel),
                    window_s=window_s,
                    polyorder=3,
                    min_observations=4,
                )
                if np.isfinite(fit.residual_rms):
                    residuals.append(fit.residual_rms)
        if residuals:
            value = float(np.median(residuals))
            bounds.append(value)
            print(
                f"  {span}-sample window ({window_s:5.3f} s)  median residual "
                f"{value:.5f} normalized_frame  ({value * 1000:.2f} milli-frame)"
            )

    if not bounds:
        print("  no landmark produced a residual")
        return None

    tightest = min(bounds)
    print(f"  -> noise upper bound {tightest:.5f} normalized_frame")
    return tightest


def section_sweep(noise: float, markdown: bool) -> None:
    """window_s x polyorder against the analytical trajectories, per frame rate."""
    print(f"\n== accuracy sweep (noise sigma = {noise:.5f} normalized_frame) ==")

    for fps in FRAME_RATES:
        rows: list[tuple[float, int, float, float, float, float, float, float]] = []
        for polyorder in POLYORDERS:
            for window_s in WINDOWS_S:
                results = [
                    measure_accuracy(trajectory, fps, noise, window_s, polyorder)
                    for trajectory in trajectories()
                ]
                # Worst case across the three, not the mean: a configuration that
                # is excellent on a pure tone and poor on the pulse is not a
                # configuration to default to. The peak errors come from the
                # pulse alone, which is the trajectory with one unambiguous
                # velocity maximum to locate.
                pulse = results[-1]
                rows.append(
                    (
                        window_s,
                        polyorder,
                        max(r.position_rms for r in results),
                        max(r.velocity_rms for r in results),
                        max(r.acceleration_rms for r in results),
                        min(r.valid_fraction for r in results),
                        pulse.peak_time_error_s,
                        pulse.peak_speed_error,
                    )
                )

        usable = [row for row in rows if row[5] > 0.0 and np.isfinite(row[3])]
        print(f"\n  --- {fps} fps ---")
        if not usable:
            print("    no configuration produced a value at this frame rate")
            continue

        header = (
            "window_s",
            "order",
            "pos RMS",
            "vel RMS",
            "acc RMS",
            "valid",
            "peak dt ms",
            "peak err",
        )
        if markdown:
            print("| " + " | ".join(header) + " |")
            print("| " + " | ".join("---" for _ in header) + " |")
        else:
            print(
                f"    {'window':>8} {'order':>5} {'pos RMS':>10} {'vel RMS':>9} "
                f"{'acc RMS':>10} {'valid':>6} {'peak dt':>9} {'peak err':>9}"
            )

        for row in rows:
            window, order, pos, vel, acc, valid, peak_dt, peak_err = row
            if markdown:
                print(
                    f"| {window:.3f} | {order} | {pos:.5f} | {vel:.4f} | {acc:.2f} | "
                    f"{valid:.0%} | {peak_dt * 1000:.1f} | {peak_err:+.1%} |"
                )
            else:
                print(
                    f"    {window:>8.3f} {order:>5} {pos:>10.5f} {vel:>9.4f} {acc:>10.2f} "
                    f"{valid:>5.0%} {peak_dt * 1000:>8.1f}ms {peak_err:>+8.1%}"
                )

        for label, index, lower_is_better in (
            ("position RMS", 2, True),
            ("velocity RMS", 3, True),
            ("acceleration RMS", 4, True),
            ("peak speed error", 7, True),
        ):
            finite = [row for row in usable if np.isfinite(row[index])]
            if not finite:
                continue
            best = min(finite, key=lambda row: abs(row[index]) if lower_is_better else -row[index])
            print(f"    best {label:>18}: window {best[0]:.3f} s, order {best[1]}")


def _sampling_patterns(fps: float, duration_s: float, seed: int = 7) -> dict[str, NDArray[np.float64]]:
    """Timelines a consumer camera actually produces.

    Small jitter is the least interesting of these and the easiest to dismiss.
    The two that matter are a rate change -- phone slow-motion switches rate
    mid-clip, and a single mean interval describes neither half -- and dropped
    frames, where the nominal rate is right for most samples and wrong exactly
    where the motion was fastest.
    """
    rng = np.random.default_rng(seed)
    uniform = np.arange(0.0, duration_s, 1.0 / fps)

    patterns: dict[str, NDArray[np.float64]] = {"uniform": uniform}

    for fraction in (0.25, 0.5):
        offsets = fraction / fps * (rng.random(uniform.size) - 0.5)
        patterns[f"jitter-{fraction:.0%}"] = np.sort(uniform + offsets)

    # Slow-motion: the first half at `fps`, the second at a quarter of it.
    half = duration_s / 2
    fast = np.arange(0.0, half, 1.0 / fps)
    slow = np.arange(half, duration_s, 4.0 / fps)
    patterns["rate-change-4x"] = np.concatenate([fast, slow])

    # Dropped frames, clustered where the motion is fastest rather than spread
    # uniformly -- which is how tracking actually fails.
    keep = np.ones(uniform.size, dtype=bool)
    centre = int(uniform.size * 0.5)
    burst = rng.choice(np.arange(centre - 30, centre + 30), size=20, replace=False)
    keep[burst] = False
    patterns["dropped-frames"] = uniform[keep]

    return patterns


def section_nonuniform(noise: float) -> None:
    """What assuming uniform sampling costs when the sampling is not uniform.

    This is the measurement that decides the design. If fitting on true
    timestamps were no better than applying a fixed Savitzky-Golay kernel and
    calling the spacing uniform, there would be no reason to solve a least-
    squares problem per sample. Run at low noise as well as measured noise,
    because at high noise the timing error hides underneath the noise and the
    comparison stops being about timing at all.
    """
    print("\n== cost of assuming uniform sampling (velocity RMS error) ==")

    fps = 120.0
    window_s = SmoothingConfig().window_s
    polyorder = SmoothingConfig().polyorder

    for label, sigma in (("noise-free", 0.0), (f"sigma={noise:.5f}", noise)):
        print(f"\n  --- {label} ---")
        print(
            f"    {'sampling':>16} {'trajectory':>16} {'true-t':>10} {'assumed-uniform':>16} {'ratio':>8}"
        )
        rng = np.random.default_rng(11)
        for name, t in _sampling_patterns(fps, SWING_DURATION_S).items():
            for trajectory in trajectories():
                truth_v = trajectory.velocity(t)
                observed = trajectory.position(t) + sigma * rng.standard_normal(t.size)

                fit = local_polynomial_fit(
                    t,
                    observed,
                    np.ones_like(t),
                    window_s=window_s,
                    polyorder=polyorder,
                    min_observations=polyorder + 1,
                )

                # The mistake being priced: run the stock kernel and take the
                # sample spacing to be the mean interval.
                width = max(polyorder + 2, round(window_s * fps) | 1)
                width = min(width, (t.size - 1) | 1)
                assumed = savgol_filter(
                    observed,
                    width,
                    polyorder,
                    deriv=1,
                    delta=float(np.mean(np.diff(t))),
                    mode="interp",
                )

                inner = _interior(t.size)
                ours = _rms(fit.velocity[inner] - truth_v[inner])
                theirs = _rms(assumed[inner] - truth_v[inner])
                ratio = theirs / ours if ours else float("nan")
                print(
                    f"    {name:>16} {trajectory.name:>16} {ours:>10.4f} {theirs:>16.4f} "
                    f"{ratio:>7.2f}x"
                )


def _median_seconds(fn: Callable[[], object], repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _measure_solver(frames: int, fps: int) -> tuple[int, float, float]:
    """One row of the solver comparison.

    A function rather than a loop body so the timed closures capture this
    call's arrays rather than whatever the loop variable holds when they
    eventually run.
    """
    t = np.arange(frames) / fps
    y = np.sin(2 * np.pi * 3 * t)
    w = np.ones(frames)

    fast = local_polynomial_fit(t, y, w, window_s=0.1, polyorder=3, min_observations=4)

    # The same windows solved by SVD: an independent check on the fast path.
    reference = _fit_via_pinv(t, y, w, window_s=0.1, polyorder=3)
    scale = float(np.nanmax(np.abs(reference)))
    diff = float(np.nanmax(np.abs(fast.value - reference))) / (scale if scale else 1.0)

    seconds_fast = _median_seconds(
        lambda: local_polynomial_fit(t, y, w, window_s=0.1, polyorder=3, min_observations=4)
    )
    seconds_slow = _median_seconds(lambda: _fit_via_pinv(t, y, w, window_s=0.1, polyorder=3))
    return fast.window_samples_max, diff, seconds_slow / seconds_fast


def section_solver() -> None:
    """Normal equations against the pseudo-inverse: agreement and speed."""
    print("\n== solver: normal equations vs pseudo-inverse ==")
    print(f"    {'frames':>8} {'fps':>5} {'window pts':>11} {'max rel diff':>13} {'speedup':>8}")

    for frames, fps in ((240, 60), (1000, 120), (14400, 240)):
        width, diff, speedup = _measure_solver(frames, fps)
        print(f"    {frames:>8} {fps:>5} {width:>11} {diff:>13.2e} {speedup:>7.1f}x")


def _fit_via_pinv(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    *,
    window_s: float,
    polyorder: int,
) -> NDArray[np.float64]:
    """The same fit solved by SVD, as an independent check on the fast path."""
    half = window_s / 2
    lo = np.searchsorted(t, t - half, "left")
    hi = np.searchsorted(t, t + half, "right")
    width = int((hi - lo).max())
    index = lo[:, None] + np.arange(width)[None, :]
    inside = index < hi[:, None]
    index = np.where(inside, index, 0)

    u = (t[index] - t[:, None]) / half
    usable = inside & np.isfinite(y[index]) & (w[index] > 0)
    sqrt_w = np.sqrt(np.where(usable, w[index], 0.0))
    design = (u[..., None] ** np.arange(polyorder + 1)) * sqrt_w[..., None]
    target = np.where(usable, y[index], 0.0) * sqrt_w

    coefficients = np.einsum("npk,nk->np", np.linalg.pinv(design), target)
    supported = usable.sum(axis=1) >= polyorder + 1
    return np.where(supported, coefficients[:, 0], np.nan)


def section_throughput(poses: Path | None) -> None:
    """Cost of filtering every landmark of a clip."""
    print("\n== throughput: filtering all 33 landmarks x 3 axes ==")
    print(f"    {'source':>28} {'frames':>8} {'median ms':>10} {'ms/frame':>9}")

    if poses is None:
        print("    (skipped: pass --poses <pose.parquet> to measure on real data)")
        return

    sequence = read_sequence(poses)
    config = FilterConfig()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        filter_sequence(sequence, config)
        samples.append((time.perf_counter() - start) * 1000)

    median = float(np.median(samples))
    frames = len(sequence.frames)
    print(
        f"    {Path(sequence.video_path).name[:28]:>28} {frames:>8} {median:>10.1f} "
        f"{median / frames if frames else float('nan'):>9.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, help="Pose Parquet file for the real-data sections.")
    parser.add_argument(
        "--noise",
        type=float,
        help="Noise sigma for the sweep. Default: measured from --poses, else 0.002.",
    )
    parser.add_argument(
        "--only",
        choices=SECTIONS,
        action="append",
        help="Run only these sections. Repeatable.",
    )
    parser.add_argument("--markdown", action="store_true", help="Emit sweep tables as markdown.")
    args = parser.parse_args()

    selected = tuple(args.only) if args.only else SECTIONS

    print(f"machine: {platform.platform()} / {platform.processor() or platform.machine()}")
    print(f"numpy {np.__version__}")

    measured: float | None = None
    if "noise" in selected:
        measured = section_noise(args.poses)

    # Fallback only if nothing was measured: stated as a fallback in the output,
    # so a sweep run without footage cannot be mistaken for one with it.
    noise = args.noise if args.noise is not None else measured
    if noise is None:
        noise = 0.002
        if "sweep" in selected or "nonuniform" in selected:
            print(f"\n  ! no noise measured; using fallback sigma {noise} for the sweep")

    if "sweep" in selected:
        section_sweep(noise, args.markdown)
    if "nonuniform" in selected:
        section_nonuniform(noise)
    if "solver" in selected:
        section_solver()
    if "throughput" in selected:
        section_throughput(args.poses)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
