#!/usr/bin/env python3
"""Measure what two recordings of a swing can and cannot say about each other.

Four sweeps, and the second is the phase's argument.

    warp        does the phase-relative clock recover a known time warp, and
                what does the obvious alternative cost?
    camera      how far apart do two cameras have to be before an unchanged
                swing reports a difference?
    resolution  what frame rate does a given difference need before the pair
                can resolve it?
    clips       what does the real reference footage produce?
    cost        what does a comparison cost to compute and to send?

`camera` is the one to read. It films **one unchanged swing** from a series of
azimuths and asks what the engine reports as different. Every other phase in this
project has found a number that stays flat while the thing it appears to describe
moves; this finds the mirror image -- a number that moves several-fold while the
thing it appears to describe does not move at all. Nothing in either recording
distinguishes that from a swing that changed, which is why the comparison layer
gates on the camera before it gates on anything else.

**Ground truth in the same sense as `benchmark_reconstruct.py` and no stronger.**
The body in the camera sweep is an input, there is no pose estimator in it, and
the landmark noise is an explicit displacement rather than an estimator's
structured error. The degrees it reports are a floor: a real estimator loses the
shoulders to motion blur exactly where this fixture is perfect.

Usage:
    uv run --project python python scripts/benchmark_compare.py
    uv run --project python python scripts/benchmark_compare.py --sweep camera
    uv run --project python python scripts/benchmark_compare.py --sweep clips
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.biomechanics import compute_metrics  # noqa: E402
from analyzer.comparison import SwingInput, compare, normalise  # noqa: E402
from analyzer.contracts.camera import CameraRole  # noqa: E402
from analyzer.contracts.comparison import (  # noqa: E402
    ComparisonConfig,
    TrajectoryChannel,
)
from analyzer.contracts.filtering import FilterConfig, SmoothingConfig  # noqa: E402
from analyzer.contracts.metrics import MetricName  # noqa: E402
from analyzer.contracts.pose import Landmark, LandmarkSpace  # noqa: E402
from analyzer.filtering.landmarks import filter_sequence  # noqa: E402
from analyzer.ingestion.probe import probe  # noqa: E402
from analyzer.paths import cache_dir  # noqa: E402
from analyzer.phases import detect_phases  # noqa: E402
from analyzer.phases.signals import swing_signals  # noqa: E402
from analyzer.pose.estimator import resolve_model  # noqa: E402
from analyzer.pose.store import read_sequence  # noqa: E402
from tests.synthetic import BODY, SwingShape, swing_sequence  # noqa: E402
from tests.synthetic_body3d import (  # noqa: E402
    DURATION_S,
    SyntheticView,
    look_at,
    pose_sequence_for,
)

# Phase 3 measured this scatter on real footage, in frame widths; at 1920 px it
# is the 2.7 px every other benchmark here quotes.
REAL_FOOTAGE_SIGMA_PX = 2.7

FILTER = FilterConfig(smoothing=SmoothingConfig(window_s=0.12))

# The synthetic body's shoulders span 0.4 torso lengths, which the view detector
# correctly calls oblique. Widened here to what a face-on recording looks like,
# so the sweeps measure the comparison rather than re-measuring Phase 6's gate.
FACE_ON_BODY: dict[int, tuple[float, float]] = {
    **BODY,
    int(Landmark.LEFT_SHOULDER): (0.40, 0.25),
    int(Landmark.RIGHT_SHOULDER): (0.60, 0.25),
    int(Landmark.LEFT_HIP): (0.44, 0.50),
    int(Landmark.RIGHT_HIP): (0.56, 0.50),
}


@dataclass(frozen=True)
class Clip:
    label: str
    path: str
    slow_motion: float = 1.0
    window_s: float | None = None


# The two real pairs this repository can build, and they are the useful part.
PAIRS: tuple[tuple[Clip, Clip], ...] = (
    (
        Clip("tour, DTL, 7x slow", "data/rory/dtl/rory_dtl.mp4", slow_motion=7.0),
        Clip("tour, DTL, 7x slow, second swing", "data/rory/dtl/rory_dtl_2.mp4", slow_motion=7.0),
    ),
    (
        Clip("amateur, face-on, 30 fps", "data/amateur/face-on/PW_face-on.mp4"),
        Clip("amateur, DTL, 30 fps", "data/amateur/dtl/iron_dtl.mp4"),
    ),
    (
        Clip("tour, face-on, 7x slow", "data/rory/face-on/rory_face_on.mp4", slow_motion=7.0),
        Clip("tour, DTL, 7x slow", "data/rory/dtl/rory_dtl.mp4", slow_motion=7.0),
    ),
)


def flat_swing(shape: SwingShape, *, fps: float = 120.0, digest: str = "a") -> SwingInput:
    """One clip from the two-dimensional fixture, through the chain the app runs."""
    sequence = swing_sequence(
        duration_s=shape.finish_s + 0.4, fps=fps, shape=shape, body=FACE_ON_BODY
    )
    sequence.video_content_key.digest = digest * 64
    filtered = filter_sequence(sequence, FILTER, space=LandmarkSpace.FRAME_WIDTHS)
    detected = detect_phases(filtered, None)
    return SwingInput(
        path=f"/data/{digest}.mov",
        content_key=sequence.video_content_key,
        filtered=filtered,
        phases=detected,
        metrics=compute_metrics(filtered, detected),
    )


# --- warp -----------------------------------------------------------------


def sweep_warp(_args: argparse.Namespace) -> None:
    """Does the phase-relative clock recover a warp whose answer is known?

    `tests/synthetic.py` builds its hand arc from the **fraction** through each
    phase, so two swings differing only in their event times are one swing under
    a piecewise-linear time warp. The residual after normalising is therefore the
    map's own error, because the truth is zero.

    The second column is the alternative worth pricing: stretch each clip's
    takeaway-to-finish interval to the same length and leave it at that. It is
    the obvious thing to build and it is wrong in a way that grows with exactly
    the quantity a golfer is most likely to differ by.
    """
    baseline = SwingShape()
    reference = flat_swing(baseline)
    reference_clock = normalise.build_clock(reference.phases)
    reference_signals = swing_signals(reference.filtered)
    axis = normalise.positions(241)

    print("One swing, warped in time. The residual is the map's own error.\n")
    print(f"{'tempo':>7}  {'backswing':>10}  {'downswing':>10}  {'phase-relative':>15}  {'uniform':>9}")

    for shape in (
        SwingShape(takeaway_s=0.50, top_s=1.30, impact_s=1.70),
        SwingShape(takeaway_s=0.50, top_s=1.10, impact_s=1.60),
        SwingShape(takeaway_s=0.50, top_s=0.90, impact_s=1.30),
        SwingShape(takeaway_s=0.50, top_s=1.70, impact_s=2.00),
        SwingShape(takeaway_s=0.80, top_s=2.00, impact_s=2.55),
    ):
        target = flat_swing(shape, digest="b")
        target_clock = normalise.build_clock(target.phases)
        if not (reference_clock.usable and target_clock.usable):
            print(f"{'-':>7}  a clip in this pair produced no clock")
            continue
        target_signals = swing_signals(target.filtered)

        left = normalise.resample(
            reference_clock, reference_signals.t, reference_signals.height, axis
        )
        right = normalise.resample(target_clock, target_signals.t, target_signals.height, axis)
        phase_relative = _residual(left.value, right.value)

        uniform = _uniform_residual(
            reference_clock, target_clock, reference_signals, target_signals, axis
        )

        backswing = reference_clock.knots[1].timestamp_s - reference_clock.knots[0].timestamp_s
        target_backswing = target_clock.knots[1].timestamp_s - target_clock.knots[0].timestamp_s
        downswing = target_clock.knots[2].timestamp_s - target_clock.knots[1].timestamp_s
        print(
            f"{target_backswing / downswing:>7.2f}  {target_backswing:>10.3f}  "
            f"{downswing:>10.3f}  {phase_relative:>15.4f}  {uniform:>9.4f}"
        )
        _ = backswing

    print(
        "\nResiduals are hand height in frame widths; the signal ranges over about "
        "0.30, so 0.01 is three per cent of it."
    )
    print(
        "The uniform column is a single linear stretch of takeaway-to-finish. It is "
        "exact only where the two swings happen to share a tempo, and every row "
        "where they do not is a comparison between the top of one swing and a point "
        "part-way down the other."
    )


def _uniform_residual(reference_clock, target_clock, reference_signals, target_signals, axis):  # type: ignore[no-untyped-def]
    """The residual under one linear stretch of takeaway-to-finish.

    Built by giving the target clip a clock whose interior knots are placed
    *proportionally* rather than at its own events, which is exactly what a
    uniform stretch does to it.
    """
    start = target_clock.knots[0].timestamp_s
    span = target_clock.knots[-1].timestamp_s - start
    reference_fractions = [
        (knot.timestamp_s - reference_clock.knots[0].timestamp_s)
        / (reference_clock.knots[-1].timestamp_s - reference_clock.knots[0].timestamp_s)
        for knot in reference_clock.knots
    ]

    stretched = target_clock.model_copy(deep=True)
    for knot, fraction in zip(stretched.knots, reference_fractions, strict=True):
        knot.timestamp_s = start + fraction * span

    left = normalise.resample(
        reference_clock, reference_signals.t, reference_signals.height, axis
    )
    right = normalise.resample(stretched, target_signals.t, target_signals.height, axis)
    return _residual(left.value, right.value)


def _residual(left: np.ndarray, right: np.ndarray) -> float:
    both = np.isfinite(left) & np.isfinite(right)
    if not np.any(both):
        return float("nan")
    return float(np.max(np.abs(left[both] - right[both])))


# --- camera ---------------------------------------------------------------


def _view_at(azimuth_deg: float, *, fps: float = 120.0, distance_m: float = 3.4) -> SyntheticView:
    """A camera `azimuth_deg` round the player from face-on, same height and lens."""
    # The fixture's own camera builder, so the sweep's lenses match the rig's.
    from tests.synthetic_body3d import _camera

    centre = np.array([0.0, 1.15, 0.0])
    angle = np.radians(azimuth_deg)
    position = np.array([-distance_m * np.sin(angle), 1.35, distance_m * np.cos(angle)])
    rotation, translation = look_at(position, centre)
    return SyntheticView(
        role=CameraRole.FACE_ON,
        intrinsics=_camera(1920, 1080, fx=1400.0, distortion=(0.0, 0.0, 0.0, 0.0)),
        rotation=rotation,
        translation=translation,
        fps=fps,
    )


def _from_camera(azimuth_deg: float, *, noise_px: float, seed: int, digest: str) -> SwingInput:
    """One unchanged swing, as recorded from a camera that many degrees round."""
    view = _view_at(azimuth_deg)
    sequence = pose_sequence_for(
        view, duration_s=DURATION_S, noise_px=noise_px, seed=seed, path=f"/data/{digest}.mov"
    )
    sequence.video_content_key.digest = digest * 64
    filtered = filter_sequence(sequence, FILTER, space=LandmarkSpace.FRAME_WIDTHS)
    detected = detect_phases(filtered, None)
    return SwingInput(
        path=f"/data/{digest}.mov",
        content_key=sequence.video_content_key,
        filtered=filtered,
        phases=detected,
        metrics=compute_metrics(filtered, detected),
    )


def sweep_camera(args: argparse.Namespace) -> None:
    """One unchanged swing, filmed from a series of azimuths.

    Nothing about the body changes between any two rows: the same `body_at`
    function is projected through cameras that differ only in where they stand.
    Every difference the table reports is the camera, and the point is that
    nothing in either recording says so.

    Several seeds per azimuth, because landmark noise moves the view estimate as
    well as the metrics -- and occasionally moves the takeaway onto frame zero,
    which leaves a clip with no address phase and therefore no view at all. Those
    seeds are counted rather than dropped quietly: it is a real failure mode and
    the amateur reference footage shows it too.
    """
    print(
        "One 3D swing, unchanged, filmed from cameras a few degrees apart. The body "
        f"is identical in every row; landmark scatter is {REAL_FOOTAGE_SIGMA_PX} px, "
        f"median of {args.repeats} seeds."
    )
    print("A difference here is a difference nobody made.\n")

    square = _from_camera(0.0, noise_px=REAL_FOOTAGE_SIGMA_PX, seed=11, digest="a")
    reference_turn = square.metrics.get(MetricName.SHOULDER_TURN, _top())

    print(
        f"{'azimuth':>8}  {'span':>6}  {'disagree':>9}  {'shoulder turn':>14}  "
        f"{'reported as':>16}  {'hand speed':>11}  {'usable':>7}"
    )
    for azimuth in (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0):
        spans, disagreements, turns, verdicts, curves = [], [], [], [], []
        for offset in range(args.repeats):
            moved = _from_camera(
                azimuth, noise_px=REAL_FOOTAGE_SIGMA_PX, seed=21 + offset, digest="b"
            )
            view = moved.metrics.view
            if view is None or not np.isfinite(view.shoulder_span_ratio):
                continue

            result = compare(square, moved)
            spans.append(view.shoulder_span_ratio)
            if result.camera.span_disagreement is not None:
                disagreements.append(result.camera.span_disagreement)

            turn = moved.metrics.get(MetricName.SHOULDER_TURN, _top())
            if turn is not None:
                turns.append(turn.value)

            difference = result.difference(MetricName.SHOULDER_TURN, _top())
            refusal = result.refusal(MetricName.SHOULDER_TURN, _top())
            verdicts.append(
                f"{difference.difference:+.1f} deg"
                if difference is not None
                else (refusal.refusal.value if refusal is not None else "-")
            )

            speed = result.trajectory(TrajectoryChannel.HAND_SPEED)
            if speed is not None:
                curves.append(
                    speed.refusal.value if speed.refusal is not None else speed.resolved_fraction
                )

        if not spans:
            print(f"{azimuth:>7.0f}  no seed at this azimuth produced an address phase")
            continue

        numeric = [value for value in curves if isinstance(value, float)]
        curve = (
            f"{float(np.median(numeric)) * 100:.0f}%"
            if numeric
            else (str(curves[0]) if curves else "-")
        )
        print(
            f"{azimuth:>7.0f}  {float(np.median(spans)):>6.2f}  "
            f"{_percent(float(np.median(disagreements)) if disagreements else None):>9}  "
            f"{(float(np.median(turns)) if turns else float('nan')):>13.1f}  "
            f"{_modal(verdicts):>16}  {curve:>11}  {len(spans):>4}/{args.repeats}"
        )

    baseline = reference_turn.value if reference_turn is not None else float("nan")
    print(
        f"\nThe swing's shoulder turn is {baseline:.1f} degrees seen square on. "
        "The column beside it is the same turn, unchanged, read from further round."
    )
    print(
        "The last two columns are the comparison's answer. Where it names a "
        "refusal, a gate caught the camera; where it prints a number, the engine "
        "is reporting a difference in a swing that did not change."
    )


def _modal(values: list[str]) -> str:
    """The commonest verdict, so one unlucky seed does not name the row."""
    if not values:
        return "-"
    return max(set(values), key=values.count)


def _top():  # type: ignore[no-untyped-def]
    from analyzer.contracts.phases import SwingEvent

    return SwingEvent.TOP


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


# --- resolution -----------------------------------------------------------


def sweep_resolution(_args: argparse.Namespace) -> None:
    """What frame rate a given difference in shape needs before a pair can see it.

    Two swings that reach the top by different routes -- same four instants, a
    different arc between them -- filmed at a series of rates. The bracket on a
    trajectory sample is each clip's event ambiguity carried onto the axis, so it
    falls with the frame rate and the fraction of the swing that resolves climbs.
    """
    shallow, steep = SwingShape(arc_top_angle=2.2), SwingShape(arc_top_angle=1.6)

    print("Two swings of different shape at one tempo, filmed at a series of rates.\n")
    print(f"{'fps':>6}  {'height':>8}  {'speed':>8}  {'largest':>9}  {'metrics':>8}")
    for fps in (30, 60, 120, 240, 480):
        window = 0.12 if fps >= 60 else 0.20
        config = FilterConfig(smoothing=SmoothingConfig(window_s=window))
        left = _at_rate(shallow, fps, config, "a")
        right = _at_rate(steep, fps, config, "b")
        if left is None or right is None:
            print(f"{fps:>6}  no swing detected at this rate and window")
            continue

        result = compare(left, right)
        height = result.trajectory(TrajectoryChannel.HAND_HEIGHT)
        speed = result.trajectory(TrajectoryChannel.HAND_SPEED)
        print(
            f"{fps:>6}  {_fraction(height):>8}  {_fraction(speed):>8}  "
            f"{(height.largest_difference if height and height.largest_difference else float('nan')):>9.3f}  "
            f"{len(result.differences):>8}"
        )

    print(
        "\nThe first two columns are the fraction of the sampled positions where the "
        "two curves differ by more than the pair can resolve. The difference itself "
        "does not change down the table -- the `largest` column is the same number "
        "at every rate, because it is a fact about the two swings. What changes is "
        "how much of it a pair of recordings can attribute to them."
    )
    print(
        "The window is set per row rather than left to the engine, so that the "
        "only thing varying down the table is the frame rate. The engine would now "
        "resolve one itself -- the 0.10 s default holds three samples at 30 fps "
        "where a degree-4 fit needs five, and `filter_sequence` widens to the "
        "narrowest width that fits rather than emitting nothing. Pinning it here "
        "keeps that resolution from becoming a second variable."
    )


def _at_rate(shape: SwingShape, fps: float, config: FilterConfig, digest: str) -> SwingInput | None:
    sequence = swing_sequence(
        duration_s=shape.finish_s + 0.4, fps=fps, shape=shape, body=FACE_ON_BODY
    )
    sequence.video_content_key.digest = digest * 64
    filtered = filter_sequence(sequence, config, space=LandmarkSpace.FRAME_WIDTHS)
    detected = detect_phases(filtered, None)
    if not detected.detected:
        return None
    return SwingInput(
        path=f"/data/{digest}.mov",
        content_key=sequence.video_content_key,
        filtered=filtered,
        phases=detected,
        metrics=compute_metrics(filtered, detected),
    )


def _fraction(channel) -> str:  # type: ignore[no-untyped-def]
    if channel is None or channel.refusal is not None:
        return "-"
    return f"{channel.resolved_fraction * 100:.0f}%"


# --- clips ----------------------------------------------------------------


def _load(clip: Clip) -> SwingInput | None:
    metadata = probe(REPO_ROOT / clip.path).metadata
    entry, _ = resolve_model(None)
    poses = cache_dir() / "poses" / metadata.content_key.as_path_segment() / f"{entry.name}.parquet"
    if not poses.exists():
        return None

    # The engine's own default where the clip can support it, because what this
    # sweep reports is what the app would show. The 30 fps clips name their own
    # window: Phase 3's floor means the default emits nothing at all there.
    config = (
        FilterConfig(smoothing=SmoothingConfig(window_s=clip.window_s))
        if clip.window_s
        else FilterConfig()
    )
    sequence = read_sequence(poses)
    filtered = filter_sequence(
        sequence,
        config,
        space=LandmarkSpace.FRAME_WIDTHS,
        slow_motion_factor=clip.slow_motion,
    )
    detected = detect_phases(filtered)
    return SwingInput(
        path=clip.path,
        content_key=sequence.video_content_key,
        filtered=filtered,
        phases=detected,
        metrics=compute_metrics(filtered, detected),
    )


def sweep_clips(_args: argparse.Namespace) -> None:
    """Every pair this repository can actually build."""
    print(f"{'pair':<46}  {'views':<26}  {'cameras':<9}  {'diff':>4}  {'refused':>7}  curves")
    for reference, target in PAIRS:
        left, right = _load(reference), _load(target)
        label = f"{Path(reference.path).name} / {Path(target.path).name}"
        if left is None or right is None:
            print(f"{label:<46}  no extraction -- run `analyzer extract` on both first")
            continue

        result = compare(left, right)
        views = f"{result.reference.view.value} / {result.target.view.value}"
        cameras = "same" if result.camera.consistent else "moved"
        curves = (
            "drawn"
            if result.trajectories and result.trajectories[0].refusal is None
            else (result.trajectories[0].refusal.value if result.trajectories else "none")
        )
        print(
            f"{label:<46}  {views:<26}  {cameras:<9}  "
            f"{len(result.differences):>4}  {len(result.refused):>7}  {curves}"
        )
        if result.differences:
            for entry in result.differences:
                print(f"{'':<46}    {entry.label}: {entry.difference:+.3g} {entry.unit.value}")
        counts = Counter(entry.refusal.value for entry in result.refused)
        if counts:
            print(
                f"{'':<46}    refusals: "
                + ", ".join(f"{name} x{count}" for name, count in counts.most_common())
            )
        if not result.computed:
            print(f"{'':<46}    no comparison: {result.warnings[0][:100]}")

    print(
        "\nTwo of the three pairs are a face-on camera against a down-the-line one, "
        "so every projected quantity is refused and only the timings survive. The "
        "third is the interesting one: two swings by one player from one position, "
        "and it refuses too -- the second clip begins at the takeaway, a clip with "
        "no address phase has no measured view, and a comparison that cannot say "
        "where either camera stood will not compare a projection."
    )
    print(
        "That is a capture instruction rather than a defect. Start recording before "
        "the player is set up to the ball: the address phase is where the view, the "
        "rotation baseline and the hand-path origin all come from, and a clip "
        "without one costs every projected comparison in this table."
    )


# --- cost -----------------------------------------------------------------


def sweep_cost(args: argparse.Namespace) -> None:
    """What a comparison costs, against what is already spent to reach it."""
    shallow = flat_swing(SwingShape(arc_top_angle=2.2))
    steep = flat_swing(SwingShape(arc_top_angle=1.6), digest="b")

    print(f"{'samples':>8}  {'compare':>9}  {'serialise':>10}  {'payload':>9}")
    for samples in (61, 121, 241, 481):
        config = ComparisonConfig(samples=samples)
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = compare(shallow, steep, config)
            timings.append(time.perf_counter() - start)

        start = time.perf_counter()
        encoded = json.dumps(result.model_dump(mode="json"))
        serialise = time.perf_counter() - start

        print(
            f"{samples:>8}  {np.median(timings) * 1000:>8.1f} ms  "
            f"{serialise * 1000:>9.1f} ms  {len(encoded) / 1024:>7.0f} KB"
        )

    print(
        "\nAgainst roughly 1.3 s per clip to extract poses, and both clips have to "
        "be extracted before any of this runs. Nothing is cached, for the reason "
        "Phase 3 measured and every phase since has re-checked."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="Timing repeats.")
    parser.add_argument(
        "--sweep",
        choices=("warp", "camera", "resolution", "clips", "cost", "all"),
        default="all",
    )
    arguments = parser.parse_args()

    sweeps = {
        "warp": sweep_warp,
        "camera": sweep_camera,
        "resolution": sweep_resolution,
        "clips": sweep_clips,
        "cost": sweep_cost,
    }
    chosen = [arguments.sweep] if arguments.sweep != "all" else list(sweeps)
    for name in chosen:
        print(f"\n=== {name} ===\n")
        sweeps[name](arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
