#!/usr/bin/env python3
"""Measure what club tracking is worth, against a shaft whose angle is an input.

**Ground truth in the same sense as Phases 8 and 9, and no stronger.** The shaft
is drawn by `tests/synthetic_club.py` at an angle that is a number put in there,
so an angular error is a subtraction rather than a judgement. What the fixture
does not contain is a photograph: no defocus, no rolling shutter, no compression,
no flat background that is actually a driving range, and a club 1.1 torso lengths
long where a real one is about 2.5 -- a longer lever smears proportionally more.
**Every rate below is a floor.**

The questions these sweeps exist to answer are not "how accurate is the angle".
That turns out to be easy: half a degree, everywhere the club is found at all.
They are about *when it is found*, and whether anything the tracker reports about
itself can tell you when it has gone wrong:

    blur        how much motion smear does a Hough-based detector survive?
    shutter     what does that mean for a capture, per phase of the swing?
    clutter     does the temporal check beat picking the strongest line?
    occlusion   what happens when the body hides part of the shaft?
    frame_rate  what do 30, 60, 120 and 240 fps each buy?
    cost        how long does any of this take?

`shutter` is the one that matters, and what it measures is the gap between a
clip's overall detection rate and its **downswing** detection rate. They differ
by a factor of two, and the first is the number a detector would naturally print.

Usage:
    uv run --project python python scripts/benchmark_club.py
    uv run --project python python scripts/benchmark_club.py --sweep shutter
    uv run --project python python scripts/benchmark_club.py --repeats 3
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

from analyzer.club import HoughShaftDetector, frame_evidence, track_shafts
from analyzer.club.track import FrameEvidence, TrackResult
from analyzer.contracts.club import ClubConfig
from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.phases import (
    PhaseConfig,
    SwingEvent,
    SwingPhase,
    SwingPhases,
)
from analyzer.contracts.pose import LandmarkSpace
from analyzer.filtering.landmarks import filter_sequence
from analyzer.phases import detect_phases
from tests import synthetic_club as fixture

# An angular error past this is not a noisy measurement of the club, it is a
# measurement of something else -- a door frame, or the club read backwards. The
# median error on a found shaft is half a degree, so there is no grey zone here
# and the count is insensitive to where exactly the line is drawn.
WRONG_DEG = 5.0

# Columns of vertical background lines used by the clutter sweep, in pixels, in
# the order they are added. Placed where the hands pass during the swing -- the
# grip spans x = 280 to 720 -- because a background line the hands never come
# near is rejected on distance before anything else is asked of it.
CLUTTER_COLUMNS = (620.0, 400.0, 520.0, 340.0)


@dataclass(frozen=True)
class Outcome:
    """One tracked swing, scored against the angles it was rendered from."""

    frames: int
    tracked: int
    coverage: float
    downswing_coverage: float
    median_error_deg: float
    p95_error_deg: float
    wrong_frames: int
    wrong_confidence: float
    max_blur_px: float
    head_fraction: float
    impact_frame: int | None
    kinematic_frame: int | None


def _errors(rendered: list[fixture.RenderedFrame], result: TrackResult) -> list[float]:
    """Absolute angular error at every tracked frame, in degrees.

    Wrapped onto (-180, 180] before taking the magnitude, so a shaft reported
    backwards scores 180 rather than 0 -- which is the failure this phase spends
    the most effort on and would be invisible to an unwrapped subtraction.
    """
    errors: list[float] = []
    for truth, frame in zip(rendered, result.frames, strict=True):
        if frame.shaft is None:
            continue
        delta = (frame.shaft.angle_deg - truth.true_angle_fw + 180.0) % 360.0 - 180.0
        errors.append(abs(delta))
    return errors


def _score(
    rendered: list[fixture.RenderedFrame], phases: SwingPhases, result: TrackResult
) -> Outcome:
    errors = _errors(rendered, result)
    wrong = [
        frame.shaft.confidence.overall
        for truth, frame in zip(rendered, result.frames, strict=True)
        if frame.shaft is not None
        and abs((frame.shaft.angle_deg - truth.true_angle_fw + 180.0) % 360.0 - 180.0)
        > WRONG_DEG
    ]
    downswing = next(
        (
            entry
            for entry in result.phase_coverage
            if entry.phase is SwingPhase.DOWNSWING
        ),
        None,
    )
    kinematic = phases.event(SwingEvent.IMPACT)
    return Outcome(
        frames=len(rendered),
        tracked=result.tracked_frames,
        coverage=result.tracked_frames / len(rendered) if rendered else 0.0,
        downswing_coverage=downswing.coverage if downswing else float("nan"),
        median_error_deg=float(np.median(errors)) if errors else float("nan"),
        p95_error_deg=float(np.percentile(errors, 95.0)) if errors else float("nan"),
        wrong_frames=len(wrong),
        wrong_confidence=float(np.median(wrong)) if wrong else float("nan"),
        max_blur_px=max(frame.blur_px for frame in rendered)
        if rendered
        else float("nan"),
        head_fraction=result.quality.head_fraction if result.quality else float("nan"),
        impact_frame=result.impact.frame_index if result.impact else None,
        kinematic_frame=kinematic.frame_index if kinematic else None,
    )


def _phases_for(fps: float) -> SwingPhases:
    """Detect the swing from the pose sequence that shares the clip's clock.

    Recomputed per frame rate rather than cached, because the phase boundaries
    are in frame indices and a different frame rate is a different indexing of
    the same instants.
    """
    _, poses = fixture.swing_with_poses(fps=fps)
    filtered = filter_sequence(poses, FilterConfig(), space=LandmarkSpace.FRAME_WIDTHS)
    return detect_phases(filtered, PhaseConfig())


def _evidence(
    rendered: list[fixture.RenderedFrame], detector: HoughShaftDetector
) -> list[FrameEvidence]:
    return [
        frame_evidence(
            item.frame,
            item.anchor,
            detector,
            geometry=fixture.FRAME,
            torso_length=fixture.TORSO_LENGTH_FW,
            timestamp_s=item.frame.timestamp_s,
        )
        for item in rendered
    ]


def _run(
    *,
    fps: float = 120.0,
    shutter_fraction: float = 0.25,
    clutter: int = 0,
    occlusion: tuple[int, int, int, int] | None = None,
    seed: int = 0,
    config: ClubConfig | None = None,
) -> tuple[list[fixture.RenderedFrame], list[FrameEvidence], SwingPhases, TrackResult]:
    rendered = fixture.swing(
        fps=fps,
        shutter_fraction=shutter_fraction,
        clutter_x=CLUTTER_COLUMNS[:clutter],
        occlusion=occlusion,
        seed=seed,
    )
    detector = HoughShaftDetector(config)
    evidence = _evidence(rendered, detector)
    phases = _phases_for(fps)
    result = track_shafts(
        evidence, config or ClubConfig(), phases=phases, frame_interval_s=1.0 / fps
    )
    return rendered, evidence, phases, result


def _heading(title: str, header: str) -> None:
    print(f"\n{title}")
    print("-" * max(len(title), len(header)))
    print(header)


def _pct(value: float) -> str:
    return "     -" if not np.isfinite(value) else f"{value:5.0%} "


def sweep_blur(repeats: int) -> None:
    """Detection against the length of the smear, one frame at a time.

    The purest form of the measurement: one rendered frame, one shaft, no
    tracker, no swing. The only variable is how far the club head travelled
    during the exposure, which is the quantity a shutter speed buys down.
    """
    _heading(
        "Blur -- a single frame, club at 45 degrees",
        " smear at head | found | support | angle error",
    )
    detector = HoughShaftDetector()
    for sweep_deg in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0):
        found = 0
        supports: list[float] = []
        errors: list[float] = []
        blur = 0.0
        for seed in range(repeats):
            item = fixture.render(angle_deg=45.0, sweep_deg=sweep_deg, seed=seed * 17)
            blur = item.blur_px
            result = detector.detect(item.frame, item.anchor)
            if not result.candidates:
                continue
            best = result.candidates[0]
            found += 1
            supports.append(best.support)
            errors.append(abs(best.angle_deg - item.angle_deg))
        rate = found / repeats
        support = f"{np.median(supports):7.2f}" if supports else "      -"
        error = f"{np.median(errors):8.2f} deg" if errors else "        -"
        print(f" {blur:9.1f} px  | {_pct(rate)}| {support} | {error}")
    print(
        "  The cliff is between 9 and 14 px and it is a cliff, not a decline: a smeared\n"
        "  shaft does not become a weaker line, it stops being a line. Multiply a clip's\n"
        "  club-head speed in px/s by its exposure to see which side of it the capture is."
    )


def sweep_shutter(repeats: int) -> None:
    """Coverage per swing phase against the shutter. **The headline.**"""
    _heading(
        "Shutter -- 120 fps, and what the aggregate rate hides",
        " shutter | max smear | overall | address | backswing | DOWNSWING | follow | err p95",
    )
    for shutter in (0.03, 0.125, 0.25, 0.5, 1.0):
        rows: list[tuple[float, dict[SwingPhase, float], float, float]] = []
        for seed in range(repeats):
            rendered, _, phases, result = _run(shutter_fraction=shutter, seed=seed * 31)
            outcome = _score(rendered, phases, result)
            coverage = {entry.phase: entry.coverage for entry in result.phase_coverage}
            rows.append(
                (outcome.coverage, coverage, outcome.p95_error_deg, outcome.max_blur_px)
            )

        overall = float(np.median([row[0] for row in rows]))
        blur = float(np.median([row[3] for row in rows]))
        p95 = float(np.median([row[2] for row in rows]))

        def phase(which: SwingPhase) -> float:
            values = [row[1].get(which, float("nan")) for row in rows]
            finite = [value for value in values if np.isfinite(value)]
            return float(np.median(finite)) if finite else float("nan")

        print(
            f" {shutter:7.3f} | {blur:6.1f} px | {_pct(overall)}| {_pct(phase(SwingPhase.ADDRESS))}|"
            f" {_pct(phase(SwingPhase.BACKSWING))}  | {_pct(phase(SwingPhase.DOWNSWING))}  |"
            f" {_pct(phase(SwingPhase.FOLLOW_THROUGH))}| {p95:6.2f} deg"
        )
    print(
        "  Read the last two columns against the third. The overall rate is dominated by\n"
        "  address and the follow-through, where the club is nearly still; the downswing is\n"
        "  where every metric worth computing lives and it is the column that collapses."
    )


def sweep_clutter(repeats: int) -> None:
    """The temporal check against picking the strongest line.

    The comparison this phase's design rests on. `evidence-only` is what a
    per-frame detector would report: the candidate with the most edge support,
    which is what a Hough transform ranks by and what a confidence built on
    support alone would be confident about.
    """
    _heading(
        "Clutter -- vertical background lines through the hands, 120 fps, 1/4 shutter",
        " lines | tracked | wrong | conf. when wrong || evidence-only: kept | wrong",
    )
    config = ClubConfig()
    for count in range(len(CLUTTER_COLUMNS) + 1):
        tracked: list[int] = []
        wrong: list[int] = []
        confidences: list[float] = []
        baseline_kept: list[int] = []
        baseline_wrong: list[int] = []

        for seed in range(repeats):
            rendered, evidence, phases, result = _run(clutter=count, seed=seed * 13)
            outcome = _score(rendered, phases, result)
            tracked.append(outcome.tracked)
            wrong.append(outcome.wrong_frames)
            if np.isfinite(outcome.wrong_confidence):
                confidences.append(outcome.wrong_confidence)

            kept, missed = _evidence_only(rendered, evidence, config)
            baseline_kept.append(kept)
            baseline_wrong.append(missed)

        confidence = f"{np.median(confidences):6.2f}" if confidences else "     -"
        print(
            f" {count:5d} | {int(np.median(tracked)):7d} | {int(np.median(wrong)):5d} |"
            f" {confidence}           || {int(np.median(baseline_kept)):19d} |"
            f" {int(np.median(baseline_wrong)):5d}"
        )
    print(
        "  The tracker refuses frames the baseline accepts, and the frames it refuses are\n"
        "  the wrong ones. A stationary line scores almost nothing on continuity however\n"
        "  good its edges are, which is the whole of the difference between the two halves\n"
        "  of this table."
    )


def _evidence_only(
    rendered: list[fixture.RenderedFrame],
    evidence: list[FrameEvidence],
    config: ClubConfig,
) -> tuple[int, int]:
    """What a per-frame detector alone would report: the best-supported line.

    No prediction, no margin over a rival, no run length -- the candidate with
    the most edge evidence behind it, subject only to the support floor. This is
    not a mode the engine offers; it is the thing the tracker is compared against,
    and it is what a classical shaft detector without a temporal stage *is*.
    """
    kept = 0
    wrong = 0
    for truth, frame in zip(rendered, evidence, strict=True):
        if not frame.candidates:
            continue
        best = max(
            frame.candidates,
            key=lambda candidate: (candidate.support, candidate.length),
        )
        if best.support < config.min_support:
            continue
        kept += 1
        delta = (best.angle_deg - truth.true_angle_fw + 180.0) % 360.0 - 180.0
        if abs(delta) > WRONG_DEG:
            wrong += 1
    return kept, wrong


def sweep_occlusion(repeats: int) -> None:
    """What the player's own body costs, as a block over part of the shaft."""
    _heading(
        "Occlusion -- an opaque block over the club, 120 fps, 1/4 shutter",
        " covers      | tracked | downswing | wrong | conf. when wrong | head seen",
    )
    height = fixture.FRAME.height
    boxes: tuple[tuple[str, tuple[int, int, int, int] | None], ...] = (
        ("nothing", None),
        ("upper third", (0, 0, fixture.FRAME.width, height // 3)),
        ("upper half", (0, 0, fixture.FRAME.width, height // 2)),
        ("a wide band", (0, height // 4, fixture.FRAME.width, 3 * height // 4)),
    )
    for label, box in boxes:
        outcomes = []
        for seed in range(repeats):
            rendered, _, phases, result = _run(occlusion=box, seed=seed * 7)
            outcomes.append(_score(rendered, phases, result))
        confidences = [o.wrong_confidence for o in outcomes if np.isfinite(o.wrong_confidence)]
        confidence = f"{np.median(confidences):6.2f}" if confidences else "     -"
        print(
            f" {label:11s} | {_pct(float(np.median([o.coverage for o in outcomes])))}|"
            f" {_pct(float(np.median([o.downswing_coverage for o in outcomes])))}  |"
            f" {int(np.median([o.wrong_frames for o in outcomes])):5d} |"
            f" {confidence}           |"
            f" {_pct(float(np.median([o.head_fraction for o in outcomes])))}"
        )
    print(
        "  An occluded shaft is mostly refused rather than guessed at, which is what the\n"
        "  gaps in the coverage are, and the club head is the first thing to go -- what\n"
        "  survives an occlusion is a direction without a length.\n"
        "\n"
        "  **The `upper third` row is the failure this phase cannot detect.** A rectangular\n"
        "  occluder has a long straight boundary, and where that boundary runs near the\n"
        "  hands it is a better shaft than the shaft: sharp, stationary, and unopposed once\n"
        "  the real club is hidden behind it. The tracker steps onto it and continuity keeps\n"
        "  it there, because a stationary line agrees perfectly with a prediction extrapolated\n"
        "  from two frames already on it. The confidence column is the point: those frames\n"
        "  score the same as the correct ones. Nothing here notices."
    )


def sweep_frame_rate(repeats: int) -> None:
    """What a frame rate buys, at a fixed exposure fraction and at a fixed exposure.

    Two different questions, and only the second is the one a buyer is asking.
    Holding the *fraction* fixed makes a faster camera a shorter exposure for
    free, which flatters it; holding the exposure fixed at 1/500 s isolates what
    the extra frames alone are worth.
    """
    _heading(
        "Frame rate -- 1/4 shutter, then a fixed 1/500 s exposure",
        " fps | 1/4 shutter: smear  overall  DOWNSWING || 1/500 s: smear  overall  DOWNSWING",
    )
    for fps in (30.0, 60.0, 120.0, 240.0):
        fixed_fraction = (1.0 / 500.0) * fps
        row = []
        for shutter in (0.25, min(1.0, fixed_fraction)):
            outcomes = []
            for seed in range(repeats):
                rendered, _, phases, result = _run(
                    fps=fps, shutter_fraction=shutter, seed=seed * 11
                )
                outcomes.append(_score(rendered, phases, result))
            row.append(
                (
                    float(np.median([o.max_blur_px for o in outcomes])),
                    float(np.median([o.coverage for o in outcomes])),
                    float(np.median([o.downswing_coverage for o in outcomes])),
                )
            )
        print(
            f" {fps:3.0f} | {row[0][0]:17.1f} px {_pct(row[0][1])} {_pct(row[0][2])} ||"
            f" {row[1][0]:12.1f} px {_pct(row[1][1])} {_pct(row[1][2])}"
        )
    print(
        "  A 30 fps clip cannot support the filter's default window at all, so its hand\n"
        "  anchors are refused before the club is ever searched for -- which is Phase 3's\n"
        "  floor reappearing here rather than anything about the club."
    )


def sweep_impact(repeats: int) -> None:
    """Where the club head puts impact, against where the hands put it."""
    _heading(
        "Impact -- club head against Phase 4's hand-speed peak",
        " shutter | downswing | club head | kinematic | delta",
    )
    for shutter in (0.03, 0.125, 0.25, 0.5, 1.0):
        rows = []
        for seed in range(repeats):
            rendered, _, phases, result = _run(shutter_fraction=shutter, seed=seed * 23)
            rows.append(_score(rendered, phases, result))
        offered = [row for row in rows if row.impact_frame is not None]
        coverage = float(np.median([row.downswing_coverage for row in rows]))
        if not offered:
            print(
                f" {shutter:7.3f} | {_pct(coverage)}  |   refused |         - |     -"
            )
            continue
        club = int(np.median([row.impact_frame for row in offered]))
        kinematic = offered[0].kinematic_frame
        delta = "-" if kinematic is None else f"{club - kinematic:+d}"
        print(
            f" {shutter:7.3f} | {_pct(coverage)}  | {club:9d} |"
            f" {'-' if kinematic is None else kinematic:>9} | {delta:>5} frames"
        )
    print(
        "  Refused wherever the downswing was not tracked, which is most of the table. The\n"
        "  one row that answers agrees with Phase 4 to a frame -- and that is a check of the\n"
        "  arithmetic, not of the physics: this fixture puts the hands' peak speed and the\n"
        "  club head's lowest point at the same instant by construction. On real footage the\n"
        "  hands peak first, which is the whole reason the two estimates are worth having."
    )


def measure_cost(repeats: int) -> None:
    """Per-frame cost, against the pose extraction that has to happen first."""
    rendered = fixture.swing(fps=120.0, shutter_fraction=0.25)
    detector = HoughShaftDetector()
    phases = _phases_for(120.0)

    detect_times: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        evidence = _evidence(rendered, detector)
        detect_times.append(time.perf_counter() - started)

    started = time.perf_counter()
    track_shafts(evidence, ClubConfig(), phases=phases, frame_interval_s=1.0 / 120.0)
    tracking = time.perf_counter() - started

    frames = len(rendered)
    per_frame = float(np.median(detect_times)) * 1000.0 / frames
    _heading("Cost", " stage                     | median   | note")
    print(
        f" detect ({frames} frames)        | {per_frame:6.2f} ms | per frame, 1000x1000 search region"
    )
    print(
        f" track                     | {tracking * 1000:6.2f} ms | whole clip, on candidates alone"
    )
    print(
        "  Against ~17 ms a frame to extract poses, which has to have happened first. The\n"
        "  search region is what costs: it is a disc of 3.2 torso lengths, which on this\n"
        "  fixture is the whole frame."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Seeds per measurement.")
    parser.add_argument(
        "--sweep",
        choices=(
            "blur",
            "shutter",
            "clutter",
            "occlusion",
            "frame_rate",
            "impact",
            "cost",
            "all",
        ),
        default="all",
    )
    arguments = parser.parse_args()

    print(
        f"Synthetic shaft: {fixture.SHAFT_TORSO:.1f} torso lengths on a "
        f"{fixture.FRAME.width}x{fixture.FRAME.height} frame, torso {fixture.TORSO_PX:.0f} px, "
        f"peak turn {fixture.peak_rate_deg_s():.0f} deg/s"
    )
    print(
        "Not a photograph: no defocus, no compression, a flat background, and a club a "
        "little\nover half the length of a real one relative to the body. A floor, not an "
        "estimate."
    )

    chosen = arguments.sweep
    if chosen in ("blur", "all"):
        sweep_blur(arguments.repeats)
    if chosen in ("shutter", "all"):
        sweep_shutter(arguments.repeats)
    if chosen in ("clutter", "all"):
        sweep_clutter(arguments.repeats)
    if chosen in ("occlusion", "all"):
        sweep_occlusion(arguments.repeats)
    if chosen in ("frame_rate", "all"):
        sweep_frame_rate(arguments.repeats)
    if chosen in ("impact", "all"):
        sweep_impact(arguments.repeats)
    if chosen in ("cost", "all"):
        measure_cost(arguments.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
