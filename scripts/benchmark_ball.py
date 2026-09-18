#!/usr/bin/env python3
"""Measure what ball detection is worth, against a ball that vanishes at a known frame.

**Ground truth in the same sense as Phases 8, 9 and 10, and no stronger.** The
ball is drawn by `tests/synthetic_ball.py` in every frame before a departure that
is a number put in there and in none after, so an error in the located instant is
a subtraction of integers rather than a judgement. What the fixture does not
contain is a photograph: no defocus, no rolling shutter, no compression, no grass
that is a thousand individually bright blades, and a ball drawn as a uniform disc
where a real one is a lit sphere. **Every rate below is a floor.**

The question these sweeps exist to answer is not "how often is the ball found".
That turns out to be easy on a clip where it is in plain view, and it is also the
wrong question: this phase reports an instant, and a detector that finds the ball
in 99% of frames and misses the last one before impact is wrong by a frame while
reporting 99%.

    departure   is the located instant the one the ball actually went at?
    frame_rate  what does the bracket cost at 30, 60, 120 and 240 fps?
    contrast    how far can the ball fade towards its background?
    clutter     does a second stationary object take the identification?
    occlusion   does a ball being covered look like a ball being struck?
    truncation  does a clip that ends too soon say so?
    agreement   how far do the four impact estimates sit from each other?
    cost        how long does any of this take?

`agreement` is the one Phase 11 exists for, and it carries a caveat large enough
that the sweep prints it: this fixture puts the hands' peak speed, the hands'
lowest point, the club head's arrival and the ball's departure at **one instant
by construction**, because all four are driven by the same arc. It can therefore
show that the four methods agree when they should, and it cannot measure the bias
between them. Only footage can, and `docs/ROADMAP.md` records what the one clip
in this project with a visible impact said.

Usage:
    uv run --project python python scripts/benchmark_ball.py
    uv run --project python python scripts/benchmark_ball.py --sweep agreement
    uv run --project python python scripts/benchmark_ball.py --repeats 3
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

from analyzer.ball import ContrastBlobDetector, track_ball
from analyzer.ball.detector import BallAnchor
from analyzer.ball.extract import frame_evidence
from analyzer.ball.track import BallEvidence, BallTrackResult
from analyzer.club import HoughShaftDetector, track_shafts
from analyzer.club.extract import frame_evidence as club_frame_evidence
from analyzer.contracts.ball import BallConfig
from analyzer.contracts.club import ClubConfig
from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.impact import ImpactSource
from analyzer.contracts.phases import PhaseConfig, SwingPhases
from analyzer.contracts.pose import LandmarkSpace
from analyzer.filtering.landmarks import filter_sequence
from analyzer.impact import fuse_impact
from analyzer.phases import detect_phases
from tests import synthetic_ball as fixture
from tests import synthetic_club as club

# Positions for stationary distractors, in pixels: ball-sized objects that do not
# move. A tee marker, a second range ball, a white shoe. Placed near the ball but
# outside its drift bound, so they compete for the identification rather than for
# any single frame's choice -- which is the competition that matters, and the one
# `EstablishedBall.margin` is the only check on.
_DISTRACTORS = (
    (fixture.BALL_PX[0] + 70.0, fixture.BALL_PX[1] + 4.0),
    (fixture.BALL_PX[0] - 95.0, fixture.BALL_PX[1] - 6.0),
    (fixture.BALL_PX[0] + 150.0, fixture.BALL_PX[1] + 10.0),
)


@dataclass(frozen=True)
class Outcome:
    """One detected clip, scored against the frame the ball was removed at."""

    truth_frame: int
    located_frame: int | None
    error_frames: int | None
    interval_ms: float | None
    coverage: float
    established_margin: float | None
    confidence: float | None
    establishment: float | None
    abruptness: float | None
    permanence: float | None


def _phases_for(fps: float, poses: object) -> SwingPhases:
    filtered = filter_sequence(poses, FilterConfig(), space=LandmarkSpace.FRAME_WIDTHS)  # type: ignore[arg-type]
    return detect_phases(filtered, PhaseConfig())


def _anchor() -> BallAnchor:
    """The address hand position, which is where `extract.py` puts the region.

    Fixed for the clip rather than per frame: the ball does not move, so a region
    that followed the hands would sweep away from it exactly through the downswing.
    """
    return BallAnchor(
        x=club.ADDRESS_GRIP_PX[0], y=club.ADDRESS_GRIP_PX[1], torso_px=club.TORSO_PX
    )


def _evidence(
    rendered: list[club.RenderedFrame], detector: ContrastBlobDetector
) -> list[BallEvidence]:
    anchor = _anchor()
    evidence: list[BallEvidence] = []
    for item in rendered:
        detection = detector.detect(item.frame, anchor)
        evidence.append(
            frame_evidence(
                item.frame.index,
                item.frame.timestamp_s,
                detection.candidates,
                geometry=club.FRAME,
                torso_length=club.TORSO_LENGTH_FW,
                refusal=detection.refusal,
            )
        )
    return evidence


def _run(
    *,
    fps: float = 120.0,
    shutter_fraction: float = 0.25,
    ball_grey: float = fixture.BALL_GREY,
    fade_frames: int = 0,
    distractors: int = 0,
    distractor_departure: int | None = None,
    duration_s: float | None = None,
    seed: int = 0,
    config: BallConfig | None = None,
) -> tuple[list[club.RenderedFrame], SwingPhases, BallTrackResult, fixture.BallTruth]:
    kwargs: dict[str, object] = {
        "fps": fps,
        "shutter_fraction": shutter_fraction,
        "ball_grey": ball_grey,
        "fade_frames": fade_frames,
        "distractors": _DISTRACTORS[:distractors],
        "distractor_departure": distractor_departure,
        "seed": seed,
    }
    if duration_s is not None:
        kwargs["duration_s"] = duration_s

    rendered, poses, truth = fixture.swing_with_ball(**kwargs)  # type: ignore[arg-type]
    resolved = config or BallConfig()
    evidence = _evidence(rendered, ContrastBlobDetector(resolved))
    phases = _phases_for(fps, poses)
    result = track_ball(
        evidence, resolved, phases=phases, torso_length=club.TORSO_LENGTH_FW
    )
    return rendered, phases, result, truth


def _score(result: BallTrackResult, truth: fixture.BallTruth) -> Outcome:
    departure = result.departure
    established = result.established
    return Outcome(
        truth_frame=truth.departure_frame,
        located_frame=departure.frame_index if departure else None,
        error_frames=(departure.frame_index - truth.departure_frame) if departure else None,
        interval_ms=departure.interval_s * 1000.0 if departure else None,
        coverage=result.pre_departure_coverage,
        established_margin=established.margin if established else None,
        confidence=departure.confidence.overall if departure else None,
        establishment=departure.confidence.establishment if departure else None,
        abruptness=departure.confidence.abruptness if departure else None,
        permanence=departure.confidence.permanence if departure else None,
    )


def _heading(title: str, header: str) -> None:
    print(f"\n{title}")
    print("-" * max(len(title), len(header)))
    print(header)


def _pct(value: float | None) -> str:
    return "     -" if value is None or not np.isfinite(value) else f"{value:5.0%} "


def _num(value: float | None, width: int = 5, places: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "-".rjust(width)
    return f"{value:{width}.{places}f}"


def _frames(value: int | None) -> str:
    return "refused" if value is None else f"{value:+d}"


def sweep_departure(repeats: int) -> None:
    """The located instant against the frame the ball was removed at.

    The headline measurement of this phase, and the only one whose right answer
    is an integer.
    """
    _heading(
        "Departure -- located instant against the rendered one",
        " shutter | truth | located | error | bracket  | coverage | confidence",
    )
    for shutter in (0.03, 0.125, 0.25, 0.5, 1.0):
        outcomes = [
            _score(*_run(shutter_fraction=shutter, seed=seed * 17)[2:])
            for seed in range(max(1, repeats))
        ]
        best = outcomes[0]
        errors = [item.error_frames for item in outcomes if item.error_frames is not None]
        print(
            f" {shutter:7.3f} | {best.truth_frame:5d} | "
            f"{'refused' if best.located_frame is None else f'{best.located_frame:7d}'} | "
            f"{_frames(int(np.median(errors)) if errors else None):>5} | "
            f"{_num(best.interval_ms, 5, 1)} ms | {_pct(best.coverage)}| "
            f"{_num(best.confidence)}"
        )
    print(
        "  The shutter changes nothing here, and that is the result. A motion-blurred club\n"
        "  is what Phase 10 measured a cliff against; a ball at rest is not moving, so no\n"
        "  exposure smears it. What the shutter would reach is the frame before impact, in\n"
        "  which the club head arrives -- and that frame is the last one the ball is drawn in."
    )


def sweep_frame_rate(repeats: int) -> None:
    """What the frame rate buys, which for this phase is the bracket and nothing else."""
    _heading(
        "Frame rate -- the bracket is one frame interval, always",
        "  fps | truth | located | error | bracket  | Phase 4 window | ratio",
    )
    for fps in (30.0, 60.0, 120.0, 240.0):
        rendered, phases, result, truth = _run(fps=fps, seed=0)
        outcome = _score(result, truth)
        filtered_window = float("nan")
        poses = fixture.swing_with_ball(fps=fps)[1]
        filtered = filter_sequence(poses, FilterConfig(), space=LandmarkSpace.FRAME_WIDTHS)
        filtered_window = filtered.report.config.smoothing.window_s * 1000.0
        ratio = (
            filtered_window / outcome.interval_ms
            if outcome.interval_ms and outcome.interval_ms > 0
            else float("nan")
        )
        print(
            f" {fps:4.0f} | {outcome.truth_frame:5d} | "
            f"{'refused' if outcome.located_frame is None else f'{outcome.located_frame:7d}'} | "
            f"{_frames(outcome.error_frames):>5} | {_num(outcome.interval_ms, 5, 1)} ms | "
            f"{_num(filtered_window, 10, 1)} ms | {_num(ratio, 5, 0)}x"
        )
    print(
        "  The last column is what this phase is for. Phase 4's impact is a peak located in\n"
        "  a signal smoothed over that window, so the window is the scale on which it could\n"
        "  have moved; the ball's bracket is an interval impact is inside. The two are not\n"
        "  the same kind of claim, and the ratio is how much narrower the bracket is anyway."
    )


def sweep_contrast(repeats: int) -> None:
    """How far the ball can fade towards the turf it sits on before it is lost."""
    _heading(
        "Contrast -- ball intensity driven towards the turf below it",
        " ball grey | step | located | error | coverage | est. margin | confidence",
    )
    for grey in (250.0, 200.0, 170.0, 150.0, 130.0, 115.0, 105.0):
        outcome = _score(*_run(ball_grey=grey, seed=0)[2:])
        print(
            f" {grey:9.0f} | {grey - fixture.GROUND_GREY:4.0f} | "
            f"{'refused' if outcome.located_frame is None else f'{outcome.located_frame:7d}'} | "
            f"{_frames(outcome.error_frames):>5} | {_pct(outcome.coverage)}| "
            f"{_num(outcome.established_margin, 11)} | {_num(outcome.confidence)}"
        )
    print(
        f"  The turf is drawn at {fixture.GROUND_GREY}, so the step is the ball's contrast\n"
        "  against what it sits on. This is the capture variable for this phase, the way the\n"
        "  shutter is for Phase 10, and `data/README.md` asks for a contrasting surface for\n"
        "  this reason rather than as a matter of taste."
    )


def sweep_clutter(repeats: int) -> None:
    """Whether a second stationary ball-shaped object takes the identification."""
    _heading(
        "Clutter -- stationary rivals that do not depart",
        " distractors | located | error | est. margin | rival at | confidence",
    )
    for count in range(4):
        rendered, phases, result, truth = _run(distractors=count, seed=0)
        outcome = _score(result, truth)
        established = result.established
        rival = (
            "-"
            if established is None or established.runner_up_distance_torso is None
            else f"{established.runner_up_distance_torso:.2f} torso"
        )
        print(
            f" {count:11d} | "
            f"{'refused' if outcome.located_frame is None else f'{outcome.located_frame:7d}'} | "
            f"{_frames(outcome.error_frames):>5} | {_num(outcome.established_margin, 11)} | "
            f"{rival:>10} | {_num(outcome.confidence)}"
        )

    _heading(
        "Clutter -- a rival that departs at the same instant",
        " case                      | located | error | est. margin | warned",
    )
    rendered, phases, result, truth = _run(
        distractors=1, distractor_departure=fixture.departure_frame(120.0), seed=0
    )
    outcome = _score(result, truth)
    warned = any("emptied and stayed empty" in note for note in result.warnings)
    print(
        f" one rival, departs too    | "
        f"{'refused' if outcome.located_frame is None else f'{outcome.located_frame:7d}'} | "
        f"{_frames(outcome.error_frames):>5} | {_num(outcome.established_margin, 11)} | "
        f"{'yes' if warned else 'NO':>6}"
    )
    print(
        "  The case a single view cannot resolve. Two stationary objects vanish at the same\n"
        "  instant and nothing in the picture says which was struck; the instant is reported\n"
        "  because both give the same one, and the warning says the identification was a\n"
        "  choice rather than a reading."
    )


def sweep_occlusion(repeats: int) -> None:
    """Whether a ball being covered can be told from a ball being struck.

    The failure this phase's central caveat is about: at impact the club head is
    at the ball, so the two are the same picture at the instant itself. The only
    thing that separates them is what led up to it.
    """
    _heading(
        "Occlusion -- the ball dimmed over N frames before it goes",
        " fade frames | located | error | abruptness | confidence | warned",
    )
    for fade in (0, 2, 4, 6, 10):
        rendered, phases, result, truth = _run(fade_frames=fade, seed=0)
        outcome = _score(result, truth)
        warned = any("usual strength" in note for note in result.warnings)
        print(
            f" {fade:11d} | "
            f"{'refused' if outcome.located_frame is None else f'{outcome.located_frame:7d}'} | "
            f"{_frames(outcome.error_frames):>5} | {_num(outcome.abruptness, 10)} | "
            f"{_num(outcome.confidence)} | {'yes' if warned else 'no':>6}"
        )
    print(
        "  A struck ball is as visible in its final frame as in its first. Something moving\n"
        "  in front of it is not, and that is what the abruptness factor reads. Where it\n"
        "  falls, the instant may run early by however long the covering took."
    )


def sweep_truncation(repeats: int) -> None:
    """Whether a clip that ends too soon after impact says so."""
    _heading(
        "Truncation -- clip ending N frames after the ball goes",
        " frames after | located | error | permanence | confidence | warned",
    )
    departure_s = fixture.departure_frame(120.0) / 120.0
    for after in (2, 6, 12, 24, 48):
        duration = departure_s + after / 120.0
        rendered, phases, result, truth = _run(duration_s=duration, seed=0)
        outcome = _score(result, truth)
        warned = any("absence was verified" in note for note in result.warnings)
        located = "refused" if outcome.located_frame is None else f"{outcome.located_frame:7d}"
        print(
            f" {after:12d} | {located} | {_frames(outcome.error_frames):>5} | "
            f"{_num(outcome.permanence, 10)} | {_num(outcome.confidence)} | "
            f"{'yes' if warned else 'no':>6}"
        )
    print(
        "  Nothing at the instant itself distinguishes a ball that left from one something\n"
        "  moved in front of; only the absence lasting does. A clip that ends before the\n"
        "  absence can be verified has not made that check, and scores accordingly rather\n"
        "  than scoring the check it could not make as passed."
    )


def sweep_agreement(repeats: int) -> None:
    """How far the four impact estimates sit from each other, and from the truth.

    **The sweep this phase exists for, and the one whose caveat is largest.** In
    this fixture the hands' peak speed, the hands' lowest point, the club head's
    arrival and the ball's departure are all driven by one arc and coincide by
    construction. So agreement here demonstrates that the fusion wires the four
    together correctly and that none of them is out by a frame for a mechanical
    reason -- and it cannot measure the bias between them, because the fixture has
    none to measure.
    """
    _heading(
        "Agreement -- four estimates of one instant",
        " shutter | truth | ball | club | hand low | hand speed | reported by",
    )
    for shutter in (0.03, 0.125, 0.25, 0.5):
        rendered, phases, result, truth = _run(shutter_fraction=shutter, seed=0)

        club_detector = HoughShaftDetector(ClubConfig())
        club_evidence = [
            club_frame_evidence(
                item.frame,
                item.anchor,
                club_detector,
                geometry=club.FRAME,
                torso_length=club.TORSO_LENGTH_FW,
                timestamp_s=item.frame.timestamp_s,
            )
            for item in rendered
        ]
        club_result = track_shafts(
            club_evidence, ClubConfig(), phases=phases, frame_interval_s=1.0 / 120.0
        )

        ball_report = _as_ball_report(result)
        club_report = _as_club_report(club_result)
        fused = fuse_impact(
            phases,
            ball=ball_report,
            club=club_report,
            smoothing_window_s=0.05,
            frame_interval_s=1.0 / 120.0,
        )
        if fused is None:
            print(f" {shutter:7.3f} | {truth.departure_frame:5d} | no estimate at all")
            continue

        def at(source: ImpactSource) -> str:
            entry = fused.candidate(source)  # noqa: B023 - consumed within this iteration
            return "    -" if entry is None else f"{entry.frame_index:5d}"

        print(
            f" {shutter:7.3f} | {truth.departure_frame:5d} | {at(ImpactSource.BALL_DEPARTURE)} |"
            f" {at(ImpactSource.CLUB_HEAD)} | {at(ImpactSource.HAND_LOW):>8} |"
            f" {at(ImpactSource.HAND_SPEED):>10} | {fused.source.value}"
        )
    print(
        "  Every column is the same instant here **by construction**: one arc drives the\n"
        "  hands, the club and the ball, so this measures the wiring rather than the physics.\n"
        "  On a real swing the hands peak before the club head arrives, and the size of that\n"
        "  gap is exactly what nothing in this project could measure before this phase -- and\n"
        "  still cannot, from a fixture. It needs footage with a visible ball."
    )


def _as_ball_report(result: BallTrackResult) -> object:
    """Wrap a tracker result in just enough of a report for the fusion to read it.

    The fusion takes the contracts rather than the internal results, which is
    right -- it is the layer that reconciles what crosses the engine boundary --
    and a benchmark that drove the whole extract path would have to encode a video
    first. This is the seam that lets it not.
    """
    from analyzer.contracts.ball import BallDetectorInfo, BallTrackingReport
    from analyzer.contracts.pose import FrameGeometry

    return BallTrackingReport(
        detected=result.observed_frames > 0,
        video_path="<fixture>",
        detector=BallDetectorInfo(name="fixture", method="fixture", opencv_version="fixture"),
        geometry=FrameGeometry(width=club.FRAME.width, height=club.FRAME.height),
        torso_length=club.TORSO_LENGTH_FW,
        frame_count=len(result.frames),
        observed_frames=result.observed_frames,
        pre_departure_coverage=result.pre_departure_coverage,
        established=result.established,
        departure=result.departure,
        quality=result.quality,
        config=BallConfig(),
    )


def _as_club_report(result: object) -> object:
    from analyzer.contracts.club import ClubDetectorInfo, ClubTrackingReport
    from analyzer.contracts.pose import FrameGeometry

    return ClubTrackingReport(
        tracked=bool(getattr(result, "tracked_frames", 0)),
        video_path="<fixture>",
        detector=ClubDetectorInfo(name="fixture", method="fixture", opencv_version="fixture"),
        geometry=FrameGeometry(width=club.FRAME.width, height=club.FRAME.height),
        torso_length=club.TORSO_LENGTH_FW,
        frame_count=len(getattr(result, "frames", [])),
        tracked_frames=getattr(result, "tracked_frames", 0),
        coverage=0.0,
        impact=getattr(result, "impact", None),
        config=ClubConfig(),
    )


def measure_cost(repeats: int) -> None:
    """Per-frame cost, against the pose extraction that has to happen first."""
    rendered, poses, _ = fixture.swing_with_ball(fps=120.0)
    detector = ContrastBlobDetector(BallConfig())
    phases = _phases_for(120.0, poses)

    detect_times: list[float] = []
    evidence: list[BallEvidence] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        evidence = _evidence(rendered, detector)
        detect_times.append(time.perf_counter() - started)

    started = time.perf_counter()
    track_ball(evidence, BallConfig(), phases=phases, torso_length=club.TORSO_LENGTH_FW)
    tracking = time.perf_counter() - started

    frames = len(rendered)
    per_frame = float(np.median(detect_times)) * 1000.0 / frames
    _heading("Cost", " stage                     | median   | note")
    print(
        f" detect ({frames} frames)        | {per_frame:6.2f} ms | per frame, "
        f"{club.FRAME.width}x{club.FRAME.height} search region"
    )
    print(
        f" track                     | {tracking * 1000:6.2f} ms | whole clip, on candidates alone"
    )
    print(
        "  Against ~17 ms a frame to extract poses, which has to have happened first. The\n"
        "  search region is what costs: a disc of 1.5 torso lengths around the feet, which on\n"
        "  this fixture is most of the lower frame. A rectangular structuring element is what\n"
        "  makes it affordable -- OpenCV separates that one and nothing else, and an\n"
        "  elliptical one of the same size costs 66 ms a frame on its own."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Seeds per measurement.")
    parser.add_argument(
        "--sweep",
        choices=(
            "departure",
            "frame_rate",
            "contrast",
            "clutter",
            "occlusion",
            "truncation",
            "agreement",
            "cost",
            "all",
        ),
        default="all",
    )
    arguments = parser.parse_args()

    print(
        f"Synthetic ball: {fixture.BALL_RADIUS_PX:.1f} px radius on a "
        f"{club.FRAME.width}x{club.FRAME.height} frame, torso {club.TORSO_PX:.0f} px, "
        f"removed at frame {fixture.departure_frame(120.0)} of a 120 fps clip"
    )
    print(
        "Not a photograph: no defocus, no compression, flat turf where a range is a thousand\n"
        "bright blades, and a ball drawn as a uniform disc where a real one is a lit sphere.\n"
        "A floor, not an estimate."
    )

    chosen = arguments.sweep
    if chosen in ("departure", "all"):
        sweep_departure(arguments.repeats)
    if chosen in ("frame_rate", "all"):
        sweep_frame_rate(arguments.repeats)
    if chosen in ("contrast", "all"):
        sweep_contrast(arguments.repeats)
    if chosen in ("clutter", "all"):
        sweep_clutter(arguments.repeats)
    if chosen in ("occlusion", "all"):
        sweep_occlusion(arguments.repeats)
    if chosen in ("truncation", "all"):
        sweep_truncation(arguments.repeats)
    if chosen in ("agreement", "all"):
        sweep_agreement(arguments.repeats)
    if chosen in ("cost", "all"):
        measure_cost(arguments.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
