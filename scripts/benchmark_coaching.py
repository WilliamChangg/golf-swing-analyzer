#!/usr/bin/env python3
"""What the coaching engine can conclude, and what stops it.

Four sweeps, and the first two are the phase's result rather than its
instrumentation.

    inventory   what every rule is refused for, before any clip is involved
    resolution  the capture rate at which a tempo comparison starts to mean
                something, computed from the arithmetic and checked against the
                reference footage
    clips       the engine run over every reference clip that has an extraction
    guard       the rejection breakdown over a corpus of plausible wrong answers

`inventory` prints a table that does not depend on a recording at all: seven of
twelve rules are refused before a camera is switched on, and the reasons divide
into "nobody published a number for this" and "somebody published a number about
a different quantity". No footage changes either count.

`resolution` is the measurement that decides whether the three rules which *can*
fire ever do. A tempo ratio's uncertainty comes from one frame of ambiguity at
the top, and that one frame moves the ratio far more than it moves either
duration -- so the question "can this camera tell a 3:1 swing from a 2:1 swing"
has an answer in frames per second, and it is not 30.

Usage:
    uv run --project python python scripts/benchmark_coaching.py
    uv run --project python python scripts/benchmark_coaching.py --sweep resolution
    uv run --project python python scripts/benchmark_coaching.py --sweep clips
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.biomechanics import compute_metrics  # noqa: E402
from analyzer.biomechanics.registry import definition  # noqa: E402
from analyzer.coaching.bracket import bracket_for  # noqa: E402
from analyzer.coaching.engine import coach  # noqa: E402
from analyzer.coaching.guard import allowance_for, inspect  # noqa: E402
from analyzer.coaching.registry import known_rules  # noqa: E402
from analyzer.contracts.coaching import FindingRefusal, ThresholdMethod  # noqa: E402
from analyzer.contracts.filtering import FilterConfig, SmoothingConfig  # noqa: E402
from analyzer.contracts.metrics import MetricName  # noqa: E402
from analyzer.contracts.pose import LandmarkSpace  # noqa: E402
from analyzer.filtering.landmarks import filter_sequence  # noqa: E402
from analyzer.ingestion.probe import probe  # noqa: E402
from analyzer.paths import cache_dir  # noqa: E402
from analyzer.phases import detect_phases  # noqa: E402
from analyzer.pose.estimator import resolve_model  # noqa: E402
from analyzer.pose.store import read_sequence  # noqa: E402
from tests.synthetic_coaching import phases as fake_phases  # noqa: E402
from tests.synthetic_coaching import tempo_metrics  # noqa: E402


@dataclass(frozen=True)
class Clip:
    """A reference recording, and what it takes to get a swing out of it.

    The window and the factor are part of the fixture rather than defaults: a
    30 fps clip cannot be filtered with the shipped 0.10 s window -- three
    samples will not support a degree-4 fit -- and a slow-motion clip has no
    record of its own playback factor. Both were found by running this on real
    footage, and both are stated here rather than hidden in a helper.
    """

    label: str
    path: str
    window_s: float | None = None
    slow_motion: float = 1.0


CLIPS: tuple[Clip, ...] = (
    Clip("amateur, face-on, 30 fps", "data/amateur/face-on/PW_face-on.mp4", window_s=0.17),
    Clip("amateur, down the line, 30 fps", "data/amateur/dtl/iron_dtl.mp4", window_s=0.17),
    Clip("tour, face-on, 7x slow", "data/rory/face-on/rory_face_on.mp4", slow_motion=7.0),
    Clip("tour, down the line, 7x slow", "data/rory/dtl/rory_dtl.mp4", slow_motion=7.0),
)

# Plausible wrong answers, of the kind a language model writes. Each is paired
# with what it should be caught by, so the sweep reports whether the guard
# catches them for the right reason rather than merely catching them.
ADVERSARIAL: tuple[tuple[str, str], ...] = (
    ("Your downswing took 0.367 s, against a tour average of 0.250 s.", "number"),
    ("That is about 15% slower than the players this is measured against.", "number"),
    ("This is well short of the ninety degrees of a full turn.", "number"),
    ("Forty-five degrees of separation is the figure usually quoted.", "number"),
    ("Your clubhead is arriving at 0.367 mph.", "unit"),
    ("The hands travelled 0.367 yards through the downswing.", "unit"),
    ("That is 0.367 metres of movement towards the ball.", "unit"),
    ("Your clubhead speed will suffer from a downswing this long.", "quantity"),
    ("Expect a lower ball speed with this tempo.", "quantity"),
    ("This costs you carry distance.", "quantity"),
    ("A downswing this slow tends to produce a slice.", "ball flight"),
    ("You will hook it from this position.", "ball flight"),
    ("The slow downswing causes an inconsistent strike.", "causal"),
    ("This leads to a loss of speed through the ball.", "causal"),
)

FAITHFUL: tuple[str, ...] = (
    "Your downswing took 0.367 s, which is 0.100 s longer than the top of the published range.",
    "The downswing came in at 0.37 s. The published range tops out at 0.267 s.",
    "Measured 0.367 s, against a range of 0.200 s to 0.267 s.",
)


def sweep_inventory(_args: argparse.Namespace) -> None:
    """What the registry can conclude, before any recording is involved."""
    rows = []
    for rule in known_rules():
        source = rule.source
        if rule.no_threshold:
            verdict = "no published number"
        elif not source.permitted_bases:
            verdict = "no measurement protocol"
        elif definition(rule.metric).basis not in source.permitted_bases:
            verdict = "measures another quantity"
        else:
            verdict = "usable"
        rows.append((rule.rule_id, source.method.value, verdict))

    width = max(len(row[0]) for row in rows)
    print(f"{'rule':<{width}}  {'source method':<22}  verdict")
    for rule_id, method, verdict in rows:
        print(f"{rule_id:<{width}}  {method:<22}  {verdict}")

    counts = Counter(row[2] for row in rows)
    print(f"\n{len(rows)} rules: " + ", ".join(f"{count} {name}" for name, count in counts.items()))
    conventions = [
        rule for rule in known_rules() if rule.source.method is ThresholdMethod.CONVENTION
    ]
    numberless = [rule for rule in conventions if rule.no_threshold]
    print(
        f"{len(conventions)} cite a source with no published measurement protocol, "
        f"and {len(numberless)} of those publish no number either. No recording "
        "changes either count."
    )


def sweep_resolution(_args: argparse.Namespace) -> None:
    """The capture rate at which a tempo comparison starts to mean something.

    Held at one swing -- 0.80 s of backswing over 0.2333 s of downswing, which is
    the amateur reference clip exactly -- and varied only in the clock. Everything
    below is arithmetic on the bracket, and the last column is what the engine
    actually did with it.
    """
    backswing, downswing = 0.800, 7.0 / 30.0
    metrics = tempo_metrics(backswing, downswing)
    ratio = metrics.get(MetricName.TEMPO_RATIO)
    assert ratio is not None  # noqa: S101 - the fixture builds it

    print(f"backswing {backswing:.3f} s, downswing {downswing:.3f} s, tempo {ratio.value:.2f}:1")
    print("band 2.43 to 3.80, which is 1.37 wide\n")
    print(f"{'fps':>6}  {'bracket':>9}  {'to nearer edge':>15}  outcome")
    for fps in (30, 60, 120, 240, 480):
        interval = 1.0 / fps
        bracket, _ = bracket_for(ratio, metrics, interval)
        assert bracket is not None  # noqa: S101 - a ratio always has one here
        distance = min(abs(ratio.value - 2.43), abs(ratio.value - 3.80))
        report = coach(metrics, fake_phases(interval_s=interval))
        outcome = (
            "resolved" if report.finding("tempo.ratio") is not None else "cannot resolve the band"
        )
        print(f"{fps:>6}  {bracket:>9.3f}  {distance:>15.3f}  {outcome}")

    distance = min(abs(ratio.value - 2.43), abs(ratio.value - 3.80))
    crossover = next(
        (
            fps
            for fps in range(24, 241)
            if (bracket_for(ratio, metrics, 1.0 / fps)[0] or float("inf")) < distance
        ),
        None,
    )
    print(
        f"\nThis swing's tempo becomes resolvable at {crossover} fps. Below that the "
        "camera cannot tell which side of the published band the swing is on, "
        "whatever the ratio happens to read."
    )
    print(
        "The bracket is one frame of ambiguity at the top, which lengthens the "
        "backswing and shortens the downswing at once. That is why it falls with "
        "the frame rate faster than a duration's does."
    )


def _run(clip: Clip):  # type: ignore[no-untyped-def]
    metadata = probe(REPO_ROOT / clip.path).metadata
    entry, _ = resolve_model(None)
    poses = cache_dir() / "poses" / metadata.content_key.as_path_segment() / f"{entry.name}.parquet"
    if not poses.exists():
        return None
    config = (
        FilterConfig(smoothing=SmoothingConfig(window_s=clip.window_s))
        if clip.window_s
        else FilterConfig()
    )
    filtered = filter_sequence(
        read_sequence(poses),
        config,
        space=LandmarkSpace.FRAME_WIDTHS,
        slow_motion_factor=clip.slow_motion,
    )
    detected = detect_phases(filtered)
    metrics = compute_metrics(filtered, detected)
    return coach(metrics, detected, times=filtered.t.tolist())


def sweep_clips(_args: argparse.Namespace) -> None:
    """The engine over every reference clip that has a pose extraction."""
    reports = [(clip, _run(clip)) for clip in CLIPS]

    print(f"{'clip':<32}  {'view':<14}  {'ms/frame':>9}  {'found':>5}  {'refused':>7}  fired")
    for clip, report in reports:
        if report is None:
            print(f"{clip.label:<32}  no extraction -- run `analyzer extract` on it first")
            continue
        interval = report.frame_interval_s
        fired = ", ".join(entry.rule_id for entry in report.findings) or "-"
        print(
            f"{clip.label:<32}  {report.view.value:<14}  "
            f"{(interval * 1000 if interval else float('nan')):>9.1f}  "
            f"{len(report.findings):>5}  {len(report.refused):>7}  {fired}"
        )

    print("\nRefusals across all of them, by reason:")
    tally: Counter[FindingRefusal] = Counter()
    for _, report in reports:
        if report is not None:
            tally.update(entry.refusal for entry in report.refused)
    for refusal, count in tally.most_common():
        print(f"  {refusal.value:<22} {count}")


def sweep_guard(_args: argparse.Namespace) -> None:
    """The rejection breakdown, over answers written to look right.

    Both halves are reported. A guard that rejects everything is not protecting
    anybody -- it has turned the phrasing layer off in a way that looks like it is
    working -- so the faithful rewordings are counted too, and any of them that
    fails is a false rejection printed by name.
    """
    # The downswing finding, because the corpus is written about a downswing.
    # An allowance built from a different finding would reject the faithful
    # rewordings too, and the sweep would report a guard that works perfectly
    # while measuring nothing. It did exactly that on its first run.
    metrics = tempo_metrics(0.671, 0.367)
    report = coach(metrics, fake_phases(interval_s=1.0 / 210.0))
    finding = report.finding("tempo.downswing")
    assert finding is not None  # noqa: S101 - the fixture is built to produce one
    allowance = allowance_for(finding)

    caught: Counter[str] = Counter()
    missed = []
    for candidate, expected in ADVERSARIAL:
        offences = inspect(candidate, allowance)
        if offences:
            caught[expected] += 1
        else:
            missed.append(candidate)
    total = len(ADVERSARIAL)
    print(f"{sum(caught.values())} of {total} inventions rejected, by check:")
    for kind, count in sorted(caught.items()):
        print(f"  {kind:<12} {count}")
    for candidate in missed:
        print(f"  MISSED: {candidate}")

    false_rejections = [text for text in FAITHFUL if inspect(text, allowance)]
    print(f"\n{len(FAITHFUL) - len(false_rejections)} of {len(FAITHFUL)} faithful rewordings kept")
    for text in false_rejections:
        print(f"  FALSE REJECTION: {text}")

    started = time.perf_counter()
    for _ in range(200):
        inspect(ADVERSARIAL[0][0], allowance)
    elapsed = (time.perf_counter() - started) / 200
    print(f"\nguard: {elapsed * 1e6:.0f} us per candidate")


def sweep_cost(_args: argparse.Namespace) -> None:
    """What reaching a conclusion costs, once the measuring is done."""
    metrics = tempo_metrics(0.671, 0.367)
    detection = fake_phases(interval_s=1.0 / 210.0)

    started = time.perf_counter()
    for _ in range(500):
        coach(metrics, detection)
    per_call = (time.perf_counter() - started) / 500
    print(f"coach() over {len(known_rules())} rules: {per_call * 1000:.2f} ms")
    print(
        "Against seconds per clip for pose extraction. Nothing here is worth "
        "caching, which is the same conclusion Phase 3 and Phase 5 reached about "
        "the layers below."
    )


SWEEPS = {
    "inventory": sweep_inventory,
    "resolution": sweep_resolution,
    "clips": sweep_clips,
    "guard": sweep_guard,
    "cost": sweep_cost,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=sorted(SWEEPS), help="Run one sweep instead of all.")
    args = parser.parse_args()

    for name in [args.sweep] if args.sweep else list(SWEEPS):
        print(f"\n=== {name} ===")
        SWEEPS[name](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
