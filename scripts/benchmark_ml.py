#!/usr/bin/env python3
"""Measure what the learned detector is worth, and what a leaky split is worth.

**Every number this prints is measured on a generated corpus and describes the
machinery, not golf.** `tests/synthetic_labels.py` draws swings from event times
that are inputs to the generator, so a detector scored against them is being
asked to recover parameters it was effectively told. No clip in this project has
a hand-labelled swing in it, and `analyzer/ml/evaluate.py` refuses to mark any of
these scores publishable for exactly that reason.

What a fixture *can* measure honestly is a property of method rather than of the
sport, and that is what the sweeps below are for:

    leakage     what does splitting by clip instead of by player buy you?
    players     how many golfers does the held-out number stop moving at?
    receptive   does a model that cannot see a whole swing find the top?
    rate        what does the canonical grid cost at each sample rate?
    baseline    the rule-based detector and the model on one held-out set
    cost        how long does any of this take?

`leakage` is the one this phase exists to make. The two splits differ in nothing
but which clips go where -- same corpus, same seed, same architecture, same
epochs -- so the gap between what they report is the overstatement a leaky
evaluation would have published, measured rather than asserted.

Usage:
    uv run --project python python scripts/benchmark_ml.py
    uv run --project python python scripts/benchmark_ml.py --sweep leakage
    uv run --project python python scripts/benchmark_ml.py --seeds 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.contracts.ml import TCNConfig
from analyzer.contracts.phases import SwingEvent
from analyzer.ml.compare import compare_detectors
from analyzer.ml.dataset import Dataset, build_dataset
from analyzer.ml.evaluate import evaluate, noise_floor_ms, predict_classes, rule_classes
from analyzer.ml.features import FEATURE_SPEC
from analyzer.ml.splits import SplitRefused, plan_split, random_clip_split, split_dataset
from analyzer.ml.train import train
from tests.synthetic_labels import make_corpus

# Kept small so the whole file runs in a couple of minutes on a laptop. The
# corpus is the variable that matters for the leakage sweep -- more players make
# the grouped split harder and the leaky one no harder at all.
DEFAULT_PLAYERS = 12
DEFAULT_SESSIONS = 2
DEFAULT_SWINGS = 3
BENCH_CONFIG = TCNConfig(epochs=40, patience=10)

CAVEAT = (
    "Synthetic corpus: events are generator inputs, not observations. These "
    "numbers describe the machinery and are not an accuracy claim about golf."
)


@dataclass(frozen=True)
class Run:
    """One trained model's held-out result."""

    macro_f1: float
    events: dict[SwingEvent, float | None]
    seconds: float


def corpus_dataset(players: int, sessions: int, swings: int, seed: int) -> Dataset:
    corpus = make_corpus(players=players, sessions=sessions, swings=swings, seed=seed)
    return build_dataset(corpus.labels, corpus.provider())


def run_once(
    built: Dataset,
    *,
    seed: int,
    planner: object = None,
    config: TCNConfig = BENCH_CONFIG,
) -> Run | None:
    """Train and score once, or None when the set cannot be split honestly."""
    try:
        split = split_dataset(built, seed=seed, planner=planner)
    except SplitRefused:
        return None

    started = time.perf_counter()
    model, _ = train(split.train, split.val, config=config, seed=seed)
    report = evaluate(
        "model", [predict_classes(model, example) for example in split.test], split.test
    )
    return Run(
        macro_f1=report.macro_f1,
        events={error.event: error.mae_ms for error in report.events},
        seconds=time.perf_counter() - started,
    )


def _mean(values: Sequence[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def _format(value: float | None, suffix: str = " ms") -> str:
    return "-" if value is None else f"{value:.1f}{suffix}"


def sweep_leakage(args: argparse.Namespace) -> None:
    """What a split by clip reports that a split by player does not.

    The single measurement this phase was built to make. Nothing differs between
    the two columns except which clips went into the held-out set.
    """
    built = corpus_dataset(args.players, args.sessions, args.swings, args.seed)
    grouped: list[Run] = []
    leaked: list[Run] = []
    for offset in range(args.seeds):
        seed = args.seed + offset
        by_player = run_once(built, seed=seed)
        by_clip = run_once(built, seed=seed, planner=random_clip_split)
        if by_player is not None:
            grouped.append(by_player)
        if by_clip is not None:
            leaked.append(by_clip)

    print(f"\n{args.seeds} seeds, {built.summary.clips} clips, {built.summary.players} players")
    print(
        f"{'quantity':<20}{'by player':>18}{'by clip':>18}{'difference':>13}{'readable':>10}"
    )
    print("-" * 79)

    verdicts: list[bool] = []
    for name, values in (
        ("macro F1", [([run.macro_f1 for run in grouped], [run.macro_f1 for run in leaked])]),
    ):
        honest, dishonest = values[0]
        gap = statistics.fmean(dishonest) - statistics.fmean(honest)
        spread = _spread(honest) + _spread(dishonest)
        verdicts.append(abs(gap) > spread)
        print(
            f"{name:<20}{_pm(honest, '{:.3f}'):>18}{_pm(dishonest, '{:.3f}'):>18}"
            f"{gap:>+13.3f}{('yes' if abs(gap) > spread else 'no'):>10}"
        )

    for event in SwingEvent:
        honest_values = [value for run in grouped if (value := run.events.get(event)) is not None]
        leaked_values = [value for run in leaked if (value := run.events.get(event)) is not None]
        if not honest_values or not leaked_values:
            print(f"{event.value + ' MAE':<20}{'-':>18}{'-':>18}{'-':>13}{'-':>10}")
            continue
        gap = statistics.fmean(honest_values) - statistics.fmean(leaked_values)
        spread = _spread(honest_values) + _spread(leaked_values)
        verdicts.append(abs(gap) > spread)
        print(
            f"{event.value + ' MAE':<20}{_pm(honest_values):>18}{_pm(leaked_values):>18}"
            f"{f'{gap:+.1f} ms':>13}{('yes' if abs(gap) > spread else 'no'):>10}"
        )

    print(
        "\n`difference` is by-player minus by-clip, so a positive number is the leaky\n"
        "split reporting a better result. `readable` is whether that difference is\n"
        "larger than the seed-to-seed spread of the two columns added together."
    )
    if not any(verdicts):
        print(
            "\nNothing here is readable, and that is the result. This corpus cannot\n"
            "measure what leakage costs: its golfers differ by six generator parameters,\n"
            "so a model that has seen seven of them has seen the space, and holding one\n"
            "out asks nothing. Real golfers differ in ways a generator does not know how\n"
            "to vary -- which is the same sentence Phase 11 had to write about measuring\n"
            "impact agreement on a fixture where one arc drove everything.\n"
            "\nThe split is grouped by player anyway. The argument for it was never this\n"
            "measurement; it is that a swing and its near-duplicate cannot be on opposite\n"
            "sides of a question, and no measurement is needed to see that."
        )
    print(f"\n{CAVEAT}")


def _spread(values: Sequence[float]) -> float:
    """Half the range across seeds. The scale a difference has to beat to be real."""
    if len(values) < 2:
        return float("inf")
    return (max(values) - min(values)) / 2.0


def _pm(values: Sequence[float], template: str = "{:.1f}") -> str:
    """mean +- half-range, so a reader sees the scatter beside the number."""
    return f"{template.format(statistics.fmean(values))} +-{template.format(_spread(values))}"


def sweep_players(args: argparse.Namespace) -> None:
    """How the held-out number moves as the corpus gains golfers."""
    print(f"\n{'players':>9}{'clips':>8}{'macro F1':>11}{'top MAE':>11}{'impact MAE':>13}")
    print("-" * 52)
    for players in (3, 4, 6, 9, args.players):
        built = corpus_dataset(players, args.sessions, args.swings, args.seed)
        runs = [
            run for offset in range(args.seeds)
            if (run := run_once(built, seed=args.seed + offset)) is not None
        ]
        if not runs:
            print(f"{players:>9}{built.summary.clips:>8}{'refused':>11}")
            continue
        print(
            f"{players:>9}{built.summary.clips:>8}"
            f"{statistics.fmean(run.macro_f1 for run in runs):>11.3f}"
            f"{_format(_mean([run.events.get(SwingEvent.TOP) for run in runs])):>11}"
            f"{_format(_mean([run.events.get(SwingEvent.IMPACT) for run in runs])):>13}"
        )
    print(
        "\nA held-out set of two golfers is two golfers. The figure moving with the\n"
        "corpus size is the measurement; its value at any one size is not."
    )
    print(f"\n{CAVEAT}")


def sweep_receptive(args: argparse.Namespace) -> None:
    """Whether a model that cannot see a whole swing can find the top of one."""
    built = corpus_dataset(args.players, args.sessions, args.swings, args.seed)
    print(f"\n{'layers':>7}{'field':>9}{'seconds':>10}{'params':>9}{'macro F1':>11}{'top MAE':>11}")
    print("-" * 57)
    for layers in (3, 4, 5, 6, 7):
        config = BENCH_CONFIG.model_copy(update={"layers": layers})
        runs = [
            run for offset in range(args.seeds)
            if (run := run_once(built, seed=args.seed + offset, config=config)) is not None
        ]
        if not runs:
            continue
        field = config.receptive_field
        from analyzer.ml.tcn import SwingTCN

        print(
            f"{layers:>7}{field:>9}{field / FEATURE_SPEC.sample_rate_hz:>10.2f}"
            f"{SwingTCN(len(FEATURE_SPEC.names), config).parameter_count():>9}"
            f"{statistics.fmean(run.macro_f1 for run in runs):>11.3f}"
            f"{_format(_mean([run.events.get(SwingEvent.TOP) for run in runs])):>11}"
        )
    print(
        "\nThe expectation was that a field shorter than a swing would lose the top,\n"
        "which is defined by what happens on both sides of it. The table does not\n"
        "show that, and the reason is a property of the fixture rather than a\n"
        "discovery about swings: these hands follow one analytic arc, so the speed\n"
        "profile in a quarter of a second already says where in the swing it is. On\n"
        "footage where a golfer pauses at the top, waggles at address or decelerates\n"
        "into a checked follow-through, the local picture stops being sufficient.\n"
        "\nThe shipped default is six layers, chosen on that argument and not on this\n"
        "table, which cannot distinguish any of these rows from any other."
    )
    print(f"\n{CAVEAT}")


def sweep_rate(args: argparse.Namespace) -> None:
    """What the canonical grid costs, before any model is involved.

    No training here. A label is a frame of the source clip, the grid is a
    resampling of it, and mapping a label onto the grid and back is a lower bound
    on the error of anything that works on the grid.
    """
    print(f"\n{'fps':>6}{'grid':>8}{'floor':>10}{'round trip':>13}{'worst':>9}")
    print("-" * 46)
    for fps in (30.0, 60.0, 120.0, 240.0):
        # Every session forced to one rate, so the column varies in nothing else.
        corpus = make_corpus(players=3, sessions=1, swings=2, seed=args.seed, rates=(fps,))
        built = build_dataset(corpus.labels, corpus.provider())
        errors: list[float] = []
        for example in built.examples:
            if not example.label.is_swing:
                continue
            per_frame_ms = 1000.0 * example.label.real_seconds_per_frame
            for entry in example.label.events:
                index = int(np.argmin(np.abs(example.features.source_frames - entry.frame_index)))
                landed = int(example.features.source_frames[index])
                errors.append(abs(landed - entry.frame_index) * per_frame_ms)
        print(
            f"{fps:>6.0f}{FEATURE_SPEC.sample_rate_hz:>8.0f}"
            f"{1000 * FEATURE_SPEC.resample_floor_s:>10.1f}"
            f"{statistics.fmean(errors):>13.1f}{max(errors):>9.1f}"
        )
    print(
        "\nThe floor is half a grid sample and it is spent before a model runs. At\n"
        "240 fps it is larger than the 4.2 ms bracket Phase 11 gets from watching\n"
        "the ball leave: a learned detector does not replace an observation."
    )
    print(f"\n{CAVEAT}")


def sweep_baseline(args: argparse.Namespace) -> None:
    """The rule-based detector and the model, one held-out set, one metric."""
    built = corpus_dataset(args.players, args.sessions, args.swings, args.seed)
    split = split_dataset(built, seed=args.seed)
    model, report = train(split.train, split.val, config=BENCH_CONFIG, seed=args.seed)
    comparison = compare_detectors(model, split.test, model_id="benchmark")

    floor = noise_floor_ms(split.test)
    print(
        f"\n{len(split.test)} held-out clips from {comparison.players} players; "
        f"this set resolves {floor:.1f} ms"
    )
    print(f"{'event':<12}{'rules':>10}{'model':>10}{'difference':>13}{'verdict':>20}")
    print("-" * 65)
    for row in comparison.rows:
        print(
            f"{row.event.value:<12}{_format(row.rule_mae_ms):>10}{_format(row.model_mae_ms):>10}"
            f"{_format(row.difference_ms):>13}{row.verdict:>20}"
        )
    print(
        f"\nmacro F1: rules {comparison.rule.macro_f1:.3f}, model {comparison.model.macro_f1:.3f}"
    )
    print(f"{comparison.verdict}")
    print(f"refused because: {comparison.model.claim_refusal}")
    print(f"\ntrained in {report.elapsed_s:.1f} s over {len(report.epochs)} epochs")
    print(
        "\n**The comparison is rigged in the model's favour here, and the rigging is\n"
        "worth understanding because a version of it survives into real data.** The\n"
        "model is trained on these labels, so it learns the *convention* they were\n"
        "made with; the rules brought their own. On this corpus the two conventions\n"
        "genuinely differ at the takeaway: the generator's takeaway is the instant its\n"
        "easing function leaves zero, and Phase 4's is where hand speed crosses 5% of\n"
        "its peak, which on a sin-squared ramp is several frames later. The model is\n"
        "not more accurate there. It has been told which definition is being marked.\n"
        "\nA human labelled set narrows this and does not close it: a person marks the\n"
        "takeaway where they can see the club move, which is a third convention again."
    )
    print(f"\n{CAVEAT}")


def sweep_cost(args: argparse.Namespace) -> None:
    """Where the time goes, so a later decision about caching has a number."""
    started = time.perf_counter()
    corpus = make_corpus(
        players=args.players, sessions=args.sessions, swings=args.swings, seed=args.seed
    )
    generated = time.perf_counter() - started

    started = time.perf_counter()
    built = build_dataset(corpus.labels, corpus.provider())
    featurised = time.perf_counter() - started

    split = split_dataset(built, seed=args.seed)
    started = time.perf_counter()
    model, report = train(split.train, split.val, config=BENCH_CONFIG, seed=args.seed)
    trained = time.perf_counter() - started

    started = time.perf_counter()
    for example in split.test:
        predict_classes(model, example)
    inference = time.perf_counter() - started

    started = time.perf_counter()
    for example in split.test:
        rule_classes(example)
    rules = time.perf_counter() - started

    samples = built.summary.samples
    print(f"\n{built.summary.clips} clips, {samples} samples at {FEATURE_SPEC.sample_rate_hz:g} Hz")
    print(f"{'stage':<28}{'total':>10}{'per clip':>12}")
    print("-" * 50)
    for name, seconds, count in (
        ("generate the corpus", generated, built.summary.clips),
        ("filter + featurise", featurised, built.summary.clips),
        (f"train ({len(report.epochs)} epochs)", trained, built.summary.clips),
        ("model inference", inference, len(split.test)),
        ("read the rule detector", rules, len(split.test)),
    ):
        print(f"{name:<28}{seconds:>9.2f}s{1000 * seconds / max(count, 1):>11.1f}ms")
    print(
        f"\nThe model has {report.parameters} parameters and a "
        f"{report.receptive_field_s:.2f} s receptive field."
    )
    print(f"\n{CAVEAT}")


def sweep_refusal(_args: argparse.Namespace) -> None:
    """What the split says about the footage this project actually has.

    Not a sweep. It runs the real decision on the real shape of the repository's
    data -- one player, four clips -- and prints the refusal, because that
    refusal is Phase 12's result and it should be reproducible by running
    something rather than by reading a document.
    """
    corpus = make_corpus(players=1, sessions=2, swings=2, seed=0)
    report = plan_split(list(corpus.labels.clips), label_digest=corpus.labels.digest())
    print(f"\n{len(corpus.labels.clips)} clips, {len(corpus.labels.players)} player")
    print(f"refused: {report.refused}")
    print(f"\n{report.refusal}\n")
    print(
        "This is the shape of the footage in data/: four clips of one golfer. The\n"
        "machinery above runs on it up to this point and stops here, which is the\n"
        "correct place to stop."
    )


SWEEPS = {
    "leakage": sweep_leakage,
    "players": sweep_players,
    "receptive": sweep_receptive,
    "rate": sweep_rate,
    "baseline": sweep_baseline,
    "cost": sweep_cost,
    "refusal": sweep_refusal,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=sorted(SWEEPS), help="Run one sweep instead of all.")
    parser.add_argument("--players", type=int, default=DEFAULT_PLAYERS)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--swings", type=int, default=DEFAULT_SWINGS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds", type=int, default=3, help="How many seeds to average a sweep over."
    )
    args = parser.parse_args()

    selected = [args.sweep] if args.sweep else list(SWEEPS)
    for name in selected:
        print(f"\n=== {name} ===")
        SWEEPS[name](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
