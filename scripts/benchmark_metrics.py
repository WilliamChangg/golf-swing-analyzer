#!/usr/bin/env python3
"""Measure what the biomechanics layer costs, and how much of the clip it uses.

Two questions, both of which decide something.

**Is it worth caching?** Phase 3 measured filtering at milliseconds against
seconds of extraction and concluded that caching it would add a schema to
version and invalidate for no measurable gain. The metric layer sits on top of
filtering and phase detection, so the same question has to be asked again rather
than assumed to have the same answer.

**How much of a clip survives to be measured?** A metric is only produced where
its landmarks were tracked at the instant it is anchored to, so the count of
metrics produced against metrics refused is a property of the footage. It is
reported here because it is the number that says whether a capture was good
enough, and it is not visible from a timing figure.

Usage:
    uv run --project python python scripts/benchmark_metrics.py data/face-on/swing.mp4
    uv run --project python python scripts/benchmark_metrics.py <clip> --window 0.15
    uv run --project python python scripts/benchmark_metrics.py <clip> --repeats 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.biomechanics import compute_metrics  # noqa: E402
from analyzer.contracts.filtering import FilterConfig, SmoothingConfig  # noqa: E402
from analyzer.filtering.landmarks import filter_sequence  # noqa: E402
from analyzer.ingestion.probe import probe  # noqa: E402
from analyzer.paths import cache_dir  # noqa: E402
from analyzer.phases import detect_phases  # noqa: E402
from analyzer.pose.estimator import resolve_model  # noqa: E402
from analyzer.pose.store import read_sequence  # noqa: E402


def resolve_poses(path: Path, model: str | None) -> Path:
    if path.suffix == ".parquet":
        return path
    metadata = probe(path).metadata
    entry, _ = resolve_model(model)
    poses = cache_dir() / "poses" / metadata.content_key.as_path_segment() / f"{entry.name}.parquet"
    if not poses.exists():
        raise SystemExit(
            f"No extracted poses for {path.name} with model '{entry.name}'.\n"
            f"Run: uv run --project python analyzer extract {path} --model {entry.name}"
        )
    return poses


def median_ms(work, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        work()
        timings.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(timings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+", help="Video files, or pose Parquet files.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--window", type=float, default=None, help="Filtering window in seconds.")
    parser.add_argument("--polyorder", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    smoothing = SmoothingConfig(
        **{
            key: value
            for key, value in (("window_s", args.window), ("polyorder", args.polyorder))
            if value is not None
        }
    )
    config = FilterConfig(smoothing=smoothing)

    header = f"{'clip':<28}{'frames':>8}{'filter':>10}{'phases':>10}{'metrics':>10}{'produced':>11}{'refused':>9}"
    print(header)
    print("-" * len(header))

    for path in args.paths:
        sequence = read_sequence(resolve_poses(path, args.model))

        filter_ms = median_ms(lambda s=sequence: filter_sequence(s, config), args.repeats)
        filtered = filter_sequence(sequence, config)

        phases_ms = median_ms(lambda f=filtered: detect_phases(f), args.repeats)
        phases = detect_phases(filtered)

        metrics_ms = median_ms(
            lambda f=filtered, p=phases: compute_metrics(f, p), args.repeats
        )
        result = compute_metrics(filtered, phases)

        print(
            f"{path.name:<28}{len(sequence.frames):>8}"
            f"{filter_ms:>9.1f}{'ms':>1}{phases_ms:>9.1f}{'ms':>1}{metrics_ms:>9.1f}{'ms':>1}"
            f"{len(result.metrics):>11}{len(result.refused):>9}"
        )

    print(
        "\nMedian of "
        f"{args.repeats} runs each. Filtering dominates: it fits a polynomial at every "
        "sample of 33 landmarks, while the metric layer reads a few dozen frames of a "
        "result that is already in memory."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
