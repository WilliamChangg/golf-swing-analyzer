#!/usr/bin/env python3
"""Measure the desktop analysis path on a named, reproducible input.

The output records the input's content key, every analysis option, code revision,
wall time, and OS process-memory samples.  That makes a number comparable to a
future run only when the things that can change it are visible beside it.

The benchmark has two modes:

* ``baseline`` disables Phase 17 result caches and writes each pose extraction
  to a temporary path, so it measures a genuine re-analysis.
* ``warm`` first populates the content/config cache, then measures the same UI
  workflow again.  It never deletes a user's existing cache.

Usage:
    uv run --project python python scripts/benchmark.py data/amateur/face-on/PW_face-on.mp4
    uv run --project python python scripts/benchmark.py <clip> --repeats 5 --json > benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.contracts.filtering import FilterConfig, SmoothingConfig
from analyzer.dispatch import call
from analyzer.ingestion import probe_video
from analyzer.performance import PerformanceRecorder


@contextmanager
def cache_disabled() -> Iterator[None]:
    """Turn off only small analysis-result caches, restoring the environment."""
    previous = os.environ.get("GSA_DISABLE_ANALYSIS_CACHE")
    os.environ["GSA_DISABLE_ANALYSIS_CACHE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GSA_DISABLE_ANALYSIS_CACHE", None)
        else:
            os.environ["GSA_DISABLE_ANALYSIS_CACHE"] = previous


def revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_call(method: str, params: dict[str, object]) -> dict[str, object]:
    recorder = PerformanceRecorder(method)
    call(method, params, performance=recorder)
    return recorder.as_dict()


def workflow(video: Path, *, output: Path | None, config: FilterConfig) -> list[dict[str, object]]:
    """Exactly the serial workflow the Swing screen invokes on Analyse."""
    extract: dict[str, object] = {"path": str(video)}
    if output is not None:
        extract["output"] = str(output)
    source = str(output) if output is not None else str(video)
    options: dict[str, object] = {"path": source, "filter": config.model_dump(mode="json")}
    return [
        run_call("extract_poses", extract),
        run_call("detect_phases", options),
        run_call("compute_metrics", options),
        run_call("coach_swing", options),
        run_call("pose_overlay", {**options, "start_frame": 0}),
    ]


def median_by_operation(runs: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """Reduce repetitions without hiding the individual stage measurements."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        for report in run:
            grouped.setdefault(str(report["operation"]), []).append(report)

    result: list[dict[str, object]] = []
    for operation, reports in grouped.items():
        elapsed = [float(entry["elapsed_s"]) for entry in reports]
        stages: dict[str, list[float]] = {}
        for report in reports:
            for stage in report["stages"]:  # type: ignore[index]
                entry = stage  # pyright: ignore[reportUnknownVariableType]
                stages.setdefault(str(entry["name"]), []).append(float(entry["elapsed_s"]))
        result.append(
            {
                "operation": operation,
                "median_elapsed_s": statistics.median(elapsed),
                "stage_median_s": {name: statistics.median(values) for name, values in stages.items()},
            }
        )
    return result


def render(summary: dict[str, object]) -> None:
    print(f"input: {summary['input']['path']}")  # type: ignore[index]
    print(f"content: {summary['input']['content_key']}")  # type: ignore[index]
    print(f"repetitions: {summary['repeats']}  (median; seconds)")
    for mode in ("baseline", "warm"):
        print(f"\n{mode}")
        print(f"  {'operation':<20} {'median':>10}  nested stages")
        for row in summary[mode]["median"]:  # type: ignore[index]
            stages = ", ".join(
                f"{name} {value * 1000:.1f}ms"
                for name, value in row["stage_median_s"].items()  # type: ignore[index]
                if not name.startswith("rpc:")
            )
            print(f"  {row['operation']:<20} {row['median_elapsed_s']:>9.3f}s  {stages}")
    print("\nRSS samples are in --json output. On macOS they are process peak RSS, not allocation attribution.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="A video with (or from which to create) a pose cache.")
    parser.add_argument("--repeats", type=int, default=3, help="Measured repetitions per mode.")
    parser.add_argument("--window-s", type=float, default=None, help="Override smoothing support in seconds.")
    parser.add_argument("--json", action="store_true", help="Emit the full reproducibility record as JSON.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    video = args.video.expanduser().resolve()
    metadata = probe_video(video)
    smoothing = SmoothingConfig(**({"window_s": args.window_s} if args.window_s else {}))
    config = FilterConfig(smoothing=smoothing)

    baseline: list[list[dict[str, object]]] = []
    # Temporary output forces extraction work without clearing a user cache.
    with tempfile.TemporaryDirectory(prefix="gsa-benchmark-") as temporary, cache_disabled():
        for number in range(args.repeats):
            baseline.append(
                workflow(video, output=Path(temporary) / f"poses-{number}.parquet", config=config)
            )

    # Prime the same content/config key once. This setup is deliberately not in
    # the reported samples: warm means a repeat analysis, not a first analysis
    # that happens to write its caches.
    workflow(video, output=None, config=config)
    warm = [workflow(video, output=None, config=config) for _ in range(args.repeats)]

    summary: dict[str, object] = {
        "schema_version": 1,
        "revision": revision(),
        "platform": platform.platform(),
        "python": sys.version,
        "input": {
            "path": str(video),
            "content_key": metadata.content_key.model_dump(mode="json"),
            "frames": metadata.timing.frame_count,
            "duration_s": metadata.timing.duration_s,
        },
        "config": config.model_dump(mode="json"),
        "repeats": args.repeats,
        "baseline": {"runs": baseline, "median": median_by_operation(baseline)},
        "warm": {"runs": warm, "median": median_by_operation(warm)},
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        render(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
