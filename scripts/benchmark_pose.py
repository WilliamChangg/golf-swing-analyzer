#!/usr/bin/env python3
"""Measure pose estimation cost for each model variant.

**Read the detection rate before reading the timings.** MediaPipe's pose
landmarker takes two different paths through its graph: when no pose is found it
runs the *detector* on every frame, and when one is found it runs the *landmark*
model and only re-detects when tracking is lost. Those cost different amounts.
A run over footage with nobody in it therefore measures the detector-only path,
which is a real path worth knowing the cost of and is **not** the throughput you
will see on a swing.

So this script refuses to recommend a default model unless it actually found
poses. Choosing lite over heavy on the strength of timings from an empty frame
would be exactly the kind of number this project is not allowed to publish.

Usage:
    uv run --project python python scripts/benchmark_pose.py --video data/raw/swing.mov
    uv run --project python python scripts/benchmark_pose.py --frames 200 --markdown
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.ingestion.probe import probe  # noqa: E402
from analyzer.ingestion.reader import OpenCVFrameSource  # noqa: E402
from analyzer.pose.estimator import PoseEstimationError  # noqa: E402
from analyzer.pose.mediapipe_estimator import MediaPipePoseEstimator  # noqa: E402

VARIANTS = ("pose_landmarker_lite", "pose_landmarker_full", "pose_landmarker_heavy")

# Enough frames that per-frame cost dominates the first-call warm-up, few enough
# that three models finish in about a minute.
DEFAULT_FRAMES = 300

# Below this, the timings describe the detector-only path rather than tracking,
# and no model recommendation may rest on them.
_MEANINGFUL_DETECTION_RATE = 0.5


@dataclass(frozen=True)
class Measurement:
    model: str
    load_ms: float
    frames: int
    detected: int
    total_s: float
    nudges: int

    @property
    def ms_per_frame(self) -> float:
        return self.total_s * 1000 / self.frames if self.frames else 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.total_s if self.total_s else 0.0

    @property
    def detection_rate(self) -> float:
        return self.detected / self.frames if self.frames else 0.0


def measure(video: Path, model: str, max_frames: int) -> Measurement:
    probed = probe(video)

    started = time.perf_counter()
    estimator = MediaPipePoseEstimator(model)
    load_ms = (time.perf_counter() - started) * 1000

    detected = 0
    frames = 0
    try:
        with OpenCVFrameSource(probed, cache_bytes=1) as source:
            # The first frame pays a one-off graph warm-up, so it is decoded and
            # estimated before the clock starts.
            warmup = source.frame_at(0)
            estimator.estimate(warmup)

            started = time.perf_counter()
            for frame in source.frames(start=1, stop=max_frames + 1):
                if estimator.estimate(frame).detected:
                    detected += 1
                frames += 1
            total_s = time.perf_counter() - started
    finally:
        estimator.close()

    return Measurement(
        model=model,
        load_ms=load_ms,
        frames=frames,
        detected=detected,
        total_s=total_s,
        nudges=estimator.timestamp_nudges,
    )


def render(results: list[Measurement], video: Path, frames_available: int, markdown: bool) -> None:
    print(f"{'video':>22}: {video.name}")
    print(f"{'frames available':>22}: {frames_available}")
    print(f"{'machine':>22}: {platform.machine()} / {platform.system()} {platform.release()}")
    print()

    if markdown:
        print("| Model | Load | ms/frame | Frames/s | Poses found |")
        print("| ----- | ---- | -------- | -------- | ----------- |")
        for r in results:
            print(
                f"| {r.model.removeprefix('pose_landmarker_')} | {r.load_ms:.0f} ms | "
                f"{r.ms_per_frame:.1f} | {r.fps:.0f} | "
                f"{r.detected}/{r.frames} ({r.detection_rate:.0%}) |"
            )
    else:
        header = f"{'model':<10} {'load':>9} {'ms/frame':>10} {'fps':>8} {'poses found':>16}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r.model.removeprefix('pose_landmarker_'):<10} {r.load_ms:>7.0f} ms "
                f"{r.ms_per_frame:>10.1f} {r.fps:>8.0f} "
                f"{f'{r.detected}/{r.frames} ({r.detection_rate:.0%})':>16}"
            )

    print()
    best_rate = max((r.detection_rate for r in results), default=0.0)
    if best_rate < _MEANINGFUL_DETECTION_RATE:
        print(
            "NO RECOMMENDATION. No model found a pose in most frames, so these timings\n"
            "measure the detector-only path taken when nobody is in shot, not the\n"
            "tracking path taken on real footage. Re-run against a clip with a person\n"
            "in it before choosing a default:\n"
            "    uv run --project python python scripts/benchmark_pose.py --video <clip>"
        )
        return

    fastest = min(results, key=lambda r: r.ms_per_frame)
    print(
        f"Fastest at a usable detection rate: {fastest.model} "
        f"({fastest.ms_per_frame:.1f} ms/frame, {fastest.detection_rate:.0%} found)."
    )
    print(
        "Speed is not the only criterion -- landmark accuracy differs between variants\n"
        "and is not measured here. Phase 12 is where accuracy gets a real evaluation set."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Clip to measure.")
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES, help="Frames to time per model."
    )
    parser.add_argument(
        "--models",
        default=",".join(VARIANTS),
        help="Comma-separated manifest model names.",
    )
    parser.add_argument("--markdown", action="store_true", help="Emit a markdown table.")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"No such file: {args.video}")

    available = probe(args.video).metadata.timing.frame_count
    results: list[Measurement] = []

    for model in (name.strip() for name in args.models.split(",") if name.strip()):
        try:
            results.append(measure(args.video, model, args.frames))
        except PoseEstimationError as exc:
            # A model that is not downloaded is skipped and said to be skipped,
            # rather than silently dropping out of the comparison.
            print(f"skipped {model}: {exc}", file=sys.stderr)

    if not results:
        raise SystemExit("No models could be measured.")

    render(results, args.video, available, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
