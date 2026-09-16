#!/usr/bin/env python3
"""Measure ingestion cost: probing, and decode throughput per backend.

Answers the question Phase 1 has to answer with data rather than assumption --
whether hardware decode is worth using -- and records the probe cost that the
metadata cache exists to avoid paying twice.

Every figure this prints is measured here. Nothing in the documentation may
state a throughput number that did not come out of this script, and the machine
it ran on is printed alongside the numbers because they mean nothing without it.

Usage:
    uv run --project python python scripts/benchmark_decode.py
    uv run --project python python scripts/benchmark_decode.py --video path/to/clip.mp4
    uv run --project python python scripts/benchmark_decode.py --repeats 5 --markdown
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.ingestion import (  # noqa: E402
    DecodeBackend,
    FFmpegPipeFrameSource,
    OpenCVFrameSource,
    VideoMetadataCache,
    probe,
    probe_video,
)

# Large enough that per-frame cost dominates process startup, and representative
# of real capture: 1080p at a high frame rate is what the capture protocol asks
# for. Regenerated rather than committed, because it is megabytes.
_SYNTHETIC = "bench_1080p120.mp4"
_SYNTHETIC_ARGS = (
    "testsrc2=size=1920x1080:rate=120:duration=5",
    ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"],
)


@dataclass(frozen=True)
class Timing:
    """Repeated measurements of one operation."""

    label: str
    samples_s: list[float]
    frames: int

    @property
    def median_s(self) -> float:
        return statistics.median(self.samples_s)

    @property
    def fps(self) -> float:
        return self.frames / self.median_s

    @property
    def spread(self) -> str:
        return f"{min(self.samples_s) * 1000:.0f}-{max(self.samples_s) * 1000:.0f}"


@contextmanager
def _stopwatch() -> Iterator[Callable[[], float]]:
    start = time.perf_counter()
    elapsed = 0.0

    def read() -> float:
        return elapsed

    try:
        yield read
    finally:
        elapsed = time.perf_counter() - start


def _time(label: str, frames: int, repeats: int, run: Callable[[], int]) -> Timing:
    samples: list[float] = []
    actual = frames
    for _ in range(repeats):
        start = time.perf_counter()
        actual = run()
        samples.append(time.perf_counter() - start)
    return Timing(label=label, samples_s=samples, frames=actual)


def ensure_video(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise SystemExit(f"No such file: {path}")
        return path

    target = REPO_ROOT / "data" / _SYNTHETIC
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    source, encode = _SYNTHETIC_ARGS
    print(f"Generating {target.relative_to(REPO_ROOT)} ...")
    subprocess.run(  # noqa: S603
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", source]
        + encode
        + [str(target)],
        check=True,
    )
    return target


def _decode_all_opencv(path: Path) -> int:
    with OpenCVFrameSource(path, cache_bytes=1) as source:
        return sum(1 for _ in source.frames())


def _decode_all_ffmpeg(path: Path, *, hardware: bool) -> int:
    with FFmpegPipeFrameSource(path, hardware=hardware, cache_bytes=1) as source:
        return sum(1 for _ in source.frames())


def run(video: Path, repeats: int) -> tuple[list[Timing], list[tuple[str, str]]]:
    probed = probe(video)
    metadata = probed.metadata
    frames = metadata.timing.frame_count

    timings: list[Timing] = []

    # Probe cost: what the metadata cache saves on a re-open.
    timings.append(
        _time("probe (2 ffprobe passes)", 1, repeats, lambda: (probe(video), 1)[1])
    )

    cache = VideoMetadataCache(REPO_ROOT / "data" / ".benchmark-cache")
    probe_video(video, cache=cache)  # prime
    timings.append(
        _time("probe (cache hit)", 1, repeats, lambda: (probe_video(video, cache=cache), 1)[1])
    )

    # Decode throughput. The frame cache is disabled so every run does real work.
    timings.append(
        _time("decode: opencv", frames, repeats, lambda: _decode_all_opencv(video))
    )
    timings.append(
        _time(
            "decode: ffmpeg cpu",
            frames,
            repeats,
            lambda: _decode_all_ffmpeg(video, hardware=False),
        )
    )
    timings.append(
        _time(
            "decode: ffmpeg videotoolbox",
            frames,
            repeats,
            lambda: _decode_all_ffmpeg(video, hardware=True),
        )
    )

    facts = [
        ("video", video.name),
        (
            "geometry",
            f"{metadata.stream.display_width}x{metadata.stream.display_height} "
            f"{metadata.stream.codec_name} {metadata.stream.pix_fmt}",
        ),
        ("frames", str(frames)),
        ("duration", f"{metadata.timing.duration_s:.3f} s"),
        (
            "measured fps",
            f"{metadata.timing.measured_fps:.3f}" if metadata.timing.measured_fps else "-",
        ),
        ("variable frame rate", str(metadata.timing.is_vfr)),
        ("machine", f"{platform.machine()} / {platform.system()} {platform.release()}"),
        ("repeats", str(repeats)),
    ]
    return timings, facts


def render(timings: list[Timing], facts: list[tuple[str, str]], markdown: bool) -> None:
    for key, value in facts:
        print(f"{key:>22}: {value}")
    print()

    if markdown:
        print("| Operation | Median | Range (ms) | Frames/s |")
        print("| --------- | ------ | ---------- | -------- |")
        for timing in timings:
            throughput = f"{timing.fps:.0f}" if timing.frames > 1 else "-"
            print(
                f"| {timing.label} | {timing.median_s * 1000:.0f} ms | "
                f"{timing.spread} | {throughput} |"
            )
        return

    width = max(len(t.label) for t in timings)
    for timing in timings:
        throughput = f"{timing.fps:>8.0f} fps" if timing.frames > 1 else " " * 12
        print(
            f"{timing.label:<{width}}  {timing.median_s * 1000:>8.1f} ms median  "
            f"[{timing.spread} ms]  {throughput}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=Path, default=None, help="Clip to measure. Defaults to a generated one."
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per measurement.")
    parser.add_argument("--markdown", action="store_true", help="Emit a markdown table.")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    video = ensure_video(args.video)
    timings, facts = run(video, args.repeats)
    render(timings, facts, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
