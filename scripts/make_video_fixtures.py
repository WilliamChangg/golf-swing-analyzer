#!/usr/bin/env python3
"""Generate the committed video test fixtures.

The fixtures under `python/tests/fixtures/video/` are committed to git rather
than generated at test time. They are a few kilobytes each, and committing them
makes the ingestion tests hermetic: they do not depend on ffmpeg being installed
on the test machine, nor on a particular ffmpeg version choosing to write a
display matrix the way this one does.

This script is how they were produced, and re-running it regenerates them. It is
not run by the test suite or by CI.

Usage:
    uv run --project python python scripts/make_video_fixtures.py
    uv run --project python python scripts/make_video_fixtures.py --verify
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "python" / "tests" / "fixtures" / "video"

# Small enough that all fixtures together are a few tens of kilobytes, large
# enough that a decoded frame is still a real image.
WIDTH, HEIGHT = 320, 240
RATE = 30
DURATION_S = 2

# Deterministic encoder settings. Without these the output bytes vary between
# runs and machines, which would make every regeneration a spurious diff.
_ENCODE = [
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "30",
    "-pix_fmt",
    "yuv420p",
    "-g",
    "15",
    # x264 embeds its version and options in the bitstream; stripping them keeps
    # the file reproducible across encoder builds.
    "-x264-params",
    "log-level=none",
    "-fflags",
    "+bitexact",
    "-flags:v",
    "+bitexact",
    "-movflags",
    "+faststart",
]

_SOURCE = f"testsrc2=size={WIDTH}x{HEIGHT}:rate={RATE}:duration={DURATION_S}"


def _run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(  # noqa: S603
        list(args), capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(args)}\n{proc.stderr}")
    return proc


def _ffmpeg(*args: str) -> None:
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args])


def _probe(path: Path) -> dict[str, object]:
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    result: dict[str, object] = json.loads(proc.stdout)
    return result


def build_cfr(path: Path) -> None:
    """Constant frame rate: 30 fps, 2 s, 60 frames."""
    _ffmpeg("-f", "lavfi", "-i", _SOURCE, "-fps_mode", "cfr", *_ENCODE, str(path))


def build_vfr(path: Path) -> None:
    """Variable frame rate: 30 fps for the first second, 15 fps for the second.

    A rate change part-way through a clip, which is what a phone does when it
    drops frame rate in low light. `-fps_mode vfr` keeps the original
    presentation timestamps instead of resampling to a constant rate, so the
    gap is preserved in the container.
    """
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        _SOURCE,
        "-vf",
        f"select='if(lt(n,{RATE}),1,not(mod(n,2)))'",
        "-fps_mode",
        "vfr",
        *_ENCODE,
        str(path),
    )


def build_rotated(path: Path, ccw_degrees: int) -> None:
    """A clip carrying a display matrix, as phone recordings do.

    The frames themselves are stored unrotated -- byte-identical to the CFR
    fixture, since this is a stream copy -- and only the container's transform
    says how they should be presented. Decoders disagree about whether to apply
    it, which is the trap this fixture exists to test.

    `ccw_degrees` is counter-clockwise, matching ffmpeg's convention. That was
    verified by pixel comparison rather than read off the documentation: a clip
    built with `-display_rotation 90` and then decoded with ffmpeg's default
    autorotation comes out equal to `cv2.ROTATE_90_COUNTERCLOCKWISE` applied to
    the unrotated source. `verify()` re-checks the metadata half of that claim,
    and `tests/test_reader.py` re-checks the pixel half.
    """
    source = path.with_name("_tmp_upright.mp4")
    build_cfr(source)
    try:
        # -display_rotation is an input option: it sets the rotation on the
        # stream being read, which the mov muxer then writes as a tkhd matrix.
        _ffmpeg(
            "-display_rotation",
            str(ccw_degrees),
            "-i",
            str(source),
            "-c",
            "copy",
            "-fflags",
            "+bitexact",
            "-movflags",
            "+faststart",
            str(path),
        )
    finally:
        source.unlink(missing_ok=True)


def build_audio_only(path: Path) -> None:
    """A media file with no video stream at all."""
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-c:a",
        "aac",
        "-b:a",
        "32k",
        "-fflags",
        "+bitexact",
        str(path),
    )


FIXTURES: tuple[tuple[str, str], ...] = (
    ("cfr_30fps.mp4", "constant 30 fps, 60 frames, no rotation"),
    ("vfr_30_to_15fps.mp4", "variable frame rate: 30 fps then 15 fps"),
    ("rotated_ccw90.mp4", "display matrix requesting 90 degrees counter-clockwise"),
    ("rotated_180.mp4", "display matrix requesting 180 degrees"),
    ("audio_only.m4a", "no video stream"),
)

# Expected counter-clockwise display rotation for each rotated fixture.
_ROTATED: tuple[tuple[str, int], ...] = (("rotated_ccw90.mp4", 90), ("rotated_180.mp4", 180))


def generate() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    build_cfr(FIXTURE_DIR / "cfr_30fps.mp4")
    build_vfr(FIXTURE_DIR / "vfr_30_to_15fps.mp4")
    for name, ccw in _ROTATED:
        build_rotated(FIXTURE_DIR / name, ccw)
    build_audio_only(FIXTURE_DIR / "audio_only.m4a")


def verify() -> int:
    """Check each fixture still has the property it was built to exercise.

    Run after regenerating: an ffmpeg upgrade that silently stopped writing the
    display matrix would otherwise turn the rotation tests into tests of nothing.
    """
    failures: list[str] = []

    for name, description in FIXTURES:
        path = FIXTURE_DIR / name
        if not path.exists():
            failures.append(f"{name}: missing")
            continue
        print(f"{name:24} {path.stat().st_size:>7} bytes  {description}")

    def video_stream(name: str) -> dict[str, object]:
        streams = _probe(FIXTURE_DIR / name).get("streams", [])
        assert isinstance(streams, list)  # noqa: S101
        for stream in streams:
            if stream.get("codec_type") == "video":
                assert isinstance(stream, dict)  # noqa: S101
                return stream
        raise SystemExit(f"{name}: no video stream")

    for name, expected in _ROTATED:
        stream = video_stream(name)
        side_data = stream.get("side_data_list") or []
        assert isinstance(side_data, list)  # noqa: S101
        rotations = [d.get("rotation") for d in side_data if "rotation" in d]
        if not rotations:
            failures.append(f"{name}: no display matrix was written by this ffmpeg")
            continue
        ccw = round(float(rotations[0])) % 360  # ffprobe reports counter-clockwise
        if ccw != expected:
            failures.append(f"{name}: expected {expected} degrees ccw, got {ccw}")

    streams = _probe(FIXTURE_DIR / "audio_only.m4a").get("streams", [])
    assert isinstance(streams, list)  # noqa: S101
    if any(s.get("codec_type") == "video" for s in streams):
        failures.append("audio_only.m4a: unexpectedly contains a video stream")

    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only check the existing fixtures; do not regenerate them.",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe must be on PATH to build the fixtures.")

    if not args.verify:
        generate()
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
