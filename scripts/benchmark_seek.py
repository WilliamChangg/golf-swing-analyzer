#!/usr/bin/env python3
"""Measure whether a seek by time lands on the frame it was asked for.

Phase 14 puts a video behind frame numbers that Phases 4 to 13 produce, and a
video element cannot be asked for a frame -- only for a time. So the question
this script exists to answer is not "is the arithmetic right", which is a test,
but "does a decoder agree", which is a measurement.

Two maps are compared over every frame of every clip given:

    naive     frame / nominal_fps, the container's declared average rate
    measured  `SeekIndex.seek_targets_s`, the midpoint of the interval the
              frame is actually displayed for

and each is checked against the presentation timestamps the probe read. That is
a statement about the map, and it is the headline table.

**A second table asks OpenCV to perform the seeks, and it does not adjudicate
anything -- it measures OpenCV.** That was worth finding out rather than
assuming: `cv2.VideoCapture.set(CAP_PROP_POS_MSEC)` lands a frame early on a
sizeable minority of seeks once the decoder has been read from, which is the
same inaccuracy Phase 1 already found and works around with `_scan_to` when
seeking by index. A reference implementation that is itself wrong cannot tell
two maps apart, and on three of the five clips here it does not.

It is still printed, for two reasons. It is the evidence for the paragraph
above, and where OpenCV's seek *is* reliable the measured map is dramatically
better through it -- which is what a midpoint target is for: a decoder that
rounds has somewhere to round to.

The real verification is a real player, because a player is the consumer. That
is `e2e/player.spec.ts`, which drives a Chromium video element and reads back
the presentation time of the frame it actually displayed.

Usage:
    uv run --project python python scripts/benchmark_seek.py
    uv run --project python python scripts/benchmark_seek.py --video path/to/clip.mp4
    uv run --project python python scripts/benchmark_seek.py --markdown
"""

from __future__ import annotations

import argparse
import bisect
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.contracts.video import SeekIndex  # noqa: E402
from analyzer.ingestion import probe_video, seek_index  # noqa: E402

# The clips this repository can speak about. Two fixtures chosen for their
# timing rather than their content, and the reference footage, because the
# interesting failures turned out to be on clips nobody would suspect.
DEFAULT_CLIPS = (
    "python/tests/fixtures/video/cfr_30fps.mp4",
    "python/tests/fixtures/video/vfr_30_to_15fps.mp4",
    "data/amateur/face-on/PW_face-on.mp4",
    "data/amateur/dtl/iron_dtl.mp4",
    "data/rory/face-on/rory_face_on.mp4",
)


@dataclass(frozen=True)
class Verdict:
    """How one map performed on one clip."""

    name: str
    frames: int
    missed: int
    max_error: int

    @property
    def rate(self) -> float:
        return self.missed / self.frames if self.frames else 0.0


def displayed_frame(timestamps: list[float], at: float) -> int:
    """Which frame is on screen at `at`, by the rule every player implements.

    The last frame whose presentation time has been reached. Clamped at zero
    because seeking before the first frame shows the first frame rather than
    an error.
    """
    return max(bisect.bisect_right(timestamps, at) - 1, 0)


def against_the_index(index: SeekIndex, targets: list[float]) -> Verdict:
    """Score a map against the timestamps the probe read.

    A statement about the model and not about any decoder. For the measured map
    this is close to a tautology -- the targets are midpoints of these very
    intervals -- and it is reported anyway, because it is the number the naive
    map fails on, and a reader comparing the two needs both computed the same
    way.
    """
    errors = [
        abs(displayed_frame(index.timestamps_s, target) - frame)
        for frame, target in enumerate(targets)
    ]
    return Verdict(
        name="model",
        frames=len(targets),
        missed=sum(1 for error in errors if error),
        max_error=max(errors, default=0),
    )


def against_a_decoder(clip: Path, index: SeekIndex, targets: list[float]) -> Verdict | None:
    """Score a map by asking a decoder to seek to each target.

    Seeks by *time*, with `CAP_PROP_POS_MSEC`, because that is what a player
    does -- not `frame_at`, which seeks by index and would be measuring a
    different question. The frame it landed on is read back from the position
    the decoder reports, which is the decoder's own opinion and the only one
    available.

    Returns None when OpenCV cannot open the clip, rather than failing the run:
    a backend that will not read a file is a fact about the backend, and the
    model column above is still worth printing.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - opencv is a declared dependency
        return None

    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        return None

    errors: list[int] = []
    try:
        for frame, target in enumerate(targets):
            capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
            position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ok, _ = capture.read()
            if not ok:
                # Past the end of what this decoder will produce. Counted as a
                # miss rather than skipped: a player that cannot reach the frame
                # cannot show it, whatever the arithmetic says.
                errors.append(abs(index.frame_count - 1 - frame))
                continue
            errors.append(abs(displayed_frame(index.timestamps_s, position) - frame))
    finally:
        capture.release()

    return Verdict(
        name="decoder",
        frames=len(errors),
        missed=sum(1 for error in errors if error),
        max_error=max(errors, default=0),
    )


def measure(clip: Path) -> dict[str, object]:
    """Both maps, both ways, for one clip."""
    index = seek_index(clip)
    metadata = probe_video(clip)
    nominal = metadata.timing.nominal_fps

    naive = (
        [frame / nominal for frame in range(index.frame_count)] if nominal else None
    )
    measured = index.seek_targets_s

    return {
        "clip": clip,
        "frames": index.frame_count,
        "is_vfr": metadata.timing.is_vfr,
        "nominal_fps": nominal,
        "measured_fps": metadata.timing.measured_fps,
        "naive_model": against_the_index(index, naive) if naive else None,
        "naive_decoder": against_a_decoder(clip, index, naive) if naive else None,
        "measured_model": against_the_index(index, measured),
        "measured_decoder": against_a_decoder(clip, index, measured),
    }


def _cell(verdict: Verdict | None) -> str:
    if verdict is None:
        return "—"
    return f"{verdict.missed} / {verdict.frames}" + (
        f" (±{verdict.max_error})" if verdict.missed else ""
    )


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]], *, markdown: bool) -> None:
    if markdown:
        print("| " + " | ".join(header) + " |")
        print("| " + " | ".join("---" for _ in header) + " |")
        for row in rows:
            print("| " + " | ".join(row) + " |")
        return

    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]
    print("  ".join(name.ljust(widths[i]) for i, name in enumerate(header)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def report(results: list[dict[str, object]], *, markdown: bool) -> None:
    """Print both tables. Markdown for pasting into the README, plain otherwise."""
    rows = []
    for entry in results:
        clip = entry["clip"]
        assert isinstance(clip, Path)  # noqa: S101 - narrows for the name below
        nominal = entry["nominal_fps"]
        measured_fps = entry["measured_fps"]
        rows.append(
            (
                clip.name,
                str(entry["frames"]),
                {True: "yes", False: "no", None: "unknown"}[entry["is_vfr"]],  # type: ignore[index]
                f"{nominal:.3f}" if isinstance(nominal, float) else "—",
                f"{measured_fps:.3f}" if isinstance(measured_fps, float) else "—",
                _cell(entry["naive_model"]),  # type: ignore[arg-type]
                _cell(entry["measured_model"]),  # type: ignore[arg-type]
            )
        )

    print("## Which frame the map asks for, against the timestamps the container carries\n")
    _table(
        (
            "clip",
            "frames",
            "vfr",
            "declared fps",
            "measured fps",
            "frame / declared fps",
            "measured midpoints",
        ),
        rows,
        markdown=markdown,
    )

    decoder_rows = [
        (
            entry["clip"].name,  # type: ignore[union-attr]
            _cell(entry["naive_decoder"]),  # type: ignore[arg-type]
            _cell(entry["measured_decoder"]),  # type: ignore[arg-type]
        )
        for entry in results
        if entry["measured_decoder"] is not None
    ]
    if not decoder_rows:
        return

    print("\n## The same targets, seeked by OpenCV. This measures OpenCV.\n")
    _table(
        ("clip", "frame / declared fps", "measured midpoints"),
        decoder_rows,
        markdown=markdown,
    )
    print(
        "\nMisses here are dominated by `CAP_PROP_POS_MSEC` landing a frame early, which it\n"
        "does whether the target is right or wrong -- so this table cannot tell the two maps\n"
        "apart, and on most clips it does not. Where the seek is reliable the midpoints win\n"
        "by a wide margin, which is the argument for them: a decoder that rounds has\n"
        "somewhere to round to. A real player is measured in e2e/player.spec.ts."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        action="append",
        type=Path,
        help="A clip to measure. Repeatable. Defaults to the fixtures and reference footage.",
    )
    parser.add_argument("--markdown", action="store_true", help="Emit a Markdown table.")
    args = parser.parse_args()

    clips = args.video or [REPO_ROOT / name for name in DEFAULT_CLIPS]
    present = [clip for clip in clips if clip.exists()]
    for clip in clips:
        if not clip.exists():
            print(f"skipped (not present): {clip}", file=sys.stderr)
    if not present:
        print("No clips to measure.", file=sys.stderr)
        return 1

    print(f"# {platform.platform()} / OpenCV seek, {len(present)} clip(s)\n")
    results = [measure(clip) for clip in present]
    report(results, markdown=args.markdown)

    missed = sum(
        entry["measured_model"].missed  # type: ignore[union-attr]
        for entry in results
        if isinstance(entry["measured_model"], Verdict)
    )
    if missed:
        # Would mean the midpoints do not sit inside their own intervals, which
        # is an arithmetic failure rather than a decoder one. Loud, because the
        # table above would otherwise report it as a quiet column of numbers.
        print(
            f"\nThe measured map missed {missed} frame(s) against its own timestamps. "
            "That is a bug in seek_index, not a property of any clip.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
