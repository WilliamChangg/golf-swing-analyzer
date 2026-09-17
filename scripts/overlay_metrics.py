#!/usr/bin/env python3
"""Draw each measured metric on the frame it was measured from.

A biomechanics engine that returns "shoulder turn 50.2 degrees" is impossible to
check from its output. The number is plausible for almost any swing, so a sign
error, a swapped landmark or a baseline taken from the wrong frame all produce
something that looks exactly as right as the truth does. The only way to know is
to put the measurement back on the picture it came from and look at it.

This renders the anchor frames -- address, top, impact -- with the segments each
metric was computed from drawn on them and the values written alongside, plus a
CSV of every metric with its confidence factors and source frames.

It earned its place immediately: the first run of the equivalent in Phase 4
found the shoulder angle wrapping between +-180, and the first pass of this one
showed the shoulder-turn baseline being set by a post-impact frame where the
estimator had the player 12% wider than they can be.

Usage:
    uv run --project python python scripts/overlay_metrics.py data/face-on/swing.mp4
    uv run --project python python scripts/overlay_metrics.py <clip> --window 0.15
    uv run --project python python scripts/overlay_metrics.py <clip> --out-dir /tmp/debug
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import cv2  # type: ignore[import-untyped]  # noqa: E402

from analyzer.biomechanics import compute_metrics  # noqa: E402
from analyzer.biomechanics.anchors import build_anchors  # noqa: E402
from analyzer.biomechanics.body import Body, body_from  # noqa: E402
from analyzer.contracts.filtering import FilterConfig, SmoothingConfig  # noqa: E402
from analyzer.contracts.metrics import MetricSet  # noqa: E402
from analyzer.contracts.pose import Landmark  # noqa: E402
from analyzer.coordinates import frame_widths_to_pixels  # noqa: E402
from analyzer.filtering.landmarks import filter_sequence  # noqa: E402
from analyzer.ingestion.probe import probe  # noqa: E402
from analyzer.ingestion.reader import OpenCVFrameSource  # noqa: E402
from analyzer.paths import cache_dir  # noqa: E402
from analyzer.phases import detect_phases  # noqa: E402
from analyzer.pose.estimator import resolve_model  # noqa: E402
from analyzer.pose.store import read_sequence  # noqa: E402

# BGR, since OpenCV. Chosen to stay apart in greyscale: these get pasted into
# issues at least as often as they are looked at in colour.
SPINE = (60, 160, 240)
SHOULDERS = (200, 120, 40)
HIPS = (80, 200, 120)
LEGS = (200, 200, 60)
ARMS = (160, 90, 220)
HANDS = (60, 60, 230)
TEXT = (250, 250, 250)
SHADOW = (20, 20, 20)


def resolve_poses(path: Path, model: str | None) -> Path:
    """Find the landmarks for a video, or accept a Parquet path directly."""
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


def to_pixels(body: Body, point: np.ndarray) -> tuple[int, int] | None:
    """Frame widths back to pixels, for drawing.

    The inverse of the conversion the whole engine runs on, which makes it the
    one useful check on that conversion: if the aspect handling were wrong, the
    skeleton drawn here would not sit on the person.
    """
    if not np.all(np.isfinite(point)):
        return None
    pixels = frame_widths_to_pixels(point, body.geometry)
    return int(round(float(pixels[0]))), int(round(float(pixels[1])))


def segment(image: np.ndarray, body: Body, a: np.ndarray, b: np.ndarray, colour, width: int = 3):
    start, end = to_pixels(body, a), to_pixels(body, b)
    if start is None or end is None:
        return
    cv2.line(image, start, end, colour, width, cv2.LINE_AA)
    for place in (start, end):
        cv2.circle(image, place, width + 2, colour, -1, cv2.LINE_AA)


def label(image: np.ndarray, text: str, origin: tuple[int, int], colour=TEXT, scale: float = 0.6):
    """Text with a dark outline, so it stays readable over any footage."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, scale, SHADOW, 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, scale, colour, 1, cv2.LINE_AA)


def draw_skeleton(image: np.ndarray, body: Body, frame: int) -> None:
    """The segments every metric in this phase is computed from, and only those."""
    points = {name: body.points[name][frame] for name in body.points}

    segment(image, body, body.hip_mid[frame], body.shoulder_mid[frame], SPINE, 4)
    segment(
        image,
        body,
        points[Landmark.LEFT_SHOULDER],
        points[Landmark.RIGHT_SHOULDER],
        SHOULDERS,
        4,
    )
    segment(image, body, points[Landmark.LEFT_HIP], points[Landmark.RIGHT_HIP], HIPS, 4)

    for hip, knee, ankle in (
        (Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
        (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    ):
        segment(image, body, points[hip], points[knee], LEGS)
        segment(image, body, points[knee], points[ankle], LEGS)

    for shoulder, elbow, wrist in (
        (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
        (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
    ):
        segment(image, body, points[shoulder], points[elbow], ARMS)
        segment(image, body, points[elbow], points[wrist], ARMS)

    hand = to_pixels(body, body.hand.position[frame])
    if hand is not None and body.hand.valid[frame]:
        cv2.circle(image, hand, 9, HANDS, 2, cv2.LINE_AA)

    # The player's left marked explicitly. Anatomical left and the left of the
    # frame are opposite sides for a player facing the camera, and every sign
    # convention in this engine turns on not confusing them.
    left_shoulder = to_pixels(body, points[Landmark.LEFT_SHOULDER])
    if left_shoulder is not None:
        label(image, "L", (left_shoulder[0] + 10, left_shoulder[1] - 8), SHOULDERS, 0.7)


def annotate(image: np.ndarray, result: MetricSet, frame: int, title: str) -> None:
    """Every metric whose source frames include this one, written on it."""
    label(image, title, (16, 34), TEXT, 0.85)

    here = [metric for metric in result.metrics if frame in metric.source_frames]
    y = 66
    for metric in here:
        value = f"{metric.value:+.1f} deg" if metric.unit == "degrees" else f"{metric.value:+.3f}"
        if metric.unit == "seconds":
            value = f"{metric.value:.3f} s"
        elif metric.unit == "ratio":
            value = f"{metric.value:.2f}:1"
        label(image, f"{metric.label}: {value}  (conf {metric.confidence.overall:.2f})", (16, y))
        y += 24

    if result.view is not None:
        label(
            image,
            f"view: {result.view.view.value} "
            f"(shoulders {result.view.shoulder_span_ratio:.2f} torso at address)",
            (16, y + 8),
            SHOULDERS,
        )
        y += 24
    if result.lead_side is not None and result.lead_side.side is not None:
        label(image, f"lead side: {result.lead_side.side.value}", (16, y + 8), SHOULDERS)


def write_csv(destination: Path, result: MetricSet) -> None:
    """Every metric with its factors, which is what gets attached to a bug report."""
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "label",
                "value",
                "unit",
                "basis",
                "event",
                "phase",
                "confidence",
                "observation",
                "anchor",
                "method",
                "source_frames",
                "view",
                "interpretation",
                "methodology",
            ]
        )
        for metric in result.metrics:
            factors = metric.confidence
            writer.writerow(
                [
                    metric.name.value,
                    metric.label,
                    f"{metric.value:.6f}",
                    metric.unit.value,
                    metric.basis.value,
                    metric.event.value if metric.event else "",
                    metric.phase.value if metric.phase else "",
                    f"{factors.overall:.4f}",
                    f"{factors.observation:.4f}",
                    f"{factors.anchor:.4f}",
                    f"{factors.method:.4f}",
                    " ".join(str(frame) for frame in metric.source_frames),
                    metric.view.value,
                    metric.interpretation,
                    metric.methodology,
                ]
            )

        writer.writerow([])
        writer.writerow(["refused", "reason"])
        for entry in result.refused:
            writer.writerow([entry.name.value, entry.reason])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Video file, or a pose Parquet file.")
    parser.add_argument("--model", default=None, help="Which extraction to use.")
    parser.add_argument("--window", type=float, default=None, help="Filtering window in seconds.")
    parser.add_argument("--polyorder", type=int, default=None, help="Filter polynomial degree.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the output.")
    args = parser.parse_args()

    smoothing = SmoothingConfig(
        **{
            key: value
            for key, value in (("window_s", args.window), ("polyorder", args.polyorder))
            if value is not None
        }
    )
    config = FilterConfig(smoothing=smoothing)

    sequence = read_sequence(resolve_poses(args.path, args.model))
    filtered = filter_sequence(sequence, config)
    phases = detect_phases(filtered)
    result = compute_metrics(filtered, phases)

    out_dir = args.out_dir or (REPO_ROOT / "build" / "metrics" / args.path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "metrics.csv", result)
    print(f"wrote {out_dir / 'metrics.csv'}")

    if not result.computed:
        for warning in result.warnings:
            print(f"! {warning}")
        return 1

    body = body_from(filtered)
    anchors = build_anchors(phases)
    wanted = {
        name: anchor
        for name, anchor in (
            ("address", anchors.address),
            ("top", anchors.top),
            ("impact", anchors.impact),
        )
        if anchor is not None
    }

    video = Path(sequence.video_path)
    if not video.exists():
        print(f"! {video} has moved, so no frames could be drawn.")
        return 0

    # The address frame drawn is the last of the phase rather than its median.
    # A median over an interval has no frame of its own, and the last one is
    # both a real picture and the closest to the motion starting.
    chosen = {name: anchor.frames[-1] for name, anchor in wanted.items()}

    with OpenCVFrameSource(probe(video)) as source:
        for name, frame in chosen.items():
            image = source.frame_at(frame).image.copy()
            draw_skeleton(image, body, frame)
            annotate(image, result, frame, f"{name} - frame {frame}")
            destination = out_dir / f"{name}_{frame:04d}.png"
            cv2.imwrite(str(destination), image)
            print(f"wrote {destination}")

    for warning in result.warnings:
        print(f"! {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
