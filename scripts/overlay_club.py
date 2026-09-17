#!/usr/bin/env python3
"""Draw the tracked shaft on the frames it was found in, and the gaps where it was not.

A club tracker that returns "shaft at -37 degrees, confidence 0.82" is impossible
to check from its output. Every plausible angle is plausible, so following a door
frame, reading the shaft backwards and tracking the club correctly all produce
numbers that look equally right. The only way to know is to put the line back on
the picture it came from and look at it.

This is `overlay_metrics.py`'s counterpart for Phase 10, and it is the
deliverable for "overlay rendering": the app's club canvas is Phase 14.5, and it
should draw something that has already been checked rather than being the thing
that checks it. The same reasoning kept Phase 5 and Phase 9 out of the UI.

**The gaps are the point.** Most of what this phase produces on real footage is
refusals, so a frame with no shaft is written too, stamped with the reason -- a
contact sheet of only the successes would show a tracker that works.

    green    tracked, confident
    amber    tracked, marginal
    red      refused, with the reason
    dotted   the club head was not reached, so the far end is where the
             evidence stopped rather than where the club ends

Usage:
    uv run --project python python scripts/overlay_club.py data/face-on/swing.mp4
    uv run --project python python scripts/overlay_club.py <clip> --slow-motion 8
    uv run --project python python scripts/overlay_club.py <clip> --every 4 --out-dir /tmp/club
    uv run --project python python scripts/overlay_club.py <clip> --video
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import cv2  # type: ignore[import-untyped]
from analyzer.club import HoughShaftDetector, track_club
from analyzer.contracts.club import (
    ClubConfig,
    ClubFrame,
    ClubTrackingReport,
)
from analyzer.contracts.filtering import FilterConfig, SmoothingConfig
from analyzer.contracts.phases import PhaseConfig
from analyzer.contracts.pose import LandmarkSpace
from analyzer.coordinates import frame_widths_to_pixels
from analyzer.filtering.landmarks import filter_sequence
from analyzer.ingestion.probe import probe
from analyzer.ingestion.reader import OpenCVFrameSource
from analyzer.paths import cache_dir
from analyzer.phases import SignalError, detect_phases
from analyzer.pose.estimator import resolve_model
from analyzer.pose.store import read_sequence

# BGR, since OpenCV. Chosen to stay apart in greyscale: these get pasted into
# issues at least as often as they are looked at in colour.
CONFIDENT = (90, 210, 90)
MARGINAL = (60, 190, 240)
REFUSED = (70, 70, 230)
RIVAL = (200, 120, 200)
TRAIL = (230, 180, 80)
GRIP = (240, 240, 240)

# Below this the shaft is drawn amber rather than green. It is a presentation
# threshold and nothing computes from it -- `ClubConfig.min_confidence` is what
# decides whether a frame is emitted at all.
MARGINAL_BELOW = 0.6


def draw_shaft(image: np.ndarray, entry: ClubFrame, report: ClubTrackingReport) -> None:
    """One frame's shaft, or a stamp saying why there is none."""
    height = image.shape[0]
    shaft = entry.shaft
    if shaft is None:
        reason = entry.refusal.value if entry.refusal else "unknown"
        cv2.putText(
            image,
            f"no shaft: {reason}",
            (16, height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            REFUSED,
            2,
            cv2.LINE_AA,
        )
        return

    points = frame_widths_to_pixels(
        np.array([[shaft.grip_x, shaft.grip_y], [shaft.tip_x, shaft.tip_y]]),
        report.geometry,
    )
    grip = (int(round(points[0, 0])), int(round(points[0, 1])))
    tip = (int(round(points[1, 0])), int(round(points[1, 1])))
    colour = CONFIDENT if shaft.confidence.overall >= MARGINAL_BELOW else MARGINAL

    if shaft.reaches_head:
        cv2.line(image, grip, tip, colour, 3, cv2.LINE_AA)
        cv2.circle(image, tip, 8, colour, 2, cv2.LINE_AA)
    else:
        # Dotted, because the far end is where the evidence stopped rather than
        # where the club ends. Drawing it solid would make a shaft direction look
        # like a club-head position, which is the single most misleading thing
        # this overlay could do.
        _dotted(image, grip, tip, colour)
    cv2.circle(image, grip, 6, GRIP, -1, cv2.LINE_AA)

    if shaft.runner_up_angle_deg is not None:
        # What the tracker was choosing between. A frame where this sits on a
        # door frame and the accepted line sits on the club is the picture that
        # explains why margin is measured against a rival at all.
        length = float(np.hypot(tip[0] - grip[0], tip[1] - grip[1]))
        radians = np.radians(-shaft.runner_up_angle_deg)  # back into pixel convention
        rival = (
            int(round(grip[0] + length * np.cos(radians))),
            int(round(grip[1] + length * np.sin(radians))),
        )
        cv2.line(image, grip, rival, RIVAL, 1, cv2.LINE_AA)

    lines = [
        f"frame {entry.frame_index}"
        + (f"  {entry.phase.value}" if entry.phase is not None else ""),
        f"shaft {shaft.angle_deg:+.1f} deg"
        + ("" if shaft.reaches_head else "   (head not reached)"),
        f"conf {shaft.confidence.overall:.2f}"
        f"  = sup {shaft.confidence.support:.2f}"
        f" x mar {shaft.confidence.margin:.2f}"
        f" x con {shaft.confidence.continuity:.2f}",
    ]
    if shaft.tip_speed_px_s is not None:
        lines.append(f"head {shaft.tip_speed_px_s:,.0f} px/s")
    for index, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (16, 36 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour if index else GRIP,
            2,
            cv2.LINE_AA,
        )


def _dotted(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    colour: tuple[int, int, int],
) -> None:
    steps = max(2, int(np.hypot(end[0] - start[0], end[1] - start[1]) / 12))
    for step in range(steps):
        if step % 2:
            continue
        a = step / steps
        b = min(1.0, (step + 1) / steps)
        first = (
            int(start[0] + a * (end[0] - start[0])),
            int(start[1] + a * (end[1] - start[1])),
        )
        second = (
            int(start[0] + b * (end[0] - start[0])),
            int(start[1] + b * (end[1] - start[1])),
        )
        cv2.line(image, first, second, colour, 3, cv2.LINE_AA)


def draw_trail(image: np.ndarray, report: ClubTrackingReport, upto: int) -> None:
    """Where the club head has been, over the frames it was actually observed.

    Only the observed ones. Joining across a gap would draw a smooth arc through
    the part of the swing the tracker refused, which is precisely the claim this
    phase spends its effort not making.
    """
    previous: tuple[int, int] | None = None
    for entry in report.frames:
        if entry.frame_index > upto:
            break
        if entry.shaft is None or not entry.shaft.reaches_head:
            previous = None
            continue
        point = frame_widths_to_pixels(
            np.array([entry.shaft.tip_x, entry.shaft.tip_y]), report.geometry
        )
        current = (int(round(float(point[0]))), int(round(float(point[1]))))
        if previous is not None:
            cv2.line(image, previous, current, TRAIL, 2, cv2.LINE_AA)
        previous = current


def write_csv(report: ClubTrackingReport, destination: Path) -> None:
    """Every frame, tracked or not, with the numbers behind the verdict."""
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "time_s",
                "phase",
                "tracked",
                "refusal",
                "angle_deg",
                "length_torso",
                "reaches_head",
                "confidence",
                "support",
                "margin",
                "continuity",
                "runner_up_deg",
                "candidates",
                "rate_deg_s",
                "tip_speed_px_s",
            ]
        )
        for entry in report.frames:
            shaft = entry.shaft
            writer.writerow(
                [
                    entry.frame_index,
                    f"{entry.timestamp_s:.5f}",
                    entry.phase.value if entry.phase else "",
                    int(entry.detected),
                    entry.refusal.value if entry.refusal else "",
                    "" if shaft is None else f"{shaft.angle_deg:.2f}",
                    "" if shaft is None else f"{shaft.length_torso:.3f}",
                    "" if shaft is None else int(shaft.reaches_head),
                    "" if shaft is None else f"{shaft.confidence.overall:.3f}",
                    "" if shaft is None else f"{shaft.confidence.support:.3f}",
                    "" if shaft is None else f"{shaft.confidence.margin:.3f}",
                    "" if shaft is None else f"{shaft.confidence.continuity:.3f}",
                    ""
                    if shaft is None or shaft.runner_up_angle_deg is None
                    else f"{shaft.runner_up_angle_deg:.2f}",
                    "" if shaft is None else shaft.candidates,
                    ""
                    if shaft is None or shaft.angular_rate_deg_s is None
                    else f"{shaft.angular_rate_deg_s:.1f}",
                    ""
                    if shaft is None or shaft.tip_speed_px_s is None
                    else f"{shaft.tip_speed_px_s:.0f}",
                ]
            )


def resolve_poses(video: Path, model: str | None) -> Path:
    entry, _ = resolve_model(model)
    metadata = probe(video).metadata
    poses = (
        cache_dir()
        / "poses"
        / metadata.content_key.as_path_segment()
        / f"{entry.name}.parquet"
    )
    if not poses.exists():
        raise SystemExit(
            f"No extracted poses for {video.name} with model '{entry.name}'.\n"
            f"Run: analyzer extract {video} --model {entry.name}"
        )
    return poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--model", default=None, help="Which extraction supplies the anchors."
    )
    parser.add_argument(
        "--window", type=float, default=None, help="Filtering window in seconds."
    )
    parser.add_argument("--slow-motion", type=float, default=1.0, dest="slow_motion")
    parser.add_argument(
        "--every", type=int, default=1, help="Write every nth frame as a still image."
    )
    parser.add_argument(
        "--video",
        action="store_true",
        dest="as_video",
        help="Write an annotated clip instead of stills. Easier to judge a track from.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    arguments = parser.parse_args()

    video = arguments.video
    if not video.exists():
        raise SystemExit(f"{video} does not exist.")

    out_dir = arguments.out_dir or (REPO_ROOT / "build" / "club" / video.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    smoothing = (
        SmoothingConfig(window_s=arguments.window)
        if arguments.window
        else SmoothingConfig()
    )
    sequence = read_sequence(resolve_poses(video, arguments.model))
    filtered = filter_sequence(
        sequence,
        FilterConfig(smoothing=smoothing),
        space=LandmarkSpace.FRAME_WIDTHS,
        slow_motion_factor=arguments.slow_motion,
    )
    try:
        phases = detect_phases(filtered, PhaseConfig())
    except SignalError:
        phases = None

    report = track_club(
        video, filtered, HoughShaftDetector(), phases=phases, config=ClubConfig()
    )

    write_csv(report, out_dir / "club.csv")
    print(f"wrote {out_dir / 'club.csv'}")
    print(
        f"tracked {report.tracked_frames} of {report.frame_count} frames "
        f"({report.coverage:.0%} clip-wide)"
    )
    for entry in report.phase_coverage:
        print(
            f"  {entry.phase.value:15s} {entry.tracked:4d}/{entry.frames:<4d} {entry.coverage:.0%}"
        )

    with OpenCVFrameSource(probe(video)) as source:
        writer = None
        if arguments.as_video:
            geometry = report.geometry
            writer = cv2.VideoWriter(
                str(out_dir / "club.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, 1.0 / max(1e-6, _median_interval(report))),
                (geometry.width, geometry.height),
            )

        for entry in report.frames:
            if not arguments.as_video and entry.frame_index % max(1, arguments.every):
                continue
            image = source.frame_at(entry.frame_index).image.copy()
            draw_trail(image, report, entry.frame_index)
            draw_shaft(image, entry, report)
            if writer is not None:
                writer.write(image)
            else:
                cv2.imwrite(str(out_dir / f"frame_{entry.frame_index:05d}.png"), image)

        if writer is not None:
            writer.release()
            print(f"wrote {out_dir / 'club.mp4'}")
        else:
            print(f"wrote stills to {out_dir}")

    for note in report.warnings:
        print(f"! {note}")
    return 0 if report.tracked else 1


def _median_interval(report: ClubTrackingReport) -> float:
    times = [entry.timestamp_s for entry in report.frames]
    if len(times) < 2:
        return 1.0 / 30.0
    return float(np.median(np.diff(times)))


if __name__ == "__main__":
    raise SystemExit(main())
