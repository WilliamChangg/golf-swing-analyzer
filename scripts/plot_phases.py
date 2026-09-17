#!/usr/bin/env python3
"""Plot the signals swing detection reads, and dump them per frame.

A detector that returns four frame numbers is impossible to debug from its
output. When the top comes back two frames late there is no way to tell from
`top=38` whether the speed minimum was in the wrong place, the height maximum
was, or the filter smoothed both into something that no longer has a minimum.
This draws the signals the rules actually key on, marks where each event landed,
and writes the same numbers as a CSV so a disagreement can be examined frame by
frame rather than argued about.

The CSV is the more useful half. It is what gets attached to a bug report, and
what Phase 12's labelling tool will compare its ground truth against.

Usage:
    uv run --project python python scripts/plot_phases.py data/face-on/swing.mp4
    uv run --project python python scripts/plot_phases.py <poses.parquet> --window 0.15
    uv run --project python python scripts/plot_phases.py <clip> --out-dir /tmp/debug
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.contracts.filtering import FilterConfig, SmoothingConfig  # noqa: E402
from analyzer.contracts.phases import PhaseConfig, SwingPhases  # noqa: E402
from analyzer.filtering.landmarks import filter_sequence  # noqa: E402
from analyzer.ingestion.probe import probe  # noqa: E402
from analyzer.paths import cache_dir  # noqa: E402
from analyzer.phases import detect_phases, swing_signals  # noqa: E402
from analyzer.phases.signals import SwingSignals  # noqa: E402
from analyzer.pose.estimator import resolve_model  # noqa: E402
from analyzer.pose.store import read_sequence  # noqa: E402

# Colours per phase. Chosen to stay distinguishable in greyscale, since these
# plots get pasted into issues and printed as often as they are viewed.
PHASE_COLOURS = {
    "address": "#dce3ea",
    "backswing": "#a8c6e0",
    "downswing": "#f0b8a8",
    "follow_through": "#c3dbc0",
}

EVENT_COLOURS = {
    "takeaway": "#2f6f9f",
    "top": "#7a4fa3",
    "impact": "#c0392b",
    "finish": "#1e8449",
}


def resolve_poses(path: Path, model: str | None) -> Path:
    """Find the landmarks for a video, or accept a Parquet path directly."""
    if path.suffix == ".parquet":
        return path

    metadata = probe(path).metadata
    entry, _ = resolve_model(model)
    poses = (
        cache_dir()
        / "poses"
        / metadata.content_key.as_path_segment()
        / f"{entry.name}.parquet"
    )
    if not poses.exists():
        raise SystemExit(
            f"No extracted poses for {path.name} with model '{entry.name}'.\n"
            f"Run: uv run --project python analyzer extract {path} --model {entry.name}"
        )
    return poses


def write_csv(destination: Path, signals: SwingSignals, result: SwingPhases) -> None:
    """One row per frame, with every signal a rule looked at.

    Including the ones that did not decide anything: when an event is in the
    wrong place, the question is usually which signal disagreed with which, and
    that cannot be answered from the two the rule happened to use.
    """
    events = {entry.frame_index: entry.event.value for entry in result.events}

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "timestamp_s",
                "phase",
                "event",
                "hand_x",
                "hand_height",
                "hand_speed",
                "visibility",
                "observed",
                "valid",
                "shoulder_angle_deg",
                "hip_angle_deg",
            ]
        )

        def number(value: float) -> str:
            return "" if not np.isfinite(value) else f"{value:.6f}"

        for index in range(len(signals)):
            phase = result.phase_at(index)
            writer.writerow(
                [
                    index,
                    f"{signals.t[index]:.6f}",
                    phase.value if phase else "",
                    events.get(index, ""),
                    number(signals.hand.position[index, 0]),
                    number(signals.height[index]),
                    number(signals.speed[index]),
                    number(signals.hand.visibility[index]),
                    int(signals.hand.observed[index]),
                    int(signals.hand.valid[index]),
                    number(signals.shoulder_angle_deg[index]),
                    number(signals.hip_angle_deg[index]),
                ]
            )


def write_plot(
    destination: Path, signals: SwingSignals, result: SwingPhases, title: str
) -> None:
    """Three stacked panels sharing a time axis."""
    import matplotlib

    # Chosen before pyplot is imported: this runs headless in CI and from a
    # terminal, and the default backend would try to open a window.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    figure.suptitle(title, fontsize=12)

    for interval in result.phases:
        colour = PHASE_COLOURS.get(interval.phase.value, "#eeeeee")
        for axis in axes:
            axis.axvspan(interval.start_s, interval.end_s, color=colour, zorder=0)

    for entry in result.events:
        colour = EVENT_COLOURS.get(entry.event.value, "#444444")
        for axis in axes:
            axis.axvline(entry.timestamp_s, color=colour, linewidth=1.4, zorder=3)
        axes[0].annotate(
            f"{entry.event.value}\n{entry.confidence.overall:.2f}",
            xy=(entry.timestamp_s, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -12),
            textcoords="offset points",
            fontsize=8,
            color=colour,
        )

    axes[0].plot(signals.t, signals.speed, color="#222222", linewidth=1.4)
    axes[0].set_ylabel("hand speed\n(widths/s)")

    # The threshold the takeaway and finish are found by, drawn so a crossing
    # that looks wrong can be checked against the line that produced it.
    if np.isfinite(signals.peak_speed):
        axes[0].axhline(
            result.config.moving_fraction * signals.peak_speed,
            color="#888888",
            linestyle="--",
            linewidth=1.0,
        )

    axes[1].plot(signals.t, signals.height, color="#222222", linewidth=1.4)
    axes[1].set_ylabel("hand height\n(upward)")

    axes[2].plot(
        signals.t,
        signals.shoulder_angle_deg,
        color="#2f6f9f",
        linewidth=1.2,
        label="shoulders",
    )
    axes[2].plot(
        signals.t, signals.hip_angle_deg, color="#c0392b", linewidth=1.2, label="hips"
    )
    axes[2].set_ylabel("projected\nangle (deg)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right", fontsize=8)

    # Where the estimator saw nothing. Drawn on every panel because an event
    # near one of these is an event to distrust.
    untracked = ~signals.hand.observed
    for axis in axes:
        axis.fill_between(
            signals.t,
            0,
            1,
            where=untracked,
            transform=axis.get_xaxis_transform(),
            color="#000000",
            alpha=0.10,
            zorder=1,
            step="mid",
        )
        axis.grid(alpha=0.25, zorder=2)

    figure.tight_layout()
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Video, or a pose Parquet file.")
    parser.add_argument("--model", help="Which extraction to use, when given a video.")
    parser.add_argument("--window", type=float, help="Filtering window in seconds.")
    parser.add_argument("--polyorder", type=int, help="Degree of the local polynomial.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.cwd(),
        help="Where to write the plot and CSV.",
    )
    args = parser.parse_args()

    smoothing: dict[str, float | int] = {}
    if args.window is not None:
        smoothing["window_s"] = args.window
    if args.polyorder is not None:
        smoothing["polyorder"] = args.polyorder
    config = FilterConfig(smoothing=SmoothingConfig(**smoothing))  # type: ignore[arg-type]

    poses = resolve_poses(args.path, args.model)
    sequence = read_sequence(poses)
    filtered = filter_sequence(sequence, config)
    signals = swing_signals(filtered)
    result = detect_phases(filtered, PhaseConfig())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(sequence.video_path).stem
    plot_path = args.out_dir / f"{stem}-phases.png"
    csv_path = args.out_dir / f"{stem}-phases.csv"

    title = (
        f"{stem} — {'swing detected' if result.detected else 'NO SWING DETECTED'} — "
        f"window {config.smoothing.window_s:g}s, degree {config.smoothing.polyorder}, "
        f"hand from {signals.hand.source.value}"
    )
    write_plot(plot_path, signals, result, title)
    write_csv(csv_path, signals, result)

    print(f"wrote {plot_path}")
    print(f"wrote {csv_path}")

    for entry in result.events:
        print(
            f"  {entry.event.value:9s} frame {entry.frame_index:4d}  "
            f"t={entry.timestamp_s:6.3f}s  confidence {entry.confidence.overall:.2f}"
        )
    for warning in result.warnings:
        print(f"  ! {warning}")

    return 0 if result.detected else 1


if __name__ == "__main__":
    raise SystemExit(main())
