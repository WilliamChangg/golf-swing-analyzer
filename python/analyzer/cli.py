"""Command-line entry point.

Shares the dispatch table with the RPC worker, so the analysis engine is fully
usable from a terminal without the desktop app. That matters for debugging, for
CI, and for benchmark scripts, none of which should need Rust in the loop.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from analyzer import __version__
from analyzer.calibration.apply import distortion_displacement
from analyzer.contracts.ball import BallTrackingReport
from analyzer.contracts.calibration import (
    BoardFamily,
    BoardSpec,
    CalibrationQuality,
    CameraCalibration,
    CameraRig,
    StereoCalibration,
)
from analyzer.contracts.club import ClubTrackingReport
from analyzer.contracts.coaching import CoachingReport, Comparison, FindingRefusal
from analyzer.contracts.filtering import SequenceFilterReport
from analyzer.contracts.health import EnvironmentReport, HealthStatus
from analyzer.contracts.impact import FusedImpact
from analyzer.contracts.metrics import (
    CameraView,
    Metric,
    MetricBasis,
    MetricGroup,
    MetricSet,
    MetricUnit,
)
from analyzer.contracts.phases import SwingPhase, SwingPhases
from analyzer.contracts.pose import PoseExtractionResult
from analyzer.contracts.progress import ProgressUpdate
from analyzer.contracts.projects import Project, ProjectList
from analyzer.contracts.reconstruction import ReconstructionReport
from analyzer.contracts.rpc import EngineError
from analyzer.contracts.sync import SyncModel
from analyzer.contracts.video import TimestampSource, VideoMetadata
from analyzer.dispatch import call
from analyzer.progress import CallbackReporter

app = typer.Typer(
    name="analyzer",
    help="Golf swing biomechanics analysis engine.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()

# Colours are chosen so status is still distinguishable without colour, because
# the same output is read from CI logs.
_STATUS_STYLE: dict[HealthStatus, str] = {
    HealthStatus.OK: "green",
    HealthStatus.DEGRADED: "yellow",
    HealthStatus.MISSING: "red",
    HealthStatus.ERROR: "bold red",
}

_STATUS_GLYPH: dict[HealthStatus, str] = {
    HealthStatus.OK: "OK",
    HealthStatus.DEGRADED: "DEGRADED",
    HealthStatus.MISSING: "MISSING",
    HealthStatus.ERROR: "ERROR",
}


def _render_report(report: EnvironmentReport) -> None:
    table = Table(title="Environment components", title_justify="left", expand=True)
    table.add_column("Component", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Version", no_wrap=True)
    table.add_column("Detail", overflow="fold")

    for component in report.components:
        style = _STATUS_STYLE[component.status]
        detail = component.detail
        if component.remediation:
            detail = f"{detail}\n[dim]fix: {component.remediation}[/dim]"
        table.add_row(
            component.name,
            f"[{style}]{_STATUS_GLYPH[component.status]}[/{style}]",
            component.version or "-",
            detail,
        )

    console.print(table)

    compute = report.compute
    compute_table = Table(title="Compute", title_justify="left", expand=True)
    compute_table.add_column("Property", no_wrap=True)
    compute_table.add_column("Value", overflow="fold")
    compute_table.add_row("torch version", compute.torch_version or "-")
    compute_table.add_row("selected device", compute.selected_device)
    compute_table.add_row("MPS available / built", f"{compute.mps_available} / {compute.mps_built}")
    compute_table.add_row("CUDA available", str(compute.cuda_available))
    compute_table.add_row("CPU fallback", str(compute.cpu_fallback_available))
    compute_table.add_row("MediaPipe delegate", compute.mediapipe_delegate)
    compute_table.add_row("ffmpeg hwaccels", ", ".join(compute.ffmpeg_hwaccels) or "-")
    console.print(compute_table)

    platform = report.platform
    console.print(
        f"[dim]{platform.system} {platform.release} / {platform.machine} / "
        f"{platform.cpu_count} CPUs / Python {platform.python_version}[/dim]"
    )

    for warning in report.warnings:
        console.print(f"[yellow]![/yellow] {warning}")

    style = _STATUS_STYLE[report.overall_status]
    console.print(
        Panel(
            f"[{style}]{_STATUS_GLYPH[report.overall_status]}[/{style}]",
            title="Overall",
            expand=False,
        )
    )


@app.command()
def doctor(
    as_json: bool = typer.Option(
        False, "--json", help="Emit the raw report as JSON instead of a table."
    ),
) -> None:
    """Check that this machine can run the analysis pipeline."""
    report = call("doctor")
    assert isinstance(report, EnvironmentReport)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        _render_report(report)

    # Non-zero exit when something is actually wrong, so scripts and CI can gate
    # on this. DEGRADED is deliberately a success: a CPU-only machine is a
    # supported configuration, not a failure.
    if report.overall_status in (HealthStatus.MISSING, HealthStatus.ERROR):
        raise typer.Exit(code=1)


def _render_metadata(metadata: VideoMetadata) -> None:
    stream, timing = metadata.stream, metadata.timing

    table = Table(title=Path(metadata.path).name, title_justify="left", expand=True)
    table.add_column("Property", no_wrap=True)
    table.add_column("Value", overflow="fold")

    geometry = f"{stream.display_width}x{stream.display_height}"
    if (stream.display_width, stream.display_height) != (stream.coded_width, stream.coded_height):
        geometry += f"  (stored {stream.coded_width}x{stream.coded_height})"

    rotation = f"{stream.rotation_ccw_degrees}deg ccw"
    if stream.rotation_source:
        rotation += f"  (from {stream.rotation_source})"

    table.add_row("container", metadata.container_format)
    table.add_row("codec", f"{stream.codec_name} {stream.profile or ''}".strip())
    table.add_row("pixel format", stream.pix_fmt or "-")
    table.add_row("display size", geometry)
    table.add_row("rotation", rotation)
    table.add_row("frames", str(timing.frame_count))
    table.add_row("duration", f"{timing.duration_s:.3f} s")
    table.add_row("timestamp span", f"{timing.timestamp_span_s:.3f} s")
    table.add_row(
        "frame rate",
        f"measured {timing.measured_fps:.4f}" if timing.measured_fps else "measured -",
    )
    table.add_row("", f"nominal {timing.nominal_fps:.4f}" if timing.nominal_fps else "nominal -")
    table.add_row("timestamp source", timing.source.value)

    # A colour and a word, so the answer survives a colourless CI log.
    if timing.is_vfr is None:
        table.add_row("variable frame rate", "[yellow]UNKNOWN[/yellow]")
    elif timing.is_vfr:
        table.add_row("variable frame rate", "[yellow]YES[/yellow]")
    else:
        table.add_row("variable frame rate", "[green]NO[/green]")

    if timing.intervals is not None:
        stats = timing.intervals
        table.add_row(
            "frame intervals",
            f"median {stats.median_s * 1000:.3f} ms, "
            f"range {stats.min_s * 1000:.3f}-{stats.max_s * 1000:.3f} ms, "
            f"{stats.irregular_count} irregular of {timing.frame_count - 1} "
            f"(tick {stats.quantum_s * 1e6:.1f} us)",
        )

    table.add_row("size", f"{metadata.file_size_bytes:,} bytes")
    table.add_row(
        "content key", f"{metadata.content_key.algorithm.value}:{metadata.content_key.digest[:16]}"
    )

    console.print(table)

    for warning in metadata.warnings:
        console.print(f"[yellow]![/yellow] {warning}")

    if timing.source is TimestampSource.DECODED_FRAMES and timing.is_vfr is False:
        console.print(
            "[dim]Frame times are evenly spaced; index/fps arithmetic is valid here.[/dim]"
        )


def _render_extraction(result: PoseExtractionResult) -> None:
    stats, model = result.stats, result.model

    table = Table(title=Path(result.video_path).name, title_justify="left", expand=True)
    table.add_column("Property", no_wrap=True)
    table.add_column("Value", overflow="fold")

    detected = f"{stats.frames_detected} of {stats.frames_processed}"
    rate_style = "green" if stats.detection_rate >= 0.9 else "yellow"
    table.add_row("frames with a pose", f"[{rate_style}]{detected}[/{rate_style}]")
    table.add_row("detection rate", f"{stats.detection_rate:.1%}")
    table.add_row("time per frame", f"{stats.ms_per_frame:.2f} ms")
    table.add_row("total time", f"{stats.elapsed_s:.2f} s")
    table.add_row(
        "mean visibility",
        f"{stats.mean_visibility:.3f}" if stats.mean_visibility is not None else "not measured",
    )
    table.add_row("model", f"{model.name} ({model.variant}, {model.precision})")
    table.add_row("inference on", model.delegate)
    table.add_row("model digest", model.sha256[:16])
    table.add_row("output", result.output_path)

    console.print(table)

    for warning in result.warnings:
        console.print(f"[yellow]![/yellow] {warning}")


@app.command()
def probe(
    path: Annotated[Path, typer.Argument(help="Video file to inspect.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw metadata as JSON instead of a table.")
    ] = False,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-probe even if a cached result exists.")
    ] = False,
) -> None:
    """Read a video's container metadata without decoding it."""
    try:
        metadata = call("probe_video", {"path": str(path), "refresh": refresh})
    except EngineError as exc:
        # An unusable file is an expected outcome here, not a crash, so it gets
        # the same message and remedy the desktop app shows rather than a stack
        # trace with the useful part buried in it.
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc

    assert isinstance(metadata, VideoMetadata)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(metadata.model_dump(mode="json"), indent=2))
    else:
        _render_metadata(metadata)


def _render_filter(report: SequenceFilterReport) -> None:
    smoothing, gate, gaps = report.config.smoothing, report.config.gate, report.config.gaps

    summary = Table(title=f"Filtering ({report.space.value})", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")

    summary.add_row("frames", str(report.samples))
    summary.add_row("landmarks", str(len(report.landmarks)))
    summary.add_row("window", f"{smoothing.window_s:g} s, degree {smoothing.polyorder}")
    summary.add_row(
        "confidence gate",
        f"visibility >= {gate.min_visibility:g}, presence >= {gate.min_presence:g}",
    )
    summary.add_row("max bridged gap", f"{gaps.max_gap_s:g} s")

    valid = report.mean_valid_fraction
    style = "green" if valid >= 0.9 else "yellow" if valid > 0 else "red"
    summary.add_row("mean usable frames", f"[{style}]{valid:.1%}[/{style}]")
    summary.add_row("elapsed", f"{report.elapsed_s * 1000:.1f} ms")
    console.print(summary)

    # The landmarks every later phase depends on, rather than all 33: a wall of
    # rows buries the ones a reader is actually checking.
    watched = {
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
    }
    by_index = {entry.landmark: entry for entry in report.landmarks}

    detail = Table(title="Key landmarks", title_justify="left", expand=True)
    for column in ("landmark", "usable", "gated", "no detection", "filled", "blocked"):
        detail.add_column(column, no_wrap=True)
    detail.add_column("longest gap", no_wrap=True)
    detail.add_column("residual (x,y,z)", overflow="fold")

    for name, index in watched.items():
        entry = by_index.get(index)
        if entry is None:
            continue
        usable_style = "green" if entry.valid_fraction >= 0.9 else "yellow"
        detail.add_row(
            name.lower(),
            f"[{usable_style}]{entry.valid_samples}/{entry.samples}[/{usable_style}]",
            str(entry.gated_out),
            str(entry.never_detected),
            str(entry.filled),
            str(entry.blocked),
            f"{entry.longest_gap_s * 1000:.0f} ms",
            ", ".join("-" if value != value else f"{value:.5f}" for value in entry.residual_rms),
        )
    console.print(detail)

    console.print(
        f"[dim]Units: position {report.landmarks[0].position_unit.value}, "
        f"velocity {report.landmarks[0].velocity_unit.value}. "
        "Neither is a calibrated metric quantity.[/dim]"
        if report.landmarks
        else ""
    )

    for warning in report.warnings:
        console.print(f"[yellow]![/yellow] {warning}")


@app.command(name="filter")
def filter_poses(
    path: Annotated[
        Path, typer.Argument(help="Pose Parquet file, or the video it was extracted from.")
    ],
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction to filter, when given a video.")
    ] = None,
    space: Annotated[
        str,
        typer.Option("--space", help="Reference frame: frame_widths, image or hip_local."),
    ] = "frame_widths",
    window: Annotated[
        float | None, typer.Option("--window", help="Fitting window in seconds.")
    ] = None,
    polyorder: Annotated[
        int | None, typer.Option("--polyorder", help="Degree of the local polynomial.")
    ] = None,
    max_gap: Annotated[
        float | None, typer.Option("--max-gap", help="Longest absence to bridge, in seconds.")
    ] = None,
    slow_motion: Annotated[
        float,
        typer.Option(
            "--slow-motion",
            help="How many times slower than real time the clip plays (8 for 8x slo-mo).",
        ),
    ] = 1.0,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw report as JSON instead of a table.")
    ] = False,
) -> None:
    """Smooth stored landmarks, differentiate them, and report what was refused."""
    smoothing: dict[str, object] = {}
    if window is not None:
        smoothing["window_s"] = window
    if polyorder is not None:
        smoothing["polyorder"] = polyorder

    config: dict[str, object] = {}
    if smoothing:
        config["smoothing"] = smoothing
    if max_gap is not None:
        config["gaps"] = {"max_gap_s": max_gap}

    params: dict[str, object] = {
        "path": str(path),
        "space": space,
        "model": model,
        "slow_motion_factor": slow_motion,
    }
    if config:
        params["config"] = config

    try:
        report = call("filter_poses", params)
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc

    assert isinstance(report, SequenceFilterReport)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
        return

    _render_filter(report)

    # Non-zero when the clip yielded nothing usable, so a script driving this
    # learns that no trajectory came out rather than reading an empty table.
    if report.mean_valid_fraction == 0.0:
        raise typer.Exit(code=1)


_PHASE_STYLE: dict[str, str] = {
    "address": "blue",
    "backswing": "cyan",
    "downswing": "magenta",
    "follow_through": "green",
}


def _render_phases(result: SwingPhases) -> None:
    hand = result.hand

    summary = Table(title="Swing detection", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")

    verdict = "[green]YES[/green]" if result.detected else "[yellow]NO[/yellow]"
    summary.add_row("swing detected", verdict)
    summary.add_row("frames", str(result.frames))
    summary.add_row(
        "hand tracked from", f"{hand.source.value} ({hand.valid_frames}/{hand.total_frames} frames)"
    )
    summary.add_row("hand travel", f"{hand.travel:.3f} frame widths")
    summary.add_row("torso length", f"{hand.torso_length:.3f} frame widths")
    summary.add_row("travel", f"{hand.travel_ratio:.2f} torso lengths")
    summary.add_row("peak hand speed", f"{hand.peak_speed:.3f} frame widths/s")
    console.print(summary)

    if result.detected:
        events = Table(title="Events", title_justify="left", expand=True)
        for column in (
            "event",
            "frame",
            "time",
            "confidence",
            "margin",
            "visibility",
            "resolution",
        ):
            events.add_column(column, no_wrap=True)
        events.add_column("corroboration", no_wrap=True)

        for entry in result.events:
            factors = entry.confidence
            style = (
                "green" if factors.overall >= 0.6 else "yellow" if factors.overall > 0 else "red"
            )
            corroboration = (
                "-"
                if entry.corroboration_delta_s is None
                else f"frame {entry.corroboration_frame} ({entry.corroboration_delta_s * 1000:+.0f} ms)"
            )
            events.add_row(
                entry.event.value,
                str(entry.frame_index),
                f"{entry.timestamp_s:.3f} s",
                f"[{style}]{factors.overall:.2f}[/{style}]",
                f"{factors.margin:.2f}",
                f"{factors.visibility:.2f}",
                f"{factors.resolution:.2f}",
                corroboration,
            )
        console.print(events)

        intervals = Table(title="Phases", title_justify="left", expand=True)
        for column in ("phase", "frames", "start", "duration", "confidence"):
            intervals.add_column(column, no_wrap=True)
        for interval in result.phases:
            style = _PHASE_STYLE.get(interval.phase.value, "white")
            intervals.add_row(
                f"[{style}]{interval.phase.value}[/{style}]",
                f"{interval.start_frame}-{interval.end_frame}",
                f"{interval.start_s:.3f} s",
                f"{interval.duration_s:.3f} s",
                f"{interval.confidence:.2f}",
            )
        console.print(intervals)

        console.print(
            "[dim]Impact is estimated from hand kinematics; nothing here sees the ball "
            "or the club. Confidence is margin x visibility x resolution.[/dim]"
        )

    for warning in result.warnings:
        console.print(f"[yellow]![/yellow] {warning}")


@app.command()
def phases(
    path: Annotated[
        Path, typer.Argument(help="Pose Parquet file, or the video it was extracted from.")
    ],
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction to use, when given a video.")
    ] = None,
    window: Annotated[
        float | None, typer.Option("--window", help="Filtering window in seconds.")
    ] = None,
    polyorder: Annotated[
        int | None, typer.Option("--polyorder", help="Degree of the filter's local polynomial.")
    ] = None,
    slow_motion: Annotated[
        float,
        typer.Option(
            "--slow-motion",
            help="How many times slower than real time the clip plays (8 for 8x slo-mo).",
        ),
    ] = 1.0,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw result as JSON instead of tables.")
    ] = False,
) -> None:
    """Locate the takeaway, top, impact and finish in a clip."""
    smoothing: dict[str, object] = {}
    if window is not None:
        smoothing["window_s"] = window
    if polyorder is not None:
        smoothing["polyorder"] = polyorder

    params: dict[str, object] = {
        "path": str(path),
        "model": model,
        "slow_motion_factor": slow_motion,
    }
    if smoothing:
        params["filter"] = {"smoothing": smoothing}

    try:
        result = call("detect_phases", params)
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc

    assert isinstance(result, SwingPhases)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    _render_phases(result)

    # Non-zero when no swing was found, so a script driving this learns that no
    # events came out rather than reading an empty table as success.
    if not result.detected:
        raise typer.Exit(code=1)


# Short tags, because the basis belongs in every row and the full sentence for
# each is in the footer. Kept in the table rather than dropped for the meaning
# column: the two answer different questions, and a reader needs both -- what
# kind of claim the number is, and what it corresponds to on the body.
_BASIS_TAG: dict[MetricBasis, str] = {
    MetricBasis.TEMPORAL: "clock",
    MetricBasis.IMAGE_PLANE: "image",
    MetricBasis.PROJECTED_ANGLE: "proj angle",
    MetricBasis.FORESHORTENED_ANGLE: "foreshort",
    MetricBasis.SPATIAL: "3D",
}

_GROUP_TITLES: dict[MetricGroup, str] = {
    MetricGroup.POSTURE: "Posture",
    MetricGroup.ROTATION: "Rotation",
    MetricGroup.ARMS: "Hands and arms",
    MetricGroup.TIMING: "Timing",
}


def _format_measure(value: float, unit: MetricUnit, uncertainty: float | None = None) -> str:
    """Render a value with its unit, at a precision the measurement can support.

    The uncertainty rides alongside where there is one. A turn of 53 degrees and
    a turn of 53 give-or-take 13 are different findings, and only one of them is
    worth telling a player.

    Takes the parts rather than a `Metric` so that a finding's evidence renders
    identically to the metrics panel. The same number shown two ways in one
    application is a reader's problem to reconcile, and there is no reason to
    give them one.
    """
    if unit is MetricUnit.DEGREES:
        if uncertainty is not None:
            return f"{value:+.1f} +/-{uncertainty:.0f} deg"
        return f"{value:+.1f} deg"
    if unit is MetricUnit.SECONDS:
        return f"{value:.3f} s"
    if unit is MetricUnit.RATIO:
        return f"{value:.2f} : 1"
    if unit is MetricUnit.TORSO_LENGTHS_PER_S:
        return f"{value:.2f} torso/s"
    if unit is MetricUnit.METRES_PER_S:
        return f"{value:.2f} m/s"
    return f"{value:+.3f} torso"


def _format_value(metric: Metric) -> str:
    return _format_measure(metric.value, metric.unit, metric.uncertainty)


def _render_metrics(result: MetricSet) -> None:
    summary = Table(title="Biomechanics", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")

    summary.add_row("metrics computed", f"{len(result.metrics)}")
    summary.add_row("refused", f"{len(result.refused)}")
    summary.add_row("frames", str(result.frames))
    summary.add_row(
        "frame geometry",
        f"{result.geometry.width}x{result.geometry.height} "
        f"(aspect {result.geometry.aspect_ratio:.4f})",
    )
    summary.add_row("torso length", f"{result.torso_length:.3f} frame widths")
    # Stated on every result, not only when something was refused: a reader who
    # sees no refusals must not conclude the camera geometry was known.
    summary.add_row(
        "calibration",
        {
            "none": "[yellow]none[/yellow] — these carry the lens's unmeasured distortion",
            "intrinsics": "[green]intrinsics[/green] — lens removed; still image-plane only",
            "stereo": "[green]stereo[/green] — triangulation possible (Phase 9)",
        }[result.calibration.value],
    )

    view = result.view
    if view is not None:
        style = "green" if view.view is not CameraView.UNKNOWN else "yellow"
        summary.add_row(
            "camera view",
            f"[{style}]{view.view.value}[/{style}] "
            f"(confidence {view.confidence:.2f}, shoulders {view.shoulder_span_ratio:.2f} "
            f"torso at address)",
        )

    lead = result.lead_side
    if lead is not None:
        named = lead.side.value if lead.side is not None else "[yellow]undetermined[/yellow]"
        summary.add_row("lead side", f"{named} (margin {lead.margin:.2f})")
    console.print(summary)

    for reference in result.references:
        verdict = (
            "square at address"
            if reference.square_at_address
            else "[yellow]not square at address[/yellow]"
        )
        console.print(
            f"[dim]{reference.landmarks}: {reference.span:.2f} torso lengths at address, "
            f"widest {reference.widest_span:.2f} on frame {reference.widest_frame} "
            f"({reference.excess:.2f}x) -- {verdict}[/dim]"
        )

    for group in MetricGroup:
        entries = result.by_group(group)
        if not entries:
            continue
        table = Table(title=_GROUP_TITLES[group], title_justify="left", expand=True)
        for column in ("metric", "value", "conf", "obs", "anchor", "method", "basis"):
            table.add_column(column, no_wrap=True)
        table.add_column("what it means from this view", overflow="fold")

        for metric in entries:
            factors = metric.confidence
            style = (
                "green" if factors.overall >= 0.6 else "yellow" if factors.overall > 0 else "red"
            )
            table.add_row(
                metric.label,
                _format_value(metric),
                f"[{style}]{factors.overall:.2f}[/{style}]",
                f"{factors.observation:.2f}",
                f"{factors.anchor:.2f}",
                f"{factors.method:.2f}",
                _BASIS_TAG[metric.basis],
                metric.interpretation,
            )
        console.print(table)

    if result.refused:
        refusals = Table(title="Refused", title_justify="left", expand=True)
        refusals.add_column("metric", no_wrap=True)
        refusals.add_column("why", overflow="fold")
        # One row per distinct reason: a metric refused at three anchors for the
        # same cause is one fact about the recording, not three.
        seen: set[tuple[str, str]] = set()
        for entry in result.refused:
            key = (entry.name.value, entry.reason)
            if key in seen:
                continue
            seen.add(key)
            refusals.add_row(entry.name.value, entry.reason)
        console.print(refusals)

    console.print(
        "[dim]basis: clock = from the timestamps, unaffected by camera position; "
        "image = measured in the image plane, blind to motion towards the camera; "
        "proj angle = an angle in the picture, not a 3D joint angle; "
        "foreshort = rotation inferred from foreshortening, a magnitude only; "
        "3D = triangulated from two calibrated views, in metres, and the only rows "
        "here that measure the body rather than a picture of it.\n"
        "Lengths are in torso lengths, which is framing-independent but not metric, "
        "except on 3D rows. Confidence is observation x anchor x method.[/dim]"
    )

    for warning in result.warnings:
        console.print(f"[yellow]![/yellow] {warning}")


@app.command()
def metrics(
    path: Annotated[
        Path, typer.Argument(help="Pose Parquet file, or the video it was extracted from.")
    ],
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction to use, when given a video.")
    ] = None,
    window: Annotated[
        float | None, typer.Option("--window", help="Filtering window in seconds.")
    ] = None,
    polyorder: Annotated[
        int | None, typer.Option("--polyorder", help="Degree of the filter's local polynomial.")
    ] = None,
    slow_motion: Annotated[
        float,
        typer.Option(
            "--slow-motion",
            help="How many times slower than real time the clip plays (8 for 8x slo-mo).",
        ),
    ] = 1.0,
    project: Annotated[
        int | None,
        typer.Option(
            "--project",
            help="Apply this project's camera calibration. The clip must be one of its clips.",
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw result as JSON instead of tables.")
    ] = False,
) -> None:
    """Measure the biomechanics metrics for a swing."""
    smoothing: dict[str, object] = {}
    if window is not None:
        smoothing["window_s"] = window
    if polyorder is not None:
        smoothing["polyorder"] = polyorder

    params: dict[str, object] = {
        "path": str(path),
        "model": model,
        "slow_motion_factor": slow_motion,
    }
    if smoothing:
        params["filter"] = {"smoothing": smoothing}
    if project is not None:
        params["project_id"] = project

    try:
        result = call("compute_metrics", params)
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc

    assert isinstance(result, MetricSet)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    _render_metrics(result)

    # Non-zero when nothing was measured, so a script driving this learns that no
    # metrics came out rather than reading an empty set as success.
    if not result.computed:
        raise typer.Exit(code=1)


def _format_ms(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.1f} ms" if signed else f"{value:.1f} ms"


def _render_sync(result: SyncModel) -> None:
    summary = Table(title="Synchronisation", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")

    verdict = "[green]YES[/green]" if result.aligned else "[yellow]NO[/yellow]"
    summary.add_row("aligned", verdict)
    summary.add_row("method", result.method.value if result.method else "-")
    for role, clip in (("reference", result.reference), ("target", result.target)):
        rate = f"{1.0 / clip.median_interval_s:.1f} fps" if clip.median_interval_s > 0 else "-"
        slow = "" if clip.slow_motion_factor == 1.0 else f", {clip.slow_motion_factor:g}x slo-mo"
        summary.add_row(role, f"{clip.name} ({clip.frames} frames, {rate}{slow})")

    mapping = result.time_map
    if mapping is not None:
        uncertainty = mapping.offset_uncertainty_s
        offset = f"{mapping.offset_s * 1000:+.1f} ms"
        if uncertainty is not None:
            offset += f" +/-{uncertainty * 1000:.1f} ms"
        summary.add_row("offset (target - reference)", offset)
        if mapping.rate_estimated:
            spread = (
                f" +/-{mapping.rate_uncertainty:.5f}"
                if mapping.rate_uncertainty is not None
                else ""
            )
            summary.add_row("clock rate", f"[yellow]{mapping.rate:.5f}{spread}[/yellow]")
        else:
            summary.add_row("clock rate", "[dim]1.0 (assumed, not measured)[/dim]")
        summary.add_row(
            "anchored over",
            f"{mapping.support_start_s:.3f}-{mapping.support_end_s:.3f} s of the reference",
        )

    overlap = result.overlap
    if overlap is not None:
        summary.add_row(
            "overlap",
            f"{overlap.duration_s:.3f} s ({overlap.reference_fraction:.0%} of the reference, "
            f"{overlap.target_fraction:.0%} of the target)",
        )
    console.print(summary)

    quality, confidence = result.quality, result.confidence
    if quality is not None and confidence is not None:
        scores = Table(title="How well it is determined", title_justify="left", expand=True)
        scores.add_column("Property", no_wrap=True)
        scores.add_column("Value", overflow="fold")

        style = (
            "green"
            if confidence.overall >= 0.6
            else "yellow"
            if confidence.overall > 0.1
            else "red"
        )
        scores.add_row("confidence", f"[{style}]{confidence.overall:.2f}[/{style}]")
        scores.add_row(
            "",
            f"[dim]agreement {confidence.agreement:.2f} x anchors {confidence.anchors:.2f} "
            f"x stability {confidence.stability:.2f}[/dim]",
        )
        scores.add_row("frame-rate floor", _format_ms(quality.quantisation_floor_ms))
        # The distinction that matters most in this table: a residual of zero
        # because the anchors agreed, against no residual at all because there
        # was nothing left over to disagree.
        if quality.residual_rms_ms is None:
            scores.add_row(
                "anchor residual",
                f"[yellow]not measurable[/yellow] ({quality.degrees_of_freedom} spare "
                "degrees of freedom -- the fit passes through every anchor)",
            )
        else:
            ratio = quality.residual_rms_ms / max(quality.quantisation_floor_ms, 1e-9)
            scores.add_row(
                "anchor residual",
                f"{_format_ms(quality.residual_rms_ms)} rms, "
                f"{_format_ms(quality.residual_max_ms)} worst ({ratio:.1f}x the floor)",
            )
        if quality.method_disagreement_ms is not None:
            comparable = mapping is not None and not mapping.rate_estimated
            note = "" if comparable else " [dim](not comparable: a rate was fitted)[/dim]"
            scores.add_row(
                "vs cross-correlation", _format_ms(quality.method_disagreement_ms) + note
            )
        console.print(scores)

    if result.anchors:
        anchors = Table(title="Anchors", title_justify="left", expand=True)
        for column in ("instant", "source", "reference frame", "target frame", "conf", "residual"):
            anchors.add_column(column, no_wrap=True)
        residuals = {entry.label: entry for entry in result.residuals}
        for anchor in result.anchors:
            entry = residuals.get(anchor.label)
            anchors.add_row(
                anchor.label,
                anchor.source.value,
                f"{anchor.reference_frame} ({anchor.reference_s:.3f} s)",
                f"{anchor.target_frame} ({anchor.target_s:.3f} s)",
                f"{anchor.confidence:.2f}",
                "-" if entry is None else _format_ms(entry.residual_ms, signed=True),
            )
        console.print(anchors)

    correlation = result.correlation
    if correlation is not None:
        rival = (
            "no rival scored"
            if correlation.rival_correlation is None
            else f"best rival {correlation.rival_correlation:.3f} at "
            f"{(correlation.rival_offset_s or 0.0) * 1000:+.0f} ms"
        )
        console.print(
            f"[dim]Cross-correlation: peak {correlation.peak_correlation:.3f} at "
            f"{correlation.peak_offset_s * 1000:+.1f} ms over {correlation.overlap_s:.2f} s "
            f"({correlation.samples} samples at {correlation.grid_interval_s * 1000:.2f} ms); "
            f"{rival}.[/dim]"
        )

    if result.refusal:
        console.print(f"[red]{result.refusal}[/red]")

    for warning in result.warnings:
        console.print(f"[yellow]![/yellow] {warning}")


def _parse_anchor(raw: str) -> dict[str, object]:
    """Parse `label=reference_frame:target_frame`.

    Terse because picking four instants on a command line should not need four
    flags each. The error names the form rather than the parse failure, because
    what a user needs here is the shape, not which colon was missing.
    """
    label, _, frames = raw.partition("=")
    reference, _, target = frames.partition(":")
    try:
        return {
            "label": label.strip() or "manual",
            "reference_frame": int(reference),
            "target_frame": int(target),
        }
    except ValueError as exc:
        raise typer.BadParameter(
            f"'{raw}' is not an anchor. The form is label=REFERENCE_FRAME:TARGET_FRAME, "
            "for example impact=204:252."
        ) from exc


@app.command()
def sync(
    reference: Annotated[
        Path, typer.Argument(help="The clip whose clock everything is stated in.")
    ],
    target: Annotated[Path, typer.Argument(help="The clip to align to the reference.")],
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction to use, when given videos.")
    ] = None,
    window: Annotated[
        float | None,
        typer.Option(
            "--window",
            help="Filtering window in seconds, shared by both clips. The coarser clip sets it.",
        ),
    ] = None,
    method: Annotated[
        str | None,
        typer.Option("--method", help="Force a method: events, correlation or manual."),
    ] = None,
    anchor: Annotated[
        list[str] | None,
        typer.Option(
            "--anchor",
            help="A manual pick, as label=REFERENCE_FRAME:TARGET_FRAME. Repeatable.",
        ),
    ] = None,
    slow_motion_reference: Annotated[
        float,
        typer.Option("--slow-motion-reference", help="Slow-motion factor of the reference clip."),
    ] = 1.0,
    slow_motion_target: Annotated[
        float, typer.Option("--slow-motion-target", help="Slow-motion factor of the target clip.")
    ] = 1.0,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw result as JSON instead of tables.")
    ] = False,
) -> None:
    """Relate two clips' clocks, and report how well the relation is known."""
    params: dict[str, object] = {
        "reference": {
            "path": str(reference),
            "model": model,
            "slow_motion_factor": slow_motion_reference,
        },
        "target": {
            "path": str(target),
            "model": model,
            "slow_motion_factor": slow_motion_target,
        },
    }
    if window is not None:
        params["filter"] = {"smoothing": {"window_s": window}}
    if method is not None:
        params["sync"] = {"method": method}
    if anchor:
        params["anchors"] = [_parse_anchor(entry) for entry in anchor]

    try:
        result = call("sync_clips", params)
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc

    assert isinstance(result, SyncModel)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    _render_sync(result)

    # Non-zero when no map came out, so a script driving this learns that the
    # clips were not aligned rather than reading a refusal as an offset of zero.
    if not result.aligned:
        raise typer.Exit(code=1)


projects = typer.Typer(
    name="project",
    help="Sessions: the clips of one swing, and how their clocks relate.",
    no_args_is_help=True,
)
app.add_typer(projects)


def _render_project(project: Project) -> None:
    summary = Table(title=f"{project.name} (id {project.id})", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")
    summary.add_row("created", project.created_at.astimezone().strftime("%Y-%m-%d %H:%M"))
    if project.notes:
        summary.add_row("notes", project.notes)
    summary.add_row("clips", str(len(project.clips)))
    console.print(summary)

    if project.clips:
        clips = Table(title="Clips", title_justify="left", expand=True)
        for column in ("id", "role", "file", "slow motion", "on disk"):
            clips.add_column(column, no_wrap=True)
        for clip in project.clips:
            clips.add_row(
                str(clip.id),
                clip.role.value,
                clip.name,
                "-" if clip.slow_motion_factor == 1.0 else f"{clip.slow_motion_factor:g}x",
                "[green]yes[/green]" if clip.exists else "[red]missing[/red]",
            )
        console.print(clips)

    for stored in project.syncs:
        mapping = stored.model.time_map
        offset = "refused" if mapping is None else f"{mapping.offset_s * 1000:+.1f} ms"
        confidence = (
            "-" if stored.model.confidence is None else f"{stored.model.confidence.overall:.2f}"
        )
        console.print(
            f"[dim]sync {stored.reference_clip_id} -> {stored.target_clip_id}: {offset} "
            f"(confidence {confidence}, {stored.updated_at.astimezone():%Y-%m-%d %H:%M})[/dim]"
        )

    for warning in project.warnings:
        console.print(f"[yellow]![/yellow] {warning}")


def _project_call(method: str, params: dict[str, object]) -> BaseModel:
    try:
        return call(method, params)
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc


@projects.command("create")
def project_create(
    name: Annotated[str, typer.Argument(help="What this session is.")],
    notes: Annotated[str, typer.Option("--notes", help="Free text.")] = "",
) -> None:
    """Create an empty project."""
    result = _project_call("create_project", {"name": name, "notes": notes})
    assert isinstance(result, Project)  # noqa: S101 - narrows the dispatch return type
    _render_project(result)


@projects.command("list")
def project_list() -> None:
    """List every project, newest first."""
    result = _project_call("list_projects", {})
    assert isinstance(result, ProjectList)  # noqa: S101 - narrows the dispatch return type

    if not result.projects:
        console.print("[dim]No projects yet. Create one with `analyzer project create`.[/dim]")
        console.print(f"[dim]Database: {result.database_path}[/dim]")
        return

    table = Table(title="Projects", title_justify="left", expand=True)
    for column in ("id", "name", "created", "clips", "missing"):
        table.add_column(column, no_wrap=True)
    for project in result.projects:
        missing = sum(1 for clip in project.clips if not clip.exists)
        table.add_row(
            str(project.id),
            project.name,
            project.created_at.astimezone().strftime("%Y-%m-%d"),
            str(len(project.clips)),
            "-" if missing == 0 else f"[red]{missing}[/red]",
        )
    console.print(table)
    console.print(f"[dim]Database: {result.database_path}[/dim]")


@projects.command("show")
def project_show(
    project_id: Annotated[int, typer.Argument(help="Which project.")],
) -> None:
    """Show one project, its clips and its stored alignments."""
    result = _project_call("get_project", {"project_id": project_id})
    assert isinstance(result, Project)  # noqa: S101 - narrows the dispatch return type
    _render_project(result)


@projects.command("add")
def project_add(
    project_id: Annotated[int, typer.Argument(help="Which project.")],
    path: Annotated[Path, typer.Argument(help="Video file to attach.")],
    role: Annotated[str, typer.Option("--role", help="face_on, down_the_line or other.")] = "other",
    slow_motion: Annotated[
        float, typer.Option("--slow-motion", help="How many times slower than real time.")
    ] = 1.0,
    label: Annotated[str, typer.Option("--label", help="Free text.")] = "",
) -> None:
    """Attach a clip to a project. The clip is identified by content, not by path."""
    result = _project_call(
        "add_clip",
        {
            "project_id": project_id,
            "path": str(path),
            "role": role,
            "slow_motion_factor": slow_motion,
            "label": label,
        },
    )
    assert isinstance(result, Project)  # noqa: S101 - narrows the dispatch return type
    _render_project(result)


@projects.command("delete")
def project_delete(
    project_id: Annotated[int, typer.Argument(help="Which project.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete a project, its clips and its alignments. The video files are untouched."""
    if not yes:
        # A project is the one thing here that cannot be recomputed from the
        # footage, so deleting one is confirmed rather than assumed. Everything
        # else this CLI removes can be rebuilt by re-running a command.
        project = _project_call("get_project", {"project_id": project_id})
        assert isinstance(project, Project)  # noqa: S101 - narrows the dispatch return type
        console.print(
            f"Deleting project {project.id} ({project.name}) with {len(project.clips)} clip(s) "
            f"and {len(project.syncs)} stored alignment(s). The video files are not touched."
        )
        typer.confirm("Continue?", abort=True)

    result = _project_call("delete_project", {"project_id": project_id})
    assert isinstance(result, ProjectList)  # noqa: S101 - narrows the dispatch return type
    console.print(f"[dim]Deleted. {len(result.projects)} project(s) remain.[/dim]")


@projects.command("remove")
def project_remove(
    project_id: Annotated[int, typer.Argument(help="Which project.")],
    clip_id: Annotated[int, typer.Argument(help="Which clip.")],
) -> None:
    """Detach a clip from a project. Its stored alignments go with it."""
    result = _project_call("remove_clip", {"project_id": project_id, "clip_id": clip_id})
    assert isinstance(result, Project)  # noqa: S101 - narrows the dispatch return type
    _render_project(result)


@projects.command("relocate")
def project_relocate(
    project_id: Annotated[int, typer.Argument(help="Which project.")],
    clip_id: Annotated[int, typer.Argument(help="Which clip.")],
    path: Annotated[Path, typer.Argument(help="Where the file is now.")],
) -> None:
    """Point a clip at a moved file, keeping its recorded content key."""
    result = _project_call(
        "relocate_clip", {"project_id": project_id, "clip_id": clip_id, "path": str(path)}
    )
    assert isinstance(result, Project)  # noqa: S101 - narrows the dispatch return type
    _render_project(result)


@projects.command("sync")
def project_sync(
    project_id: Annotated[int, typer.Argument(help="Which project.")],
    reference_clip: Annotated[
        int | None, typer.Option("--reference-clip", help="Clip id to use as the reference.")
    ] = None,
    target_clip: Annotated[
        int | None, typer.Option("--target-clip", help="Clip id to align to the reference.")
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="Which extraction to use.")] = None,
    window: Annotated[
        float | None, typer.Option("--window", help="Filtering window in seconds, shared by both.")
    ] = None,
    method: Annotated[
        str | None, typer.Option("--method", help="events, correlation or manual.")
    ] = None,
    anchor: Annotated[
        list[str] | None,
        typer.Option("--anchor", help="A manual pick, as label=REFERENCE_FRAME:TARGET_FRAME."),
    ] = None,
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Store the result on the project.")
    ] = True,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw result as JSON instead of tables.")
    ] = False,
) -> None:
    """Align two of a project's clips, using each clip's own slow-motion factor."""
    params: dict[str, object] = {
        "project_id": project_id,
        "reference_clip_id": reference_clip,
        "target_clip_id": target_clip,
        "model": model,
        "save": save,
    }
    if window is not None:
        params["filter"] = {"smoothing": {"window_s": window}}
    if method is not None:
        params["sync"] = {"method": method}
    if anchor:
        params["anchors"] = [_parse_anchor(entry) for entry in anchor]

    result = _project_call("sync_project", params)
    assert isinstance(result, SyncModel)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    _render_sync(result)
    if not result.aligned:
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the engine version."""
    typer.echo(__version__)


@app.command()
def extract(
    path: Annotated[Path, typer.Argument(help="Video file to extract poses from.")],
    model: Annotated[
        str | None,
        typer.Option("--model", help="Manifest model name. Defaults to the manifest's choice."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Where to write the Parquet file.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw result as JSON instead of a table.")
    ] = False,
) -> None:
    """Run pose estimation over every frame of a clip and store the landmarks."""
    params: dict[str, object] = {"path": str(path), "model": model}
    if output is not None:
        params["output"] = str(output)

    # The same reporter seam the desktop app uses for notifications drives a
    # terminal progress bar here, which is the point of it being a seam.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        # Redirected to stderr so `--json` output stays a clean pipe.
        disable=as_json,
    ) as progress:
        bar = progress.add_task("Estimating pose", total=None)

        def on_update(update: ProgressUpdate) -> None:
            progress.update(
                bar,
                completed=update.current,
                total=update.total,
                description="Loading the model"
                if update.stage == "starting"
                else "Estimating pose",
            )

        try:
            result = call("extract_poses", params, CallbackReporter(on_update))
        except EngineError as exc:
            progress.stop()
            console.print(f"[red]{exc}[/red]")
            remediation = (exc.data or {}).get("remediation")
            if remediation:
                console.print(f"[dim]fix: {remediation}[/dim]")
            raise typer.Exit(code=1) from exc

    assert isinstance(result, PoseExtractionResult)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    _render_extraction(result)

    # Non-zero when nothing was found: a caller scripting this needs to know the
    # run produced no usable landmarks, and an exit code is how it finds out.
    if result.stats.frames_detected == 0:
        raise typer.Exit(code=1)


# Last in the file on purpose. Typer registers a command when its decorator
# runs, so anything defined below this block is absent when the module is


def _run_with_progress(
    method: str, params: dict[str, object], description: str, *, quiet: bool = False
) -> BaseModel:
    """Call a long method, driving a terminal bar from the same reporter seam the app uses.

    Factored out here because board detection is the third long method in this
    CLI and the bar wiring was already written twice. The error handling is part
    of it: a failure has to stop the bar before printing, or Rich leaves the
    partly-drawn bar sitting over the message.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        # Redirected to stderr so `--json` output stays a clean pipe.
        disable=quiet,
    ) as progress:
        bar = progress.add_task(description, total=None)

        def on_update(update: ProgressUpdate) -> None:
            progress.update(bar, completed=update.current, total=update.total)

        try:
            return call(method, params, CallbackReporter(on_update))
        except EngineError as exc:
            progress.stop()
            console.print(f"[red]{exc}[/red]")
            remediation = (exc.data or {}).get("remediation")
            if remediation:
                console.print(f"[dim]fix: {remediation}[/dim]")
            raise typer.Exit(code=1) from exc


calibrate = typer.Typer(
    name="calibrate",
    help="Measure a camera's geometry, so the system knows what a pixel means.",
    no_args_is_help=True,
)
app.add_typer(calibrate)


def _render_coverage(quality: CalibrationQuality) -> Table:
    """The diagnostic half, which is the half that decides whether to trust the fit."""
    table = Table(title="Coverage", title_justify="left", expand=True)
    table.add_column("What the board views sampled", no_wrap=True)
    table.add_column("Value", overflow="fold")

    found = quality.coverage
    table.add_row("views used", str(found.views))
    table.add_row("corners", str(found.corners))
    table.add_row("frame area visited", f"{found.image_fraction:.0%}")
    table.add_row("corners near the edge", f"{found.edge_fraction:.0%}")
    table.add_row("tilt spread", f"{found.tilt_range_deg:.0f} deg")
    table.add_row("apparent size spread", f"{found.scale_range:.2f}x")
    table.add_row("spare degrees of freedom", str(quality.degrees_of_freedom))
    return table


def _render_camera_calibration(result: CameraCalibration) -> None:
    summary = Table(title="Camera calibration", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")

    found = result.intrinsics
    verdict = "[green]USABLE[/green]" if result.usable else "[red]REFUSED[/red]"
    summary.add_row("usable", verdict)
    summary.add_row("role", result.role.value)
    summary.add_row("frame size", f"{found.image_width}x{found.image_height}")
    summary.add_row("focal length", f"fx {found.fx:.1f} px, fy {found.fy:.1f} px")
    summary.add_row("optical centre", f"({found.cx:.1f}, {found.cy:.1f}) px")
    summary.add_row(
        "field of view",
        f"{found.horizontal_fov_deg:.1f} deg across, {found.vertical_fov_deg:.1f} deg down"
        "  [dim](check this against the lens)[/dim]",
    )
    summary.add_row("distortion model", found.model.value)
    summary.add_row(
        "distortion", ", ".join(f"{value:+.4f}" for value in found.distortion) or "none"
    )

    # The three numbers that answer three different questions, in the order the
    # module docstring puts them: the fit, the answer, and then the cause.
    quality = result.quality
    summary.add_row(
        "reprojection error",
        f"{quality.rms_reprojection_px:.3f} px rms, {quality.max_reprojection_px:.3f} px worst "
        "[dim](how well the model fits these views)[/dim]",
    )
    if found.fx_uncertainty is not None:
        ratio = found.fx_uncertainty / found.fx
        style = "green" if ratio <= 0.02 else "yellow" if ratio <= 0.05 else "red"
        summary.add_row(
            "focal uncertainty",
            f"[{style}]{found.fx_uncertainty:.2f} px ({ratio:.2%})[/{style}] "
            "[dim](whether these views determined it)[/dim]",
        )
    else:
        summary.add_row("focal uncertainty", "[dim]not reported[/dim]")

    edge, worst = distortion_displacement(found)
    summary.add_row(
        "this lens moves a pixel by",
        f"{edge:.1f} px at the frame edge, {worst:.1f} px at worst",
    )
    if result.notes:
        summary.add_row("notes", result.notes)
    console.print(summary)
    console.print(_render_coverage(quality))

    detection = result.detection
    console.print(
        f"[dim]Scanned {detection.frames_scanned} frame(s); board found in "
        f"{detection.frames_with_board}; {detection.views_used} distinct views kept.[/dim]"
    )
    for warning in [*detection.warnings, *result.warnings]:
        console.print(f"[yellow]! {warning}[/yellow]")
    if result.refusal:
        console.print(Panel(result.refusal, title="Refused", border_style="red"))


def _render_stereo(result: StereoCalibration) -> None:
    summary = Table(title="Stereo extrinsics", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")

    verdict = "[green]USABLE[/green]" if result.usable else "[red]REFUSED[/red]"
    summary.add_row("usable", verdict)
    summary.add_row("cameras", f"{result.reference_role.value} -> {result.target_role.value}")
    summary.add_row(
        "baseline",
        f"{result.baseline_m:.3f} m  [dim](check this with a tape measure)[/dim]",
    )
    summary.add_row("convergence", f"{result.convergence_deg:.1f} deg between the optical axes")
    summary.add_row(
        "reprojection error",
        f"{result.quality.rms_reprojection_px:.3f} px rms "
        "[dim](a point located by one camera, seen by the other)[/dim]",
    )

    pairing = result.pairing
    summary.add_row("paired views", f"{pairing.pairs} of {pairing.candidates} candidates")
    if pairing.median_time_error_ms is not None:
        summary.add_row("pairing, in time", f"{pairing.median_time_error_ms:.1f} ms median")
    if pairing.median_board_speed_px_s is not None:
        summary.add_row("board speed", f"{pairing.median_board_speed_px_s:.0f} px/s median")
    if pairing.worst_pairing_error_px is not None:
        summary.add_row(
            "pairing, in pixels",
            f"{pairing.worst_pairing_error_px:.2f} px worst "
            "[dim](sync uncertainty x board speed)[/dim]",
        )
    console.print(summary)
    console.print(_render_coverage(result.quality))

    for entry in pairing.dropped:
        console.print(f"[dim]dropped: {entry}[/dim]")
    for warning in result.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")
    if result.refusal:
        console.print(Panel(result.refusal, title="Refused", border_style="red"))


def _board_params(
    squares: str, square_mm: float, marker_mm: float | None, family: str, legacy: bool
) -> dict[str, object]:
    width, _, height = squares.partition("x")
    if not height:
        raise typer.BadParameter("Give the board as WxH, for example 7x5.", param_hint="--squares")
    return {
        "squares_x": int(width),
        "squares_y": int(height),
        "square_length_mm": square_mm,
        "marker_length_mm": marker_mm,
        "family": family,
        "legacy_pattern": legacy,
    }


@calibrate.command(name="board")
def calibrate_board(
    output: Annotated[Path, typer.Argument(help="Where to write the printable board image.")],
    squares: Annotated[str, typer.Option("--squares", help="Board squares, as WxH.")] = "7x5",
    square_mm: Annotated[
        float, typer.Option("--square-mm", help="Intended printed square size, in millimetres.")
    ] = 35.0,
    family: Annotated[str, typer.Option("--family", help="ArUco dictionary.")] = "DICT_5X5_100",
    dpi: Annotated[float, typer.Option("--dpi", help="Rendering resolution.")] = 600.0,
) -> None:
    """Generate a Charuco board to print.

    Generated here rather than downloaded, so the board that is printed and the
    board that is looked for are the same object by construction.
    """
    import cv2

    from analyzer.calibration.board import generate_board_image

    spec = BoardSpec(
        squares_x=int(squares.partition("x")[0]),
        squares_y=int(squares.partition("x")[2]),
        square_length_m=square_mm / 1000.0,
        marker_length_m=0.75 * square_mm / 1000.0,
        family=BoardFamily(family),
    )
    image = generate_board_image(spec, pixels_per_metre=dpi / 0.0254)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        console.print(f"[red]Could not write {output}.[/red]")
        raise typer.Exit(code=1)

    console.print(f"Wrote [bold]{output}[/bold] ({image.shape[1]}x{image.shape[0]} px).")
    console.print(
        Panel(
            f"Print this at 100% scale -- no 'fit to page'. Then [bold]measure one square "
            f"with a ruler[/bold] and pass what you measure as --square-mm, not "
            f"{square_mm:.0f}.\n\n"
            "Every metric-scale claim this system ever makes descends from that one "
            "measured length, and a page silently scaled to 96% makes every future "
            "distance wrong by 4% with nothing anywhere looking amiss.\n\n"
            "Mount it on something rigid and flat. The method assumes the board is a "
            "plane, and a sheet held in one hand is not.",
            title="Before you use it",
            border_style="yellow",
        )
    )


@calibrate.command(name="camera")
def calibrate_camera_command(
    source: Annotated[
        Path, typer.Argument(help="A video of the board, or a directory of board photographs.")
    ],
    role: Annotated[
        str, typer.Option("--role", help="Which camera: face_on, down_the_line or other.")
    ] = "other",
    project: Annotated[
        int | None, typer.Option("--project", help="Store the result on this project's rig.")
    ] = None,
    squares: Annotated[str, typer.Option("--squares", help="Board squares, as WxH.")] = "7x5",
    square_mm: Annotated[
        float,
        typer.Option("--square-mm", help="Square size measured on the printed sheet, in mm."),
    ] = 35.0,
    marker_mm: Annotated[
        float | None, typer.Option("--marker-mm", help="Marker size. Defaults to 0.75 of a square.")
    ] = None,
    family: Annotated[str, typer.Option("--family", help="ArUco dictionary.")] = "DICT_5X5_100",
    legacy: Annotated[
        bool, typer.Option("--legacy-pattern", help="The board came from OpenCV before 4.6.")
    ] = False,
    stride: Annotated[int, typer.Option("--stride", help="Frames to skip, for a video.")] = 5,
    model: Annotated[
        str | None,
        typer.Option("--distortion", help="pinhole, radial_tangential_4 or radial_tangential_5."),
    ] = None,
    notes: Annotated[
        str,
        typer.Option("--notes", help="Lens, zoom, stabilisation -- what the file cannot record."),
    ] = "",
    as_json: Annotated[bool, typer.Option("--json", help="Emit the raw result as JSON.")] = False,
) -> None:
    """Measure one camera's intrinsics from footage of a Charuco board."""
    params: dict[str, object] = {
        "source": str(source),
        "role": role,
        "board": _board_params(squares, square_mm, marker_mm, family, legacy),
        "stride": stride,
        "notes": notes,
    }
    if project is not None:
        params["project_id"] = project
    if model is not None:
        params["calibration"] = {"distortion_model": model}

    result = _run_with_progress("calibrate_camera", params, "Detecting the board", quiet=as_json)
    assert isinstance(result, CameraCalibration)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
    else:
        _render_camera_calibration(result)

    # Non-zero when the calibration should not be used, so a script driving this
    # learns that rather than reading a refused calibration as a usable one.
    if not result.usable:
        raise typer.Exit(code=1)


@calibrate.command(name="stereo")
def calibrate_stereo_command(
    project: Annotated[int, typer.Argument(help="The project whose cameras these are.")],
    reference: Annotated[Path, typer.Argument(help="The reference camera's board footage.")],
    target: Annotated[Path, typer.Argument(help="The target camera's board footage.")],
    reference_role: Annotated[
        str, typer.Option("--reference-role", help="Which camera the reference footage is.")
    ] = "face_on",
    target_role: Annotated[
        str, typer.Option("--target-role", help="Which camera the target footage is.")
    ] = "down_the_line",
    offset: Annotated[
        float | None,
        typer.Option("--offset", help="Target clock minus reference clock, in seconds."),
    ] = None,
    offset_uncertainty: Annotated[
        float,
        typer.Option("--offset-uncertainty", help="How well that offset is known, in seconds."),
    ] = 0.0,
    squares: Annotated[str, typer.Option("--squares", help="Board squares, as WxH.")] = "7x5",
    square_mm: Annotated[float, typer.Option("--square-mm", help="Square size, in mm.")] = 35.0,
    marker_mm: Annotated[float | None, typer.Option("--marker-mm")] = None,
    family: Annotated[str, typer.Option("--family")] = "DICT_5X5_100",
    legacy: Annotated[bool, typer.Option("--legacy-pattern")] = False,
    stride: Annotated[int, typer.Option("--stride", help="Frames to skip, for a video.")] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the raw result as JSON.")] = False,
) -> None:
    """Measure where two already-calibrated cameras stand relative to each other.

    Both cameras must be calibrated first: the stereo fit holds the intrinsics
    fixed rather than re-fitting them, because they are the better-determined
    quantity and a joint fit would trade focal length against baseline.
    """
    params: dict[str, object] = {
        "project_id": project,
        "reference_source": str(reference),
        "target_source": str(target),
        "reference_role": reference_role,
        "target_role": target_role,
        "board": _board_params(squares, square_mm, marker_mm, family, legacy),
        "stride": stride,
        "offset_uncertainty_s": offset_uncertainty,
    }
    if offset is not None:
        params["offset_s"] = offset

    result = _run_with_progress("calibrate_stereo", params, "Detecting the board", quiet=as_json)
    assert isinstance(result, StereoCalibration)  # noqa: S101

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
    else:
        _render_stereo(result)

    if not result.usable:
        raise typer.Exit(code=1)


@calibrate.command(name="show")
def calibrate_show(
    project: Annotated[int, typer.Argument(help="Which project's rig to report.")],
    as_json: Annotated[bool, typer.Option("--json", help="Emit the raw result as JSON.")] = False,
) -> None:
    """What is known about a project's cameras, and what that permits."""
    try:
        result = call("get_calibration", {"project_id": project})
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    assert isinstance(result, CameraRig)  # noqa: S101

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    status = result.status()
    style = {"none": "red", "intrinsics": "yellow", "stereo": "green"}[status.value]
    console.print(
        Panel(
            f"[{style}]{status.value.upper()}[/{style}] -- "
            + {
                "none": "no camera geometry is known, so every measurement is a statement "
                "about the image plane and carries the lens's unmeasured distortion.",
                "intrinsics": "the lens is measured and can be removed from the landmarks. "
                "Still no depth: a calibrated camera knows which direction a pixel came "
                "from, not how far away it was.",
                "stereo": "both cameras and their relative pose are known, so two views of "
                "one instant can be triangulated. Phase 9 is what does that.",
            }[status.value],
            title="What this project may claim",
            border_style=style,
        )
    )

    for _role, entry in sorted(result.cameras.items(), key=lambda item: item[0].value):
        console.print()
        _render_camera_calibration(entry)
    if result.stereo is not None:
        console.print()
        _render_stereo(result.stereo)


@calibrate.command(name="clear")
def calibrate_clear(
    project: Annotated[int, typer.Argument(help="Which project to forget the calibration of.")],
) -> None:
    """Forget a project's calibration. The footage is untouched."""
    try:
        call("clear_calibration", {"project_id": project})
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Project {project} is now uncalibrated.")


def _render_reconstruction(result: ReconstructionReport) -> None:
    """The three numbers, each labelled with the question it answers.

    Laid out like the calibration review for the same reason: a reader who is
    shown only the reprojection error will take it for the accuracy, and it is
    not -- it is blind along the epipolar line, which is the direction the error
    actually moves a point. So the convergence angle and the bone check sit
    beside it, and each row says what it is evidence of.
    """
    summary = Table(title="3D reconstruction", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")

    verdict = "[green]YES[/green]" if result.reconstructed else "[yellow]NO[/yellow]"
    summary.add_row("reconstructed", verdict)
    summary.add_row(
        "cameras",
        f"{result.reference_role.value} ({result.reference_name}) -> "
        f"{result.target_role.value} ({result.target_name})",
    )
    summary.add_row("frame", f"metres, centred on the {result.reference_role.value} camera")
    summary.add_row(
        "rig", f"{result.baseline_m:.2f} m apart, {result.convergence_deg:.0f} deg apart"
    )
    summary.add_row("frames", f"{result.reconstructed_frames} of {result.frames} carry a point")

    quality = result.quality
    if quality is not None:
        summary.add_row("coverage", f"{quality.coverage:.0%} of landmark-frames")
        summary.add_row(
            "ray convergence",
            f"{_deg(quality.median_convergence_deg)} median, {_deg(quality.min_convergence_deg)} "
            "worst  [dim](the gate: depth error goes as 1/sin)[/dim]",
        )
        summary.add_row(
            "reprojection error",
            f"{_px(quality.median_reprojection_px)} median, {_px(quality.max_reprojection_px)} "
            "worst  [dim](blind along the epipolar line)[/dim]",
        )
        summary.add_row(
            "positional uncertainty",
            f"{_mm(quality.median_uncertainty_m)} median, {_mm(quality.p95_uncertainty_m)} p95 "
            f"[dim](worst direction, from {quality.pixel_sigma_px:.2f} px landmark scatter)[/dim]",
        )
        if quality.worst_bone_variation is not None:
            style = (
                "green"
                if quality.worst_bone_variation <= result.config.max_bone_variation
                else "yellow"
            )
            summary.add_row(
                "worst bone variation",
                f"[{style}]{quality.worst_bone_variation:.1%}[/{style}]  "
                "[dim](the check the residual cannot make)[/dim]",
            )

    pairing = result.pairing
    if pairing is not None:
        summary.add_row(
            "pairing",
            f"{pairing.method}, {pairing.resampled_frames} frames"
            + (
                f", {pairing.outside_overlap} outside the overlap"
                if pairing.outside_overlap
                else ""
            ),
        )
        if pairing.max_pairing_error_px is not None:
            summary.add_row(
                "sync cost, in pixels",
                f"{pairing.max_pairing_error_px:.2f} px at the fastest landmark "
                f"[dim](nearest-frame pairing would cost "
                f"{_px(pairing.nearest_frame_error_px)})[/dim]",
            )
    console.print(summary)

    if quality is not None and quality.bones:
        bones = Table(title="Reconstructed segments", title_justify="left", expand=True)
        bones.add_column("Segment", no_wrap=True)
        bones.add_column("Median", justify="right")
        bones.add_column("Variation", justify="right")
        bones.add_column("", no_wrap=True)
        for entry in sorted(quality.bones, key=lambda row: -row.variation)[:8]:
            mark = "" if entry.stable else "[yellow]unstable[/yellow]"
            bones.add_row(
                entry.name,
                f"{entry.median_length_m * 100:.1f} cm",
                f"{entry.variation:.1%}",
                mark,
            )
        console.print(bones)

    if quality is not None and quality.symmetry:
        worst = max(quality.symmetry, key=lambda row: row.disagreement)
        console.print(
            f"[dim]Left/right agreement: worst is {worst.segment} at "
            f"{worst.disagreement:.1%} ({worst.left_m * 100:.1f} vs "
            f"{worst.right_m * 100:.1f} cm).[/dim]"
        )

    for warning in result.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")
    if result.refusal:
        console.print(Panel(result.refusal, title="Refused", border_style="red"))


def _deg(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} deg"


def _px(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f} px"


def _mm(value: float | None) -> str:
    return "-" if value is None else f"{value * 1000:.1f} mm"


@app.command()
def reconstruct(
    project: Annotated[int, typer.Argument(help="Which project's pair to triangulate.")],
    reference_clip: Annotated[
        int | None,
        typer.Option("--reference-clip", help="Which clip sets the clock and the frame."),
    ] = None,
    target_clip: Annotated[
        int | None, typer.Option("--target-clip", help="The second view.")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction to use for both clips.")
    ] = None,
    window: Annotated[
        float | None, typer.Option("--window", help="Filtering window in seconds, for both clips.")
    ] = None,
    align: Annotated[
        bool,
        typer.Option("--align", help="Re-align the pair now instead of using the stored sync."),
    ] = False,
    nearest_frame: Annotated[
        bool,
        typer.Option(
            "--nearest-frame",
            help="Pair nearest frames instead of resampling. For comparison; measurably worse.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw report as JSON instead of tables.")
    ] = False,
) -> None:
    """Triangulate two calibrated, aligned views into 3D positions in metres."""
    params: dict[str, object] = {"project_id": project, "model": model, "align": align}
    if reference_clip is not None:
        params["reference_clip_id"] = reference_clip
    if target_clip is not None:
        params["target_clip_id"] = target_clip
    if window is not None:
        params["filter"] = {"smoothing": {"window_s": window}}
    if nearest_frame:
        params["reconstruction"] = {"resample": False}

    try:
        result = call("reconstruct", params)
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc

    assert isinstance(result, ReconstructionReport)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    _render_reconstruction(result)

    # Non-zero when nothing came out, so a script driving this learns that the
    # pair produced no positions rather than reading an empty report as success.
    if not result.reconstructed:
        raise typer.Exit(code=1)


def _render_club(result: ClubTrackingReport) -> None:
    """Coverage per phase first, because the aggregate rate is the misleading one.

    Laid out like the calibration and reconstruction reviews and for the same
    reason a third time: a reader shown one headline number will take it for the
    quality, and here that number is wrong in a predictable direction. Detection
    is easy where the club is slow, so the clip-wide rate is an average over the
    frames nobody wants to measure. The per-phase table is what the verdict rests
    on and it goes first.
    """
    summary = Table(title="Club tracking", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")

    summary.add_row("tracked", "[green]YES[/green]" if result.tracked else "[yellow]NO[/yellow]")
    summary.add_row("detector", f"{result.detector.name}  [dim]{result.detector.method}[/dim]")
    summary.add_row("frames", f"{result.tracked_frames} of {result.frame_count} carry a shaft")
    summary.add_row(
        "coverage",
        f"{result.coverage:.0%}  [dim](clip-wide -- see the per-phase table, not this)[/dim]",
    )
    if result.unanchored_frames:
        summary.add_row(
            "runs",
            f"{result.unanchored_frames}  [dim](separately seeded; a track in pieces is "
            "weaker than one piece)[/dim]",
        )
    if result.slow_motion_factor != 1.0:
        summary.add_row("slow motion", f"{result.slow_motion_factor:g}x  [dim]as supplied[/dim]")
    console.print(summary)

    if result.phase_coverage:
        phases = Table(title="Coverage by phase", title_justify="left", expand=True)
        phases.add_column("Phase", no_wrap=True)
        phases.add_column("Tracked", justify="right")
        phases.add_column("Coverage", justify="right")
        phases.add_column("Confidence", justify="right")
        phases.add_column("Club-head speed", justify="right")
        for entry in result.phase_coverage:
            style = (
                "green" if entry.coverage >= 0.8 else "yellow" if entry.coverage >= 0.5 else "red"
            )
            # The downswing is emphasised because it is the row the verdict rests
            # on, and it is the one a reader skims past on the way to the total.
            name = entry.phase.value
            if entry.phase is SwingPhase.DOWNSWING:
                name = f"[bold]{name}[/bold]"
            phases.add_row(
                name,
                f"{entry.tracked}/{entry.frames}",
                f"[{style}]{entry.coverage:.0%}[/{style}]",
                "-" if entry.median_confidence is None else f"{entry.median_confidence:.2f}",
                "-"
                if entry.median_tip_speed_px_s is None
                else f"{entry.median_tip_speed_px_s:,.0f} px/s",
            )
        console.print(phases)
    else:
        console.print(
            "[yellow]No swing was detected, so coverage is clip-wide only.[/yellow] "
            "[dim]That number is dominated by the frames where the club is nearly still.[/dim]"
        )

    quality = result.quality
    if quality is not None:
        detail = Table(title="Quality", title_justify="left", expand=True)
        detail.add_column("Measure", no_wrap=True)
        detail.add_column("Value", overflow="fold")
        detail.add_row(
            "support",
            f"{_opt(quality.median_support)} median  [dim](edge evidence -- the fit)[/dim]",
        )
        detail.add_row(
            "margin",
            f"{_opt(quality.median_margin)} median  [dim](over the best rival -- the check "
            "support cannot make)[/dim]",
        )
        detail.add_row(
            "continuity",
            f"{_opt(quality.median_continuity)} median  [dim](agreement with the predicted "
            "direction)[/dim]",
        )
        detail.add_row(
            "confidence", f"{_opt(quality.median_confidence)} median  [dim](the product)[/dim]"
        )
        detail.add_row(
            "shaft turn rate",
            f"{_rate(quality.median_angular_rate_deg_s)} median, "
            f"{_rate(quality.max_angular_rate_deg_s)} peak",
        )
        if quality.max_blur_px is not None:
            style = (
                "red"
                if quality.max_blur_px > 20
                else "yellow"
                if quality.max_blur_px > 10
                else "green"
            )
            detail.add_row(
                "club-head smear",
                f"[{style}]up to {quality.max_blur_px:.0f} px[/{style}]  [dim](at a 360-degree "
                "shutter; a shutter n times faster divides it by n, and nothing in the file "
                "records which)[/dim]",
            )
        detail.add_row(
            "club head seen",
            f"{quality.head_fraction:.0%} of tracked frames  [dim](elsewhere a shaft direction "
            "with no length)[/dim]",
        )
        console.print(detail)

    if result.impact is not None:
        impact = result.impact
        delta = "" if impact.delta_s is None else f", {impact.delta_s * 1000:+.0f} ms from it"
        console.print(
            f"[bold]Impact from the club head:[/bold] frame {impact.frame_index} "
            f"({impact.timestamp_s:.3f} s){delta}. "
            f"[dim]Phase 4's hand-speed estimate is frame {impact.kinematic_frame}; the two are "
            "independent and neither replaces the other.[/dim]"
        )

    if result.refusals:
        counts = ", ".join(
            f"{reason.value} {count}" for reason, count in sorted(result.refusals.items())
        )
        console.print(f"[dim]Frames carrying nothing, by reason: {counts}[/dim]")

    if result.refusal:
        console.print(f"[yellow]{result.refusal}[/yellow]")
    for note in result.warnings:
        console.print(f"[yellow]note:[/yellow] {note}")


def _opt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f} deg/s"


@app.command()
def club(
    video: Annotated[Path, typer.Argument(help="The clip. A video, not a pose file.")],
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction supplies the hand anchors.")
    ] = None,
    window: Annotated[
        float | None, typer.Option("--window", help="Filtering window in seconds.")
    ] = None,
    slow_motion: Annotated[
        float, typer.Option("--slow-motion", help="How many times slower than real time.")
    ] = 1.0,
    min_confidence: Annotated[
        float | None,
        typer.Option("--min-confidence", help="Below this a frame emits nothing."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw report as JSON instead of tables.")
    ] = False,
) -> None:
    """Track the club shaft through a clip, refusing where the evidence will not carry it."""
    params: dict[str, object] = {
        "path": str(video),
        "model": model,
        "slow_motion_factor": slow_motion,
    }
    if window is not None:
        params["filter"] = {"smoothing": {"window_s": window}}
    if min_confidence is not None:
        params["club"] = {"min_confidence": min_confidence}

    try:
        result = _run_with_progress("track_club", params, "Tracking the club")
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc

    assert isinstance(result, ClubTrackingReport)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    _render_club(result)

    # Non-zero when nothing was tracked, so a script driving this learns that the
    # clip produced no shaft rather than reading an empty report as success.
    if not result.tracked:
        raise typer.Exit(code=1)


def _render_ball(result: BallTrackingReport) -> None:
    """The departure first, because it is the only thing here anyone wants.

    Laid out unlike every other review in this file, and deliberately. The club,
    the calibration and the reconstruction reviews all lead with a table of how
    well the thing was measured, because in each of those the measurement *is* the
    output. Here the output is one instant, the per-frame track exists only to
    support it, and a reader shown the coverage first would be reading the working
    rather than the answer.
    """
    if result.departure is not None:
        departure = result.departure
        confidence = departure.confidence
        style = "green" if confidence.overall >= 0.6 else "yellow"
        console.print(
            Panel(
                f"Impact is between frame [bold]{departure.last_seen_frame}[/bold] "
                f"({departure.last_seen_s:.3f} s), the last frame carrying a ball, and frame "
                f"[bold]{departure.first_absent_frame}[/bold] ({departure.first_absent_s:.3f} s), "
                f"the first carrying none.\n"
                f"That bracket is [bold]{departure.interval_s * 1000:.1f} ms[/bold] wide -- one "
                "frame interval, and impact is inside it.\n"
                f"Confidence [{style}]{confidence.overall:.2f}[/{style}]  [dim]= established "
                f"{confidence.establishment:.2f} x abrupt {confidence.abruptness:.2f} x "
                f"permanent {confidence.permanence:.2f}[/dim]",
                title="Ball departure -- an observation, not an estimate",
                title_align="left",
                border_style=style,
            )
        )
        if departure.kinematic_frame is not None and departure.delta_s is not None:
            direction = "early" if departure.delta_s > 0 else "late"
            console.print(
                f"[dim]Phase 4's hand-speed peak is frame {departure.kinematic_frame}, which runs "
                f"{abs(departure.delta_s) * 1000:.0f} ms {direction} against this. "
                "`analyzer impact` reconciles every estimate.[/dim]"
            )
    elif result.established is not None:
        console.print(
            "[yellow]A ball was found and it never left.[/yellow] [dim]No impact is reported "
            "from it -- a practice swing, a clip that ends before contact, and a miss all look "
            "exactly like this.[/dim]"
        )

    summary = Table(title="Ball detection", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")
    summary.add_row("found", "[green]YES[/green]" if result.detected else "[yellow]NO[/yellow]")
    summary.add_row("detector", f"{result.detector.name}  [dim]{result.detector.method}[/dim]")
    summary.add_row("frames", f"{result.observed_frames} of {result.frame_count} carry a ball")
    summary.add_row(
        "coverage before departure",
        f"{result.pre_departure_coverage:.0%}  [dim](after it the ball is correctly absent, "
        "so a clip-wide rate would count the follow-through against it)[/dim]",
    )
    if result.slow_motion_factor != 1.0:
        summary.add_row("slow motion", f"{result.slow_motion_factor:g}x  [dim]as supplied[/dim]")
    console.print(summary)

    established = result.established
    if established is not None:
        detail = Table(title="Identification", title_justify="left", expand=True)
        detail.add_column("Measure", no_wrap=True)
        detail.add_column("Value", overflow="fold")
        detail.add_row(
            "agreed on by",
            f"{established.frames} of {established.searched_frames} frames searched",
        )
        margin_style = (
            "green"
            if established.margin >= 0.5
            else "yellow"
            if established.margin >= 0.2
            else "red"
        )
        rival = (
            ""
            if established.runner_up_distance_torso is None
            else f"  [dim](next-best stationary candidate "
            f"{established.runner_up_distance_torso:.2f} torso lengths away)[/dim]"
        )
        detail.add_row(
            "margin",
            f"[{margin_style}]{established.margin:.2f}[/{margin_style}]{rival}",
        )
        detail.add_row(
            "size",
            f"{established.radius_torso:.3f} torso lengths radius  [dim](a golf ball is "
            "0.047)[/dim]",
        )
        detail.add_row(
            "scatter",
            f"{established.spread_torso:.4f} torso lengths  [dim](a teed ball scatters by the "
            "detector's noise and nothing else)[/dim]",
        )
        console.print(detail)

    quality = result.quality
    if quality is not None:
        detail = Table(title="Quality", title_justify="left", expand=True)
        detail.add_column("Measure", no_wrap=True)
        detail.add_column("Value", overflow="fold")
        detail.add_row(
            "contrast",
            f"{_opt(quality.median_contrast)} median  [dim](against the ring around it -- "
            "the fit)[/dim]",
        )
        detail.add_row(
            "margin",
            f"{_opt(quality.median_margin)} median  [dim](over a co-located rival)[/dim]",
        )
        detail.add_row(
            "stillness",
            f"{_opt(quality.median_stillness)} median  [dim](against the established "
            "position)[/dim]",
        )
        detail.add_row(
            "confidence", f"{_opt(quality.median_confidence)} median  [dim](the product)[/dim]"
        )
        if quality.radius_px is not None:
            style = (
                "red" if quality.radius_px < 3 else "yellow" if quality.radius_px < 5 else "green"
            )
            detail.add_row(
                "ball size on the sensor",
                f"[{style}]{quality.radius_px:.1f} px[/{style}] radius  [dim](under about 3 px "
                "a disc has no shape left to be round)[/dim]",
            )
        if quality.drift_torso is not None:
            detail.add_row(
                "camera drift",
                f"{quality.drift_torso:.3f} torso lengths  [dim](the ball did not move, so "
                "this is the camera: handheld, stabilised, or panned)[/dim]",
            )
        console.print(detail)

    if result.refusals:
        counts = ", ".join(
            f"{reason.value} {count}" for reason, count in sorted(result.refusals.items())
        )
        console.print(f"[dim]Frames carrying no ball, by reason: {counts}[/dim]")

    if result.refusal:
        console.print(f"[yellow]{result.refusal}[/yellow]")
    for note in result.warnings:
        console.print(f"[yellow]note:[/yellow] {note}")


@app.command()
def ball(
    video: Annotated[Path, typer.Argument(help="The clip. A video, not a pose file.")],
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction places the search region.")
    ] = None,
    window: Annotated[
        float | None, typer.Option("--window", help="Filtering window in seconds.")
    ] = None,
    slow_motion: Annotated[
        float, typer.Option("--slow-motion", help="How many times slower than real time.")
    ] = 1.0,
    min_confidence: Annotated[
        float | None,
        typer.Option("--min-confidence", help="Below this a frame emits nothing."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw report as JSON instead of tables.")
    ] = False,
) -> None:
    """Find the ball, and the frame at which it stopped being there."""
    params: dict[str, object] = {
        "path": str(video),
        "model": model,
        "slow_motion_factor": slow_motion,
    }
    if window is not None:
        params["filter"] = {"smoothing": {"window_s": window}}
    if min_confidence is not None:
        params["ball"] = {"min_confidence": min_confidence}

    try:
        result = _run_with_progress("detect_ball", params, "Looking for the ball")
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc

    assert isinstance(result, BallTrackingReport)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    _render_ball(result)

    # Non-zero when no ball was found at all, so a script driving this learns that
    # the clip produced nothing rather than reading an empty report as success. A
    # ball that was found and did not depart is a success: it is the right answer
    # for a practice swing.
    if not result.detected:
        raise typer.Exit(code=1)


def _render_impact(result: FusedImpact) -> None:
    """One instant, then every method's disagreement with it.

    The disagreements are a table rather than a footnote because they are the
    measurement this command exists to make: how far the estimates that do not
    see the ball sit from the one that does.
    """
    style = "green" if result.observed else "yellow"
    kind = "observed" if result.observed else "inferred"
    bracket = (
        f"impact is inside a {result.uncertainty_s * 1000:.1f} ms bracket"
        if result.uncertainty_is_bracket and result.uncertainty_s is not None
        else f"located within about {result.uncertainty_s * 1000:.0f} ms"
        if result.uncertainty_s is not None
        else "with no error bar available"
    )
    console.print(
        Panel(
            f"Frame [bold]{result.frame_index}[/bold] at [bold]{result.timestamp_s:.3f} s[/bold], "
            f"from [bold]{result.source.value}[/bold] -- {bracket}.\n"
            f"[dim]{result.methodology}[/dim]",
            title=f"Impact ({kind})",
            title_align="left",
            border_style=style,
        )
    )

    table = Table(title="Every estimate, and its distance from the one above", expand=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Frame", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Uncertainty", justify="right")
    table.add_column("Confidence", justify="right")
    for candidate in result.candidates:
        chosen = candidate.source is result.source
        name = f"[bold]{candidate.source.value}[/bold]" if chosen else candidate.source.value
        delta = (
            "[dim]reported[/dim]"
            if chosen
            else "-"
            if candidate.delta_s is None
            else f"{candidate.delta_s * 1000:+.0f} ms / {candidate.delta_frames:+d} f"
        )
        table.add_row(
            name,
            str(candidate.frame_index),
            f"{candidate.timestamp_s:.3f} s",
            delta,
            "-" if candidate.uncertainty_s is None else f"{candidate.uncertainty_s * 1000:.0f} ms",
            "-" if candidate.confidence is None else f"{candidate.confidence:.2f}",
        )
    console.print(table)

    if not result.observed:
        console.print(
            "[dim]Only the ball_departure row is an observation. The rest are read off the "
            "player's motion, and their uncertainties are a scale rather than a bracket -- "
            "nothing guarantees the instant is inside them.[/dim]"
        )
    for note in result.warnings:
        console.print(f"[yellow]note:[/yellow] {note}")


@app.command()
def impact(
    video: Annotated[Path, typer.Argument(help="The clip. A video, not a pose file.")],
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction supplies the landmarks.")
    ] = None,
    window: Annotated[
        float | None, typer.Option("--window", help="Filtering window in seconds.")
    ] = None,
    slow_motion: Annotated[
        float, typer.Option("--slow-motion", help="How many times slower than real time.")
    ] = 1.0,
    with_club: Annotated[
        bool,
        typer.Option(
            "--club/--no-club",
            help="Also track the club for its independent estimate. Costs a second pass.",
        ),
    ] = True,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw result as JSON instead of tables.")
    ] = False,
) -> None:
    """Reconcile every impact estimate for a clip into one instant with one provenance."""
    params: dict[str, object] = {
        "path": str(video),
        "model": model,
        "slow_motion_factor": slow_motion,
        "with_club": with_club,
    }
    if window is not None:
        params["filter"] = {"smoothing": {"window_s": window}}

    try:
        result = _run_with_progress("locate_impact", params, "Locating impact")
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc

    assert isinstance(result, FusedImpact)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    _render_impact(result)

    # Exit zero even when the instant was inferred rather than seen. Every other
    # command here reserves a non-zero exit for "produced nothing usable", and an
    # inference clearly labelled as one is usable -- a clip with no visible ball
    # is the ordinary case, not a failure. A script that needs an *observation*
    # reads `observed` out of `--json`, which exists for exactly that.


# --- Phase 12: labelling, datasets and the learned detector -----------------
#
# These do not go through `dispatch.call`. Every command above is a thin shell
# over an RPC method because the desktop app calls the same method; nothing here
# is reachable from the app, and will not be -- training is a developer operation
# over a corpus that does not ship. Routing it through the IPC boundary would add
# a schema to version, a worker to spawn and a progress protocol to thread,
# in exchange for a capability no UI has.
#
# The imports are function-local for the reason `dispatch` uses them: torch is
# several seconds of import, and `analyzer doctor` must not pay for it.


@app.command()
def label(
    video: Annotated[Path, typer.Argument(help="The clip to label.")],
    player: Annotated[str, typer.Option("--player", help="Who is swinging. Required.")],
    session: Annotated[str, typer.Option("--session", help="Which capture sitting. Required.")],
    labeller: Annotated[str, typer.Option("--labeller", help="Who is marking. Required.")],
    swing: Annotated[
        str | None, typer.Option("--swing", help="Id within the session. Defaults to the filename.")
    ] = None,
    slow_motion: Annotated[
        float, typer.Option("--slow-motion", help="How many times slower than real time.")
    ] = 1.0,
    labels: Annotated[
        Path | None,
        typer.Option("--labels", help="Where to write. Defaults to the data directory."),
    ] = None,
) -> None:
    """Step a clip frame by frame and mark the four swing events.

    Player and session are required and have no defaults. They are the grouping
    keys a held-out split is built from, and a wrong one is invisible: the model
    trains on a golfer it will later be tested against, and every number after
    that is inflated with nothing to show for it.
    """
    from analyzer.ingestion import ProbeError, probe_video
    from analyzer.ingestion.reader import open_video
    from analyzer.ml.labels import write_label
    from analyzer.ml.labeltool import LabelSession, run_window

    try:
        metadata = probe_video(video)
    except ProbeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    # `measured_fps` rather than the container's declared rate: a label's
    # uncertainty is in frames and only becomes a time through this number, so it
    # has to be the rate the file actually has. None on a one-frame clip, which
    # the session refuses.
    session_state = LabelSession(
        clip_path=video,
        frames=metadata.timing.frame_count,
        fps=metadata.timing.measured_fps or metadata.timing.nominal_fps or 0.0,
        player_id=player,
        session_id=session,
        swing_id=swing or video.stem,
        labeller=labeller,
        slow_motion_factor=slow_motion,
        content_key=metadata.content_key,
    )

    written: list[Path] = []

    def save(produced: object) -> None:
        written.append(write_label(produced, labels))  # type: ignore[arg-type]

    source = open_video(video)
    try:
        run_window(
            session_state,
            _FrameWindow(source),
            window=f"label {video.name}",
            on_save=save,
        )
    finally:
        source.close()

    if written:
        console.print(f"[green]Saved[/green] {written[-1]}")
        return
    if session_state.dirty:
        console.print(
            "[yellow]Nothing was saved; the marks made in this session are gone.[/yellow]"
        )
        raise typer.Exit(code=1)
    console.print("[dim]Nothing marked.[/dim]")


class _FrameWindow:
    """Frame lookup by index for the labelling window.

    A three-line adapter so `labeltool` never imports the ingestion layer: the
    tool works on images, and which decoder produced them is the caller's
    business. BGR because that is what OpenCV draws and displays.
    """

    def __init__(self, source: object) -> None:
        self._source = source

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        import cv2  # type: ignore[import-untyped]

        frame = self._source.frame_at(index)  # type: ignore[attr-defined]
        return cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)


def _render_label_set(label_set: object) -> None:
    """The set, its groups, and whether it can be split at all."""
    from analyzer.ml.splits import plan_split

    clips = label_set.clips  # type: ignore[attr-defined]
    table = Table(title="Labelled clips", expand=True)
    table.add_column("Player", no_wrap=True)
    table.add_column("Sessions", justify="right")
    table.add_column("Clips", justify="right")
    table.add_column("Swings", justify="right")
    table.add_column("No swing", justify="right")
    for player in sorted({clip.player_id for clip in clips}):
        owned = [clip for clip in clips if clip.player_id == player]
        table.add_row(
            player,
            str(len({clip.session_key for clip in owned})),
            str(len(owned)),
            str(sum(1 for clip in owned if clip.is_swing)),
            str(sum(1 for clip in owned if not clip.is_swing)),
        )
    console.print(table)
    console.print(
        f"[dim]{len(clips)} clips, {len(label_set.players)} players, "  # type: ignore[attr-defined]
        f"{len(label_set.sessions)} sessions, provenance "  # type: ignore[attr-defined]
        f"{label_set.provenance.value}, digest {label_set.digest()[:12]}[/dim]"  # type: ignore[attr-defined]
    )

    plan = plan_split(list(clips), label_digest=label_set.digest())  # type: ignore[attr-defined]
    if plan.refused:
        console.print(
            Panel(
                plan.refusal or "",
                title="No split is possible",
                title_align="left",
                border_style="yellow",
            )
        )
        return
    counts = ", ".join(f"{role.value} {count}" for role, count in plan.clips.items())
    console.print(f"[green]A split exists[/green]: {counts}")
    for note in plan.warnings:
        console.print(f"[yellow]note:[/yellow] {note}")


@app.command()
def labels(
    directory: Annotated[
        Path | None,
        typer.Option("--labels", help="Where the labels are. Defaults to the data directory."),
    ] = None,
) -> None:
    """List the labelled set, and say whether it can support a held-out split."""
    from analyzer.ml.labels import LabelStoreError, load_label_set

    try:
        label_set = load_label_set(directory, strict=False)
    except LabelStoreError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not label_set.clips:
        console.print(
            Panel(
                "No labels yet. Phase 12's machinery is built and has nothing to run on.\n"
                "Produce one with: analyzer label <clip> --player <id> --session <id> "
                "--labeller <id>",
                title="Empty label set",
                title_align="left",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1)

    _render_label_set(label_set)


def _render_dataset(summary: object) -> None:
    table = Table(title="Dataset", expand=True)
    table.add_column("Quantity", no_wrap=True)
    table.add_column("Value", justify="right")
    rows = (
        ("clips", f"{summary.clips} ({summary.swings} swings, {summary.non_swings} without)"),  # type: ignore[attr-defined]
        ("players / sessions", f"{summary.players} / {summary.sessions}"),  # type: ignore[attr-defined]
        ("samples", f"{summary.samples} at {summary.sample_rate_hz:g} Hz"),  # type: ignore[attr-defined]
        ("class imbalance", f"{summary.imbalance_ratio:.1f}x"),  # type: ignore[attr-defined]
        ("feature spec", f"v{summary.feature_version} {summary.feature_digest[:12]}"),  # type: ignore[attr-defined]
        ("label digest", summary.label_digest[:12]),  # type: ignore[attr-defined]
        ("provenance", summary.provenance.value),  # type: ignore[attr-defined]
        (
            "resolution floor",
            f"{summary.resample_floor_ms:.1f} ms grid"  # type: ignore[attr-defined]
            + (
                f" + {summary.median_label_uncertainty_ms:.1f} ms labels"  # type: ignore[attr-defined]
                if summary.median_label_uncertainty_ms is not None  # type: ignore[attr-defined]
                else ""
            ),
        ),
    )
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)

    classes = Table(title="Samples per class", expand=True)
    classes.add_column("Class", no_wrap=True)
    classes.add_column("Samples", justify="right")
    classes.add_column("Share", justify="right")
    for frame_class, count in summary.class_samples.items():  # type: ignore[attr-defined]
        classes.add_row(
            frame_class.value,
            str(count),
            f"{summary.class_fraction[frame_class]:.1%}",  # type: ignore[attr-defined]
        )
    console.print(classes)

    for entry in summary.dropped:  # type: ignore[attr-defined]
        console.print(f"[yellow]dropped:[/yellow] {entry}")
    for note in summary.warnings:  # type: ignore[attr-defined]
        console.print(f"[yellow]note:[/yellow] {note}")


@app.command()
def dataset(
    directory: Annotated[
        Path | None, typer.Option("--labels", help="Where the labels are.")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction supplies the landmarks.")
    ] = None,
    search: Annotated[
        list[Path] | None,
        typer.Option("--search", help="Directories to look for moved footage in."),
    ] = None,
) -> None:
    """Build the training arrays from the labelled set, and report what they are."""
    from analyzer.ml.dataset import DatasetError, build_dataset
    from analyzer.ml.labels import LabelStoreError, load_label_set
    from analyzer.ml.provider import cached_pose_provider

    try:
        label_set = load_label_set(directory)
        built = build_dataset(label_set, cached_pose_provider(model, search=tuple(search or ())))
    except (DatasetError, LabelStoreError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    _render_dataset(built.summary)


def _render_split(report: object) -> None:
    table = Table(title="Split, grouped by player", expand=True)
    table.add_column("Role", no_wrap=True)
    table.add_column("Players", justify="right")
    table.add_column("Clips", justify="right")
    for role, count in report.clips.items():  # type: ignore[attr-defined]
        table.add_row(role.value, str(report.players[role]), str(count))  # type: ignore[attr-defined]
    console.print(table)

    leak = report.leakage  # type: ignore[attr-defined]
    if leak is not None:
        style = "green" if leak.passed else "bold red"
        console.print(
            f"[{style}]leak check: players {leak.player_overlap}, sessions "
            f"{leak.session_overlap}, clips {leak.clip_overlap}[/{style}]"
        )
    for note in report.warnings:  # type: ignore[attr-defined]
        console.print(f"[yellow]note:[/yellow] {note}")


def _render_evaluation(report: object, title: str) -> None:
    table = Table(title=title, expand=True)
    table.add_column("Event", no_wrap=True)
    table.add_column("Scored", justify="right")
    table.add_column("Missed", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("Bias", justify="right")
    table.add_column("In bracket", justify="right")
    for error in report.events:  # type: ignore[attr-defined]
        table.add_row(
            error.event.value,
            str(error.scored),
            str(error.missed),
            "-" if error.mae_ms is None else f"{error.mae_ms:.1f} ms",
            "-" if error.bias_ms is None else f"{error.bias_ms:+.1f} ms",
            "-"
            if error.within_label_uncertainty is None
            else f"{error.within_label_uncertainty:.0%}",
        )
    console.print(table)
    console.print(
        f"[dim]macro F1 {report.macro_f1:.3f} over {report.samples} tracked samples; "  # type: ignore[attr-defined]
        f"this set resolves {report.noise_floor_ms:.1f} ms[/dim]"  # type: ignore[attr-defined]
    )
    if not report.claims_permitted:  # type: ignore[attr-defined]
        console.print(
            Panel(
                report.claim_refusal or "",  # type: ignore[attr-defined]
                title="Not publishable",
                title_align="left",
                border_style="yellow",
            )
        )
    for note in report.warnings:  # type: ignore[attr-defined]
        console.print(f"[yellow]note:[/yellow] {note}")


@app.command()
def train(
    directory: Annotated[
        Path | None, typer.Option("--labels", help="Where the labels are.")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction supplies the landmarks.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Everything random is seeded from this.")] = 0,
    epochs: Annotated[
        int | None, typer.Option("--epochs", help="Maximum passes over the set.")
    ] = None,
    device: Annotated[
        str, typer.Option("--device", help="torch device. cpu is the default.")
    ] = "cpu",
    search: Annotated[
        list[Path] | None,
        typer.Option("--search", help="Directories to look for moved footage in."),
    ] = None,
) -> None:
    """Train the learned detector, score it against the rules, and register it.

    Every step refuses rather than improvising: no labels, no dataset; too few
    players, no split; synthetic or small held-out set, no publishable number.
    A run that gets all the way through still prints what its score may not be
    used for.
    """
    from analyzer.contracts.ml import ModelCard, TCNConfig
    from analyzer.ml.compare import compare_detectors
    from analyzer.ml.dataset import DatasetError, build_dataset
    from analyzer.ml.labels import LabelStoreError, load_label_set
    from analyzer.ml.provider import cached_pose_provider
    from analyzer.ml.registry import model_id, save_model
    from analyzer.ml.splits import SplitRefused, split_dataset
    from analyzer.ml.train import train as run_training

    try:
        label_set = load_label_set(directory)
        built = build_dataset(label_set, cached_pose_provider(model, search=tuple(search or ())))
    except (DatasetError, LabelStoreError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    _render_dataset(built.summary)

    try:
        split = split_dataset(built, seed=seed)
    except SplitRefused as exc:
        console.print(
            Panel(
                exc.report.refusal or "",
                title="Refusing to split",
                title_align="left",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1) from exc

    _render_split(split.report)

    config = TCNConfig() if epochs is None else TCNConfig(epochs=epochs)
    identifier = model_id(
        label_digest=built.summary.label_digest,
        feature_digest=built.summary.feature_digest,
        seed=seed,
        config=config,
    )
    with console.status("Training"):
        trained, report = run_training(
            split.train,
            split.val,
            config=config,
            seed=seed,
            device=device,
            label_digest=built.summary.label_digest,
        )
    console.print(
        f"[dim]{report.parameters} parameters, receptive field "
        f"{report.receptive_field_samples} samples ({report.receptive_field_s:.2f} s), "
        f"best epoch {report.best_epoch} of {len(report.epochs)}, "
        f"{report.elapsed_s:.1f} s[/dim]"
    )
    for note in report.warnings:
        console.print(f"[yellow]note:[/yellow] {note}")

    comparison = compare_detectors(
        trained,
        split.test,
        model_id=identifier,
        device=device,
        label_digest=built.summary.label_digest,
    )
    _render_evaluation(comparison.rule, "Rule-based detector, held-out clips")
    _render_evaluation(comparison.model, "Learned detector, held-out clips")

    rows = Table(title="Same clips, same metric", expand=True)
    rows.add_column("Event", no_wrap=True)
    rows.add_column("Rules", justify="right")
    rows.add_column("Model", justify="right")
    rows.add_column("Difference", justify="right")
    rows.add_column("Verdict", no_wrap=True)
    for row in comparison.rows:
        rows.add_row(
            row.event.value,
            "-" if row.rule_mae_ms is None else f"{row.rule_mae_ms:.1f} ms",
            "-" if row.model_mae_ms is None else f"{row.model_mae_ms:.1f} ms",
            "-" if row.difference_ms is None else f"{row.difference_ms:+.1f} ms",
            row.verdict,
        )
    console.print(rows)
    console.print(comparison.verdict)

    card = ModelCard(
        model_id=identifier,
        created_at=datetime.now(UTC).isoformat(),
        architecture="dilated TCN, per-sample classification",
        config=config,
        feature_spec=built.spec,
        feature_digest=built.summary.feature_digest,
        label_digest=built.summary.label_digest,
        dataset=built.summary,
        split=split.report,
        training=report,
        evaluation=comparison.model,
    )
    directory_written = save_model(trained, card)
    console.print(f"[green]Registered[/green] {identifier} in {directory_written}")


@app.command()
def models() -> None:
    """List the trained models, and what each one is allowed to claim."""
    from analyzer.ml.registry import list_models

    cards = list_models()
    if not cards:
        console.print("[dim]No trained models. Run `analyzer train`.[/dim]")
        raise typer.Exit(code=1)

    table = Table(title="Model registry", expand=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Labels", no_wrap=True)
    table.add_column("Features", no_wrap=True)
    table.add_column("Held out", justify="right")
    table.add_column("Macro F1", justify="right")
    table.add_column("Publishable", no_wrap=True)
    for card in cards:
        evaluation = card.evaluation
        table.add_row(
            card.model_id,
            card.label_digest[:8],
            card.feature_digest[:8],
            "-" if evaluation is None else f"{evaluation.clips} clips / {evaluation.players}p",
            "-" if evaluation is None else f"{evaluation.macro_f1:.3f}",
            "-" if evaluation is None else ("yes" if evaluation.claims_permitted else "no"),
        )
    console.print(table)


_REFUSAL_TAG: dict[FindingRefusal, str] = {
    FindingRefusal.NO_SWING: "no swing",
    FindingRefusal.NO_METRIC: "not measured",
    FindingRefusal.NO_THRESHOLD: "no threshold exists",
    FindingRefusal.BASIS_NOT_PERMITTED: "wrong kind of number",
    FindingRefusal.VIEW_MISMATCH: "wrong camera",
    FindingRefusal.SUPPLIED_TIMEBASE: "supplied timebase",
    FindingRefusal.LOW_CONFIDENCE: "low confidence",
    FindingRefusal.NO_UNCERTAINTY: "no uncertainty",
    FindingRefusal.UNRESOLVED: "cannot resolve",
}

_COMPARISON_STYLE: dict[Comparison, str] = {
    Comparison.BELOW: "yellow",
    Comparison.WITHIN: "green",
    Comparison.ABOVE: "yellow",
}


def _render_coaching(report: CoachingReport) -> None:
    summary = Table(title="Coaching", title_justify="left", expand=True)
    summary.add_column("Property", no_wrap=True)
    summary.add_column("Value", overflow="fold")
    summary.add_row("rules considered", str(report.rules_considered))
    summary.add_row("findings", f"{len(report.findings)}")
    summary.add_row("refused", f"{len(report.refused)}")
    summary.add_row("camera view", report.view.value)
    if report.frame_interval_s is not None:
        summary.add_row(
            "clock resolution",
            f"{report.frame_interval_s * 1000:.1f} ms between frames "
            f"({1.0 / report.frame_interval_s:.0f} per real second)",
        )
    phrasing = report.phrasing
    summary.add_row(
        "phrasing",
        f"{phrasing.mode.value}"
        + (f" via {phrasing.provider}" if phrasing.provider else "")
        + (
            f" — {phrasing.accepted} kept, {phrasing.rejected} rejected"
            if phrasing.attempted
            else ""
        ),
    )
    console.print(summary)

    for finding in report.findings:
        style = _COMPARISON_STYLE[finding.comparison]
        body = [
            finding.observation,
            "",
            f"[dim]source:[/dim] {finding.source.citation}",
            f"[dim]measured on:[/dim] {finding.source.population}",
        ]
        if finding.phrased is not None:
            body.insert(1, f"\n[dim]phrased:[/dim] {finding.phrased}")
        for item in finding.evidence:
            shown = item.frames if len(item.frames) <= 6 else [item.frames[0], item.frames[-1]]
            span = (
                f"frames {shown[0]}-{shown[-1]}"
                if len(item.frames) > 6
                else "frames " + ", ".join(str(frame) for frame in shown)
            )
            body.append(
                f"[dim]evidence:[/dim] {item.label} = "
                f"{_format_measure(item.value, item.unit, item.uncertainty)} "
                f"({span}, confidence {item.confidence:.2f})"
            )
        console.print(
            Panel(
                "\n".join(body),
                title=f"[{style}]{finding.comparison.value}[/{style}] {finding.title}",
                title_align="left",
                subtitle=f"[dim]{finding.rule_id}[/dim]",
                subtitle_align="right",
            )
        )

    if report.refused:
        refusals = Table(title="Refused", title_justify="left", expand=True)
        refusals.add_column("rule", no_wrap=True)
        # The tag rides with the reason rather than taking a column of its own:
        # the reasons are the readable part and a third narrow column squeezes
        # them into a ribbon on an eighty-column terminal.
        refusals.add_column("why", overflow="fold", ratio=2)
        for entry in report.refused:
            refusals.add_row(
                entry.rule_id,
                f"[yellow]{_REFUSAL_TAG[entry.refusal]}[/yellow] — {entry.reason}",
            )
        console.print(refusals)

    for rejection in report.phrasing.rejections:
        console.print(
            f"[red]guard[/red] {rejection.rule_id}: {rejection.offence} "
            f"({rejection.token!r}) — the engine's own sentence was used instead."
        )
    if report.phrasing.unavailable:
        console.print(f"[yellow]![/yellow] {report.phrasing.unavailable}")

    console.print(
        "[dim]Every finding above cites the frames its numbers were measured on. "
        "No sentence here contains a number absent from that evidence — the same "
        "guard is applied to the engine's own words and to any model's.\n"
        "There is no score: a single number summarising a swing would need a scale "
        "relating degrees of turn to seconds of tempo, and nobody has measured one.[/dim]"
    )

    for warning in report.warnings:
        console.print(f"[yellow]![/yellow] {warning}")


@app.command()
def coach(
    path: Annotated[
        Path, typer.Argument(help="Pose Parquet file, or the video it was extracted from.")
    ],
    model: Annotated[
        str | None, typer.Option("--model", help="Which extraction to use, when given a video.")
    ] = None,
    window: Annotated[
        float | None, typer.Option("--window", help="Filtering window in seconds.")
    ] = None,
    slow_motion: Annotated[
        float,
        typer.Option(
            "--slow-motion",
            help="How many times slower than real time the clip plays (8 for 8x slo-mo).",
        ),
    ] = 1.0,
    project: Annotated[
        int | None,
        typer.Option("--project", help="Apply this project's camera calibration."),
    ] = None,
    phrasing: Annotated[
        str | None,
        typer.Option(
            "--phrase-with",
            help=(
                "Base URL of a language model on THIS machine to reword the findings. "
                "Loopback only. Off by default, and off is complete."
            ),
        ),
    ] = None,
    phrasing_model: Annotated[
        str | None, typer.Option("--phrase-model", help="Which local model to ask for.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw report as JSON instead of tables.")
    ] = False,
) -> None:
    """Reach the conclusions a swing's measurements support, and refuse the rest."""
    params: dict[str, object] = {
        "path": str(path),
        "model": model,
        "slow_motion_factor": slow_motion,
    }
    if window is not None:
        params["filter"] = {"smoothing": {"window_s": window}}
    if project is not None:
        params["project_id"] = project
    if phrasing is not None:
        params["coaching"] = {
            "phrasing": "local",
            "phrasing_endpoint": phrasing,
            "phrasing_model": phrasing_model,
        }

    try:
        result = call("coach_swing", params)
    except EngineError as exc:
        console.print(f"[red]{exc}[/red]")
        remediation = (exc.data or {}).get("remediation")
        if remediation:
            console.print(f"[dim]fix: {remediation}[/dim]")
        raise typer.Exit(code=1) from exc

    assert isinstance(result, CoachingReport)  # noqa: S101 - narrows the dispatch return type

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    _render_coaching(result)

    # Non-zero when no conclusion was reached, so a script driving this learns
    # that the swing produced nothing rather than reading silence as approval.
    if not result.findings:
        raise typer.Exit(code=1)


# executed as `python -m analyzer.cli` -- the commands exist under the installed
# `analyzer` entry point, which imports the module fully first, and silently do
# not under the module form. Keeping the two entry points equivalent means this
# stays at the bottom.
if __name__ == "__main__":
    app()
