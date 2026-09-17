"""Command-line entry point.

Shares the dispatch table with the RPC worker, so the analysis engine is fully
usable from a terminal without the desktop app. That matters for debugging, for
CI, and for benchmark scripts, none of which should need Rust in the loop.
"""

from __future__ import annotations

import json
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
from analyzer.contracts.filtering import SequenceFilterReport
from analyzer.contracts.health import EnvironmentReport, HealthStatus
from analyzer.contracts.metrics import (
    CameraView,
    Metric,
    MetricBasis,
    MetricGroup,
    MetricSet,
    MetricUnit,
)
from analyzer.contracts.phases import SwingPhases
from analyzer.contracts.pose import PoseExtractionResult
from analyzer.contracts.progress import ProgressUpdate
from analyzer.contracts.projects import Project, ProjectList
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
}

_GROUP_TITLES: dict[MetricGroup, str] = {
    MetricGroup.POSTURE: "Posture",
    MetricGroup.ROTATION: "Rotation",
    MetricGroup.ARMS: "Hands and arms",
    MetricGroup.TIMING: "Timing",
}


def _format_value(metric: Metric) -> str:
    """Render a value with its unit, at a precision the measurement can support.

    The uncertainty rides alongside where there is one. A turn of 53 degrees and
    a turn of 53 give-or-take 13 are different findings, and only one of them is
    worth telling a player.
    """
    if metric.unit is MetricUnit.DEGREES:
        if metric.uncertainty is not None:
            return f"{metric.value:+.1f} +/-{metric.uncertainty:.0f} deg"
        return f"{metric.value:+.1f} deg"
    if metric.unit is MetricUnit.SECONDS:
        return f"{metric.value:.3f} s"
    if metric.unit is MetricUnit.RATIO:
        return f"{metric.value:.2f} : 1"
    if metric.unit is MetricUnit.TORSO_LENGTHS_PER_S:
        return f"{metric.value:.2f} torso/s"
    return f"{metric.value:+.3f} torso"


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
        "foreshort = rotation inferred from foreshortening, a magnitude only.\n"
        "Lengths are in torso lengths, which is framing-independent but not metric. "
        "Confidence is observation x anchor x method.[/dim]"
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
# executed as `python -m analyzer.cli` -- the commands exist under the installed
# `analyzer` entry point, which imports the module fully first, and silently do
# not under the module form. Keeping the two entry points equivalent means this
# stays at the bottom.
if __name__ == "__main__":
    app()
