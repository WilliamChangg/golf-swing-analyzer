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
from analyzer.contracts.rpc import EngineError
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

    params: dict[str, object] = {"path": str(path), "space": space, "model": model}
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

    params: dict[str, object] = {"path": str(path), "model": model}
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
    """Render a value with its unit, at a precision the measurement can support."""
    if metric.unit is MetricUnit.DEGREES:
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

    params: dict[str, object] = {"path": str(path), "model": model}
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
