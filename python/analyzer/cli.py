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
        str, typer.Option("--space", help="Coordinate space: image or hip_local.")
    ] = "image",
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


@app.command()
def version() -> None:
    """Print the engine version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()


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
