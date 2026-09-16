"""Command-line entry point.

Shares the dispatch table with the RPC worker, so the analysis engine is fully
usable from a terminal without the desktop app. That matters for debugging, for
CI, and for benchmark scripts, none of which should need Rust in the loop.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from analyzer import __version__
from analyzer.contracts.health import EnvironmentReport, HealthStatus
from analyzer.dispatch import call

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


@app.command()
def version() -> None:
    """Print the engine version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
