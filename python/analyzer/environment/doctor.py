"""Assembles the environment health report.

This is the only analysis-engine method implemented in Phase 0, chosen because
it exercises the entire architecture end-to-end -- desktop UI, Tauri command,
Rust engine client, stdio JSON-RPC, typed contract -- while reporting only facts
that were actually measured on the running machine.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime

from analyzer.contracts.health import (
    ComponentStatus,
    EnvironmentReport,
    HealthStatus,
    aggregate_status,
)
from analyzer.environment.ffmpeg import list_hwaccels, probe_ffmpeg, probe_ffprobe
from analyzer.environment.hardware import (
    DEVICE_CPU,
    compute_info,
    platform_info,
    probe_python,
    probe_torch,
)
from analyzer.environment.models import probe_models

# Import name -> human label for the packages the pipeline depends on.
# torch is probed separately because it also feeds the compute summary.
_REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("cv2", "opencv"),
    ("mediapipe", "mediapipe"),
    ("pyarrow", "pyarrow"),
)


def _probe_package(module_name: str, label: str) -> ComponentStatus:
    """Import a package and report its real, self-declared version."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return ComponentStatus(
            name=label,
            status=HealthStatus.MISSING,
            detail=f"Could not import `{module_name}`: {exc}",
            remediation="Run `uv sync` in the python/ directory.",
        )
    except Exception as exc:  # noqa: BLE001 - a bad install must not abort the report
        return ComponentStatus(
            name=label,
            status=HealthStatus.ERROR,
            detail=f"Importing `{module_name}` raised {type(exc).__name__}: {exc}",
            remediation="Run `uv sync --reinstall` in the python/ directory.",
        )

    version = getattr(module, "__version__", None)
    if version is None:
        return ComponentStatus(
            name=label,
            status=HealthStatus.DEGRADED,
            detail=f"`{module_name}` imported but does not expose __version__.",
        )

    return ComponentStatus(
        name=label,
        status=HealthStatus.OK,
        detail=f"`{module_name}` imported successfully.",
        version=str(version),
    )


def _build_warnings(components: list[ComponentStatus], delegate: str, device: str) -> list[str]:
    """Surface caveats that are true even when every component reports OK."""
    warnings: list[str] = []

    if delegate == DEVICE_CPU:
        warnings.append(
            "Pose inference runs on CPU: the MediaPipe Tasks Python API provides no GPU "
            "delegate on this platform. The torch device below does not change this."
        )

    if device == DEVICE_CPU:
        warnings.append(
            "No GPU acceleration detected for PyTorch; all tensor work will run on CPU."
        )

    degraded = [c.name for c in components if c.status is HealthStatus.DEGRADED]
    if degraded:
        warnings.append(f"Degraded components: {', '.join(degraded)}.")

    return warnings


def run_doctor() -> EnvironmentReport:
    """Probe the environment and return a fully measured report."""
    components: list[ComponentStatus] = [probe_python()]
    components.extend(_probe_package(mod, label) for mod, label in _REQUIRED_PACKAGES)
    components.append(probe_torch())
    components.append(probe_ffmpeg())
    components.append(probe_ffprobe())
    components.extend(probe_models())

    hwaccels = list_hwaccels()
    compute = compute_info(hwaccels)

    return EnvironmentReport(
        generated_at=datetime.now(UTC),
        overall_status=aggregate_status(components),
        platform=platform_info(),
        components=components,
        compute=compute,
        warnings=_build_warnings(components, compute.mediapipe_delegate, compute.selected_device),
    )
