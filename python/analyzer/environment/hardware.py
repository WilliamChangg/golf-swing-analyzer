"""Hardware and acceleration probing.

Everything reported here is measured at runtime. The project explicitly does not
hard-code a GPU vendor or assume CUDA: the reference development machine is
Apple silicon, where the acceleration paths are Metal (via torch MPS) and
VideoToolbox (via ffmpeg), and CUDA is simply absent.
"""

from __future__ import annotations

import logging
import os
import platform
import sys

from analyzer.contracts.health import ComponentStatus, ComputeInfo, HealthStatus, PlatformInfo

# Logs go to stderr; stdout is reserved for the RPC protocol stream.
_log = logging.getLogger(__name__)

# Torch device strings, in the order they are preferred when available.
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"
DEVICE_CPU = "cpu"

ENV_FORCE_CPU = "GSA_FORCE_CPU"


def platform_info() -> PlatformInfo:
    return PlatformInfo(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
    )


def mediapipe_delegate() -> str:
    """Delegate MediaPipe Tasks will actually use on this platform.

    The MediaPipe Tasks Python API exposes a GPU delegate on Linux and Android
    only; there is no GPU delegate for macOS. Reporting this honestly matters:
    on this machine torch can use MPS while pose inference is still CPU-bound,
    and a single "GPU: yes" indicator would misrepresent where time is spent.
    """
    if platform.system() == "Darwin":
        return DEVICE_CPU
    return DEVICE_CPU  # Conservative default until a GPU delegate is verified.


def compute_info(hwaccels: list[str] | None = None) -> ComputeInfo:
    """Probe acceleration capabilities.

    Import of torch is deliberately local: it is the slowest import in the
    project, and a doctor run should still produce a useful report when torch is
    broken or absent rather than failing at module import time.
    """
    force_cpu = os.environ.get(ENV_FORCE_CPU, "").strip().lower() in {"1", "true", "yes"}

    torch_version: str | None = None
    mps_available = False
    mps_built = False
    cuda_available = False

    try:
        import torch

        torch_version = torch.__version__
        mps_built = bool(torch.backends.mps.is_built())
        mps_available = bool(torch.backends.mps.is_available())
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken torch must not break the report
        # Swallowed on purpose: probe_torch() reports the failure as its own
        # component, so the user still sees it. Logged rather than passed
        # silently so the cause is recoverable from stderr.
        _log.debug("torch capability probe failed", exc_info=True)

    if force_cpu:
        selected = DEVICE_CPU
    elif cuda_available:
        selected = DEVICE_CUDA
    elif mps_available:
        selected = DEVICE_MPS
    else:
        selected = DEVICE_CPU

    return ComputeInfo(
        torch_version=torch_version,
        mps_available=mps_available,
        mps_built=mps_built,
        cuda_available=cuda_available,
        selected_device=selected,
        # CPU is always available as a fallback; this stays a real field because
        # it is part of the contract the UI renders and future backends may not
        # be able to make the same claim.
        cpu_fallback_available=True,
        mediapipe_delegate=mediapipe_delegate(),
        ffmpeg_hwaccels=hwaccels or [],
    )


def probe_torch() -> ComponentStatus:
    """Report torch as a component, separately from the compute summary."""
    try:
        import torch
    except ImportError as exc:
        return ComponentStatus(
            name="torch",
            status=HealthStatus.MISSING,
            detail=f"PyTorch could not be imported: {exc}",
            remediation="Run `uv sync` in the python/ directory.",
        )
    except Exception as exc:  # noqa: BLE001
        return ComponentStatus(
            name="torch",
            status=HealthStatus.ERROR,
            detail=f"PyTorch import raised {type(exc).__name__}: {exc}",
            remediation="Run `uv sync --reinstall` in the python/ directory.",
        )

    return ComponentStatus(
        name="torch",
        status=HealthStatus.OK,
        detail="PyTorch imported successfully.",
        version=torch.__version__,
    )


def probe_python() -> ComponentStatus:
    """Verify the interpreter is the pinned 3.12 series.

    MediaPipe 1.0.1 declares support for 3.9-3.12 only, so running outside that
    range is a real problem rather than a cosmetic one.
    """
    major, minor = sys.version_info[:2]
    version = platform.python_version()

    if (major, minor) != (3, 12):
        return ComponentStatus(
            name="python",
            status=HealthStatus.DEGRADED,
            detail=(
                f"Running Python {version}; this project pins 3.12 because "
                "MediaPipe 1.0.1 does not declare support beyond 3.12."
            ),
            version=version,
            remediation="Run `uv sync` in python/ to rebuild the environment on Python 3.12.",
        )

    return ComponentStatus(
        name="python",
        status=HealthStatus.OK,
        detail=f"Python {version} at {sys.executable}",
        version=version,
    )


def select_torch_device(requested: str = "auto") -> tuple[str, str | None]:
    """Resolve a training device at runtime, proving allocation before selecting it.

    CPU override wins even over an explicit accelerator. Later kernel failures
    still propagate: silently restarting a partially trained run is not fallback.
    """
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device '{requested}'; use auto, cpu, cuda or mps.")
    if os.environ.get(ENV_FORCE_CPU, "").strip().lower() in {"1", "true", "yes"}:
        return DEVICE_CPU, "CPU forced by GSA_FORCE_CPU."
    if requested == DEVICE_CPU:
        return DEVICE_CPU, None
    info = compute_info()
    selected = info.selected_device if requested == "auto" else requested
    if selected == DEVICE_CPU:
        return DEVICE_CPU, "No accelerator available; using CPU."
    available = info.cuda_available if selected == DEVICE_CUDA else info.mps_available
    if not available:
        return DEVICE_CPU, f"Requested {selected} is unavailable; using CPU."
    import torch

    try:
        # Copying back synchronises the operation; allocation alone is not proof.
        probe = torch.ones((2, 2), device=selected)
        (probe @ probe).cpu()
    except (RuntimeError, NotImplementedError) as exc:
        return DEVICE_CPU, f"{selected} runtime probe failed; using CPU: {exc}"
    return selected, None
