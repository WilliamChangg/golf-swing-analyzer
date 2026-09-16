"""FFmpeg availability and capability probing.

FFmpeg is a hard requirement: ffprobe is the authoritative source of video
metadata from Phase 1 onwards, because container-level facts (rotation matrix,
variable frame rate, real presentation timestamps) are not reliably exposed by
higher-level decoding wrappers.
"""

from __future__ import annotations

import re
import shutil
import subprocess

from analyzer.contracts.health import ComponentStatus, HealthStatus

# Version banner looks like: "ffmpeg version 9.0.1 Copyright (c) ..."
_VERSION_RE = re.compile(r"^(?:ffmpeg|ffprobe) version (\S+)")

_PROBE_TIMEOUT_S = 10

_INSTALL_HINT = "Install with `brew install ffmpeg` (macOS) and ensure it is on PATH."


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        args,
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT_S,
        check=False,
    )


def _probe_binary(name: str) -> ComponentStatus:
    """Probe one ffmpeg-family binary for presence and version."""
    path = shutil.which(name)
    if path is None:
        return ComponentStatus(
            name=name,
            status=HealthStatus.MISSING,
            detail=f"{name} was not found on PATH.",
            remediation=_INSTALL_HINT,
        )

    try:
        proc = _run([path, "-version"])
    except (OSError, subprocess.SubprocessError) as exc:
        return ComponentStatus(
            name=name,
            status=HealthStatus.ERROR,
            detail=f"Found {name} at {path} but could not execute it: {exc}",
            remediation=_INSTALL_HINT,
        )

    if proc.returncode != 0:
        return ComponentStatus(
            name=name,
            status=HealthStatus.ERROR,
            detail=f"`{name} -version` exited with code {proc.returncode}.",
            remediation=_INSTALL_HINT,
        )

    first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
    match = _VERSION_RE.match(first_line)
    if match is None:
        # It ran, so it is usable; we just could not parse the banner.
        return ComponentStatus(
            name=name,
            status=HealthStatus.DEGRADED,
            detail=f"{name} is executable at {path} but its version banner was not recognised.",
            remediation=None,
        )

    return ComponentStatus(
        name=name,
        status=HealthStatus.OK,
        detail=f"Found at {path}.",
        version=match.group(1),
    )


def probe_ffmpeg() -> ComponentStatus:
    return _probe_binary("ffmpeg")


def probe_ffprobe() -> ComponentStatus:
    return _probe_binary("ffprobe")


def list_hwaccels() -> list[str]:
    """Return hardware decoders reported by `ffmpeg -hwaccels`.

    Returns an empty list when ffmpeg is absent or the call fails; callers treat
    that as "no hardware decode available" and fall back to software decoding.
    """
    path = shutil.which("ffmpeg")
    if path is None:
        return []

    try:
        proc = _run([path, "-hide_banner", "-hwaccels"])
    except (OSError, subprocess.SubprocessError):
        return []

    if proc.returncode != 0:
        return []

    lines = [line.strip() for line in proc.stdout.splitlines()]
    # First line is the header "Hardware acceleration methods:".
    return [line for line in lines if line and not line.endswith(":")]
