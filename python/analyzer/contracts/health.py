"""Typed contracts for the environment health check.

These Pydantic models are the single source of truth for the Rust <-> Python
boundary. `scripts/gen_types.py` derives JSON Schema from them and generates the
TypeScript definitions under `packages/types/src/generated/`; CI fails if the
generated output drifts from these models.

Every field here must be *measured*. Nothing in this module may report a
capability that has not been probed on the running machine.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# Bump when a breaking change is made to any model in this module. The desktop
# app checks this against the value it was generated from and refuses to render
# a report it does not understand, rather than silently mis-displaying fields.
HEALTH_SCHEMA_VERSION = 1


class HealthStatus(StrEnum):
    """Outcome of a single environment probe.

    OK        - present and usable
    DEGRADED  - present but limited (e.g. a slower fallback path is in use)
    MISSING   - not installed / not found; the user can fix this
    ERROR     - the probe itself failed unexpectedly
    """

    OK = "ok"
    DEGRADED = "degraded"
    MISSING = "missing"
    ERROR = "error"


class ComponentStatus(BaseModel):
    """Result of probing one required component of the analysis environment."""

    name: str = Field(description="Stable identifier, e.g. 'ffmpeg' or 'torch'.")
    status: HealthStatus
    detail: str = Field(description="Human-readable result of the probe.")
    version: str | None = Field(
        default=None,
        description="Version string as reported by the component itself, never inferred.",
    )
    remediation: str | None = Field(
        default=None,
        description="Concrete action the user can take when status is not OK.",
    )


class PlatformInfo(BaseModel):
    """Facts about the host, read from the interpreter and OS."""

    system: str
    release: str
    machine: str
    cpu_count: int | None
    python_version: str
    python_executable: str


class ComputeInfo(BaseModel):
    """Measured acceleration capabilities.

    `mediapipe_delegate` is reported separately from the torch device because the
    two do not share a backend: on macOS the MediaPipe Tasks Python API has no
    GPU delegate, so pose inference runs on CPU even when torch has MPS.
    """

    torch_version: str | None = None
    mps_available: bool = Field(description="torch.backends.mps.is_available()")
    mps_built: bool = Field(description="torch.backends.mps.is_built()")
    cuda_available: bool = Field(description="torch.cuda.is_available()")
    selected_device: str = Field(
        description="Preferred torch device for auto selection; runtime probing may fall back to CPU."
    )
    cpu_fallback_available: bool
    mediapipe_delegate: str = Field(
        description="Delegate MediaPipe will use on this platform ('cpu' or 'gpu')."
    )
    ffmpeg_hwaccels: list[str] = Field(
        default_factory=list,
        description="Hardware decoders reported by `ffmpeg -hwaccels`.",
    )


class EnvironmentReport(BaseModel):
    """Complete environment health report returned by the `doctor` method."""

    schema_version: int = HEALTH_SCHEMA_VERSION
    generated_at: datetime
    overall_status: HealthStatus = Field(
        description="Worst status across all components, in the order OK < DEGRADED < MISSING < ERROR."
    )
    platform: PlatformInfo
    components: list[ComponentStatus]
    compute: ComputeInfo
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal caveats the user should know about, e.g. CPU-only inference.",
    )


# Ordering used to reduce many component results to one overall status.
_SEVERITY: dict[HealthStatus, int] = {
    HealthStatus.OK: 0,
    HealthStatus.DEGRADED: 1,
    HealthStatus.MISSING: 2,
    HealthStatus.ERROR: 3,
}


def aggregate_status(components: list[ComponentStatus]) -> HealthStatus:
    """Reduce component results to the single worst status.

    An empty list means nothing was probed, which is itself an error rather than
    a pass -- reporting OK here would be the exact kind of false green this
    project is built to avoid.
    """
    if not components:
        return HealthStatus.ERROR
    return max((c.status for c in components), key=lambda s: _SEVERITY[s])
