"""Typed contracts shared across the desktop <-> engine boundary.

These models are the single source of truth for the IPC schema. TypeScript
definitions are generated from them by scripts/gen_types.py; CI fails if the
checked-in output drifts.
"""

from analyzer.contracts.health import (
    HEALTH_SCHEMA_VERSION,
    ComponentStatus,
    ComputeInfo,
    EnvironmentReport,
    HealthStatus,
    PlatformInfo,
    aggregate_status,
)
from analyzer.contracts.rpc import (
    EngineError,
    ErrorCode,
    RpcError,
    RpcErrorResponse,
    RpcNotification,
    RpcRequest,
    RpcResponse,
)

__all__ = [
    "HEALTH_SCHEMA_VERSION",
    "ComponentStatus",
    "ComputeInfo",
    "EngineError",
    "EnvironmentReport",
    "ErrorCode",
    "HealthStatus",
    "PlatformInfo",
    "RpcError",
    "RpcErrorResponse",
    "RpcNotification",
    "RpcRequest",
    "RpcResponse",
    "aggregate_status",
]
