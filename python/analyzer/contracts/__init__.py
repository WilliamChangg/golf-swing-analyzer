"""Typed contracts shared across the desktop <-> engine boundary.

These models are the single source of truth for the IPC schema. TypeScript
definitions are generated from them by scripts/gen_types.py; CI fails if the
checked-in output drifts.
"""

from analyzer.contracts.cache import ContentKey, HashAlgorithm
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
from analyzer.contracts.video import (
    VIDEO_SCHEMA_VERSION,
    IntervalStats,
    RotationDegrees,
    TimestampSource,
    VideoMetadata,
    VideoStreamInfo,
    VideoTiming,
)

__all__ = [
    "HEALTH_SCHEMA_VERSION",
    "VIDEO_SCHEMA_VERSION",
    "ComponentStatus",
    "ComputeInfo",
    "ContentKey",
    "EngineError",
    "EnvironmentReport",
    "ErrorCode",
    "HashAlgorithm",
    "HealthStatus",
    "IntervalStats",
    "PlatformInfo",
    "RotationDegrees",
    "RpcError",
    "RpcErrorResponse",
    "RpcNotification",
    "RpcRequest",
    "RpcResponse",
    "TimestampSource",
    "VideoMetadata",
    "VideoStreamInfo",
    "VideoTiming",
    "aggregate_status",
]
