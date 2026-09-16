"""JSON-RPC 2.0 envelopes for the desktop <-> engine boundary.

The desktop app spawns the analysis engine as a child process and speaks
newline-delimited JSON-RPC 2.0 over stdin/stdout. This deliberately avoids a
local HTTP server: an application whose central promise is that video never
leaves the machine should not open a listening socket to do its own internal
messaging. See docs/decisions/ADR-0005-engine-boundary.md
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JSONRPC_VERSION: Literal["2.0"] = "2.0"


class ErrorCode:
    """JSON-RPC reserved codes plus this application's own range.

    The reserved codes (-32768..-32000) keep their standard meanings so a
    generic JSON-RPC client behaves sensibly. Application errors use -32000 and
    below-in-range values are avoided.
    """

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Application-specific (outside the reserved range).
    ANALYSIS_FAILED = -31000
    UNSUPPORTED_INPUT = -31001


class RpcError(BaseModel):
    code: int
    message: str
    data: dict[str, Any] | None = None


class RpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: int | str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class RpcResponse(BaseModel):
    """A success response. `result` is the serialized contract model."""

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: int | str
    result: dict[str, Any]


class RpcErrorResponse(BaseModel):
    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: int | str | None
    error: RpcError


class RpcNotification(BaseModel):
    """Engine -> desktop message with no reply, used for progress streaming.

    Notifications carry no `id`, which is how the client distinguishes them from
    responses to its own requests.
    """

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class EngineError(Exception):
    """Raised by engine methods to produce a structured RPC error.

    Carrying the code on the exception keeps the dispatch layer free of
    method-specific error mapping.
    """

    def __init__(
        self,
        message: str,
        code: int = ErrorCode.INTERNAL_ERROR,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.data = data

    def to_rpc_error(self) -> RpcError:
        return RpcError(code=self.code, message=str(self), data=self.data)
