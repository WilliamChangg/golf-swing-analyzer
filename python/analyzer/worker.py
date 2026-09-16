"""Newline-delimited JSON-RPC 2.0 worker over stdin/stdout.

Run as `python -m analyzer.worker`. The desktop app spawns this as a child
process and keeps it alive for the session, so models stay resident across
requests (loading them dominates per-request cost) and long analyses can stream
progress notifications back.

Protocol: exactly one JSON object per line, in both directions.

  ->  {"jsonrpc":"2.0","id":1,"method":"doctor","params":{}}
  <-  {"jsonrpc":"2.0","id":1,"result":{...}}
  <-  {"jsonrpc":"2.0","method":"progress","params":{...}}   (no id: notification)

stdout is reserved for protocol frames. Any library that writes to stdout would
corrupt the stream, so `sys.stdout` is rebound to stderr at startup and frames
are written to the original handle. MediaPipe and its TensorFlow Lite runtime do
emit chatter, which makes this a practical necessity rather than a precaution.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from analyzer import __version__
from analyzer.contracts.health import HEALTH_SCHEMA_VERSION
from analyzer.contracts.progress import PROGRESS_NOTIFICATION, PROGRESS_SCHEMA_VERSION
from analyzer.contracts.rpc import (
    EngineError,
    ErrorCode,
    RpcError,
    RpcErrorResponse,
    RpcNotification,
    RpcRequest,
    RpcResponse,
)
from analyzer.dispatch import METHODS, call
from analyzer.progress import CallbackReporter, ThrottledReporter


class Worker:
    """Reads requests from a stream and writes responses to another."""

    def __init__(self, stdin: TextIO, protocol_out: TextIO) -> None:
        self._stdin = stdin
        self._out = protocol_out

    def _write(self, payload: dict[str, Any]) -> None:
        """Emit one protocol frame and flush.

        Flushing every frame is required: the parent process blocks on a reply,
        so a buffered response would present as a hang.
        """
        self._out.write(json.dumps(payload, separators=(",", ":"), default=str))
        self._out.write("\n")
        self._out.flush()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._write(RpcNotification(method=method, params=params or {}).model_dump())

    def _error(self, request_id: int | str | None, error: RpcError) -> None:
        self._write(RpcErrorResponse(id=request_id, error=error).model_dump())

    def _handle_line(self, line: str) -> None:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            self._error(None, RpcError(code=ErrorCode.PARSE_ERROR, message=f"Invalid JSON: {exc}"))
            return

        if not isinstance(raw, dict):
            self._error(
                None,
                RpcError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"Expected a JSON object, got {type(raw).__name__}.",
                ),
            )
            return

        # Recover the id before validation so a malformed request can still be
        # correlated with the caller's pending promise instead of orphaning it.
        raw_id = raw.get("id")
        request_id: int | str | None = raw_id if isinstance(raw_id, int | str) else None

        try:
            request = RpcRequest.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            self._error(
                request_id,
                RpcError(code=ErrorCode.INVALID_REQUEST, message=f"Malformed request: {exc}"),
            )
            return

        # Progress is throttled here rather than in the methods: a per-frame
        # report at 240 fps would cost more in framing and pipe traffic than the
        # work it describes. The request id travels in the payload because a
        # JSON-RPC notification has no id of its own, and a client with two
        # calls in flight needs to know which one moved.
        reporter = ThrottledReporter(
            CallbackReporter(
                lambda event: self.notify(
                    PROGRESS_NOTIFICATION,
                    event.model_copy(update={"request_id": request.id}).model_dump(mode="json"),
                )
            )
        )

        try:
            result = call(request.method, request.params, reporter)
        except EngineError as exc:
            self._error(request.id, exc.to_rpc_error())
            return

        self._write(RpcResponse(id=request.id, result=result.model_dump(mode="json")).model_dump())

    def serve_forever(self) -> int:
        """Process requests until stdin closes. Returns a process exit code."""
        self.notify(
            "ready",
            {
                "engine_version": __version__,
                "health_schema_version": HEALTH_SCHEMA_VERSION,
                "progress_schema_version": PROGRESS_SCHEMA_VERSION,
                "methods": sorted(METHODS),
            },
        )

        for line in self._stdin:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                self._handle_line(stripped)
            except BrokenPipeError:
                # Parent went away mid-write; nothing left to report to.
                return 0
        return 0


def main() -> int:
    protocol_out = sys.stdout
    # Anything that prints to stdout from here on lands on stderr instead,
    # keeping the protocol stream clean.
    sys.stdout = sys.stderr

    worker = Worker(stdin=sys.stdin, protocol_out=protocol_out)
    try:
        return worker.serve_forever()
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
