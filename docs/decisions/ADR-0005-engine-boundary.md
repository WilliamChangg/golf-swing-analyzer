# ADR-0005: stdio JSON-RPC for the desktop ↔ engine boundary

**Status:** Accepted · 2026-09-15

## Context

The desktop shell is Rust + TypeScript (Tauri 2); the analysis engine is Python.
They need a boundary that supports request/response now and streamed progress
from long-running analyses later.

Three options were considered.

**Local HTTP server (FastAPI/uvicorn).** Familiar, easy to debug with curl, and
streaming is straightforward. But it opens a listening socket on a product whose
central promise is that video never leaves the machine. Even bound to localhost
that is a real surface: any local process can reach it, it needs an auth story
it would not otherwise need, and it can collide on ports. Paying that cost for
an application's _internal_ messaging is poor value.

**PyO3 / embedded interpreter.** No IPC at all. But it welds the Python engine
into the Rust build: the analysis code could no longer be run, tested, or
profiled without compiling Rust, and swapping a model implementation would mean
rebuilding the desktop app. That conflicts directly with keeping CV/ML inference
separate from the UI and with keeping models replaceable.

**Child process over stdio.** No socket, no port, no auth surface. The engine
dies with its parent. The Python side stays an ordinary package that can be run
from a terminal, tested with pytest, and profiled on its own.

## Decision

Spawn the engine as a **long-lived child process** speaking **newline-delimited
JSON-RPC 2.0** over stdin/stdout.

Design points that follow from this:

- **Long-lived, not one-shot.** Measured on the reference machine: spawn to
  `ready` is 163 ms, the first `doctor` call 1373 ms, subsequent calls 127 ms
  median. A process-per-call design would pay the ~1.4 s framework import on
  every request.
- **stdout is reserved for protocol frames.** `analyzer.worker` rebinds
  `sys.stdout` to stderr at startup, because MediaPipe and its TFLite runtime
  print to stdout and would otherwise corrupt the stream.
- **Reading happens on a dedicated thread feeding an mpsc channel.** A blocking
  read on a hung child cannot be interrupted; without the channel there would be
  no way to implement a timeout, and a wedged engine would freeze the UI.
- **Notifications carry no `id`.** That is how the client distinguishes
  engine-initiated progress messages from replies to its own requests. Phase 2
  forwards them to the UI as Tauri events.
- **Both entry points share one dispatch table** (`analyzer.dispatch`), so a
  method implemented once is reachable from the app and from `analyzer <cmd>`.

## Consequences

- No network surface, and no port configuration for the user.
- Debugging requires piping JSON by hand rather than curl; the Typer CLI exists
  partly to make that unnecessary.
- Packaging must resolve `uv` and the Python project at runtime. GUI apps on
  macOS do not inherit the login shell's `PATH`, so `engine::resolve` searches
  well-known install locations explicitly and supports `GSA_UV_BIN` /
  `GSA_PYTHON_DIR` overrides.
- Concurrency is currently serialised by a mutex around the worker. One method
  exists, so this is adequate; multiplexing concurrent requests by id is a
  change local to `engine/mod.rs` when a phase needs it.
