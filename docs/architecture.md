# Architecture

## Overview

Three processes, one machine, no network.

```
┌─────────────────────────────────────────────────────────────┐
│ Tauri application                                           │
│                                                             │
│  ┌──────────────────────┐        ┌───────────────────────┐  │
│  │ WebView (React/TS)   │        │ Rust core             │  │
│  │                      │        │                       │  │
│  │  features/health     │ invoke │  commands.rs          │  │
│  │  lib/ipc.ts  ────────┼───────►│  engine/mod.rs        │  │
│  │  components/ui       │        │  engine/resolve.rs    │  │
│  └──────────────────────┘        └───────────┬───────────┘  │
└───────────────────────────────────────────────┼─────────────┘
                                                │ stdin/stdout
                                                │ NDJSON-RPC 2.0
                                    ┌───────────▼───────────┐
                                    │ Python engine         │
                                    │                       │
                                    │  worker.py / cli.py   │
                                    │  dispatch.py          │
                                    │  contracts/           │
                                    │  environment/         │
                                    └───────────────────────┘
```

The Rust core is a **transport and process supervisor**. It contains no computer
vision, no biomechanics, and no analysis state. Everything measurable lives in
Python, which keeps it runnable and testable without compiling Rust.

## The engine boundary

Newline-delimited JSON-RPC 2.0 over stdin/stdout. Rationale, alternatives
considered, and the design constraints that follow are in
[ADR-0005](decisions/ADR-0005-engine-boundary.md). In short: no listening socket
on a local-first product, and no welding Python into the Rust build.

### Call path for `doctor`

```
HealthScreen (mount effect)
  └─ lib/ipc.ts  doctor()
      └─ @tauri-apps/api  invoke("doctor")
          └─ commands.rs  doctor(State<Engine>)
              └─ engine::Engine::request("doctor", {})
                  ├─ spawn worker if absent or dead
                  ├─ write  {"jsonrpc":"2.0","id":N,"method":"doctor",...}\n
                  └─ read lines until id == N  (skipping notifications)
                      └─ analyzer.worker  →  dispatch.call("doctor")
                          └─ environment.doctor.run_doctor()
                              ├─ probe_python / _probe_package × 5 / probe_torch
                              ├─ probe_ffmpeg / probe_ffprobe / list_hwaccels
                              └─ probe_models  (sha256 vs manifest)
```

Failures are typed at every hop. `EngineError.kind` is one of `spawn`,
`transport`, `protocol`, `method`, `timeout`; the UI renders a different
explanation and remediation for each, because "uv is not installed" and "the
worker crashed" call for different user actions.

## Typed contracts

Pydantic models in `python/analyzer/contracts/` are the **single source of
truth** for anything crossing the boundary.

```
python/analyzer/contracts/health.py   (Pydantic, authoritative)
        │  scripts/gen_types.py
        ▼
packages/types/src/generated/schema/EnvironmentReport.schema.json
        │  json-schema-to-typescript
        ▼
packages/types/src/generated/EnvironmentReport.ts
```

CI runs `npm run gen:types:check`, which regenerates and fails on any diff. A
Python contract change that is not reflected in the TypeScript therefore breaks
the build rather than drifting until it fails at runtime.

Two post-processing steps keep the output reviewable: per-field `title`s are
stripped (Pydantic emits one per field, which the generator turns into a wall of
single-use aliases), and structurally identical numbered duplicates
(`HealthStatus1`) are collapsed.

Types that exist only on the TypeScript side — `EngineError`, `EngineResult` —
are hand-written in `packages/types/src/ipc.ts`, because they describe the
transport rather than engine data.

## Layering

```
contracts/      typed models shared with the desktop app
paths/          filesystem layout resolution
environment/    hardware, tooling, model probing
dispatch/       method registry
worker, cli     entry points
```

Later phases add `ingestion`, `pose`, `filtering`, `phases`, `biomechanics`,
`coaching` as sibling packages with Protocol-typed seams (`FrameSource`,
`PoseEstimator`, `ClubDetector`, `BallDetector`). Golf-specific reasoning is
confined to `biomechanics`, `phases`, and `coaching`; everything below is
general computer vision.

## Honest capability reporting

The health screen reports torch's device and MediaPipe's delegate as **separate
rows**, because they do not share a backend. On this machine torch selects MPS
while MediaPipe runs on CPU — the MediaPipe Tasks Python API has no GPU delegate
on macOS. A single "GPU: available" indicator would claim GPU pose inference the
system cannot perform, so the UI states both and adds an explicit caveat.

This generalises: the system never reports a capability it has not measured.
Later phases extend the same rule to calibration status gating metric-scale 3D
claims, and to detector confidence gating club/ball output.

## Phase 0 baseline measurements

Measured 2026-09-15 on Apple M1 Pro (10-core CPU, 16-core GPU, 16 GB), macOS
26.4.1, via `scripts/` and the harness in this repository.

| Measurement                             | Value        |
| --------------------------------------- | ------------ |
| Worker spawn → `ready` notification     | 163 ms       |
| First `doctor` call (framework imports) | 1373 ms      |
| `doctor` call, warm (median of 5)       | 127 ms       |
| `doctor` call, warm (min / max)         | 121 / 136 ms |
| Components probed per call              | 12           |

The ~11x gap between first and warm calls is the entire justification for a
long-lived worker; a process-per-call design would pay the import cost every
time.

No other performance figures exist yet, because no analysis pipeline exists yet.

## Testing strategy

| Layer                                 | Tool           | Covers                                                                                      |
| ------------------------------------- | -------------- | ------------------------------------------------------------------------------------------- |
| Contracts, probes, dispatch, protocol | pytest (48)    | serialization, status aggregation, hash/size verification, error normalisation, RPC framing |
| Transport framing, path resolution    | cargo test (9) | notification vs reply, id correlation, malformed frames, `uv`/project discovery             |
| IPC wrappers, component rendering     | Vitest (15)    | error normalisation, status rendering, remediation display, failure states                  |
| UI flows                              | Playwright (4) | layout, engine data rendering, failure panel                                                |

Playwright drives the Vite dev server with a stubbed Tauri bridge rather than the
packaged WebView; driving the real WebView needs `tauri-driver` plus a platform
WebDriver, which is not wired up yet. See Limitations in the README.
