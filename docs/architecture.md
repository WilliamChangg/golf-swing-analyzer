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
│  │  features/video   ───┼───────►│  engine/mod.rs        │  │
│  │  lib/ipc.ts          │        │  engine/resolve.rs    │  │
│  │  components/ui       │        │                       │  │
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
                                    │  ingestion/           │
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
python/analyzer/contracts/{health,video}.py   (Pydantic, authoritative)
        │  scripts/gen_types.py
        ▼
packages/types/src/generated/schema/*.schema.json
        │  json-schema-to-typescript
        ▼
packages/types/src/generated/{EnvironmentReport,VideoMetadata}.ts
```

Each root model listed in `gen_types.py`'s `EXPORTS` becomes one schema and one
`.ts` file; nested models are inlined by the generator, so `VideoTiming` and
`IntervalStats` arrive with `VideoMetadata` rather than needing their own entry.

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
hashing/        content digests for verification and cache keys
paths/          filesystem layout resolution
environment/    hardware, tooling, model probing
ingestion/      container inspection and frame decoding
dispatch/       method registry
worker, cli     entry points
```

Later phases add `pose`, `filtering`, `phases`, `biomechanics`, `coaching` as
sibling packages with Protocol-typed seams (`PoseEstimator`, `ClubDetector`,
`BallDetector`), following `ingestion`'s `FrameSource`. Golf-specific reasoning
is confined to `biomechanics`, `phases`, and `coaching`; everything below is
general computer vision.

## Video ingestion

Two facts about consumer video decide whether anything downstream can be
trusted, and both fail silently rather than loudly.

**Rotation.** Phones store upright frames sideways and attach a display matrix.
Decoders disagree about applying it — OpenCV's `CAP_PROP_ORIENTATION_AUTO` reads
back as `1` on a fresh capture, so it rotates by default — and getting it wrong
produces a sideways skeleton rather than an error. Auto-rotation is therefore
switched off in every frame source and the rotation applied in exactly one
function, `apply_display_rotation`, in the counter-clockwise direction ffmpeg's
display matrix uses. That direction was settled by experiment, not documentation
(the two disagree), and is pinned by a test that decodes a rotated fixture both
ways and asserts the pixels are identical.

**Frame timing.** `frame_index / fps` is not the time a frame was taken on a
variable-rate clip, and phones produce those routinely. Time comes from the
container's presentation timestamps, read from the packet index. Those are
handled as integer time-base ticks rather than ffprobe's six-decimal `pts_time`,
because the printed form rounds by about as much as the jitter that distinguishes
constant from variable rate: a 30 fps clip in a 1/15360 time base prints
intervals of both 0.033333 and 0.033334, but in ticks it is exactly 512 every
time. The verdict therefore needs no tolerance beyond one tick — the finest
difference the container's own clock can express.

```
probe_video(path)
  ├─ check_readable            missing / directory / zero bytes
  ├─ content_key               sampled sha256 -> cache lookup
  ├─ ffprobe -show_streams     codec, geometry, display matrix
  ├─ ffprobe packet=pts,flags  presentation timestamps, sorted
  └─ VideoMetadata + FrameIndex
```

Packets leave a container in _decode_ order, not presentation order, so they are
sorted once here and every consumer downstream can treat index `i` as the i-th
frame a decoder will hand it.

`FrameSource` is the seam: `OpenCVFrameSource` (in-process, random access, the
default) and `FFmpegPipeFrameSource` (subprocess, optionally VideoToolbox). Both
yield display-oriented frames timed from the index, and are cross-checked
against each other in the test suite. Which one to use, and why hardware decode
is not the default, is [ADR-0007](decisions/ADR-0007-decode-backend.md).

## Honest capability reporting

The health screen reports torch's device and MediaPipe's delegate as **separate
rows**, because they do not share a backend. On this machine torch selects MPS
while MediaPipe runs on CPU — the MediaPipe Tasks Python API has no GPU delegate
on macOS. A single "GPU: available" indicator would claim GPU pose inference the
system cannot perform, so the UI states both and adds an explicit caveat.

This generalises: the system never reports a capability it has not measured.
Later phases extend the same rule to calibration status gating metric-scale 3D
claims, and to detector confidence gating club/ball output.

## Baseline measurements

Measured on Apple M1 Pro (10-core CPU, 16-core GPU, 16 GB), macOS 26.4.1, via
`scripts/` and the harness in this repository.

### Engine boundary (Phase 0, 2026-09-15)

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

### Ingestion (Phase 1, 2026-09-16)

`scripts/benchmark_decode.py`, 600 frames of 1920x1080 H.264, median of 5 runs.

| Operation                      | Median  | Frames/s |
| ------------------------------ | ------- | -------- |
| probe (2 ffprobe passes)       | 72.6 ms | —        |
| probe (metadata cache hit)     | 1.6 ms  | —        |
| decode: OpenCV, in-process     | 564 ms  | **1065** |
| decode: ffmpeg subprocess, CPU | 1349 ms | 445      |
| decode: ffmpeg + VideoToolbox  | 2332 ms | 257      |

Hardware decode is the slowest of the three because the pipeline needs BGR
frames in system memory: the GPU readback costs more than the decode it saves.
The 45x probe gap is what the content-keyed metadata cache exists for.

No figures exist yet for pose, filtering, or metrics, because none of those
exist yet.

## Testing strategy

| Layer                                 | Tool            | Covers                                                                                                                                       |
| ------------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Contracts, probes, dispatch, protocol | pytest (170)    | serialization, status aggregation, hash verification, error normalisation, RPC framing, rotation conventions, VFR detection, decode, caching |
| Transport framing, path resolution    | cargo test (9)  | notification vs reply, id correlation, malformed frames, `uv`/project discovery                                                              |
| IPC wrappers, component rendering     | Vitest (34)     | error normalisation, status rendering, remediation display, metadata panels, failure states                                                  |
| UI flows                              | Playwright (10) | layout, engine data rendering, import flow, screen switching, failure panel                                                                  |

The ingestion tests are split between pure parsing tests, which take ffprobe
output as literal strings and need no ffmpeg, and integration tests that run the
real thing against five committed fixture clips (constant rate, variable rate,
two rotations, audio-only). The fixtures are committed rather than generated so
the suite does not depend on the local ffmpeg writing a display matrix the same
way; `scripts/make_video_fixtures.py --verify` re-checks that they still carry
the properties the tests rely on, and CI runs it.

Playwright drives the Vite dev server with a stubbed Tauri bridge rather than the
packaged WebView; driving the real WebView needs `tauri-driver` plus a platform
WebDriver, which is not wired up yet. See Limitations in the README.
