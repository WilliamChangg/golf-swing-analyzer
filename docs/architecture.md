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
progress/       reporting from long-running methods
environment/    hardware, tooling, model probing
ingestion/      container inspection and frame decoding
pose/           landmark estimation, storage, per-landmark series
dispatch/       method registry
worker, cli     entry points
```

Later phases add `filtering`, `phases`, `biomechanics`, `coaching` as sibling
packages with Protocol-typed seams (`ClubDetector`, `BallDetector`), following
`ingestion`'s `FrameSource` and `pose`'s `PoseEstimator`. Golf-specific
reasoning is confined to `biomechanics`, `phases`, and `coaching`; everything
below is general computer vision.

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

## Pose estimation

`PoseEstimator` is the seam: MediaPipe is one implementation and none of its
types reach a caller. That is what makes the model replaceable, and it is also
what makes the pipeline testable — most of the pose tests run the whole path
with a fake estimator, so decode, estimate, persist and reload are exercised
without loading a model.

**Two coordinate spaces, kept apart by name.** `IMAGE` is normalised to the
displayed frame and is the only space in which a landmark can be drawn on video.
`HIP_LOCAL` is what MediaPipe calls "world landmarks": approximate metres,
centred on the hips, oriented to the body. They carry no information about where
the camera was or how far away the subject stood. Calling them "world" would
collide with the genuinely calibrated world coordinates Phase 9 produces by
triangulation, and the two are not interchangeable, so the word does not appear
in the pose contracts at all.

**Landmarks are stored long, not wide.** One row per (frame, space, landmark),
so adding a landmark or a space adds rows rather than columns and a file written
today still reads after Phase 9. Frames with no detection are written as NaN
rather than omitted: the file stays rectangular, and a gap arrives at the
filtering layer already in the representation Phase 3 needs in order to refuse
to interpolate across it. Schema version, source video, model digest, thresholds
and run statistics live in the Parquet metadata, so a result always traces back
to the artifact that produced it.

**The clock MediaPipe sees is not the clock that is recorded.**
`detect_for_video` takes integer milliseconds and requires them to increase
strictly; at 480 fps frames are 2.08 ms apart, so rounding alone eventually
repeats a value and the call is rejected. That clock is forced forward when it
must be, and the count is reported. Every timestamp that reaches a stored
sequence or a metric still comes from the container index — the nudged value is
only used by MediaPipe's tracker to order frames.

## Progress

Long methods report through a `ProgressReporter` and have no idea what is on the
other end. Three things are:

```
extract_poses  ──▶ ProgressTracker ──▶ ThrottledReporter ──▶ RPC notification
                                                          ──▶ Rich progress bar (CLI)
                                                          ──▶ list (tests)
```

Throttling lives in the worker rather than the call sites, because extraction
reports once per frame and at 240 fps that would cost more in framing and pipe
traffic than the work being described. The first update and the last always get
through: a bar stuck below 100% on finished work reads as a hang.

A JSON-RPC notification has no `id`, so the request id travels in the payload.
Rust re-emits progress as a Tauri `engine://progress` event, and — the reason
this matters beyond cosmetics — **every frame from the worker resets the
engine's timeout**. That turns a fixed request budget into an inactivity
timeout, which is what lets a 14,400-frame extraction outlast it while a wedged
worker still gives up.

## Honest capability reporting

The health screen reports torch's device and MediaPipe's delegate as **separate
rows**, because they do not share a backend. On this machine torch selects MPS
while MediaPipe runs on CPU — the MediaPipe Tasks Python API has no GPU delegate
on macOS. A single "GPU: available" indicator would claim GPU pose inference the
system cannot perform, so the UI states both and adds an explicit caveat.

Phase 2 found the cost of getting this wrong in the codebase's own foundation.
The Phase 0 check reported `mediapipe` as OK because `import mediapipe`
succeeded — and pose inference aborted the process on first use, so the report
was a green light for a pipeline that could not run a frame. The check now runs
one real inference, in a child process, because an abort cannot be caught
in-process and an in-process probe would have taken the engine down with it.
See [ADR-0008](decisions/ADR-0008-mediapipe-1.0.0.md).

The general rule that follows: **a probe must exercise the capability, not a
proxy for it.** Import is a proxy. The same reasoning is why ingestion decodes a
frame to settle rotation rather than trusting a property, and why the decode
benchmark measures both backends rather than assuming the hardware one wins.

This generalises further: the system never reports a capability it has not
measured. Later phases extend the same rule to calibration status gating
metric-scale 3D claims, and to detector confidence gating club/ball output.

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

Probing costs far more than it did in Phase 1 — 1326 ms against the 72.6 ms a
packet-only pass took — because per-frame timestamps are now read from the
frames a decoder emits. That was not a performance regression accepted for
tidiness: packet timestamps are simply wrong on containers with an edit list,
and wrong silently. The ~900x gap to a cache hit is what the content-keyed
metadata cache exists for.

### Pose (Phase 2, 2026-09-16)

`scripts/benchmark_pose.py`, on two real swings: a 68-frame 720x1280 face-on
clip and a 239-frame 1920x1080 down-the-line clip.

| Model | Load    | ms/frame (face-on / DTL) | Frames/s | Poses found |
| ----- | ------- | ------------------------ | -------- | ----------- |
| lite  | ~190 ms | 11.6 / 11.0              | 86 / 91  | **100%**    |
| full  | ~85 ms  | 17.7 / 17.2              | 56 / 58  | **100%**    |
| heavy | ~120 ms | 67.1 / 66.3              | 15 / 15  | **100%**    |

Per-frame cost barely moves with resolution, because MediaPipe resizes to a
fixed input; it moves a great deal with which path the graph takes. On a clip
with nobody in it `heavy` measured 27.4 ms/frame, against 67.1 ms here — the
detector is cheap and the landmark model is not. That is why the benchmark
refuses to recommend a model below a 50% detection rate.

The health check also gained about a second, spent running one real pose
inference in a child process rather than trusting an import.

No figures exist yet for filtering or metrics, because neither exists yet.

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

The pose tests follow the same shape. Most run against a fake estimator, which
is the evidence that `PoseEstimator` is a real seam rather than a shortcut; the
MediaPipe adapter's own behaviour — colour order, no-detection handling,
determinism — is tested separately against a real model. CI downloads the models
and sets `GSA_REQUIRE_MODELS`, so a runner that failed to fetch them fails
rather than skipping the suite and reporting green.

Playwright drives the Vite dev server with a stubbed Tauri bridge rather than the
packaged WebView; driving the real WebView needs `tauri-driver` plus a platform
WebDriver, which is not wired up yet. See Limitations in the README.
