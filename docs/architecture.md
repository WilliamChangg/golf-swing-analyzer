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
filtering/      smoothing, gap policy, derivatives
phases/         swing event detection
dispatch/       method registry
worker, cli     entry points
```

Later phases add `biomechanics` and `coaching` as sibling packages with
Protocol-typed seams (`ClubDetector`, `BallDetector`), following `ingestion`'s
`FrameSource`, `pose`'s `PoseEstimator` and `filtering`'s `FilterStage`.
Golf-specific reasoning is confined to `phases`, `biomechanics` and `coaching`;
everything below is general computer vision that would serve any moving body.

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

## Temporal filtering

Pose estimation produces a position per frame, independently, with no notion
that the frames form a motion. This layer turns that into a trajectory that can
be differentiated, and decides which parts of it are supported well enough to
use.

**The fit runs on real timestamps.** Savitzky–Golay is a local least-squares
polynomial fit whose coefficients collapse into a fixed convolution kernel
_because the samples are uniformly spaced_ — an assumption Phase 1 established
that consumer footage frequently violates. Rather than resample onto a uniform
grid first, which interpolates the data before any of it is measured, the
weighted least-squares problem is solved at each evaluation point over a window
defined in seconds. Savitzky–Golay is then the special case this reduces to, and
the test suite asserts that equivalence against SciPy rather than claiming it.
Measured, the difference is nothing on uniform input and ~10x on a clip whose
rate changes. [ADR-0009](decisions/ADR-0009-local-polynomial-filtering.md).

**Derivatives come from the fit.** Velocity is the polynomial's linear
coefficient and acceleration twice its quadratic one, so position, velocity and
acceleration all describe the same curve. Finite-differencing smoothed positions
would apply a second, unstated filter with a much worse noise response and leave
the reported velocity inconsistent with the reported position. The `Signal` type
enforces the corollary: any stage that changes values drops the derivatives that
described them, because a stale derivative is exactly the kind of wrong number
that survives review.

```
LandmarkSeries
  └─ ConfidenceGateStage   low visibility/presence -> the same NaN as no detection
      └─ GapPolicyStage    short absences bridged; long ones marked `blocked`
          └─ LocalPolynomialStage   fit, and refuse where support is missing
              └─ FilterPipeline     verifies blocked samples came out empty
```

**Refusal is checked, not trusted.** Two different rules prevent two different
fabrications, and neither substitutes for the other. Bracketing inside the fit
stops extrapolation past the observed range. The gap policy decides how far
interpolation may go, and marks the rest `blocked` — and the pipeline re-checks
on the way out that nothing acquired a value there, so a stage added later
cannot quietly fill a refused gap.

Every stage returns a report of what it did, so the difference between a clip's
frame count and the count of frames carrying a usable position is always
attributable: gated detections, refused gaps, or windows with too little
support, each counted separately.

**Where it declines entirely.** The defaults need five samples per window, which
30 fps footage cannot supply over 0.10 s. Such a clip gets nothing, and the
report names the minimum window its measured rate would support. That is the
same rule as everywhere else in this system — report what was measured, refuse
what was not — applied to a case where the honest answer is unhelpful.

## Swing phase detection

The first layer that knows what a golf swing is. It reduces filtered
trajectories to a handful of scalar signals and reads four events off their
shape — deterministically, with no model, because the shape is not subtle and a
rule that can be read is a rule that can be argued with.

```
hand speed |          ___                    /\
           |      ___/   \__  <- top        /  \   <- impact
           |  ___/          \__          __/    \___
           |_/                  \_______/           \____
            address   backswing      downswing   follow-through
```

**Searches are nested rather than sequential.** Impact is found first, being the
clearest feature in the signal; the top is then found _before_ impact, the
takeaway _before_ the top, the finish _after_ impact. Each search is bounded by
an event already located, so a rule cannot place the top after impact — that
ordering is impossible by construction rather than checked afterwards.

**Two sign conventions are handled once, here.** Image y increases downward, so
a larger y is a _lower_ hand; it is converted into an explicitly named upward
`height` at the boundary and the raw y is never used again. And a shoulder line
has an orientation rather than a direction, so its angle is folded onto
(−90, 90] — taken as a vector angle, a nearly level line sits beside the ±180
discontinuity and flips the full 360 every time the tilt crosses zero, which for
shoulders and hips is most of a swing.

**Confidence is three measured factors, reported separately.** Margin (how
clearly the signal singles the instant out), visibility (how well the landmark
was seen _around that instant_, which a clip-wide mean cannot answer), and
resolution — whether the smoothing window the clip's frame rate forced is short
enough to resolve an event of that duration at all. Their product is the
headline, but the factors are what make a low score actionable: "0.3" is not,
and "the frame rate cannot resolve this" is.

Resolution is the factor that stops a 24 fps clip reporting a confident impact.
Such a clip forces a smoothing window about as long as a downswing, and no
amount of clarity in the resulting curve makes that measurable.

**Refusal is measured, not assumed.** A clip produces no events at all unless
the hands ranged far enough — judged in the subject's own torso lengths, so the
verdict does not depend on where the camera was put. The same test is applied to
the backswing and downswing separately, because a clip that is still except for
one twitch can otherwise be organised into a swing shape made almost entirely of
noise.

Impact is a kinematic estimate rather than an observation: nothing at this layer
sees the ball or the club, and hand speed peaks slightly before the club reaches
the ball. It is corroborated against an independent signal — the lowest point
the hands reach after the top — and Phases 10 and 11 replace that corroboration
with real evidence.

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

### Filtering (Phase 3, 2026-09-16)

`scripts/benchmark_filter.py`. Velocity RMS error against analytical
trajectories at 120 fps, at a noise level of 0.0014 normalized_frame measured
from real footage:

| Sampling        | True timestamps | Assumed uniform | Ratio     |
| --------------- | --------------- | --------------- | --------- |
| uniform         | 0.0308          | 0.0309          | **1.00x** |
| jitter, 50%     | 0.0402          | 0.0567          | 1.41x     |
| rate change, 4x | 0.0836          | 0.8926          | **10.7x** |
| dropped frames  | 0.0637          | 0.5740          | **9.0x**  |

Filtering all 33 landmarks in three axes takes 12.3 ms for a 68-frame clip and
17.5 ms for a 240-frame clip — against ~1.2 s to extract poses for the same 68
frames. Nothing is cached as a result.

No figures exist yet for metrics, because they do not exist yet.

## Testing strategy

| Layer                                 | Tool            | Covers                                                                                                                                                                         |
| ------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Contracts, probes, dispatch, protocol | pytest (449)    | serialization, status aggregation, hash verification, error normalisation, RPC framing, rotation conventions, VFR detection, decode, caching, pose, filtering, phase detection |
| Transport framing, path resolution    | cargo test (12) | notification vs reply, id correlation, malformed frames, `uv`/project discovery                                                                                                |
| IPC wrappers, component rendering     | Vitest (73)     | error normalisation, status rendering, remediation display, metadata panels, failure states, frame-by-frame inspection                                                         |
| UI flows                              | Playwright (21) | layout, engine data rendering, import flow, screen switching, failure panel, phase timeline scrubbing                                                                          |

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
