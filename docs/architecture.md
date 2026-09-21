# Architecture

The desktop presents evidence computed by a local Python engine. Rust supervises
that engine and mediates filesystem access; it contains no biomechanics or
computer-vision algorithms. The CLI uses the same dispatch layer without Tauri.

## Processes and transport

```text
React WebView
  lib/ipc.ts → Tauri invoke
                   ↓
Rust commands.rs → engine/{mod,resolve}.rs
                   ↓ stdin/stdout, newline-delimited JSON-RPC
Python worker.py → dispatch.py → typed analysis modules
                   ↑
                 cli.py
```

The engine boundary has no listening socket. Dependency/model installation uses
network downloads, and optional coaching phrasing can call a configured loopback
server; neither belongs to the default pixel-processing path.

Rust starts the worker on demand and serializes requests through the supervised
process. Progress notifications are forwarded as Tauri events. The timeout
measures inactivity, not total job duration, so a long extraction that keeps
reporting progress can complete. Typed spawn, transport, protocol, method and
timeout errors reach the UI. A dead worker is replaced on a subsequent request.
Stdout is reserved for protocol messages; diagnostics belong on stderr.

Runtime resolution looks for `uv` and an ancestor `python/pyproject.toml`, with
`GSA_UV_BIN` and `GSA_PYTHON_DIR` overrides. A built executable is therefore not
a self-contained distribution. Bundling the Python interpreter, dependencies and
model assets remains open deployment work.

## File access and the player

The WebView's asset scope starts empty. `commands::choose_clip` opens a native
dialog and admits the chosen path, allowing the video element to decode it.
There is no frontend command that simply grants an arbitrary supplied path.
Video is served through the asset protocol rather than copied into JSON-RPC.

`frame_index` supplies actual presentation times. The player seeks within the
intended frame's presentation interval and reconciles against the frame reported
painted by the browser. The pose overlay uses the analysis coordinate transform
and preserves missing support. The Three-D view shares the player's frame index.
[ADR-0018](decisions/ADR-0018-seeking-by-measured-time.md) describes the seek policy.

## Contracts

`python/analyzer/contracts/` owns Pydantic models for requests and results.
`scripts/gen_types.py` exports JSON Schema and generates corresponding TypeScript
under `packages/types/src/generated/`. Handwritten IPC types describe transport
outcomes rather than duplicating engine data. `npm run gen:types:check` fails if
the generated files have drifted.

The generator strips redundant field titles, supplies draft-07 tuple forms for
`prefixItems`, and wraps documented references to prevent the TypeScript generator
from duplicating shared types. Contract changes require regeneration and any
appropriate persisted-schema version change.

## Analysis layers

| Module                                      | Responsibility                                                             |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| `contracts/`, `coordinates.py`              | Shared data, units, frames and explicit conversions                        |
| `ingestion/`                                | Probe, frame timestamps, rotation, decoding and metadata cache             |
| `pose/`                                     | Estimator interface, MediaPipe adapter, extraction and Parquet storage     |
| `filtering/`                                | Confidence gates, gaps and polynomial fits on real timestamps              |
| `phases/`                                   | Hand signals and heuristic swing events                                    |
| `sync/`                                     | Cross-camera clock mapping and uncertainty                                 |
| `calibration/`                              | Board detection, intrinsics, stereo pose and coverage gates                |
| `reconstruction/`                           | Matched-time triangulation and geometric quality checks                    |
| `biomechanics/`                             | View-aware projected and calibrated spatial measurements                   |
| `club/`, `ball/`, `impact.py`               | Classical object evidence and impact reconciliation                        |
| `coaching/`                                 | Basis-gated rules, cited findings and optional guarded phrasing            |
| `comparison/`                               | Phase-relative alignment and camera-gated differences                      |
| `ml/`                                       | Labels, feature extraction, grouped splits, temporal models and evaluation |
| `overlay.py`, `scene.py`                    | Bounded payloads for the player and 3D viewport                            |
| `projects/`, `performance/`, `environment/` | Persistent sessions, instrumentation, caches and runtime/model management  |

Coordinates are corrected below the filter so derivatives inherit the same
scale and sign. Calibrated `CAMERA` positions come from reconstruction, not from
converting a single pose sequence. `WORLD` is defined as a concept but not
produced. See [Coordinate systems](coordinate-systems.md).

Filter window resolution happens before fitting. Unsupported requested windows
widen by default with warnings and applied configuration in the result; stereo
synchronization resolves a common window for both inputs. The original strict
low-rate refusal is superseded by
[ADR-0021](decisions/ADR-0021-resolving-the-smoothing-window.md).

Biomechanics receives reconstruction as an optional input because a single-clip
measurement should not itself orchestrate a two-camera workflow. Coaching reads
metrics and phases, never pixels. Comparison shares coaching's bracket logic
but adds bounds across independent recordings. The guides explain the methods:
[Computer vision](computer-vision.md), [Biomechanics](biomechanics.md),
[Modeling](modeling.md).

## Desktop workflow and errors

The Swing screen serially requests extraction, phases, metrics, coaching and
pose overlay. Starting analysis clears earlier results. A stage failure stops
later requests and leaves its error visible for retry. Sessions persist clip
membership and camera roles. Three-D requires a usable aligned stereo project;
Compare treats its inputs as separate swings.

The default phase path is heuristic. Club tracking, ball detection, reconciled
impact and temporal training are CLI workflows; their existence does not imply
that they are invoked by the Swing screen's Analyse button. Environment provides
runtime probes and manifest-pinned pose-model management.

## Persistence and cache identity

`paths.py` centralizes locations and environment overrides.

| State                               | macOS default                                       | Lifetime                                        |
| ----------------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| Pose models and manifest            | Repository `models/`                                | Pinned artifacts; `GSA_MODELS_DIR` overrides    |
| Metadata, Parquet, report caches    | `~/Library/Caches/golf-swing-analyzer`              | Recomputable; `GSA_CACHE_DIR` overrides         |
| Projects, labels, temporal registry | `~/Library/Application Support/golf-swing-analyzer` | User state to back up; `GSA_DATA_DIR` overrides |

Other platforms use XDG cache/data roots with conventional home-directory
fallbacks. `GSA_REPO_ROOT` overrides engine repository discovery. SQLite projects
store clip membership, roles, synchronization and calibration; original videos
remain at their own paths. A sampled content key identifies media for reuse,
not cryptographic integrity of every byte. Relocation updates clip paths.

Pose reuse checks the content key, model name and installed artifact digest.
Metrics and coaching JSON cache keys include complete request configuration and
pose extraction identity, including model metadata and extraction time. Invalid
cache contents are misses. Project-backed analysis bypasses these result caches
because calibration or clock state can change without a change to the video.
`GSA_DISABLE_ANALYSIS_CACHE=1` disables small result caches, not pose reuse.

## Model installation and runtime reporting

The typed pose inventory reports version digest, backend, selected device and
input requirements. Downloads stage to a unique neighbouring temporary file,
verify size/hash and replace atomically. A failed transfer preserves the installed
model. Update means this application's manifest pin; deliberate re-pinning is a
developer operation, separate from the UI.

The environment doctor probes dependencies and runs real pose inference in a
child process so a native abort becomes a failed component rather than destroying
the worker. Inventory verification only checks bytes. Torch device selection tests
an accelerator before choosing it and records CPU fallback; MediaPipe explicitly
uses its CPU delegate. See [Modeling](modeling.md).

## Measurements and verification

`PerformanceRecorder` instruments RPC operations and nested stages with elapsed
time and OS process-memory samples. On macOS RSS is peak process RSS, not memory
allocated by an individual stage. `scripts/benchmark.py` follows the desktop's
serial workflow with fresh and primed-cache modes.

The [generated appendix](benchmarks.md) links retained benchmark data and source
fingerprints. Historical measurements and architectural rationale remain in the
[roadmap](ROADMAP.md) and [decision records](decisions/ADR-0005-engine-boundary.md).
Current figures are not copied by hand into this document.

[Testing](testing.md) covers numerical, pipeline, transport, component and browser
checks. Browser tests stub the native bridge; no automated packaged-WebView suite
exists. Real pose execution and synthetic accuracy tests answer different
questions, and neither supplies a labelled golfer evaluation set.
