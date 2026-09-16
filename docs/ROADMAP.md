# Roadmap

Tracking checklist for the build. One phase at a time; at each boundary — run
tests, run the app, verify, document, record measurements, commit. Do not
advance past a broken phase.

**Progress: Phases 0-3 complete (4 / 21).**

| #   | Phase                 | Status      | Exit criterion                                             |
| --- | --------------------- | ----------- | ---------------------------------------------------------- |
| 0   | Foundation            | ✅ **Done** | App launches; doctor reports the real measured environment |
| 1   | Video ingestion       | ✅ **Done** | Correct metadata on VFR and rotated fixtures               |
| 2   | Single-camera pose    | ✅ **Done** | Landmarks persisted and reloadable; estimator swappable    |
| 3   | Temporal filtering    | ✅ **Done** | Error bounds met against analytical trajectories           |
| 4   | Swing phase detection | ⬜ Next     | Phases correct on real swings, inspectable frame-by-frame  |
| 5   | Biomechanics engine   | ⬜          | Metrics carry units, confidence, methodology               |
| 6   | DTL + coordinates     | ⬜          | Conventions documented and tested                          |
| 7   | Two-camera sync       | ⬜          | Measured sync residual in ms                               |
| 8   | Camera calibration    | ⬜          | Reprojection error reported; status gates claims           |
| 9   | 3D reconstruction     | ⬜          | Reconstruction error measured on synthetic ground truth    |
| 10  | Club tracking         | ⬜          | Shaft tracked; low confidence emits nothing                |
| 11  | Ball detection        | ⬜          | Impact-frame agreement measured                            |
| 12  | Temporal ML           | ⬜          | Leak-free splits; metrics only from a real labelled set    |
| 13  | Coaching engine       | ⬜          | Every finding cites computed evidence                      |
| 14  | Desktop UI            | ⬜          | Full workflow end-to-end                                   |
| 15  | 3D visualisation      | ⬜          | Scrub stays in sync with video                             |
| 16  | Swing comparison      | ⬜          | Differences shown, no "better/worse" score                 |
| 17  | Performance           | ⬜          | Before/after numbers recorded                              |
| 18  | Model management      | ⬜          | Backends reported; CPU fallback proven                     |
| 19  | Test hardening        | ⬜          | Numerical + pipeline + UI suites green                     |
| 20  | Documentation         | ⬜          | Docs match reality                                         |

---

## Phase 0 — Foundation ✅

- [x] **0.1 Toolchain** — rustup (rustc 1.98.1), uv 0.12.15, uv-managed CPython 3.12.14
- [x] **0.2 Repo + git** — `.gitignore`, `.gitattributes`, `.editorconfig`
- [x] **0.3 Python package** — pinned deps, `uv.lock`, ruff + mypy strict
- [x] **0.4 Contracts** — `ComponentStatus`, `ComputeInfo`, `EnvironmentReport`, RPC envelopes
- [x] **0.5 Doctor** — probes Python, 6 packages, ffmpeg/ffprobe, hwaccels, MPS/CUDA, model hashes
- [x] **0.6 Worker + CLI** — NDJSON-RPC loop, Typer CLI, shared dispatch table
- [x] **0.7 Tauri 2** — engine client with timeout, respawn, typed errors; minimal capabilities
- [x] **0.8 Frontend** — React 19.2.8, TS 6.0.3, Tailwind 4.3.3; health screen
- [x] **0.9 Codegen** — Pydantic → JSON Schema → TypeScript, with drift check
- [x] **0.10 Tests** — 48 pytest, 9 cargo, 15 Vitest, 4 Playwright
- [x] **0.11 CI** — 5 jobs: python, typescript, contracts, rust, e2e
- [x] **0.12 Docs** — README, architecture, ADR-0001..0006, this roadmap
- [x] **0.13 Capture protocol** — `data/README.md`
- [x] **0.14 Verify + commit** — app launched, negative test passed, baselines recorded

**Measured:** worker spawn → ready 163 ms · first `doctor` 1373 ms · warm
`doctor` 127 ms median · 12 components probed.

**Deliberate deviations from the original plan:**

- `tauri-plugin-shell` / `-dialog` / `-fs` were **not** added. The engine
  subprocess is spawned from Rust with `std::process`, which needs no plugin;
  the shell plugin exists to let the _frontend_ spawn processes, which this app
  should not do. Dialog and fs are added in Phase 1 when video import needs them.
- A `status-badge` component replaced generic `badge`/`alert` shadcn primitives,
  since every use is status presentation and an exhaustive `Record<Status, …>`
  makes a new status a type error until it is given a presentation.

---

## Phase 1 — Video ingestion ✅

`python/analyzer/ingestion/{probe,reader,cache}.py`

Two correctness traps handled explicitly rather than discovered later:
**rotation metadata** (phone clips carry a display matrix; decoders apply it
inconsistently, and getting it wrong silently biases every angle downstream) and
**variable frame rate** (phone slow-mo is frequently VFR, so time must come from
real presentation timestamps, never `frame / fps`).

- [x] **1.1 Contracts** — `VideoMetadata`, `VideoStreamInfo`, `VideoTiming`,
      `IntervalStats`, `ContentKey`; rotation in one named direction, `is_vfr`
      nullable so "unknown" is distinct from "no"
- [x] **1.2 ffprobe parse** — two passes (headers, then the packet index);
      timestamps handled in integer time-base ticks
- [x] **1.3 `FrameSource` Protocol** — OpenCV implementation, byte-bounded LRU
      frame cache, verified seeking
- [x] **1.4 VideoToolbox path + CPU** — `FFmpegPipeFrameSource`; CPU is the
      default, on measurement ([ADR-0007](decisions/ADR-0007-decode-backend.md))
- [x] **1.5 Content-hash cache keys** — sampled sha256, explicitly not an
      integrity check; on-disk `VideoMetadataCache`
- [x] **1.6 `probe_video`** — RPC method, CLI `analyzer probe`, metadata panel,
      dialog plugin + file picker
- [x] **1.7 Tests** — 94 added: CFR, VFR, rotated, truncated, corrupt,
      zero-byte, audio-only, missing, directory
- [x] **1.8 Measure + commit** — decode throughput per backend recorded below

**Measured** (`scripts/benchmark_decode.py`, 600 frames of 1920x1080 H.264,
median of 5, Apple M1 Pro / macOS 26.4.1):

| Operation                      | Median  | Frames/s |
| ------------------------------ | ------- | -------- |
| probe (2 ffprobe passes)       | 72.6 ms | —        |
| probe (metadata cache hit)     | 1.6 ms  | —        |
| decode: OpenCV (in-process)    | 564 ms  | **1065** |
| decode: ffmpeg subprocess, CPU | 1349 ms | 445      |
| decode: ffmpeg + VideoToolbox  | 2332 ms | 257      |

**Deliberate deviations from the original plan:**

- **The VideoToolbox/CPU relationship is inverted.** The plan assumed hardware
  decode would be the fast path with CPU as fallback. Measured, VideoToolbox is
  1.7x slower than software decode and 4.1x slower than decoding in process:
  this pipeline needs BGR frames in system memory, so a hardware-decoded frame
  must be read back off the GPU, and the transfer costs more than the decode it
  saves. CPU is the default; VideoToolbox is built, tested, and opt-in. Full
  reasoning in [ADR-0007](decisions/ADR-0007-decode-backend.md).
- **Two frame sources, not one.** OpenCV cannot reach VideoToolbox at all — its
  bundled FFmpeg reports `VIDEO_ACCELERATION_NONE` whatever is requested — so
  1.4 required an out-of-process ffmpeg reader. The second implementation earns
  its keep independently: it cross-checks the first, and the two are asserted to
  produce the same pixels and the same timestamps.
- **`is_vfr` has no tolerance threshold.** Reading integer time-base ticks rather
  than ffprobe's six-decimal `pts_time` removes the print-rounding that would
  otherwise need one, so the rule is simply "any interval more than one tick from
  the median". One tick is the container clock's own resolution; below that a
  difference is not measurable. This deliberately flags a single dropped frame in
  a long clip, because `frame / fps` is wrong after it either way.
- **`tauri-plugin-fs` still not added.** Video import needs a file _picker_, not
  filesystem access: the dialog returns a path, the WebView hands it to Rust, and
  the engine does the reading. Only `dialog:allow-open` was granted.
- **Video fixtures are committed, not generated at test time.** Five clips
  totalling ~185 KB, built by `scripts/make_video_fixtures.py`. Generating them
  per-run would make the suite depend on the local ffmpeg choosing to write a
  display matrix the same way, which is the very thing under test.
- **No capture-quality warning on low frame rate.** A "30 fps is too coarse for a
  downswing" caveat would be useful but is golf-specific reasoning, and
  `architecture.md` confines that to `biomechanics`/`phases`/`coaching`.
  Ingestion reports the measured rate; Phase 4 is where it gets judged.

## Phase 2 — Single-camera pose ✅

`PoseEstimator` Protocol with MediaPipe as one implementation; MediaPipe types
must not leak past the adapter. VIDEO running mode with real timestamps. Keep
normalized and world landmarks in separate fields — world landmarks are
hip-centred and only roughly metric, and are **not** calibrated world coordinates.

- [x] **2.1 Contracts** — `Landmark` (33, values are the model's own indices),
      `LandmarkPoint`, `PoseFrame`, `PoseSequence`, `LandmarkSeries`,
      `PoseExtractionResult`, `ProgressUpdate`
- [x] **2.2 MediaPipe adapter** — behind `PoseEstimator`; no MediaPipe type
      reaches a caller. BGR→RGB, VIDEO mode, strictly-increasing clock
- [x] **2.3 Parquet store** — long format, schema-versioned, self-describing;
      undetected frames stored as NaN rather than omitted
- [x] **2.4 Progress** — reporter seam → JSON-RPC notification → Tauri event →
      progress bar; also drives a Rich bar in the CLI
- [x] **2.5 Tests** — 114 added: round-trip, mapping completeness, no-detection
      frames, determinism, colour order, clock monotonicity, throttling
- [x] **2.6 Measure ms/frame; pick the default** — measured on two real swings;
      default kept at `full`, on the data, for the reason below

**Measured** (`scripts/benchmark_pose.py`, Apple M1 Pro / macOS 26.4.1), on two
real swings — a 68-frame 720x1280 face-on clip and a 239-frame 1920x1080
down-the-line clip:

| Model | Load    | ms/frame (face-on / DTL) | Frames/s | Poses found |
| ----- | ------- | ------------------------ | -------- | ----------- |
| lite  | ~190 ms | 11.6 / 11.0              | 86 / 91  | **100%**    |
| full  | ~85 ms  | 17.7 / 17.2              | 56 / 58  | **100%**    |
| heavy | ~120 ms | 67.1 / 66.3              | 15 / 15  | **100%**    |

**The default stays `pose_landmarker_full`, and that is a decision made on the
data rather than despite it.** All three variants find a pose in every frame, so
detection rate does not discriminate between them. Speed does, but not in a way
that binds: a swing clip is seconds long, so the spread is 0.8 s against 1.2 s
of work on the face-on clip. The criterion that would actually discriminate --
landmark accuracy -- is not measured anywhere yet, and picking the least
accurate variant to save half a second on a measurement system would be
optimising the wrong quantity. Revisit when Phase 12 provides an evaluation set.

Note how far the earlier synthetic figures were off: on a clip with nobody in
it, `heavy` measured 27.4 ms/frame, against 67.1 ms on a real swing. MediaPipe
runs the detector when it finds nothing and the landmark model when it does, and
those cost very different amounts. The benchmark's refusal to recommend below a
50% detection rate is what stopped that becoming a wrong default.

**Deliberate deviations from the original plan:**

- **2.6 is not finished, and the benchmark says so rather than guessing.** No
  footage containing a person exists in this repository, and MediaPipe takes two
  different paths through its graph: with no pose found it runs the _detector_
  every frame, and with one found it runs the _landmark_ model and re-detects
  only when tracking is lost. The table above therefore measures the path taken
  when nobody is in shot — a real path, and not the throughput of a real swing.
  `benchmark_pose.py` refuses to recommend a model below a 50% detection rate,
  so the default stays the manifest's `pose_landmarker_full`. One command
  finishes this once a swing clip exists:
  `uv run --project python python scripts/benchmark_pose.py --video <clip>`
- **MediaPipe re-pinned to 1.0.0** ([ADR-0008](decisions/ADR-0008-mediapipe-1.0.0.md)).
  1.0.1 aborts the process when the pose graph opens on macOS arm64. Phase 0's
  health check had reported MediaPipe as OK on the strength of an import, so the
  check now runs one real inference in a child process — an abort cannot be
  caught in-process, and a probe that takes the engine down with it reports
  nothing.
- **"World landmarks" are called `HIP_LOCAL`.** MediaPipe's name would collide
  with the genuinely-calibrated world coordinates Phase 9 produces, and the two
  are not interchangeable: these are hip-centred, only roughly metric, and carry
  no camera geometry. The word "world" appears nowhere in the pose contracts.
- **The request timeout became an inactivity timeout.** Extraction over a 60 s
  clip at 240 fps is 14,400 frames, well past any fixed budget. Every frame the
  worker sends — progress included — resets the clock, so a slow job is allowed
  to be slow while a wedged one still gives up.
- **`ProgressEvent` renamed `ProgressUpdate`**, because the generated TypeScript
  would otherwise collide with the DOM's built-in `ProgressEvent`.

## Phase 3 — Temporal filtering ✅

`python/analyzer/filtering/{signal,gating,gaps,localpoly,smoothing,pipeline,landmarks}.py`

The numerical core. **Savitzky–Golay assumes uniform sampling**, which VFR input
violates. Rather than resample to a uniform grid first, the local least-squares
problem is solved at each sample on the real timestamps, which makes
Savitzky–Golay the uniform-grid special case rather than an alternative —
asserted against SciPy to floating-point precision. Velocity and acceleration are
coefficients of that fit, never finite differences of smoothed data.
Full reasoning in [ADR-0009](decisions/ADR-0009-local-polynomial-filtering.md).

- [x] **3.1 Confidence gating** — visibility and presence thresholded
      separately; a rejected detection becomes the same NaN as no detection, so
      one representation of absence reaches the gap policy
- [x] **3.2 Gap policy** — linear-in-time bridging under `max_gap_s`; longer
      absences, and any leading or trailing one, marked `blocked` and left NaN
- [x] **3.3 Local polynomial fit** — window in seconds, solved per sample;
      refuses where support is insufficient or the point is unbracketed
- [x] **3.4 Analytic derivatives** — velocity `c1`, acceleration `2*c2`, from the
      same polynomial, so the three are mutually consistent
- [x] **3.5 `FilterStage` pipeline** — composable, each stage reporting what it
      did; the pipeline verifies that blocked samples are empty on the way out
- [x] **3.6 Tests** — 115 added (292 → 407): SciPy equivalence, polynomial
      exactness, RMS bounds against closed-form trajectories, Hypothesis properties
- [x] **3.7 Measure + commit** — defaults chosen from the sweep below

**Measured** (`scripts/benchmark_filter.py`, Apple M1 Pro / macOS 26.4.1).
Worst case over three analytical trajectories at 120 fps, sigma = 0.0014
normalized_frame measured from real footage:

| order | window     | pos RMS     | vel RMS    | acc RMS  | peak speed err |
| ----- | ---------- | ----------- | ---------- | -------- | -------------- |
| 2     | 0.10 s     | 0.00194     | 0.6226     | 13.69    | −8.4%          |
| 3     | 0.10 s     | 0.00195     | 0.0435     | 13.68    | −1.1%          |
| **4** | **0.10 s** | **0.00077** | **0.0437** | **3.29** | **−1.1%**      |
| 4     | 0.125 s    | 0.00072     | 0.0609     | 2.33     | −0.7%          |

Degree 2 is not viable — it underestimates peak speed by 8%, an error Phase 4
would inherit when locating impact. Degree 4 beats degree 3 on acceleration by
~4x at equal velocity error. The 0.10 s window is preferred over 0.125 s because
velocity is what phase detection keys on, and it is 28% better there.

Cost of assuming uniform sampling, same noise level (velocity RMS):

| Sampling        | True timestamps | Assumed uniform | Ratio     |
| --------------- | --------------- | --------------- | --------- |
| uniform         | 0.0308          | 0.0309          | **1.00x** |
| jitter, 50%     | 0.0402          | 0.0567          | 1.41x     |
| rate change, 4x | 0.0836          | 0.8926          | **10.7x** |
| dropped frames  | 0.0637          | 0.5740          | **9.0x**  |

Throughput: 33 landmarks x 3 axes in 12.3 ms (68 frames) and 17.5 ms (240
frames), against ~1.2 s for pose extraction on the same 68 frames.

**Deliberate deviations from the original plan:**

- **No resampling step, so 3.3 is a generalisation rather than an adaptation.**
  The plan offered "resample to a uniform grid, or local polynomial regression on
  true timestamps" as alternatives. Only the second was built, because resampling
  interpolates the data before anything is measured and commits to a grid and a
  kernel whose frequency response never appears in the output. The uniform case
  is recovered exactly, and a test pins it against SciPy.
- **"Confidence-weighted interpolation" ended up weighting filled samples at
  zero by default.** An interpolated value is a function of the neighbours that
  are already inside the fitting window, so giving it weight counts them twice.
  The confidence _is_ interpolated and carried, and `filled_weight` exposes the
  behaviour for anyone who wants a fit anchored rather than free — documented as
  what it is.
- **The defaults expose a frame-rate floor, and the system states it.** At 30 fps
  a 0.10 s window holds three samples, which cannot determine a degree-4 fit, so
  nothing is emitted — and the report names the minimum window that clip's
  measured rate would support. Both reference clips here are 24–30 fps, so this
  is the ordinary case. Silently widening the window was rejected: it produces
  numbers that are worse in a way nothing would report. This is also the first
  quantitative backing for the ≥120 fps the capture protocol asks for, which
  Phase 1 deliberately left as reasoning.
- **Bracketing does not stop interior gaps being crossed**, which is easy to
  assume it does. It prevents extrapolation past the observed range; how far
  interpolation may go is `GapPolicy`'s decision, delivered as a `blocked` mask
  and re-checked by the pipeline on the way out.
- **Nothing is persisted.** Filtering a clip is milliseconds against seconds of
  pose extraction, so a cache would add a schema to version and invalidate for no
  measurable gain. Decided on the measurement, and it changes if that does.
- **No TypeScript contracts generated.** `SequenceFilterReport` is reachable over
  RPC and from `analyzer filter`, but no UI consumes it yet, and exporting types
  nothing renders would make the app's type surface look like a plan rather than
  a description. Phase 4's inspector (4.5) is where it earns an export.
- **An unrelated environment fault was diagnosed**: the `analyzer` console script
  intermittently failing to import was macOS setting `UF_HIDDEN` on the venv's
  `.pth` files, which CPython's `site` module skips. `chflags nohidden` fixes it;
  reinstalling only appeared to, by rewriting the file.

## Phase 4 — Swing phase detection ⬜

Deterministic, no ML. Confidence from signal margin and landmark visibility — a
computed number, not a constant.

- [ ] 4.1 Signals: wrist speed, hand height, shoulder/hip line angles
- [ ] 4.2 Rule-based events + `SwingPhases` contract
- [ ] 4.3 Per-phase confidence
- [ ] 4.4 `scripts/plot_phases.py` debug plots + per-frame CSV
- [ ] 4.5 In-app frame-by-frame inspector
- [ ] 4.6 Tests: synthetic signals, truncated clip, no swing, double swing
- [ ] 4.7 Validate against recorded swings; commit

## Phase 5 — Biomechanics engine ⬜

Every result is a `Metric{name, value, unit, phase, confidence, source_frames,
methodology}`. From one 2D camera, shoulder/pelvis rotation are _projected_
angles under an assumed viewing geometry — labelled as such, with X-factor
flagged as an approximation.

- [ ] 5.1 Vector/angle primitives + tests
- [ ] 5.2 Posture: spine angle, knee flexion, hip position, head position
- [ ] 5.3 Rotation: shoulder, pelvis, separation, X-factor (2D-projected)
- [ ] 5.4 Hand/arm: hand path, lead/trail arm angles
- [ ] 5.5 Timing: backswing/downswing duration, tempo ratio, transition→impact
- [ ] 5.6 `Metric` contract + registry + confidence propagation
- [ ] 5.7 Tests with hand-computed expected angles
- [ ] 5.8 Commit

## Phase 6 — DTL + coordinate systems ⬜

- [ ] 6.1 Image / normalized / camera / world frame types + `docs/coordinate-systems.md`
- [ ] 6.2 DTL metrics: hand depth, spine angle, shaft orientation, club path, head movement
- [ ] 6.3 View-tagged metrics so face-on and DTL never conflate
- [ ] 6.4 Tests: round-trip conversions, known-projection fixtures
- [ ] 6.5 Commit

## Phase 7 — Two-camera synchronisation ⬜

- [ ] 7.1 Two-video project model (SQLite)
- [ ] 7.2 `SyncModel` / `TimeMap` contract
- [ ] 7.3 Manual sync UI (pick address/impact per camera)
- [ ] 7.4 Automatic sync: cross-correlation of hand-speed + event refinement
- [ ] 7.5 Report residual (ms) and confidence
- [ ] 7.6 Tests: known offsets, mismatched fps, partial overlap
- [ ] 7.7 Measure sync error vs manual ground truth; commit

## Phase 8 — Camera calibration ⬜

Charuco over plain chessboard (tolerates occlusion, unambiguous correspondences).
`CalibrationStatus` gates what the system may claim.

- [ ] 8.1 Calibration contracts (K, distortion, R, T, RMS, status)
- [ ] 8.2 Charuco detection + intrinsics
- [ ] 8.3 Stereo extrinsics
- [ ] 8.4 Capture/review UI with per-view reprojection error
- [ ] 8.5 Status gating enforced across the metric layer
- [ ] 8.6 Tests: synthetic board projections with known parameters
- [ ] 8.7 Document reconstruction limits; commit

## Phase 9 — Multi-view 3D reconstruction ⬜

- [ ] 9.1 DLT triangulation + non-linear refinement
- [ ] 9.2 Reprojection error + bone-length consistency checks
- [ ] 9.3 Per-joint per-frame confidence
- [ ] 9.4 Synthetic ground-truth validation harness
- [ ] 9.5 Hard gate: no metric claims without sufficient calibration
- [ ] 9.6 Measure reconstruction error on synthetic truth; commit

## Phase 10 — Club tracking ⬜

- [ ] 10.1 `ClubDetector` Protocol + `ClubFrame` contract
- [ ] 10.2 ROI → Canny → probabilistic Hough → geometric filtering
- [ ] 10.3 Temporal tracking + occlusion handling
- [ ] 10.4 Club-head position + trajectory
- [ ] 10.5 Overlay rendering
- [ ] 10.6 Tests: synthetic shafts, blurred, occluded
- [ ] 10.7 Measure detection rate and per-frame cost; commit

## Phase 11 — Ball detection ⬜

- [ ] 11.1 `BallDetector` Protocol + confidence
- [ ] 11.2 Impact corroboration fused with the Phase 4 estimate
- [ ] 11.3 Measure agreement between ball-based and kinematic impact frames
- [ ] 11.4 Commit

## Phase 12 — Temporal ML ⬜

Splits grouped by **session and player**, never by individual swing — swings from
one session are near-duplicates and would leak.

- [ ] 12.1 Labelling tool + label schema (**before** any training)
- [ ] 12.2 Dataset builder with feature versioning
- [ ] 12.3 Group-aware train/val/test splits
- [ ] 12.4 TCN baseline, seeded, checkpointed
- [ ] 12.5 Metrics, confusion matrix, model registry
- [ ] 12.6 Compare against the rule-based detector on the same held-out set
- [ ] 12.7 Commit (metrics only if a real labelled set exists)

## Phase 13 — Coaching engine ⬜

LLM layer receives **only** structured findings, never video, and is off by
default. Output is validated against the evidence; any number not present in the
evidence is rejected.

- [ ] 13.1 `Finding` contract + rule engine
- [ ] 13.2 Rule set with sourced, documented thresholds
- [ ] 13.3 Evidence linking (metric → frames → overlay)
- [ ] 13.4 Optional local LLM phrasing layer
- [ ] 13.5 Guard rejecting invented numbers
- [ ] 13.6 Tests: rule firing, insufficient evidence, LLM guard
- [ ] 13.7 Commit

## Phase 14 — Desktop UI ⬜

- [ ] 14.1 Project management (SQLite) + import flow
- [ ] 14.2 Dual video player with frame-accurate seek
- [ ] 14.3 Phase timeline
- [ ] 14.4 Transport: play/pause, 0.25/0.5/1x, frame step, jump to impact/top
- [ ] 14.5 Pose + club overlay canvas
- [ ] 14.6 Metrics panel surfacing methodology and confidence
- [ ] 14.7 Findings panel linked to evidence frames
- [ ] 14.8 Playwright flows for the full workflow
- [ ] 14.9 Commit

## Phase 15 — 3D visualisation ⬜

- [ ] 15.1 R3F scene + camera controls
- [ ] 15.2 Skeleton + trajectory from reconstruction output
- [ ] 15.3 Shared timeline store (video ↔ 3D)
- [ ] 15.4 Confidence + calibration status encoded in the viewport
- [ ] 15.5 Tests: store sync, degenerate frames
- [ ] 15.6 Commit

## Phase 16 — Swing comparison ⬜

- [ ] 16.1 Phase-relative temporal normalisation
- [ ] 16.2 Comparison contracts + diff computation
- [ ] 16.3 Comparison UI (overlaid trajectories, metric deltas)
- [ ] 16.4 Tests: normalisation against known warps
- [ ] 16.5 Commit

## Phase 17 — Performance engineering ⬜

- [ ] 17.1 Per-stage timing and memory instrumentation
- [ ] 17.2 `scripts/benchmark.py` with reproducible inputs
- [ ] 17.3 Baseline recorded **before** any optimisation
- [ ] 17.4 Optimise measured bottlenecks only
- [ ] 17.5 Incremental/cached re-analysis via content + config hashes
- [ ] 17.6 Before/after table; commit

## Phase 18 — Local model management ⬜

- [ ] 18.1 `Model{name, version, backend, device, input_requirements}` registry
- [ ] 18.2 Runtime device selection (no hard-coded GPU)
- [ ] 18.3 CPU fallback proven by a forced-fallback test
- [ ] 18.4 Download/verify/update flow in the UI
- [ ] 18.5 Commit

## Phase 19 — Testing hardening ⬜

- [ ] 19.1 Numerical: angles, vectors, derivatives, filtering, transforms, triangulation, alignment
- [ ] 19.2 Pipeline: metadata, pose, missing/low-confidence landmarks, phases, cache invalidation
- [ ] 19.3 UI: project creation, import, analysis, timeline, metric rendering
- [ ] 19.4 Synthetic fixture library
- [ ] 19.5 Coverage reporting in CI
- [ ] 19.6 Commit

## Phase 20 — Documentation ⬜

- [ ] 20.1 README: all 15 sections filled with real content
- [ ] 20.2 `docs/biomechanics.md`, `docs/computer-vision.md`, `docs/modeling.md`
- [ ] 20.3 Limitations stated explicitly
- [ ] 20.4 Metric appendix generated from benchmark output, every figure traceable
- [ ] 20.5 Commit

---

## Standing rules

**Measurements.** Numbers enter docs only from a benchmark script or a test run,
never from estimation. Every published figure names the script that produced it
and the machine it ran on.

**Honesty gates.**

| Condition               | Consequence                     |
| ----------------------- | ------------------------------- |
| Uncalibrated cameras    | No metric-scale 3D claims       |
| Single 2D view          | Rotations labelled as projected |
| No labelled dataset     | No accuracy claim               |
| Low detector confidence | Emit nothing, never a guess     |
| Probe not run           | Report unknown, never OK        |
