# Roadmap

Tracking checklist for the build. One phase at a time; at each boundary — run
tests, run the app, verify, document, record measurements, commit. Do not
advance past a broken phase.

**Progress: Phases 0-10 complete (11 / 21).**

| #   | Phase                 | Status      | Exit criterion                                                 |
| --- | --------------------- | ----------- | -------------------------------------------------------------- |
| 0   | Foundation            | ✅ **Done** | App launches; doctor reports the real measured environment     |
| 1   | Video ingestion       | ✅ **Done** | Correct metadata on VFR and rotated fixtures                   |
| 2   | Single-camera pose    | ✅ **Done** | Landmarks persisted and reloadable; estimator swappable        |
| 3   | Temporal filtering    | ✅ **Done** | Error bounds met against analytical trajectories               |
| 4   | Swing phase detection | ✅ **Done** | Phases correct on real swings, inspectable frame-by-frame      |
| 5   | Biomechanics engine   | ✅ **Done** | Metrics carry units, confidence, methodology                   |
| 6   | DTL + coordinates     | ✅ **Done** | Conventions documented and tested                              |
| 7   | Two-camera sync       | ✅ **Done** | Measured sync residual in ms                                   |
| 8   | Camera calibration    | ✅ **Done** | Coverage gates claims; reprojection error shown for what it is |
| 9   | 3D reconstruction     | ✅ **Done** | Reconstruction error measured on synthetic ground truth        |
| 10  | Club tracking         | ✅ **Done** | Shaft tracked; low confidence emits nothing                    |
| 11  | Ball detection        | ⬜ Next     | Impact-frame agreement measured                                |
| 12  | Temporal ML           | ⬜          | Leak-free splits; metrics only from a real labelled set        |
| 13  | Coaching engine       | ⬜          | Every finding cites computed evidence                          |
| 14  | Desktop UI            | ⬜          | Full workflow end-to-end                                       |
| 15  | 3D visualisation      | ⬜          | Scrub stays in sync with video                                 |
| 16  | Swing comparison      | ⬜          | Differences shown, no "better/worse" score                     |
| 17  | Performance           | ⬜          | Before/after numbers recorded                                  |
| 18  | Model management      | ⬜          | Backends reported; CPU fallback proven                         |
| 19  | Test hardening        | ⬜          | Numerical + pipeline + UI suites green                         |
| 20  | Documentation         | ⬜          | Docs match reality                                             |

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

`data/dtl/iron_dtl.mp4` (96 frames, 30 fps, added after the first pass) detects
just as cleanly — takeaway frame 7, top 27, impact 38, finish 49, confidences
0.94 / 0.79 / 0.81 / 0.00 — but only after a fix it forced. See the hand-source
deviation below.

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

## Phase 4 — Swing phase detection ✅

`python/analyzer/phases/{signals,detect}.py`

The first golf-specific package; everything below it is general computer vision.
Deterministic, no ML — the events are defined by the shape of the hand-speed
signal (one minimum between two maxima, with the minimum at the highest point
the hands reach), and a rule that can be read is a rule that can be argued with.
Phase 12 trains a detector and compares it against this one, which is only
meaningful because a readable baseline exists first.

- [x] **4.1 Signals** — hand speed and height, shoulder and hip line angles.
      Image y points down, so height is converted once and the raw y never used
      again; line angles are folded onto (−90, 90] because a shoulder line has
      an orientation, not a direction
- [x] **4.2 Rule-based events** — `SwingPhases` contract; four events bounding
      four phases. Searches are nested (impact, then the top before it, then the
      takeaway before that), so a misordering is impossible by construction
- [x] **4.3 Per-event confidence** — margin × visibility × resolution, each
      reported separately because "0.3" is not actionable and "the frame rate
      cannot resolve this" is
- [x] **4.4 `scripts/plot_phases.py`** — three-panel plot with phase shading and
      event markers, plus a per-frame CSV
- [x] **4.5 In-app inspector** — scrubbable phase timeline, frame stepping,
      event jump targets, and the current frame's phase always on screen
- [x] **4.6 Tests** — 60 added (42 pytest, 12 Vitest, 6 Playwright): synthetic
      swings with known event times, truncated clip, no swing, double swing
- [x] **4.7 Validated on recorded swings; commit**

**Measured** on `data/face-on/PW_face-on.mp4` (68 frames, 30 fps, 0.15 s
window). Every event matches a hand reading of the signal:

| Event    | Frame | Time    | Confidence | margin | visibility | resolution |
| -------- | ----- | ------- | ---------- | ------ | ---------- | ---------- |
| takeaway | 13    | 0.433 s | 0.97       | 1.00   | 0.97       | 1.00       |
| top      | 38    | 1.267 s | 0.64       | 0.84   | 0.92       | 0.83       |
| impact   | 48    | 1.600 s | 0.67       | 1.00   | 0.81       | 0.83       |
| finish   | 65    | 2.167 s | 0.00       | 0.00   | 0.63       | 1.00       |

Impact is corroborated by an independent signal — the lowest point the hands
reach after the top — which lands on frame 47, one frame (33 ms) away. The
finish scores zero because the clip ends before the hands stop, which is a fact
about the recording rather than about the detector.

**The down-the-line clip is refused, and the refusal is the useful part.** Its
descent takes 2.13 s, over the 1 s a downswing can last — a club falls faster
than that unaided, so that motion is someone lowering it. The detector also
reports that the hands were untracked for 1.92 s from 8.05 s, which at 24 fps
with a slow shutter is almost certainly where the real swing is: motion blur
smears the wrists exactly when they move fastest, and pose estimation loses them.

**Deliberate deviations from the original plan:**

- **The hand source is chosen by longest unbroken tracking, not by frame
  count.** Down-the-line footage hides one wrist behind the other, and which one
  it hides changes through the swing: on `iron_dtl.mp4` the right wrist is
  visible through the address and backswing while the left is not, and the left
  takes over once the right is lost after impact. Counting frames picks the left
  (59 against 48), which contains no address and no backswing, and the clip is
  refused. Counting the longest unbroken run picks the right (48 against 44) and
  every event is found. A swing is one continuous event, so the run that must
  contain it is the thing worth maximising.
- **Four events, not five.** `ADDRESS` is a phase with no event of its own,
  because its last frame _is_ the frame before the takeaway. An address event
  would sit one frame from the takeaway and imply a precision the signal does
  not carry.
- **The "is this a swing" gate is hand travel in torso lengths, not a speed
  ratio.** Peak hand speed against the speed seen while still was tried first
  and reports a swing on a clip of someone standing motionless — on pure noise
  that ratio compares the largest spike with the median and comfortably exceeds
  ten. A noise-based gate was tried second and is worse: the filter's residual
  is _exactly zero_ whenever the smoothing window holds as many samples as the
  polynomial has coefficients, which is every clip below about 60 fps at the
  shipped defaults. Judging travel against the subject's own torso is
  independent of framing and always available.
- **The takeaway search is bounded by the backswing's speed peak, not by the
  top.** The hands come almost to rest at the top, so a search for the last
  still stretch before the top finds that pause and reports a backswing of a few
  frames. The face-on clip only avoided this by luck — its speed at the top was
  0.130 against a 0.131 threshold. Found by a synthetic fixture, not by the real
  footage.
- **An upper bound on the downswing was added** (`max_downswing_s`). The lower
  bounds alone accept a two-second descent as a swing, which is what the DTL
  clip contains.
- **Tempo is not computed here.** The backswing-to-downswing ratio is a
  biomechanics metric and belongs to Phase 5, where it acquires a unit, a
  confidence and a methodology. Computing it here would put one number in two
  places with two provenances.
- **`FilteredLandmark` gained `visibility` and `observed`** (a Phase 3 type).
  Confidence needs per-frame visibility around an event, which a clip-wide mean
  cannot give, and the tracking diagnostic needs to distinguish "the estimator
  saw nothing" from "the filter had too little support at the clip edge" — one
  is a camera problem and the other a window-width one.
- **`matplotlib` is now a declared dev dependency.** It arrives transitively
  through mediapipe today, and a dependency that works by accident stops working
  without warning.
- **The debug plot found a defect on its first run**, which is the argument for
  4.4 existing: the projected shoulder and hip angles were wrapping between
  ±180° every time the line crossed level.

## Phase 5 — Biomechanics engine ✅

`python/analyzer/biomechanics/{geometry,body,anchors,registry,posture,rotation,arms,timing,compute}.py`

Every result is a `Metric{name, value, unit, basis, event/phase, confidence,
source_frames, methodology}`. The field that does the most work is `basis`,
which says what **kind** of claim the number is — a duration, an image-plane
distance, a projected angle, or a rotation inferred from foreshortening. From
one uncalibrated camera nothing here sees three dimensions, and that is carried
in the type rather than in a disclaimer. See
[ADR-0010](decisions/ADR-0010-projected-biomechanics.md).

- [x] **5.1 Primitives** — plane coordinates (the aspect fix and the y flip,
      both applied exactly once), interior and unsigned angles, tilt from
      vertical, line tilt, foreshortening, projection onto a direction
- [x] **5.2 Posture** — spine tilt, left/right knee flex, hip sway, head sway
      and head lift, the last three measured from the median address pose
- [x] **5.3 Rotation** — shoulder turn, pelvis turn, X-factor by foreshortening
      against an address baseline; shoulder and pelvis tilt, which need no
      baseline and survive any view
- [x] **5.4 Hand/arm** — hand path summed along the arc, peak hand speed in
      torso lengths per second, lead and trail arm angles with the lead side
      **measured** rather than assumed
- [x] **5.5 Timing** — backswing, downswing, follow-through, takeaway-to-impact
      and the tempo ratio; the only metrics here a camera position cannot spoil
- [x] **5.6 Contract + registry + propagation** — `Metric`, `MetricSet`,
      `RefusedMetric`; one declaration per quantity so a unit cannot drift from
      its documentation; `Anchor` carries Phase 4's confidence into every metric
      measured at it
- [x] **5.7 Tests** — 93 added (all pytest): primitives against 3-4-5 triangles
      and Thales' theorem, metrics against a body built from known angles on a
      deliberately non-square frame
- [x] **5.8 Verified on recorded swings; commit**

**Measured** on `data/face-on/PW_face-on.mp4` (68 frames, 30 fps, 0.15 s
window), at the top of the backswing. Every value was checked against the frame
it came from using `scripts/overlay_metrics.py`:

| Metric         | Value    | Confidence | obs  | anchor | method |
| -------------- | -------- | ---------- | ---- | ------ | ------ |
| Shoulder turn  | 50.2°    | 0.49       | 1.00 | 0.64   | 0.77   |
| Pelvis turn    | 35.3°    | 0.37       | 1.00 | 0.64   | 0.58   |
| X-factor       | 14.9°    | 0.37       | 1.00 | 0.64   | 0.58   |
| Spine tilt     | +3.6°    | 0.50       | 1.00 | 0.64   | 0.79   |
| Lead arm angle | 134.9°   | 0.39       | 0.98 | 0.64   | 0.62   |
| Tempo ratio    | 2.67 : 1 | 0.57       | 1.00 | 0.64   | 0.89   |

39 metrics produced, none refused. Lead side inferred as **left** (margin 1.00),
which the overlay confirms: the player sets up with the left hand above the
right and takes the club over their right shoulder.

`data/dtl/iron_dtl.mp4` produces 33 metrics and **refuses 3**. Its shoulders
project 0.10 torso lengths at address and 7.2× that mid-swing, so shoulder
turn, pelvis turn and X-factor are refused as not measurable from that view —
which is correct, because a down-the-line camera does not contain the
measurement. Tilts, posture, hands and timing all survive.

**Cost** (`scripts/benchmark_metrics.py`, median of 9):

| Clip           | Frames | Filter  | Phases | Metrics |
| -------------- | ------ | ------- | ------ | ------- |
| PW_face-on.mp4 | 68     | 17.9 ms | 0.3 ms | 1.9 ms  |
| iron_dtl.mp4   | 96     | 20.5 ms | 0.3 ms | 1.6 ms  |

Against ~1.3 s to extract poses for the same clip. Nothing is cached, for the
same measured reason as Phase 3.

**Deliberate deviations from the original plan:**

- **Phase 6.1a was pulled forward, partly.** The plan left the anisotropic-
  distance defect to Phase 6, and also said Phase 5's angles and lengths would
  not survive it — both of which are true, so the phase could not be done
  honestly without it. `PoseSequence` gains `FrameGeometry` (pose schema
  version 1 → 2, so cached extractions are refused and re-run), `FilteredSequence`
  carries it, and `biomechanics.geometry.plane_coordinates` applies the
  correction once. On the reference clips the aspect ratio is 1.7778, so a true
  45° line was reading as 29.4°. **What remains for Phase 6** is pushing the
  correction below the biomechanics layer so Phase 4's own travel ratios get it
  too, and re-measuring its thresholds against that.
- **The rotation baseline is the address pose, not the widest view in the clip.**
  The latter was built first and is worse; the reference footage broke it on the
  first run. See ADR-0010 — a maximum over a clip is a maximum over its noise.
- **`line_tilt_deg` orders its points by image x rather than folding onto
  (−90, 90].** The fold, carried over from Phase 4, reverses any leftward vector
  and so silently redefines the sign: "positive means the right shoulder is
  higher" held only while the player's right shoulder was on the right of the
  frame, and a player facing the camera has it on the left. Found by two of the
  engine's own outputs disagreeing about handedness on the reference clip.
  `phases/signals.py` was corrected to the same convention so the two cannot
  diverge; no Phase 4 detection rule reads those angles, so no event moved.
- **Lead and trail arms are measured, not configured.** At the top the hands sit
  over the trail shoulder; projecting their offset onto the shoulder line names
  the side. Where the view does not show it the side is `None` and those metrics
  are refused, rather than assuming the player is right-handed.
- **Rotation is not reported at address.** Address is the baseline it is
  measured against, so the value there is zero by construction — a restatement
  of the method rather than a measurement of the swing.
- **"Transition → impact" is not emitted.** The transition is the top, so it is
  the downswing duration under another name. Reporting it twice would give a
  reader two numbers that can never disagree and no way to know that in advance.
  Same reasoning that kept tempo out of Phase 4.
- **No confidence factor is a chosen constant.** Every `method` value is
  computed from something measured — frame-rate quantisation, the in-plane
  fraction of a segment, or the sine of the angle the arccos returned.
- **No UI.** Phase 4 built an inspector because checking an event's _timing_
  needs the video frame by frame. A metric is checked by drawing it on the frame
  it came from, which `scripts/overlay_metrics.py` does; the app's metric
  presentation belongs with the rest of the workflow in Phase 14.
- **`python -m analyzer.cli` was missing commands.** `if __name__ == "__main__":
app()` sat mid-file, so Typer never registered the commands defined below it
  under the module entry point while the installed `analyzer` script had them
  all. Moved to the bottom, with a comment saying why it stays there.

## Phase 6 — DTL + coordinate systems ✅

`python/analyzer/coordinates.py` · `python/analyzer/biomechanics/view.py` ·
[docs/coordinate-systems.md](coordinate-systems.md)

Two things, and they turn out to be the same thing twice. A measurement needs a
**reference frame** before its arithmetic means anything, and it needs a
**camera view** before its result means anything about a body. Phase 6 makes
both explicit, measured and carried.

- [x] **6.1 Frame types** — `LandmarkSpace` becomes the full vocabulary:
      `IMAGE`, `FRAME_WIDTHS`, `HIP_LOCAL`, plus `CAMERA` and `WORLD` named but
      refused with the phase that supplies them. Conversions and the conventions
      document
- [x] **6.1a The aspect correction moved below the filter** — into
      `pose/series.py`, so phase detection reads isotropic, upward-positive
      coordinates too. Thresholds re-measured, the two `torso_length` figures
      reconciled
- [x] **6.2 DTL metrics** — hand depth added; spine angle, head movement and hip
      travel gained their down-the-line readings. **Shaft orientation and club
      path are not built:** both need the club, which Phase 10 tracks
- [x] **6.3 View-tagged metrics** — `CameraView` measured from the shoulder
      line at address; every `Metric` carries the view and an `interpretation`;
      metrics a view cannot support are refused centrally
- [x] **6.4 Tests** — 18 added (all pytest): round trips across three frame
      shapes, pixel agreement by two routes, the refusals, view classification
      on face-on / down-the-line / oblique fixtures, and the tagging
- [x] **6.5 Verified on both reference clips; commit**

**Measured.** The view verdict, from the projected shoulder width at address:

| Clip           | Shoulders at address | Openness | View          | Confidence |
| -------------- | -------------------- | -------- | ------------- | ---------- |
| PW_face-on.mp4 | 0.83 torso           | 0.90     | face_on       | 1.00       |
| iron_dtl.mp4   | 0.10 torso           | 0.14     | down_the_line | 0.80       |

A factor of eight between them, against thresholds at 0.55 and 0.30.

The same computation, read through the view — **spine tilt at address is +4.6°
of lateral side bend face-on and +35.5° of forward posture angle down the
line.** Both are correct; neither is interpretable without knowing where the
camera stood, and before this phase nothing recorded that.

**What moving the correction cost and bought.** Phase 4's numbers changed, since
it had been measuring in the anisotropic frame:

| Quantity                 | Before | After   |
| ------------------------ | ------ | ------- |
| travel ratio (face-on)   | 3.86   | 2.92    |
| travel ratio (iron DTL)  | 2.30   | 2.14    |
| impact frame (face-on)   | 48     | **46**  |
| `torso_length` agreement | no     | **yes** |

The impact frame moving is the substantive one, and it moved the right way. The
hand-arc low point is frame 47. Impact is defined as peak hand speed, which the
README documents as running _marginally early_ because the hands peak before the
club reaches the ball — so 46 against a low point of 47 is consistent with that
and 48 was not. The anisotropic frame had been under-weighting vertical motion,
which biased the speed peak towards the most horizontal part of the swing.

`min_travel_ratio` (0.5) and `min_phase_travel_ratio` (0.15) were **re-measured
and left alone**: the worst margin across the reference clips is 2.14 against
0.5 on the gate and 1.19 against 0.15 on the phase bound, so the separation is
4x and 8x respectively and nothing warranted moving.

**Deliberate deviations from the original plan:**

- **Shaft orientation and club path are not delivered.** 6.2 lists them and both
  need the club tracked, which is Phase 10. Nothing here fakes them; the DTL
  metrics that landed are the ones a body alone supports.
- **`FRAME_WIDTHS` is a `LandmarkSpace`, not a parallel enum.** The plan said
  "image / normalized / camera / world frame types", which suggested a second
  vocabulary alongside the existing one. Two enums covering overlapping concepts
  is how a caller ends up converting between them; extending the one that is
  already threaded through `unit_for`, the reports and the contracts keeps a
  single answer to "which frame is this in".
- **`CAMERA` and `WORLD` are declared without being implemented.** Ordinarily
  this project does not name what it cannot do. A reference frame is the
  exception: the point is that Phase 8 and Phase 9 add the _capability_ rather
  than the concept, and until then asking for one raises an error naming the
  phase instead of reporting an unknown value.
- **The z channel is carried through the conversion rather than dropped.**
  MediaPipe documents its IMAGE z as roughly the scale of x, so it is already in
  frame widths and passes through unchanged. That avoids making the frame a
  two-dimensional special case in a filter built for three channels. It does not
  make z a measurement, and nothing above reads it.
- **The view has one gate, not two.** Rotation had been refused down the line by
  its own shoulder-span check, which measured the same number the view detector
  uses against a threshold five hundredths away. The view is now the single
  owner of "this recording does not contain this quantity", and the gate is
  applied centrally in `compute.py` — to the refusals as well as the values, so
  a blocked metric is refused once, naming the view, rather than once per anchor
  naming a symptom.
- **`DOWN_THE_LINE` means _along_ the target line, not _behind_ the player.**
  Shoulder foreshortening is identical from both ends and nothing else here
  separates them. The overlay showed this: `data/dtl/iron_dtl.mp4` is filmed
  from in _front_ of the player despite its name, and the classification is
  still right. The cost is the sign of anything measured along the frame's
  horizontal axis, so hand depth is reported as an image direction rather than
  as "towards the player".
- **`line_tilt_deg` lost its y negation** in `phases/signals.py`, since the
  frame arriving there already points up. No event moved: nothing in detection
  reads those angles.

## Post-Phase 6 — what tour-pro footage found ✅

`python/analyzer/biomechanics/view.py` · `rotation.py` · `pose/series.py`

Two reference clips of a tour professional were added after Phase 6. Both
contain a fuller turn than any earlier footage and both are slow motion, and
between them they falsified two things the project had already documented as
settled. Details in the amendment to
[ADR-0010](decisions/ADR-0010-projected-biomechanics.md).

- [x] **Slow motion is declarable** — `--slow-motion` / `slow_motion_factor`,
      applied to the timestamps once, below the filter, so every duration, speed
      and window lands on a real clock. Reported on the three contracts that
      carry durations, because the timestamps afterwards are real seconds and no
      longer index the video
- [x] **The detector says when a clip looks like slow motion** — a swing's shape
      with every phase long in the same proportion, and the smallest factor that
      would fit
- [x] **Rotation carries a measured uncertainty** — `Metric.uncertainty`, in
      degrees, from the projected span's stability over a real-time window,
      propagated through the arccos. Refused above a bound, and reported as
      unknown where too few frames exist to measure it
- [x] **Tests** — 18 added (all pytest)

**Measured.** MediaPipe reports a visibility of **1.00** for both shoulders on a
swing where its own span estimate varies between 0.27 and 0.68 of the address
width across 27 frames of near-static pose — an implied turn of 47 to 74 degrees.
No confidence the estimator supplies can catch that, which is why stability is
now measured from the geometry.

`data/face-on/rory_face_on.mp4` at a factor of 7:

| Metric              | Value       | Note                                  |
| ------------------- | ----------- | ------------------------------------- |
| Shoulder turn (top) | +53.0 ± 5°  | the most trustworthy of the rotations |
| Pelvis turn (top)   | +27.9 ± 15° | at the refusal bound                  |
| X-factor (top)      | +25.2 ± 16° | nearly worthless, and now says so     |
| Tempo               | 1.83 : 1    | ratio, but see the caveat below       |

Every event resolves at 1.00 once the clock is right, which no 30 fps clip in
this project manages: slow motion is a high-speed capture.

**The first observed impact in the project.** The ball is on the tee at frame
360 and gone at 361. Peak hand speed — Phase 4's primary estimate — lands 17 to
40 frames away depending on the assumed factor; the lowest point of the hand arc,
carried only as corroboration, lands within 4 and is stable across it. **Phase 11
should evaluate swapping them before adding anything.** One clip is not enough to
change it now.

**Open, and not resolved here:**

- The two tour clips give apparent tempos of 1.83 and 1.3 against the ~3:1 a
  tour swing is known for. A uniform slow-motion factor cannot change a ratio,
  so either the clips are speed-ramped — plausible for social-media reposts — or
  the takeaway is misplaced on slowed footage. Not determined; until it is,
  treat those clips as geometry-only.
- The factor itself is supplied, not measured, and it moves events. Nothing in
  the video can recover it.

## Phase 7 — Two-camera synchronisation ✅

`python/analyzer/sync/{timemap,correlate,anchors,align}.py` ·
`python/analyzer/projects/store.py` ·
[ADR-0011](decisions/ADR-0011-affine-time-map.md)

Two cameras, two clocks, neither aware of the other. Everything Phase 8 and
Phase 9 do rests on relating them, because triangulating two views is only
meaningful for two views of the _same instant_ — and the error in the relation
propagates into every reconstructed point, so it has to be carried rather than
assumed away.

- [x] **7.1 Two-video project model (SQLite)** — the first state in this engine
      that cannot be recomputed, so it lives under `data_dir()` rather than in
      the cache. Clips identified by content, not by path; foreign keys on;
      a schema version refused rather than guessed at
- [x] **7.2 `SyncModel` / `TimeMap` contract** — an affine map stated at the
      anchor centroid, so its two uncertainties are independent and combine in
      quadrature. `rate_estimated` says whether the second parameter was
      measured or assumed
- [x] **7.3 Manual sync UI** — both clips drawn on one clock, frame steppers per
      camera, pinned instants replacing the detected events entirely
- [x] **7.4 Automatic sync** — masked normalised cross-correlation of hand speed
      (FFT, partial overlap handled per lag) and an affine fit to the paired
      events. Both always computed; neither ever averaged into the other
- [x] **7.5 Report residual (ms) and confidence** — residual against the
      quantisation floor the two frame rates impose, plus a decomposed
      confidence and the gap between the two independent estimators
- [x] **7.6 Tests** — 91 added (60 sync, 31 projects) plus 18 Vitest and 8
      Playwright: known offsets, mismatched fps, partial overlap, a stale schema
- [x] **7.7 Measured against known offsets; commit**

**Measured** (`scripts/benchmark_sync.py`, Apple M1 Pro / macOS 26.4.1). One
synthetic swing sampled by two simulated cameras, at the sigma = 0.0014 frame
widths Phase 3 measured from real footage; median absolute error over 9 seeds:

| Pair          | Floor   | Clean  | With landmark noise |
| ------------- | ------- | ------ | ------------------- |
| 240 + 240 fps | 1.7 ms  | —      | **0.5 ms**          |
| 120 + 120 fps | 3.4 ms  | 0.0 ms | **0.4 – 4.2 ms**    |
| 120 + 30 fps  | 9.9 ms  | —      | **0.3 ms**          |
| 30 + 30 fps   | 13.6 ms | —      | **0.5 ms**          |

**The benchmark overturned the design's central assumption.** Four swing events
look like four clocks and are not. Perturbing the landmarks and watching where
each lands over eight seeds: the top moves 8 ms, impact 117 ms, the finish
350 ms and the takeaway **542 ms** — the last two are threshold crossings on a
signal that is barely moving there, and one seed put the takeaway on frame zero.
Anchoring on all four gives 14–27 ms; correlating the two speed signals gives
0.4–1.8 ms. So the offset now comes from the correlation and the events supply
the two things it cannot produce, a clock rate and a residual.

**What a second camera makes measurable.** Phase 6 recorded that a slow-motion
factor "is supplied, not measured, and nothing in the video can recover it".
True of one clip, false of two:

| Factor supplied | True rate | Fitted | Estimated? |
| --------------- | --------- | ------ | ---------- |
| correct         | 1.0000    | 1.0000 | no         |
| 2% wrong        | 0.9804    | 1.0000 | no         |
| 10% wrong       | 0.9091    | 0.8869 | **yes**    |
| 20% wrong       | 0.8333    | 0.7874 | **yes**    |
| 40% wrong       | 0.7143    | 0.7592 | **yes**    |

The _ratio_ of two supplied factors is recoverable even though neither clip can
recover its own. A rate further than 2% from 1.0 is reported as a slow-motion
problem, because real camera clocks do not differ by anything approaching that.

**The frame-rate floor is the number to know before buying a camera.** Two at
30 fps give 13.6 ms. Replacing one with a 240 fps camera gives 9.7 ms and can
never beat 9.6 ms however fast it gets, because the 30 fps clip contributes that
much alone. Replacing both gives 1.7 ms. Upgrading one camera of a pair is worth
a factor of sqrt(2) at most.

**On the only real pair in this project, the answer is "these are not the same
swing", and that is the useful part.** `rory_face_on.mp4` against `rory_dtl.mp4`
aligns at −258 ms with a residual of 95.8 ms rms against a 2.4 ms floor — 40x —
and per-anchor residuals of −82, −92, +90 and −116 ms. Confidence 0.10. Nothing
here can see that two cameras were pointed at one event; what it can see is that
no offset and no clock rate reconcile two different tempos.

Cost: 0.5 ms to align two 312-frame clips, on signals already filtered. Nothing
is cached, for the same measured reason as Phase 3.

**Deliberate deviations from the original plan:**

- **7.7 is measured against known offsets, not against manual ground truth, and
  the benchmark says so in its own docstring.** Ground truth for synchronisation
  needs two cameras that genuinely filmed one swing at once with their clocks
  related by something outside this system — a clapperboard, a flash, a genlock.
  No such recording exists here. What is measured is one synthetic swing sampled
  twice, which establishes that the arithmetic recovers an offset that was put
  in, and cannot establish that a real pair does: the simulated cameras see the
  same projection of the same body, and two real cameras do not. One command
  finishes this once a simultaneous pair exists:
  `uv run --project python python scripts/benchmark_sync.py --real <a> <b>`
- **The offset does not come from the events.** 7.4 said "cross-correlation of
  hand-speed + event refinement", which assumes the events refine the
  correlation. Measured, it is the other way round by a factor of fifteen. The
  events are kept because they are the only evidence that can determine a clock
  rate or produce a residual, and both matter — but they do not set the offset.
  Full reasoning in [ADR-0011](decisions/ADR-0011-affine-time-map.md).
- **The rate is refused by default, on a significance test rather than a
  threshold.** A rate fitted from four instants over 1.5 s carries a fractional
  error of about a frame divided by the span, and applied ten seconds away that
  is worse than assuming the clocks agree. It is kept only when it sits more than
  two standard errors from 1.0. Dropping an insignificant one moved a clean
  pair's confidence from 0.72 to 0.95 with no change in accuracy.
- **Phase 4's event confidence does not detect a badly located event**, which is
  a finding about Phase 4 rather than about this phase. It scores a takeaway at
  0.95 on seeds where that takeaway landed 500 ms from the truth, so gating
  anchors on it changes nothing measurable (22–28 ms at every threshold from 0.0
  to 0.5). Recorded here; not fixed here.
- **The residual stops separating "same swing" from "different swing" once the
  footage is noisy.** A correctly aligned pair scatters 34 ms and an 8% tempo
  difference scatters 44 ms. What does separate them is the correlation peak's
  margin over its nearest rival — 0.36, 0.09, and 0.01 at a 15% difference — so
  that margin is what `SyncConfidence.agreement` reports for a map whose offset
  came from the correlation. The residual is still shown, and still refuses
  above `max_residual_ms`.
- **Both clips share one smoothing window.** Smoothing them differently would
  shift the features the alignment keys on by an amount nothing measures, so the
  coarser clip sets the window for the pair — and a 30 fps camera cannot support
  the 0.10 s default at all. That arrives here as a refusal naming the window
  that clip's rate would support, carried up from Phase 3's own report.
- **`Project` and `ProjectList` are not exported to TypeScript.** They are
  reachable over RPC and from `analyzer project`, but no UI renders them yet —
  project management is Phase 14.1 — and exporting types nothing draws would make
  the app's type surface a description of the plan rather than of the app. Same
  reasoning that kept `SequenceFilterReport` out until Phase 4 built its panel.
- **A stored alignment is JSON inside a row, not four normalised tables.** A
  `SyncModel` is a nested document read whole and never queried by part;
  normalising it would create four tables to keep in step with one Pydantic
  model that is already the authoritative definition. The schema version is
  repeated in its own column so a stale row can be found without being parsed.
- **The synthetic swing fixture moved to `tests/synthetic.py`.** Phase 7 needs
  one swing sampled by two cameras with different clocks, which is impossible
  while the fixture only knows how to produce a clip. A second copy of it would
  have been a second definition of what a swing looks like.

## Phase 8 — Camera calibration ✅

`python/analyzer/calibration/{board,detect,intrinsics,stereo,apply}.py` ·
[ADR-0012](decisions/ADR-0012-calibration-coverage.md)

The first layer that turns a picture back into a statement about space. It also
contains the phase's whole argument, which is a **negative**: the number every
calibration tool prints as its quality measure is blind to the one failure that
matters, and so is the parameter uncertainty that looks like the principled
replacement. Only coverage catches it, so coverage is the gate.

- [x] **8.1 Calibration contracts** — `CameraIntrinsics` (K, distortion, and the
      fit's own parameter uncertainties), `CoverageReport`, `CalibrationQuality`,
      `StereoCalibration`, `PairingReport`, `CameraRig`, and `CalibrationStatus`
      with three values rather than two
- [x] **8.2 Charuco detection + intrinsics** — board generated from the same spec
      it is detected with; views selected for being _different_, not for
      existing; the distortion model named rather than defaulted
- [x] **8.3 Stereo extrinsics** — intrinsics held fixed; frames paired through
      Phase 7's time map, with the sync uncertainty converted into **pixels** by
      multiplying it by the board's measured image speed
- [x] **8.4 Capture/review UI** — per-view reprojection error, the coverage
      numbers that decide usability, and a map of where in the frame each board
      view actually landed
- [x] **8.5 Status gating enforced across the metric layer** —
      `MetricDefinition.requires` and a central gate in `compute.py`, mirroring
      the view gate. It blocks nothing today, and a test asserts that it would
- [x] **8.6 Tests** — 59 added: a rendered board photographed by a camera whose
      parameters are inputs, detected by the real detector
- [x] **8.7 Document reconstruction limits; commit**

**Measured** (`scripts/benchmark_calibration.py`, Apple M1 Pro / macOS 26.4.1).
A synthetic camera at fx = 1400 px on a 1920x1080 frame with k1 = −0.28; median
of 3 seeds, 14 board views each. **A floor, not an estimate**: the renderer has
no blur, no defocus and no bowed sheet.

The headline, sweeping how much the board was tilted:

| tilt spread | RMS px | reported σ(fx) | **true fx error** |
| ----------- | ------ | -------------- | ----------------- |
| ±0°         | 0.220  | 0.012%         | **6.33%**         |
| ±2°         | 0.210  | 0.012%         | **10.04%**        |
| ±10°        | 0.269  | 1.22%          | **23.25%**        |
| ±20°        | 0.245  | 0.656%         | 0.10%             |
| ±35°        | 0.218  | 0.314%         | **0.05%**         |

The residual is flat across a range over which the focal length error varies by
a factor of four hundred, and **the reported uncertainty is thirty times smaller
exactly where the answer is worst** — because the distortion coefficients absorb
the degeneracy and leave a tightly determined wrong answer. That second half
overturned this phase's own design. Full reasoning in
[ADR-0012](decisions/ADR-0012-calibration-coverage.md).

Frame coverage, and the distortion model, both measured the same way:

| capture         | model   | fx err | k1 err | error at the frame edge |
| --------------- | ------- | ------ | ------ | ----------------------- |
| corners reached | rt4     | 0.05%  | 0.003  | **2.1 px**              |
| corners reached | rt5     | 0.05%  | 0.004  | 9.5 px                  |
| corners reached | pinhole | 9.64%  | 0.280  | 202.3 px                |
| centred only    | rt4     | 0.14%  | 0.001  | 34.4 px                 |
| centred only    | rt5     | 0.13%  | 0.004  | **180.6 px**            |

OpenCV's fifth coefficient is worth a factor of four at the edge on a good
capture and a factor of five in the other direction on a poor one, so the
default is four. Eight well-spread views are enough (0.08% focal error); twenty
give 0.02%.

**What the lens does to a landmark, which is what Phase 8 buys a single-camera
user.** Undistortion moves a point near the frame edge by:

| lens                  | at the edge | worst  | h-fov |
| --------------------- | ----------- | ------ | ----- |
| phone main (k1 −0.28) | 171.5 px    | 199 px | 68.9° |
| phone wide (k1 −0.42) | 219.9 px    | 248 px | 93.7° |
| mild (k1 −0.10)       | 34.0 px     | 41 px  | 56.1° |
| long lens (k1 −0.02)  | 2.5 px      | 3.0 px | 35.5° |

On 1920x1080. Every angle, distance and speed Phases 4 to 6 compute is taken
from landmark positions carrying that displacement, and nothing in those numbers
shows it. This is the one thing a calibration does for a single camera — and it
is still not depth.

**What it changes on real footage, and what it does not.** Undistorting
`data/face-on/PW_face-on.mp4` with a plausible phone lens (fx = 0.73 x frame
width, k1 = -0.28) moves the metrics by:

| metric                 | uncalibrated | undistorted | delta     |
| ---------------------- | ------------ | ----------- | --------- |
| Trail arm angle (top)  | 154.75°      | 155.93°     | +1.19°    |
| Shoulder turn (impact) | 24.39°       | 25.26°      | +0.87°    |
| X-factor (impact)      | 6.10°        | 6.79°       | +0.69°    |
| Shoulder turn (top)    | 50.21°       | 50.62°      | +0.41°    |
| Peak hand speed        | 19.32        | 19.91       | **+3.0%** |
| Hand path length       | 3.304        | 3.360       | +1.7%     |

**and moves no swing event at all** -- takeaway 13, top 38, impact 46, finish 65
either way. Phase detection reads the shape of a speed curve, and a smooth
radial correction does not change where its extrema sit; the metrics read
absolute angles and distances, and it does.

**That lens is assumed, not measured**, because nobody has calibrated the phone
that shot this clip. What the table establishes is the _size_ of the correction
on real footage, not a corrected result -- and it is modest here because the
player is near the middle of the frame, which is where the lens bends least.
A swing framed closer to the edge inherits more of the 171 px.

**Two unsynchronised cameras can be stereo-calibrated, and the condition is
measurable.** Both cameras at 30 fps, clocks related to 12 ms:

| board             | image speed | pairing error | pairs | baseline err | rotation err |
| ----------------- | ----------- | ------------- | ----- | ------------ | ------------ |
| held 1 s          | 0.0 px/s    | 0.00 px       | 7     | **0.104%**   | **0.031°**   |
| held 0.2 s        | 0.0 px/s    | 0.00 px       | 7     | 0.104%       | 0.031°       |
| held **3 frames** | 0.0 px/s    | 0.00 px       | 7     | 0.104%       | 0.031°       |
| held 2 frames     | —           | —             | 2     | refused      | —            |
| never stops       | —           | —             | 0     | refused      | —            |

**Three frames of stillness is the whole requirement** — a tenth of a second at
30 fps. The sync error is multiplied by the board's image speed, so a still
board makes an unsynchronised pair as good as a genlocked one; a moving one is
refused rather than fitted badly. The images are equally sharp in every row,
which is why this is measured rather than left as advice.

**Cost:** 3.7 ms/frame to detect the board at 1920x1080, 13.5 ms to fit
intrinsics from 14 views.

**Deliberate deviations from the original plan:**

- **The quality gate is not the reprojection error, and finding that out
  overturned the design.** 8.1 lists "RMS" among the contracts and the obvious
  reading is that RMS is what `CalibrationStatus` keys on. It cannot be: the
  sweep above shows it flat while the answer moves by 400x. The intended
  replacement — OpenCV's own parameter standard deviations — fails the same test
  and fails it _backwards_. Coverage is the gate; the other two are reported for
  what they each genuinely say. [ADR-0012](decisions/ADR-0012-calibration-coverage.md).
- **8.5 blocks nothing, and that is the assertion rather than an omission.**
  Every metric in the registry measures the image plane, which an uncalibrated
  camera supplies; a calibration makes those measurements cleaner and does not
  promote one of them into a claim about three dimensions. The gate is built,
  enforced and tested against a registry entry that demands stereo, so Phase 9's
  metrics meet machinery that predates them rather than a check written on the
  day the first metric needs relaxing.
- **`CalibrationStatus` has three values, because `INTRINSICS` is not half of
  `STEREO`.** A calibrated single camera knows which _direction_ a pixel came
  from and nothing about how far away it was. `apply.bearings` returns unit
  vectors for exactly that reason, and there is deliberately no function here
  that returns a 3D point.
- **Undistortion went into `pose/series.py`, below the filter**, next to the
  aspect correction and the slow-motion factor, for the third instance of the
  same reason: the filter is linear, so a correction applied to positions before
  fitting emerges correctly signed in the velocity and acceleration rather than
  needing a second correction kept in step by hand.
- **The sync error reaches stereo calibration as a length in pixels, not as
  milliseconds.** Phase 7 reports `TimeMap.uncertainty_at`; multiplying it by the
  board's measured image speed puts it in the same unit as the reprojection error
  it would otherwise be mistaken for. This is what makes the capture instruction
  a measurement rather than folklore.
- **Board speed is measured from _every_ detection, not from the selected
  views** — and getting that wrong first is what the stereo benchmark caught.
  Selection keeps one view per board position, so differencing selected views
  measures the speed of carrying the board _between_ positions, which is large
  however patiently it was held at either end. Pairing now runs over all
  detections and distinctness is selected on the pairs.
- **A mis-paired stereo capture is caught by the residual, not by the clock.**
  Board stations a second apart with an offset wrong by exactly a second pair
  every frame with its neighbour, simultaneous to the millisecond and showing the
  board in two different places. No time-based check can see that; the fit's
  residual explodes and refuses. So stereo _does_ gate on reprojection error —
  there it is measuring a correspondence rather than a model's fit to its own data.
- **`CameraRole` moved to `contracts/camera.py`.** A rig is keyed by role and a
  project holds a rig, which made `projects` and `calibration` mutually
  dependent. A module for one enum is worth it when the alternative is a cycle;
  `projects` re-exports it so nothing else changed.
- **The projects database gained a real migration** (version 1 → 2, adding
  `rigs`). Phase 7 recorded "one version and no upgrade path yet, which is the
  honest state". A project is the only state here that cannot be recomputed from
  the files, so the second version migrates rather than refuses. A _newer_
  database is still refused by name.
- **A stereo calibration holds both intrinsics fixed.** Re-fitting them jointly
  lets the optimiser trade focal length against baseline over the small angular
  range two cameras share, replacing two calibrations each measured from a
  proper spread of board views with one determined by whatever both cameras
  happened to see at once.
- **No real calibration footage exists in this project**, so 8.7's limits are
  documented against a synthetic camera and the benchmark says so in its own
  docstring. One command finishes this once a board has been filmed:
  `uv run --project python python scripts/benchmark_calibration.py --real <dir>`

## Phase 9 — Multi-view 3D reconstruction ✅

`python/analyzer/reconstruction/{pairing,triangulate,skeleton,reconstruct}.py` ·
`python/analyzer/biomechanics/spatial.py` ·
[ADR-0013](decisions/ADR-0013-epipolar-blindness.md)

Two calibrated rays meet, and the result is metres — the first thing this engine
has ever said about a body rather than about a picture of one. It also contains
the phase's whole argument, which is a **negative** of exactly the shape Phase 8
found one layer down and is not the same fact: the number that scores a
triangulation is blind to half of the error that matters, and blind to it in a
nameable direction.

- [x] **9.1 DLT triangulation + non-linear refinement** — Hartley's linear
      solution as a seed, then batched Gauss-Newton on the actual reprojection
      error in both views. `StereoGeometry` composes the rig once, so the
      reference camera sits at the origin and every point is a `CAMERA`
      coordinate by construction rather than by a later transform
- [x] **9.2 Reprojection error + bone-length consistency** — and the finding that
      these answer different questions. Bone variation is the check the residual
      cannot make; left-right symmetry is corroboration, which is **less** than it
      was designed to be (below)
- [x] **9.3 Per-joint per-frame confidence** — ray convergence angle, reprojection
      residual, and a positional uncertainty in **metres** propagated through the
      triangulation's own Jacobian from a pixel sigma measured on the clip
- [x] **9.4 Synthetic ground-truth harness** — `tests/synthetic_body3d.py`: a
      swing that exists in three dimensions, with exact bone lengths by inverse
      kinematics, filmed by two cameras with different lenses, different frame
      rates and different clocks. The first pair in this project that genuinely
      is one swing from two viewpoints
- [x] **9.5 Hard gate made real** — six metrics declaring `requires=STEREO`, so
      the gate Phase 8 built and tested against nothing now blocks something
- [x] **9.6 Tests** — 50 added (all pytest), plus 8 earlier assertions updated
      that Phase 9 was written to invalidate
- [x] **9.7 Measured on synthetic truth; commit**

**Measured** (`scripts/benchmark_reconstruct.py`, Apple M1 Pro / macOS 26.4.1).
A 1.78 m synthetic body filmed by a 120 fps face-on camera and a 60 fps
down-the-line one, 4.8 m apart at 90°. **A floor, not an estimate**: there is no
pose estimator in the fixture, so no blur, no occlusion, no mis-tracked wrist.

Accuracy against isotropic landmark noise, over the eight landmarks that move:

| landmark sigma | median     | p95         | hands p95   | reprojection | uncertainty |
| -------------- | ---------- | ----------- | ----------- | ------------ | ----------- |
| 0.5 px         | 1.0 mm     | 1.9 mm      | 1.9 mm      | 0.16 px      | 1.0 mm      |
| **2.7 px**     | **5.5 mm** | **10.4 mm** | **10.5 mm** | 0.85 px      | 5.4 mm      |
| 5.0 px         | 10.1 mm    | 19.3 mm     | 19.4 mm     | 1.55 px      | 10.1 mm     |
| 10.0 px        | 20.4 mm    | 38.5 mm     | 38.3 mm     | 2.45 px      | 20.2 mm     |

2.7 px is the 0.0014 frame widths Phase 3 measured on real footage, at 1920 px
wide — so the bolded row is what this pipeline would do on a perfect capture of a
real swing.

**The headline is the negative.** Displace every landmark in one view _along its
epipolar line_ — the direction in which a wrong depth is a perfect fit — and:

| along-epipolar | median 3D error | **reprojection** | bone variation |
| -------------- | --------------- | ---------------- | -------------- |
| 0 px           | 0.0 mm          | **0.00 px**      | 2.6%           |
| 1 px           | 1.4 mm          | **0.00 px**      | 4.8%           |
| 2 px           | 2.7 mm          | **0.00 px**      | 9.6%           |
| 4 px           | 5.4 mm          | **0.00 px**      | 19.1%          |
| 8 px           | 10.8 mm         | **0.00 px**      | 37.8%          |

Not insensitive — **exactly zero**, because every one of those detections has a
perfect 3D explanation. The same pixel counts applied isotropically give
0.31 / 0.63 / 1.25 / 2.51 px, so the residual is a real measurement of the
component it can see and is not the accuracy. Bone variation is what notices.

**The gate is the geometry, and it is flat in the residual too.** Camera
separation, at 2.7 px of noise:

| separation | ray angle | median error | **reprojection** | uncertainty | 1/sin |
| ---------- | --------- | ------------ | ---------------- | ----------- | ----- |
| 90°        | 94°       | 5.5 mm       | **0.85 px**      | 5.4 mm      | 1.0x  |
| 45°        | 48°       | 6.5 mm       | **0.84 px**      | 8.7 mm      | 1.3x  |
| 30°        | 32°       | 8.2 mm       | **0.84 px**      | 12.7 mm     | 1.9x  |
| 15°        | 16°       | 14.3 mm      | **0.84 px**      | 25.0 mm     | 3.6x  |
| 8°         | 9°        | 26.0 mm      | **0.84 px**      | 46.6 mm     | 6.7x  |

Flat to two decimal places over a range where the error grows 4.7x. That is
Phase 8's tilt sweep again, one layer up, and it is why `min_convergence_deg` is
the gate and is checked **before** the residual.

**Phase 8's capture instruction does not survive the subject moving.** Stereo
calibration tolerates unsynchronised cameras because pairing error is
`sync_error × image_speed` and a board can be held still. Nothing in a swing is:

| target fps | pairing       | hands p95   | nearest-frame cost     |
| ---------- | ------------- | ----------- | ---------------------- |
| 240        | resampled     | 0.0 mm      | 1.6 px                 |
| 240        | nearest frame | 2.4 mm      | 1.6 px                 |
| 120        | nearest frame | 2.4 mm      | 3.3 px                 |
| 60         | **resampled** | **0.0 mm**  | 6.5 px                 |
| 60         | nearest frame | **12.3 mm** | 6.5 px                 |
| 30         | refused       | —           | Phase 3's window floor |

No landmark noise in that table, so every millimetre is the pairing. Resampling
uses cubic Hermite interpolation of the position _and velocity_ Phase 3 already
fitted — no new kernel and no second smoothing pass.

**The 3D rotations, against angles that are inputs.** The fixture turns the
shoulders 92° and the pelvis 45° about a declared axis, so X-factor has a true
value — which no single-camera measurement in this project has ever had:

| landmark sigma | shoulder                                          | pelvis | X-factor | reported ± | worst error |
| -------------- | ------------------------------------------------- | ------ | -------- | ---------- | ----------- |
| 0.0 px         | 92.0°                                             | 45.0°  | 47.0°    | 0.0°       | 0.0°        |
| 1.0 px         | 92.3°                                             | 44.6°  | 47.3°    | 1.1°       | 0.9°        |
| 2.7 px         | 91.6°                                             | 44.5°  | 47.2°    | 3.0°       | 0.5°        |
| 5.0 px         | — no swing detected: **Phase 4 refuses the clip** |        |          |            |             |

Truth: 92 / 45 / 47. The last row is the finding: **at 5 px of scatter the
reconstruction is fine and the phase detector is not**, so there is no instant to
anchor a rotation to. On noisy footage the limit on a 3D metric is Phase 4, not
the triangulation.

Non-linear refinement is worth a little and is kept: 5.7 → 5.5 mm median at
2.7 px, residual 1.02 → 0.85 px.

**Cost:** 70 ms to reconstruct 312 frames (8,778 points), against ~95 ms to
filter both clips and ~1.3 s per clip to extract poses in the first place.
Nothing is cached, for the same measured reason as Phase 3.

**Deliberate deviations from the original plan:**

- **9.2's two checks are not equals, and finding that out corrected this
  phase's own design.** Left-right symmetry was built on the reasoning that a
  consistent depth bias would give a _stably_ wrong bone length that the
  variation check could not see. Measured, that reasoning is wrong: a swing
  rotates the body, so a displacement constant in the camera's frame is not
  constant relative to the bone. Displacing one elbow 5 cm along the optical axis
  makes the forearm's variation 11% — past the bound — while the left-right
  disagreement is 2.6%, well inside it. Symmetry is kept because it **localises**
  (it names which side is worse, which a per-segment number does not), and it is
  documented as corroboration rather than as the instrument it was meant to be.
- **`WORLD` is not delivered, and the obstacle is not arithmetic.** Phase 6
  recorded that WORLD "needs a calibrated stereo pair, which arrives in Phase 9".
  A stereo pair is necessary and not sufficient: a scene-fixed frame also needs a
  gravity direction and a target line, and the cameras know neither their own
  attitude nor which way the shot goes. Both fall out of a capture that lays the
  calibration board flat on the ground with one edge along the target line, which
  `data/README.md` does not currently ask for. So Phase 9 produces `CAMERA` and
  names what WORLD is missing. The cost is specific: a **3D spine tilt** needs a
  vertical and is therefore absent from the metric family. Nothing else is —
  lengths, joint angles and speeds are invariant to the frame, and the rotations
  are taken about the body's own measured spine axis.
- **`CAMERA` is produced and still refused by `require_reachable`.** It is not a
  conversion of one clip's landmarks, it is a measurement made from two of them,
  and a `landmark_series` call that returned it would have had to invent the depth
  the projection destroyed. The error message names triangulation rather than a
  phase number.
- **The target clip is resampled, not paired.** 9.1 does not mention pairing at
  all, which quietly assumes the two cameras' frames correspond. They do not, and
  Phase 8's answer to that — pair nearest frames, hold the subject still — is
  unavailable on a swing. The table above is the measurement that decided it.
- **3D velocity is not a finite difference of the reconstructed track.**
  Differentiating the triangulation's least-squares condition through time gives
  `X_dot = (J'J)^-1 J' x_obs_dot`, which assembles the 3D velocity from the two
  views' _fitted_ image velocities. That is Phase 3's rule — a derivative comes
  from the fit, never from differencing what the fit produced — applied one layer
  up, and it is exact at a converged reconstruction.
- **The pixel sigma that drives every uncertainty is measured from the clip**,
  as the filter's own residual RMS, rather than chosen. A residual of exactly
  zero is **not** treated as evidence of perfect landmarks — that is what an
  exactly-determined local fit produces every time — so that case falls back to
  the 0.0014 frame widths Phase 3 measured on real footage, and says so.
- **Six new metric names rather than promoting the existing five.**
  `SHOULDER_TURN` is a rotation inferred from how much a line shortened in one
  picture and `SHOULDER_TURN_3D` is the angle between two measured directions in
  space. They will disagree, the disagreement is informative, and one name whose
  meaning depended on whether a rig happened to be calibrated would make two
  numbers that cannot be compared look like one that can. The 3D ones carry
  `meaning` rather than `meanings`, and an empty `refused_in` — a reconstructed
  quantity means the same thing from every camera position, and a down-the-line
  clip that refuses `SHOULDER_TURN` supports `SHOULDER_TURN_3D` from the same
  footage.
- **The sign of a 3D turn is measured, not assumed.** "Away from the target"
  needs the target line and "clockwise from above" needs an up; neither is
  available. What is available is the swing, so the direction the shoulders had
  turned at the top defines positive for the whole clip.
- **The fixture's framing is set by measurement, twice.** At 4.2 m the torso
  spans 0.072 frame widths and Phase 4 refuses to call the clip a swing at
  ordinary noise levels, so the cameras moved to 3.4 m. And the clock offset is
  deliberately not a whole number of frames: at exactly 0.35 s every 120 fps
  reference instant maps onto an exact 60 fps target frame, so nearest-frame
  pairing is _exact_ and the sweep measuring what it costs measures zero.
- **A camera convention bug the reconstruction could not see.** `look_at` built
  with `cross(world_up, forward)` produces two cameras that are both upside down
  and mirrored. Triangulation is unaffected — the reference frame is merely
  rotated, and the measured error was zero — and Phase 4 caught it immediately,
  because the hands reached their _lowest_ point at the top of the backswing.
  There is now a test asserting the convention directly.
- **No UI.** Phase 4 built an inspector because checking an event's timing needs
  the video frame by frame; a reconstruction is checked by its own bones and by
  the numbers `analyzer reconstruct` prints. The 3D viewport is Phase 15, and it
  should render something this phase has already validated rather than being the
  thing that validates it.
- **The rig is checked before any footage is read, and the first version was
  not.** Running `analyzer reconstruct` on an uncalibrated project reported "no
  extracted poses for this clip" — the last thing that went wrong rather than the
  first thing that was wrong — because the dispatcher aligned the pair and loaded
  both clips before `reconstruct_pair` looked at the rig. `require_stereo_rig`
  is now split out and called up front: every check in it reads the project and
  none reads the footage, which is what makes it cheap enough to run first.
  Found by running the command, not by a test.
- **Three pre-existing TypeScript errors were fixed**, in Phase 8's
  `CalibrationPanel`. They are not Phase 9's, and `npm run check:all` was already
  red on `cf41d97`: an `exactOptionalPropertyTypes` violation passing an
  `undefined` tone, an optional `observations[]` used as required, and a test
  variable TypeScript narrows to `null` because its assignment happens inside a
  callback. Fixed rather than left, because a phase boundary that cannot be
  verified green is not a phase boundary.
- **No real reconstruction exists in this project**, and the benchmark says so in
  its own docstring. It needs two calibrated cameras that filmed one swing
  simultaneously; Phase 7 established that the only two-angle pair here is not the
  same swing, and Phase 8 that no real calibration footage exists. Everything
  above is synthetic, and the errors are a floor.

## Phase 10 — Club tracking ✅

`python/analyzer/club/{detector,hough,geometry,track,extract}.py` ·
[ADR-0014](decisions/ADR-0014-club-evidence-and-coverage.md)

The first thing this engine measures that is **not a body**. Everything below
reads landmarks a model was trained to find, which arrive carrying the model's
own opinion of how well it saw them. A golf shaft has no model, no landmark index
and no reported visibility — it has edges. The phase's argument is the shape
Phases 8 and 9 both found one and two layers down, and it is two different facts:
the number a line detector reports about itself ranks a door frame above a golf
club, and it does so most decisively in the frames that matter most.

- [x] **10.1 `ClubDetector` Protocol + `ClubFrame` contract** — a shaft reported
      as a **ray from the hands**: an origin the pose layer supplies, a direction
      the image determines, and a length it frequently does not. `ClubFrame`
      carries an observation or a named reason there is none, for every frame
- [x] **10.2 ROI → Canny → probabilistic Hough → geometric filtering** — Canny
      thresholds from the searched region's own median rather than fixed; every
      bound stated in torso lengths; `support` measured along the ray **from the
      hands**, which makes "the shaft reaches the grip" a measurement instead of
      a second threshold
- [x] **10.3 Temporal tracking + occlusion handling** — seeded where the evidence
      is best and grown outward, not started at frame zero. The predicted
      direction **scores** candidates and never **supplies** one, so an occluded
      frame emits nothing and leaves a visible hole
- [x] **10.4 Club-head position + trajectory** — emitted only where the evidence
      ran to the end of the club, which on a smeared downswing is a minority of
      frames. Plus a club-head impact estimate, independent of Phase 4's and
      deliberately not replacing it
- [x] **10.5 Overlay rendering** — `scripts/overlay_club.py`, drawing the tracked
      shaft, the rival it was chosen over, the head trail over observed frames
      only, and **the refused frames stamped with the reason**
- [x] **10.6 Tests** — 91 added (all pytest; 814 → 905): a rendered shaft at
      known angles, blurred, occluded and cluttered; the tracker against
      constructed evidence; the whole path through a real container
- [x] **10.7 Measure detection rate and per-frame cost; commit**

**Measured** (`scripts/benchmark_club.py`, Apple M1 Pro / macOS 26.4.1). A shaft
drawn at an angle that is an input, smeared by a declared exposure. **A floor,
not an estimate**: no defocus, no compression, a flat background, and a club 1.1
torso lengths long where a real one is about 2.5 — a longer lever smears
proportionally more.

The headline is the gap between two numbers a reader would assume are the same.
Coverage per swing phase at 120 fps, against the exposure as a fraction of the
frame interval:

| shutter | max smear | overall | address | backswing | **DOWNSWING** | follow | err p95 |
| ------- | --------- | ------- | ------- | --------- | ------------- | ------ | ------- |
| 0.03    | 0.8 px    | 100%    | 100%    | 99%       | **100%**      | 100%   | 0.63°   |
| 0.125   | 3.5 px    | 99%     | 100%    | 99%       | **100%**      | 98%    | 0.69°   |
| 0.25    | 6.9 px    | 93%     | 100%    | 98%       | **83%**       | 73%    | 0.79°   |
| 0.5     | 13.9 px   | 86%     | 100%    | 93%       | **65%**       | 62%    | 1.15°   |
| 1.0     | 27.7 px   | 64%     | 100%    | 58%       | **25%**       | 23%    | 1.18°   |

A tracker reporting "64% coverage" has tracked a quarter of the downswing. The
address column is 100% in every row and is a large share of the aggregate, so it
carries no information about the capture at all — which is why coverage is
reported per phase and `ClubTrackingReport.coverage` is documented as the number
**not** to read alone.

**Blur is a cliff, not a decline**, and that is what makes it a capture problem
with no processing fix:

| smear at the head | found    | support | angle error |
| ----------------- | -------- | ------- | ----------- |
| 0.0 px            | **100%** | 1.00    | 0.44°       |
| 4.8 px            | **100%** | 1.00    | 0.88°       |
| 9.6 px            | **100%** | 1.00    | 0.90°       |
| 14.4 px           | **0%**   | —       | —           |
| 28.8 px           | **0%**   | —       | —           |

An exposure draws a rotating club as a fan with no edge in it. Where the club is
found the angle is right to about half a degree, and where it is not, lowering a
threshold recovers nothing — so there is deliberately no sensitivity setting in
`ClubConfig` trading detection rate against accuracy. There is nothing to trade.

**The shutter is the lever; the frame rate is not.** Holding the exposure at
1/500 s rather than at a fraction of the interval:

| fps | 1/4 shutter: smear / overall / downswing | 1/500 s: smear / overall / downswing |
| --- | ---------------------------------------- | ------------------------------------ |
| 30  | 27.6 px / 62% / —                        | 6.6 px / 96% / —                     |
| 60  | 13.8 px / 87% / 62%                      | 6.6 px / 97% / **92%**               |
| 120 | 6.9 px / 92% / 81%                       | 6.7 px / 96% / **90%**               |
| 240 | 3.5 px / 100% / 99%                      | 6.7 px / 96% / **88%**               |

Four times the frames buy nothing once the shutter is fixed. The 30 fps rows have
no downswing column because no swing is detected there at all — Phase 3's window
floor reappearing, rather than anything about the club.

**What the temporal check is worth**, against picking the best-supported
candidate, which is what a Hough transform ranks by and what a per-frame detector
_is_:

| background lines | tracker: kept / wrong | evidence only: kept / wrong |
| ---------------- | --------------------- | --------------------------- |
| 0                | 289 / **0**           | 296 / **0**                 |
| 1                | 285 / **0**           | 294 / **4**                 |
| 2                | 285 / **0**           | 294 / **7**                 |
| 3                | 277 / **0**           | 297 / **15**                |
| 4                | 279 / **0**           | 298 / **17**                |

The tracker keeps about 20 fewer frames and gets all of them right. A vertical
line through the hands scores **the same support as the club and more length**,
so evidence alone ranks it first — and once the club smears it is the only
candidate left. What separates them is that a door frame does not rotate.

**The case it does not handle, and cannot see.** A rectangular occluder's
boundary is a long straight line; where it runs near the hands it is sharp,
stationary and, once the real club is hidden behind it, unopposed:

| occluder    | tracked | downswing | wrong | **conf. when wrong** | head seen |
| ----------- | ------- | --------- | ----- | -------------------- | --------- |
| nothing     | 93%     | 79%       | 0     | —                    | 97%       |
| upper third | 54%     | 44%       | 18    | **0.98**             | 8%        |
| upper half  | 47%     | 31%       | 4     | **0.81**             | 97%       |
| a wide band | 64%     | 38%       | 0     | —                    | 100%      |

Those eighteen frames score 0.98 against the 0.99 a correct frame carries. All
three factors are satisfied and the answer is the occluder's edge; continuity
cannot reject a stationary line the tracker is _already on_, because such a line
agrees perfectly with a prediction extrapolated from two frames already on it.
Recorded rather than fixed, the way Phase 7 recorded that Phase 4's event
confidence does not detect a badly located event. The two things that would catch
it — a rule that the shaft must rotate during the backswing, and a detector that
has seen a half-occluded club — are a golf norm and a learned model respectively,
and both need the labelled set Phase 12 builds.

**Impact from the club head, against Phase 4's hand-speed peak:**

| shutter | downswing coverage | club head | kinematic | delta    |
| ------- | ------------------ | --------- | --------- | -------- |
| 0.03    | 100%               | 203       | 204       | −1 frame |
| 0.125   | 100%               | 203       | 204       | −1 frame |
| 0.25    | 83%                | refused   | —         | —        |
| 0.5     | 69%                | refused   | —         | —        |
| 1.0     | 27%                | refused   | —         | —        |

Refused wherever the downswing was not more than half tracked, which is most of
the table. The two rows that answer agree with Phase 4 to a frame — and that is a
check of the **arithmetic**, not of the physics: this fixture puts the hands' peak
speed and the club head's lowest point at the same instant by construction. On
real footage the hands peak first, which is the whole reason two estimates are
worth having.

**Cost:** 6.2 ms/frame to detect over a search region of 3.2 torso lengths (the
whole frame on this fixture), and 4.0 ms to track a 312-frame clip once the
candidates exist. Against ~17 ms/frame to extract the poses that must come first.
Nothing is cached, for the same measured reason as Phase 3.

**On real footage, and this is the part worth reading.** The tour-pro clips are
the only recordings here a club can be looked for in, and they are 30 fps
social-media reposts of slow-motion captures — the worst case the tables above
describe. `rory_face_on.mp4` at a factor of 7:

| phase          | tracked | coverage | median confidence | club-head speed |
| -------------- | ------- | -------- | ----------------- | --------------- |
| address        | 27/161  | 17%      | 0.54              | 6,233 px/s      |
| backswing      | 56/142  | 39%      | 0.61              | 5,765 px/s      |
| downswing      | 18/78   | **23%**  | 0.69              | 8,744 px/s      |
| follow-through | 23/145  | 16%      | 0.68              | 8,690 px/s      |

21% clip-wide, in 29 separate runs, with the club head reached in 16% of tracked
frames. The clip reports up to **217 px** of smear at a 360-degree shutter, which
is twenty times the bound the blur sweep measured — so most of this clip could
not have been tracked by anything of this kind, and the system says so rather
than producing a line anyway.

**Checked by eye with `scripts/overlay_club.py`, the tracked frames are the
club.** At address the ray runs from the hands down the shaft to the ball; in the
downswing it lies along the shaft to the head. That is the confirmation the
overlay exists for, and it is not something any number in the report could have
supplied.

**And the failure the synthetic occlusion sweep predicted is there, in the real
clip, at the confidence it predicted.** Nine of the 139 tracked frames — 6.5% —
follow the vertical **edge of the yardage sign** behind the player rather than
the club, at a median confidence of **0.98**. The sign's edge is long, sharp,
stationary and passes through the hands at the finish, exactly where the club has
gone behind the player's head; there is no rival to bring the margin down and a
stationary line matches a prediction extrapolated from two frames already on it.
All nine fall _after_ the detected swing, so per-phase coverage happens not to
report them — which is luck rather than design, and is recorded as such.

The down-the-line clip inverts the usual ordering, which is worth knowing: 8%
clip-wide but **25% through the downswing**, its best-covered phase. Down the
line the club points towards the camera at address and is foreshortened past the
minimum length, so "detection is easiest where the club is slowest" turns out to
be a face-on observation rather than a general one.

**Deliberate deviations from the original plan:**

- **The detector is stateless and the tracker is offline, which inverts the seam
  Phase 2 established.** `PoseEstimator` documents that implementations are
  stateful and must be called in increasing timestamp order. A `ClubDetector`
  sees one frame, knows nothing about any other, and returns **every** candidate
  rather than a decision. That is what makes the seeding possible — the track
  starts where the evidence is best, which is address or the top, and grows into
  the downswing, where a forward-only tracker would arrive carrying whatever it
  had picked up on the way. It is Phase 4's nested searches one layer along. It
  also removes the failure that makes classical trackers untrustworthy: a
  detector told what the tracker believes narrows its search towards it, confirms
  it, and narrows further.
- **`margin` is measured on `support × continuity`, not on evidence alone**, and
  that was a decision rather than an implementation detail. The question it
  answers is "given everything known at this frame, was the choice clear". At a
  frame where a smeared club sits at 0.6 support against a door frame at 1.0, the
  choice _is_ clear, because the door frame scores nothing on continuity —
  scoring the margin on evidence alone reports that frame as ambiguous and
  refuses a **correct** detection at exactly the point in the swing where this
  phase is hardest.
- **Both ends of a line through the hands are offered as candidates.** A door
  frame and a club pointing the other way are the same pixels, so taking the
  farther endpoint silently picks one — and measured, it picked wrong for runs of
  several frames and reported it at a confidence of 1.00. Offering both puts them
  against each other: with no prediction the margin collapses and the frame is
  refused, and with a tracked neighbour the prediction separates them.
- **The transform is asked for _fragments_, not shafts**, and conflating the two
  cost the club entirely on the first pass. `ClubConfig.min_shaft_length_torso`
  is how far a shaft must reach **from the hands**; Hough's `minLineLength` is how
  long one unbroken run of collinear edge pixels must be, and an antialiased
  diagonal does not produce one long run. A clean 275-pixel shaft came back as
  four fragments of 100–200 px, every one of which a 200-pixel minimum discarded
  — reporting "no candidate" with the club in plain view under 779 edge pixels.
  The transform now gets a quarter of the bound and the geometry measures the
  thing the bound is about.
- **`support` is measured in a band around the ray, because a bar has no edge
  down its middle.** Canny finds the two _sides_ of a shaft and the ray from grip
  to tip runs between them: sampled on the ray alone a clean shaft scores 0.005,
  and 1.000 once the band covers its own thickness. The band scales with the
  torso, because the shaft's apparent thickness does.
- **A rate is never measured across a tracking gap.** Nothing says how many
  half-turns the shaft made while unobserved, which is the same reason
  `unwrap_deg` refuses to carry phase across a break. Taking the nearest tracked
  neighbour across a three-frame hole reported **8,050 °/s on a club turning at
  700**; adjacent frames only, and a frame beside a gap gets a one-sided rate.
- **A frame the temporal check rejected may not be re-seeded.** The first version
  of the growth loop re-seeded any unassigned frame, and a seed is accepted on
  evidence alone — so a shaft jumping 157° in one frame at 120 fps was refused as
  `DISCONTINUOUS` by the walk and then re-admitted by the next seed at a
  confidence of 1.0. Seeding is now only for frames with no tracked neighbour
  inside the prediction window, which is exactly the set that has not already
  been judged.
- **The club-head impact estimate is gated on downswing coverage**, not on
  anything local to itself. Without that gate a clip whose head blurred away
  through impact reports the lowest point of whatever survived — measured, **38
  frames late**, with both local checks passing: the minimum sat inside the
  observed set and its neighbours were adjacent frames. Only the coverage of the
  phase the minimum was supposed to fall in can see that.
- **No lens correction is applied, and the half-measure is what is ruled out.**
  Every other consumer of `filter_sequence` undistorts the landmarks when a
  calibration exists. Doing that here would be actively wrong: the detector
  searches the **raw** frame, so an undistorted anchor points at a place in the
  image where the hands are not. Correcting properly means undistorting the
  frame, and it matters for a second reason a landmark does not have — a straight
  club in the world is a _curved_ line in a distorted image, so the straight-line
  model the transform rests on is itself violated near the frame edge.
- **The per-frame observations cross the engine boundary**, which departs from
  `PoseExtractionResult` and `ReconstructionReport`. Two things make this the
  different case: a club track is _one_ object per frame rather than 33 landmarks
  in two spaces, so a swing's worth is kilobytes where a pose sequence is
  megabytes; and a detector is checked by **drawing** it, so a report that omitted
  the geometry would force every consumer to re-run the detection to see it.
- **`ClubTrackingReport` is not exported to TypeScript.** It is reachable over RPC
  and from `analyzer club`, and no UI renders it — the club overlay canvas is
  Phase 14.5, which is where it earns an export. Same reasoning that kept
  `SequenceFilterReport` out until Phase 4 built its panel and `Project` out until
  Phase 14.1.
- **No UI, and `scripts/overlay_club.py` is the deliverable for 10.5.** Phase 4
  built an inspector because checking an event's _timing_ needs the video frame by
  frame; a detector is checked by drawing its output on the frame it came from,
  which is what Phase 5 and Phase 9 concluded too. The overlay writes the refused
  frames as well as the tracked ones, because a contact sheet of only the
  successes would show a tracker that works.
- **`pixels_to_frame_widths` was added to `coordinates.py`**, and it is the first
  time that direction has been needed. Every layer up to here measured landmarks a
  model had already normalised, so pixels were something the system converted
  _to_, for drawing. Club tracking is the first thing that takes a measurement off
  the pixel grid itself.
- **A shaft angle is a direction and is unwrapped; a shoulder line is an
  orientation and is folded.** The two operations are opposites, they now both
  exist in this codebase, and a reader who confuses them gets a plausible number
  either way — so each says in its own docstring which kind of object it is for.
  A swing carries the shaft through more than a full turn and across ±180 exactly
  once, in the downswing.
- **No club-head speed metric, and no shaft-plane metric.** Phase 6.2 deferred
  "shaft orientation and club path" here, and what this phase can honestly supply
  is a shaft **direction**, well determined, in the frames where the club was
  found. A club-head speed needs a length the image frequently does not contain,
  and a club path needs the head observed through the downswing, which is exactly
  where it is not. The biomechanics registry is unchanged; promoting a shaft angle
  into a `Metric` needs a decision about what a metric measured in 44% of a
  downswing means, and that belongs with the coverage gate rather than beside it.
- **Club tracking has never been run on real footage.** No clip in this
  repository has been club-tracked and checked by eye, because the reference
  clips are 24–30 fps with the hands already lost to motion blur through the
  downswing — Phase 4 refuses one of them outright for that reason. One command
  finishes this once a fast-shutter clip exists:
  `uv run --project python python scripts/overlay_club.py <clip> --video`

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
