# Roadmap

Tracking checklist for the build. One phase at a time; at each boundary — run
tests, run the app, verify, document, record measurements, commit. Do not
advance past a broken phase.

**Progress: Phases 0-16 complete (17 / 21).**

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
| 11  | Ball detection        | ✅ **Done** | Impact-frame agreement measured                                |
| 12  | Temporal ML           | ✅ **Done** | Leak-free splits; **no metrics — no labelled set exists**      |
| 13  | Coaching engine       | ✅ **Done** | Every finding cites computed evidence; 9 of 12 rules refuse    |
| 14  | Desktop UI            | ✅ **Done** | Full workflow end-to-end; seek verified in a real browser      |
| 15  | 3D visualisation      | ✅ **Done** | Scrub stays in sync; the viewpoint reports what it hides       |
| 16  | Swing comparison      | ✅ **Done** | Differences shown, no score; the camera is gated first         |
| 17  | Performance           | ⬜ Next     | Before/after numbers recorded                                  |
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

## Phase 11 — Ball detection ✅

`python/analyzer/ball/{detector,blob,track,extract}.py` · `python/analyzer/impact.py` ·
[ADR-0015](decisions/ADR-0015-ball-departure-and-impact-precedence.md)

The first thing this engine measures by **watching something stop existing**.
Every layer below reads a presence — where a landmark is, where a shaft points,
how deep a joint sits — and the reading is taken off a thing that is in the
picture. A golf ball sits in plain view doing nothing for hundreds of frames and
then is not there, and the instant this phase exists to find is the boundary
between those two states.

Tracking the ball in flight is the obvious ask and is unanswerable on consumer
footage: it leaves at ~70 m/s, which is a metre-long smear at 1/60 s and out of
frame before it has been drawn sharply once. Phase 10 met the same wall with the
club head. So this measures the ball where the ball is **easy** — at rest — and
reads impact off the edge of that interval.

- [x] **11.1 `BallDetector` Protocol + confidence** — one Protocol, one classical
      implementation (ROI → top-hat → components → geometry), and **two** separate
      confidences computed from disjoint evidence: `BallConfidence` scores one
      frame's observation, `DepartureConfidence` scores the instant. The
      identification is circular — the ball is picked out partly by the fact that
      it leaves — and the partition is what closes the circle rather than hiding it
- [x] **11.2 Impact corroboration fused with the Phase 4 estimate** —
      `analyzer/impact.py`, reporting **one** instant with one provenance and one
      error bar. Four estimates now exist and they are **ranked, not averaged**:
      three are biased in a known direction and one is not, and averaging moves the
      answer away from the truth while producing a provenance nobody can name.
      Every other estimate is kept beside it with its delta, because those deltas
      are the measurement. **Nothing reaches back down into Phase 4**
- [x] **11.3 Measure agreement between ball-based and kinematic impact frames** —
      `scripts/benchmark_ball.py`, and on the reference footage
- [x] **11.4 Commit**

**The uncertainty is one frame interval, and it is a bracket.** Nothing here
estimates a peak, fits a curve or smooths a signal, so nothing here has a
resolution that degrades. The ball was present at one frame and absent at the
next, so impact is **inside**:

| fps | located | error | bracket | Phase 4's window | ratio |
| --- | ------- | ----- | ------- | ---------------- | ----- |
| 30  | exact   | 0     | 33.3 ms | 100 ms           | 3x    |
| 60  | exact   | 0     | 16.7 ms | 100 ms           | 6x    |
| 120 | exact   | 0     | 8.3 ms  | 100 ms           | 12x   |
| 240 | exact   | 0     | 4.2 ms  | 100 ms           | 24x   |

Phase 4's number is the width of the smoothing window its frame rate forced,
which is a **scale the peak could have moved on**, not an interval it is inside.
The two are not comparable and `FusedImpact.uncertainty_is_bracket` says which is
which.

**The shutter, which decided Phase 10, decides nothing here** — a ball at rest is
not moving, so no exposure smears it. Located exactly at every shutter from 0.03
to a full 360°. What replaces it as the capture variable is **contrast against
the surface**, and it is a cliff in the same way:

| ball vs turf | located | coverage |
| ------------ | ------- | -------- |
| 155 levels   | exact   | 100%     |
| 105          | exact   | 100%     |
| 75           | refused | 22%      |
| 35           | refused | 0%       |

**What refuses, and why each is right:**

| case                          | outcome                  | what says so              |
| ----------------------------- | ------------------------ | ------------------------- |
| practice swing, ball stays    | no instant               | nothing departed          |
| ball rolls off before the top | no instant               | outside the strike window |
| clip ends 2 frames after      | instant, confidence 0.06 | permanence 0.06           |
| a rival that also departs     | instant, margin **0.00** | warned                    |
| ball covered over 10 frames   | 5 frames early           | abruptness 0.70, warned   |

The occlusion row is the phase's **documented failure**, not a bug: at impact the
club head is at the ball, so a covered ball and a departed one are the same
picture. It only matters when the covering starts _before_ contact, and then the
instant runs early by about half the covering — reported rather than fixed.

### The first observed impact, and the number it settles

`data/face-on/rory_face_on.mp4` at a factor of 7. The ball is bracketed between
frames 360 and 361 — which is exactly what the Post-Phase-6 section recorded by
eye, now measured, with no warnings and every factor clean:

| source           | frame | delta                   | uncertainty    |
| ---------------- | ----- | ----------------------- | -------------- |
| `ball_departure` | 361   | **reported**            | 5 ms (bracket) |
| `hand_low`       | 362   | +5 ms / +1 frame        | 100 ms (scale) |
| `hand_speed`     | 381   | **+95 ms / +20 frames** | 100 ms (scale) |

**This settles the question Phase 6 left open.** The roadmap asked whether to
swap Phase 4's primary estimate for the lowest point of the hand arc. The arc low
lands within **one frame** of the observation; peak hand speed lands twenty
frames away — and **late**, which is the opposite of the direction the physics
predicts and which `contracts/phases.py` has warned about since Phase 4. So the
fusion's precedence puts `hand_low` above `hand_speed`, and Phase 4's own
estimate is left exactly as it was: an engine in which a number changes depending
on which other analyses happened to run cannot be compared across clips.

One clip is not a correction. It is one clip, and the labelled set is Phase 12.

**Cost:** 16 ms/frame to detect over 1.5 torso lengths of ground, and 34 ms to
track a 312-frame clip once the candidates exist. Against ~17 ms/frame to extract
the poses that must come first. A **rectangular** structuring element is what
makes that affordable: OpenCV separates that one and nothing else, and an
elliptical one of the same size costs 66 ms a frame on its own.

### What real footage found, and the fixture did not ✅

Six defects, every one of which produced a confident wrong answer rather than an
error, and five of which the synthetic fixture passed cleanly.

- **The search region was anchored at the hands, and contained the sky.** "Within
  a club length of the hands" is true and is a much weaker statement than "on the
  ground": it is a disc of ~2.5 torso lengths centred chest-high, which on a
  vertical phone clip is the whole picture. The identification looks for a small
  round thing that sits still for hundreds of frames and then stops being there,
  and a patch of cloud between two branches is rounder, better resolved and
  stiller than a ball 4 px across. The clip established on a point 126 px from the
  top of the frame and reported impact **117 frames late**. Anchored at the
  **ankle midpoint** — the pose layer knows where the ground is — the same clip
  finds the ball.
- **The drift reference was fixed, and the camera was not.** "A teed ball does not
  move" is a statement about the world; this layer works in the image, where a
  stationary ball moves whenever the camera does — handheld, stabilised, or a
  repost with a slow pan. The ball's offset grew monotonically from 11 px to 14 px
  through the swing while its contrast stayed healthy at 0.52–0.62, so `stillness`
  decayed under the bound and the run ended **58 frames early**, with contrast
  good, margin 1.00 and a plausible confidence. The reference now **follows**, which
  is Phase 10's continuity argument in a different medium, and
  `BallQuality.drift_torso` reports the camera motion instead of absorbing it.
- **`min_circularity` was tuned on a fixture and rejected the only real ball in
  the project.** A rendered ball scores 0.89 and everything else falls below 0.4,
  which invites a bound at 0.65 and the belief that shape identifies a golf ball.
  The real ball — 4.4 px in radius, compressed, with a ragged boundary — scores
  **0.59**, while grass and compression blocks in the same size band score 0.11 to
  0.71. At that size shape does not separate them at all. The bound is now a
  sanity filter and the identification rests on persistence.
- **The candidate cap sorted by area, which discards a golf ball.** A ball is one
  of the _smallest_ things in the response; the real ball sat outside the largest
  32 regions of its own frame. The size band now runs **before** the cap, and what
  survives is ranked by closeness to a ball's expected size.
- **A frame-widths distance was compared against a torso-lengths bound** — loose by
  a factor of about four, which merged a teed ball with a tee marker 70 px away
  into one object that then never departed. Found by the fixture's distractor case;
  the ordinary clean clip passed throughout.
- **Permanence was measured on the tracker's verdicts rather than on the
  detector's candidates.** A dimming ball falls under the confidence bound and ends
  the run early, so every frame after it is one the tracker rejected — and an
  acceptance-based score reads 1.0 while the ball is plainly still in the picture.
  Reading candidates notices; reading verdicts does not.

**Open, and not resolved here:**

- The club-head estimate never answered on any reference clip. Phase 10 already
  recorded why: they are 24–30 fps with the hands lost to motion blur through the
  downswing, and a club-head impact is refused whenever the downswing is less than
  half tracked. The fusion has been exercised with three of its four sources on
  real footage and all four only on the fixture.
- **One clip has an observed impact.** Every statement above about the bias of the
  kinematic estimate rests on it, and a bias measured once is an anecdote with a
  number attached.
- The synthetic fixture **cannot** measure the agreement this phase exists to
  measure: one arc drives the hands, the club and the ball, so all four estimates
  coincide by construction. It can show the fusion is wired correctly and nothing
  more, and `scripts/benchmark_ball.py --sweep agreement` prints that caveat above
  its own table.

## Phase 12 — Temporal ML ✅

`python/analyzer/ml/{labels,labeltool,features,dataset,splits,tcn,train,evaluate,compare,registry,provider}.py` ·
`python/analyzer/contracts/{labels,ml}.py` ·
[ADR-0016](decisions/ADR-0016-labels-groups-and-the-noise-floor.md)

**The phase's result is a refusal, and the machinery that earns the right to make
it.** Everything the checklist asked for is built, tested and exercised end to
end. It has never been run on a golf swing that a person labelled, because no
such clip exists in this project, so **not one accuracy number here is a
statement about golf** — and the engine says so itself rather than leaving it to
a reader: `EvaluationReport.claims_permitted` is computed, and on every set this
phase can currently build it is false.

- [x] **12.1 Labelling tool + label schema** — `labeltool.py` is a pure state
      machine, a renderer and a thin OpenCV window; only the window lacks a test.
      The schema requires three things a "frame number per event" format loses:
      **who** marked it, **how sure they were** (`uncertainty_frames`, required,
      no default), and **whose swing it is** (`player_id` and `session_id`,
      required, no default). Built before anything consumed a label
- [x] **12.2 Dataset builder with feature versioning** — `FeatureSpec` hashes
      itself, so a checkpoint records the digest of the definition it was trained
      on and `registry.load_model` refuses a mismatch. Every channel is divided by
      the subject's **torso**, never by a statistic of the clip
- [x] **12.3 Group-aware train/val/test splits** — grouped by **player**.
      `LeakageCheck` is recomputed from the assignment that came out, so it can
      catch a splitter it was not written alongside — and it does: `random_clip_split`
      is kept as a measuring instrument and fails the check by construction
- [x] **12.4 TCN baseline, seeded, checkpointed** — 19,525 parameters, six dilated
      layers, a 127-sample receptive field which the report states in **seconds**
      (2.12 s) because samples alone say nothing about whether it can see a swing
- [x] **12.5 Metrics, confusion matrix, model registry** — per-event localisation
      in frames _and_ real milliseconds, per-class F1, a 5x5 confusion matrix, and
      a noise floor below which no claim is permitted
- [x] **12.6 Compare against the rule-based detector on the same held-out set** —
      same clips, same metric, same filtered trajectories, and a verdict word that
      is "indistinguishable" whenever the gap is inside what the labels resolve
- [x] **12.7 Commit** — with no metrics, because no real labelled set exists

### The refusal, which is the deliverable

```
$ analyzer labels
No labels yet. Phase 12's machinery is built and has nothing to run on.

$ uv run --project python python scripts/benchmark_ml.py --sweep refusal
4 clips, 1 player
refused: True

A train/validation/test split needs at least 3 players and this set has 1.
Splitting by clip instead would put the same golfer on both sides, and every
number measured after that would be about how well the model recognises a
person it has already been trained on. More clips of the same player do not
help; more players do.
```

That last sentence is the phase's finding. The four reference clips in `data/`
are one golfer. Labelling all four would not produce a split; nor would forty of
them. The input this phase is short of is **people**, not footage, and no amount
of engineering substitutes.

### The noise floor, and the ceiling it puts on the whole approach

An error smaller than the labels' own uncertainty is not an achievement; it is a
measurement below the resolution of the instrument. The floor has two parts and
they add:

| part          | what it is                                   | value   |
| ------------- | -------------------------------------------- | ------- |
| label bracket | median of what labellers said they could see | per set |
| resampling    | half a sample of the 60 Hz feature grid      | 8.3 ms  |

`scripts/benchmark_ml.py --sweep rate` measures the second half directly, by
putting a label's frame onto the grid and reading it back:

| clip fps | grid | floor  | round trip | worst  |
| -------- | ---- | ------ | ---------- | ------ |
| 30       | 60   | 8.3 ms | 0.0 ms     | 0.0 ms |
| 60       | 60   | 8.3 ms | 0.0 ms     | 0.0 ms |
| 120      | 60   | 8.3 ms | 4.6 ms     | 8.3 ms |
| 240      | 60   | 8.3 ms | 4.0 ms     | 8.3 ms |

**At 240 fps the grid costs more than watching the ball leave.** Phase 11 brackets
impact to 4.2 ms at that frame rate by observing an absence; the resampling here
spends 8.3 ms before a model has run a single convolution. A learned detector is
not a replacement for an observation, and on the one quantity both can report,
the observation wins before the comparison starts.

### What the synthetic corpus could and could not measure

`tests/synthetic_labels.py` generates players, sessions and swings whose events
are inputs to the generator. It can show the plumbing is right. It was also meant
to answer "what is a leaky split worth", and it **cannot**, which is the same
shape of caveat Phase 11 had to write about measuring impact agreement on a
fixture where one arc drove every estimate.

`--sweep leakage --seeds 5`, 72 clips, 12 players, identical corpus and seeds on
both sides:

| quantity     | by player     | by clip (leaky) | difference | larger than the scatter? |
| ------------ | ------------- | --------------- | ---------- | ------------------------ |
| macro F1     | 0.972 ±0.007  | 0.973 ±0.013    | +0.001     | no                       |
| takeaway MAE | 25.5 ±15.0 ms | 25.6 ±9.0 ms    | −0.1 ms    | no                       |
| top MAE      | 14.3 ±7.1 ms  | 12.2 ±6.9 ms    | +2.1 ms    | no                       |
| impact MAE   | 4.5 ±3.3 ms   | 6.6 ±6.2 ms     | −2.1 ms    | no                       |
| finish MAE   | 14.0 ±8.8 ms  | 12.8 ±11.5 ms   | +1.2 ms    | no                       |

Nothing is readable, and the reason is a property of the fixture: its golfers
differ by six generator parameters, so a model that has seen seven of them has
seen the space and holding one out asks nothing. Real golfers differ in ways a
generator does not know how to vary. **The split is grouped by player anyway** —
the argument for it was never this measurement, it is that a swing and its
near-duplicate cannot sit on opposite sides of a question.

What the corpus _can_ show is that the held-out number has not settled
(`--sweep players --seeds 3`):

| players | clips | macro F1 | top MAE | impact MAE |
| ------- | ----- | -------- | ------- | ---------- |
| 3       | 18    | 0.869    | 35.0 ms | 130.0 ms   |
| 4       | 24    | 0.942    | 16.1 ms | 7.2 ms     |
| 6       | 36    | 0.939    | 21.7 ms | 8.3 ms     |
| 9       | 54    | 0.957    | 15.6 ms | 5.3 ms     |
| 12      | 72    | 0.974    | 11.4 ms | 3.9 ms     |

Still improving at twelve. A corpus whose score still moves with its own size has
not yet measured a method; it is measuring its sample.

### Why the comparison is rigged, and the part of the rigging that is real

On the synthetic held-out set the model looks far better than the rules at two of
the four events:

| event    | rules    | model   | verdict           |
| -------- | -------- | ------- | ----------------- |
| takeaway | 174.2 ms | 10.8 ms | model             |
| top      | 8.3 ms   | 8.3 ms  | indistinguishable |
| impact   | 12.5 ms  | 5.8 ms  | indistinguishable |
| finish   | 168.3 ms | 11.7 ms | model             |

**The model is not more accurate at the takeaway. It has been told which
definition is being marked.** The generator's takeaway is the instant its easing
function leaves zero; Phase 4's is where hand speed crosses 5% of its peak, which
on a sin-squared ramp is several frames later. The model is trained on the labels,
so it learns the labeller's convention; the rules brought their own.

A human-labelled set narrows this and does not close it — a person marks the
takeaway where they can _see_ the club move, which is a third convention again.
Any comparison between a learned detector and a rule is partly a measurement of
whose definition the labels encode, and that is a permanent property of the
exercise rather than a flaw in this corpus.

### The other things measured

The receptive field, on the same corpus (`--sweep receptive --seeds 2`):

| layers | field | seconds | params | macro F1 | top MAE |
| ------ | ----- | ------- | ------ | -------- | ------- |
| 3      | 15    | 0.25    | 10,021 | 0.959    | 5.4 ms  |
| 4      | 31    | 0.52    | 13,189 | 0.964    | 8.8 ms  |
| 5      | 63    | 1.05    | 16,357 | 0.963    | 15.4 ms |
| 6      | 127   | 2.12    | 19,525 | 0.973    | 9.6 ms  |
| 7      | 255   | 4.25    | 22,693 | 0.968    | 13.8 ms |

The expectation was that a field shorter than a swing would lose the top, which
is defined by what happens on both sides of it. The table does not show that, and
the reason is again the fixture: these hands follow one analytic arc, so a quarter
of a second of speed profile already says where in the swing it is. The shipped
default is six layers, chosen on the argument and **not** on this table, which
cannot distinguish any row from any other.

Cost, 72 clips on the reference machine (`--sweep cost`):

| stage                  | total  | per clip |
| ---------------------- | ------ | -------- |
| generate the corpus    | 1.64 s | 22.8 ms  |
| filter + featurise     | 2.68 s | 37.2 ms  |
| train, 40 epochs       | 5.93 s | 82.3 ms  |
| model inference        | 0.01 s | 0.7 ms   |
| read the rule detector | 0.00 s | 0.0 ms   |

Training the whole corpus costs less than extracting poses from one clip. That is
the correct shape for this phase: the expensive thing was never the model.

### What the build found on the way

- **A 30 fps clip cannot be filtered with the default window, so a dataset builder
  meeting Phase 3's seam loses most real footage.** The 0.10 s window holds three
  samples at 30 fps and the degree-4 fit needs five, so every landmark comes back
  empty and the clip is dropped for having no torso. `filter_config_for` widens
  the window to the narrowest one the rate supports. This is not free — a wider
  window flattens the velocity peak — so the grid equalises the **sample rate and
  not the smoothing**, and `DatasetSummary` reports when a set contains both.
- **Phase 4's tiling had to be reproduced exactly, and a test pins it.** Address
  runs from frame zero; the follow-through ends _on_ the finish; everything after
  is outside every phase, which is why the classifier has five classes and not
  four. A tiling that drifted would turn every comparison in 12.6 into a
  comparison of two conventions.
- **The features had to reverse one of Phase 4's own decisions, for a reason that
  does not contradict it.** `phases/signals.py` refuses hip-local coordinates
  because its rule is about the highest point the hands reach and a moving origin
  makes "highest" mean something else. A learned detector needs the opposite:
  where the hands are relative to the body, with the camera removed — otherwise a
  handheld clip drifting upward through the downswing presents rising hands the
  model cannot distinguish from real ones, and Phase 11 found exactly that drift
  on real footage. Each layer states which quantity it wants.

**Open, and not resolved here:**

- No real labelled set. Everything above rests on generated swings.
- The labelling window itself has never been driven by a person in this
  repository. Its state machine, its overlay and its key mapping are tested; the
  OpenCV loop around them is forty lines with no decisions in it and no test.
- `random_clip_split` measures nothing on the corpus available, so the cost of a
  leaky split remains an argument rather than a number.

## Phase 13 — Coaching engine ✅

`python/analyzer/coaching/{registry,engine,bracket,evidence,guard,phrasing,local}.py` ·
`python/analyzer/contracts/coaching.py` ·
[ADR-0017](decisions/ADR-0017-borrowed-thresholds-and-the-guard.md)

**Twelve rules, and across four reference clips two of them ever fire.** Five can
reach a comparison at all; seven are refused before a clip is looked at. That is
not a gap in the engine: it is what happens when every borrowed threshold is made
to declare how it was measured, and most of golf coaching's numbers turn out to
have been measured in three dimensions, or not measured at all.

- [x] **13.1 `Finding` contract + rule engine** — a finding carries the
      comparison, the margin, the bracket it had to clear, the source with its
      population and method, and the evidence. **No severity and no score**, and
      a test asserts the absence of those field names
- [x] **13.2 Rule set with sourced, documented thresholds** —
      `ThresholdSource.permitted_bases` decides what a number may be compared
      against, `CONVENTION` permits nothing, and every band states the arithmetic
      that produced it from the source's own published figures
- [x] **13.3 Evidence linking** — metric → frames → real-clock instants, all or
      none so the two lists are never misaligned. Frames index the video file;
      timestamps do not, on a slow-motion clip
- [x] **13.4 Optional local LLM phrasing layer** — off by default, loopback
      enforced in code, handed the findings as JSON and never a frame, a landmark
      or a path
- [x] **13.5 Guard rejecting invented numbers** — numbers, units, quantities
      nothing here measures, ball-flight outcomes and causal claims. Applied to
      this engine's own sentences too
- [x] **13.6 Tests** — 164 added (all pytest)
- [x] **13.7 Commit** — with the measurements below

### The registry, before any clip is involved

`scripts/benchmark_coaching.py --sweep inventory`:

| verdict                 | rules | what would change it    |
| ----------------------- | ----- | ----------------------- |
| usable                  | 5     | —                       |
| no measurement protocol | 4     | somebody publishing one |
| no published number     | 3     | somebody publishing one |

Six of the twelve cite a source with no published measurement protocol; two of
those publish no number either. **No recording changes either count.**

The pair that names the phase sits either side of one gap: `rotation.x_factor_top`
has a number (forty-five degrees, from a 1992 magazine article) and no protocol;
`rotation.x_factor_top_3d` has a protocol (three-dimensional capture) and no
number this project has read a figure it can cite.

### What four real clips produced

`scripts/benchmark_coaching.py --sweep clips`:

| clip                     | view          | ms/frame | found | refused | fired                        |
| ------------------------ | ------------- | -------- | ----- | ------- | ---------------------------- |
| amateur, face-on, 30 fps | face_on       | 33.3     | **0** | 12      | —                            |
| amateur, DTL, 30 fps     | down_the_line | 33.3     | 2     | 10      | tempo.ratio, tempo.downswing |
| tour, face-on, 7x slow   | face_on       | 4.8      | 1     | 11      | tempo.ratio                  |
| tour, DTL, 7x slow       | down_the_line | 4.8      | 1     | 11      | tempo.ratio                  |

Every finding on every clip is temporal. Nothing measured in the image plane
produced a comparison this engine would stand behind, from either camera
position, on either golfer.

### The measurement that decides whether the survivors ever fire

A tempo ratio's bracket is what one frame of ambiguity at the top moves it by,
and the top is shared: it lengthens the backswing and shortens the downswing at
once. Held at the amateur clip's own swing — 0.800 s over 0.233 s, tempo 3.43 —
and varying only the clock (`--sweep resolution`):

| fps | bracket | to the nearer band edge | outcome                 |
| --- | ------- | ----------------------- | ----------------------- |
| 30  | 0.738   | 0.371                   | cannot resolve the band |
| 60  | 0.341   | 0.371                   | resolved                |
| 120 | 0.164   | 0.371                   | resolved                |
| 240 | 0.081   | 0.371                   | resolved                |

**At 30 fps the measurement cannot resolve the band**; the crossover for this
swing is 56 fps. The published band is 1.37 wide and one frame at the top is
worth 0.74 of it. The tempo figure every golf app shows is not resolvable by the
camera that most of them are pointed at.

Tour Tempo itself publishes frame counts — 18/6, 21/7, 24/8 — and not a ratio,
which is the more careful of the two. 3:1 exactly is a ratio of two integers.

### What the engine tells a tour professional

On `rory_face_on.mp4` at a factor of 7, the one rule that fires reports a tempo
of **1.83:1** against a tour band of 2.43–3.80, with a bracket of 0.04. The
numbers are right and the conclusion is not about the swing: the finding cites
the backswing, the downswing and the frames each came from, and following those
frames leads to the takeaway — which the Phase 6 amendment had already flagged as
suspect on slowed footage.

That is the argument for citing evidence, and it is not transparency in the
abstract. **A finding that carries its frames is a finding somebody can discover
is wrong. A sentence of advice is not.**

### The guard

`--sweep guard`, over fourteen plausible inventions and three faithful
rewordings:

| check                            | caught |
| -------------------------------- | ------ |
| number not in evidence           | 4      |
| unit this system cannot measure  | 3      |
| quantity nothing here measures   | 3      |
| claim about where the ball went  | 2      |
| causal claim nothing here tested | 2      |

14 of 14 rejected, 3 of 3 faithful rewordings kept, 34 µs per candidate.

The four number cases include two with no digits in them — "ninety degrees" and
"forty-five degrees" — because the invention this engine most needs to catch is
the one golf coaching quotes in words.

**The guard's first catch was this engine's own sentence.** The same-clip
template read "more than the X degrees the two measurements leave open"; the
guard read the spelled-out "two" as a quantity absent from the evidence and
rejected it. The template was reworded. Teaching the guard to ignore small
counting words would have widened the hole it exists to close.

### Cost

`coach()` over twelve rules: **0.05 ms**. Against seconds per clip for pose
extraction, which is the same conclusion Phases 3 and 5 reached about the layers
below: nothing here is worth caching.

### What the build found on the way

- **A supplied slow-motion factor disqualifies every comparison in seconds, and
  no comparison that is a ratio.** Nothing in a conformed file records the
  factor, so a duration measured from such a clip is a measured number multiplied
  by a guess. A ratio of two durations from the same clip divides it back out —
  the same argument that puts lengths in torso lengths, one dimension over. The
  gate is `SUPPLIED_TIMEBASE`, and it exists because the first run of the engine
  reported the tour clip's backswing against a band in seconds without blinking.
- **The gate order matters, and `BASIS_NOT_PERMITTED` has to sit above
  `NO_METRIC`.** A reader told "this clip could not measure your shoulder turn"
  goes and re-films; the re-filmed clip is then refused for the basis instead.
- **The evidence's frame indices had to be kept out of the number allowance.** A
  metric measured over the address phase cites 161 consecutive frames. Folding
  those into the set of quotable numbers would licence every small integer in the
  language, and "ninety degrees of shoulder turn" would be accepted on any clip
  with a frame 90.

**Open, and not resolved here:**

- The phrasing layer has never been run against a language model in this
  repository, because none is installed on the machine it was built on. Every
  path is tested against fakes.
- The guard's spelled-out-number table is finite and starts at two. "One" is
  excluded because in English it is usually a pronoun, so a model writing "one
  degree of tilt" gets through.
- The guard cannot tell whether a sentence is about the _right_ finding. A model
  given two findings could describe the first with the second's numbers and pass
  every check. This is why `observation` is never replaced and `phrased` sits
  beside it.
- `posture.spine_tilt_impact` and every other same-clip rule on a projected angle
  refuses as `NO_UNCERTAINTY`, because Phase 6 quantifies uncertainty for the
  foreshortened rotations and nothing else. Those quantities are measurable; what
  is missing is a measurement of how well.

## Phase 14 — Desktop UI ✅

`apps/desktop/src/features/{player,analysis,projects}/` ·
`python/analyzer/{overlay.py,contracts/overlay.py}` ·
[ADR-0018](decisions/ADR-0018-seeking-by-measured-time.md)

**A video element cannot be asked for a frame.** It is asked for a time, and its
decoder shows whichever frame is being displayed then — while every panel this
project has built since Phase 4 reports frame indices. That gap is the whole
phase: the conversion between them is a measurement, it is made in the engine,
and the result is **checked against what the browser reports painting** rather
than assumed.

- [x] **14.1 Project management + import flow** — sessions in SQLite, clips
      attached by content with a **declared** camera role; the picker runs in
      Rust, which is what makes the asset protocol's scope mean anything
- [x] **14.2 Dual video player with frame-accurate seek** — `SeekIndex` carries
      every frame's presentation time and the midpoint to seek to; the player
      reports the frame it actually landed on
- [x] **14.3 Phase timeline** — lifted out of `PhasesPanel` into one component,
      so the timeline under the video and the one in the inspector cannot form
      two opinions about where the backswing is
- [x] **14.4 Transport** — play/pause, 0.25/0.5/1x, frame step, event jumps.
      Stepping is relative to the frame **on screen**, not the one last asked for
- [x] **14.5 Pose + club overlay canvas** — the _filtered_ landmarks, converted
      back to drawing coordinates by the engine; three states, and a blocked
      landmark is drawn as absent
- [x] **14.6 Metrics panel** — basis, decomposed confidence, measured
      uncertainty, methodology, and the refusals in their own list
- [x] **14.7 Findings panel linked to evidence frames** — every citation is a
      jump target; an empty result is presented as a result
- [x] **14.8 Playwright flows** — 18 added, including the seek verification
      against a real decoder
- [x] **14.9 Commit** — with the measurements below

### What the obvious implementation would have cost

`scripts/benchmark_seek.py`, over the clips this repository contains. Each cell
is frames landed on the wrong one, against the timestamps the container carries:

| clip                | frames | vfr | declared fps | measured fps | `frame / declared fps` | measured midpoints |
| ------------------- | ------ | --- | ------------ | ------------ | ---------------------- | ------------------ |
| cfr_30fps.mp4       | 60     | no  | 30.000       | 30.000       | 0 / 60                 | **0 / 60**         |
| vfr_30_to_15fps.mp4 | 45     | yes | 23.684       | 22.759       | 39 / 45 (±6)           | **0 / 45**         |
| PW_face-on.mp4      | 68     | no  | 27.470       | 30.000       | 56 / 68 (±5)           | **0 / 68**         |
| iron_dtl.mp4        | 96     | yes | 30.063       | 30.063       | 0 / 96                 | **0 / 96**         |
| rory_face_on.mp4    | 652    | no  | 30.006       | 30.000       | 651 / 652 (±1)         | **0 / 652**        |

**Three different failures, and only one of them is variable frame rate.**

`PW_face-on.mp4` is constant-rate, evenly spaced at exactly 30 fps, and its
container **declares 27.470** — an 8.4% error that drifts six frames by the end
of a 2.3 s clip. This is one of the two clips every phase since Phase 4 has been
measured on.

`rory_face_on.mp4` has a declared rate correct to four decimal places and still
misses on 651 of its 652 frames, every one of them by exactly one. `i / 30.006`
lands a few hundred nanoseconds _below_ frame `i`'s presentation time, and a
frame boundary has another frame on the other side of it. That is the argument
for midpoints, and it survives getting the rate exactly right.

### The measurement that is not circular

The table above compares two maps against the timestamps the container carries,
which is a statement about arithmetic. `benchmark_seek.py` also asks OpenCV to
perform the seeks, and **that column measures OpenCV**:
`cv2.VideoCapture.set(CAP_PROP_POS_MSEC)` lands a frame early on a sizeable
minority of seeks once the decoder has been read from — the same inaccuracy
Phase 1 found and works around with `_scan_to`. On three of the five clips it
cannot tell the two maps apart.

So the evidence is a real player: `e2e/player.spec.ts` serves the real fixture
bytes, Chromium decodes them, and `requestVideoFrameCallback` reports what was
painted. Every measured target lands on its own frame; the naive map is measured
alongside it in the same browser, on the same clip, in the same run, and does not.

### What the browser test found that the design did not predict

- **Chromium reports media time in whole microseconds.** A frame at
  0.13333333… seconds comes back as `0.133333`, a third of a microsecond _below_
  the container's timestamp. Looked up strictly, it falls into the previous
  frame's interval — so the player would have reported a spurious one-frame miss
  on about half the frames of every 30 fps clip. `frameAt` allows one
  microsecond, which is the resolution of the number being looked up and four
  thousand times smaller than one frame at 240 fps.
- **A media element will not seek in a resource that does not advertise byte
  ranges.** Without `Accept-Ranges` Chromium reports `seekable` as empty and
  silently clamps every `currentTime` back to zero — while still loading the
  clip, reporting its duration and buffering it end to end. It looks exactly
  like a broken player rather than a missing header.
- **The last frame of a clip can be unreachable.** Its target lies past the end
  of the media, because how long it is displayed is not recorded; the browser
  clamps to the declared duration, and on `vfr_30_to_15fps.mp4` that duration
  _equals the last frame's own presentation time_. No map can reach it. The
  player reports it as a residual rather than hiding it.

### The permission this cost

Phases 0 and 1 both recorded that the WebView never reads a file. A `<video>`
element **is** the WebView reading a file, and no frame-accurate player avoids
it. So the line is crossed as narrowly as the platform allows: the asset
protocol's scope ships **empty**, a path enters it one file at a time and only
through `commands::choose_clip`, and **there is deliberately no command that
takes a path and grants access to it**. That last point is what carries the
guarantee — the frontend cannot name a file it wants read — so `choose_clip`
opens its own dialog in Rust, because a grant has to be tied to something the
caller could not have fabricated.

`dialog:allow-open` stays, and that was reconsidered rather than assumed. It
serves the pickers that choose footage **the engine** reads and the WebView does
not, and none of those paths can reach the asset scope because no command would
put one there. `tauri-plugin-fs` is still absent and the WebView still cannot
spawn a process.

### Deliberate deviations from the original plan

- **"Dual video player" ships as one player.** 14.2 asks for two, and the
  two-camera screen it implies needs a second clip _aligned to the first_ — which
  is a `SyncModel`, which a project has only after `analyzer project sync` has
  run on a genuine simultaneous pair. This repository contains no such pair:
  Phase 7 measured its only candidate at a 95.8 ms residual against a 2.4 ms
  floor and concluded the two clips are **not the same swing**. Building a
  side-by-side player here would have meant shipping a screen whose correctness
  could not be demonstrated on any footage that exists. The frame-accurate
  single player is the part that is demonstrable, and the second one is a
  component beside it rather than a rebuild — `SeekIndex` is per clip and the
  time map that would relate two of them is already a contract.
- **A router still has not earned its place.** Phase 0 said one would arrive
  with project management. It did not: what justifies a router is a URL worth
  addressing, and nothing here is addressed from outside the window. No second
  window, no deep link, no history to be wrong about.
- **Two contracts were added that no earlier phase needed.** `SeekIndex`,
  because the engine read the per-frame timestamps from Phase 1 onward and threw
  them away at the contract boundary; and `PoseOverlay`, because the landmarks
  worth drawing are the _filtered_ ones and converting them back into drawing
  coordinates needs the aspect ratio, the filter's validity mask and the lens.
  A UI doing that arithmetic would have been a second opinion about all three.
- **The overlay draws filtered landmarks, not the estimator's output.** An
  overlay here is an instrument: it exists so a number can be checked against
  the frame it came from. Drawn from the raw landmarks it would sit slightly
  elsewhere than the thing being checked, and every disagreement would be
  unattributable. The cost is that a lens-corrected skeleton genuinely does not
  sit on the pixels underneath it — by up to the tens of pixels Phase 8 measured
  near the frame edge — and `PoseOverlay.undistorted` puts that on screen rather
  than leaving it to be discovered.
- **Four contracts were exported to TypeScript, each on the terms its own phase
  set.** `Project`/`ProjectList` were withheld until project management existed
  (14.1); `CoachingReport` until a findings panel drew it (14.7), and the
  rendering questions that exclusion predicted did get answered by having to
  answer them — a refusal is its own list rather than a greyed finding, and a
  citation is a jump target rather than a link, because the video is on the same
  screen. `SequenceFilterReport` remains out on its original terms: no panel
  draws it, and Phase 14 renders the filter's _warnings_, not its report.
- **The type generator needed fixing rather than the contract.**
  `tuple[Landmark, Landmark]` is JSON Schema 2020-12 `prefixItems`, which
  `json-schema-to-typescript` does not read — it emits `[unknown, unknown]`, a
  type that compiles, carries nothing, and fails at the point of use.
  `gen_types.py` now also writes the draft-07 tuple form. Reshaping
  `POSE_CONNECTIONS` to a `list[list[Landmark]]` would have given up "exactly
  two" in both languages to satisfy a generator in one.
- **The app opens on the workflow, not on the environment report.** Diagnostics
  were the right front door while the environment was the only thing the app
  could tell you about.
- **`preload="auto"` on the video.** A player built to be scrubbed should not
  seek into an unfetched part of a local file; a swing clip is seconds long and
  already on the machine.
- **An unrelated defect in the e2e harness was found and fixed.** The fixture
  stubbed `unregisterListener` on `__TAURI_INTERNALS__`; the event plugin keeps
  its own `__TAURI_EVENT_PLUGIN_INTERNALS__`, so every progress subscription
  teardown had been throwing an unhandled rejection inside the page — passing
  the test while leaving the teardown path untested. Invisible until Phase 14
  put four subscribe/unsubscribe cycles behind one click.

### Cost

`seek_index` is one probe of the container index: **72.6 ms**, the figure Phase 1
measured, and deliberately uncached for Phase 3's reason. `pose_overlay` over a
whole clip is the filter plus a conversion — milliseconds against the seconds
pose extraction already cost. Nothing new is cached.

**Open, and not resolved here:**

- The seek map is verified in **Chromium**. The app ships on WKWebView, and
  `tauri-driver` is not wired up, so the packaged binary's player has no
  automated coverage. The residual display is the mitigation and not a
  substitute: it makes a WKWebView that behaves differently visible to the
  person using it rather than silent.
- No two-camera screen, for the reason above: no pair of clips in this
  repository is two views of one swing.

## Phase 15 — 3D visualisation ✅

`python/analyzer/{scene.py,contracts/scene.py}` ·
`apps/desktop/src/features/scene/` ·
[ADR-0019](decisions/ADR-0019-the-viewpoint-is-part-of-the-measurement.md)

**A drawing is made from somewhere, and a report is not.** That is the whole
phase. Phase 9 measured how well two cameras locate a joint and reported a
number; Phase 15 draws the joint, and discovers that the same measurement looks
like a confident dot or a ten-centimetre smear depending only on where the
reader is standing — with the default viewpoint, the one almost nobody moves,
being the most flattering one available.

- [x] **15.1 Scene + camera controls** — orbit, zoom, and a default that is the
      **reference camera itself**, with the focal length the calibration
      measured. Not R3F; see the deviation below, and the measurement behind it
- [x] **15.2 Skeleton + trajectory from reconstruction output** — a new
      `ReconstructionScene` contract, because `ReconstructionReport` deliberately
      carries no points. Bones drawn only where both ends survived; hand paths
      **broken** at a refused instant rather than drawn across
- [x] **15.3 Shared timeline** — not a store: one number. The viewport reads the
      frame `useFramePlayer` reports the browser **painted**, so there is no
      second clock to drift. What is added is the guard that number cannot
      supply — a content-key check that the video is the recording the geometry
      came from
- [x] **15.4 Confidence + calibration status in the viewport** — the calibration
      status, the ray convergence, the uncertainty in millimetres, and the number
      this phase exists for: **what fraction of that uncertainty the current
      viewpoint can show**
- [x] **15.5 Tests** — 23 pytest, 68 Vitest, 7 Playwright: the projection pinned
      across two languages, degenerate frames, and the scrub against a real
      decoder
- [x] **15.6 Commit** — with the measurements below

### The headline, and it is the third instance of the same shape

`scripts/benchmark_viewport.py --sweep convergence`, Apple M1 Pro / macOS
26.4.1. The synthetic stereo fixture at the 2.7 px landmark scatter Phase 3
measured on real footage, with the two cameras brought together from a right
angle — which is what someone filming with two phones on one side of a bay does:

| separation | ray angle | true sigma  | visible fraction | **sigma on screen** |
| ---------- | --------- | ----------- | ---------------- | ------------------- |
| 90°        | 94°       | 5.5 mm      | 0.96             | **5.3 mm**          |
| 60°        | 64°       | 7.3 mm      | 0.74             | **5.3 mm**          |
| 45°        | 48°       | 9.4 mm      | 0.57             | **5.3 mm**          |
| 30°        | 32°       | 13.7 mm     | 0.39             | **5.3 mm**          |
| 20°        | 21°       | 20.2 mm     | 0.26             | **5.3 mm**          |
| 15°        | 17°       | **25.6 mm** | 0.20             | **5.2 mm**          |

**The last column is flat to one decimal place while the third grows 4.7x.** A
reader looking at the default view of a 15° capture sees the same smear as a
reader looking at a 90° one, because the direction that grew is the direction
that camera is looking along. Phase 8 found a residual flat across a 400x change
in focal error; Phase 9 found one flat across a 4.7x change in 3D error; this is
the same blindness a third time, and it is worse than both because it is a
picture rather than a number somebody might think to distrust.

Walking round the body, at the same two captures:

| orbit from reference | 90° pair: visible | on screen | 15° pair: visible | on screen |
| -------------------- | ----------------- | --------- | ----------------- | --------- |
| 0° (reference)       | 0.96              | 5.3 mm    | **0.20**          | 5.2 mm    |
| 15°                  | 0.97              | 5.3 mm    | 0.17              | 4.3 mm    |
| 30°                  | 0.97              | 5.4 mm    | 0.36              | 9.2 mm    |
| 45°                  | 0.98              | 5.4 mm    | 0.57              | 14.4 mm   |
| 60°                  | 0.98              | 5.3 mm    | 0.75              | 19.0 mm   |
| 90°                  | 0.96              | 5.3 mm    | **0.98**          | 24.7 mm   |

A well-conditioned pair barely changes — its ellipsoid is nearly a sphere and
there is no bad direction to find. A shallow one hides four fifths of its error
at zero and gives all of it up by ninety. **On screen this is unmissable**: the
15° capture drawn from the reference camera is a tidy skeleton with small round
halos, and the same frame orbited 90° is a stick figure inside four flat
ellipses wider than the body.

### The claim that makes a hand-written projection safe to ship

`benchmark_viewport.py --sweep projection`, over 8,778 reconstructed points:

| points | max disagreement with `StereoGeometry.project` |
| ------ | ---------------------------------------------- |
| 8,778  | **0.000e+00 px**                               |

Placed at the reference camera and projected with the intrinsics the scene
carries, every point lands exactly where the engine puts it — which is where
that camera saw the landmark and where `PoseOverlay` draws it. The viewport's
default view and the video are then two renderings of one measurement.
`projection-truth.json` carries 32 of those points and their pixels into
TypeScript, and `projection.test.ts` asserts the same thing there, so the two
implementations cannot drift apart without a test failing.

### Cost

`benchmark_viewport.py --sweep cost`, plus `JSON.parse` measured in Node:

| frames | build    | serialise | parse   | payload | per frame |
| ------ | -------- | --------- | ------- | ------- | --------- |
| 68     | 26.5 ms  | 5.8 ms    | —       | 0.98 MB | 14.0 KB   |
| 96     | 42.0 ms  | 8.0 ms    | —       | 1.37 MB | 13.9 KB   |
| 312    | 158.3 ms | 23.7 ms   | 11.9 ms | 3.98 MB | 12.5 KB   |

Against roughly 1.3 s per clip to extract poses and ~70 ms to reconstruct.
A scene point carries a position, six covariance elements and three diagnostics
against an overlay point's two coordinates, which is what set `MAX_SCENE_FRAMES`
at **600** rather than the overlay's 2000. Nothing is cached, for Phase 3's
measured reason.

### Deliberate deviations from the original plan

- **No React Three Fiber, and no WebGL.** 15.1 asks for an R3F scene; what
  shipped is a projection written in `projection.ts` and an SVG figure. Three
  reasons, in the order they decided it. **(1) The exit criterion is a
  measurement.** "Scrub stays in sync with video" has to be _checked_, and Phase
  14 established what checking means here — a real browser, a real decoder, and
  the frame it reports painting looked back up. A WebGL canvas exposes nothing
  about what it drew, so verifying it is a screenshot diff, which cannot say
  _which frame_ is on screen. Every joint in the viewport is a DOM node carrying
  its landmark, so `e2e/scene.spec.ts` asserts that the wrist moved 112 px when
  the painted frame moved 20 — against a fixture that travels a known centimetre
  per frame. **(2) The uncertainty ellipse needs the projection in hand**: what
  is drawn per joint is `A S A'`, the 3D covariance pushed through this camera's
  own Jacobian, which is not a scaled sphere. **(3) The scene is small** — about
  150 nodes a frame, which is not what a scene graph is for. The cost is stated
  rather than hidden: no depth buffer, no lighting, and drawing order standing in
  for occlusion. That is the right trade for a stick figure whose job is to show
  where a measurement is weak, and it would not survive a mesh.
- **15.3's "shared timeline store" is not a store, and building one would have
  been the bug.** Two timelines kept in step is two things that can drift; the
  viewport instead reads the single frame index `useFramePlayer` already
  publishes — the frame the **browser reported painting**, not the one that was
  requested. What a shared number cannot supply is whether the pixels and the
  geometry are the same recording, so `sceneMatchesClip` compares the scene's
  `reference_content_key` against the seek index's. That failure is the one worth
  guarding: the video plays, the skeleton moves, the frame numbers agree, and the
  body on screen has nothing to do with the footage behind it.
- **The uncertainty ellipses are magnified, and the factor is drawn in the
  picture.** At true scale a joint determined to 5.4 mm at 3.4 m through a
  1400 px focal length is 2.2 px of a 1920-wide frame — about two thirds of one
  screen pixel. Drawn honestly at 1x the feature shows nothing in exactly the
  cases it exists for. So it is exaggerated, ×20 by default, with
  `uncertainty ×20 · 1σ` rendered in the corner of every frame that has one and
  ×1 offered so the true scale can be seen for what it is. The millimetres in the
  panel remain the truth; the ellipse carries the part a number cannot, which is
  the shape and how it changes as the reader moves.
- **`uncertainty_covariance` is new; `positional_uncertainty` is untouched.**
  Phase 9's scalar is what every gate and report uses and it is cheaper. The
  covariance is built from `J'J`'s own eigendecomposition rather than by
  inverting it — the systems a shallow convergence angle produces are nearly
  singular and a direct inverse loses precision first in the smallest eigenvalue,
  which is the largest axis of the covariance and the entire reason for computing
  one. A test asserts the two agree to 1e-9, because "the same by construction"
  across two functions is a claim and not a guarantee.
- **`ReconstructedSequence` gained a `refusal` array.** The report counts Phase
  9's five refusal reasons per landmark, which is what a reader of numbers needs;
  a viewport draws the individual hole and has to say which of the five it is.
  The reason could not be re-derived downstream, because every diagnostic array
  is masked to NaN for a refused point — refusal removes the evidence a reason
  would be inferred from.
- **The type generator needed fixing again, and again rather than the contract.**
  Pydantic writes a documented reference as `{"$ref": ..., "description": ...}`,
  which `json-schema-to-typescript` treats as an anonymous schema and inlines a
  **copy** of: one `Vec3` used by five fields came out as `Vec3` plus `Vec31`
  through `Vec35`. `gen_types.py` now rewrites those into draft-07's
  one-element `allOf`, which the generator reads as a reference. Dropping the
  descriptions would have de-duplicated the types too, at the cost of deleting
  the sentence saying `up` points along the image's -y — exactly the kind of
  convention that produces a plausible picture rather than an error. The fix
  improved every other generated file: a shared model now keeps its own docstring
  instead of having it overwritten by whichever field referenced it.
- **The viewport is a fifth screen, not a panel on Swing.** It is the only screen
  that needs a _pair_ — two clips, a calibration and an alignment — and none of
  the three is recoverable from the loose file the Swing screen works on.
- **The reference clip is still picked by hand.** The project knows its path and
  the WebView still may not read it: Phase 14's asset scope ships empty and
  `choose_clip` opens its own dialog in Rust precisely so that a grant is tied to
  something the frontend could not have fabricated. A command taking a path would
  have saved a click and undone the only thing making that scope mean anything.
  The content key is what confirms the right file was chosen.
- **No ground plane, no horizon, no gravity.** The scene is `CAMERA` metres.
  Phase 9 recorded that a scene-fixed frame needs a measured vertical and a
  target line and that a stereo pair supplies neither; a grid on the floor would
  be a drawing of that assumption. The orbit turns about the reference camera's
  own up, which is a statement about the picture and is labelled as one.
- **A `scene.fixture.ts` sits beside the tests and is imported by nothing in the
  app.** Its scene starts at frame 10, refuses one landmark on one frame and
  every landmark on another. A fixture starting at zero would let a viewport that
  indexed positionally pass every test and draw the wrong body on every windowed
  scene.

### Verified

`analyzer scene <project>` on the only two-clip project this repository can build
refuses before reading any footage — _"Triangulation needs both cameras
calibrated and their relative pose measured; this project has 'none'"_ — which is
Phase 9's `require_stereo_rig` running first, so the reported failure is the
first thing that was wrong rather than the last thing that went wrong.

The viewport itself was run in a browser against a real reconstruction of the
synthetic stereo fixture and **looked at**, which is how Phase 9 caught a camera
convention its own error metric could not see. The body is the right way up, the
hand path arcs up and back, and at frame 60 the arms are at the top. At 90°
separation the panel reads 5.4 mm / 92% visible / drawn as 4.9 mm; at 15° it
reads 22.8 mm / 21% / **drawn as 4.9 mm** — the benchmark's flat column, live on
screen, with nine landmarks refused for rays too shallow to intersect.

**Open, and not resolved here:**

- **Nothing here has been run on a real reconstruction**, and the benchmark says
  so in its own docstring. Phase 7 established that this repository's only
  two-angle pair is not the same swing and Phase 8 that no real calibration
  footage exists, so every millimetre above comes from a fixture whose body is an
  input and which contains no pose estimator. The viewpoint arithmetic is exact —
  it is geometry, and does not care where the points came from — and the
  millimetres are a floor.
- The viewport is verified in **Chromium**, the app ships on WKWebView, and
  `tauri-driver` is still not wired up. Phase 14's limitation, unchanged.
- **`MAX_SCENE_FRAMES` is 600 because of a JSON encoding.** A columnar payload
  would cut it several-fold; it was not taken because the per-point object is
  what carries the "null exactly when refused" invariant into TypeScript, and a
  positional array would move that check out of the type system and into every
  consumer. Worth revisiting in Phase 17, where it is a measured bottleneck or it
  is nothing.

## Phase 16 — Swing comparison ✅

`python/analyzer/comparison/{normalise,diff,compare}.py` ·
`python/analyzer/contracts/comparison.py` ·
`apps/desktop/src/features/compare/` ·
[ADR-0020](decisions/ADR-0020-a-difference-between-recordings.md)

**Two recordings differ for reasons that have nothing to do with the two
swings**, and the largest of those reasons turns out to be one nobody measures:
where the tripod was. The same shoulder turn, on a body that does not move,
reads as 57.7° from square on and 32.9° from thirty degrees round. Every phase
since Phase 8 has found a number blind to its own capture; this one finds the
mirror image — a number that moves 25 degrees while the thing it appears to
describe does not move at all.

- [x] **16.1 Phase-relative temporal normalisation** — a piecewise-linear clock
      with a knot at each of the four events, so 0 is the takeaway, 1 the top,
      2 impact and 3 the finish. Linear **in time** within a phase, never in
      frame number. Every sample carries the event ambiguity propagated onto the
      axis, which is exact and cancels the phase duration
- [x] **16.2 Comparison contracts + diff computation** — `SwingComparison`,
      `MetricDifference`, `RefusedDifference` with nine typed refusals, and a
      `CameraAgreement` that gates every projected quantity before any of them
      is looked at. **No score, no severity, no "better"**, and a test asserts
      the absence of those field names in both languages
- [x] **16.3 Comparison UI** — a sixth screen. Overlaid trajectories with the
      bracket drawn as a **band** rather than left implicit, metric deltas with
      their bracket itemised by source, and both clocks on screen because the
      normalisation destroyed the timing to make the plots possible
- [x] **16.4 Tests** — 110 added (73 pytest, 30 Vitest, 7 Playwright): the
      normalisation against warps whose answer is zero by construction, every
      gate, the invariant that a clip compared with itself reports nothing, and
      the plots laid out by a real browser against two real engine outputs
- [x] **16.5 Commit** — with the measurements below

### The measurement, and it is the opposite shape to every previous one

`scripts/benchmark_compare.py --sweep camera`, Apple M1 Pro / macOS 26.4.1. One
**unchanged** 3D swing — same body, same instants, same joint angles — projected
through cameras that differ only in where they stand, at the 2.7 px landmark
scatter Phase 3 measured on real footage. Median of five seeds:

| azimuth | address span | spans disagree | **shoulder turn reported** | verdict          |
| ------- | ------------ | -------------- | -------------------------- | ---------------- |
| 0°      | 1.05 torso   | 0%             | **57.7°**                  | unresolved       |
| 2°      | 1.05         | 1%             | 57.6°                      | unresolved       |
| 5°      | 1.04         | 1%             | 56.9°                      | unresolved       |
| 10°     | 1.02         | 3%             | 55.1°                      | unresolved       |
| 15°     | 0.99         | 6%             | 52.1°                      | unresolved       |
| 20°     | 0.95         | 11%            | 47.7°                      | **camera_moved** |
| 30°     | 0.83         | 24%            | **32.9°**                  | **camera_moved** |

**Nothing about the body changes down that table.** Every row is still classified
`face_on`, because Phase 6's three labels are for deciding whether a recording
contains a measurement at all and a camera can move thirty degrees round a player
without leaving one. A comparison that trusted the label would report a 25-degree
change in the most quoted number in golf instruction, on a swing nobody made.

The arithmetic is not subtle once written down. From a camera `a` degrees off
broadside, a line that truly turned `t` projects `cos(a + t)` against an address
span of `cos(a)`, so the reported angle is `arccos(cos(a + t) / cos(a))` — equal
to `t` only at `a = 0`. Fifty degrees seen from ten degrees off reads as
fifty-six; from twenty, sixty-five.

### Why the gate is the span and not the angle derived from it

`openness` is the address span over the widest that line was seen in the clip,
which is the cosine of how far off broadside the camera stood — so its arccosine
is an azimuth, and gating on that is the obvious move. It does not survive
contact: **the arccosine is flat near broadside, so 1% of landmark noise in
`openness` comes out as 8.1 degrees.** Phase 5 already recorded the reference
footage reading 12% wider than the player can physically be at its widest frame,
which is 28 degrees. A gate on the derived angle refuses every pair ever filmed.

So the verdict is the **span disagreement**, which is measured and
well-conditioned, and the azimuth is used only to widen a bracket — where being
pessimistic is the safe direction. Full reasoning in
[ADR-0020](decisions/ADR-0020-a-difference-between-recordings.md).

### Does the normalisation recover a warp whose answer is known?

`--sweep warp`. `tests/synthetic.py` builds its hand arc from the **fraction**
through each phase, so two swings differing only in their event times are one
swing under a piecewise-linear time warp — and the residual after normalising is
the map's own error, because the truth is zero. Hand height in frame widths, over
a signal that ranges about 0.30:

| tempo | backswing | downswing | **phase-relative** | uniform stretch |
| ----- | --------- | --------- | ------------------ | --------------- |
| 1.92  | 0.767 s   | 0.400 s   | **0.0000**         | 0.0000          |
| 1.17  | 0.583 s   | 0.500 s   | **0.0049**         | 0.1685          |
| 0.98  | 0.392 s   | 0.400 s   | **0.0073**         | 0.2085          |
| 3.59  | 1.108 s   | 0.308 s   | **0.0080**         | 0.2643          |
| 2.04  | 1.142 s   | 0.558 s   | **0.0041**         | 0.0393          |

The last column prices the obvious alternative: one linear stretch of
takeaway-to-finish, which is what a normalisation does if it does not know what a
swing is. It is exact only where the two tempos happen to match and is **thirty
times worse** where they do not — and tempo is the quantity two golfers are most
likely to differ by, so it is worst exactly where it is used.

### What the normalisation destroys, deliberately

The knots coincide by construction, so **every timing difference between the two
swings is divided out by the map**. Nothing in an overlaid trajectory can say
that one player reached the top later. That comparison lives in the metric
differences, as a duration with a bracket of its own, and the clock's four knots
are carried onto the screen so a reader can see what was removed.

### What a difference has to clear, and what a faster camera buys

`--sweep resolution`. Two swings of genuinely different shape at one tempo, so
the difference is a fact about the pair and does not change down the table:

| fps | hand height resolved | hand speed resolved | largest difference |
| --- | -------------------- | ------------------- | ------------------ |
| 30  | 39%                  | 39%                 | −0.492             |
| 60  | 61%                  | 61%                 | −0.492             |
| 120 | 80%                  | 78%                 | −0.492             |
| 240 | 89%                  | 89%                 | −0.492             |
| 480 | 94%                  | 96%                 | −0.492             |

The difference is constant; what grows is how much of it a pair of recordings can
attribute to the swings rather than to the clock. At 30 fps three fifths of the
swing carries a difference nobody can call one.

### What the real footage produces

`--sweep clips`, over the three pairs this repository can build:

| pair                    | views                   | differences | refused | commonest refusal |
| ----------------------- | ----------------------- | ----------- | ------- | ----------------- |
| rory_dtl / rory_dtl_2   | down_the_line / unknown | 0           | 32      | view_mismatch ×27 |
| PW_face-on / iron_dtl   | face_on / down_the_line | **4**       | 36      | view_mismatch ×35 |
| rory_face_on / rory_dtl | face_on / down_the_line | **1**       | 38      | view_mismatch ×34 |

Two of the three are a face-on camera against a down-the-line one, so every
projected quantity refuses and only the timings survive — a tempo difference of
1.63 on the amateur pair and 0.572 on the tour pair. That second figure
independently corroborates Phase 7, which found those two clips are not the same
swing.

**The third pair is the interesting one, and it is a capture instruction.** Two
swings by one player from one position — the only such pair here — and it refuses
as well, because `rory_dtl_2.mp4` begins at the takeaway. A clip with no address
phase has no measured view, and a comparison that cannot say where either camera
stood will not compare a projection. **Start recording before the player is set
up to the ball**: the address phase is where the view, the rotation baseline and
the hand-path origin all come from.

### Cost

`--sweep cost`, median of five:

| samples | compare  | serialise | payload |
| ------- | -------- | --------- | ------- |
| 61      | 6.1 ms   | 1.8 ms    | 68 KB   |
| **121** | **11.2** | **1.4**   | **111** |
| 241     | 21.7     | 2.4       | 197     |
| 481     | 42.5     | 4.6       | 371     |

Against roughly 1.3 s per clip to extract poses, twice. 121 samples is the
default: one every fortieth of a phase, finer than the frame rate of any clip
here and coarse enough that the payload stays small. Nothing is cached, for
Phase 3's measured reason.

### Deliberate deviations from the original plan

- **Brackets are summed across the two clips, not combined in quadrature, which
  contradicts `coaching.combined_bracket` on purpose.** That function bounds the
  difference of two measurements sharing a systematic error — a foreshortening
  baseline taken at address is the same baseline at both anchors of one clip, and
  a shared error largely cancels. **Two separate recordings share nothing.** A
  sum of bounds is a bound; a quadrature of bounds is not. Same question, two
  clips apart, opposite answer.
- **`coaching.bracket` is imported rather than re-derived.** What one recording
  can resolve is one question with one answer, and two layers computing it
  separately would eventually disagree about a number a reader sees in both. It
  makes `comparison` depend on `coaching`, which is a dependency between two
  packages at the same level and is the right price.
- **`Direction` has two members and there is no "the same".** Two values closer
  together than the pair can resolve produce `UNRESOLVED`, which says the
  recordings could not tell them apart rather than that the swings agreed — an
  absence of evidence, filed where a reader can see what would resolve it. The
  refusal carries **both values and the bracket**, because what is refused is the
  claim, not the numbers.
- **The bracket on a trajectory sample is a range, not a slope.** A linearisation
  under-reports exactly at a peak, which on a hand-speed curve is the middle of
  the downswing — the part of a swing anybody comparing two of them cares about.
  The bracket is instead the largest the signal moves anywhere inside that
  instant's own ambiguity window, which is a bound rather than an estimate, and a
  test checks it directly by re-reading the signal from a clock shifted a whole
  frame.
- **The window is widened by one sample at each end before that range is taken**,
  and that is not slack. A reading inside the window is interpolated between the
  two clip samples bracketing it, and one of that pair may sit outside the edge —
  so a range over the strictly-interior samples is not a bound. The test above is
  what found it.
- **The event ambiguity is widened by a disagreeing corroboration.** Phase 4
  corroborates impact with the low point of the hand arc; where two independent
  estimates of one instant sit 40 ms apart, 40 ms is the ambiguity and quoting one
  frame would be taking the finer of two numbers that contradict each other.
- **A refused channel carries no samples.** Two curves drawn on one axis is a
  comparison whatever the caption says, so a channel the camera gate refused is
  not drawn at all — the same argument Phase 15 made for why a picture is worse
  than a number somebody might think to distrust.
- **No video element on the Compare screen.** Phase 14 put a player behind the
  frame indices because a metric is checked against the frame it came from. Two
  players would double that and deliver neither: the frames a difference cites are
  in two different files and only one can be on screen. The frames are printed
  instead, for the Swing screen to follow.
- **The synthetic body had to be widened for these tests.** `tests/synthetic.BODY`
  projects its shoulders at 0.4 torso lengths, which the view detector correctly
  calls oblique — so every projected comparison over that fixture refuses, which
  is the right answer for that body and not the case under test. The test module
  and the benchmark each widen it to the 0.8 a face-on recording produces, and say
  so.

### Verified

`analyzer compare data/amateur/face-on/PW_face-on.mp4 data/amateur/dtl/iron_dtl.mp4
--window 0.17` reports the two cameras 108° apart, refuses all 36 projected
quantities by name, and compares the four timings — including a tempo of 3.43:1
against 1.80:1, which clears a bracket of 1.05. The target clip's impact carries
an ambiguity of **100 ms rather than 33**, because Phase 4's hand-arc corroboration
disagrees with the speed peak by three frames there. That is the widened-ambiguity
rule firing on real footage rather than on a fixture.

The Compare screen was run in a browser against two real engine outputs and
**looked at**, which is how Phase 15 caught a convention its own metric could not
see. Two things were wrong in the picture and are fixed: the plots carried no
vertical scale at all, so a reader saw the shape of a difference and not its size;
and `largest_difference`, which the engine had been computing since 16.2, reached
nothing on screen. Both are now in the figure — two axis labels and a caption
reading "the largest −1.73 at impact".

### Open, and not resolved here

- **The trajectory bracket covers _when_ a sample was taken, not _how well_ the
  value at it was measured.** The azimuth-0 row of the camera sweep measures what
  that leaves: **5% of sampled positions report a resolved difference on a swing
  that did not change**, from landmark noise alone at two seeds. Closing it needs
  a per-sample landmark uncertainty, and Phase 4 recorded why the filter's own
  residual cannot supply one — it is exactly zero whenever the smoothing window
  holds as many samples as the polynomial has coefficients, which is every clip
  below about 60 fps at the shipped defaults.
- **The span gate assumes both clips show one body.** A broader-shouldered player
  projects a broader line from the same place, so two golfers filmed from one
  tripod can fail it. Nothing in either recording separates that from a camera
  that moved, and both explain a difference the swing did not make.
- **The horizontal bracket is one frame, and Phase 7 measured the takeaway moving
  sixty-five.** One frame per event is the same convention `coaching.bracket`
  applies to a duration and is a floor; under landmark noise at 120 fps Phase 7
  watched the top move 8 ms and the takeaway 542 ms. Nothing in a single clip
  measures that, so the refusals here are a lower bound on the refusals warranted
  — and the takeaway end of every plot is the least trustworthy part of it.
- **The camera sweep's degrees are a floor.** The 3D fixture's body is an input,
  there is no pose estimator in it, and the noise is an explicit displacement
  rather than an estimator's structured error. A real estimator loses the
  shoulders to motion blur exactly where this fixture is perfect.

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
