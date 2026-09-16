# Golf Swing Analyzer

Local-first golf swing biomechanics analysis from face-on and down-the-line
video. Pose and club motion are extracted with computer vision, swing phases are
segmented deterministically, and biomechanics metrics are computed with explicit
units, confidence, and methodology. All processing runs on your machine; video
never leaves it.

> **Status: Phases 0-3 of 21 complete.** The foundation, typed engine boundary,
> environment health check, video ingestion, single-camera pose extraction, and
> temporal filtering are built and verified. No phase detection, metrics, or
> coaching exist yet. Sections below marked _Not yet implemented_ say so rather
> than describing features that do not exist. See
> [docs/ROADMAP.md](docs/ROADMAP.md).

---

## 1. Project overview

The goal is a serious computer-vision system, not a wrapper that forwards video
to a language model. The analysis pipeline produces structured quantitative
evidence; any LLM layer added later only translates that evidence into coaching
language and is forbidden from inventing measurements.

Design commitments that shape everything else:

- **Every displayed number comes from a real computation.** No placeholders, no
  illustrative values, no estimated benchmarks.
- **Capabilities are reported as measured.** If pose inference runs on CPU, the
  UI says so even when a GPU is present.
- **Uncertainty is first-class.** Metrics carry confidence and methodology;
  detectors emit nothing rather than guessing.
- **Subsystems are swappable.** CV models sit behind typed interfaces so a
  learned detector can replace a classical baseline without touching callers.

## 2. Architecture

```
apps/desktop/          Tauri 2 + React 19 + TypeScript + Tailwind 4
  src/                   WebView UI
  src-tauri/             Rust core: transport + process supervision
packages/types/        Shared types (generated from Pydantic + hand-written IPC)
python/analyzer/       Analysis engine
models/                Model manifest (weights fetched, not committed)
docs/                  Architecture, decisions, roadmap
scripts/               Bootstrap, codegen, model download, health check
```

The desktop app spawns the Python engine as a long-lived child process and
speaks newline-delimited JSON-RPC 2.0 over stdin/stdout — no local HTTP server,
no listening port. Full diagram and call trace in
[docs/architecture.md](docs/architecture.md); the rationale is
[ADR-0005](docs/decisions/ADR-0005-engine-boundary.md).

## 3. Installation

Requirements: macOS on Apple silicon (the only platform verified so far),
Node >= 20.19, FFmpeg, and Xcode Command Line Tools. `uv` and Rust are installed
by the bootstrap script if missing.

```bash
git clone <repo-url> golf-swing-analyzer
cd golf-swing-analyzer
./scripts/bootstrap.sh
```

The script verifies prerequisites, installs `uv` and Rust if absent, creates the
Python 3.12 environment, downloads and hash-verifies the pose models, and runs
the health check. It is idempotent and does not modify your shell profile.

If `cargo` is not on your `PATH` in new shells, add:

```bash
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
```

## 4. Development

```bash
npm run dev          # launch the desktop app (Vite + Tauri, hot reload)
npm run doctor       # environment health check in the terminal
npm run check:all    # every lint, typecheck and test suite
npm run gen:types    # regenerate TypeScript from the Pydantic contracts
```

Per-language commands: `npm run py:test`, `npm run py:lint`,
`npm run py:typecheck`, `npm run rs:test`, `npm run rs:lint`, `npm test`,
`npm run test:e2e`.

**Changing a contract:** edit the Pydantic model in
`python/analyzer/contracts/`, then run `npm run gen:types`. CI fails if the
checked-in TypeScript does not match.

## 5. Running analysis

_Partially implemented — Phases 4-13 outstanding._ A clip can be imported,
inspected, run through pose estimation, and filtered into trajectories with
derivatives — from the app's **Video** screen or from a terminal:

```bash
uv run --project python analyzer probe   path/to/swing.mov   # container metadata
uv run --project python analyzer extract path/to/swing.mov   # pose landmarks
uv run --project python analyzer extract path/to/swing.mov --model pose_landmarker_lite
uv run --project python analyzer filter  path/to/swing.mov   # smooth + differentiate
uv run --project python analyzer filter  path/to/swing.mov --window 0.15 --polyorder 4
```

Extraction writes landmarks to a Parquet file keyed by the video's content, and
reports what it measured: frames processed, how many contained a pose, time per
frame, and the model digest it ran with.

Filtering reads those landmarks back and produces position, velocity and
acceleration per landmark, along with an account of what it refused: detections
below the confidence gate, gaps too long to bridge, and windows without enough
support to fit. Nothing is persisted — filtering a clip costs milliseconds
against seconds of extraction. No metrics or coaching consume it yet.

The engine methods are `doctor`, `probe_video`, `extract_poses` and
`filter_poses`.

## 6. Supported video formats

Whatever FFmpeg can demux; tested against MP4/MOV with H.264. Two properties of
consumer recordings are handled explicitly, because both fail silently rather
than loudly if they are got wrong.

**Rotation.** Phones store upright frames sideways with a display matrix saying
how to present them, and decoders disagree about whether they apply it — OpenCV
does by default. Frames are always delivered in display orientation, rotated in
exactly one place, and the metadata reports coded and display dimensions
separately so the difference is visible rather than implied.

**Variable frame rate.** Frame times come from the container's presentation
timestamps, never from `frame_index / fps`. A clip whose intervals are not
uniform is flagged, with the measurement behind the flag shown next to it. When
a container carries no timestamps at all, the verdict is reported as _unknown_
rather than as _constant_.

Rejected with a specific reason and a suggested fix: files that do not exist,
directories, zero-byte files (an interrupted copy), corrupt containers, and
media with no video stream. A truncated recording is accepted and reported with
the frame count it actually has, alongside the larger count its header claims.

Decoding runs on the CPU by default. VideoToolbox is available but measured
slower for this pipeline — see
[ADR-0007](docs/decisions/ADR-0007-decode-backend.md).

## 7. Hardware requirements

Verified reference machine: Apple M1 Pro (10-core CPU, 16-core GPU), 16 GB RAM,
macOS 26.4.1.

Acceleration is detected at runtime and never hard-coded to a vendor. On this
machine:

| Capability         | State                                                    |
| ------------------ | -------------------------------------------------------- |
| PyTorch device     | MPS (Metal)                                              |
| CUDA               | Unavailable — Apple silicon                              |
| MediaPipe delegate | **CPU** — the Tasks Python API has no macOS GPU delegate |
| FFmpeg hwaccel     | VideoToolbox                                             |
| CPU fallback       | Always available                                         |

The MediaPipe row is called out because it is easy to misreport: torch having
MPS does not make pose inference GPU-accelerated.

## 8. Computer vision pipeline

_Partially implemented — Phases 4-11 outstanding._ Three stages are built.

**Ingestion.** Container inspection and frame decoding behind a `FrameSource`
interface that yields display-oriented frames carrying real presentation
timestamps. Two implementations (in-process OpenCV, and an ffmpeg subprocess
that can use VideoToolbox) are cross-checked against each other.

**Pose.** A `PoseEstimator` interface with MediaPipe behind it, producing 33
landmarks per frame in two coordinate spaces, stored as Parquet and reloadable.
The spaces are kept apart deliberately: `IMAGE` is normalised to the frame and
is the only space a landmark can be drawn in, while `HIP_LOCAL` is MediaPipe's
"world" output — hip-centred, only roughly metric, and carrying no camera
geometry. It is **not** calibrated world coordinates, and no metric claim rests
on it. Real world coordinates arrive in Phase 9 from stereo triangulation.

**Filtering.** Landmark trajectories are smoothed and differentiated by local
polynomial regression solved at each sample on the clip's real timestamps.
Savitzky–Golay is the uniform-grid special case of that, and the test suite pins
the equivalence against SciPy to floating-point precision — but the general form
is what runs, because a fixed convolution kernel silently biases every derivative
on variable-rate footage. Velocity and acceleration are coefficients of the same
fit rather than finite differences of smoothed positions, so the three are
mutually consistent. Low-confidence detections become absences, absences longer
than the gap policy stay absent, and windows without enough observations emit
nothing at all. See [ADR-0009](docs/decisions/ADR-0009-local-polynomial-filtering.md).

Phase detection, club and ball tracking are not built. Planned stages and their
ordering are in [docs/ROADMAP.md](docs/ROADMAP.md).

## 9. 3D reconstruction methodology

_Not yet implemented — Phases 8-9._ The system will distinguish
`UNCALIBRATED` / `INTRINSIC_ONLY` / `STEREO_CALIBRATED` and will refuse to make
metric-scale claims without stereo calibration.

## 10. Biomechanics methodology

_Not yet implemented — Phase 5._ Every metric will carry name, value, unit,
phase, confidence, source frames, and methodology. Angles derived from a single
2D camera will be labelled as projected, not true 3D rotation.

## 11. Model architecture

_No model is trained — Phase 12._ No accuracy figure will be published without a
real labelled evaluation set with session- and player-grouped splits.

Currently vendored: MediaPipe Pose Landmarker (lite/full/heavy, float16), pinned
by sha256 in `models/manifest.json`. See
[ADR-0006](docs/decisions/ADR-0006-model-pinning.md). It sits behind a
`PoseEstimator` interface and its types do not escape the adapter, so replacing
it — with another variant, or with something learned later — touches nothing
that consumes poses.

MediaPipe itself is pinned to **1.0.0 exactly**: 1.0.1 aborts the process when
the pose graph opens on macOS arm64. The health check now runs a real inference
rather than trusting an import, which is what caught it. See
[ADR-0008](docs/decisions/ADR-0008-mediapipe-1.0.0.md).

## 12. Testing

```bash
npm run check:all
```

| Suite      | Count | Scope                                                                                                |
| ---------- | ----- | ---------------------------------------------------------------------------------------------------- |
| pytest     | 407   | contracts, environment probes, model verification, dispatch, RPC framing, ingestion, pose, filtering |
| cargo test | 12    | protocol framing, id correlation, `uv`/project resolution                                            |
| Vitest     | 60    | IPC error normalisation, health screen, video metadata rendering, extraction panel                   |
| Playwright | 15    | UI layout, engine-data rendering, import flow, failure panels                                        |

All 494 pass as of Phase 3. (The counts above are what the suites report today;
earlier revisions of this table understated them.)

The ingestion tests are deliberately split. Parsing logic is tested against
literal ffprobe output and needs no FFmpeg installed, so the rotation and
variable-rate rules are pinned independently of any particular FFmpeg build.
Integration tests then run the real thing against five committed fixture clips —
constant rate, variable rate, two rotations, audio-only — plus corrupt,
truncated, and zero-byte files built at test time.

The rotation convention gets its own guard: a rotated fixture is decoded twice,
once letting FFmpeg auto-rotate and once rotating it in our own code, and the
pixels must be identical. A future FFmpeg that changed the sign fails the suite
instead of quietly transposing every measurement thereafter.

The pose tests follow the same split. Most of them run against a fake estimator
rather than MediaPipe — which is the evidence that `PoseEstimator` is a real
seam, since decode, estimate, persist and reload all run with the model replaced
and nothing else changed. The MediaPipe adapter's own behaviour is tested
separately: that it converts BGR to RGB before inference, that a frame with no
pose is recorded rather than dropped, and that two runs over one clip agree
exactly.

The filtering tests are where numerical claims get checked against something
independent, and they come in three kinds. **Equivalence:** on a uniform grid the
fit must reproduce SciPy's `savgol_filter` for value, velocity and acceleration
to floating-point precision — an independent implementation agreeing to ~1e-14 is
a far stronger statement than a tolerance someone chose. **Exactness:** a
polynomial of degree at most the fit's order must come back perfectly, on
arbitrary non-uniform sampling, derivatives included. **Error bounds:** RMS
limits against trajectories with closed-form derivatives, with the bounds taken
from `scripts/benchmark_filter.py` rather than from whatever the code currently
emits, so a change that halves the accuracy fails rather than passing quietly.

Hypothesis covers the invariants a worked example would miss: adding a constant
shifts position and leaves derivatives alone, shifting all timestamps changes
nothing, and reversing time negates velocity while preserving acceleration —
which is the property that catches a sign error in the local coordinate.

CI installs FFmpeg and downloads the pose models, and sets `GSA_REQUIRE_FFMPEG`
and `GSA_REQUIRE_MODELS` so that a runner missing either **fails** rather than
skipping — a skipped suite and a passing one look identical in a summary.

## 13. Performance benchmarks

Measured on the reference machine. Every figure here came out of a script in
`scripts/`; none is estimated.

**Engine boundary** (2026-09-15):

| Measurement                             | Value   |
| --------------------------------------- | ------- |
| Worker spawn → ready                    | 163 ms  |
| First `doctor` call (framework imports) | 1373 ms |
| `doctor` call, warm (median of 5)       | 127 ms  |
| Components probed per call              | 12      |

That ~11x cold/warm gap is why the engine is a long-lived process rather than
one invocation per call.

**Ingestion** (2026-09-16, `scripts/benchmark_decode.py`, 600 frames of
1920x1080 H.264, median of 5):

| Operation                      | Median  | Frames/s |
| ------------------------------ | ------- | -------- |
| probe (2 ffprobe passes)       | 72.6 ms | —        |
| probe (metadata cache hit)     | 1.6 ms  | —        |
| decode: OpenCV, in-process     | 564 ms  | **1065** |
| decode: ffmpeg subprocess, CPU | 1349 ms | 445      |
| decode: ffmpeg + VideoToolbox  | 2332 ms | 257      |

Hardware decode is the slowest of the three: this pipeline needs BGR frames in
system memory, so a hardware-decoded frame has to be read back off the GPU, and
that transfer costs more than the decode it saved.

**Pose estimation** (2026-09-16, `scripts/benchmark_pose.py`) on two real
swings — a 68-frame 720x1280 face-on clip and a 239-frame 1920x1080
down-the-line clip:

| Model | Load    | ms/frame (face-on / DTL) | Frames/s | Poses found |
| ----- | ------- | ------------------------ | -------- | ----------- |
| lite  | ~190 ms | 11.6 / 11.0              | 86 / 91  | **100%**    |
| full  | ~85 ms  | 17.7 / 17.2              | 56 / 58  | **100%**    |
| heavy | ~120 ms | 67.1 / 66.3              | 15 / 15  | **100%**    |

The default is `full`. All three find a pose in every frame, so detection rate
does not discriminate; speed does, but a swing clip is seconds long, so the
spread is under half a second of work. The criterion that would discriminate —
landmark accuracy — is not measured anywhere yet, and choosing the least
accurate variant to save that half second would be optimising the wrong
quantity.

Measuring this on synthetic footage first was instructive about how wrong the
easy number can be: on a clip with nobody in it, `heavy` measured 27.4 ms/frame
against 67.1 ms on a real swing, because MediaPipe runs the detector when it
finds nothing and the landmark model when it does. The benchmark refuses to
recommend a model below a 50% detection rate for exactly that reason.

**Temporal filtering** (2026-09-16, `scripts/benchmark_filter.py`). Accuracy is
against analytical trajectories with closed-form derivatives, at a landmark noise
level of 0.0014 normalized_frame measured from real footage. Worst case over
three trajectories at 120 fps:

| order | window     | pos RMS     | vel RMS    | acc RMS  | peak speed err |
| ----- | ---------- | ----------- | ---------- | -------- | -------------- |
| 2     | 0.10 s     | 0.00194     | 0.6226     | 13.69    | −8.4%          |
| 3     | 0.10 s     | 0.00195     | 0.0435     | 13.68    | −1.1%          |
| **4** | **0.10 s** | **0.00077** | **0.0437** | **3.29** | **−1.1%**      |
| 4     | 0.125 s    | 0.00072     | 0.0609     | 2.33     | −0.7%          |

That table is how the defaults were chosen rather than a report on them. Degree 2
is not viable — it underestimates peak speed by 8%, an error phase detection
would inherit when locating impact. Degree 4 beats degree 3 on acceleration by
about 4x at the same velocity error.

What assuming uniform sampling costs, same noise level, velocity RMS:

| Sampling        | True timestamps | Assumed uniform | Ratio     |
| --------------- | --------------- | --------------- | --------- |
| uniform         | 0.0308          | 0.0309          | **1.00x** |
| jitter, 50%     | 0.0402          | 0.0567          | 1.41x     |
| rate change, 4x | 0.0836          | 0.8926          | **10.7x** |
| dropped frames  | 0.0637          | 0.5740          | **9.0x**  |

The first row is why the general method is affordable: on genuinely uniform input
it costs nothing measurable. The last two are why it is necessary, and both are
ordinary properties of phone footage.

Throughput: 33 landmarks in three axes takes 12.3 ms for a 68-frame clip and
17.5 ms for a 240-frame clip, against ~1.2 s to extract poses for the same 68
frames. Filtering is not a bottleneck, which is why nothing is cached.

No figures exist yet for metrics, because they do not exist yet. A general
benchmark harness arrives in Phase 17.

## 14. Limitations

- **macOS/Apple silicon only, so far.** Nothing is known to be Windows- or
  Linux-incompatible, but neither has been tested, and the MediaPipe wheel
  pinned here is macOS-arm64.
- **Pose inference is CPU-bound on macOS.** No GPU delegate exists for the
  MediaPipe Tasks Python API on this platform.
- **Playwright does not drive the real WebView.** It runs against the Vite dev
  server with the Tauri IPC bridge stubbed. Real-WebView automation needs
  `tauri-driver` and a platform WebDriver, which is not set up.
- **Engine requests are serialised.** A mutex guards the worker; concurrent
  request multiplexing is not implemented because nothing needs it yet.
- **No packaging story yet.** `npm run dev` runs from the repository and
  resolves the Python project by walking up from the working directory. A
  bundled app needs the engine shipped as a sidecar; that is not built.
- **Ingestion is verified on H.264 in MP4/MOV only.** Other codecs and
  containers are likely to work, since FFmpeg does the demuxing, but nothing
  else has been tested and no claim is made for it.
- **A half-turn rotation is taken on trust.** A 90 or 270 degree rotation is
  verified against the decoded frame's dimensions; 180 degrees changes no
  dimension, so there the decoder's own property read-back is the only evidence
  available.
- **`FFmpegPipeFrameSource` restarts to seek backwards.** It is built for a
  sequential pass. Random access uses the OpenCV source, which is the default.
- **Pose landmark accuracy is not measured at all.** Detection _rate_ is
  reported because it is counted; nothing here says whether the landmarks that
  were found are in the right place. That needs a labelled set, which is
  Phase 12.
- **Filtering needs about 60 fps or better at its default settings.** A 0.10 s
  window with a degree-4 fit needs five samples, and 30 fps supplies three. Such
  a clip gets no values at all, plus a message naming the minimum window its
  measured rate would support — the alternative, widening the window silently,
  produces numbers that are worse in a way nothing reports. Both reference clips
  used during development are 24–30 fps, so this is the ordinary case rather than
  an edge one, and it is the first quantitative backing for the ≥120 fps the
  capture protocol asks for.
- **Filter accuracy is measured against models of swing motion, not a swing.**
  The trajectories in the benchmark have exact derivatives, which real footage
  cannot until Phase 12 provides labelled landmarks. They were chosen to resemble
  swing dynamics; no claim is made that they match one, and the defaults should
  be re-derived against ground truth when it exists.
- **MediaPipe runs on CPU.** The Tasks Python API has no macOS GPU delegate, and
  the health check reports the delegate it measured rather than the one it would
  prefer.
- **The app icon is a placeholder** — a solid colour, not designed art.

## 15. Future work

Phases 1-20: video ingestion, pose extraction, temporal filtering, swing phase
detection, biomechanics metrics, DTL analysis, two-camera synchronisation,
camera calibration, 3D reconstruction, club tracking, ball detection, temporal
ML, the coaching engine, desktop visualisation, 3D rendering, swing comparison,
performance work, and documentation. Sequencing, deliverables, and exit criteria
per phase are in [docs/ROADMAP.md](docs/ROADMAP.md).

## Licence

MIT.
