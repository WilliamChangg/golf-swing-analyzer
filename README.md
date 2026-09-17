# Golf Swing Analyzer

Local-first golf swing biomechanics analysis from face-on and down-the-line
video. Pose and club motion are extracted with computer vision, swing phases are
segmented deterministically, and biomechanics metrics are computed with explicit
units, confidence, and methodology. All processing runs on your machine; video
never leaves it.

> **Status: Phases 0-6 of 21 complete.** The foundation, typed engine boundary,
> environment health check, video ingestion, single-camera pose extraction,
> temporal filtering, swing phase detection, the biomechanics metric engine, and
> explicit coordinate frames with measured camera-view tagging are built and
> verified. No club tracking, 3D reconstruction or coaching exists yet. Sections below marked _Not yet implemented_ say so rather than describing
> features that do not exist. See [docs/ROADMAP.md](docs/ROADMAP.md).

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
docs/                  Architecture, coordinate systems, decisions, roadmap
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

_Partially implemented — Phases 7-13 outstanding._ A clip can be imported,
inspected, run through pose estimation, filtered into trajectories with
derivatives, segmented into swing phases, and measured — the first four from the
app's **Video** screen, and all of them from a terminal:

```bash
uv run --project python analyzer probe   path/to/swing.mov   # container metadata
uv run --project python analyzer extract path/to/swing.mov   # pose landmarks
uv run --project python analyzer extract path/to/swing.mov --model pose_landmarker_lite
uv run --project python analyzer filter  path/to/swing.mov   # smooth + differentiate
uv run --project python analyzer filter  path/to/swing.mov --window 0.15 --polyorder 4
uv run --project python analyzer phases  path/to/swing.mov   # takeaway/top/impact/finish
uv run --project python analyzer metrics path/to/swing.mov   # biomechanics
uv run --project python analyzer metrics path/to/swing.mov --slow-motion 8
```

Extraction writes landmarks to a Parquet file keyed by the video's content, and
reports what it measured: frames processed, how many contained a pose, time per
frame, and the model digest it ran with.

Filtering reads those landmarks back and produces position, velocity and
acceleration per landmark, along with an account of what it refused: detections
below the confidence gate, gaps too long to bridge, and windows without enough
support to fit. Nothing is persisted — filtering a clip costs milliseconds
against seconds of extraction.

Phase detection locates the takeaway, top, impact and finish, and reports a
confidence for each built from three measured factors. A clip with no swing in
it produces no events and an explanation, rather than events at frame zero. The
app's **Swing phases** panel adds a scrubbable timeline for stepping through the
clip frame by frame, which is the only way to actually check whether an event
landed where it should. `scripts/plot_phases.py` writes the same signals as a
plot and a per-frame CSV.

Metrics turn those phases into measured quantities — posture, rotation, hands
and arms, timing — each with a unit, the frames it came from, a decomposed
confidence and a methodology. Metrics that cannot honestly be computed from a
given recording are listed as refusals with the reason, rather than omitted.
`scripts/overlay_metrics.py` draws each value on the frame it was measured from,
which is the only way to tell a correct angle from a plausible one.

The engine methods are `doctor`, `probe_video`, `extract_poses`, `filter_poses`,
`detect_phases` and `compute_metrics`.

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

_Partially implemented — Phases 7-11 outstanding._ Four stages are built; what
sits on top of them is in §10 and §11.

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

**Swing phases.** The first golf-specific layer: everything under it would serve
any moving body. Four events — takeaway, top, impact, finish — are located from
the shape of the hand-speed signal, deterministically and without a model. Each
carries a confidence built from three measured factors (how clearly the signal
singles the instant out, how well the landmark was seen around it, and whether
the frame rate can resolve an event of that duration at all), reported
separately because a single number hides which of them was weak.

Impact is a **kinematic estimate, not an observation** — nothing here sees the
ball or the club — and it is corroborated against an independent signal, the
lowest point the hands reach after the top. On the reference clip the two agree
to within one frame. Phases 10 and 11 replace the corroboration with real
evidence.

Club and ball tracking are not built. Planned stages and their ordering are in
[docs/ROADMAP.md](docs/ROADMAP.md).

## 9. 3D reconstruction methodology

_Not yet implemented — Phases 8-9._ The system will distinguish
`UNCALIBRATED` / `INTRINSIC_ONLY` / `STEREO_CALIBRATED` and will refuse to make
metric-scale claims without stereo calibration.

## 10. Coordinate systems and camera views

Every number here is a measurement taken in some reference frame, and most ways
to get one wrong produce a plausible number rather than an error. Full
conventions in [docs/coordinate-systems.md](docs/coordinate-systems.md).

| Frame          | Units           | y    | Isotropic | Metric | Status              |
| -------------- | --------------- | ---- | --------- | ------ | ------------------- |
| `IMAGE`        | x/W, y/H        | down | **no**    | no     | stored              |
| `FRAME_WIDTHS` | x/W, (H−y_px)/W | up   | yes       | no     | derived on read     |
| `HIP_LOCAL`    | approx. metres  | up   | yes       | approx | stored              |
| `CAMERA`       | metres          | —    | yes       | yes    | **Phase 8, absent** |
| `WORLD`        | metres          | —    | yes       | yes    | **Phase 9, absent** |

`IMAGE` is what a pose estimator emits and a bad frame to measure in, for two
reasons that both fail silently: it is **anisotropic** (x divided by the frame
width and y by its height, so on a 1080x1920 clip the same displacement measures
0.5625 times as much vertically, and a line genuinely at 45° reads as 29.4°) and
its **y points down** (so a missing flip inverts the top of the backswing into
the bottom).

`FRAME_WIDTHS` fixes both, and the conversion happens **once, below the filter**.
That placement is the point. A consumer cannot forget it — before Phase 6 the
correction lived above phase detection, which therefore measured hand travel in
the uncorrected frame — and the derivatives come out right for free, because the
filter is linear and a flip applied to positions before fitting emerges correctly
signed in the velocity.

`CAMERA` and `WORLD` are named without being implemented, which is the one place
this project declares what it cannot do. A later phase should add the
_capability_, not the concept; asking for either today raises an error naming the
phase that supplies it.

**Camera views are a separate question, and the one that decides what a number
means.** The view is measured from the projected width of the shoulder line at
address, in torso lengths — 0.83 on the face-on reference clip against 0.10 on
the down-the-line one. Every metric carries the view it was measured in and an
`interpretation` saying what it corresponds to on the body from there.

Spine tilt is the worked example. The identical computation gives **+4.6° of
lateral side bend** face-on and **+35.5° of forward posture angle** down the
line. Neither is interpretable without knowing where the camera stood.

Metrics a view cannot support are refused rather than relabelled: shoulder turn,
pelvis turn and X-factor down the line, hand depth face-on. An oblique camera is
reported as `UNKNOWN` rather than rounded to the nearer label, and then nothing
is blocked by the view alone — each metric's own conditions decide.

## 11. Biomechanics methodology

Every metric carries a name, a value, a unit, the swing event or phase it was
measured at, the frames it came from, a three-part confidence, and a methodology
in words that is never omitted. It also carries a **basis**, and that field is
the one doing the real work.

Everything measured so far comes from a single uncalibrated camera, which
flattens three dimensions into two. No arithmetic afterwards puts the third one
back. So rather than a disclaimer in a document, the kind of claim a number
represents is a typed field on the contract:

| basis                 | what the number is                                  |
| --------------------- | --------------------------------------------------- |
| `temporal`            | a duration, or a ratio of durations                 |
| `image_plane`         | a distance or speed between two points in the frame |
| `projected_angle`     | the angle between two segments as they appear       |
| `foreshortened_angle` | rotation inferred from how much a segment shortened |

`temporal` is the only one of the four that measures the body rather than a
picture of it. A consumer that wants to say "your shoulders turned 50 degrees"
has to go past a field saying it is a foreshortening estimate. Full reasoning in
[ADR-0010](docs/decisions/ADR-0010-projected-biomechanics.md).

**Rotation is measured, not assumed.** A shoulder line of fixed real width
projects to less as it turns away from the camera, by the cosine of the turn, so
`arccos(span / span_at_address)` recovers the angle from two lengths in the same
image — focal length, distance and sensor size all cancel in the ratio. It
cannot tell a turn from its mirror image, so values are magnitudes; and it is
ill-conditioned near zero, where `dθ/d(span)` goes as `1/sin θ`, which is
exactly why `sin θ` is the confidence factor rather than a number someone chose.

**The method's premise is checked rather than trusted.** Foreshortening needs
the player square to the camera at address. If the body line projects more than
1.25× wider anywhere in the clip than it did at address, that did not happen,
and the rotation metrics are refused with the measured ratio in the reason. On
the reference footage the two cases are not close: 1.12× face-on against 7.2×
down-the-line.

**Confidence is three measured factors and their product**, matching how Phase 4
reports an event:

| factor        | what it measures                                              |
| ------------- | ------------------------------------------------------------- |
| `observation` | reported visibility of the landmarks used, on the frames used |
| `anchor`      | the Phase 4 confidence of the event or phase measured at      |
| `method`      | how sharply the method itself pins the quantity down          |

None of the three is a policy constant. `method` is computed from the data for
each basis: the fraction of a duration that is not frame-rate quantisation, the
fraction of a segment lying in the image plane, or the sine of the angle the
arccos returned. The factors answer _how well this method determined this
quantity as defined_ — not how close it is to an anatomical truth, which is what
`basis` is for. Mixing the two into one scalar would make it uninterpretable,
because "the landmark was blurry" and "a camera cannot see rotation" have
completely different fixes.

**Lengths are in torso lengths**, not pixels or frame widths. A frame width
halves when the camera is moved twice as far away; the player's torso does not.
It is not a metric unit and does not pretend to be one.

**Lead and trail arms are worked out from the video.** Which physical arm leads
depends on handedness, which no pose sequence states, and assuming right-handed
would silently mislabel every left-handed player. At the top of the backswing
the hands sit over the _trail_ shoulder, so projecting their offset from the
chest onto the shoulder line names the side. Down the line that line points
nearly at the camera and the projection is noise — so no side is named and the
lead/trail metrics are refused, which is correct, because that view does not
show it.

Twenty quantities are registered: spine tilt, left and right knee flex, hip
sway, head sway and lift; shoulder turn, pelvis turn, X-factor, shoulder and
pelvis tilt; hand path length, peak hand speed, lead and trail arm angle; and
backswing, downswing, follow-through and takeaway-to-impact durations with the
tempo ratio. Each is declared once, with its unit and basis, so a unit written
next to a computation cannot drift from the one in the documentation.

## 12. Model architecture

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

## 13. Testing

```bash
npm run check:all
```

| Suite      | Count | Scope                                                                                                                      |
| ---------- | ----- | -------------------------------------------------------------------------------------------------------------------------- |
| pytest     | 604   | contracts, environment probes, model verification, dispatch, RPC framing, ingestion, pose, filtering, phases, biomechanics |
| cargo test | 12    | protocol framing, id correlation, `uv`/project resolution                                                                  |
| Vitest     | 73    | IPC error normalisation, health screen, video metadata rendering, extraction panel, swing inspector                        |
| Playwright | 21    | UI layout, engine-data rendering, import flow, failure panels, frame-by-frame phase inspection                             |

All 710 pass as of Phase 6.

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

Phase detection is judged the same way, against synthetic swings built from a
_velocity profile_ that is integrated into a trajectory, so the instants the
detector is meant to find are inputs rather than things read off a plot
afterwards. The hands travel a circular arc rather than a vertical line, because
the simpler fixture gets impact wrong in a way that matters: on a real swing the
hands are at the bottom of their arc and moving horizontally at impact, so peak
speed and lowest position coincide, and a vertical fixture separates them and
makes the corroboration check meaningless. That fixture caught a defect the
reference footage did not — the takeaway search finding the pause at the top of
the backswing instead of the address, which the face-on clip survived only
because its speed at the top was 0.130 against a 0.131 threshold.

The biomechanics tests take the same line one step further: the synthetic body
is written in **plane coordinates** — x rightwards, y upwards, both in frame
widths — and converted backwards into the normalised image coordinates an
estimator would have produced, on a deliberately non-square 1080x1920 frame. The
engine then has to undo the aspect ratio and the y flip to get back to the
numbers the body was built with. A fixture written directly in image coordinates
could not tell a correct engine from one that skipped both.

The expected angles are facts rather than agreements. A knee built at 25 degrees
of flex must read 25; an arm whose elbow sits on the segment between shoulder
and wrist must read 180; an arm whose elbow sits on the circle with that segment
as its diameter must read 90, by Thales' theorem. Turn schedules are held
constant across the instants they are measured at, because a local polynomial
fit reproduces a constant exactly — so the recovered shoulder turn is the
constructed one, not whatever the smoothing of a curve happened to leave.

CI installs FFmpeg and downloads the pose models, and sets `GSA_REQUIRE_FFMPEG`
and `GSA_REQUIRE_MODELS` so that a runner missing either **fails** rather than
skipping — a skipped suite and a passing one look identical in a summary.

## 14. Performance benchmarks

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

**Biomechanics** (2026-09-16, `scripts/benchmark_metrics.py`, median of 9, on
the two reference swings at a 0.15 s window):

| Clip           | Frames | Filter  | Phases | Metrics | Produced | Refused |
| -------------- | ------ | ------- | ------ | ------- | -------- | ------- |
| PW_face-on.mp4 | 68     | 17.8 ms | 0.3 ms | 2.1 ms  | 39       | 1       |
| iron_dtl.mp4   | 96     | 19.9 ms | 0.3 ms | 1.8 ms  | 35       | 3       |

Against ~1.3 s to extract poses for the same clip. Filtering dominates the three
because it fits a polynomial at every sample of 33 landmarks, while the metric
layer reads a few dozen frames of a result already in memory. Nothing downstream
of extraction is cached, for the same measured reason as Phase 3.

The last two columns are the more interesting measurement, and they run both
ways. The down-the-line clip refuses shoulder turn, pelvis turn and X-factor,
because that view does not contain them; the face-on clip refuses hand depth,
because the axis it measures along runs along the target line there and is a
different quantity under the same name. Each clip measures what its camera
position supports and says what it cannot.

A general benchmark harness arrives in Phase 17.

## 15. Limitations

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
- **Impact is inferred from the hands, and runs marginally early.** Hand speed
  peaks slightly before the club reaches the ball, so the reported frame is a
  kinematic estimate with a known bias in a known direction. It is corroborated
  against the lowest point of the hand arc, and Phases 10-11 will replace that
  with club and ball evidence.
- **Rotation is unreliable on a full turn, and now says so.** Once a player
  turns far enough for the far shoulder to pass behind the torso, MediaPipe
  infers its position from a body prior — and reports a visibility of 1.00 while
  doing it. On tour-pro footage the implied shoulder turn varies by 27 degrees
  across frames where the player barely moves. Stability is now measured from
  the geometry and reported as an uncertainty in degrees, and a turn is refused
  above a bound; but the underlying landmark is still a guess, and a wide
  uncertainty is a warning rather than a correction.
- **An uncertainty needs enough frames to measure.** At 30 fps the stability
  window holds three frames, which cannot separate a jumping landmark from a
  turning body, so the uncertainty is reported as unknown. Slow-motion footage
  and high frame rates are where it becomes available.
- **Slow motion has to be declared.** Nothing in a conformed file records the
  playback rate, so a slow-motion clip is refused until `--slow-motion` is
  supplied. The factor is not a relabelling of the clock: the smoothing window
  is in real seconds too, so it decides how many frames that window holds and
  therefore where events land.
- **Impact is estimated worse than its own corroboration, on the one clip
  where that can be checked.** The ball leaves the tee between two known frames
  in the tour-pro face-on clip. Peak hand speed — Phase 4's primary estimate —
  lands 17 to 40 frames away; the lowest point of the hand arc, carried only as
  corroboration, lands within 4 and is stable across the assumed slow-motion
  factor. One clip is not enough to change the estimator, and Phase 11 should
  evaluate it first.
- **Nothing here is metric.** Lengths are in torso lengths and the frame they
  are measured in is the frame's own width. The same swing filmed from twice the
  distance gives the same torso-length numbers and different frame-width ones,
  which is the point — but neither is centimetres, and nothing here can produce
  centimetres until a calibrated camera does in Phase 8.
- **The camera view is inferred from one measurement.** The shoulder line's
  projected width at address separates face-on from down-the-line by a factor of
  eight on the reference clips, which is a wide margin, but it is one signal and
  its thresholds rest on an anatomical assumption — that shoulder width is a
  fairly stable multiple of torso length — rather than on a labelled set.
- **`DOWN_THE_LINE` does not say which end of the target line.** A camera behind
  the player and one in front foreshorten the shoulder line identically. The
  measurable consequences are the same, but the sign of anything measured along
  the frame's horizontal axis is therefore an image direction rather than an
  anatomical one.
- **Nothing in the metric layer sees three dimensions.** Every angle is a
  projection and every distance is an image-plane distance, including the ones
  named after 3D quantities. Motion directly towards or away from the camera
  contributes nothing to any of them. This is recorded per metric in the `basis`
  field rather than left to a reader to remember, but it is a limitation of the
  input and no amount of labelling removes it. Phases 8 and 9 do.
- **Shoulder and pelvis turn are magnitudes with no direction.** Foreshortening
  is a cosine, and a cosine is even, so a turn one way and its mirror image
  produce the same number. Recovering the sign needs depth.
- **Rotation needs a face-on recording, and refuses without one.** The
  foreshortening baseline is the body line's projected width at address, which
  only stands in for its true width if the player was square to the camera
  there. Down-the-line clips fail that and have their rotation metrics refused;
  they keep tilts, posture, hand and timing metrics. The baseline is also only a
  lower bound on the true width in general — a player never quite square to the
  camera has every turn under-reported by an amount nothing here can measure.
- **Metric thresholds are structural, not golf norms.** As in Phase 4, the
  numbers in `MetricConfig` exist to decide when a measurement is too
  ill-conditioned to report, not to describe what a swing should look like.
- **Biomechanics is validated on one swing and one synthetic body.** The
  synthetic fixture pins the arithmetic against angles known by construction,
  and every value on the face-on reference clip was checked against the frame it
  came from with `scripts/overlay_metrics.py`. Neither is ground truth: nobody
  has measured this player's actual shoulder turn. That needs the labelled set
  Phase 12 builds.
- **Phase detection is validated on two swings.** The face-on reference clip is
  the only recording here containing a swing the pipeline can see; the
  face-on and iron down-the-line reference clips both detect cleanly; the driver
  down-the-line clip is 24 fps and loses the wrists to motion blur through the
  part where the swing happens, so it is refused. Every event was checked by
  hand against the signal, which is not the same as being checked against ground
  truth — that needs the labelled set Phase 12 builds.
- **Detection thresholds are structural bounds, not golf norms.** They exist to
  reject motion that cannot be a swing (a two-second descent, hands that never
  travel further than a fraction of the subject's torso), and are deliberately
  loose. Numbers tight enough to describe what a swing _should_ look like would
  need a labelled set.
- **Filter accuracy is measured against models of swing motion, not a swing.**
  The trajectories in the benchmark have exact derivatives, which real footage
  cannot until Phase 12 provides labelled landmarks. They were chosen to resemble
  swing dynamics; no claim is made that they match one, and the defaults should
  be re-derived against ground truth when it exists.
- **MediaPipe runs on CPU.** The Tasks Python API has no macOS GPU delegate, and
  the health check reports the delegate it measured rather than the one it would
  prefer.
- **The app icon is a placeholder** — a solid colour, not designed art.

## 16. Future work

Phases 7-20: two-camera synchronisation, camera calibration, 3D reconstruction, club tracking, ball
detection, temporal ML, the coaching engine, desktop visualisation, 3D
rendering, swing comparison, performance work, model management, test hardening
and documentation. Sequencing, deliverables, and exit criteria per phase are in
[docs/ROADMAP.md](docs/ROADMAP.md).

Phase 6 cleared the debt Phase 5 carried: the aspect correction now happens once
below the filter, so phase detection and the metric layer measure in the same
isotropic frame and their two `torso_length` figures agree. Phase 7 begins the
two-camera work that eventually makes the projections in §11 unnecessary.

## Licence

MIT.
