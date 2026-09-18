# Golf Swing Analyzer

Local-first golf swing biomechanics analysis from face-on and down-the-line
video. Pose and club motion are extracted with computer vision, swing phases are
segmented deterministically, and biomechanics metrics are computed with explicit
units, confidence, and methodology. All processing runs on your machine; video
never leaves it.

> **Status: Phases 0-13 of 21 complete.** The foundation, typed engine boundary,
> environment health check, video ingestion, single-camera pose extraction,
> temporal filtering, swing phase detection, the biomechanics metric engine,
> explicit coordinate frames with measured camera-view tagging, two-camera time
> alignment, camera calibration, multi-view 3D reconstruction, club shaft
> tracking, ball detection, the temporal-ML apparatus and the coaching engine are
> built and verified. **Nothing has been reconstructed, calibrated or
> club-tracked from real footage** — no board capture and no simultaneous
> two-camera recording exists in this repository, and the club figures come from
> a rendered shaft whose angle is an input, so every figure in §9 and §10a is
> synthetic and is a floor. The ball is the exception: impact has been
> **observed** on one real clip, and §10b says what that one clip does and does
> not settle. The coaching engine ships twelve rules of which **nine are refused
> on every recording this system can currently make**, for reasons §12 sets out.
> Sections below marked _Not yet implemented_ say so rather than describing
> features that do not exist. See [docs/ROADMAP.md](docs/ROADMAP.md).

---

## 1. Project overview

The goal is a serious computer-vision system, not a wrapper that forwards video
to a language model. The analysis pipeline produces structured quantitative
evidence; the LLM layer, now built, only rewords that evidence, is off by
default, never sees a frame, and is held to a deterministic guard that discards
any sentence containing a number the evidence does not.

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

_Partially implemented — the desktop UI is Phase 14._ A clip can be imported,
inspected, run through pose estimation, filtered into trajectories with
derivatives, segmented into swing phases, measured, aligned against a second
camera, reconstructed into 3D positions in metres with both cameras calibrated,
and reasoned about. All of it from the app's **Video** screen except project
management, reconstruction and coaching, and all of it from a terminal:

```bash
uv run --project python analyzer probe   path/to/swing.mov   # container metadata
uv run --project python analyzer extract path/to/swing.mov   # pose landmarks
uv run --project python analyzer extract path/to/swing.mov --model pose_landmarker_lite
uv run --project python analyzer filter  path/to/swing.mov   # smooth + differentiate
uv run --project python analyzer filter  path/to/swing.mov --window 0.15 --polyorder 4
uv run --project python analyzer phases  path/to/swing.mov   # takeaway/top/impact/finish
uv run --project python analyzer metrics path/to/swing.mov   # biomechanics
uv run --project python analyzer metrics path/to/swing.mov --slow-motion 8

# Two cameras, one swing.
uv run --project python analyzer sync faceon.mov dtl.mov     # relate their clocks
uv run --project python analyzer sync faceon.mov dtl.mov --slow-motion-target 8
uv run --project python analyzer sync faceon.mov dtl.mov --anchor impact=204:252

# Camera geometry: what a pixel means.
uv run --project python analyzer calibrate board board.png      # one to print
uv run --project python analyzer calibrate camera board.mov --role face_on --square-mm 34.6
uv run --project python analyzer calibrate stereo 1 faceon_board.mov dtl_board.mov
uv run --project python analyzer calibrate show 1               # what may be claimed

# Sessions, which remember which clips belong together.
uv run --project python analyzer project create "Tuesday range"
uv run --project python analyzer project add 1 faceon.mov --role face_on
uv run --project python analyzer project add 1 dtl.mov --role down_the_line --slow-motion 8
uv run --project python analyzer project sync 1                # stored on the project
uv run --project python analyzer project list

# Two calibrated, aligned views: metres, in three dimensions.
uv run --project python analyzer reconstruct 1
uv run --project python analyzer reconstruct 1 --json
uv run --project python analyzer metrics faceon.mov --project 1   # now includes the 3D metrics

# The club. Coverage per swing phase first, because the clip-wide rate misleads.
uv run --project python analyzer club faceon.mov
uv run --project python analyzer club faceon.mov --slow-motion 8
uv run --project python analyzer club faceon.mov --json

# The ball, and the one instant this system can observe rather than infer.
uv run --project python analyzer ball   faceon.mov                # where it was, and when it left
uv run --project python analyzer impact faceon.mov                # every estimate, reconciled
uv run --project python analyzer impact faceon.mov --no-club      # skip the second decode

# Labels, and the learned detector that cannot yet be trained on them.
uv run --project python analyzer label faceon.mov --player rory --session range-01 --labeller wc
uv run --project python analyzer labels                           # the set, and whether it splits
uv run --project python analyzer dataset                          # features, classes, imbalance
uv run --project python analyzer train                            # split, fit, score, register
uv run --project python analyzer models                           # what each may claim

# Conclusions, and the far longer list of things that cannot be concluded.
uv run --project python analyzer coach faceon.mov
uv run --project python analyzer coach faceon.mov --slow-motion 7
uv run --project python analyzer coach faceon.mov --phrase-with http://127.0.0.1:11434/api/generate
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

Reconstruction turns two calibrated, aligned views into 3D positions in metres,
and reports three numbers rather than one: how well the two views agree
(reprojection error), what the capture could possibly determine (the ray
convergence angle), and an independent check the first cannot make (whether the
reconstructed bones keep their length). It refuses by name when the rig is not
stereo, when a calibration does not describe the footage, or when the clocks have
not been related — never at a default offset of zero, which is both a plausible
answer and the wrong one.

Synchronisation relates two cameras' clocks, which everything above two views
depends on: triangulating a point is only meaningful for two views of the same
instant. It reports an offset with an uncertainty, a clock rate only when the
evidence supports one, and — the number that matters most — how far the swing
events in the two clips disagree, measured against the floor the two frame rates
impose. **It cannot tell you that both cameras filmed the same swing**; it aligns
swing-shaped signals, and two different swings align perfectly happily. What they
cannot do is agree about phase durations, so that disagreement survives as
residual, and a residual many times the floor is the system saying so.

Club tracking finds the shaft in each frame and reports it as a **ray from the
hands**: a direction the image determines to about half a degree, and a club-head
position only where the edge evidence ran to the end of the club. It prints
coverage **per swing phase** before anything else, because the clip-wide rate is
dominated by address and the follow-through — where the club is nearly still —
and can be high while the downswing contains nothing. Frames that emit nothing
are counted by reason: no hands to anchor on, no edges, no candidate, an
ambiguous frame, an impossible rotation, or a confidence below the bound.
`scripts/overlay_club.py` draws the tracked shaft on the frames it was found in
**and the refused frames stamped with the reason**, which is the only way to tell
a tracked club from a tracked door frame.

Ball detection is the only thing here that **observes** impact rather than
inferring it, and it does so by watching the ball stop existing. Tracking a ball
in flight is unanswerable on consumer footage — 70 m/s is a metre-long smear at
1/60 s — so this measures the ball where it is easy, at rest, and reads impact
off the edge of that interval: the last frame carrying a ball and the first
carrying none. **The uncertainty is one frame interval and it is a bracket**,
because nothing in that path smooths or fits anything. It reports a refusal for a
practice swing, for a ball that leaves outside the window a strike can happen in,
and for a clip that ends before the absence can be verified — all of which are
correct answers rather than failures.

`analyzer impact` reconciles every estimate a clip supports into **one** instant
with one provenance and one error bar. They are ranked, not averaged: three of
the four are biased in a known direction and one is not, and averaging moves the
answer away from the truth while producing a number nobody can reason about.
Every estimate that answered is kept beside the reported one with its delta,
because those deltas are how the bias of the kinematic estimate gets measured.
Nothing there reaches back down and changes what `analyzer phases` reports.

A **project** records which clips belong to one session and stores their
alignment. It is the only state in this system that cannot be recomputed from the
video, so it lives in `~/Library/Application Support/golf-swing-analyzer`
(`$XDG_DATA_HOME` elsewhere) rather than in the cache, and clips are identified by
content so moving a file does not break the record of what it is.

Coaching is the only layer here that draws a conclusion rather than measuring
one, and most of what it does is decline to. Each rule declares the source of the
threshold it compares against — who published it, on whom, and **by what
method** — and a threshold measured by three-dimensional motion capture is not a
threshold on a rotation this engine inferred from how much a line shortened in
one photograph. Nine of the twelve shipped rules are refused for that reason or
because no number has ever been published for the quantity at all. Every finding
that survives carries the frames its numbers were measured on, and no sentence in
the report — including the engine's own — may contain a number absent from that
evidence. §12 has the numbers and
[ADR-0017](docs/decisions/ADR-0017-borrowed-thresholds-and-the-guard.md) the
reasoning.

The engine methods are `doctor`, `probe_video`, `extract_poses`, `filter_poses`,
`detect_phases`, `compute_metrics`, `coach_swing`, `track_club`, `detect_ball`,
`locate_impact`, `sync_clips`, `sync_project`,
`create_project`, `list_projects`, `get_project`, `delete_project`, `add_clip`,
`remove_clip`, `relocate_clip`, `calibrate_camera`, `calibrate_stereo`,
`get_calibration`, `clear_calibration` and `reconstruct`.

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

_Complete through Phase 12._ Six stages are built; what sits on top of them is
in §10, §11 and §13.

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
on it. Genuinely metric coordinates come from triangulating two calibrated views
of one instant — §9 — and they are camera-centred rather than scene-fixed.

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

Impact is a **kinematic estimate, not an observation** — nothing at that layer
sees the ball or the club — and it is corroborated against an independent signal,
the lowest point the hands reach after the top. Club tracking and ball detection,
below, supply real evidence, and on the one reference clip where the ball can be
seen leaving they settle which of the two hand signals to believe: the arc low
lands within **one frame** of the observation and the speed peak **twenty**.

**Club tracking.** A `ClubDetector` interface with a Canny/Hough implementation
behind it, anchored on the hands the pose layer already found. A shaft is
reported as a **ray from the grip**: an origin supplied by the pose layer, a
direction the image determines to about half a degree, and a club-head position
only where the edge evidence ran to the end of the club — which on real footage
is a minority of frames.

Three numbers are reported rather than one, because the obvious one is misleading
in a predictable direction. Edge support scores how much of the line the image
drew; a **margin** over the best rival says whether anything else in frame fits
as well; and **per-phase coverage** says whether the frames that matter carry a
shaft at all. A stationary door frame through the hands scores the same support
as the club and more length, so evidence alone ranks the background first — and
does so most decisively through the downswing, where the club is the blurriest
thing in the picture and the background is the sharpest. See
[ADR-0014](docs/decisions/ADR-0014-club-evidence-and-coverage.md).

The detector is **stateless** and the tracker runs **offline**, which is the
opposite of the pose seam and is deliberate: a track is seeded where the evidence
is best — address, or the top — and grown outward into the downswing, rather than
starting at frame zero and committing. The predicted direction scores candidates
and never supplies one, so a frame with nothing acceptable emits nothing and
leaves a visible hole.

**Ball detection.** A `BallDetector` interface with a morphological top-hat
implementation behind it, anchored not on the hands but on the **ankles** — a
teed ball rests on the ground, and the pose layer knows where the ground is.
Anchoring on the hands instead says only "within a club length", which on a
vertical phone clip is a region containing the sky, and a patch of cloud between
two branches is rounder, better resolved and stiller than a golf ball a few
pixels across.

What it measures is not the ball but the **frame the ball stops being in the
picture**. A ball in flight is unanswerable at consumer frame rates and a teed
ball is the easiest object in the clip, so the measurement is taken where the
evidence is and impact is read off the edge of it. That gives the property no
other impact estimate here has: an uncertainty of exactly one frame interval,
which is a **bracket** rather than a scale — the ball was present at one end of
it and absent at the other.

Two confidences are reported rather than one, computed from **disjoint** evidence,
because the identification is circular: the ball is picked out of the stationary
candidates partly by the fact that it leaves. One scores a frame's observation
(contrast, a co-located rival, drift); the other scores the instant (how much of
the lead-in carried a ball, how abruptly it went, whether it stayed gone).
Neither reads the other's inputs, and a test asserts it. What survives that
partition is reported rather than solved: if two stationary objects both depart,
nothing in one view says which was struck, and the identification margin falls to
zero. See
[ADR-0015](docs/decisions/ADR-0015-ball-departure-and-impact-precedence.md).

**Reconciling the estimates.** Four now exist across three phases. They are
ranked rather than averaged — an observation beats a measurement of a coinciding
quantity, which beats a proxy, and a proxy with no known bias beats one with a
known bias — and every estimate that answered is kept beside the reported one
with its delta. Those deltas are the point: they are how the bias of a kinematic
estimate becomes a number instead of a caveat.

## 9. 3D reconstruction methodology

Two calibrated rays meet, and the result is metres. On a synthetic swing whose 3D
positions are inputs, filmed by two simulated cameras at 90°, the reconstruction
lands within **5.5 mm at the median and 10.4 mm at the 95th percentile** at the
landmark scatter Phase 3 measured on real footage — a floor rather than an
estimate, since there is no pose estimator in the fixture.

**The number that scores a triangulation is blind to half of the error.** A point
detected in camera 1 defines a ray, and every 3D point on that ray projects into
camera 2 along a single line. Displace camera 2's detection _across_ that line
and no 3D point explains it, so it lands in the residual. Displace it _along_ the
line and a point further up or down the ray explains it perfectly — the fit is
exact and the depth is wrong. Measured, the residual is not merely insensitive
to that but **exactly zero**:

| along-epipolar displacement | 3D error | reprojection | bone variation |
| --------------------------- | -------- | ------------ | -------------- |
| 0 px                        | 0.0 mm   | **0.00 px**  | 2.6%           |
| 2 px                        | 2.7 mm   | **0.00 px**  | 9.6%           |
| 8 px                        | 10.8 mm  | **0.00 px**  | 37.8%          |

For the camera pair this system recommends — one face-on, one down the line — the
epipolar lines run nearly horizontally in both images, and nothing makes a pose
estimator's horizontal error smaller than its vertical one.

So three numbers are reported, each labelled with its own question. **The gate is
the ray convergence angle**, because depth error scales as `1/sin` of it and it
is a property of where the tripods went: closing the cameras from 90° to 8° moves
the error 4.7x and leaves the residual flat at 0.84 px. **The independent check
is bone length**, because a point sliding along its ray changes its distance to
its neighbours and a bone does not change length during a swing — no ground truth
and no anatomical table required. Full reasoning in
[ADR-0013](docs/decisions/ADR-0013-epipolar-blindness.md).

**Phase 8's capture instruction does not survive the subject moving.** Stereo
calibration tolerates unsynchronised cameras because the pairing error is
`sync_error × image_speed` and a board can be held still. Nothing in a swing is,
so pairing nearest frames costs 2.4–12.3 mm at the hands; the target clip is
resampled onto the reference clock instead, by cubic Hermite interpolation of the
position and velocity the Phase 3 filter already fitted.

**What comes out is camera-centred metres, not a scene frame.** A scene-fixed
frame needs a gravity direction and a target line, and a stereo pair supplies
neither. Every length, angle and speed between two reconstructed points is
unaffected, so the six metrics this unlocks — shoulder turn, pelvis turn and
X-factor about the body's own measured spine axis, both true elbow angles, and
peak hand speed in metres per second — need no scene frame. A 3D _spine tilt_
would, and is therefore absent.

`CalibrationStatus` is `none`, `intrinsics` or `stereo`, and the metric layer
gates on it centrally. **Three values rather than two, because a calibrated
camera is not a 3D camera.** A calibrated pixel names a _direction_; how far
along that direction anything sat is exactly what the projection destroyed. Two
such rays intersect and one does not, so `intrinsics` permits removing the lens
from a landmark and permits no metric-scale claim whatever. `apply.bearings`
returns unit vectors for that reason, and nothing in the calibration package
returns a 3D point.

Both prerequisites are gated, and a reconstruction refuses by name when either is
missing.

**Prerequisite one, Phase 7: the two cameras' clocks.** Triangulating a point
from two views is only meaningful for two views of the same instant. An offset
known to 3.4 ms on a 120 fps pair bounds how much of the disagreement between
two views is real parallax and how much is one camera looking a frame later —
the difference between a reconstruction error and a timing error, which are not
separable after the fact.
[ADR-0011](docs/decisions/ADR-0011-affine-time-map.md).

**Prerequisite two, Phase 8: what a pixel means.** Charuco board, intrinsics per
camera, extrinsics between them, all measured and all gated. The gate is the
interesting part, because the obvious one does not work:

| number                 | what it answers                         | gates?             |
| ---------------------- | --------------------------------------- | ------------------ |
| RMS reprojection error | how well the model fits these views     | gross failure only |
| reported σ(fx)         | what the fit says about its own spread  | a backstop         |
| coverage               | what the views could possibly determine | **yes**            |

A board held square to the camera at one distance cannot separate focal length
from distance, so it determines almost nothing — and fits beautifully. Measured,
the residual stays at 0.21–0.27 px while the focal length error moves from 0.05%
to 23%, and the parameter uncertainty the fit reports is _smallest_ where the
answer is worst. Full reasoning in
[ADR-0012](docs/decisions/ADR-0012-calibration-coverage.md).

The two cameras do not need a genlock. What stereo extrinsics actually need is
that nothing moved between the two frames, so the sync uncertainty is multiplied
by the board's measured image speed and reported as a displacement in **pixels**
— the same unit as the reprojection error it would otherwise be mistaken for.
Three frames of stillness per board position is the whole requirement, and that
recovers the baseline to 0.104% and the rotation to 0.031°.

## 10. Coordinate systems and camera views

Every number here is a measurement taken in some reference frame, and most ways
to get one wrong produce a plausible number rather than an error. Full
conventions in [docs/coordinate-systems.md](docs/coordinate-systems.md).

| Frame          | Units           | y    | Isotropic | Metric  | Status           |
| -------------- | --------------- | ---- | --------- | ------- | ---------------- |
| `IMAGE`        | x/W, y/H        | down | **no**    | no      | stored           |
| `FRAME_WIDTHS` | x/W, (H−y_px)/W | up   | yes       | no      | derived on read  |
| `HIP_LOCAL`    | approx. metres  | up   | yes       | approx  | stored           |
| `CAMERA`       | metres          | down | yes       | **yes** | **triangulated** |
| `WORLD`        | metres          | —    | yes       | yes     | **absent**       |

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

## 12. Coaching methodology

The layer that turns measurements into sentences, and the first one where being
wrong is expensive rather than embarrassing. A wrong angle is a wrong number; a
wrong finding is a golfer changing their swing.

Twelve rules ship. **Five can reach a comparison at all**, and seven are refused
before a clip is looked at, because the refusal is about the threshold rather
than about the footage. On the four reference clips only two rules have ever
produced a finding, and both are timing rules.

### Why a borrowed threshold usually cannot be used

Golf coaching runs on numbers with no papers behind them. Ninety degrees of
shoulder turn, forty-five degrees of X-factor, three-to-one tempo. Hard-coded as
`THRESHOLD = 90.0` they look identical, and they are not. So every rule carries a
`ThresholdSource` stating who published the number, on whom, and by what method —
and the method decides what the number may be compared against:

| the source measured with         | it is a threshold on | rules |
| -------------------------------- | -------------------- | ----- |
| three-dimensional motion capture | a `spatial` metric   | 1     |
| video                            | what that camera saw | 3     |
| this clip, against itself        | its own uncertainty  | 2     |
| no stated protocol               | **nothing at all**   | 6     |

A `convention` source permits no basis. That is not an oversight awaiting a
citation: a number with no published measurement protocol has no quantity
attached to it, so there is nothing here it could be a threshold on.

The reference footage shows what the alternative costs. On a tour
professional's face-on clip this engine measures a **shoulder turn of 53.0 ± 5.5
degrees at the top**. The player is not restricted; the measurement is a
foreshortening estimate from one camera, which under-reports by tens of degrees
and says so in its `basis`. A rule comparing that against "about ninety" —
however heavily caveated — tells a tour player to turn more.

Two rules sit either side of the same gap, and the pair is the point:
`rotation.x_factor_top` has a number (from a 1992 magazine article) and no
protocol; `rotation.x_factor_top_3d` has a protocol and no number this project
has read a figure it can cite. Both are in the registry, refused for different
reasons, because a coaching engine that silently omitted the numbers every golf
app displays would look like it had forgotten them.

### What a comparison has to clear

A value is only reported as being on one side of a band if it clears the band by
more than its own **bracket**. Durations get one frame interval. Angles and
distances get whatever uncertainty the metric layer measured, and are refused
where it measured none. The tempo ratio gets a figure of its own, because the
backswing and downswing share an endpoint: one frame of ambiguity at the top
lengthens one and shortens the other at once.

That turns out to decide whether the three usable rules ever fire
(`scripts/benchmark_coaching.py --sweep resolution`, on a 0.800 s backswing over
a 0.233 s downswing — the amateur reference clip exactly):

| fps | bracket on the ratio | distance to the nearer band edge | outcome     |
| --- | -------------------- | -------------------------------- | ----------- |
| 30  | 0.738                | 0.371                            | **refused** |
| 60  | 0.341                | 0.371                            | reported    |
| 120 | 0.164                | 0.371                            | reported    |
| 240 | 0.081                | 0.371                            | reported    |

**At 30 fps a swing's tempo cannot be compared against the published band at
all.** The band is 1.37 wide and one frame at the top is worth 0.74 of it; the
crossover for this swing is 56 fps. The number every golf app puts on its front
page is not resolvable by the camera most of them are pointed at. The source is
more careful than its readers: Tour Tempo publishes frame counts — 18/6, 21/7,
24/8 — and not a ratio, because 3:1 exactly is a ratio of two integers rather
than a measurement with a resolution.

A related gate refuses absolute durations on any clip whose slow-motion factor
was supplied rather than measured, which is every slow-motion clip: nothing in a
conformed file records it. The **ratio** survives, because a factor stretching
both durations equally divides out of their quotient — the same argument that
puts lengths in torso lengths, one dimension over.

### What four real clips produced

`scripts/benchmark_coaching.py --sweep clips`:

| clip                     | view          | ms/frame | findings | refused | fired                        |
| ------------------------ | ------------- | -------- | -------- | ------- | ---------------------------- |
| amateur, face-on, 30 fps | face_on       | 33.3     | **0**    | 12      | —                            |
| amateur, DTL, 30 fps     | down_the_line | 33.3     | 2        | 10      | tempo.ratio, tempo.downswing |
| tour, face-on, 7x slow   | face_on       | 4.8      | 1        | 11      | tempo.ratio                  |
| tour, DTL, 7x slow       | down_the_line | 4.8      | 1        | 11      | tempo.ratio                  |

Every finding on every clip is temporal. Nothing measured in the image plane
produced a comparison this engine would stand behind, from either camera
position, on either golfer.

The tour clip's surviving finding reports a tempo of **1.83:1** against a tour
band of 2.43–3.80. The numbers are right and the conclusion is not about the
swing: the finding cites the backswing, the downswing and the frames each came
from, and following those frames leads to the takeaway, which this project had
already flagged as suspect on slowed footage. That is the argument for citing
evidence. **A finding that carries its frames can be discovered to be wrong; a
sentence of advice cannot.**

### The guard, and the language layer under it

No sentence in a report may contain a number its evidence does not. The guard
checks numerals to the precision they are written at, spelled-out numbers,
units this system cannot produce, units it can produce but did not here,
quantities nothing in the project measures, claims about where the ball went, and
causal claims. Over a corpus of fourteen plausible inventions it rejects all
fourteen while keeping three faithful rewordings, at 34 µs per candidate.

It runs over the **engine's own sentences** in the test suite as well as over any
model's. That is not symmetry for its own sake — the first thing it ever caught
was a rule template that said "the two measurements", where the spelled-out "two"
was a quantity absent from the evidence. The template was reworded rather than
the guard relaxed.

The phrasing layer is off by default, and off is a complete configuration: every
finding already has a sentence. Turned on, it talks to a model on this machine —
loopback is enforced in code, not documented as a convention — and receives the
findings as JSON with no frame, no landmark and no file path in it. A candidate
with one offence is discarded whole, not repaired, and the engine's own sentence
ships. **No language model has ever phrased a finding in this repository**, because
none is installed on the machine it was built on; every path is exercised against
fakes, and `PhrasingReport` carries the counts that would make a later run
evidence rather than an impression.

**There is no score.** No severity, no grade, no ranking between findings, and a
test asserts the absence of those field names. A single number summarising a
swing would be the most quoted output of this system and the one with the least
behind it: it would need a scale relating degrees of turn to seconds of tempo,
and nobody has measured one.

## 13. Model architecture

**No model is trained on real swings, and none can be.** Phase 12 built the whole
apparatus — labelling tool, versioned features, player-grouped splits, a TCN
baseline, metrics, a model registry and a comparison against the rule-based
detector — and then ran it on the footage this project has:

```
$ analyzer labels
No labels yet. Phase 12's machinery is built and has nothing to run on.
```

The four reference clips are one golfer. A train/validation/test split needs
players to hold out, so the splitter refuses, and labelling all four clips would
not change that. **The input this is short of is people, not footage.**

What exists, in `python/analyzer/ml/`:

| Piece                 | What it is                                                              |
| --------------------- | ----------------------------------------------------------------------- |
| `labeltool.py`        | A pure state machine, a renderer, and a thin OpenCV window around them  |
| `contracts/labels.py` | Who marked it, how sure they were, and whose swing it is — all required |
| `features.py`         | 10 channels at 60 Hz, torso-normalised, versioned by a hash of the spec |
| `splits.py`           | Grouped by **player**; leakage measured on the output, not asserted     |
| `tcn.py`              | 19,525 parameters, six dilated layers, a 2.12 s receptive field         |
| `evaluate.py`         | Per-event error in ms, and the gates that refuse to publish it          |
| `registry.py`         | Weights plus the card that says what they mean; feature digest enforced |

**The honesty gate lives in the type.** `EvaluationReport.claims_permitted` is
computed rather than asserted by a caller, and it is false whenever the set is
synthetic, has too few held-out players or clips, or the measured error is smaller
than the labels' own uncertainty. That last one is the failure that looks like
success: a detector cannot be shown to be more accurate than the numbers it is
being scored against.

Two figures set the ceiling on what a learned detector can be worth here. The
60 Hz feature grid costs **8.3 ms** on the placement of any event, before a
convolution has run; Phase 11 brackets impact to **4.2 ms** at 240 fps by watching
the ball stop being there. On the one quantity both can report, the observation
wins before the comparison starts. See
[ADR-0016](docs/decisions/ADR-0016-labels-groups-and-the-noise-floor.md).

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

## 14. Testing

```bash
npm run check:all
```

| Suite      | Count | Scope                                                                                                                                                                                                                                                                                                                                                                     |
| ---------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pytest     | 1,255 | contracts, environment probes, model verification, dispatch, RPC framing, ingestion, pose, filtering, phases, biomechanics, time alignment, project storage, camera calibration, 3D reconstruction, club tracking, ball detection, impact fusion, labelling, features, splits, training, evaluation, model registry, coaching rules, evidence linking, the phrasing guard |
| cargo test | 12    | protocol framing, id correlation, `uv`/project resolution                                                                                                                                                                                                                                                                                                                 |
| Vitest     | 99    | IPC error normalisation, health screen, video metadata rendering, extraction panel, swing inspector, two-camera alignment, calibration review                                                                                                                                                                                                                             |
| Playwright | 34    | UI layout, engine-data rendering, import flow, failure panels, frame-by-frame phase inspection, alignment flow, calibration review                                                                                                                                                                                                                                        |

All 1,400 pass as of Phase 13.

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

The club tests split along the seam rather than along the feature. The detector
is tested against **rendered frames** whose shaft angle is an input, so an
angular error is a subtraction; the tracker is tested against **constructed
candidates**, so a test can state exactly what the image offered — a club and a
door frame of equal support, a rotation no club could make, a gap in the middle
of a swing — and assert what is done with it. Driving those cases through a
renderer would mean tuning pixels until the detector produced the situation under
test, which tests the fixture.

The ones that matter most assert a **negative**, as in Phases 8 and 9: that a
stationary background line through the hands scores the same edge support as the
club and more length, so ranking by evidence puts a door frame first; that a
smeared shaft stops being a line rather than becoming a weaker one, so no
threshold recovers it; and that two equally good readings of the same pixels are
refused rather than resolved by a coin toss the confidence would then describe as
a measurement. Those fail if anyone later decides the Hough score is the
confidence.

The ball tests split the same way and add a third layer, because this phase can
be wrong in a way no per-frame assertion reaches. The detector runs against
rendered pixels; the tracker against constructed candidate positions; and an
end-to-end suite renders a swing, detects every frame and compares the located
instant with the frame the ball was removed at. That comparison is an **equality
of integers** — the fixture draws the ball in every frame before a departure that
is an input and in none after — so there is no tolerance to argue about, and it
is the only place a mistake in the units or the frame conversion would surface.

Almost every assertion here is about a refusal or a gate, because the failures
are all confident: that a ball which comes back was occluded rather than struck;
that a departure before the top is not an impact; that a one-frame speck in the
follow-through cannot move the answer by fifty; that a clip ending too soon scores
the check it could not make as **not made** rather than as passed. One of them
pins the phase's central structural claim directly — halving the contrast of
every frame in a clip must not move the departure's confidence at all, because
the two confidences are computed from disjoint evidence and a version that wired
them together would be an argument with itself.

CI installs FFmpeg and downloads the pose models, and sets `GSA_REQUIRE_FFMPEG`
and `GSA_REQUIRE_MODELS` so that a runner missing either **fails** rather than
skipping — a skipped suite and a passing one look identical in a summary.

## 15. Performance benchmarks

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

### Two-camera synchronisation

`scripts/benchmark_sync.py`. **Against known offsets, not against ground truth.**
No simultaneous two-camera recording exists in this project, so what is measured
is one synthetic swing sampled by two simulated cameras with the offset as an
input. That establishes the method recovers what was put in; it does not
establish that a real pair works, because the simulated cameras see the same
projection of the same body and two real cameras do not.

Median absolute error over 9 seeds, at the landmark noise measured from real
footage in Phase 3:

| Pair          | Frame-rate floor | Recovered offset error |
| ------------- | ---------------- | ---------------------- |
| 240 + 240 fps | 1.7 ms           | **0.5 ms**             |
| 120 + 120 fps | 3.4 ms           | **0.4 – 4.2 ms**       |
| 120 + 30 fps  | 9.9 ms           | **0.3 ms**             |
| 30 + 30 fps   | 13.6 ms          | **0.5 ms**             |

The floor is what the two frame rates permit for a single located instant. The
correlation beats it because it averages hundreds of samples; anchoring on the
four swing events instead gives 14–27 ms on the same pairs, which is what decided
the design. Aligning two 312-frame clips costs 0.5 ms.

### Camera calibration

`scripts/benchmark_calibration.py`. **A floor, not an estimate.** The camera is
an input — a synthetic one at fx = 1400 px on a 1920x1080 frame with k1 = −0.28
— and a Charuco board is rendered through it and detected by the real detector.
What the renderer does not contain is everything that makes real calibration
footage hard: motion blur, rolling shutter, defocus, JPEG ringing on the marker
borders, and a printed sheet that has bowed off flat.

The headline sweep, over the one thing a person controls. Median of 3 seeds, 14
board views each:

| tilt spread | RMS px | reported σ(fx) | **true fx error** |
| ----------- | ------ | -------------- | ----------------- |
| ±0°         | 0.220  | 0.012%         | **6.33%**         |
| ±2°         | 0.210  | 0.012%         | **10.04%**        |
| ±10°        | 0.269  | 1.22%          | **23.25%**        |
| ±20°        | 0.245  | 0.656%         | 0.10%             |
| ±35°        | 0.218  | 0.314%         | **0.05%**         |

The reprojection error is flat across a range over which the focal length error
moves by a factor of four hundred, and the uncertainty the fit reports about
itself is thirty times _smaller_ where the answer is worst. That is why coverage
is the gate: it is the only one of the three that describes the capture rather
than the fit. [ADR-0012](docs/decisions/ADR-0012-calibration-coverage.md).

Frame coverage and the distortion model, measured as error at the frame edge —
the quantity a landmark actually inherits:

| capture         | model   | fx err | error at the frame edge |
| --------------- | ------- | ------ | ----------------------- |
| corners reached | rt4     | 0.05%  | **2.1 px**              |
| corners reached | rt5     | 0.05%  | 9.5 px                  |
| corners reached | pinhole | 9.64%  | 202.3 px                |
| centred only    | rt4     | 0.14%  | 34.4 px                 |
| centred only    | rt5     | 0.13%  | **180.6 px**            |

OpenCV's fifth distortion coefficient is worth a factor of four at the edge on a
good capture, and a factor of five in the other direction on a poor one, because
a term the data does not determine takes whatever value cancels the residual
where the board was and diverges where it was not. Four is the default.

What the lens does to a landmark, which is what a calibration buys a
single-camera user:

| lens                  | at the frame edge | worst  | h-fov |
| --------------------- | ----------------- | ------ | ----- |
| phone main (k1 −0.28) | 171.5 px          | 199 px | 68.9° |
| phone wide (k1 −0.42) | 219.9 px          | 248 px | 93.7° |
| mild (k1 −0.10)       | 34.0 px           | 41 px  | 56.1° |
| long lens (k1 −0.02)  | 2.5 px            | 3.0 px | 35.5° |

On the reference swing `data/face-on/PW_face-on.mp4`, undistorting with a
plausible phone lens moves the trail arm angle at the top by +1.19°, shoulder
turn at impact by +0.87°, and peak hand speed by 3.0% — and moves **no swing
event**, because phase detection reads the shape of a speed curve and a smooth
radial correction does not move its extrema. That lens is assumed rather than
measured, so the table says how large the correction is on real footage, not
what the corrected numbers are.

Stereo extrinsics, both cameras at 30 fps with their clocks related to 12 ms:

| board             | pairing error | usable pairs | baseline error | rotation error |
| ----------------- | ------------- | ------------ | -------------- | -------------- |
| held 1 s          | 0.00 px       | 7            | **0.104%**     | **0.031°**     |
| held **3 frames** | 0.00 px       | 7            | 0.104%         | 0.031°         |
| held 2 frames     | —             | 2            | refused        | —              |
| never stops       | —             | 0            | refused        | —              |

Three frames of stillness — a tenth of a second at 30 fps — is the entire
requirement, and the images are equally sharp in every row.

Cost: 3.7 ms per frame to detect the board at 1920x1080, and 13.5 ms to fit
intrinsics from 14 views.

### Club tracking

`scripts/benchmark_club.py`. **Against a shaft whose angle is an input, not
against real footage.** The club is drawn by `tests/synthetic_club.py` at a known
direction and smeared by a declared exposure, so an angular error is a
subtraction. There is no defocus, no compression, a flat background, and a club
1.1 torso lengths long where a real one is about 2.5 — a longer lever smears
proportionally more. **Every rate below is a floor.**

Detection against the length of the smear, one frame at a time:

| smear at the club head | shaft found | support | angle error |
| ---------------------- | ----------- | ------- | ----------- |
| 0.0 px                 | **100%**    | 1.00    | 0.44°       |
| 4.8 px                 | **100%**    | 1.00    | 0.88°       |
| 9.6 px                 | **100%**    | 1.00    | 0.90°       |
| 14.4 px                | **0%**      | —       | —           |
| 28.8 px                | **0%**      | —       | —           |

A cliff, not a decline. Where the club is found the angle is right to about half
a degree; where it is not, no threshold recovers it.

Coverage per swing phase at 120 fps, against the exposure as a fraction of the
frame interval — **the table this phase exists to produce**:

| shutter | max smear | overall | address | backswing | **downswing** | follow | err p95 |
| ------- | --------- | ------- | ------- | --------- | ------------- | ------ | ------- |
| 0.03    | 0.8 px    | 100%    | 100%    | 99%       | **100%**      | 100%   | 0.63°   |
| 0.125   | 3.5 px    | 99%     | 100%    | 99%       | **100%**      | 98%    | 0.69°   |
| 0.25    | 6.9 px    | 93%     | 100%    | 98%       | **83%**       | 73%    | 0.79°   |
| 0.5     | 13.9 px   | 86%     | 100%    | 93%       | **65%**       | 62%    | 1.15°   |
| 1.0     | 27.7 px   | 64%     | 100%    | 58%       | **25%**       | 23%    | 1.18°   |

The address column is 100% in every row, so it carries no information about the
capture at all — and it is a large share of the overall figure. That is why
coverage is reported per phase and the aggregate is documented as the number not
to read alone.

**The shutter is the lever; the frame rate is not.** Holding the exposure fixed
at 1/500 s rather than at a fraction of the interval:

| fps | 1/4 shutter: smear / overall / downswing | 1/500 s: smear / overall / downswing |
| --- | ---------------------------------------- | ------------------------------------ |
| 30  | 27.6 px / 62% / —                        | 6.6 px / 96% / —                     |
| 60  | 13.8 px / 87% / 62%                      | 6.6 px / 97% / **92%**               |
| 120 | 6.9 px / 92% / 81%                       | 6.7 px / 96% / **90%**               |
| 240 | 3.5 px / 100% / 99%                      | 6.7 px / 96% / **88%**               |

Once the shutter is fixed, four times the frames buy nothing for club detection.
The 30 fps rows have no downswing column because no swing is detected at all
there — Phase 3's window floor, reappearing rather than anything about the club.

Background clutter, against what a per-frame detector would do — `evidence only`
takes the best-supported candidate, which is what a Hough transform ranks by:

| vertical background lines | tracker: kept / wrong | evidence only: kept / wrong |
| ------------------------- | --------------------- | --------------------------- |
| 0                         | 289 / **0**           | 296 / **0**                 |
| 1                         | 285 / **0**           | 294 / **4**                 |
| 2                         | 285 / **0**           | 294 / **7**                 |
| 3                         | 277 / **0**           | 297 / **15**                |
| 4                         | 279 / **0**           | 298 / **17**                |

And the case it does not handle, which is the honest end of the table:

| occluder    | tracked | downswing | wrong | **conf. when wrong** | head seen |
| ----------- | ------- | --------- | ----- | -------------------- | --------- |
| nothing     | 93%     | 79%       | 0     | —                    | 97%       |
| upper third | 54%     | 44%       | 18    | **0.98**             | 8%        |
| upper half  | 47%     | 31%       | 4     | **0.81**             | 97%       |
| a wide band | 64%     | 38%       | 0     | —                    | 100%      |

A rectangular occluder's boundary is sharp, stationary and — once the club is
hidden behind it — unopposed, so the tracker steps onto it and continuity keeps
it there. Those frames score the same confidence as the correct ones. See
[ADR-0014](docs/decisions/ADR-0014-club-evidence-and-coverage.md).

Cost: 6.2 ms per frame to detect, over a search region of 3.2 torso lengths which
on this fixture is the whole 1000x1000 frame, and 4.0 ms to track a 312-frame clip
once the candidates exist. Against ~17 ms a frame to extract the poses that have
to come first. Nothing is cached, for the same measured reason as Phase 3.

### Ball detection and impact

`scripts/benchmark_ball.py`, against a ball drawn in every frame before a
departure frame that is an **input** and in none after — so the error in the
located instant is a subtraction of integers. No defocus, no compression, flat
turf where a range is a thousand bright blades, and a ball drawn as a uniform
disc where a real one is a lit sphere. **A floor, not an estimate** — with one
exception, below, which is a real measurement on real footage.

**What the frame rate buys, and what it does not.** The instant is located
exactly at every rate; what changes is only the width of the bracket around it:

| fps | located | error | bracket     | Phase 4's window | ratio |
| --- | ------- | ----- | ----------- | ---------------- | ----- |
| 30  | exact   | 0     | **33.3 ms** | 100 ms           | 3x    |
| 60  | exact   | 0     | **16.7 ms** | 100 ms           | 6x    |
| 120 | exact   | 0     | **8.3 ms**  | 100 ms           | 12x   |
| 240 | exact   | 0     | **4.2 ms**  | 100 ms           | 24x   |

The right-hand columns are not the same kind of claim. The ball's bracket is an
interval impact is **inside**; Phase 4's is the width of the smoothing window its
frame rate forced, which is a scale the peak could have moved on.

**The shutter decides nothing here**, which is the sharpest contrast with the
club: located exactly at every exposure from 0.03 of the frame interval to a full
360° shutter, because a ball at rest is not moving and no exposure smears it.
What replaces it as the capture variable is contrast against the surface, and
that is a cliff in the same way the club's blur is:

| ball vs turf | located   | coverage |
| ------------ | --------- | -------- |
| 155 levels   | **exact** | 100%     |
| 105          | **exact** | 100%     |
| 75           | refused   | 22%      |
| 35           | refused   | 0%       |

**What it refuses, and why each refusal is the right answer:**

| case                          | outcome                  | what says so       |
| ----------------------------- | ------------------------ | ------------------ |
| practice swing, ball stays    | no instant               | nothing departed   |
| ball rolls off before the top | no instant               | outside the window |
| clip ends 2 frames after      | instant, confidence 0.06 | permanence 0.06    |
| a rival that departs too      | instant, margin **0.00** | warned             |

And the case it gets wrong, which is the honest end of the table. At impact the
club head is at the ball, so a covered ball and a departed one are the same
picture; it only matters when the covering starts _before_ contact:

| ball dimmed over | located   | abruptness | warned |
| ---------------- | --------- | ---------- | ------ |
| 0 frames         | **exact** | 1.00       | no     |
| 2                | −1 frame  | 0.91       | no     |
| 6                | −3 frames | 0.80       | yes    |
| 10               | −5 frames | 0.70       | yes    |

The instant runs early by about half the covering. Reported rather than fixed.

**The first observed impact in this project, and the only figure here that is
not synthetic.** `data/face-on/rory_face_on.mp4` at a factor of 7 — the ball is
bracketed between frames 360 and 361, with no warnings:

| source           | frame | delta vs the observation | uncertainty    |
| ---------------- | ----- | ------------------------ | -------------- |
| `ball_departure` | 361   | **reported**             | 5 ms (bracket) |
| `hand_low`       | 362   | +5 ms / **+1 frame**     | 100 ms (scale) |
| `hand_speed`     | 381   | +95 ms / **+20 frames**  | 100 ms (scale) |

That settles a question the roadmap had left open since Phase 6. The lowest point
of the hand arc lands within **one frame** of the observation; peak hand speed —
Phase 4's primary estimate — lands twenty frames away, and **late**, which is the
opposite of the direction the physics predicts. The fusion's precedence follows
the measurement, and Phase 4's own output is left unchanged: a number that moved
depending on which other analyses had run could not be compared across clips.

**One clip is not a correction.** A bias measured once is an anecdote with a
number attached, and there is still no labelled set to settle it. Phase 12 built
the tool that produces one and the gates that refuse a claim without one; what it
found is that this project has a single golfer in it.

Cost: 16 ms per frame to detect over 1.5 torso lengths of ground, and 34 ms to
track a 312-frame clip once the candidates exist. A **rectangular** structuring
element is what makes that affordable — OpenCV decomposes that one into separable
passes and no other shape, and an elliptical element of the same size costs 66 ms
a frame on its own.

### The learned detector

`scripts/benchmark_ml.py`, on a **generated** corpus of players, sessions and
swings whose events are inputs to the generator. Nothing here is a statement
about golf, and the script prints that caveat under every table it produces.

**What the feature grid costs before a model runs.** A label's frame is mapped
onto the 60 Hz grid and read back; the error in that round trip is a floor under
anything working on the grid:

| clip fps | grid  | floor  | round trip | worst      |
| -------- | ----- | ------ | ---------- | ---------- |
| 30       | 60 Hz | 8.3 ms | 0.0 ms     | 0.0 ms     |
| 60       | 60 Hz | 8.3 ms | 0.0 ms     | 0.0 ms     |
| 120      | 60 Hz | 8.3 ms | 4.6 ms     | **8.3 ms** |
| 240      | 60 Hz | 8.3 ms | 4.0 ms     | **8.3 ms** |

Compare the table above it: at 240 fps a ball departure brackets impact to
**4.2 ms**, and the resampling here spends **8.3 ms** before a convolution has
run. On the one quantity both can report, the observation wins before the
comparison starts.

**The held-out number has not settled**, which is the measurement that matters
most on a corpus this size (`--sweep players --seeds 3`):

| players | clips | macro F1 | top MAE | impact MAE |
| ------- | ----- | -------- | ------- | ---------- |
| 3       | 18    | 0.869    | 35.0 ms | 130.0 ms   |
| 4       | 24    | 0.942    | 16.1 ms | 7.2 ms     |
| 6       | 36    | 0.939    | 21.7 ms | 8.3 ms     |
| 9       | 54    | 0.957    | 15.6 ms | 5.3 ms     |
| 12      | 72    | 0.974    | 11.4 ms | 3.9 ms     |

Still improving at twelve players. A score that moves with the size of its own
corpus is measuring the sample, not the method.

**What a leaky split is worth could not be measured.** Same corpus, same seeds,
same architecture; the only difference is whether whole players are held out
(`--sweep leakage --seeds 5`; ± is half the range across seeds):

| quantity     | by player     | by clip (leaky) | difference | beats the scatter? |
| ------------ | ------------- | --------------- | ---------- | ------------------ |
| macro F1     | 0.972 ±0.007  | 0.973 ±0.013    | +0.001     | no                 |
| takeaway MAE | 25.5 ±15.0 ms | 25.6 ±9.0 ms    | −0.1 ms    | no                 |
| top MAE      | 14.3 ±7.1 ms  | 12.2 ±6.9 ms    | +2.1 ms    | no                 |
| impact MAE   | 4.5 ±3.3 ms   | 6.6 ±6.2 ms     | −2.1 ms    | no                 |
| finish MAE   | 14.0 ±8.8 ms  | 12.8 ±11.5 ms   | +1.2 ms    | no                 |

The corpus cannot answer the question: its golfers differ by six generator
parameters, so a model that has seen seven of them has seen the space. The split
is grouped by player on the argument — a swing and its near-duplicate cannot sit
on opposite sides of a question — and not on this table.

**Cost**, 72 clips on the reference machine: 37 ms per clip to filter and
featurise, 82 ms per clip to train 40 epochs, 0.7 ms per clip to run. Training
the entire corpus costs less than extracting poses from one clip, which is the
correct shape for this phase — the expensive thing was never the model.

### The coaching engine

`scripts/benchmark_coaching.py`. Two of its four sweeps measure the registry
rather than a recording, and those are the phase's result.

**What the rules can conclude, before a camera is switched on** (`--sweep
inventory`):

| verdict                 | rules | what would change it    |
| ----------------------- | ----- | ----------------------- |
| usable                  | 5     | —                       |
| no measurement protocol | 4     | somebody publishing one |
| no published number     | 3     | somebody publishing one |

**The capture rate a tempo comparison needs** (`--sweep resolution`), held at the
amateur reference clip's own swing — 0.800 s over 0.233 s, tempo 3.43 — and
varying only the clock:

| fps | bracket | to the nearer band edge | outcome         |
| --- | ------- | ----------------------- | --------------- |
| 30  | 0.738   | 0.371                   | **cannot tell** |
| 60  | 0.341   | 0.371                   | reported        |
| 120 | 0.164   | 0.371                   | reported        |
| 240 | 0.081   | 0.371                   | reported        |

Crossover at **56 fps**. The bracket is what one frame of ambiguity at the top is
worth, and the top is shared between the two durations, so it moves the ratio far
more than it moves either of them.

**The engine over every reference clip with an extraction** (`--sweep clips`):
zero findings on the 30 fps face-on amateur clip, two on the 30 fps
down-the-line one, one on each tour clip. All four findings across all four clips
are temporal.

**The guard** (`--sweep guard`): 14 of 14 plausible inventions rejected, 3 of 3
faithful rewordings kept, **34 µs** per candidate. Four of the fourteen are
number offences and two of those contain no digits — "ninety degrees" and
"forty-five degrees" — because the invention this most needs to catch is the one
golf coaching quotes in words.

**Cost**: `coach()` over twelve rules is **0.05 ms**, against seconds per clip
for pose extraction. Nothing here is worth caching, which is what Phases 3 and 5
concluded about the layers below it.

A general benchmark harness arrives in Phase 17.

## 16. Limitations

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
- **Nothing can tell that two clips show the same swing.** Synchronisation
  aligns swing-shaped signals, and two _different_ swings align just as happily.
  What they cannot do is agree about phase durations, so the disagreement
  survives as a residual — and that residual is the only evidence on the
  question. On the only two-angle pair in this repository it is 40x the
  frame-rate floor, which is the correct verdict and is reported as a confidence
  of 0.10.
- **No calibration has ever been checked against a real camera.** Every
  published calibration figure comes from a synthetic camera whose parameters
  were inputs to the renderer that produced the board images. That establishes
  the _shape_ of the relationship the gate rests on — that the reprojection
  error is blind to a degeneracy and coverage is not — and it does not establish
  what any real phone will achieve. The renderer has no motion blur, no rolling
  shutter, no defocus and no sheet bowed off flat, so every error published is a
  floor. `benchmark_calibration.py --real <dir>` runs the same pipeline over real
  board footage and scores nothing, because there is nothing to score it against.
- **A calibration is silently invalidated by anything but a resolution change.**
  It belongs to a camera at one zoom, one lens and one capture setting. Frame
  size is the only part of that a video file records, and a mismatch there is
  refused; a different zoom, a switch to the ultra-wide, or digital
  stabilisation being left on leave no trace at all and produce a plausible
  wrong answer. `--notes` records what the file cannot, which is a convention
  rather than a check.
- **Calibrating one camera does not make it see depth, and the word
  "calibrated" invites believing it does.** A calibrated pixel is a direction.
  `CalibrationStatus` keeps `intrinsics` and `stereo` apart for that reason and
  the metric layer gates on it. The gate now blocks six metrics: on a project
  with one camera calibrated, every quantity whose basis is `spatial` is refused
  by name, because two rays are what make a point and one does not.
- **Neither camera may move between the board capture and the swing**, and
  nothing detects that one was nudged. The extrinsics describe where the cameras
  stood; a bumped tripod makes every triangulated point wrong with no symptom.
- **Stereo pairing can alias.** Board positions spaced regularly in time, with a
  clock offset wrong by exactly that spacing, pair each frame with its
  neighbour: simultaneous to the millisecond and showing the board in two
  different places. The fit's residual catches it, which is why stereo gates on
  reprojection error where intrinsics deliberately do not — but the pairing's own
  numbers look perfect.
- **Synchronisation has never been checked against real ground truth.** That
  needs two cameras that genuinely filmed one swing at once, with their clocks
  related by something outside this system — a clapperboard, a flash, a genlock.
  The published figures are recoveries of offsets that were put in
  synthetically, which is a weaker claim and is labelled as one.
- **Two clips share one smoothing window, so the coarser sets it.** Smoothing
  them differently would shift the features an alignment keys on by an amount
  nothing measures. A 30 fps camera therefore cannot support the 0.10 s default
  for either clip; the engine refuses and names the window that rate would
  support.
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
  were found are in the right place. That needs a labelled set. Phase 12 built
  the apparatus for one — a labelling tool, a schema, player-grouped splits — and
  no such set exists.
- **Filtering needs about 60 fps or better at its default settings.** A 0.10 s
  window with a degree-4 fit needs five samples, and 30 fps supplies three. Such
  a clip gets no values at all, plus a message naming the minimum window its
  measured rate would support — the alternative, widening the window silently,
  produces numbers that are worse in a way nothing reports. Both reference clips
  used during development are 24–30 fps, so this is the ordinary case rather than
  an edge one, and it is the first quantitative backing for the ≥120 fps the
  capture protocol asks for.
- **`analyzer phases` reports impact from the hands, and that has not changed.**
  It is a kinematic estimate and its contract says so. `analyzer impact` is where
  club and ball evidence now reconcile the four available estimates into one
  instant with a named provenance; nothing there reaches back down and rewrites
  Phase 4, because a number that moved depending on which other analyses had run
  could not be compared across clips.
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
- **The bias of the kinematic impact estimate is measured on exactly one clip.**
  It is the only clip in this repository where the ball can be seen leaving, and
  there peak hand speed lands **20 frames (95 ms) late** while the lowest point
  of the hand arc lands within one. That is enough to have set the fusion's
  precedence and it is not enough to be a correction — a bias measured once is an
  anecdote with a number attached, and no labelled set exists to turn it into
  more than one.
- **The ball is identified by persistence, not by looking like a ball.** At the
  size a golf ball actually occupies on consumer footage — 4.4 px in radius on
  the reference clip — shape does not separate it from grass texture and
  compression blocks, which score anywhere from 0.11 to 0.71 on the same
  circularity measure the ball scores 0.59 on. What identifies it is that one
  position holds a candidate for hundreds of consecutive frames and then stops.
  That works, and it means a **stationary object that is not a ball and does
  vanish** — a tee marker picked up, a second ball also struck — is a rival one
  view cannot resolve. `EstablishedBall.margin` falls to zero when it happens and
  the report warns, which is a disclosure rather than a fix.
- **The ball is observed stopping being visible, not being struck.** For a struck
  ball those are the same instant. They come apart when something covers the ball
  before contact — the club head crossing the line of sight on a down-the-line
  view — and then the reported instant runs early by about half the covering.
  Measured on the fixture: a ball dimmed over ten frames reads five frames early,
  with the abruptness factor at 0.70 and two warnings. Recorded rather than
  fixed.
- **A clip that ends shortly after contact cannot verify the absence.** Nothing
  at the instant itself distinguishes a ball that left from one something moved
  in front of; only the absence lasting does. Two frames of follow-through scores
  that check at 0.06 rather than passing it.
- **Ball detection has been run on one real clip.** It found the impact there
  exactly, and six defects in this phase were found by that clip rather than by
  the fixture — including a search region that contained the sky and a drift
  reference that mistook camera motion for the ball leaving. Every one of them
  produced a confident wrong answer rather than an error, which is the honest
  characterisation of what a second clip might still find.
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
  has measured this player's actual shoulder turn. That needs a labelled set,
  which Phase 12 built the apparatus for and which does not yet exist.
- **Phase detection is validated on two swings.** The face-on reference clip is
  the only recording here containing a swing the pipeline can see; the
  face-on and iron down-the-line reference clips both detect cleanly; the driver
  down-the-line clip is 24 fps and loses the wrists to motion blur through the
  part where the swing happens, so it is refused. Every event was checked by
  hand against the signal, which is not the same as being checked against ground
  truth — that needs a labelled set, which does not yet exist.
- **Detection thresholds are structural bounds, not golf norms.** They exist to
  reject motion that cannot be a swing (a two-second descent, hands that never
  travel further than a fraction of the subject's torso), and are deliberately
  loose. Numbers tight enough to describe what a swing _should_ look like would
  need a labelled set.
- **Filter accuracy is measured against models of swing motion, not a swing.**
  The trajectories in the benchmark have exact derivatives, which real footage
  cannot supply without labelled landmarks. They were chosen to resemble
  swing dynamics; no claim is made that they match one, and the defaults should
  be re-derived against ground truth when it exists.
- **MediaPipe runs on CPU.** The Tasks Python API has no macOS GPU delegate, and
  the health check reports the delegate it measured rather than the one it would
  prefer.
- **Every published club figure comes from a rendered shaft whose angle was an
  input.** The two tour-pro clips have been tracked and checked by eye — which is
  how the yardage-sign failure below was found — but nobody has labelled a shaft
  angle in a real frame, so there is nothing to score a real run against. The renderer
  has no defocus, no rolling shutter, no compression, a flat background where a
  driving range is not, and a club 1.1 torso lengths long where a real one is
  about 2.5 — a longer lever smears proportionally more. The rates are a floor.
- **Motion blur is a hard limit with no processing fix.** A shaft smeared past
  about ten pixels is not a weaker line, it is not a line: an exposure draws a
  rotating club as a fan with no edge in it. Where the club is found the angle is
  right to about half a degree, and where it is not, no threshold recovers it.
  There is deliberately no sensitivity setting, because there is nothing to
  trade. The remedy is a faster shutter, and it is the capture's to supply.
- **The exposure cannot be measured, so the blur cannot be.** Nothing in a video
  file records the shutter. What is reported is the club head's measured image
  speed times the frame interval, which is the smear at a 360-degree shutter and
  therefore an upper bound; a shutter n times faster divides it by n, and which
  it was is not recoverable.
- **A clip's overall club detection rate overstates the downswing, always.** The
  club is nearly still at address and through the follow-through, which are most
  of a clip, and fastest through the downswing, which is where every metric worth
  computing lives. Measured, 64% overall can be 25% of the downswing. Coverage is
  therefore reported per phase and the aggregate is documented as the number not
  to read alone.
- **A background line through the hands is a better line than the club.** A door
  frame, fence post or window mullion scores the same edge support and more
  length, so evidence alone ranks it first — and it is the only candidate left
  once the club blurs. What separates them is that it does not rotate, so the
  tracker's temporal check catches it where a per-frame detector cannot: on the
  synthetic swing with four background lines, picking the strongest line gets 17
  frames wrong and the tracker gets none, keeping 20 fewer frames to do it.
- **An occluder's own edge defeats all three checks, and the confidence does not
  see it.** A rectangular block over the top of the frame has a long straight
  boundary, and where that boundary runs near the hands it is sharp, stationary
  and — once the real club is hidden behind it — unopposed. The tracker steps onto
  it and continuity keeps it there, because a stationary line agrees perfectly
  with a prediction extrapolated from two frames already on it. Measured, 18 of
  172 tracked frames are wrong and they carry a confidence of **0.98** against
  the 0.99 a correct frame carries. No threshold here separates them. **The real
  footage found it on the first run**: on `rory_face_on.mp4`, nine of 139 tracked
  frames follow the vertical edge of the yardage sign behind the player rather
  than the club, at a median confidence of 0.98. A rule that the shaft must
  rotate during the backswing would catch it and is a golf norm, which needs a
  labelled set; so would a detector that has seen a half-occluded club. Phase 12
  built what is needed to produce one and did not produce one.
- **The club head is usually not observed, and the shaft direction usually is.**
  A detected segment stops where the edge evidence stops, which is short of the
  club head whenever the head is smeared — and also whenever the club points at
  the camera and is genuinely short in the picture. One view cannot separate
  those, so a club-head position is emitted only where the evidence reached the
  end of the club and `reaches_head` says which frames those are. **There is no
  club-head speed metric**, and that is why: it would need a length the image
  frequently does not contain.
- **No lens correction is applied to club tracking**, even where the project has
  a calibration. The detector searches the raw frame, so undistorting the hand
  anchor alone would point the search at a place in the image where the hands are
  not. Correcting it properly means undistorting the frame, and it matters for a
  second reason a landmark does not have: a straight club in the world is a
  _curved_ line in a distorted image, so the straight-line model a Hough
  transform rests on is itself violated near the frame edge.
- **The club-based impact estimate does not move Phase 4's answer.** It is the
  lowest observed club-head position after the top, it is independent of hand
  speed, and it is refused outright unless the downswing was more than half
  tracked — which on consumer footage it usually is not. Fusing the two is
  Phase 11's job, once the ball supplies a third piece of evidence that can
  arbitrate.
- **There is no labelled set, so there is no accuracy figure for anything.** Not
  for the pose landmarks, not for the rule-based phase detector, not for the
  learned one. The engine refuses to produce one rather than producing a weak
  one: `EvaluationReport.claims_permitted` is false on every set this project can
  currently build, with the reason attached.
- **The learned detector has only ever seen generated swings.** Its held-out
  scores are measured against events that are inputs to the generator that drew
  the motion, which makes them a test of the plumbing and nothing else. Worse,
  the comparison against the rule-based detector is rigged in the model's favour
  on that corpus: the model is trained on the labels, so it learns the convention
  they were made with, while the rules brought their own — and the two genuinely
  differ at the takeaway by several frames.
- **What a leaky split costs could not be measured.** Training under a
  player-grouped split and a clip-random one on the synthetic corpus produces no
  difference larger than the seed-to-seed scatter, because its golfers differ by
  six generator parameters and holding one out asks nothing. The split is grouped
  by player on the argument, not on that measurement.
- **The labelling window has never been driven by a person here.** Its state
  machine, overlay and key mapping are tested; the OpenCV loop around them is
  forty lines with no decisions in it and no test, because it needs a display.
- **A dataset of mixed frame rates is smoothed inconsistently.** The feature grid
  equalises the sample rate and cannot equalise the filter: a 30 fps clip needs a
  167 ms window where a 120 fps clip uses 100 ms, and a wider window flattens the
  velocity peak. `DatasetSummary` reports it rather than averaging over it.
- **Nine of the twelve coaching rules cannot fire, and no recording changes
  that.** Six rest on a number with no published measurement protocol, and three
  name a quantity for which nothing has published a threshold in any unit. This
  is a fact about what golf instruction publishes, not about this engine, and it
  is reported per rule rather than by omitting the rule.
- **At 30 fps no tempo comparison is possible at all.** One frame of ambiguity at
  the top is worth 0.74 of a band that is 1.37 wide. A phone recording at its
  default rate produces a coaching report with zero findings on it, which is the
  correct answer and an unsatisfying one.
- **A finding compares against a population, not against a target.** Tour Tempo's
  band describes tour professionals. A swing outside it is a swing unlike theirs,
  which is not the same as a swing that is working badly, and nothing in this
  system measures the second thing.
- **The phrasing layer has never met a language model.** No local model is
  installed on the machine this was built on, so the protocol, the request shape,
  the guard integration and every failure path are exercised against fakes.
- **The guard cannot tell whether a sentence is about the right finding.** A
  model given two findings could describe the first using the second's numbers
  and pass every check, because every number would be in the evidence. This is
  why `Finding.observation` is never replaced and the model's version sits beside
  it. Its spelled-out-number table is also finite and starts at two: "one" is
  excluded because in English it is usually a pronoun, so "one degree of tilt"
  gets through.
- **Most same-clip comparisons refuse for want of an uncertainty.** Phase 6
  quantifies uncertainty for the foreshortened rotations and for nothing else, so
  a rule asking whether the spine angle changed from address to impact is refused
  even though both values were measured. The quantity is measurable; what is
  missing is a measurement of how well.
- **The app icon is a placeholder** — a solid colour, not designed art.

## 17. Future work

Phases 14-20: desktop visualisation, 3D rendering, swing comparison, performance
work, model management, test hardening and documentation. Sequencing,
deliverables, and exit criteria per phase are in
[docs/ROADMAP.md](docs/ROADMAP.md).

Phase 7 began the two-camera work that makes the projections in §11 unnecessary,
and settled the first of the two things a second view needs: the relation between
the cameras' clocks, with the error carried rather than assumed away. Phase 8
settled the second — where the cameras were. Phase 9 spends both, and the six
metrics in §9 are the first numbers here that describe a body rather than a
picture of one.

Five open items carry forward:

- **Nothing here has been reconstructed from real footage.** It needs two
  calibrated cameras that filmed one swing at once, and no such recording exists
  in this repository. Every figure in §9 is synthetic and is a floor.
- **Synchronisation has never been checked against a genuinely simultaneous
  pair**, for the same reason.
- **A scene frame is one capture away.** Laying the calibration board on the
  ground with an edge along the target line would supply the vertical and the
  target line that a stereo pair does not, and with them a 3D forward spine
  tilt. `data/README.md` now asks for that footage; nothing reads it yet.
- **Phase 4 is the limit on 3D metrics under noise, not the triangulation.**
  Measured: at 5 px of landmark scatter the reconstruction is still accurate to
  millimetres and the phase detector declines to call the clip a swing, so there
  is no instant to anchor a rotation to. Phase 12 did not revisit it: a learned
  detector trained on generated swings says nothing about how either behaves
  under real landmark noise.
- **A labelled set needs more golfers, not more footage.** Phase 12 built the
  labelling tool, the schema, the player-grouped splitter and the gates that
  refuse a claim without one, and then found that this repository contains a
  single player. Three people would produce a split; eight would allow a number
  to be quoted. `data/README.md` says what to record and asks for consent in
  writing before anyone else's swing enters a training set.
- **What the coaching engine is short of divides cleanly, and only half of it is
  a recording.** Its three timing rules would fire on ordinary footage shot above
  about 60 fps, which is a capture away. `rotation.x_factor_top_3d` needs a
  stereo calibration _and_ a citable range for the quantity it triangulates, and
  the first of those is the same missing two-camera session as everything above.
  The six convention rules need neither: they need somebody to publish a
  measurement protocol for a number the sport has quoted for thirty years, and no
  phase of this project can supply one.

## Licence

MIT.
