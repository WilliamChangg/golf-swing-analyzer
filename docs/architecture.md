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
coordinates.py  reference frames and the conversions between them
hashing/        content digests for verification and cache keys
paths/          filesystem layout resolution
progress/       reporting from long-running methods
environment/    hardware, tooling, model probing
ingestion/      container inspection and frame decoding
pose/           landmark estimation, storage, per-landmark series
filtering/      smoothing, gap policy, derivatives
phases/         swing event detection
sync/           relating two cameras' clocks to each other
calibration/    what a pixel means: the lens, and where the cameras stand
reconstruction/ where two calibrated rays meet: metres, in three dimensions
club/           the shaft: a ray from the hands, and when to emit nothing
biomechanics/   measured metrics, with units, confidence and methodology
projects/       the clips of one swing, and their stored alignments (SQLite)
dispatch/       method registry
worker, cli     entry points
```

Later phases add `coaching` as a sibling package, and a `BallDetector` seam
following `ingestion`'s `FrameSource`, `pose`'s `PoseEstimator`, `filtering`'s
`FilterStage` and `club`'s `ClubDetector`. Golf-specific reasoning is confined to
`phases`, `sync`, `club`, `biomechanics` and `coaching`; everything below is
general computer vision that would serve any moving body. `sync` and `club` are
the marginal members of that set. `sync`'s mechanism — an affine time map and a
masked cross-correlation — would align any pair of recordings of anything, and
only its choice of signal (hand speed, and the four named swing events) is
golf-specific; `club`'s would follow any rigid rod held in a pair of hands, and
only the assumption that there is one is about golf.

`projects` sits apart from the rest, because it holds the only state here that
cannot be recomputed. Everything else is either input the user already has or a
derived artifact under `cache_dir()` that exists to save time; a project records
a _decision_ — these two clips are one swing, filmed from here and from there —
which nothing can recover from the files once it is lost. It therefore lives
under `data_dir()`, which backup tools do not skip.

`calibration` sits beside `coordinates` rather than above it, and is optional
in the same way `sync` is: nothing below it requires a calibration, and
everything above it is better with one. It is general computer vision — a
Charuco board and a lens model know nothing about golf — and it is the only
package below `reconstruction` whose output is a statement about the physical
world rather than about a picture of it.

`reconstruction` is the only package with **three** prerequisites, and it refuses
by name when any is missing: a stereo calibration (`calibration`), an alignment
between the two clips' clocks (`sync`), and a filtered trajectory in each
(`filtering`). That makes it the one place in the engine where the optional
layers stop being optional. It is still general computer vision — nothing in a
triangulation knows what a golf swing is — and it is the layer every earlier
phase has been qualifying its output against.

`coordinates` sits below everything and owns the one conversion the whole system
depends on. IMAGE space, as a pose estimator emits it, is anisotropic and has y
pointing downward; both are corrected exactly once, on read in `pose/series.py`,
so every layer above measures in FRAME_WIDTHS. Putting it there rather than in
the biomechanics layer is deliberate on two counts: phase detection sits below
biomechanics and would otherwise measure in the uncorrected frame, and the
filter is linear, so a conversion applied to positions before fitting emerges
correctly signed in the velocity and acceleration rather than needing a second
correction kept in step by hand. See
[coordinate-systems.md](coordinate-systems.md).

`biomechanics` has one entry point, `compute_metrics(filtered, phases, config,
calibration, reconstruction)`. It measures the **camera view** before anything
else, because a projected number is only interpretable once the direction it was
taken from is known, and tags every metric with it — the same spine tilt is
lateral side bend face-on and forward posture angle down the line. Metrics a view
cannot support are refused centrally rather than per family, so a family added
later cannot forget the check.

Its fifth argument is the reason the layering above is worth reading twice.
`compute_metrics` measures **one clip**, and a reconstruction is a fact about
**two**, so it arrives as an argument rather than being computed here. Absent, the
spatial metrics are refused by name; on a project without a stereo rig they are
refused one step earlier by the calibration gate, whose reason names the
calibration rather than the missing reconstruction — the more actionable of the
two answers, and so the one that wins.

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

## Two-camera synchronisation

Two cameras keep two clocks and neither knows about the other. Phases 8 and 9
triangulate points from two views, and triangulation is only meaningful for two
views of the _same instant_, so this layer is a prerequisite for everything above
it — and the error in its answer propagates into every reconstructed point, which
is why that error is carried rather than assumed away.

```
target_s = reference_s + offset_s + (rate - 1) * (reference_s - pivot_s)
```

**The pivot is the anchor centroid, and that is load-bearing.** It makes the two
fitted parameters uncorrelated, so the offset's and the rate's standard errors
can be reported separately and combined in quadrature. Quoted at time zero they
would be strongly correlated and two independent-looking error bars would
overstate the error near the anchors and understate it far from them.

**The rate is refused by default.** A rate fitted from four instants spanning a
second and a half carries a fractional error of about a frame divided by the
span; applied ten seconds from the anchors that is worse than assuming the two
clocks agree, which for real hardware they do to a small fraction of a percent.
It is fitted only over a long enough baseline and kept only when it sits more
than two standard errors from 1.0 — otherwise it is describing noise, and paying
for it with an uncertainty that grows with distance from the anchors while a
constant offset has none.

**Two estimators, each used for what it can determine.** A masked normalised
cross-correlation of the two hand-speed signals produces one number, an offset,
from several hundred samples. An affine fit to the paired swing events produces
an offset, a rate and a residual from four instants. They are never averaged —
averaging two estimates that disagree yields a third matching neither — and the
split between them was decided by measurement rather than by argument:

| Quantity | Comes from  | Why                                                |
| -------- | ----------- | -------------------------------------------------- |
| offset   | correlation | 0.4–1.8 ms error against 14–27 ms for four anchors |
| rate     | events      | a single lag cannot express one                    |
| residual | events      | a single lag has nothing left over to disagree     |

The events lose the offset because they are not four equally good clocks. Under
the landmark noise Phase 3 measured from real footage, the top moves 8 ms, impact
117 ms, the finish 350 ms and the takeaway 542 ms — the last two are threshold
crossings on a signal that is barely moving there.
[ADR-0011](decisions/ADR-0011-affine-time-map.md).

**Nothing may claim to beat the frame-rate floor.** An instant located to the
nearest frame carries a uniform error one interval wide, standard deviation
`interval / sqrt(12)`, and two clips contribute one each in quadrature. Every
reported uncertainty is the larger of the observed scatter and that floor, so
anchors that happen to agree better than their frame rates allow are reported at
the frame rates' limit rather than at their own lucky rounding.

**What this layer cannot tell you** is that the two clips show the same swing. It
aligns swing-shaped signals and will align two different swings just as happily;
what it cannot do is make their phase durations agree. On the only two-angle pair
in this project the residual is 40x the floor and the confidence is 0.10, which
is the correct reading of two different swings — and the honest limit of what a
pair of uncalibrated recordings can be asked.

## Camera calibration

The first layer that turns a picture back into a statement about space, and the
one whose central finding is a negative.

**The number every calibration tool prints is blind to the failure that
matters.** RMS reprojection error says how well the model fits the board views it
was given. A board held square to the camera at one distance cannot separate
focal length from distance — a longer lens further away makes the same picture —
so such a capture determines almost nothing, and fits beautifully, because the
views it fits are exactly the views it was free to fit. Measured, the residual
stays at 0.21–0.27 px across a range over which the focal length error moves from
0.05% to 23%.

**The principled-looking replacement fails backwards.** OpenCV returns a standard
deviation for every intrinsic, propagated through the fit's Jacobian, and the
design here assumed that would catch it. It is _smallest_ where the answer is
worst — 0.012% on a capture wrong by 6% — because the distortion coefficients
absorb the degeneracy and leave a tightly determined wrong answer. A covariance
computed from one set of views cannot see outside them.

So three numbers are reported and each is labelled with the question it answers:

```
rms_reprojection_px   how well the model fits the data          the fit
fx_uncertainty        what the fit says about its own spread    a check
CoverageReport        what the views could possibly determine   the gate
```

`usable` rests on the third. [ADR-0012](decisions/ADR-0012-calibration-coverage.md).

**A calibrated camera is not a 3D camera**, and `CalibrationStatus` has three
values rather than two so that nothing can read it as one. `INTRINSICS` means the
lens can be removed from a landmark — worth 171 px at the frame edge on an
ordinary phone, inherited by every angle and distance measured above it — and
means a pixel is a known _direction_. It is not a position: the distance along
that direction is exactly what the projection destroyed. `apply.bearings` returns
unit vectors for that reason, and there is deliberately no function here that
returns a 3D point. Two of those rays meet, and that is Phase 9.

```
board footage
  └─ detect        Charuco corners, per frame
      └─ select    views that differ from the ones already kept
          └─ fit   intrinsics, held to a named distortion model
              └─ judge   coverage decides; the residual catches gross failure
```

**Stereo extrinsics do not need a genlock, and the reason is measured.** Two
cameras must see the board at the same instant, and two phones do not share a
clock. What actually matters is that nothing moved between the two frames — so
the pairing error is converted into the unit it contaminates by multiplying
Phase 7's `TimeMap.uncertainty_at` by the board's observed image speed there,
giving a displacement in **pixels** directly comparable with the reprojection
error. A still board makes that term zero however badly the clocks are known.

Measured, three frames of stillness is the whole requirement: a tenth of a second
at 30 fps recovers the baseline to 0.10% and the rotation to 0.03°, and a board
that never stops yields no usable pairs at all. The images are equally sharp
either way, which is why the system measures this rather than advising it.

The one failure a clock cannot catch is aliasing — board stations a second apart
with an offset wrong by exactly a second pair each frame with its neighbour,
simultaneous to the millisecond and showing the board in two different places.
The fit's residual catches that, so stereo _does_ gate on reprojection error:
there it is measuring a correspondence rather than a model's fit to its own data.

## Multi-view 3D reconstruction

Two calibrated rays meet, and the result is metres. This is the layer every
earlier phase has been qualifying its output against — and, like the calibration
below it, its central finding is a negative.

**The reprojection residual is blind along the epipolar line.** A point detected
in camera 1 defines a ray, and every 3D point on that ray projects into camera 2
along a single line. Split camera 2's detection error into two components: the
part _across_ that line has no 3D explanation and lands in the residual, and the
part _along_ it is explained perfectly by a point further up or down the ray. The
first is visible, the second is invisible, and the second is the one that moves
the answer in depth.

Measured, it is not merely insensitive but exactly blind — displacing every
landmark along its epipolar line moves the reconstruction by a centimetre at a
residual of 0.00 px throughout:

| along-epipolar displacement | 3D error | reprojection | bone variation |
| --------------------------- | -------- | ------------ | -------------- |
| 0 px                        | 0.0 mm   | **0.00 px**  | 2.6%           |
| 2 px                        | 2.7 mm   | **0.00 px**  | 9.6%           |
| 8 px                        | 10.8 mm  | **0.00 px**  | 37.8%          |

For the pair this system recommends — one face-on, one down-the-line — the
epipolar lines run nearly horizontally in both images, and nothing makes a pose
estimator's horizontal error smaller than its vertical one. Roughly half the
error by variance lands where the residual cannot see it.

So three numbers are reported and each is labelled with its own question, the
same shape `CalibrationQuality` uses one layer down:

```
reprojection_px    how well the two views agree, across the epipolar line
convergence_deg    what the capture could possibly determine        the gate
bone variation     an independent check the residual cannot make
uncertainty_m      what the first two imply, in metres          the answer
```

**The gate is the ray convergence angle**, because depth error scales as `1/sin`
of it and it is a property of where the tripods went rather than of the fit. A
sweep from 90° to 8° of camera separation moves the error 4.7x and leaves the
residual flat at 0.84 px. [ADR-0013](decisions/ADR-0013-epipolar-blindness.md).

**Bone length is the independent check**, because a point sliding along its ray
changes its distance to its neighbours, and a bone does not change length during a
swing. It needs no ground truth and no anatomical table: the _variation_ across a
clip is error whatever the absolute value is.

```
two filtered clips
  └─ pairing       resample the target onto the reference clock
      └─ triangulate   DLT, then Gauss-Newton on reprojection error
          └─ gate      convergence, then residual, then uncertainty
              └─ skeleton   bones and symmetry, on what survived
```

**Phase 8's capture instruction has no analogue here.** Stereo calibration
survives unsynchronised cameras because the pairing error is `sync_error x
image_speed`, and a board held still drives the second factor to zero. Nothing in
a swing is still: hands reach thousands of pixels per second, so pairing nearest
frames costs 2.4–12.3 mm at the hands depending on the frame rate. The target
clip is therefore **resampled** onto the reference clock by cubic Hermite
interpolation of the position and velocity Phase 3 already fitted — no new
kernel, no second smoothing pass — and what survives is the map's own uncertainty,
converted into pixels the same way Phase 8 converted it.

**What comes out is CAMERA, not WORLD.** Triangulation gives metres centred on
the reference camera. A scene-fixed frame needs a gravity direction and a target
line, and a stereo pair supplies neither; both would fall out of a capture that
laid the board flat on the ground, which the protocol does not currently ask for.
Every length, angle and speed between two reconstructed points is unaffected,
because none of them depends on the frame — which is why the spatial metrics are
rotations about the body's own measured spine axis rather than about a vertical.

## Club tracking

The first layer that measures an **object the player is holding** rather than the
player. Everything below reads landmarks a model was trained to find, which
arrive with the model's own opinion of how well it saw them. A golf shaft has no
model, no landmark index and no reported visibility. It has edges, and the whole
of this layer is about what edges can and cannot be asked.

```
video frame + hand anchor
  └─ roi        a disc of 3.2 torso lengths around the hands
      └─ canny  thresholds from the region's own median, not fixed
          └─ hough_p   line fragments, not shafts
              └─ geometry  which fragments could be a club somebody is holding
                  └─ track   one per frame, or a named reason there is none
```

**A shaft is a ray from the hands: an origin, a direction, and often no length.**
Those three are not equally trustworthy and the contract keeps them apart. The
origin is supplied by the pose layer, the direction is determined well — half a
degree, wherever the club is found at all — and how far along the ray the club
_ends_ is frequently not determined at all. That is Phase 8's split in a
different medium: `apply.bearings` returns unit vectors and offers no function
turning one into a point, because a calibrated pixel is a direction and the
distance along it is what the projection destroyed. Here the distance is what
motion blur destroyed, and `reaches_head` says whether the question was answered.

Two things shorten a segment and one view cannot separate them: the head was
smeared, or the club was pointing at the camera and is genuinely short in the
picture — the same foreshortening Phase 5 measures rotation from. A stereo pair
would separate them.

**Three numbers, each labelled with the question it answers**, which is the shape
Phases 8 and 9 both arrived at because the failure is the same shape again:

```
support        how much edge evidence backs this line        the fit
margin         whether anything else in frame fits as well   a check
phase coverage whether the frames that matter have any       the gate
```

**Support is not the quality, and through the downswing it is anti-correlated
with correctness.** A Hough transform prefers whatever is longest, straightest
and sharpest. The club is the fastest thing in the frame and therefore the
blurriest; the door frame behind the player is stationary and stays sharp. So
evidence ranks the background _above_ the club exactly where the club matters
most — measured, a door frame through the hands scores the same support and more
length than a sharp club, and is the only candidate left once the club smears.

**The aggregate detection rate hides the downswing**, and that is the second
finding rather than a restatement of the first. Detection is easy where the club
is slow, and address plus the follow-through are most of a clip:

| shutter         | overall | address | **downswing** |
| --------------- | ------- | ------- | ------------- |
| 1/4 of interval | 93%     | 100%    | **83%**       |
| 180°            | 86%     | 100%    | **65%**       |
| 360°            | 64%     | 100%    | **25%**       |

The address column carries no information about the capture at all.
[ADR-0014](decisions/ADR-0014-club-evidence-and-coverage.md).

**The detector is stateless and the tracker is offline**, which is the opposite
of `PoseEstimator` and is what Phase 4's nested searches buy one layer along. A
frame-by-frame tracker starts at frame zero and commits; this one runs over a
clip that already exists, so it seeds where the evidence is **best** — address,
or the top — and grows outward into the downswing, where a forward-only tracker
would arrive carrying whatever it had picked up on the way. Keeping the detector
unaware of the tracker's belief also removes the failure that makes classical
trackers untrustworthy: a search narrowed by the current belief finds what it
expects and confirms it.

**The prediction scores candidates and never supplies one.** That is Phase 3's
gap rule in a different medium: a value produced where there is no observation is
an invention however smooth it looks, so a frame with nothing acceptable leaves a
hole. Most of what this layer produces on consumer footage is holes.

**Blur is a capture bound with no processing fix.** A smeared shaft does not
become a weaker line, it stops being one — found in 100% of frames up to about
9 px of smear and 0% past 14, with the angle right to half a degree either side
of nothing. So there is no sensitivity setting that trades detection rate against
accuracy, because there is nothing to trade. What the system reports instead is
`max_blur_px`: the club head's measured image speed times the frame interval,
which is the smear at a 360° shutter and an upper bound on what the clip actually
carried, since nothing in a video file records the exposure.

`club` sits with `phases`, `sync` and `biomechanics` on the golf-specific side of
the line drawn above, and it is the second marginal member of that set for the
reason `sync` is the first: its mechanism would follow any rigid rod held in a
pair of hands, and only the assumption that there is one is about golf.

## Projects

The first state in this engine that cannot be recomputed, and the reason
`data_dir()` exists alongside `cache_dir()`. SQLite rather than a directory of
JSON for three reasons in descending order of weight: a sync cannot name a clip
outside its project because a foreign key says so; deleting a project takes its
clips and alignments in one statement and cannot half-fail; and a write is atomic
against a reader in another process, which the desktop app and a terminal running
the CLI are.

A clip is identified by **content**, not by path. The same footage cannot enter
one project twice under two names, and a file that has been moved still matches
its own cached extraction — `ProjectClip.exists` reports whether the path still
resolves, because a project with a moved clip is correct and incomplete rather
than corrupt.

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
measured. Phase 8 extends it to calibration, where the rule needed sharpening:
a capability must be gated on evidence about the _capture_, not on the estimator's
opinion of its own fit. Both of the numbers a calibration reports about itself
pass a capture whose focal length is wrong by tens of percent. Later phases apply
the same rule to detector confidence gating club and ball output.

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

### Synchronisation (Phase 7, 2026-09-17)

`scripts/benchmark_sync.py`. **Not ground truth**: one synthetic swing sampled by
two simulated cameras, with the offset as an input. No simultaneous two-camera
recording exists in this project, so what this establishes is that the method
recovers an offset that was put in — not that it recovers one from a real pair,
whose two cameras see two different projections of the same body.

Median absolute error over 9 seeds, at the sigma = 0.0014 frame widths Phase 3
measured from real footage:

| Pair          | Frame-rate floor | Recovered offset error |
| ------------- | ---------------- | ---------------------- |
| 240 + 240 fps | 1.7 ms           | **0.5 ms**             |
| 120 + 120 fps | 3.4 ms           | **0.4 – 4.2 ms**       |
| 120 + 30 fps  | 9.9 ms           | **0.3 ms**             |
| 30 + 30 fps   | 13.6 ms          | **0.5 ms**             |

The floor is what the two frame rates permit, not what the method achieves; the
correlation beats it because it averages several hundred samples rather than
locating one instant. Anchoring on the four swing events instead gives 14–27 ms
on the same pairs, which is the measurement that decided which estimator supplies
the offset.

Aligning two 312-frame clips costs 0.5 ms on signals already filtered. Nothing is
cached, for the same measured reason as Phase 3.

### 3D reconstruction (Phase 9, 2026-09-17)

`scripts/benchmark_reconstruct.py`. **Not a real capture**: a synthetic body
whose 3D positions are inputs, filmed by two simulated cameras. There is no pose
estimator in the fixture, so no motion blur, no occluded hip and no mis-tracked
wrist — the errors are a **floor**.

Accuracy over the eight landmarks that move, against isotropic landmark noise:

| landmark sigma | median     | p95         | reprojection | uncertainty |
| -------------- | ---------- | ----------- | ------------ | ----------- |
| 0.5 px         | 1.0 mm     | 1.9 mm      | 0.16 px      | 1.0 mm      |
| **2.7 px**     | **5.5 mm** | **10.4 mm** | 0.85 px      | 5.4 mm      |
| 5.0 px         | 10.1 mm    | 19.3 mm     | 1.55 px      | 10.1 mm     |

2.7 px is the 0.0014 frame widths Phase 3 measured on real footage, at 1920 px
wide.

The two sweeps that decided the design both show the residual refusing to move
while the answer does — along the epipolar line it is _exactly_ 0.00 px while the
error reaches 10.8 mm, and across a camera separation sweep from 90° to 8° it is
flat at 0.84 px while the error grows 4.7x. See the section above and
[ADR-0013](decisions/ADR-0013-epipolar-blindness.md).

Reconstructing 312 frames (8,778 points) costs 70 ms, against ~95 ms to filter
both clips and ~1.3 s per clip to extract poses. Nothing is cached, for the same
measured reason as Phase 3.

## Testing strategy

| Layer                                 | Tool            | Covers                                                                                                                                                                                                                                                                                            |
| ------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contracts, probes, dispatch, protocol | pytest (905)    | serialization, status aggregation, hash verification, error normalisation, RPC framing, rotation conventions, VFR detection, decode, caching, pose, filtering, phase detection, biomechanics, camera views, time alignment, project storage, camera calibration, 3D reconstruction, club tracking |
| Transport framing, path resolution    | cargo test (12) | notification vs reply, id correlation, malformed frames, `uv`/project discovery                                                                                                                                                                                                                   |
| IPC wrappers, component rendering     | Vitest (99)     | error normalisation, status rendering, remediation display, metadata panels, failure states, frame-by-frame inspection, alignment presentation, manual anchor picking, calibration coverage                                                                                                       |
| UI flows                              | Playwright (34) | layout, engine data rendering, import flow, screen switching, failure panel, phase timeline scrubbing, two-camera alignment, calibration review                                                                                                                                                   |

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
