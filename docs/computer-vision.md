# Computer vision methodology

This guide describes the implemented path from recorded pixels to motion evidence.
See [biomechanics](biomechanics.md) for interpretation and
[modeling](modeling.md) for learned components. Source directories below are
relative to `python/analyzer/`.

## Ingestion, timestamps and persistence

`ingestion/probe.py` reads stream metadata and a frame index with ffprobe.
Presentation timestamps come from integer time-base ticks, including variable
frame intervals. Rotation is interpreted in one documented direction; displayed
width/height describe the upright frame, not necessarily the encoded raster.
`ingestion/reader.py` exposes a frame-source interface with OpenCV and FFmpeg
implementations. OpenCV's CPU decoder is the default selected by the historical
[decode benchmark](decisions/ADR-0007-decode-backend.md).

Metadata and poses are keyed by sampled video content. This key is a reuse aid,
not an integrity proof of the complete video. A moved video can reuse metadata
with its current path; an attached project clip can be explicitly relocated.
Parquet pose storage includes schema, model provenance, timestamps, image
geometry and missing detections. Missing frames remain represented as missing.

```bash
uv run --project python analyzer probe swing.mov
uv run --project python analyzer frames swing.mov
uv run --project python analyzer extract swing.mov
```

Subsequent pose-based commands read the extraction. Repeat `extract` after a
model change rather than assuming that an existing extraction used the new bytes.

## Pose estimation

`pose/estimator.py` defines the estimator interface; the MediaPipe adapter maps
its output into project contracts. It converts decoded BGR to RGB and supplies
strictly increasing timestamps in VIDEO mode. MediaPipe types do not escape the
adapter. Its CPU delegate is explicit.

The stored `IMAGE` and `HIP_LOCAL` coordinates have different meanings.
`IMAGE` is a projection; `HIP_LOCAL` is a body-centred model estimate in roughly
metric units. Neither supplies calibrated camera geometry. Planar analysis
converts image coordinates to isotropic, upward-y `FRAME_WIDTHS` before filtering.
The [coordinate reference](coordinate-systems.md) defines each transform.

Detection rate reports how often the model returned a pose, not how close the
landmarks are to anatomy. Full remains the default without a measured accuracy
ranking among the installed variants.

## Filtering and phase detection

`filtering/` gates visibility and presence separately. Failed observations become
NaN; bounded short gaps can be interpolated, while long gaps and unsupported
ends remain blocked. Interpolated samples have zero fitting weight by default
so they do not count the same neighbouring observations twice.

A local polynomial is fitted on actual timestamps. Position, velocity and
acceleration come from the same fit; derivatives are not finite differences of
smoothed positions. The configured defaults are a degree-four polynomial and a
0.10-second window. These are algorithm settings, not claimed resolution.

When the clip cannot support the requested window, `filter_sequence` widens it
to the narrowest supported window and reports both the requested and applied
width, along with a warning. `SmoothingConfig.auto_widen=False` restores strict
refusal. Synchronization resolves one common window for both cameras. This
current policy supersedes the original low-rate refusal in ADR-0009; see
[ADR-0021](decisions/ADR-0021-resolving-the-smoothing-window.md).

`phases/` derives hand-motion signals and detects takeaway, top, impact and
finish. Confidence combines observation, event shape and temporal resolution;
it is not a calibrated probability. Impact here is a hand-motion estimate.
No swing produces a refusal, and multiple swing-like motions produce warnings;
the engine is designed for one complete swing per clip, not session segmentation.

Slow-motion dilation maps playback time to real motion time. It changes durations
and derivatives and cannot reliably be read from an already conformed file.
Supply a known factor rather than treating an implausible result as a factor
estimate. The default is real-time playback.

## Synchronization and calibration

`sync/` fits an affine relation between clocks using motion correlation and
swing-event evidence. An offset can be reported even when a rate correction is
not identifiable. Residuals and frame-rate floors accompany the result. Matching
swing-shaped signals does not establish that two cameras filmed the same swing.

Calibration uses a printed ChArUco board, coverage checks and explicit distortion
models. A low reprojection error from repeated similar board views does not prove
that intrinsics are well constrained. Measure a printed square with a ruler;
that physical length establishes the scale. Keep camera lens, zoom, crop,
resolution and rig pose consistent between calibration and swing footage.

Example workflow (replace project ID `1` with the ID returned by `create`, and
`34.6` with your measured square length in millimetres):

```bash
uv run --project python analyzer project create "Stereo session"
uv run --project python analyzer project add 1 faceon.mov --role face_on
uv run --project python analyzer project add 1 dtl.mov --role down_the_line
uv run --project python analyzer extract faceon.mov
uv run --project python analyzer extract dtl.mov
uv run --project python analyzer project sync 1
uv run --project python analyzer calibrate board board.png
uv run --project python analyzer calibrate camera faceon_board.mov --role face_on --project 1 --square-mm 34.6
uv run --project python analyzer calibrate camera dtl_board.mov --role down_the_line --project 1 --square-mm 34.6
uv run --project python analyzer calibrate stereo 1 faceon_board.mov dtl_board.mov --square-mm 34.6
uv run --project python analyzer calibrate show 1
uv run --project python analyzer reconstruct 1
uv run --project python analyzer metrics faceon.mov --project 1
```

The two board recordings for stereo must depict corresponding board poses;
use `calibrate stereo --help` for clock offset and role options. The swing's
alignment is not automatically the clock offset of separately recorded board
clips. Board generation is shown here as a command reference; print and record
it before attempting the calibration commands. More detail is in the
[capture protocol](../data/README.md).

## Reconstruction and the viewport

`reconstruction/` pairs observations at the same mapped time, accounts for
camera intrinsics/distortion and triangulates into the reference camera's
coordinate frame. It requires usable stereo calibration and time alignment;
missing geometry is a refusal, not a default camera or clock.

Reprojection residual measures image agreement. Errors along epipolar directions
can leave that residual small while shifting the reconstructed point. Ray
convergence and bone-length consistency provide different checks; they are not
interchangeable with ground truth. Gravity and target-line orientation remain
unknown, so `WORLD` and gravity-relative posture are not produced.

`scene.py` prepares bounded frame windows for the Three-D viewport. Its viewpoint
panel reports how projection hides depth uncertainty. A plausible skeleton on
screen is not an independent accuracy check. The
[benchmark appendix](benchmarks.md) retains synthetic reconstruction and spatial
metric sweeps; no real calibrated stereo capture has validated those figures.

## Club, ball and impact

```bash
uv run --project python analyzer club swing.mov
uv run --project python analyzer ball swing.mov
uv run --project python analyzer impact swing.mov
uv run --project python analyzer impact swing.mov --no-club
```

These commands also need the prior pose extraction. Club tracking (`club/`)
uses edges/Hough candidates anchored at the hands, geometric plausibility and
temporal gating. A shaft direction may be supported when its endpoint is not;
club-head position is withheld without endpoint evidence. Coverage is broken
out by phase because a detector that works at address may fail through impact.
Low confidence, ambiguity and excessive motion produce explicit non-detections.
This is a classical baseline, not a trained club detector or validated club-speed
measurement.

Ball detection (`ball/`) looks for a stationary ball and its sustained departure
near the expected strike. The last present/first absent frames bracket an
observed departure; occlusion and false blobs remain possible. It does not
measure launch velocity, spin, carry or a flight trajectory.

`impact.py` reconciles available estimates by precedence, retaining provenance,
uncertainty and disagreement instead of averaging them. This independent result
does not rewrite `detect_phases` or the Swing screen's phase markers. One observed
reference clip is anecdotal validation; synthetic detector tests do not establish
an impact accuracy rate across real captures.

## Video inspection and limits of the evidence

The player uses the measured timestamp index, seeks inside the intended frame's
presentation interval, and verifies the painted frame via browser video-frame
callbacks. The overlay represents the filtered trajectory used by analysis;
it does not connect a blocked gap merely to make the drawing look continuous.

Browser tests use a real decoder but stub the native transport. They establish
seek behaviour for the committed fixtures, not support for every codec/WebView.
Numerical tests establish transform and triangulation arithmetic. Real pose
execution and overlay inspection establish neither a labelled landmark error
nor clinical or coaching validity. The next validation inputs are labelled
landmarks, simultaneous calibrated footage and observed club/ball ground truth.
