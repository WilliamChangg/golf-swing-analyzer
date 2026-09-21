# Golf Swing Analyzer

Local golf swing analysis from face-on and down-the-line video, with a desktop
player, pose overlays, phase detection, biomechanics, coaching findings and swing
comparison. Every result carries its units, source frames and limitations.

**Status: Phases 0–20 complete.** See the
[roadmap](docs/ROADMAP.md). This is a source-run research application, not a
validated coaching instrument or a self-contained desktop distribution. There is
no labelled golfer evaluation set and no real calibrated stereo validation.

## 1. Project overview

The Python engine computes evidence; the desktop makes it inspectable against
video. Measurements that the recording cannot support are reported as refusals.
There is no overall swing score. Optional language-model phrasing only rewords
computed findings, is disabled by default, and receives no video frames.

Analysis runs on the local machine. Dependency installation and model downloads
use the network. Optional phrasing calls a configured loopback model server.
Footage and derived pose files are not committed to the repository.

## 2. Architecture

```text
apps/desktop/       Tauri + React + TypeScript desktop
packages/types/     Generated contracts and handwritten IPC types
python/analyzer/    Analysis engine, CLI and NDJSON-RPC worker
models/            Pinned model manifest; downloaded weights are ignored
scripts/           Setup, code generation, benchmarks and overlays
docs/              Methodology, testing, decisions and measured evidence
```

The WebView invokes Rust commands. Rust supervises a Python worker and exchanges
JSON-RPC over stdin/stdout; the engine boundary has no HTTP server. Only files
chosen through the native dialog are admitted to the video asset scope.
Pydantic contracts generate the shared TypeScript types.

Read [Architecture](docs/architecture.md) for transport, persistence and caching,
and [Coordinate systems](docs/coordinate-systems.md) before interpreting geometry.

## 3. Installation

The verified platform is macOS on Apple silicon. Install Node **>=20.19**, FFmpeg
(including ffprobe), and Xcode Command Line Tools. Homebrew is needed by bootstrap
if `uv` is absent. From a checkout of this repository:

```bash
./scripts/bootstrap.sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
npm run doctor
npm run dev
```

Bootstrap installs Node dependencies, installs `uv` and Rust if missing, creates
the Python 3.12 environment, downloads the required pinned pose model, and runs
the environment probes. It does not change your shell profile. Initial setup
needs network access; installed models can subsequently run offline.

For an existing toolchain and reproducible dependency installation:

```bash
npm ci
uv sync --project python --locked
uv run --project python python scripts/download_models.py
```

The native executable still needs `uv`, the Python project and model files.
Building an app bundle does not bundle a standalone Python runtime. Launch from
this checkout, or set `GSA_PYTHON_DIR` to its absolute `python/` path and
`GSA_UV_BIN` to the `uv` executable if discovery fails.

## 4. Development

```bash
npm run dev                 # native desktop and hot reload
npm run dev:web             # browser UI; no live Tauri engine bridge
npm run build               # native build/bundle; external engine still required
npm run check:all           # lint, types, Python/Vitest tests, Rust fmt/clippy
npm run gen:types           # regenerate contracts after Pydantic changes
```

`check:all` does **not** run every verification task. Also run `npm run rs:test`,
`npm run test:e2e`, `npm run gen:types:check`, `npm run format:check` and
`npm run docs:check`. Coverage commands and required prerequisites are in
[Testing](docs/testing.md).

Source contracts live in `python/analyzer/contracts/`; edit them there, not in
`packages/types/src/generated/`. New estimators belong behind the existing pose,
club or ball interfaces. Keep golf-specific interpretations in phases,
biomechanics and coaching.

## 5. Running analysis

In **Swing**, choose a clip, set its slow-motion factor when known, then select
**Analyse**. The screen runs extraction, phases, metrics, coaching and overlay
construction. Scrub or select a reported frame to inspect its evidence. A failed
stage stops analysis and leaves an error that can be retried.

| Screen      | Use                                                                           |
| ----------- | ----------------------------------------------------------------------------- |
| Swing       | Analyse one clip, inspect video, pose, phases, metrics and findings           |
| Sessions    | Create sessions, attach clips with camera roles, align and review calibration |
| Three-D     | Reconstruct a calibrated, aligned session and inspect its geometry            |
| Compare     | Compare separate swings on a phase-relative clock, subject to camera gates    |
| Video       | Inspect individual ingestion and analysis stages                              |
| Environment | Probe dependencies and runtime; verify, download or reinstall pose models     |

From the terminal, **extract first**. Later stages consume that persisted pose
sequence; passing a video path locates its cache rather than automatically doing
all preceding work.

```bash
uv run --project python analyzer probe path/to/swing.mov
uv run --project python analyzer extract path/to/swing.mov
uv run --project python analyzer phases path/to/swing.mov
uv run --project python analyzer metrics path/to/swing.mov
uv run --project python analyzer coach path/to/swing.mov
uv run --project python analyzer compare before.mov after.mov
```

Both inputs to `compare` need extraction. For footage playing eight times slower
than the actual motion, use `--slow-motion 8` on analysis commands that expose it;
`compare` has separate reference and target factors. This is an example input,
not an inferred factor. Use `analyzer COMMAND --help` to see available options.

[Computer vision](docs/computer-vision.md) covers the stereo workflow, club/ball
commands and player timing. [Modeling](docs/modeling.md) covers labels and training.

Projects, labels and trained models live under
`~/Library/Application Support/golf-swing-analyzer` on macOS. Recomputable files
live under `~/Library/Caches/golf-swing-analyzer`. Other platforms use XDG data
and cache directories. Back up the data directory and original videos; clearing
cache is not a backup strategy. Override with `GSA_DATA_DIR`, `GSA_CACHE_DIR` or
`GSA_MODELS_DIR` (the latter must include the model manifest).

## 6. Video formats and capture

Ingestion uses FFmpeg/ffprobe and OpenCV. Committed integration fixtures cover
H.264 MP4/MOV, variable frame rate, display rotation and audio-only rejection.
Other demuxable formats may ingest, but that does not establish codec support
in the desktop WebView. HEVC and every phone/export combination are not verified.

Actual presentation timestamps drive the pipeline; time is never inferred solely
from a frame number divided by a declared rate. Rotation is applied before
coordinates are interpreted. Slow-motion playback dilation is separate from
variable frame rate and may require a user-supplied factor.

Capture one complete swing with a visible address and finish, a stationary camera,
clear body/club visibility and enough light to limit blur. Prefer high-frame-rate
capture for fast motion. For stereo, film the **same swing simultaneously** and
retain calibration footage without moving or changing either camera. Detailed
placement, board and recording instructions are in [Capture protocol](data/README.md).

## 7. Hardware and runtime

Only Apple-silicon macOS has been exercised for the full desktop workflow.
There is no measured minimum RAM or GPU requirement and no Windows/Linux desktop
qualification. The [benchmark appendix](docs/benchmarks.md) records the hardware
actually used; it is not a performance promise for another machine.

MediaPipe pose inference explicitly uses CPU. Torch can probe CUDA or MPS and
falls back to CPU when selection/probing fails. `GSA_FORCE_CPU=1` overrides
accelerator selection; training otherwise defaults to CPU. On macOS this
MediaPipe build still initializes graphics services, even with its CPU delegate,
so a restrictive headless sandbox can abort inference.

Use **Environment → Re-check** or `npm run doctor` for measured runtime status.
Hash verification establishes artifact identity; the inference probe separately
checks that the model actually runs.

## 8. Computer vision pipeline

The engine decodes frames, extracts landmarks, gates unreliable detections,
filters on real timestamps, detects swing events and computes downstream results.
Classical club-shaft and stationary-ball detectors are additional CLI paths.
Ball departure can corroborate impact; it does not silently replace the phase
engine's hand-motion estimate.

The filter widens an unsupported smoothing window to the narrowest usable width
by default and reports the applied width and its confidence cost. It does not
recover detail absent from a low-rate or blurred recording. Long gaps remain
missing. See [Computer vision methodology](docs/computer-vision.md) and
[ADR-0021](docs/decisions/ADR-0021-resolving-the-smoothing-window.md).

## 9. Coordinates and 3D reconstruction

Image landmarks use separately normalized axes; planar measurements use frame
widths on both axes with y upward. MediaPipe's hip-local output is only roughly
metric and is not a calibrated world frame.

Two aligned, calibrated cameras can produce metric `CAMERA` coordinates through
triangulation. A low reprojection residual alone does not establish accurate
3D: ray convergence, timing, landmark correspondence and bone consistency also
matter. Gravity and a target-aligned `WORLD` frame are not produced.

The Three-D viewport reports uncertainty that its selected viewpoint may hide.
Reconstruction and spatial metric validation are synthetic; the repository has
no real simultaneous calibrated stereo capture. See
[Coordinate systems](docs/coordinate-systems.md) and the
[generated reconstruction evidence](docs/benchmarks.md).

## 10. Biomechanics

Implemented families cover posture, projected shoulder/pelvis rotation, arms and
hands, swing timing, and selected calibrated spatial quantities. Results carry
basis, camera view, units, confidence factors, methodology and source frames.
A projected angle is not interchangeable with a motion-capture joint angle.

Face-on and down-the-line cameras support different anatomical interpretations.
The engine measures view from the address geometry; a session's declared camera
role does not prove it. Unsupported measurements appear with refusal reasons.
See [Biomechanics methodology and metric catalogue](docs/biomechanics.md).

## 11. Coaching and comparison

Coaching applies evidence-backed rules to computed metrics and records why each
rule may or may not be used. Most shipped rules refuse because the available
measurement method cannot support the borrowed threshold. Confidence is not a
validated probability of correctness, and a finding is not a diagnosis.

Optional CLI phrasing uses `analyzer coach … --phrase-with` with a loopback
endpoint. The deterministic text remains available without a model. The numeric
guard rejects unsupported numbers; it does not prove that all accepted prose is
factually or semantically correct. No real language-model quality evaluation is
available. Details are in [Biomechanics](docs/biomechanics.md).

Comparison aligns takeaway, top, impact and finish, gates projected differences
by camera compatibility, and reports resolved differences or refusals. It emits
no score. Event timing uncertainty and landmark noise limit what a difference
means; an unresolved difference does not prove that two swings are identical.

## 12. Models and learning

The pose manifest pins lite, full and heavy artifacts by SHA-256. Full is the
default; no labelled landmark study establishes that it is the most accurate
choice for this application. **Environment → Manage models** installs only the
application's pinned versions. An upstream URL containing `latest` does not
make the downloaded bytes a reproducible version.

Temporal learning has a feature pipeline, a convolutional classifier, labels,
player-grouped splits, evaluation and model cards. It has **no labelled real
training/evaluation set or validated learned replacement for heuristic phases**.
`analyzer models` lists locally trained temporal models, not the pose inventory.
See [Modeling](docs/modeling.md) and [Model installation](models/README.md).

## 13. Testing

[Testing and fixtures](docs/testing.md) documents numerical invariants, pipeline
regressions, frontend tests, browser workflows, coverage floors and CI artifacts.
Synthetic tests check arithmetic against known truth; real inference tests check
execution. Neither substitutes for landmark-accuracy evaluation on golfers.

Playwright uses a stubbed Tauri bridge and a real browser video decoder. It does
not test the packaged native WebView/worker boundary. Test counts and historical
coverage measurements are recorded with their runs in [Phase 19](docs/ROADMAP.md#phase-19--testing-hardening-).

## 14. Performance and reproducibility

Use the [generated benchmark appendix](docs/benchmarks.md) for current figures,
raw samples, source fingerprints, machine, model hashes, input identity and exact
commands. Its workflow benchmark compares fresh extraction with primed cache
reuse. Synthetic reconstruction figures are identified separately from timings
on real footage. Older roadmap/ADR tables retain their historical context.

```bash
python3 scripts/document_benchmarks.py          # regenerate from retained output
npm run docs:check                            # fail if appendix has drifted
uv run --project python python scripts/benchmark.py path/to/swing.mov --repeats 5 --json
```

Pose reuse requires matching content and model identity. Metrics and coaching
cache keys include extraction identity and configuration. Project-backed results
bypass those small result caches because calibration or synchronization can
change independently of video content. See [Architecture](docs/architecture.md).

## 15. Limitations and remaining work

- **No measured golfer accuracy.** There is no labelled, player-held-out real
  evaluation set. Detection coverage, confidence and synthetic errors are not
  substitutes for it.
- **Incomplete real-capture validation.** Calibration, stereo reconstruction and
  club tracking lack real ground truth. Ball departure has been inspected on a
  single reference clip; it is not an impact-accuracy study.
- **Single-view ambiguity.** Depth, rotation sign, camera attitude and target
  direction cannot be recovered from a projected measurement. Hip-local model
  output does not remove those ambiguities.
- **Capture sensitivity.** Blur, occlusion, moving cameras, missing address,
  unknown slow-motion factors and multiple swings can undermine analysis.
  Automatic window widening trades temporal resolution for usable support.
- **Uncertainty is partial.** Detector confidences are not calibrated error
  probabilities. Timing brackets do not bound every landmark or model error.
  Comparison can report false differences caused by those unmodelled errors.
- **Deployment and validation gaps.** Native packaging still needs an external
  Python environment; other desktop platforms and packaged-WebView automation
  remain unverified. Coaching norms and language-model prose need independent
  validation before broader claims.

Further work should collect labelled diverse golfers, simultaneous calibrated
views and club/ball ground truth; validate uncertainty and comparison error;
then qualify deployment platforms and a self-contained runtime. These are open
work beyond the completed build checklist, not capabilities implied by it.

Package metadata declares MIT, but this checkout has no standalone `LICENSE`
file. Model artifact provenance is recorded in [the manifest](models/manifest.json);
upstream model terms and third-party footage permissions apply separately.
