# Golf Swing Analyzer

Local-first golf swing biomechanics analysis from face-on and down-the-line
video. Pose and club motion are extracted with computer vision, swing phases are
segmented deterministically, and biomechanics metrics are computed with explicit
units, confidence, and methodology. All processing runs on your machine; video
never leaves it.

> **Status: Phase 0 of 21 complete.** The foundation, typed engine boundary, and
> environment health check are built and verified. No analysis pipeline exists
> yet. Sections below marked _Not yet implemented_ say so rather than describing
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

_Not yet implemented — Phases 1-13._ The only engine method today is `doctor`.

## 6. Supported video formats

_Not yet implemented — Phase 1._ Planned: MP4 and MOV containers with H.264 or
HEVC, including variable-frame-rate and rotated smartphone recordings. FFmpeg
9.0.1 with VideoToolbox hardware decode is detected and reported today, but
nothing consumes it yet.

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

_Not yet implemented — Phases 1-11._ Planned stages and their ordering are in
[docs/ROADMAP.md](docs/ROADMAP.md).

## 9. 3D reconstruction methodology

_Not yet implemented — Phases 8-9._ The system will distinguish
`UNCALIBRATED` / `INTRINSIC_ONLY` / `STEREO_CALIBRATED` and will refuse to make
metric-scale claims without stereo calibration.

## 10. Biomechanics methodology

_Not yet implemented — Phase 5._ Every metric will carry name, value, unit,
phase, confidence, source frames, and methodology. Angles derived from a single
2D camera will be labelled as projected, not true 3D rotation.

## 11. Model architecture

_Not yet implemented — Phase 12._ No model is trained. No accuracy figure will
be published without a real labelled evaluation set with session- and
player-grouped splits.

Currently vendored: MediaPipe Pose Landmarker (lite/full/heavy, float16), pinned
by sha256 in `models/manifest.json`. See
[ADR-0006](docs/decisions/ADR-0006-model-pinning.md).

## 12. Testing

```bash
npm run check:all
```

| Suite      | Count | Scope                                                                    |
| ---------- | ----- | ------------------------------------------------------------------------ |
| pytest     | 48    | contracts, environment probes, model verification, dispatch, RPC framing |
| cargo test | 9     | protocol framing, id correlation, `uv`/project resolution                |
| Vitest     | 15    | IPC error normalisation, health screen rendering                         |
| Playwright | 4     | UI layout and engine-data rendering                                      |

All 76 pass as of Phase 0. Numerical algorithm tests against analytical
solutions arrive with Phase 3, which is where the first real numerics land.

## 13. Performance benchmarks

Measured on the reference machine, 2026-09-15:

| Measurement                             | Value   |
| --------------------------------------- | ------- |
| Worker spawn → ready                    | 163 ms  |
| First `doctor` call (framework imports) | 1373 ms |
| `doctor` call, warm (median of 5)       | 127 ms  |
| Components probed per call              | 12      |

That ~11x cold/warm gap is why the engine is a long-lived process rather than
one invocation per call.

No pipeline throughput figures exist yet, because no pipeline exists yet. A
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
