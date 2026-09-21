# Testing and fixtures

Run these commands from the repository root:

```bash
npm run check:all
npm run rs:test
npm run gen:types:check
npm run format:check
npm run docs:check
npm run py:coverage
npm run test:coverage
npm run test:e2e
```

`check:all` includes Python and frontend unit tests, linting, type checks and
Rust formatting/clippy. Rust tests, contract drift, formatting, documentation drift, coverage and
browser workflows have separate commands above. CI runs each of these checks.

Install frontend dependencies with `npm ci` and the browser binary with
`npx playwright install chromium` before the browser suite. Browser tests run
the Vite server locally and require an available port from `playwright.config.ts`.

Python tests require the locked environment (`uv sync --project python --locked`).
The video integration tests need FFmpeg/ffprobe; real pose tests need the pinned
models (`uv run --project python python scripts/download_models.py`). CI sets
`GSA_REQUIRE_FFMPEG=1` and `GSA_REQUIRE_MODELS=1` so missing prerequisites fail.
On macOS, the MediaPipe build needs access to graphics services even with its
CPU delegate. A sandbox denial is not an inference failure to suppress or skip.

## Coverage

Python measures all `analyzer` modules, including modules not reached by a test,
with branch coverage enabled. `--cov-config=python/pyproject.toml` is explicit
because the commands run above that file's directory. The minimum is **76%**
combined statement and branch coverage. Frontend coverage includes production
TypeScript and TSX, excluding tests, fixture data, test setup and the `main.tsx`
mounting entry point. Its minima are **80% statements, 66% branches, 74%
functions and 83% lines**. These are regression floors based on the measured
suite, not completeness or accuracy targets.

Reports are written under ignored `coverage/python/` and `coverage/typescript/`.
Open the Python `html/index.html` or TypeScript `index.html` for uncovered lines;
XML and LCOV support external reporting. CI uploads both directories as
`coverage-python` and `coverage-typescript`, including on test failure, for
14 days. Browser failures upload `test-results/` with available traces; traces
are recorded on the first CI retry. No hosted coverage service or secret is
required.

Python CLI and runtime-boundary paths still have substantial uncovered branches.
Vitest does not count browser workflows toward its coverage, so the Swing screen
can be exercised by Playwright while appearing uncovered in the Vitest report.
The reports make those gaps visible; a passing threshold does not close them.

## Shared fixture library

The library lives alongside the tests so production code cannot accidentally
depend on synthetic measurements. Reuse these builders before adding another
body, camera or clock to an individual test file.

| Fixture                              | Constructed truth                                                                          | Consumers                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| `python/tests/synthetic_signals.py`  | Seeded irregular timestamps with a rate change; cubic position and closed-form derivatives | Numerical invariants, missing-landmark pipeline tests        |
| `python/tests/synthetic.py`          | Continuous hand arc, declared event times, body dimensions, clock offset and time scaling  | Phases, filtering, synchronization, persisted pipeline       |
| `python/tests/synthetic_board.py`    | Camera intrinsics, stereo extrinsics, board dimensions and projections                     | Calibration, triangulation, analytical disparity             |
| `python/tests/synthetic_body3d.py`   | Moving body in known 3D coordinates observed from two cameras                              | Reconstruction, spatial metrics, comparison, scene rendering |
| `python/tests/synthetic_club.py`     | Shaft endpoints, motion and rendered frames                                                | Club geometry, detection and tracking                        |
| `python/tests/synthetic_ball.py`     | Ball location and departure frame                                                          | Detection, tracking and impact fusion                        |
| `python/tests/synthetic_coaching.py` | Explicit measured evidence and refusal cases                                               | Coaching rules and phrasing guard                            |
| `python/tests/synthetic_labels.py`   | Varied synthetic swings, players and labels                                                | Training, evaluation, split leakage checks                   |
| `python/tests/conftest.py`           | Fake estimator with distinguishable landmarks and missing frames; isolated cache/database  | Decode-to-pose integration, storage, dispatch                |
| `python/tests/fixtures/video/`       | Committed CFR, VFR, rotated and audio-only files                                           | Probe, decode, seek and browser video playback               |
| `e2e/fixtures.ts`                    | Declared engine responses, response sequences and recorded IPC arguments                   | Browser success, cancellation, failure and retry workflows   |

`scripts/make_video_fixtures.py --verify` checks the committed container facts.
Do not regenerate video fixtures during tests: the FFmpeg version that builds
the input would then also define the expected rotation and timing behavior.

New numerical assertions should use independent truth or invariants. Phase 19
checks a 3–4–5 triangle after rotation, translation and scaling; pixel triangles
through aspect correction; cubic derivatives under unequal frame intervals and
clock rescaling; stereo depth from analytical disparity; and affine alignment
with the clocks reversed. Returning no finite results cannot satisfy these tests.

Pipeline tests cross Parquet and public dispatch boundaries. They verify missing
poses, separately rejected visibility/presence, cache corruption and invalidation,
renamed clips, and frame identity from metadata through decoding and extraction.
Browser tests check session mutations and their IPC arguments, analysis failures
and retry, metric rendering, and seeking in a real decoded VFR clip.

## Limits

Synthetic results verify arithmetic and behavior under their stated assumptions.
They do not measure pose accuracy on golfers, coaching validity, or performance
on an independently labelled dataset. The real inference tests check execution,
not landmark ground truth.

Playwright runs the Vite application with a stubbed Tauri bridge. Its video
decoder is real; its native file picker, SQLite connection and Rust transport
are not. Python tests cover SQLite and pipeline behavior, and Rust tests cover
transport framing. There is no automated packaged-WebView integration suite.

## Documentation evidence

`npm run docs:check` regenerates the benchmark appendix in memory and compares it
with the checked-in file. CI runs this without private footage or model inference.
To refresh measured evidence, use the capture command in [the appendix](benchmarks.md);
rendering alone preserves the original run rather than manufacturing new timings.
