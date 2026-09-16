# ADR-0001: Pin Python to 3.12

**Status:** Accepted · 2026-09-15

## Context

The project targets "Python 3.12+". At the time of setup the newest CPython was
3.13/3.14, and `uv` will happily build an environment on any of them.

MediaPipe 1.0.1 — the pose landmarker this project depends on — publishes a
`py3-none-macosx_11_0_arm64` wheel whose trove classifiers declare support for
Python **3.9 through 3.12 only**. The wheel's universal `py3` tag means pip will
install it on 3.13 without complaint; the declared support simply stops at 3.12.

## Decision

Pin the Python environment to 3.12 exactly:

- `requires-python = "==3.12.*"` in `python/pyproject.toml`
- `python/.python-version` containing `3.12`
- `[tool.uv] python-preference = "only-managed"` so uv uses its own CPython
  build rather than whichever 3.12 it happens to find first

The last point matters on this machine specifically: an Anaconda 3.12.7 is on
PATH, and without `only-managed` uv built the virtualenv on top of it. That made
the project's environment depend on an unrelated third-party toolchain that may
be upgraded or removed independently. The pinned environment now uses
uv-managed CPython 3.12.14.

`analyzer.environment.hardware.probe_python()` reports a non-3.12 interpreter as
DEGRADED with this reasoning, so a drifting environment is visible rather than
mysterious.

## Consequences

- The dependency set resolves cleanly: numpy 2.5 and scipy 1.18 both require
  > = 3.12, so 3.12 is simultaneously the floor and the ceiling.
- Adopting 3.13 requires MediaPipe to declare support for it first.
- CI must install 3.12 rather than "latest".
