"""Minimal pose inference, run as a child process by the runtime probe.

A separate module *and* a separate process, because the failure this exists to
detect is an abort rather than an exception. MediaPipe 1.0.1 calls `abort()`
from inside its own graph when the pose landmarker opens on macOS arm64:

    F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
        -[DrishtiMetalHelper initWithCalculatorContext:]
        mediapipe::api2::TensorsToDetectionsCalculator::Open()

There is no exception to catch -- the interpreter is gone. Running it in a child
turns a fatal signal into an exit code the parent can report on.

The parent runs this **by file path**, not with `-m`. Importing the package
would drag in OpenCV, the ingestion layer and the rest of the engine, none of
which the question needs; by path the child imports only MediaPipe and numpy.
Nothing at module scope here imports either, so the parent can import this
module for its constants at no cost.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Printed on success, and checked for by the parent. A sentinel rather than a
# zero exit code alone, because a process that aborts after doing the work would
# otherwise be indistinguishable from one that never started it.
SUCCESS_SENTINEL = "POSE_SMOKE_OK"

# Exit codes the parent distinguishes. A *negative* return code, meaning death
# by signal, is the interesting one and is not listed here: that is the abort
# this whole module exists to catch, and no code in this file can produce it.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MODEL_MISSING = 3
EXIT_INFERENCE_FAILED = 4

# Small enough to cost nothing to allocate, large enough that the detector's
# input resizing has something to work with. Nothing is expected to be found in
# it; the question is whether the graph runs at all.
_PROBE_SIZE = 64


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(f"usage: python {Path(__file__).name} <model-path>", file=sys.stderr)
        return EXIT_USAGE

    model_path = Path(argv[0])
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return EXIT_MODEL_MISSING

    import mediapipe as mp  # type: ignore[import-untyped]
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision  # type: ignore[import-untyped]

    started = time.perf_counter()
    try:
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        try:
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.zeros((_PROBE_SIZE, _PROBE_SIZE, 3), dtype=np.uint8),
            )
            landmarker.detect(image)
        finally:
            landmarker.close()
    except Exception as exc:  # noqa: BLE001 - the whole point is to survive anything catchable
        # A catchable failure is a bad model or a bad option, which is a
        # different diagnosis from the uncatchable abort of a broken build.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INFERENCE_FAILED

    print(f"{SUCCESS_SENTINEL} {(time.perf_counter() - started) * 1000:.1f}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
