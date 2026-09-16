# ADR-0004: Constrain OpenCV to the 4.x line

**Status:** Accepted · 2026-09-15

## Context

`opencv-contrib-python` now publishes 5.0.0.93 (OpenCV 5) alongside 4.14.0.94.
MediaPipe depends on `opencv-contrib-python` with no upper bound, so a naive
install resolves to OpenCV 5.

OpenCV 5 reorganises several APIs this project depends on in later phases:

- `cv2.aruco` Charuco detection and board APIs (Phase 8, calibration)
- calibration entry points such as `calibrateCamera` / `stereoCalibrate`
- Hough transform variants used for shaft detection (Phase 10)

Adopting OpenCV 5 now would mean writing Phases 8-10 against APIs that most
published reference material does not cover, for no present benefit.

## Decision

Constrain to `opencv-contrib-python>=4.14,<5`. The resolved version is 4.14.0.

## Consequences

- Calibration and club-tracking work can follow well-documented 4.x APIs.
- The constraint must be revisited before OpenCV 4.x loses support.
- If MediaPipe ever requires OpenCV 5, this pin will surface the conflict at
  `uv sync` rather than as a runtime failure.
