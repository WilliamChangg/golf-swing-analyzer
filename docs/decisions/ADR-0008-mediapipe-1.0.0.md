# ADR-0008: Pin MediaPipe to 1.0.0, and verify inference instead of import

**Status:** Accepted (Phase 2)

## Context

Phase 0 pinned `mediapipe~=1.0.1` and reported the package as healthy on the
strength of `import mediapipe` succeeding. It does succeed. Pose inference,
however, does not run at all on this platform under 1.0.1: the first attempt to
open the pose landmarker graph aborts the process.

```
F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
*** Check failure stack trace: ***
    @  -[DrishtiMetalHelper initWithCalculatorContext:]
    @  mediapipe::api2::TensorsToDetectionsCalculator::Open()
    @  mediapipe::CalculatorNode::OpenNode()
```

`TensorsToDetectionsCalculator` unconditionally constructs a Metal helper, and
the Metal graph service is not attached, so an internal `CHECK` fails and
MediaPipe calls `abort()`. There is no exception: the interpreter is gone.

Measured across the matrix, every combination fails identically:

| Running mode | lite  | full  | heavy |
| ------------ | ----- | ----- | ----- |
| IMAGE        | abort | abort | abort |
| VIDEO        | abort | abort | abort |

Requesting `BaseOptions.Delegate.CPU` explicitly does not avoid it — the Metal
helper is constructed regardless of the delegate.

Testing adjacent versions in throwaway environments, with the same model files
and the same code:

| Version | Result                      |
| ------- | --------------------------- |
| 0.10.35 | runs                        |
| 1.0.0   | runs                        |
| 1.0.1   | aborts the process          |

So this is a regression introduced in 1.0.1, not a property of the 1.0 line.

## Decision

**Pin `mediapipe==1.0.0` exactly.** Not `~=1.0`, which would re-admit 1.0.1, and
not `<1.0.1`, which reads as a range when the fact is a single known-good
version. 1.0.0 declares the same Python support as 1.0.1 (3.9–3.12), so
[ADR-0001](ADR-0001-python-312.md)'s reasoning for the 3.12 pin is unchanged.

**Verify pose inference by running it.** The health check now runs one real
inference and reports what happened, rather than inferring capability from a
successful import.

## Consequences

The health check gained a `pose_runtime` component, and about a second of
runtime. That is the honest price of the claim it makes, and it is charged only
to the check, not to the pipeline.

It runs in a **child process**, which is not incidental. The failure being
detected is an abort, so there is nothing to catch in-process: an in-process
check would have taken the engine down with it and produced no report at all.
A child turns a fatal signal into an exit code, and the probe distinguishes the
cases that follow from it:

| Child outcome            | Reported  | Means                                   |
| ------------------------ | --------- | --------------------------------------- |
| exit 0 with the sentinel | `ok`      | inference ran, with its measured time   |
| killed by a signal       | `error`   | the build cannot open its graph here    |
| exit 4 (caught)          | `error`   | the model file is bad, the build is not |
| exit 3                   | `missing` | no model to test with                   |
| no model on disk         | `degraded`| unverified, which is not the same as false |

The last two rows matter as much as the first: "not verified" is reported as
distinct from "verified working", because the alternative is the false green
this ADR exists because of.

Upgrading MediaPipe now requires re-running the health check and seeing
`pose_runtime` report `ok`. If a later release fixes the regression the pin can
move, but only on that evidence.

## What this cost, and the lesson

Phase 0 shipped a green health check for a pipeline that could not process a
single frame, and nothing about the report hinted at it. The system was
measuring the wrong thing — presence rather than function — which is the exact
failure mode the project's own rule about never reporting an unmeasured
capability is meant to prevent.

The general form: **a probe should exercise the capability the user cares
about, not a proxy for it.** Import is a proxy. Applied elsewhere in this
codebase, this is why the ingestion layer decodes a frame to settle rotation
rather than trusting a property, and why the decode benchmark measures both
backends rather than assuming the hardware one is faster.
