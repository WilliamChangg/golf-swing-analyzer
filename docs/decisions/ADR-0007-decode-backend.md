# ADR-0007: Decode with OpenCV on the CPU; keep VideoToolbox as an opt-in

**Status:** Accepted (Phase 1)

## Context

Phase 1 called for "a VideoToolbox decode path plus a CPU fallback", on the
reasonable assumption that hardware decode would be the fast path and software
decode the safety net. Two things turned out to be true that invert that.

**OpenCV cannot reach VideoToolbox at all.** `cv2.VideoCapture` accepts a
`CAP_PROP_HW_ACCELERATION` hint, but on this platform it reports back
`VIDEO_ACCELERATION_NONE` whatever is requested:

```
request NONE  -> opened=True selected=0.0 backend=FFMPEG
request ANY   -> opened=True selected=0.0 backend=FFMPEG
```

OpenCV only defines `VIDEO_ACCELERATION_D3D11`, `_VAAPI` and `_MFX`; there is no
VideoToolbox constant, because its bundled FFmpeg is not built with that
hwaccel. Reaching VideoToolbox therefore requires an out-of-process ffmpeg, not
a flag.

**Hardware decode is slower for this pipeline.** Measured with
`scripts/benchmark_decode.py` on 600 frames of 1920x1080 H.264, median of 5
runs, Apple M1 Pro / macOS 26.4.1:

| Backend                         | Median  | Frames/s |
| ------------------------------- | ------- | -------- |
| OpenCV (in-process, CPU)        | 564 ms  | **1065** |
| ffmpeg subprocess, CPU          | 1349 ms | 445      |
| ffmpeg subprocess, VideoToolbox | 2332 ms | 257      |

VideoToolbox is 1.7x slower than software decode through the same subprocess,
and 4.1x slower than decoding in process. This is not a surprise once stated:
the pipeline needs BGR frames in system memory for MediaPipe and OpenCV, so a
hardware-decoded frame has to be downloaded from the GPU and colour-converted,
and that transfer costs more than the decode it saved. The subprocess path pays
a second cost on top — every frame crosses a pipe as raw bytes, which is what
separates 445 fps from 1065 fps.

## Decision

`OpenCVFrameSource` is the default. It decodes in process, supports random
access, and is the fastest of the three.

`FFmpegPipeFrameSource` is kept, with `hardware=True` selecting VideoToolbox. It
is opt-in and is not used by any default path.

## Consequences

The roadmap's "VideoToolbox path with CPU fallback" is inverted: CPU is the
path, and VideoToolbox is the option. The capability is still built, reachable,
and tested — including a test asserting that hardware and software decode
produce the same pixels — so the decision can be revisited with a measurement
rather than a rewrite.

It should be revisited if the pipeline ever stops needing frames on the CPU.
Hardware decode wins when the frames stay on the GPU; the readback is the cost,
not the decode. If a future phase does pose inference or rendering on the GPU,
these numbers no longer apply and the benchmark should be re-run.

Keeping `FFmpegPipeFrameSource` has a second justification independent of
acceleration: it is a second implementation of the `FrameSource` protocol, which
is what demonstrates the seam is real. The two are cross-checked against each
other in `tests/test_reader.py` — same pixels, same timestamps — so a bug in
either one is visible as a disagreement rather than as a plausible wrong answer.

## Note on the metadata cache

The same benchmark measures probing at **72.6 ms** (two ffprobe passes, the
second walking the packet index) against **1.6 ms** for a cache hit. That 45x
gap is what `VideoMetadataCache` exists for; the cache is keyed by content, so
it survives a rename and is invalidated by an edit.
