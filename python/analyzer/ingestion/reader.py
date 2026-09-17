"""Frame sources: decoding video into numpy arrays with correct time and orientation.

`FrameSource` is the seam. Everything downstream -- pose estimation, club
tracking, overlay rendering -- consumes frames through it and never touches a
decoder, so a decoder can be replaced without those layers noticing.

Two implementations exist because neither is strictly better:

`OpenCVFrameSource` decodes in-process through OpenCV's FFmpeg backend. It
supports random access, which a timeline scrubber needs, and it is the default.

`FFmpegPipeFrameSource` pipes raw frames from an ffmpeg subprocess. It exists
because OpenCV's bundled FFmpeg reports `VIDEO_ACCELERATION_NONE` on macOS
whatever is requested of it -- OpenCV only defines D3D11, VAAPI and MFX
acceleration, none of which is VideoToolbox -- so hardware decode is not
reachable in-process at all. It is sequential-first.

**Two invariants hold for both implementations**, because getting either wrong
is silent rather than loud:

*Frames come out in display orientation.* Both sources are told not to
auto-rotate, and both then apply the container's rotation through the same
function, so there is one rotation code path to be right about instead of two to
keep in agreement.

*Time comes from the container's presentation timestamps*, supplied by the probe
and never from `frame_index / fps`, which is wrong for the variable-rate clips
phones routinely produce.
"""

from __future__ import annotations

import io
import subprocess
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.video import RotationDegrees, VideoMetadata
from analyzer.ingestion.cache import DEFAULT_FRAME_CACHE_BYTES, FrameCache, FrameCacheStats
from analyzer.ingestion.probe import FrameIndex, ProbeError, VideoProbe, probe

# Counter-clockwise display rotation -> the OpenCV constant that performs it.
_ROTATION_OPS: dict[int, int] = {
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}

# Reading forward is far cheaper than seeking, which on a long-GOP stream has to
# restart from the preceding keyframe anyway. Below this many frames ahead, read
# rather than seek.
_FORWARD_SCAN_LIMIT = 48

# How long to wait for the ffmpeg subprocess, and its stderr reader, to finish.
_SHUTDOWN_TIMEOUT_S = 5


class DecodeBackend(StrEnum):
    """Which decoder a frame source is using.

    Reported rather than assumed: `FFMPEG_VIDEOTOOLBOX` means hardware decode
    was *requested*, and `OpenCVFrameSource.acceleration` records what OpenCV
    actually selected, which is not always what was asked for.
    """

    OPENCV = "opencv"
    FFMPEG_CPU = "ffmpeg_cpu"
    FFMPEG_VIDEOTOOLBOX = "ffmpeg_videotoolbox"


@dataclass(frozen=True)
class VideoFrame:
    """One decoded frame, in display orientation.

    `timestamp_s` is elapsed time from the first frame of the clip, taken from
    the container index. `decoder_timestamp_s` is what the decoder itself
    reported for the same frame, where it reports one at all; it is carried so
    the two can be compared rather than assumed to agree.
    """

    index: int
    timestamp_s: float
    image: NDArray[np.uint8]
    decoder_timestamp_s: float | None = None

    @property
    def nbytes(self) -> int:
        return int(self.image.nbytes)

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.image.shape[0]), int(self.image.shape[1])


class FrameSource(Protocol):
    """A decoded, display-oriented, correctly-timed view of a video file."""

    @property
    def metadata(self) -> VideoMetadata: ...

    def frame_at(self, index: int) -> VideoFrame:
        """Return one frame by presentation index. Raises IndexError if out of range."""
        ...

    def frames(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[VideoFrame]:
        """Iterate frames in presentation order."""
        ...

    def close(self) -> None: ...


class DecodeError(RuntimeError):
    """A frame could not be decoded from a file that probed successfully."""


def apply_display_rotation(
    image: NDArray[np.uint8], ccw_degrees: RotationDegrees
) -> NDArray[np.uint8]:
    """Rotate a stored frame into display orientation.

    `ccw_degrees` is counter-clockwise, matching ffmpeg's display matrix. The
    direction was established by experiment, and is pinned by a test that
    decodes a rotated fixture both ways -- once letting the decoder auto-rotate
    and once rotating here -- and asserts the pixels are identical.
    """
    if ccw_degrees == 0:
        return image
    return cast(NDArray[np.uint8], cv2.rotate(image, _ROTATION_OPS[ccw_degrees]))


def _validate_range(index: int, frame_count: int) -> None:
    if not 0 <= index < frame_count:
        raise IndexError(f"Frame {index} is out of range; the clip has {frame_count} frames.")


class OpenCVFrameSource:
    """Random-access frame source backed by OpenCV's FFmpeg decoder.

    OpenCV auto-rotates by default -- `CAP_PROP_ORIENTATION_AUTO` reads back as
    1 on a fresh capture -- which would silently double-rotate on top of the
    rotation applied here. It is turned off at open, and then *verified* against
    the first decoded frame's dimensions rather than trusted, because a build
    that ignores the property would otherwise produce sideways frames with no
    error anywhere.
    """

    def __init__(
        self,
        source: Path | VideoProbe,
        *,
        cache_bytes: int = DEFAULT_FRAME_CACHE_BYTES,
    ) -> None:
        self._probe = probe(source) if isinstance(source, Path) else source
        self._timestamps = self._probe.index.relative_timestamps_s()
        self._cache = FrameCache(cache_bytes)

        path = Path(self._probe.metadata.path)
        capture: Any = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise DecodeError(
                f"OpenCV could not open {path.name} for decoding, although ffprobe read it. "
                "The codec may not be supported by this OpenCV build."
            )

        self._capture = capture
        self._auto_rotate_disabled = bool(capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0))
        self._acceleration = int(capture.get(cv2.CAP_PROP_HW_ACCELERATION))
        self._next_index = 0
        self._closed = False
        self._inaccurate_seeks = 0
        # Resolved on the first decode, once there is a real frame to measure.
        self._rotation_to_apply: RotationDegrees | None = None

    @property
    def metadata(self) -> VideoMetadata:
        return self._probe.metadata

    @property
    def index(self) -> FrameIndex:
        return self._probe.index

    @property
    def backend(self) -> DecodeBackend:
        return DecodeBackend.OPENCV

    @property
    def acceleration(self) -> int:
        """The `VIDEO_ACCELERATION_*` value OpenCV actually selected."""
        return self._acceleration

    @property
    def cache_stats(self) -> FrameCacheStats:
        return self._cache.stats

    def _resolve_rotation(self, decoded_hw: tuple[int, int]) -> RotationDegrees:
        """Decide how much rotation this decoder still needs, by measuring.

        For a quarter turn the decoded frame's shape settles it outright: if it
        already matches the display dimensions, the decoder rotated despite
        being told not to, and rotating again here would be wrong.

        A half turn changes no dimension, so shape cannot settle it. There the
        property read-back is all there is, and it is used with that limitation
        stated rather than papered over.
        """
        stream = self.metadata.stream
        rotation = stream.rotation_ccw_degrees
        if rotation == 0:
            return 0

        if rotation in (90, 270):
            if decoded_hw == (stream.coded_height, stream.coded_width):
                return rotation
            if decoded_hw == (stream.display_height, stream.display_width):
                return 0
            raise DecodeError(
                f"Decoded frame is {decoded_hw[1]}x{decoded_hw[0]}, which matches neither the "
                f"coded size ({stream.coded_width}x{stream.coded_height}) nor the display size "
                f"({stream.display_width}x{stream.display_height}). Refusing to guess its "
                "orientation."
            )

        return rotation if self._auto_rotate_disabled else 0

    def _decode_next(self) -> VideoFrame:
        """Decode the frame at the current position and advance."""
        index = self._next_index
        ok, raw = self._capture.read()
        # Read the decoder's clock *after* the decode: beforehand it still holds
        # the previous frame's timestamp. Measured, because the two readings
        # differ by exactly one frame interval and either looks plausible.
        decoder_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        if not ok or raw is None:
            raise DecodeError(
                f"Decoding stopped at frame {index} of {len(self._timestamps)}. The file is "
                "probably truncated: the container index lists frames the stream does not contain."
            )

        image = cast(NDArray[np.uint8], raw)
        if self._rotation_to_apply is None:
            self._rotation_to_apply = self._resolve_rotation(
                (int(image.shape[0]), int(image.shape[1]))
            )

        self._next_index = index + 1
        return VideoFrame(
            index=index,
            timestamp_s=self._timestamps[index],
            image=apply_display_rotation(image, self._rotation_to_apply),
            decoder_timestamp_s=decoder_ms / 1000.0 if decoder_ms >= 0 else None,
        )

    def _seek(self, index: int) -> None:
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        self._next_index = index

    def _seek_landed(self, frame: VideoFrame, index: int) -> bool:
        """Whether a seek actually produced the frame that was asked for.

        Frame-number seeking on a long-GOP stream is approximate in a way that
        is invisible to the caller: OpenCV returns *a* frame and reports success
        whether or not it is the right one, and the wrong frame carries the
        right timestamp because the timestamp came from the index rather than
        from the decoder. Comparing the decoder's own clock against the index is
        what turns that into something detectable.
        """
        if frame.decoder_timestamp_s is None:
            return True  # nothing to check against; assume the seek was honoured

        intervals = self.metadata.timing.intervals
        if intervals is None:
            return True
        expected = self._probe.index.timestamps_s[index]
        return abs(frame.decoder_timestamp_s - expected) <= intervals.median_s / 2

    def _scan_to(self, index: int) -> VideoFrame:
        """Decode forward to `index`, caching everything passed on the way."""
        for _ in range(index - self._next_index):
            self._cache.put(self._decode_next())
        return self._decode_next()

    def frame_at(self, index: int) -> VideoFrame:
        _validate_range(index, len(self._timestamps))

        cached = self._cache.get(index)
        if cached is not None:
            return cast(VideoFrame, cached)

        # Reading forward is cheaper than seeking until the gap is large enough
        # that the decoder would have to restart from a keyframe anyway.
        ahead = index - self._next_index
        if 0 <= ahead <= _FORWARD_SCAN_LIMIT:
            frame = self._scan_to(index)
        else:
            self._seek(index)
            frame = self._decode_next()
            if not self._seek_landed(frame, index):
                # Scanning from the start always lands on the right frame. It is
                # slow, so it is the fallback rather than the default, but a slow
                # correct frame beats a fast wrong one.
                self._inaccurate_seeks += 1
                self._seek(0)
                frame = self._scan_to(index)

        self._cache.put(frame)
        return frame

    @property
    def inaccurate_seeks(self) -> int:
        """How many seeks landed on the wrong frame and had to be redone."""
        return self._inaccurate_seeks

    def frames(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[VideoFrame]:
        if step < 1:
            raise ValueError("step must be at least 1")
        end = len(self._timestamps) if stop is None else min(stop, len(self._timestamps))
        for index in range(start, end, step):
            yield self.frame_at(index)

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._cache.clear()
            self._closed = True

    def __enter__(self) -> OpenCVFrameSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class FFmpegPipeFrameSource:
    """Sequential frame source reading raw frames from an ffmpeg subprocess.

    This is the only way to reach VideoToolbox from this process, since OpenCV's
    bundled FFmpeg does not expose it. Hardware decode is requested, never
    assumed: if ffmpeg cannot honour `-hwaccel`, it decodes in software and says
    so on stderr, and the frames are identical either way.

    Random access backwards restarts the decode from the beginning, so this
    source suits a full sequential pass rather than a scrubber.
    """

    def __init__(
        self,
        source: Path | VideoProbe,
        *,
        hardware: bool = False,
        cache_bytes: int = DEFAULT_FRAME_CACHE_BYTES,
    ) -> None:
        self._probe = probe(source) if isinstance(source, Path) else source
        self._timestamps = self._probe.index.relative_timestamps_s()
        self._cache = FrameCache(cache_bytes)
        self._hardware = hardware
        self._path = Path(self._probe.metadata.path)

        stream = self._probe.metadata.stream
        # -noautorotate keeps ffmpeg from applying the display matrix, so the
        # frame size on the pipe is the coded size and rotation stays in one place.
        self._frame_bytes = stream.coded_width * stream.coded_height * 3
        self._coded_hw = (stream.coded_height, stream.coded_width)

        self._process: subprocess.Popen[bytes] | None = None
        self._stdout: io.BufferedReader | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._stderr_thread: threading.Thread | None = None
        self._next_index = 0
        self._closed = False
        self._start()

    @property
    def metadata(self) -> VideoMetadata:
        return self._probe.metadata

    @property
    def index(self) -> FrameIndex:
        return self._probe.index

    @property
    def backend(self) -> DecodeBackend:
        return DecodeBackend.FFMPEG_VIDEOTOOLBOX if self._hardware else DecodeBackend.FFMPEG_CPU

    @property
    def cache_stats(self) -> FrameCacheStats:
        return self._cache.stats

    def _command(self) -> list[str]:
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self._hardware:
            args += ["-hwaccel", "videotoolbox"]
        # -noautorotate leaves the display matrix unapplied, so frames arrive at
        # the coded size and rotation stays in `apply_display_rotation` alone.
        args += ["-noautorotate", "-i", str(self._path)]
        args += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        return args

    def _start(self) -> None:
        self._stop()
        try:
            process = subprocess.Popen(  # noqa: S603
                self._command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise DecodeError(f"Could not start ffmpeg: {exc}") from exc

        self._process = process
        self._next_index = 0
        # `Popen.stdout` is typed as the generic `IO[bytes]`, which does not
        # declare `readinto`. A binary pipe is always a BufferedReader, and
        # reading into a preallocated buffer is what keeps frames writable.
        self._stdout = cast(io.BufferedReader, process.stdout)

        # ffmpeg blocks once its stderr pipe fills, which on a file that
        # produces per-frame warnings would deadlock the read below.
        def drain(pipe: Any) -> None:
            for line in iter(pipe.readline, b""):
                self._stderr_tail.append(line.decode("utf-8", "replace").rstrip())

        self._stderr_thread = threading.Thread(target=drain, args=(process.stderr,), daemon=True)
        self._stderr_thread.start()

    def _stop(self) -> None:
        """Shut the subprocess down, in an order the drain thread survives.

        stderr belongs to the drain thread while it is running, so it is closed
        last: closing it underneath a blocked `readline` raises inside that
        thread instead of ending it. Terminating first gives the thread the EOF
        it is waiting for.
        """
        if self._process is None:
            return
        process, self._process = self._process, None
        thread, self._stderr_thread = self._stderr_thread, None

        if process.poll() is None:
            process.terminate()
        if process.stdout is not None:
            process.stdout.close()

        try:
            process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        if thread is not None:
            thread.join(timeout=_SHUTDOWN_TIMEOUT_S)
        if process.stderr is not None:
            process.stderr.close()

    def _decode_next(self) -> VideoFrame:
        if self._process is None or self._stdout is None:
            raise DecodeError("The ffmpeg frame source is closed.")

        index = self._next_index
        # Read straight into a numpy buffer. Going through `bytes` would make
        # the frame read-only, and every consumer that wants to draw on it would
        # have to copy; a pipe also has no obligation to hand over a whole frame
        # in one read, so a short read is topped up rather than trusted.
        buffer: NDArray[np.uint8] = np.empty(self._frame_bytes, dtype=np.uint8)
        view = memoryview(buffer.data).cast("B")
        filled = 0
        while filled < self._frame_bytes:
            got = self._stdout.readinto(view[filled:])
            if not got:
                break
            filled += got

        if filled < self._frame_bytes:
            tail = "\n".join(self._stderr_tail)
            raise DecodeError(
                f"ffmpeg produced {filled} bytes for frame {index}, expected "
                f"{self._frame_bytes}. The stream ended early."
                + (f"\nffmpeg said:\n{tail}" if tail else "")
            )

        self._next_index = index + 1
        return VideoFrame(
            index=index,
            timestamp_s=self._timestamps[index],
            image=apply_display_rotation(
                buffer.reshape((*self._coded_hw, 3)), self.metadata.stream.rotation_ccw_degrees
            ),
        )

    def frame_at(self, index: int) -> VideoFrame:
        _validate_range(index, len(self._timestamps))

        cached = self._cache.get(index)
        if cached is not None:
            return cast(VideoFrame, cached)

        if index < self._next_index:
            self._start()
        while self._next_index < index:
            self._cache.put(self._decode_next())

        frame = self._decode_next()
        self._cache.put(frame)
        return frame

    def frames(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[VideoFrame]:
        if step < 1:
            raise ValueError("step must be at least 1")
        end = len(self._timestamps) if stop is None else min(stop, len(self._timestamps))
        for index in range(start, end, step):
            yield self.frame_at(index)

    def close(self) -> None:
        if not self._closed:
            self._stop()
            self._cache.clear()
            self._closed = True

    def __enter__(self) -> FFmpegPipeFrameSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def open_video(
    path: Path,
    *,
    backend: DecodeBackend = DecodeBackend.OPENCV,
    cache_bytes: int = DEFAULT_FRAME_CACHE_BYTES,
) -> FrameSource:
    """Open a video for frame-by-frame reading.

    Defaults to OpenCV: it is in-process and supports random access. The
    VideoToolbox path is opt-in rather than automatic because it was measured to
    be slower for this pipeline, not merely unproven -- see
    `docs/decisions/ADR-0007-decode-backend.md`.
    """
    probed = probe(path)
    if backend is DecodeBackend.OPENCV:
        return OpenCVFrameSource(probed, cache_bytes=cache_bytes)
    return FFmpegPipeFrameSource(
        probed,
        hardware=backend is DecodeBackend.FFMPEG_VIDEOTOOLBOX,
        cache_bytes=cache_bytes,
    )


__all__ = [
    "DecodeBackend",
    "DecodeError",
    "FFmpegPipeFrameSource",
    "FrameSource",
    "OpenCVFrameSource",
    "ProbeError",
    "VideoFrame",
    "apply_display_rotation",
    "open_video",
]
