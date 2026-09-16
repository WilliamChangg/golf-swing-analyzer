"""Container inspection via ffprobe.

ffprobe is the authoritative source for everything in this module. Higher-level
decoding wrappers expose a convenient but lossy view of a file -- OpenCV, for
instance, reports a single `CAP_PROP_FPS` whether or not the clip has a constant
frame rate, and silently applies or ignores the display matrix depending on
build and platform. Both of those are exactly the facts that must be right, so
they are read from the container rather than from a decoder.

Two ffprobe passes are made. The first reads the format and stream headers. The
second reads the packet index, which is where per-frame presentation timestamps
live; it demuxes but does not decode, so it costs a fraction of a real read.

Timestamps are handled in integer time-base ticks for as long as possible.
ffprobe's `pts_time` is printed to six decimal places, which introduces rounding
of the same order as the jitter that distinguishes constant from variable frame
rate -- a 30 fps clip in a 1/15360 time base prints intervals of both 0.033333
and 0.033334. In ticks the same clip is exactly 512 every time, so the
constant-rate verdict needs no tolerance to reach.
"""

from __future__ import annotations

import itertools
import json
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from analyzer.contracts.video import (
    IntervalStats,
    RotationDegrees,
    TimestampSource,
    VideoMetadata,
    VideoStreamInfo,
    VideoTiming,
)
from analyzer.hashing import content_key

# Generous: the packet pass walks the whole container index, which for a long
# recording on a slow disk is not instant. It is still bounded, because a hung
# ffprobe must not become a hung engine.
_PROBE_TIMEOUT_S = 120

_INSTALL_HINT = "Install FFmpeg (`brew install ffmpeg` on macOS) and ensure ffprobe is on PATH."

# Intervals within one tick of the median are rounding, not a rate change: one
# tick is the finest difference the container's own clock can express.
_TICK_TOLERANCE = 1


class ProbeError(RuntimeError):
    """A file could not be probed, or holds nothing this system can analyse.

    Carries a remediation because every case that reaches it is something the
    user can act on -- a wrong file, a truncated download, a missing FFmpeg --
    rather than an internal fault.
    """

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


@dataclass(frozen=True)
class FrameIndex:
    """Per-frame presentation times, in presentation order.

    Packets come out of a container in *decode* order, which for any stream with
    B-frames is not the order the frames are shown in. They are sorted by
    presentation timestamp here, once, so every consumer downstream can treat
    index `i` as the i-th frame a decoder will hand it.
    """

    source: TimestampSource
    timestamps_s: tuple[float, ...]
    keyframes: tuple[bool, ...]

    def __len__(self) -> int:
        return len(self.timestamps_s)

    def relative_timestamps_s(self) -> tuple[float, ...]:
        """Timestamps rebased so the first frame is at t=0.

        Analysis wants elapsed time within the clip; a container start offset is
        an artefact of how the file was cut and carries no meaning for a swing.
        """
        if not self.timestamps_s:
            return ()
        origin = self.timestamps_s[0]
        return tuple(t - origin for t in self.timestamps_s)


@dataclass(frozen=True)
class VideoProbe:
    """Everything one probe of a file produced."""

    metadata: VideoMetadata
    index: FrameIndex


def _ffprobe_binary() -> str:
    path = shutil.which("ffprobe")
    if path is None:
        raise ProbeError("ffprobe was not found on PATH.", remediation=_INSTALL_HINT)
    return path


def _run_ffprobe(args: list[str], path: Path) -> str:
    try:
        proc = subprocess.run(  # noqa: S603
            [_ffprobe_binary(), "-v", "error", *args, str(path)],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(
            f"ffprobe did not finish within {_PROBE_TIMEOUT_S}s for {path.name}.",
            remediation="The file may be on a slow or disconnected volume. Copy it locally and retry.",
        ) from exc
    except OSError as exc:
        raise ProbeError(f"Could not run ffprobe: {exc}", remediation=_INSTALL_HINT) from exc

    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        reason = detail[-1] if detail else f"exit code {proc.returncode}"
        raise ProbeError(
            f"ffprobe could not read {path.name}: {reason}",
            remediation="The file may be corrupt, incomplete, or not a media file. Try re-copying it from the camera.",
        )

    return proc.stdout


def check_readable(path: Path) -> None:
    """Reject inputs before spending an ffprobe call on them.

    These produce clearer errors than ffprobe's would, and a zero-byte file in
    particular is common enough -- an interrupted copy from a phone -- to be
    worth naming precisely.
    """
    if not path.exists():
        raise ProbeError(
            f"No file exists at {path}.",
            remediation="Check the path; the file may have been moved or the volume unmounted.",
        )
    if path.is_dir():
        raise ProbeError(
            f"{path} is a directory, not a video file.",
            remediation="Select a video file rather than a folder.",
        )
    if path.stat().st_size == 0:
        raise ProbeError(
            f"{path.name} is empty (0 bytes).",
            remediation="The copy from the camera was probably interrupted. Copy the file again.",
        )


def _rational(raw: str | None) -> Fraction | None:
    """Parse an ffprobe rational such as '30/1'. '0/0' means 'not stated'."""
    if not raw or raw == "N/A":
        return None
    try:
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _float_or_none(raw: object) -> float | None:
    if raw is None or raw == "N/A":
        return None
    try:
        return float(cast(str | float, raw))
    except (TypeError, ValueError):
        return None


def _int_or_none(raw: object) -> int | None:
    value = _float_or_none(raw)
    return None if value is None else int(value)


def _quantize_rotation(degrees: float) -> tuple[RotationDegrees, str | None]:
    """Snap a rotation to the four orientations a frame source can produce.

    Arbitrary angles are legal in a display matrix but no decoder in this
    pipeline resamples for them, so anything that is not a quarter turn is
    rounded and said to have been rounded.
    """
    normalised = degrees % 360
    nearest = round(normalised / 90) * 90 % 360
    warning = None
    if abs(normalised - nearest) > 1.0 and abs(normalised - nearest) < 359.0:
        warning = (
            f"The display matrix requests {normalised:.1f} degrees, which is not a quarter "
            f"turn. It was rounded to {nearest} degrees; the frame is not resampled."
        )
    return cast(RotationDegrees, nearest), warning


def read_rotation(stream: dict[str, Any]) -> tuple[RotationDegrees, str | None, str | None]:
    """Extract the display rotation, in counter-clockwise degrees.

    ffprobe reports the display matrix rotation counter-clockwise. That was
    settled by experiment rather than by reading the documentation, which
    describes the sign ambiguously: a clip built with `-display_rotation 90` and
    decoded with ffmpeg's default autorotation is pixel-identical to the
    unrotated source rotated 90 degrees counter-clockwise. See
    `scripts/make_video_fixtures.py` and `tests/test_reader.py`, which assert
    both halves of that so an ffmpeg upgrade that changed the convention would
    fail the suite rather than quietly transpose every future measurement.

    The older `rotate` tag is clockwise, and is negated on the way in.
    """
    for side_data in stream.get("side_data_list") or []:
        raw = _float_or_none(side_data.get("rotation"))
        if raw is not None:
            degrees, warning = _quantize_rotation(raw)
            return degrees, "display_matrix", warning

    tag = _float_or_none((stream.get("tags") or {}).get("rotate"))
    if tag is not None:
        degrees, warning = _quantize_rotation(-tag)
        return degrees, "rotate_tag", warning

    return 0, None, None


def _select_video_stream(streams: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Pick the stream to analyse, skipping cover art.

    An attached picture is carried as a video stream of one frame. Selecting it
    would report a still image's geometry for the whole file.
    """
    candidates = [
        s
        for s in streams
        if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
    ]
    if not candidates:
        raise ProbeError(
            f"{path.name} contains no video stream.",
            remediation="Select a video recording. Audio-only files and still images cannot be analysed.",
        )
    return candidates[0]


def parse_packet_index(csv: str) -> tuple[list[int], list[bool]]:
    """Parse `pts,flags` rows into presentation-ordered ticks and keyframe flags.

    Rows without a timestamp are dropped: a packet whose presentation time is
    unknown cannot be placed on the timeline, and guessing where it belongs
    would corrupt every interval around it.
    """
    entries: list[tuple[int, bool]] = []
    for line in csv.splitlines():
        row = line.strip()
        if not row:
            continue
        parts = row.split(",")
        pts = _int_or_none(parts[0])
        if pts is None:
            continue
        flags = parts[1] if len(parts) > 1 else ""
        entries.append((pts, "K" in flags))

    entries.sort(key=lambda item: item[0])
    return [pts for pts, _ in entries], [key for _, key in entries]


def _interval_stats(deltas: list[int], time_base: Fraction) -> IntervalStats:
    median = statistics.median(deltas)
    irregular = sum(1 for d in deltas if abs(d - median) > _TICK_TOLERANCE)
    return IntervalStats(
        median_s=float(median * time_base),
        min_s=float(min(deltas) * time_base),
        max_s=float(max(deltas) * time_base),
        quantum_s=float(time_base),
        irregular_count=irregular,
        irregular_fraction=irregular / len(deltas),
    )


def timing_from_ticks(
    ticks: list[int],
    time_base: Fraction,
    *,
    nominal_fps: float | None,
    container_duration_s: float | None,
) -> VideoTiming:
    first = float(ticks[0] * time_base)
    last = float(ticks[-1] * time_base)
    span = last - first

    deltas = [b - a for a, b in itertools.pairwise(ticks)]
    intervals = _interval_stats(deltas, time_base) if deltas else None

    # How long the final frame stays on screen is not recorded anywhere, so the
    # previous interval is the best available estimate for it.
    duration = span + float(deltas[-1] * time_base) if deltas else (container_duration_s or 0.0)

    return VideoTiming(
        source=TimestampSource.PACKET_PTS,
        frame_count=len(ticks),
        first_timestamp_s=first,
        last_timestamp_s=last,
        timestamp_span_s=span,
        duration_s=duration,
        container_duration_s=container_duration_s,
        nominal_fps=nominal_fps,
        measured_fps=(len(ticks) - 1) / span if span > 0 else None,
        intervals=intervals,
        is_vfr=intervals.irregular_count > 0 if intervals else None,
    )


def _timing_from_container_rate(
    *,
    frame_count: int,
    nominal_fps: float,
    container_duration_s: float | None,
) -> VideoTiming:
    """Fallback when the container yielded no usable timestamps.

    `is_vfr` stays None rather than False: nothing was measured, so nothing is
    known, and reporting False here would be the exact false green this project
    exists to avoid.
    """
    interval = 1.0 / nominal_fps
    span = (frame_count - 1) * interval if frame_count > 1 else 0.0
    return VideoTiming(
        source=TimestampSource.CONTAINER_RATE,
        frame_count=frame_count,
        first_timestamp_s=0.0,
        last_timestamp_s=span,
        timestamp_span_s=span,
        duration_s=frame_count * interval,
        container_duration_s=container_duration_s,
        nominal_fps=nominal_fps,
        measured_fps=None,
        intervals=None,
        is_vfr=None,
    )


def _collect_warnings(
    timing: VideoTiming, stream_info: VideoStreamInfo, declared_frames: int | None
) -> list[str]:
    warnings: list[str] = []

    if timing.is_vfr and timing.intervals is not None:
        stats = timing.intervals
        warnings.append(
            f"Variable frame rate: {stats.irregular_count} of "
            f"{timing.frame_count - 1} intervals differ from the median by more than one "
            f"time-base tick (from {stats.min_s * 1000:.2f} ms to {stats.max_s * 1000:.2f} ms). "
            "Frame times must come from presentation timestamps; frame_index / fps is not "
            "valid for this clip."
        )

    if timing.source is TimestampSource.CONTAINER_RATE:
        rate = f"{timing.nominal_fps:.3f} fps" if timing.nominal_fps else "declared frame rate"
        warnings.append(
            f"No presentation timestamps were found. Frame times were synthesised from the "
            f"container's {rate}, so a variable frame rate could not be detected and would go "
            "unnoticed."
        )

    if stream_info.rotation_ccw_degrees != 0:
        geometry = (
            f"stored {stream_info.coded_width}x{stream_info.coded_height}, presented "
            f"{stream_info.display_width}x{stream_info.display_height}"
            if stream_info.rotation_ccw_degrees in (90, 270)
            else f"the frame size is unchanged at {stream_info.coded_width}x{stream_info.coded_height}"
        )
        warnings.append(
            f"The container requests a {stream_info.rotation_ccw_degrees} degree "
            f"counter-clockwise display rotation ({geometry}). Frames are returned already "
            "rotated."
        )

    if declared_frames is not None and declared_frames != timing.frame_count:
        warnings.append(
            f"The container declares {declared_frames} frames but {timing.frame_count} were "
            "found in the packet index. The index is used, since it is what a decoder will "
            "produce."
        )

    if timing.container_duration_s is not None and timing.intervals is not None:
        disagreement = abs(timing.container_duration_s - timing.duration_s)
        if disagreement > 2 * timing.intervals.median_s:
            warnings.append(
                f"The container declares a duration of {timing.container_duration_s:.3f} s but "
                f"the timestamps span {timing.duration_s:.3f} s. Timestamps are used."
            )

    return warnings


def probe(path: Path, *, full_hash: bool = False) -> VideoProbe:
    """Read everything knowable about a video file without decoding it.

    Raises `ProbeError` for anything the user can act on: a missing or empty
    file, a corrupt container, or a media file with no video in it.
    """
    path = path.expanduser()
    check_readable(path)

    raw = json.loads(
        _run_ffprobe(
            [
                "-print_format",
                "json",
                "-select_streams",
                "v",
                "-show_format",
                "-show_streams",
            ],
            path,
        )
    )

    stream = _select_video_stream(raw.get("streams") or [], path)
    fmt: dict[str, Any] = raw.get("format") or {}

    time_base = _rational(stream.get("time_base")) or Fraction(1, 1000)
    rotation, rotation_source, rotation_warning = read_rotation(stream)

    coded_width = _int_or_none(stream.get("width")) or 0
    coded_height = _int_or_none(stream.get("height")) or 0
    if coded_width <= 0 or coded_height <= 0:
        raise ProbeError(
            f"{path.name} declares a video stream with no usable dimensions "
            f"({coded_width}x{coded_height}).",
            remediation="The file header is probably damaged. Re-copy it from the camera.",
        )

    swapped = rotation in (90, 270)
    stream_info = VideoStreamInfo(
        codec_name=str(stream.get("codec_name") or "unknown"),
        codec_long_name=stream.get("codec_long_name"),
        profile=stream.get("profile"),
        pix_fmt=stream.get("pix_fmt"),
        coded_width=coded_width,
        coded_height=coded_height,
        display_width=coded_height if swapped else coded_width,
        display_height=coded_width if swapped else coded_height,
        rotation_ccw_degrees=rotation,
        rotation_source=rotation_source,
        sample_aspect_ratio=stream.get("sample_aspect_ratio"),
        bit_rate=_int_or_none(stream.get("bit_rate")),
        time_base=str(stream.get("time_base") or f"{time_base.numerator}/{time_base.denominator}"),
    )

    avg_rate = _rational(stream.get("avg_frame_rate"))
    nominal_fps = float(avg_rate) if avg_rate else None
    container_duration = _float_or_none(stream.get("duration")) or _float_or_none(
        fmt.get("duration")
    )
    declared_frames = _int_or_none(stream.get("nb_frames"))

    ticks, keyframes = parse_packet_index(
        _run_ffprobe(
            ["-select_streams", "v:0", "-show_entries", "packet=pts,flags", "-of", "csv=p=0"],
            path,
        )
    )

    if ticks:
        timing = timing_from_ticks(
            ticks,
            time_base,
            nominal_fps=nominal_fps,
            container_duration_s=container_duration,
        )
        index = FrameIndex(
            source=TimestampSource.PACKET_PTS,
            timestamps_s=tuple(float(t * time_base) for t in ticks),
            keyframes=tuple(keyframes),
        )
    else:
        frame_count = declared_frames or (
            int(container_duration * nominal_fps) if container_duration and nominal_fps else 0
        )
        if not nominal_fps or frame_count <= 0:
            raise ProbeError(
                f"{path.name} has a video stream with neither presentation timestamps nor a "
                "usable frame rate, so no timeline can be established.",
                remediation="Re-encode the clip with FFmpeg (`ffmpeg -i in.mp4 -c:v libx264 out.mp4`) to rebuild its timing.",
            )
        timing = _timing_from_container_rate(
            frame_count=frame_count,
            nominal_fps=nominal_fps,
            container_duration_s=container_duration,
        )
        index = FrameIndex(
            source=TimestampSource.CONTAINER_RATE,
            timestamps_s=tuple(i / nominal_fps for i in range(frame_count)),
            keyframes=(True,) + (False,) * (frame_count - 1),
        )

    if timing.frame_count == 0:
        raise ProbeError(
            f"{path.name} contains a video stream with no frames.",
            remediation="The recording is empty or was truncated before any frame was written.",
        )

    warnings = _collect_warnings(timing, stream_info, declared_frames)
    if rotation_warning:
        warnings.append(rotation_warning)

    metadata = VideoMetadata(
        path=str(path.resolve()),
        file_size_bytes=path.stat().st_size,
        content_key=content_key(path, full=full_hash),
        probed_at=datetime.now(UTC),
        container_format=str(fmt.get("format_name") or "unknown"),
        stream=stream_info,
        timing=timing,
        warnings=warnings,
    )
    return VideoProbe(metadata=metadata, index=index)
