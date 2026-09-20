"""Video ingestion: container inspection and frame decoding.

The two facts this package exists to get right are rotation and frame timing.
Both are recorded in the container, both are routinely mishandled by decoding
wrappers, and both fail silently rather than loudly -- a wrongly-rotated clip
produces a sideways skeleton and a clip timed by `frame / fps` produces a
confidently wrong tempo. See `probe.py` for how each is established.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from analyzer.contracts.video import SeekIndex, TimestampSource, VideoMetadata
from analyzer.hashing import content_key
from analyzer.ingestion.cache import FrameCache, VideoMetadataCache
from analyzer.ingestion.probe import (
    FrameIndex,
    ProbeError,
    VideoProbe,
    check_readable,
    probe,
)
from analyzer.ingestion.reader import (
    DecodeBackend,
    DecodeError,
    FFmpegPipeFrameSource,
    FrameSource,
    OpenCVFrameSource,
    VideoFrame,
    apply_display_rotation,
    open_video,
)


def probe_video(
    path: Path,
    *,
    refresh: bool = False,
    cache: VideoMetadataCache | None = None,
) -> VideoMetadata:
    """Read a video's metadata, using the on-disk cache when it is still valid.

    The cache is consulted by content, so the hit depends on the bytes of the
    file rather than on its name. `refresh=True` re-probes and overwrites the
    entry, which is what a user asking for a re-read expects.
    """
    path = path.expanduser()
    # Before hashing: a missing or empty file must surface as the ProbeError
    # that names what is wrong, not as whatever OSError `stat` raises first.
    check_readable(path)

    store = cache or VideoMetadataCache()
    key = content_key(path)

    if not refresh:
        cached = store.load(key)
        if cached is not None:
            return cached

    metadata = probe(path).metadata
    store.store(metadata)
    return metadata


def seek_index(path: Path) -> SeekIndex:
    """The map from a frame index to a time that will display that frame.

    **Not cached, and that is a measurement rather than an omission.** Reading
    the packet index is the 72.6 ms Phase 1 measured, against the seconds of
    pose extraction the same clip needs before any of these frames is worth
    looking at; caching it would add a schema to version and invalidate for a
    saving nothing can detect. The same reasoning Phase 3 applied to the filter.

    The seek target for frame `i` is the midpoint of the interval it is
    displayed for. The final frame has no recorded interval -- the container
    never says how long the last frame stays up -- so it borrows the last one
    observed, which is the assumption `VideoTiming.duration_s` already makes and
    is repeated here rather than invented.
    """
    path = path.expanduser()
    check_readable(path)

    probed = probe(path)
    times = probed.index.relative_timestamps_s()
    warnings: list[str] = []

    if probed.index.source is TimestampSource.CONTAINER_RATE:
        warnings.append(
            "This clip carried no presentation timestamps, so the frame times below were "
            "synthesised from the container's declared rate. Seeking by them assumes the "
            "frames are evenly spaced, which is the assumption that makes a variable-rate "
            "clip land on the wrong frame -- and nothing here can check it, because the "
            "evidence that would is the thing that is missing."
        )

    intervals = [later - earlier for earlier, later in pairwise(times)]
    last_interval = intervals[-1] if intervals else None

    # Midpoints. The final frame borrows the last observed interval; a clip of
    # one frame has no interval at all, and its own timestamp is then the only
    # answer available -- there is no boundary to be on the wrong side of.
    targets: list[float] = []
    for i, start in enumerate(times):
        span = intervals[i] if i < len(intervals) else last_interval
        targets.append(start + span / 2.0 if span is not None else start)

    return SeekIndex(
        path=str(path),
        content_key=content_key(path),
        source=probed.index.source,
        frame_count=len(times),
        timestamps_s=list(times),
        seek_targets_s=targets,
        last_interval_s=last_interval,
        warnings=warnings,
    )


__all__ = [
    "DecodeBackend",
    "DecodeError",
    "FFmpegPipeFrameSource",
    "FrameCache",
    "FrameIndex",
    "FrameSource",
    "OpenCVFrameSource",
    "ProbeError",
    "SeekIndex",
    "VideoFrame",
    "VideoMetadata",
    "VideoMetadataCache",
    "VideoProbe",
    "apply_display_rotation",
    "open_video",
    "probe",
    "probe_video",
    "seek_index",
]
