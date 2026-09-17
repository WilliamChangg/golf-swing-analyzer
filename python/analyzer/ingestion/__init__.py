"""Video ingestion: container inspection and frame decoding.

The two facts this package exists to get right are rotation and frame timing.
Both are recorded in the container, both are routinely mishandled by decoding
wrappers, and both fail silently rather than loudly -- a wrongly-rotated clip
produces a sideways skeleton and a clip timed by `frame / fps` produces a
confidently wrong tempo. See `probe.py` for how each is established.
"""

from __future__ import annotations

from pathlib import Path

from analyzer.contracts.video import VideoMetadata
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


__all__ = [
    "DecodeBackend",
    "DecodeError",
    "FFmpegPipeFrameSource",
    "FrameCache",
    "FrameIndex",
    "FrameSource",
    "OpenCVFrameSource",
    "ProbeError",
    "VideoFrame",
    "VideoMetadata",
    "VideoMetadataCache",
    "VideoProbe",
    "apply_display_rotation",
    "open_video",
    "probe",
    "probe_video",
]
