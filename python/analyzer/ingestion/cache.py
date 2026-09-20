"""Caches keyed by content rather than by path.

Two caches with different lifetimes live here.

`FrameCache` is in memory and holds decoded frames for the duration of a read.
Scrubbing a timeline or running several metrics over the same window would
otherwise decode the same frames repeatedly, and decoding is the dominant cost
of every later phase.

`VideoMetadataCache` is on disk and holds probe results between runs. Both are
keyed by a `ContentKey`, so renaming a clip keeps its cached work and replacing
a clip in place discards it -- which is the behaviour a path-keyed cache gets
backwards in both directions.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from analyzer.contracts.cache import ContentKey
from analyzer.contracts.video import VIDEO_SCHEMA_VERSION, VideoMetadata
from analyzer.paths import cache_dir

# A 1080p BGR frame is ~6 MB and a 4K frame ~25 MB, so a cache bounded by frame
# *count* is either useless on large footage or an out-of-memory error on it.
# 512 MB holds ~85 frames at 1080p, which covers a full swing at 120 fps.
DEFAULT_FRAME_CACHE_BYTES = 512 * 1024 * 1024


class CacheableFrame(Protocol):
    """What `FrameCache` needs to know about whatever it stores.

    Declared structurally so the cache does not import the reader, which imports
    the cache.
    """

    @property
    def index(self) -> int: ...

    @property
    def nbytes(self) -> int: ...


@dataclass(frozen=True)
class FrameCacheStats:
    """Measured cache behaviour, for reporting rather than for decisions."""

    hits: int
    misses: int
    evictions: int
    used_bytes: int
    capacity_bytes: int

    @property
    def hit_rate(self) -> float | None:
        """None rather than 0.0 when nothing has been looked up yet."""
        total = self.hits + self.misses
        return self.hits / total if total else None


class FrameCache:
    """Least-recently-used cache of decoded frames, bounded by total bytes.

    Not thread-safe: a frame source owns one and is itself used from a single
    thread. Sharing one across threads would need a lock, and adding one now
    would be locking against a caller that does not exist.
    """

    def __init__(self, max_bytes: int = DEFAULT_FRAME_CACHE_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._max_bytes = max_bytes
        self._entries: OrderedDict[int, CacheableFrame] = OrderedDict()
        self._used_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, index: int) -> CacheableFrame | None:
        entry = self._entries.get(index)
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(index)
        self._hits += 1
        return entry

    def put(self, frame: CacheableFrame) -> None:
        """Store a frame, evicting least-recently-used entries to stay in budget.

        A frame larger than the whole budget is simply not cached: evicting
        everything to hold one frame that will itself be evicted next is worse
        than not caching it at all.
        """
        if frame.nbytes > self._max_bytes:
            return

        existing = self._entries.pop(frame.index, None)
        if existing is not None:
            self._used_bytes -= existing.nbytes

        self._entries[frame.index] = frame
        self._used_bytes += frame.nbytes

        while self._used_bytes > self._max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._used_bytes -= evicted.nbytes
            self._evictions += 1

    def clear(self) -> None:
        self._entries.clear()
        self._used_bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> FrameCacheStats:
        return FrameCacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            used_bytes=self._used_bytes,
            capacity_bytes=self._max_bytes,
        )


class VideoMetadataCache:
    """On-disk store of probe results, keyed by content.

    A probe costs two ffprobe passes, the second of which walks the whole packet
    index; on long footage that is seconds, and the desktop app re-probes every
    time a clip is opened.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory or (cache_dir() / "video-metadata")

    def _path_for(self, key: ContentKey) -> Path:
        return self._directory / f"{key.as_path_segment()}.json"

    def load(self, key: ContentKey) -> VideoMetadata | None:
        """Return the cached metadata, or None if there is nothing usable.

        A cache entry written by an older contract is discarded rather than
        coerced: the whole point of the schema version is that a field may have
        changed meaning, and re-probing is cheap next to getting that wrong.
        """
        path = self._path_for(key)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

        if not isinstance(raw, dict) or raw.get("schema_version") != VIDEO_SCHEMA_VERSION:
            return None

        try:
            metadata = VideoMetadata.model_validate(raw)
        except ValueError:
            return None

        # The key is derived from content, so an entry filed under a key whose
        # digest does not match what it contains is a corrupted cache, not a hit.
        if metadata.content_key != key:
            return None
        return metadata

    def store(self, metadata: VideoMetadata) -> None:
        """Write an entry, treating any failure as a cache miss rather than an error.

        A read-only or full cache directory must not stop an analysis that has
        already succeeded.
        """
        path = self._path_for(metadata.content_key)
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            # Write-then-rename, so a crash mid-write cannot leave a truncated
            # entry that later reads would have to defend against.
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(metadata.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError:
            return
