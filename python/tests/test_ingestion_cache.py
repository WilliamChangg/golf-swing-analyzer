"""Cache tests.

Both caches exist to return work that was done earlier. The failure that
matters for both is therefore the same one: returning something that does not
correspond to the input, which is worse than doing the work again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from analyzer.contracts.cache import ContentKey
from analyzer.contracts.video import VIDEO_SCHEMA_VERSION, VideoMetadata
from analyzer.ingestion import probe_video
from analyzer.ingestion.cache import FrameCache, VideoMetadataCache
from analyzer.ingestion.probe import ProbeError, probe
from tests.conftest import CFR_30FPS, requires_ffprobe

MEGABYTE = 1024 * 1024


@dataclass(frozen=True)
class FakeFrame:
    """Stands in for a decoded frame; the cache only needs its index and size."""

    index: int
    nbytes: int


class TestFrameCache:
    def test_returns_what_was_stored(self) -> None:
        cache = FrameCache(max_bytes=MEGABYTE)
        cache.put(FakeFrame(index=3, nbytes=100))
        assert cache.get(3) == FakeFrame(index=3, nbytes=100)

    def test_reports_a_miss_as_none(self) -> None:
        assert FrameCache(max_bytes=MEGABYTE).get(0) is None

    def test_evicts_the_least_recently_used_entry_when_full(self) -> None:
        cache = FrameCache(max_bytes=300)
        for index in range(3):
            cache.put(FakeFrame(index=index, nbytes=100))

        cache.get(0)  # 1 is now the least recently used
        cache.put(FakeFrame(index=3, nbytes=100))

        assert cache.get(1) is None
        assert cache.get(0) is not None
        assert cache.get(3) is not None

    def test_is_bounded_by_bytes_not_by_count(self) -> None:
        """Frame sizes vary tenfold between 720p and 4K, so a count bound is meaningless."""
        cache = FrameCache(max_bytes=1000)
        cache.put(FakeFrame(index=0, nbytes=900))
        cache.put(FakeFrame(index=1, nbytes=900))
        assert len(cache) == 1
        assert cache.stats.used_bytes == 900

    def test_a_frame_larger_than_the_budget_is_not_cached(self) -> None:
        """Evicting everything for one frame that is itself next out is worse than nothing."""
        cache = FrameCache(max_bytes=500)
        cache.put(FakeFrame(index=0, nbytes=100))
        cache.put(FakeFrame(index=1, nbytes=5000))

        assert cache.get(1) is None
        assert cache.get(0) is not None

    def test_replacing_an_entry_does_not_leak_its_bytes(self) -> None:
        cache = FrameCache(max_bytes=MEGABYTE)
        cache.put(FakeFrame(index=0, nbytes=400))
        cache.put(FakeFrame(index=0, nbytes=100))
        assert cache.stats.used_bytes == 100
        assert len(cache) == 1

    def test_counts_hits_misses_and_evictions(self) -> None:
        cache = FrameCache(max_bytes=200)
        cache.put(FakeFrame(index=0, nbytes=100))
        cache.put(FakeFrame(index=1, nbytes=100))
        cache.put(FakeFrame(index=2, nbytes=100))  # evicts 0
        cache.get(1)
        cache.get(99)

        stats = cache.stats
        assert (stats.hits, stats.misses, stats.evictions) == (1, 1, 1)

    def test_hit_rate_is_unknown_before_any_lookup(self) -> None:
        """Reporting 0% when nothing was asked for would be a made-up number."""
        assert FrameCache(max_bytes=MEGABYTE).stats.hit_rate is None

    def test_hit_rate_is_measured_once_there_is_something_to_measure(self) -> None:
        cache = FrameCache(max_bytes=MEGABYTE)
        cache.put(FakeFrame(index=0, nbytes=1))
        cache.get(0)
        cache.get(1)
        assert cache.stats.hit_rate == pytest.approx(0.5)

    def test_clear_releases_everything(self) -> None:
        cache = FrameCache(max_bytes=MEGABYTE)
        cache.put(FakeFrame(index=0, nbytes=100))
        cache.clear()
        assert len(cache) == 0
        assert cache.stats.used_bytes == 0

    def test_a_zero_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            FrameCache(max_bytes=0)


@requires_ffprobe
class TestVideoMetadataCache:
    @pytest.fixture
    def metadata(self) -> VideoMetadata:
        return probe(CFR_30FPS).metadata

    def test_round_trips(self, tmp_path: Path, metadata: VideoMetadata) -> None:
        cache = VideoMetadataCache(tmp_path)
        cache.store(metadata)
        assert cache.load(metadata.content_key) == metadata

    def test_an_absent_entry_is_a_miss(self, tmp_path: Path, metadata: VideoMetadata) -> None:
        assert VideoMetadataCache(tmp_path).load(metadata.content_key) is None

    def test_an_entry_from_an_older_schema_is_discarded(
        self, tmp_path: Path, metadata: VideoMetadata
    ) -> None:
        """A field may have changed meaning, so re-probing beats reinterpreting."""
        cache = VideoMetadataCache(tmp_path)
        cache.store(metadata)

        entry = next(tmp_path.glob("*.json"))
        raw = json.loads(entry.read_text())
        raw["schema_version"] = VIDEO_SCHEMA_VERSION + 1
        entry.write_text(json.dumps(raw))

        assert cache.load(metadata.content_key) is None

    def test_a_corrupt_entry_is_a_miss_not_an_error(
        self, tmp_path: Path, metadata: VideoMetadata
    ) -> None:
        cache = VideoMetadataCache(tmp_path)
        cache.store(metadata)
        next(tmp_path.glob("*.json")).write_text("{ not json")
        assert cache.load(metadata.content_key) is None

    def test_an_entry_filed_under_the_wrong_key_is_rejected(
        self, tmp_path: Path, metadata: VideoMetadata
    ) -> None:
        """The key is derived from content, so a mismatch means a corrupted cache."""
        cache = VideoMetadataCache(tmp_path)
        cache.store(metadata)

        impostor = ContentKey(
            algorithm=metadata.content_key.algorithm,
            digest=metadata.content_key.digest,
            size_bytes=metadata.content_key.size_bytes + 1,
        )
        entry = next(tmp_path.glob("*.json"))
        entry.rename(entry.with_name(f"{impostor.as_path_segment()}.json"))

        assert cache.load(impostor) is None

    def test_an_unwritable_directory_does_not_fail_the_probe(
        self, tmp_path: Path, metadata: VideoMetadata
    ) -> None:
        """A full or read-only cache must not sink an analysis that already succeeded."""
        blocked = tmp_path / "blocked"
        blocked.write_text("this is a file, not a directory")
        VideoMetadataCache(blocked / "inside").store(metadata)


@requires_ffprobe
class TestProbeVideoCaching:
    def test_a_second_probe_is_served_from_the_cache(self, tmp_path: Path) -> None:
        cache = VideoMetadataCache(tmp_path)
        first = probe_video(CFR_30FPS, cache=cache)
        second = probe_video(CFR_30FPS, cache=cache)
        assert first == second
        assert first.probed_at == second.probed_at  # the same probe, not a repeat

    def test_refresh_re_probes(self, tmp_path: Path) -> None:
        cache = VideoMetadataCache(tmp_path)
        first = probe_video(CFR_30FPS, cache=cache)
        refreshed = probe_video(CFR_30FPS, refresh=True, cache=cache)
        assert refreshed.probed_at > first.probed_at
        assert refreshed.timing == first.timing

    def test_a_copy_under_a_different_name_hits_the_same_entry(self, tmp_path: Path) -> None:
        cache = VideoMetadataCache(tmp_path / "cache")
        original = probe_video(CFR_30FPS, cache=cache)

        copy = tmp_path / "renamed.mp4"
        copy.write_bytes(CFR_30FPS.read_bytes())
        assert probe_video(copy, cache=cache).probed_at == original.probed_at

    def test_a_missing_file_is_rejected_before_it_is_hashed(self, tmp_path: Path) -> None:
        with pytest.raises(ProbeError, match="No file exists"):
            probe_video(tmp_path / "absent.mp4")
