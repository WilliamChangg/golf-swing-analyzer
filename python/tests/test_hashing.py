"""Content hashing tests.

The sampled key is the one worth testing carefully. It exists to be fast, which
means it deliberately does not read every byte, which means the interesting
question is not "does it detect a change" in general but "does it detect a
change *where it looks*" -- and that the places it looks are stated honestly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from analyzer.contracts.cache import HashAlgorithm
from analyzer.hashing import content_key, sha256_file

CONTENT = b"fake video bytes"
CONTENT_SHA = hashlib.sha256(CONTENT).hexdigest()

# Larger than the three 1 MiB sample windows, so head, middle and tail are
# genuinely separate regions of the file.
LARGE_SIZE = 5 * 1024 * 1024


@pytest.fixture
def large_file(tmp_path: Path) -> Path:
    target = tmp_path / "large.bin"
    target.write_bytes(bytes(LARGE_SIZE))
    return target


class TestSha256File:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        target.write_bytes(CONTENT)
        assert sha256_file(target) == CONTENT_SHA

    def test_handles_file_larger_than_one_chunk(self, tmp_path: Path) -> None:
        """The reader streams in 1 MiB chunks; multi-chunk files must still hash."""
        payload = b"x" * (3 * 1024 * 1024 + 7)
        target = tmp_path / "big.bin"
        target.write_bytes(payload)
        assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


class TestContentKey:
    def test_full_key_is_a_plain_sha256_of_the_file(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        target.write_bytes(CONTENT)
        key = content_key(target, full=True)
        assert key.algorithm is HashAlgorithm.SHA256
        assert key.digest == CONTENT_SHA

    def test_sampled_key_is_labelled_as_sampled(self, tmp_path: Path) -> None:
        """It must never be mistaken for an integrity check, so it is not named like one."""
        target = tmp_path / "blob.bin"
        target.write_bytes(CONTENT)
        key = content_key(target)
        assert key.algorithm is HashAlgorithm.SHA256_SAMPLED
        assert key.digest != CONTENT_SHA

    def test_records_the_file_size(self, large_file: Path) -> None:
        assert content_key(large_file).size_bytes == LARGE_SIZE

    def test_is_stable_across_repeated_calls(self, large_file: Path) -> None:
        assert content_key(large_file) == content_key(large_file)

    def test_is_stable_across_a_rename(self, large_file: Path) -> None:
        """Keying by content is the whole point: renaming must keep cached work."""
        before = content_key(large_file)
        renamed = large_file.with_name("renamed.bin")
        large_file.rename(renamed)
        assert content_key(renamed) == before

    def test_changes_when_the_size_changes(self, large_file: Path) -> None:
        before = content_key(large_file)
        with large_file.open("ab") as handle:
            handle.write(b"more")
        assert content_key(large_file) != before

    @pytest.mark.parametrize(
        ("label", "offset"),
        [("head", 0), ("middle", LARGE_SIZE // 2), ("tail", LARGE_SIZE - 16)],
    )
    def test_detects_a_change_in_each_sampled_window(
        self, large_file: Path, label: str, offset: int
    ) -> None:
        before = content_key(large_file)
        with large_file.open("r+b") as handle:
            handle.seek(offset)
            handle.write(b"\xff" * 8)
        assert content_key(large_file) != before, label

    def test_small_files_are_sampled_whole(self, tmp_path: Path) -> None:
        """Below the window budget there is nothing to skip, so every byte counts."""
        target = tmp_path / "small.bin"
        target.write_bytes(bytes(1024))
        before = content_key(target)

        with target.open("r+b") as handle:
            handle.seek(512)
            handle.write(b"\x01")
        assert content_key(target) != before

    def test_full_and_sampled_keys_are_different_keys(self, large_file: Path) -> None:
        """Mixing the two under one cache entry would compare unlike things."""
        sampled = content_key(large_file)
        full = content_key(large_file, full=True)
        assert sampled.digest != full.digest
        assert sampled.as_path_segment() != full.as_path_segment()

    def test_path_segment_names_the_algorithm(self, large_file: Path) -> None:
        segment = content_key(large_file).as_path_segment()
        assert segment.startswith(HashAlgorithm.SHA256_SAMPLED.value)
        assert "/" not in segment
