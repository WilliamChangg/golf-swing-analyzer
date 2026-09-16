"""Content hashing for artifact verification and cache keys.

Two different jobs live here because they have genuinely different requirements.
Verifying a downloaded model must read every byte -- a partial hash would defeat
the point. Keying a cache entry for a 2 GB recording must not, because the cost
would be paid on every open and the key only has to change when the file does.
The distinction is carried in `HashAlgorithm` rather than left implicit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from analyzer.contracts.cache import ContentKey, HashAlgorithm

_CHUNK_BYTES = 1024 * 1024

# Window size and count for the sampled key. Three 1 MiB windows bound the work
# at ~3 MiB regardless of file size; the head window covers container headers,
# which is where an edit or re-mux almost always shows up.
_SAMPLE_BYTES = 1024 * 1024
_SAMPLE_COUNT = 3


def sha256_file(path: Path) -> str:
    """Digest an entire file, streaming it rather than reading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_offsets(size: int) -> list[int]:
    """Head, middle, and tail window offsets, clamped for small files."""
    if size <= _SAMPLE_BYTES * _SAMPLE_COUNT:
        return [0]
    return [0, (size - _SAMPLE_BYTES) // 2, size - _SAMPLE_BYTES]


def content_key(path: Path, *, full: bool = False) -> ContentKey:
    """Compute a cache key for a file's contents.

    With `full=True` the digest covers every byte. Otherwise it covers the file
    size and three windows of it -- see `HashAlgorithm.SHA256_SAMPLED` for why,
    and for why the result must not be treated as an integrity check.
    """
    size = path.stat().st_size

    if full:
        return ContentKey(algorithm=HashAlgorithm.SHA256, digest=sha256_file(path), size_bytes=size)

    digest = hashlib.sha256()
    # The size and each window's offset are mixed in as well as its bytes, so
    # two files that happen to share sampled content but differ in length or
    # layout cannot collide.
    digest.update(f"size={size}".encode())
    with path.open("rb") as handle:
        for offset in _sample_offsets(size):
            handle.seek(offset)
            window = handle.read(_SAMPLE_BYTES)
            digest.update(f"@{offset}+{len(window)}".encode())
            digest.update(window)

    return ContentKey(
        algorithm=HashAlgorithm.SHA256_SAMPLED, digest=digest.hexdigest(), size_bytes=size
    )
