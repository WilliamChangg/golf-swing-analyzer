"""Contracts for identifying cached analysis artifacts.

Analysis results are expensive and video files are large, so derived artifacts
are cached against the *content* of their input rather than its path: renaming a
clip must not invalidate its analysis, and replacing a clip in place must.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class HashAlgorithm(StrEnum):
    """How a `ContentKey` digest was computed.

    SHA256          - the whole file. Use when the answer must be an integrity
                      check, as it is for model weights.
    SHA256_SAMPLED  - size, plus sha256 over three fixed windows of the file.
                      **This is not an integrity check.** It exists because
                      hashing every byte of a multi-gigabyte recording on every
                      open costs seconds, and a cache key only has to change
                      when the file changes, not prove that it did not.
    """

    SHA256 = "sha256"
    SHA256_SAMPLED = "sha256-sampled-v1"


class ContentKey(BaseModel):
    """Identity of a file's contents, used to key derived artifacts.

    The algorithm travels with the digest so a cache written under one scheme is
    never silently compared against a digest produced by another.
    """

    algorithm: HashAlgorithm
    digest: str = Field(description="Lowercase hex sha256 digest.")
    size_bytes: int = Field(description="File size at the time the digest was taken.")

    def as_path_segment(self) -> str:
        """A filesystem-safe directory name for this key.

        The algorithm is part of the name, so changing the scheme produces a
        different directory instead of colliding with entries written under the
        old one.
        """
        return f"{self.algorithm.value}-{self.digest[:32]}"
