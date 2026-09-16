"""Shared test fixtures.

The video fixtures are committed rather than generated, so the ingestion suite
is hermetic: it does not depend on ffmpeg being installed to *build* its inputs,
nor on a particular ffmpeg version choosing to write a display matrix the way
the machine that produced them did. See `scripts/make_video_fixtures.py`.

Decoding them still needs ffmpeg, so those tests skip when it is absent -- with
one exception, below, so that "skipped" never silently becomes the CI result.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from analyzer.paths import ENV_CACHE_DIR

VIDEO_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "video"

CFR_30FPS = VIDEO_FIXTURE_DIR / "cfr_30fps.mp4"
VFR_30_TO_15FPS = VIDEO_FIXTURE_DIR / "vfr_30_to_15fps.mp4"
ROTATED_CCW90 = VIDEO_FIXTURE_DIR / "rotated_ccw90.mp4"
ROTATED_180 = VIDEO_FIXTURE_DIR / "rotated_180.mp4"
AUDIO_ONLY = VIDEO_FIXTURE_DIR / "audio_only.m4a"

# Facts the fixtures were built to have. Asserted against rather than read from
# the files, so a regenerated fixture that came out different fails loudly.
CFR_FRAME_COUNT = 60
CFR_FPS = 30.0
CFR_CODED_SIZE = (320, 240)
VFR_FRAME_COUNT = 45

# Set in CI. Without it a machine missing ffmpeg skips the decode tests, which
# is right locally and wrong in CI, where a skipped suite looks identical to a
# passing one.
ENV_REQUIRE_FFMPEG = "GSA_REQUIRE_FFMPEG"


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


requires_ffprobe = pytest.mark.skipif(not _have("ffprobe"), reason="ffprobe is not on PATH")
requires_ffmpeg = pytest.mark.skipif(
    not (_have("ffmpeg") and _have("ffprobe")), reason="ffmpeg is not on PATH"
)


@pytest.fixture(scope="session", autouse=True)
def _ffmpeg_required_in_ci() -> None:
    """Turn a silent skip into a failure where a skip would be misleading."""
    if os.environ.get(ENV_REQUIRE_FFMPEG) and not (_have("ffmpeg") and _have("ffprobe")):
        pytest.fail(
            f"{ENV_REQUIRE_FFMPEG} is set but ffmpeg/ffprobe are not on PATH, so the "
            "ingestion tests would be skipped rather than run."
        )


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the artifact cache at a temporary directory for every test.

    Without this, anything that probes a video writes into the developer's real
    cache directory, and a stale entry there could make a failing test pass.
    """
    directory = tmp_path / "cache"
    monkeypatch.setenv(ENV_CACHE_DIR, str(directory))
    yield directory


@pytest.fixture
def corrupt_video(tmp_path: Path) -> Path:
    """A file with a plausible name and nothing decodable in it."""
    target = tmp_path / "corrupt.mp4"
    target.write_bytes(b"\x00\x01\x02\x03not a container at all" * 64)
    return target


@pytest.fixture
def empty_video(tmp_path: Path) -> Path:
    """A zero-byte file, which is what an interrupted copy leaves behind."""
    target = tmp_path / "empty.mp4"
    target.touch()
    return target


@pytest.fixture
def truncated_video(tmp_path: Path) -> Path:
    """A real container cut off part-way through.

    The fixtures are written with `+faststart`, so the header precedes the
    frames -- exactly as a phone writes them, and as an interrupted transfer
    therefore leaves them. ffprobe reads such a file happily and reports the
    frame count from the header, which is the count the file does *not* have.
    """
    target = tmp_path / "truncated.mp4"
    target.write_bytes(CFR_30FPS.read_bytes()[: 1024 * 4])
    return target


@pytest.fixture
def headerless_video(tmp_path: Path) -> Path:
    """Truncated so early that not even the stream header survives."""
    target = tmp_path / "headerless.mp4"
    target.write_bytes(CFR_30FPS.read_bytes()[:200])
    return target
