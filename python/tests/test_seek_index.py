"""The map a video player seeks by.

Every panel in this app reports frame indices and every video element takes a
time, so something has to convert between them. These tests are about the two
ways that conversion goes wrong: using an average frame rate on footage that
does not have one, and landing on a frame boundary and being rounded to the
wrong side of it.

The test that matters most is `test_naive_rate_lands_on_the_wrong_frame`, which
does not test this module at all -- it measures what the obvious alternative
would have cost on a clip this repository already contains.
"""

from __future__ import annotations

import bisect
from itertools import pairwise
from pathlib import Path

import pytest

from analyzer.contracts.video import TimestampSource
from analyzer.ingestion import probe_video, seek_index

FIXTURES = Path(__file__).parent / "fixtures" / "video"
CFR = FIXTURES / "cfr_30fps.mp4"
VFR = FIXTURES / "vfr_30_to_15fps.mp4"
ROTATED = FIXTURES / "rotated_ccw90.mp4"


def displayed_frame(index: list[float], at: float) -> int:
    """Which frame a decoder shows at time `at`.

    The last frame whose presentation time has been reached, which is the rule
    every player implements and the rule `seek_targets_s` is built to satisfy.
    """
    return max(bisect.bisect_right(index, at) - 1, 0)


@pytest.mark.parametrize("clip", [CFR, VFR, ROTATED])
def test_every_target_displays_its_own_frame(clip: Path) -> None:
    """The round trip: seek to the target for frame i, and frame i is shown.

    This is the whole contract. It is asserted over every frame of every fixture
    rather than a sampled few, because the failure this guards against is not
    uniform -- it appears where the interval changes, which is one or two frames
    of a clip.
    """
    index = seek_index(clip)

    for frame, target in enumerate(index.seek_targets_s):
        assert displayed_frame(index.timestamps_s, target) == frame


@pytest.mark.parametrize("clip", [CFR, VFR])
def test_targets_sit_strictly_inside_their_interval(clip: Path) -> None:
    """A target is never on a boundary, which is the point of using midpoints.

    Landing exactly on `timestamps_s[i]` is asking a decoder to resolve a tie
    between frame i and frame i - 1, and which way it falls is decided by
    rounding in the container's time base, in the double the time is carried as
    and in the decoder's own comparison. None of those is under this project's
    control, so the target avoids the question.
    """
    index = seek_index(clip)
    times = index.timestamps_s

    for frame, target in enumerate(index.seek_targets_s):
        assert target > times[frame]
        if frame + 1 < len(times):
            assert target < times[frame + 1]


def test_naive_rate_lands_on_the_wrong_frame() -> None:
    """`frame / fps` misses on variable-rate footage; the measured times do not.

    The fixture changes rate from 30 fps to 15 fps half-way through, which is
    what a phone does when it runs short of light. Seeking by the container's
    declared average rate is the obvious implementation and the reason this
    contract exists: it is correct until the rate changes and then accumulates,
    and nothing about the resulting picture looks wrong.
    """
    index = seek_index(VFR)
    metadata = probe_video(VFR)
    nominal = metadata.timing.nominal_fps
    assert nominal is not None
    assert metadata.timing.is_vfr is True

    wrong = [
        frame
        for frame in range(index.frame_count)
        if displayed_frame(index.timestamps_s, frame / nominal) != frame
    ]

    # The measured targets are exact on the same clip, which is the comparison:
    # the naive map is not merely less precise, it is wrong on frames the right
    # map gets right.
    assert wrong, "the fixture no longer changes rate; this test has stopped measuring anything"
    assert all(
        displayed_frame(index.timestamps_s, target) == frame
        for frame, target in enumerate(index.seek_targets_s)
    )


def test_timestamps_are_relative_and_increasing() -> None:
    """Times start at zero and never go backwards.

    Relative because `PoseFrame.timestamp_s` is, and a player that indexed a
    different origin from the panels beside it would disagree with them by the
    container's start offset -- a quantity that means nothing about a swing.
    """
    index = seek_index(CFR)

    assert index.timestamps_s[0] == 0.0
    assert all(later > earlier for earlier, later in pairwise(index.timestamps_s))


def test_frame_count_agrees_with_the_metadata() -> None:
    """One clip, one frame count. Two would be a bug nobody would notice."""
    index = seek_index(CFR)

    assert index.frame_count == probe_video(CFR).timing.frame_count
    assert len(index.timestamps_s) == index.frame_count
    assert len(index.seek_targets_s) == index.frame_count


def test_last_frame_borrows_the_last_observed_interval() -> None:
    """The container does not record how long the final frame is displayed.

    So it is assumed to last as long as the one before it -- the same assumption
    `VideoTiming.duration_s` already makes, repeated here rather than invented,
    so the two cannot disagree about where the clip ends.
    """
    index = seek_index(CFR)
    times = index.timestamps_s

    assert index.last_interval_s == pytest.approx(times[-1] - times[-2])
    assert index.seek_targets_s[-1] == pytest.approx(times[-1] + index.last_interval_s / 2.0)


def test_content_key_identifies_the_clip() -> None:
    """Keyed by content, as every other reference to a clip in this project is."""
    assert seek_index(CFR).content_key == probe_video(CFR).content_key


def test_synthesised_times_are_reported_as_a_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip with no timestamps still gets an index, and it says what it is.

    The frames are evenly spaced by assumption rather than by measurement, which
    is exactly the assumption this contract exists to avoid making silently. It
    is not refused -- a player still has to show something -- but the warning is
    the only evidence a reader will ever get, because the data that would
    contradict it is the data that is missing.

    Reached by substituting the probe rather than by a fixture, because no
    container this repository can generate takes that branch: ffmpeg writes
    presentation timestamps, and a file with frames but no timestamps is a
    damaged one. The branch is still live -- `probe` falls back to it -- so it
    is tested where it is decidable rather than left to a clip nobody has.
    """
    import analyzer.ingestion as ingestion
    from analyzer.ingestion.probe import FrameIndex, VideoProbe

    real = ingestion.probe

    def synthesised(path: Path, *, full_hash: bool = False) -> VideoProbe:
        probed = real(path, full_hash=full_hash)
        return VideoProbe(
            metadata=probed.metadata,
            index=FrameIndex(
                source=TimestampSource.CONTAINER_RATE,
                timestamps_s=tuple(i / 30.0 for i in range(len(probed.index))),
                keyframes=probed.index.keyframes,
            ),
        )

    monkeypatch.setattr(ingestion, "probe", synthesised)
    index = seek_index(CFR)

    assert index.source is TimestampSource.CONTAINER_RATE
    assert index.warnings
    assert "synthesised" in index.warnings[0]
    # Still a usable map: the times are a guess and the arithmetic over them is
    # not, so a player gets targets that are self-consistent with the guess.
    assert all(
        displayed_frame(index.timestamps_s, target) == frame
        for frame, target in enumerate(index.seek_targets_s)
    )


def test_missing_file_is_refused_with_a_remedy(tmp_path: Path) -> None:
    """The same `ProbeError` path every other reader of a video takes."""
    from analyzer.ingestion import ProbeError

    with pytest.raises(ProbeError):
        seek_index(tmp_path / "nothing.mp4")
