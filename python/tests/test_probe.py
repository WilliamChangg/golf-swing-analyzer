"""Container inspection tests.

Split deliberately into two halves. The parsing tests are pure -- they take
ffprobe output as literal strings and need no ffmpeg -- so the logic that
decides rotation and variable frame rate is pinned independently of what any
particular ffmpeg build produces. The probe tests then run the real thing
against the committed fixtures, which is what catches an ffmpeg whose output
this parser no longer understands.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path

import pytest

from analyzer.contracts.video import TimestampSource
from analyzer.ingestion.probe import (
    ProbeError,
    check_readable,
    parse_frame_index,
    probe,
    read_rotation,
    timing_from_ticks,
)
from tests.conftest import (
    AUDIO_ONLY,
    CFR_30FPS,
    CFR_CODED_SIZE,
    CFR_FPS,
    CFR_FRAME_COUNT,
    ROTATED_180,
    ROTATED_CCW90,
    VFR_30_TO_15FPS,
    VFR_FRAME_COUNT,
    requires_ffprobe,
)

# A 1/15360 time base at 30 fps: one frame is exactly 512 ticks.
TICK = Fraction(1, 15360)
FRAME_TICKS = 512


class TestReadRotation:
    """The display matrix is counter-clockwise; the legacy tag is clockwise.

    Getting this sign wrong transposes every frame the system ever measures, and
    nothing about the output would look obviously broken, so both conventions
    are pinned here rather than left to the one fixture that happens to exercise
    them.
    """

    def test_display_matrix_is_read_as_counter_clockwise(self) -> None:
        stream = {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 90}]}
        assert read_rotation(stream) == (90, "display_matrix", None)

    def test_negative_display_matrix_rotation_normalises(self) -> None:
        stream = {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -180}]}
        degrees, source, _ = read_rotation(stream)
        assert (degrees, source) == (180, "display_matrix")

    def test_negative_quarter_turn_becomes_270(self) -> None:
        stream = {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]}
        assert read_rotation(stream)[0] == 270

    def test_legacy_rotate_tag_is_clockwise_and_is_negated(self) -> None:
        """`rotate: 90` means 90 clockwise, which is 270 counter-clockwise."""
        degrees, source, _ = read_rotation({"tags": {"rotate": "90"}})
        assert (degrees, source) == (270, "rotate_tag")

    def test_display_matrix_wins_over_the_tag(self) -> None:
        stream = {
            "side_data_list": [{"rotation": 180}],
            "tags": {"rotate": "90"},
        }
        assert read_rotation(stream)[1] == "display_matrix"

    def test_absent_rotation_is_zero_with_no_source(self) -> None:
        assert read_rotation({}) == (0, None, None)

    def test_non_quarter_turn_is_rounded_and_says_so(self) -> None:
        degrees, _, warning = read_rotation({"side_data_list": [{"rotation": 37}]})
        assert degrees == 0
        assert warning is not None
        assert "not a quarter turn" in warning

    def test_quarter_turn_produces_no_warning(self) -> None:
        assert read_rotation({"side_data_list": [{"rotation": 270}]})[2] is None


class TestParseFrameIndex:
    """ffprobe's `compact` format, one frame per line.

    Compact rather than CSV because CSV emits fields in ffprobe's own order
    rather than the order they were requested in, which makes column position a
    guess.
    """

    @staticmethod
    def _line(timestamp: str, key: str = "0") -> str:
        return f"key_frame={key}|best_effort_timestamp={timestamp}||||"

    def test_reads_timestamps_in_order(self) -> None:
        text = "\n".join(self._line(t) for t in ("0", "512", "1024"))
        ticks, _, _ = parse_frame_index(text)
        assert ticks == [0, 512, 1024]

    def test_keyframe_flags_travel_with_their_timestamps(self) -> None:
        text = "\n".join([self._line("0", "1"), self._line("512"), self._line("1024", "1")])
        ticks, keyframes, _ = parse_frame_index(text)
        assert ticks == [0, 512, 1024]
        assert keyframes == [True, False, True]

    def test_field_order_does_not_matter(self) -> None:
        """ffprobe chooses the order; the parser reads by name."""
        text = "best_effort_timestamp=7|key_frame=1"
        ticks, keyframes, _ = parse_frame_index(text)
        assert (ticks, keyframes) == ([7], [True])

    def test_frames_without_a_timestamp_are_dropped_and_counted(self) -> None:
        """Dropping one shifts every later index, so the count must reach the caller."""
        text = "\n".join([self._line("0"), self._line("N/A"), self._line("512")])
        ticks, _, dropped = parse_frame_index(text)
        assert ticks == [0, 512]
        assert dropped == 1

    def test_nothing_dropped_is_reported_as_zero(self) -> None:
        assert parse_frame_index(self._line("0"))[2] == 0

    def test_blank_lines_are_ignored(self) -> None:
        text = f"\n{self._line('0')}\n\n{self._line('512')}\n\n"
        assert parse_frame_index(text)[0] == [0, 512]

    def test_empty_input_yields_nothing(self) -> None:
        assert parse_frame_index("") == ([], [], 0)


class TestVariableFrameRateDetection:
    """`is_vfr` has no tolerance beyond the container's own clock resolution.

    Any interval more than one tick from the median means frame times are not
    uniform, which is the only question the flag has to answer: whether
    `frame_index / fps` may be used as time. One dropped frame in a long clip
    makes it invalid just as surely as a rate change does.
    """

    def test_exactly_uniform_intervals_are_not_vfr(self) -> None:
        timing = timing_from_ticks(
            [i * FRAME_TICKS for i in range(60)],
            TICK,
            nominal_fps=30.0,
            container_duration_s=2.0,
        )
        assert timing.is_vfr is False
        assert timing.intervals is not None
        assert timing.intervals.irregular_count == 0

    def test_one_tick_of_jitter_is_rounding_not_a_rate_change(self) -> None:
        """29.97 fps in a 1/1000 time base genuinely alternates 33 and 34 ms.

        That is the time base failing to divide the frame rate, not the camera
        changing rate, and calling it variable would flag almost every
        NTSC-derived clip in existence.
        """
        milliseconds = Fraction(1, 1000)
        ticks = [0]
        for i in range(1, 300):
            ticks.append(round(i * 1000 / 29.97))
        timing = timing_from_ticks(
            ticks, milliseconds, nominal_fps=29.97, container_duration_s=None
        )
        assert timing.intervals is not None
        assert timing.intervals.max_s - timing.intervals.min_s > 0  # jitter is real
        assert timing.is_vfr is False

    def test_a_two_tick_deviation_is_variable(self) -> None:
        ticks = [i * FRAME_TICKS for i in range(30)]
        ticks[15:] = [t + 2 for t in ticks[15:]]
        timing = timing_from_ticks(ticks, TICK, nominal_fps=30.0, container_duration_s=None)
        assert timing.is_vfr is True

    def test_a_dropped_frame_makes_a_clip_variable(self) -> None:
        ticks = [i * FRAME_TICKS for i in range(60) if i != 30]
        timing = timing_from_ticks(ticks, TICK, nominal_fps=30.0, container_duration_s=None)
        assert timing.is_vfr is True
        assert timing.intervals is not None
        assert timing.intervals.irregular_count == 1

    def test_measured_fps_comes_from_timestamps_not_the_declared_rate(self) -> None:
        ticks = [i * FRAME_TICKS for i in range(31)]
        timing = timing_from_ticks(ticks, TICK, nominal_fps=999.0, container_duration_s=None)
        assert timing.measured_fps == pytest.approx(30.0)
        assert timing.nominal_fps == 999.0

    def test_a_single_frame_has_no_intervals_and_no_verdict(self) -> None:
        timing = timing_from_ticks([0], TICK, nominal_fps=30.0, container_duration_s=1.0)
        assert timing.intervals is None
        assert timing.is_vfr is None
        assert timing.measured_fps is None

    def test_timestamps_are_reported_as_seconds_not_ticks(self) -> None:
        timing = timing_from_ticks(
            [FRAME_TICKS, 2 * FRAME_TICKS], TICK, nominal_fps=30.0, container_duration_s=None
        )
        assert timing.first_timestamp_s == pytest.approx(1 / 30)
        assert timing.timestamp_span_s == pytest.approx(1 / 30)


class TestCheckReadable:
    def test_missing_file_is_named_as_such(self, tmp_path: Path) -> None:
        with pytest.raises(ProbeError, match="No file exists"):
            check_readable(tmp_path / "absent.mp4")

    def test_directory_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ProbeError, match="is a directory"):
            check_readable(tmp_path)

    def test_zero_byte_file_is_rejected_before_ffprobe(self, empty_video: Path) -> None:
        with pytest.raises(ProbeError, match="empty"):
            check_readable(empty_video)

    def test_every_rejection_carries_a_remediation(self, tmp_path: Path) -> None:
        """These are all user-fixable, so none of them may be a bare error."""
        for target in (tmp_path / "absent.mp4", tmp_path):
            with pytest.raises(ProbeError) as excinfo:
                check_readable(target)
            assert excinfo.value.remediation


@requires_ffprobe
class TestProbeConstantRate:
    def test_reads_codec_and_geometry(self) -> None:
        stream = probe(CFR_30FPS).metadata.stream
        assert stream.codec_name == "h264"
        assert (stream.coded_width, stream.coded_height) == CFR_CODED_SIZE
        assert (stream.display_width, stream.display_height) == CFR_CODED_SIZE
        assert stream.pix_fmt == "yuv420p"

    def test_frame_count_and_rate_are_exact(self) -> None:
        timing = probe(CFR_30FPS).metadata.timing
        assert timing.frame_count == CFR_FRAME_COUNT
        assert timing.measured_fps == pytest.approx(CFR_FPS)
        assert timing.nominal_fps == pytest.approx(CFR_FPS)

    def test_duration_matches_the_container(self) -> None:
        timing = probe(CFR_30FPS).metadata.timing
        assert timing.duration_s == pytest.approx(2.0, abs=1e-6)
        assert timing.container_duration_s == pytest.approx(2.0, abs=1e-3)

    def test_is_not_variable_frame_rate(self) -> None:
        timing = probe(CFR_30FPS).metadata.timing
        assert timing.is_vfr is False
        assert timing.source is TimestampSource.DECODED_FRAMES

    def test_a_clean_clip_produces_no_warnings(self) -> None:
        assert probe(CFR_30FPS).metadata.warnings == []

    def test_timestamps_increase_strictly(self) -> None:
        timestamps = probe(CFR_30FPS).index.timestamps_s
        assert len(timestamps) == CFR_FRAME_COUNT
        assert all(b > a for a, b in itertools.pairwise(timestamps))

    def test_relative_timestamps_start_at_zero(self) -> None:
        relative = probe(CFR_30FPS).index.relative_timestamps_s()
        assert relative[0] == 0.0
        assert relative[-1] == pytest.approx((CFR_FRAME_COUNT - 1) / CFR_FPS)

    def test_content_key_records_the_file_size(self) -> None:
        metadata = probe(CFR_30FPS).metadata
        assert metadata.content_key.size_bytes == CFR_30FPS.stat().st_size
        assert metadata.file_size_bytes == CFR_30FPS.stat().st_size


@requires_ffprobe
class TestProbeVariableRate:
    def test_is_detected(self) -> None:
        timing = probe(VFR_30_TO_15FPS).metadata.timing
        assert timing.is_vfr is True
        assert timing.frame_count == VFR_FRAME_COUNT

    def test_reports_the_evidence_for_the_verdict(self) -> None:
        """The numbers behind the flag, so it can be checked rather than trusted."""
        stats = probe(VFR_30_TO_15FPS).metadata.timing.intervals
        assert stats is not None
        assert stats.irregular_count > 0
        assert stats.max_s == pytest.approx(2 / 30, abs=1e-6)
        assert stats.min_s == pytest.approx(1 / 30, abs=1e-6)

    def test_warns_that_frame_over_fps_is_invalid(self) -> None:
        warnings = probe(VFR_30_TO_15FPS).metadata.warnings
        assert any("Variable frame rate" in w for w in warnings)
        assert any("frame_index / fps is not" in w for w in warnings)

    def test_timestamps_are_not_evenly_spaced(self) -> None:
        timestamps = probe(VFR_30_TO_15FPS).index.timestamps_s
        intervals = {round(b - a, 4) for a, b in itertools.pairwise(timestamps)}
        assert len(intervals) > 1

    def test_measured_rate_differs_from_the_declared_rate(self) -> None:
        timing = probe(VFR_30_TO_15FPS).metadata.timing
        assert timing.measured_fps is not None
        assert timing.nominal_fps is not None
        assert abs(timing.measured_fps - timing.nominal_fps) > 0.5


@requires_ffprobe
class TestProbeRotation:
    def test_quarter_turn_swaps_the_display_dimensions(self) -> None:
        stream = probe(ROTATED_CCW90).metadata.stream
        assert stream.rotation_ccw_degrees == 90
        assert (stream.coded_width, stream.coded_height) == CFR_CODED_SIZE
        assert (stream.display_width, stream.display_height) == CFR_CODED_SIZE[::-1]
        assert stream.rotation_source == "display_matrix"

    def test_half_turn_leaves_the_dimensions_alone(self) -> None:
        stream = probe(ROTATED_180).metadata.stream
        assert stream.rotation_ccw_degrees == 180
        assert (stream.display_width, stream.display_height) == CFR_CODED_SIZE

    def test_unrotated_clip_reports_no_rotation_source(self) -> None:
        stream = probe(CFR_30FPS).metadata.stream
        assert stream.rotation_ccw_degrees == 0
        assert stream.rotation_source is None

    def test_rotation_is_warned_about(self) -> None:
        warnings = probe(ROTATED_CCW90).metadata.warnings
        assert any("rotation" in w for w in warnings)

    def test_rotation_does_not_disturb_timing(self) -> None:
        """The same frames, re-muxed with a display matrix, must time identically."""
        assert (
            probe(ROTATED_CCW90).metadata.timing.frame_count
            == probe(CFR_30FPS).metadata.timing.frame_count
        )


@requires_ffprobe
class TestProbeRejects:
    def test_audio_only_file(self) -> None:
        with pytest.raises(ProbeError, match="no video stream"):
            probe(AUDIO_ONLY)

    def test_corrupt_file(self, corrupt_video: Path) -> None:
        with pytest.raises(ProbeError) as excinfo:
            probe(corrupt_video)
        assert excinfo.value.remediation is not None

    def test_zero_byte_file(self, empty_video: Path) -> None:
        with pytest.raises(ProbeError, match="empty"):
            probe(empty_video)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ProbeError, match="No file exists"):
            probe(tmp_path / "nope.mp4")

    def test_headerless_file(self, headerless_video: Path) -> None:
        with pytest.raises(ProbeError, match="no video stream"):
            probe(headerless_video)


@requires_ffprobe
class TestProbeTruncated:
    """A truncated recording is readable, and reporting it as whole would lie.

    This is the interesting case, because nothing errors: the header survives an
    interrupted transfer and declares a frame count the file does not contain.
    Trusting the header would give every later phase a timeline running past the
    end of the footage.
    """

    def test_reports_the_frames_that_are_actually_there(self, truncated_video: Path) -> None:
        timing = probe(truncated_video).metadata.timing
        assert 0 < timing.frame_count < CFR_FRAME_COUNT

    def test_says_the_header_overstates_the_clip(self, truncated_video: Path) -> None:
        warnings = probe(truncated_video).metadata.warnings
        assert any(f"declares {CFR_FRAME_COUNT} frames" in w for w in warnings)

    def test_a_file_with_no_decodable_frames_is_refused(self, frameless_video: Path) -> None:
        """Describing a clip that cannot be read is worse than refusing it."""
        with pytest.raises(ProbeError, match="produces no frames") as excinfo:
            probe(frameless_video)
        assert excinfo.value.remediation is not None
