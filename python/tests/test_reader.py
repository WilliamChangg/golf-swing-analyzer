"""Frame source tests: orientation, timing, random access.

The three things a frame source must not get wrong are which way up a frame is,
what time it was taken at, and which frame it is. None of the three announces
itself when it goes wrong -- a transposed frame, a frame timed by index, and an
off-by-a-few seek all decode successfully and look like video -- so each is
pinned against something independent of the code under test.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from analyzer.ingestion.probe import probe
from analyzer.ingestion.reader import (
    DecodeBackend,
    FFmpegPipeFrameSource,
    OpenCVFrameSource,
    VideoFrame,
    apply_display_rotation,
    open_video,
)
from tests.conftest import (
    CFR_30FPS,
    CFR_CODED_SIZE,
    CFR_FPS,
    CFR_FRAME_COUNT,
    ROTATED_180,
    ROTATED_CCW90,
    VFR_30_TO_15FPS,
    requires_ffmpeg,
)

# OpenCV and the ffmpeg CLI both decode with libavcodec but convert YUV to BGR
# through differently-configured swscale paths, so a few least-significant bits
# differ. Measured at max 3 across the fixtures; 4 leaves a little headroom
# without being loose enough to hide a real decode difference.
_CHANNEL_TOLERANCE = 4


@requires_ffmpeg
class TestRotationConvention:
    """Pins the counter-clockwise convention against ffmpeg's own autorotation.

    `apply_display_rotation` takes ffprobe's rotation value at face value, and
    the direction that value means is the single assumption the whole ingestion
    layer rests on. Here it is checked against an independent implementation:
    ffmpeg's own autorotate, driven by the same display matrix. If a future
    ffmpeg flips the sign, this fails instead of silently transposing every
    measurement the system goes on to make.
    """

    @staticmethod
    def _decode_first(path: Path, *, autorotate: bool) -> np.ndarray:
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not autorotate:
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        ok, image = capture.read()
        capture.release()
        assert ok
        return image

    @pytest.mark.parametrize("path", [ROTATED_CCW90, ROTATED_180])
    def test_matches_the_decoder_s_own_autorotation(self, path: Path) -> None:
        stored = self._decode_first(path, autorotate=False)
        rotation = probe(path).metadata.stream.rotation_ccw_degrees
        assert np.array_equal(
            apply_display_rotation(stored, rotation),
            self._decode_first(path, autorotate=True),
        )

    def test_the_fixture_really_is_stored_unrotated(self) -> None:
        """Guards the test above: it proves nothing if the file is already upright."""
        stored = self._decode_first(ROTATED_CCW90, autorotate=False)
        assert stored.shape[:2] == CFR_CODED_SIZE[::-1]

    def test_zero_rotation_returns_the_frame_untouched(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        assert apply_display_rotation(image, 0) is image


@requires_ffmpeg
class TestOpenCVFrameSource:
    def test_yields_display_oriented_frames(self) -> None:
        with OpenCVFrameSource(ROTATED_CCW90) as source:
            stream = source.metadata.stream
            assert source.frame_at(0).shape_hw == (stream.display_height, stream.display_width)

    def test_does_not_double_rotate(self) -> None:
        """OpenCV auto-rotates by default; rotating again lands a quarter turn out.

        A 320x240 frame turned 90 degrees is 240 wide and 320 high. Turned twice
        it is 320x240 again -- the original shape -- so the frame size alone
        distinguishes a correct single rotation from a doubled one.
        """
        width, height = CFR_CODED_SIZE
        with OpenCVFrameSource(ROTATED_CCW90) as source:
            assert source.frame_at(0).shape_hw == (width, height)

    def test_half_turn_keeps_the_frame_size(self) -> None:
        with OpenCVFrameSource(ROTATED_180) as source:
            assert source.frame_at(0).shape_hw == CFR_CODED_SIZE[::-1]

    def test_frame_timestamps_agree_with_the_decoder(self) -> None:
        """The index says when a frame is shown; the decoder agrees or something is wrong."""
        with OpenCVFrameSource(CFR_30FPS) as source:
            for index in (0, 1, 17, CFR_FRAME_COUNT - 1):
                frame = source.frame_at(index)
                assert frame.decoder_timestamp_s is not None
                assert frame.decoder_timestamp_s == pytest.approx(frame.timestamp_s, abs=1e-6)

    def test_timestamps_come_from_the_container_not_the_index(self) -> None:
        """On a variable-rate clip the two answers differ, and the container wins."""
        with OpenCVFrameSource(VFR_30_TO_15FPS) as source:
            nominal = source.metadata.timing.nominal_fps
            assert nominal is not None
            frame = source.frame_at(40)
            assert frame.timestamp_s != pytest.approx(40 / nominal, abs=1e-3)
            assert frame.decoder_timestamp_s == pytest.approx(frame.timestamp_s, abs=1e-6)

    def test_random_access_agrees_with_sequential_reading(self) -> None:
        with OpenCVFrameSource(CFR_30FPS) as sequential:
            expected = {f.index: f.image.copy() for f in sequential.frames()}
        with OpenCVFrameSource(CFR_30FPS) as random_access:
            for index in (55, 3, 40, 0, 59, 21):
                assert np.array_equal(random_access.frame_at(index).image, expected[index])

    def test_seeks_that_land_wrong_are_detected_and_redone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frame-index seeking is approximate on long-GOP streams.

        The decoder reports success either way, so the wrong frame arrives
        carrying the right timestamp. Here the seek is deliberately sabotaged to
        prove the mismatch is noticed rather than returned.
        """
        with OpenCVFrameSource(CFR_30FPS) as truth:
            expected = truth.frame_at(50).image.copy()

        with OpenCVFrameSource(CFR_30FPS) as source:
            # Reaching into the private seek is the point: this simulates a
            # decoder-level fault that cannot be provoked from outside.
            original = source._seek

            def wrong_seek(index: int) -> None:
                original(max(0, index - 7))
                source._next_index = index  # pretend the seek was honoured

            monkeypatch.setattr(source, "_seek", wrong_seek)
            frame = source.frame_at(50)

            assert source.inaccurate_seeks == 1
            assert np.array_equal(frame.image, expected)

    def test_frames_can_be_written_to(self) -> None:
        """Overlay rendering draws onto decoded frames in place."""
        with OpenCVFrameSource(CFR_30FPS) as source:
            assert source.frame_at(0).image.flags.writeable

    def test_iterating_with_a_step(self) -> None:
        with OpenCVFrameSource(CFR_30FPS) as source:
            indices = [frame.index for frame in source.frames(start=10, stop=20, step=3)]
        assert indices == [10, 13, 16, 19]

    def test_iterating_yields_every_frame_once(self) -> None:
        with OpenCVFrameSource(CFR_30FPS) as source:
            indices = [frame.index for frame in source.frames()]
        assert indices == list(range(CFR_FRAME_COUNT))

    @pytest.mark.parametrize("index", [-1, CFR_FRAME_COUNT, CFR_FRAME_COUNT + 100])
    def test_out_of_range_raises(self, index: int) -> None:
        with OpenCVFrameSource(CFR_30FPS) as source, pytest.raises(IndexError):
            source.frame_at(index)

    def test_rejects_a_step_of_zero(self) -> None:
        with OpenCVFrameSource(CFR_30FPS) as source, pytest.raises(ValueError, match="step"):
            list(source.frames(step=0))

    def test_close_is_idempotent(self) -> None:
        source = OpenCVFrameSource(CFR_30FPS)
        source.close()
        source.close()

    def test_repeated_reads_hit_the_cache(self) -> None:
        with OpenCVFrameSource(CFR_30FPS) as source:
            source.frame_at(5)
            source.frame_at(5)
            assert source.cache_stats.hits == 1


@requires_ffmpeg
class TestFFmpegPipeFrameSource:
    def test_agrees_with_opencv_on_pixels(self) -> None:
        """Two independent decoders, one file: the frames must be the same picture."""
        probed = probe(CFR_30FPS)
        with OpenCVFrameSource(probed) as a, FFmpegPipeFrameSource(probed) as b:
            for index in (0, 7, CFR_FRAME_COUNT - 1):
                difference = np.abs(
                    a.frame_at(index).image.astype(np.int16)
                    - b.frame_at(index).image.astype(np.int16)
                )
                assert difference.max() <= _CHANNEL_TOLERANCE, index

    def test_agrees_with_opencv_on_timestamps(self) -> None:
        probed = probe(VFR_30_TO_15FPS)
        with OpenCVFrameSource(probed) as a, FFmpegPipeFrameSource(probed) as b:
            for index in (0, 30, 44):
                assert a.frame_at(index).timestamp_s == b.frame_at(index).timestamp_s

    def test_applies_rotation_the_same_way(self) -> None:
        with FFmpegPipeFrameSource(ROTATED_CCW90) as source:
            stream = source.metadata.stream
            assert source.frame_at(0).shape_hw == (stream.display_height, stream.display_width)

    def test_reading_backwards_restarts_the_decode(self) -> None:
        with FFmpegPipeFrameSource(CFR_30FPS) as source:
            later = source.frame_at(20).image.copy()
            earlier = source.frame_at(3)
            assert earlier.index == 3
            assert not np.array_equal(earlier.image, later)
            assert np.array_equal(source.frame_at(20).image, later)

    def test_frames_can_be_written_to(self) -> None:
        with FFmpegPipeFrameSource(CFR_30FPS) as source:
            assert source.frame_at(0).image.flags.writeable

    def test_hardware_request_is_reported_not_assumed(self) -> None:
        with FFmpegPipeFrameSource(CFR_30FPS, hardware=True) as source:
            assert source.backend is DecodeBackend.FFMPEG_VIDEOTOOLBOX

    def test_hardware_decode_produces_the_same_pixels(self) -> None:
        """If VideoToolbox ever became the default, it must change nothing visible."""
        probed = probe(CFR_30FPS)
        with (
            FFmpegPipeFrameSource(probed, hardware=False) as software,
            FFmpegPipeFrameSource(probed, hardware=True) as hardware,
        ):
            difference = np.abs(
                software.frame_at(10).image.astype(np.int16)
                - hardware.frame_at(10).image.astype(np.int16)
            )
            assert difference.max() <= _CHANNEL_TOLERANCE

    def test_close_is_idempotent(self) -> None:
        source = FFmpegPipeFrameSource(CFR_30FPS)
        source.close()
        source.close()


@requires_ffmpeg
class TestOpenVideo:
    def test_defaults_to_opencv(self) -> None:
        source = open_video(CFR_30FPS)
        try:
            assert isinstance(source, OpenCVFrameSource)
        finally:
            source.close()

    @pytest.mark.parametrize(
        "backend", [DecodeBackend.FFMPEG_CPU, DecodeBackend.FFMPEG_VIDEOTOOLBOX]
    )
    def test_ffmpeg_backends_are_selectable(self, backend: DecodeBackend) -> None:
        source = open_video(CFR_30FPS, backend=backend)
        try:
            assert isinstance(source, FFmpegPipeFrameSource)
        finally:
            source.close()

    def test_every_backend_satisfies_the_protocol(self) -> None:
        """The seam is only real if both implementations are interchangeable."""
        for backend in DecodeBackend:
            source = open_video(CFR_30FPS, backend=backend)
            try:
                frame = source.frame_at(1)
                assert isinstance(frame, VideoFrame)
                assert frame.timestamp_s == pytest.approx(1 / CFR_FPS)
                assert source.metadata.timing.frame_count == CFR_FRAME_COUNT
            finally:
                source.close()
