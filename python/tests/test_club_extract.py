"""The whole path: a video on disk, landmarks from the cache, a report out.

The tests above this one exercise the detector against rendered frames and the
tracker against constructed evidence. This one exercises the seam between them
and the things only the orchestration knows: that the hand anchors come from the
*filtered* trajectory, that a frame the filter refused arrives as `NO_GRIP`
rather than as a search around a guess, and that the report is self-contained
enough to draw.

It writes a real container and decodes it, so it needs ffmpeg, and it is the
only club test that does. That is deliberate rather than incidental: everything
else here would still pass if `OpenCVFrameSource` handed back frames in the wrong
order or `probe` mis-read a timestamp, and this is where those would show.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from analyzer.club import HoughShaftDetector, track_club
from analyzer.club.detector import ClubDetectionError
from analyzer.contracts.club import ClubConfig, ShaftRefusal
from analyzer.contracts.filtering import FilterConfig
from analyzer.contracts.phases import PhaseConfig, SwingPhase
from analyzer.contracts.pose import Landmark, LandmarkSpace
from analyzer.coordinates import frame_widths_to_pixels
from analyzer.filtering.landmarks import filter_sequence
from analyzer.phases import detect_phases
from analyzer.progress import ProgressUpdate
from tests import synthetic_club as fixture
from tests.conftest import requires_ffmpeg

FPS = 60.0


def _write_clip(frames: list[fixture.RenderedFrame], path: Path) -> Path:
    """Encode the rendered swing into a real container.

    Lossy, on purpose. Every other test here hands the detector the pixels the
    renderer produced; this one puts them through an encoder first, which is what
    happens to every clip this system will ever see.
    """
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (fixture.FRAME.width, fixture.FRAME.height),
    )
    assert writer.isOpened(), "could not open a writer for the fixture clip"
    for item in frames:
        writer.write(item.frame.image)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


@pytest.fixture
def clip(tmp_path: Path) -> tuple[Path, list[fixture.RenderedFrame], object]:
    """A swing on disk, and the pose sequence describing the same motion."""
    frames, poses = fixture.swing_with_poses(fps=FPS, shutter_fraction=0.125)
    return _write_clip(frames, tmp_path / "swing.mp4"), frames, poses


def _filtered(poses: object, **kwargs: object) -> object:
    return filter_sequence(
        poses,  # type: ignore[arg-type]
        FilterConfig(),
        space=LandmarkSpace.FRAME_WIDTHS,
        **kwargs,  # type: ignore[arg-type]
    )


@requires_ffmpeg
class TestTheWholePath:
    def test_a_swing_is_tracked_from_a_file(
        self, clip: tuple[Path, list[fixture.RenderedFrame], object]
    ) -> None:
        video, rendered, poses = clip
        filtered = _filtered(poses)
        phases = detect_phases(filtered, PhaseConfig())  # type: ignore[arg-type]

        report = track_club(
            video,
            filtered,  # type: ignore[arg-type]
            HoughShaftDetector(),
            phases=phases,
        )

        assert report.tracked
        assert report.frame_count == len(rendered)
        assert report.coverage > 0.6
        assert report.refusal is None

        errors = [
            abs((frame.shaft.angle_deg - truth.true_angle_fw + 180.0) % 360.0 - 180.0)
            for truth, frame in zip(rendered, report.frames, strict=True)
            if frame.shaft is not None
        ]
        # Loose by an order of magnitude against the measured median of about
        # half a degree, so a rounding change does not fail it and a detector
        # that starts following the wall does.
        assert float(np.median(errors)) < 5.0

    def test_the_report_carries_what_an_overlay_needs(
        self, clip: tuple[Path, list[fixture.RenderedFrame], object]
    ) -> None:
        """Self-contained on purpose, which is the departure from `PoseExtractionResult`.

        A detector is checked by drawing it on the frames it came from, so a
        report that omitted the geometry would force every consumer to re-run the
        detection in order to see it. A club track is one object per frame rather
        than 33 landmarks in two spaces, which is what makes that affordable.
        """
        video, _, poses = clip
        report = track_club(video, _filtered(poses), HoughShaftDetector())  # type: ignore[arg-type]

        assert report.geometry == fixture.FRAME
        assert report.torso_length == pytest.approx(fixture.TORSO_LENGTH_FW, rel=0.05)
        assert report.detector.name == "hough_shaft"

        drawable = next(frame for frame in report.frames if frame.shaft is not None)
        shaft = drawable.shaft
        assert shaft is not None
        pixels = frame_widths_to_pixels(
            np.array([[shaft.grip_x, shaft.grip_y], [shaft.tip_x, shaft.tip_y]]), report.geometry
        )
        assert (pixels >= -1.0).all() and (pixels[:, 0] <= report.geometry.width + 1).all()

    def test_the_grip_is_the_filtered_hand_position(
        self, clip: tuple[Path, list[fixture.RenderedFrame], object]
    ) -> None:
        """Not the raw landmark, and not a position the detector found.

        A jittering anchor would put its jitter into the shaft angle, where it is
        indistinguishable from the club moving -- and the reported grip is the
        constraint that was applied, so it has to be the one that was applied.
        """
        video, _, poses = clip
        filtered = _filtered(poses)
        report = track_club(video, filtered, HoughShaftDetector())  # type: ignore[arg-type]

        left = filtered.landmarks[Landmark.LEFT_WRIST]  # type: ignore[attr-defined]
        right = filtered.landmarks[Landmark.RIGHT_WRIST]  # type: ignore[attr-defined]
        midpoint = (left.position + right.position) / 2.0

        drawable = next(frame for frame in report.frames if frame.shaft is not None)
        shaft = drawable.shaft
        assert shaft is not None
        assert shaft.grip_x == pytest.approx(float(midpoint[drawable.frame_index, 0]))
        assert shaft.grip_y == pytest.approx(float(midpoint[drawable.frame_index, 1]))

    def test_frames_the_filter_refused_arrive_as_no_grip(
        self, clip: tuple[Path, list[fixture.RenderedFrame], object]
    ) -> None:
        """The ends of every clip, where the fit has too little support.

        A capture problem and a window-width problem produce the same absence
        here, and both are the pose layer's absence rather than the club's --
        which is what the separate reason is for.
        """
        video, _, poses = clip
        report = track_club(video, _filtered(poses), HoughShaftDetector())  # type: ignore[arg-type]
        assert report.refusals.get(ShaftRefusal.NO_GRIP, 0) > 0
        assert report.frames[0].refusal is ShaftRefusal.NO_GRIP

    def test_progress_is_reported(
        self, clip: tuple[Path, list[fixture.RenderedFrame], object]
    ) -> None:
        video, rendered, poses = clip
        updates: list[ProgressUpdate] = []

        class Collector:
            def report(self, event: ProgressUpdate) -> None:
                updates.append(event)

        track_club(video, _filtered(poses), HoughShaftDetector(), reporter=Collector())  # type: ignore[arg-type]
        assert updates
        assert updates[0].stage == "starting"
        assert updates[-1].stage == "done"
        assert updates[-1].total == len(rendered)

    def test_coverage_is_reported_per_phase_when_a_swing_is_found(
        self, clip: tuple[Path, list[fixture.RenderedFrame], object]
    ) -> None:
        video, _, poses = clip
        filtered = _filtered(poses)
        report = track_club(
            video,
            filtered,  # type: ignore[arg-type]
            HoughShaftDetector(),
            phases=detect_phases(filtered, PhaseConfig()),  # type: ignore[arg-type]
        )
        phases = {entry.phase for entry in report.phase_coverage}
        assert SwingPhase.DOWNSWING in phases
        assert all(0.0 <= entry.coverage <= 1.0 for entry in report.phase_coverage)

    def test_without_a_swing_the_report_says_the_rate_is_clip_wide(
        self, clip: tuple[Path, list[fixture.RenderedFrame], object]
    ) -> None:
        video, _, poses = clip
        report = track_club(video, _filtered(poses), HoughShaftDetector())  # type: ignore[arg-type]
        assert report.phase_coverage == []
        assert any("clip-wide" in note for note in report.warnings)

    def test_a_high_confidence_bound_tracks_less_rather_than_worse(
        self, clip: tuple[Path, list[fixture.RenderedFrame], object]
    ) -> None:
        """The exit criterion, exercised from the top of the stack.

        Raising the bound must remove frames, never change what the surviving
        ones say: a confidence threshold that moved the measurement would be a
        tuning knob on the answer rather than on what is reported.
        """
        video, _, poses = clip
        filtered = _filtered(poses)
        loose = track_club(video, filtered, HoughShaftDetector(), config=ClubConfig())  # type: ignore[arg-type]
        strict = track_club(
            video,
            filtered,  # type: ignore[arg-type]
            HoughShaftDetector(),
            config=ClubConfig(min_confidence=0.95),
        )

        assert strict.tracked_frames < loose.tracked_frames
        loose_by_index = {
            frame.frame_index: frame.shaft for frame in loose.frames if frame.shaft is not None
        }
        for frame in strict.frames:
            if frame.shaft is None:
                continue
            kept = loose_by_index.get(frame.frame_index)
            assert kept is not None
            assert frame.shaft.angle_deg == pytest.approx(kept.angle_deg)

    def test_the_cost_is_reported(
        self, clip: tuple[Path, list[fixture.RenderedFrame], object]
    ) -> None:
        """Measured rather than estimated, because the caching decision rests on it."""
        video, _, poses = clip
        report = track_club(video, _filtered(poses), HoughShaftDetector())  # type: ignore[arg-type]
        assert report.elapsed_s > 0.0
        assert report.ms_per_frame == pytest.approx(
            report.elapsed_s * 1000.0 / report.frame_count, rel=1e-6
        )


class TestRefusingToStart:
    def test_a_clip_with_no_measurable_torso_is_refused_by_name(self, tmp_path: Path) -> None:
        """Every search bound is a multiple of the torso, so there is no search.

        Named rather than reported as a failure to find a club, because the two
        have completely different fixes: one is a framing problem and the other
        is a capture or a lighting one.
        """
        _frames, poses = fixture.swing_with_poses(fps=FPS)
        stripped = poses.model_copy(deep=True)
        for frame in stripped.frames:
            for landmark in (Landmark.LEFT_HIP, Landmark.RIGHT_HIP):
                if frame.image:
                    frame.image[int(landmark)].visibility = 0.0

        with pytest.raises(ClubDetectionError, match="shoulder-to-hip"):
            track_club(
                tmp_path / "unused.mp4",
                _filtered(stripped),  # type: ignore[arg-type]
                HoughShaftDetector(),
            )
