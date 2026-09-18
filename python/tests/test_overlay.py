"""What gets drawn on top of a frame.

The overlay's job is to let a number be checked against the picture it came
from, so these tests are mostly about the ways it could quietly stop being
checkable: drawing the wrong landmarks, drawing a position nothing supports, or
putting the skeleton in a frame that does not map to the canvas.

Every fixture here is portrait, for the reason Phase 5's are deliberately not
square: on a square frame the anisotropic and isotropic coordinates are the same
numbers, so a conversion run backwards, twice, or not at all all produce
identical output and every one of those bugs hides.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from analyzer.contracts.cache import ContentKey, HashAlgorithm
from analyzer.contracts.filtering import FilterConfig, SmoothingConfig
from analyzer.contracts.overlay import OverlayState
from analyzer.contracts.pose import (
    LANDMARK_COUNT,
    POSE_CONNECTIONS,
    FrameGeometry,
    Landmark,
    LandmarkPoint,
    LandmarkSpace,
    PoseExtractionStats,
    PoseFrame,
    PoseModelInfo,
    PoseSequence,
)
from analyzer.coordinates import image_to_frame_widths
from analyzer.filtering.landmarks import filter_sequence
from analyzer.overlay import MAX_OVERLAY_FRAMES, OverlayError, build_overlay

FPS = 120.0

# Deliberately not square, and deliberately portrait: phone footage of a golf
# swing is portrait, and the aspect correction is a no-op on a square frame.
PORTRAIT = FrameGeometry(width=720, height=1280)

KEY = ContentKey(algorithm=HashAlgorithm.SHA256_SAMPLED, digest="a" * 64, size_bytes=1)


def _points(t: float, visibility: float = 0.9) -> list[LandmarkPoint]:
    """A body whose landmarks move slowly and smoothly, none of them coincident."""
    return [
        LandmarkPoint(
            x=0.3 + 0.002 * index + 0.05 * float(np.sin(t)),
            y=0.2 + 0.015 * index,
            z=0.0,
            visibility=visibility,
            presence=visibility,
        )
        for index in range(LANDMARK_COUNT)
    ]


def _sequence(
    count: int = 240, *, undetected: range | None = None, geometry: FrameGeometry = PORTRAIT
) -> PoseSequence:
    frames = []
    for index in range(count):
        detected = undetected is None or index not in undetected
        points = _points(index / FPS) if detected else []
        frames.append(
            PoseFrame(
                frame_index=index,
                timestamp_s=index / FPS,
                detected=detected,
                image=points,
                hip_local=points,
            )
        )
    found = sum(1 for frame in frames if frame.detected)
    return PoseSequence(
        video_path="/data/swing.mov",
        video_content_key=KEY,
        geometry=geometry,
        model=PoseModelInfo(
            name="fake",
            variant="fake",
            precision="float32",
            sha256="0" * 64,
            delegate="cpu",
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ),
        extracted_at=datetime(2026, 9, 18, tzinfo=UTC),
        stats=PoseExtractionStats(
            frames_processed=count,
            frames_detected=found,
            detection_rate=found / count if count else 0.0,
            elapsed_s=1.0,
            ms_per_frame=1.0,
        ),
        frames=frames,
    )


def _filtered(sequence: PoseSequence | None = None, **kwargs: object):  # type: ignore[no-untyped-def]
    return filter_sequence(
        sequence if sequence is not None else _sequence(),
        FilterConfig(smoothing=SmoothingConfig(window_s=0.15)),
        space=LandmarkSpace.FRAME_WIDTHS,
        **kwargs,  # type: ignore[arg-type]
    )


def _overlay(filtered=None, **kwargs: object):  # type: ignore[no-untyped-def]
    defaults: dict[str, object] = {
        "video_path": "/data/swing.mov",
        "content_key": KEY,
        "start_frame": 100,
        "end_frame": 104,
    }
    defaults.update(kwargs)
    return build_overlay(filtered if filtered is not None else _filtered(), **defaults)  # type: ignore[arg-type]


class TestRange:
    def test_covers_exactly_the_half_open_range(self) -> None:
        overlay = _overlay(start_frame=10, end_frame=14)

        assert [frame.frame_index for frame in overlay.frames] == [10, 11, 12, 13]
        assert overlay.start_frame == 10
        assert overlay.end_frame == 14

    def test_clamps_rather_than_refusing_past_the_end(self) -> None:
        """A canvas asking for a window around the last frame is asking sensibly.

        Making it compute the clamp itself would put the clip length in two
        places, and the second one would be in TypeScript.
        """
        overlay = _overlay(start_frame=238, end_frame=999)

        assert [frame.frame_index for frame in overlay.frames] == [238, 239]
        assert overlay.end_frame == 240

    def test_an_empty_range_is_allowed(self) -> None:
        """Zero frames is a legitimate answer, not an error.

        It is what a caller gets asking for a window that starts past the end,
        and refusing it would make a scrubber at the clip boundary raise.
        """
        assert _overlay(start_frame=500, end_frame=600).frames == []

    def test_a_range_beyond_the_limit_is_refused_by_name(self) -> None:
        """Refused, not truncated. A silently shortened range would draw a
        skeleton over the first part of a clip and nothing after it, with the
        reason visible nowhere."""
        long_clip = _filtered(_sequence(count=MAX_OVERLAY_FRAMES + 40))

        with pytest.raises(OverlayError) as caught:
            build_overlay(
                long_clip,
                video_path="/data/swing.mov",
                content_key=KEY,
                start_frame=0,
                end_frame=MAX_OVERLAY_FRAMES + 1,
            )

        assert str(MAX_OVERLAY_FRAMES) in str(caught.value)
        assert caught.value.remediation is not None


class TestCoordinates:
    def test_positions_are_the_exact_inverse_of_the_measurement_frame(self) -> None:
        """Round trip: IMAGE in, frame widths measured, IMAGE out.

        Asserted to floating-point precision rather than to a tolerance, because
        `frame_widths_to_image` is the algebraic inverse of the conversion the
        filter's input went through -- not an approximation of it. A tolerance
        here would hide a wrong aspect ratio that happened to be close.
        """
        filtered = _filtered()
        overlay = _overlay(filtered, start_frame=120, end_frame=121)

        drawn = overlay.frames[0].points
        for point in drawn:
            assert point.x is not None
            assert point.y is not None
            back = image_to_frame_widths(np.array([point.x, point.y, 0.0]), filtered.geometry)
            expected = filtered[point.landmark].position[120]
            assert back[0] == pytest.approx(expected[0], abs=1e-12)
            assert back[1] == pytest.approx(expected[1], abs=1e-12)

    def test_the_conversion_runs_in_the_right_direction(self) -> None:
        """Drawing is IMAGE in, IMAGE out -- the aspect correction must cancel.

        Which means the useful assertion is not that a portrait frame and a
        square frame disagree: they do not, and should not, because the two
        conversions are inverse and the round trip returns what went in. What
        would break is applying the correction the *wrong way* on the return
        trip, which is the defect Phase 5 found by hand on a 16:9 frame -- a
        true 45-degree line reading as 29.4.

        So the guard is explicit: the drawn y is what the estimator reported,
        and is not what the forward conversion would have produced. On a square
        frame those two are the same number and the test would prove nothing,
        which is why every fixture here is portrait.
        """
        filtered = _filtered()
        overlay = _overlay(filtered, start_frame=120, end_frame=121)
        point = overlay.frames[0].points[int(Landmark.NOSE)]
        assert point.y is not None

        reported = _points(120 / FPS)[int(Landmark.NOSE)].y
        wrong_way = float(
            image_to_frame_widths(
                np.array([0.0, filtered[Landmark.NOSE].position[120][1], 0.0]), PORTRAIT
            )[1]
        )

        assert point.y == pytest.approx(reported, abs=1e-9)
        assert point.y != pytest.approx(wrong_way, abs=1e-3)

    def test_geometry_is_carried_so_a_canvas_can_size_itself(self) -> None:
        overlay = _overlay()

        assert overlay.geometry.width == PORTRAIT.width
        assert overlay.geometry.height == PORTRAIT.height


class TestWhatIsKnown:
    def test_a_supported_observed_point_is_drawn(self) -> None:
        point = _overlay().frames[0].points[int(Landmark.LEFT_WRIST)]

        assert point.state is OverlayState.OBSERVED
        assert point.x is not None
        assert point.y is not None

    def test_a_blocked_point_carries_no_position(self) -> None:
        """The invariant the contract states: null coordinates exactly when blocked.

        A long absence is left unfilled by the gap policy, and the overlay draws
        nothing there. Interpolating across it would hide the case Phase 4 found
        on the down-the-line clip, where motion blur lost the wrists for 1.92 s
        precisely when they were moving fastest.
        """
        filtered = _filtered(_sequence(undetected=range(100, 160)))
        overlay = build_overlay(
            filtered,
            video_path="/data/swing.mov",
            content_key=KEY,
            start_frame=120,
            end_frame=126,
        )

        blocked = [
            point
            for frame in overlay.frames
            for point in frame.points
            if point.state is OverlayState.BLOCKED
        ]
        assert blocked, "the gap policy no longer blocks here; this test measures nothing"
        assert all(point.x is None and point.y is None for point in blocked)

    def test_every_drawn_point_has_both_coordinates_or_neither(self) -> None:
        """Never half a point. A canvas drawing x from one frame and y from
        nothing would put the skeleton on the frame edge without saying so."""
        filtered = _filtered(_sequence(undetected=range(100, 160)))
        overlay = build_overlay(
            filtered,
            video_path="/data/swing.mov",
            content_key=KEY,
            start_frame=90,
            end_frame=170,
        )

        for frame in overlay.frames:
            for point in frame.points:
                assert (point.x is None) == (point.y is None)
                assert (point.x is None) == (point.state is OverlayState.BLOCKED)

    def test_visibility_is_finite_even_where_nothing_was_seen(self) -> None:
        """NaN would be a number no strict JSON parser accepts.

        The store writes NaN for a frame the estimator returned nothing on, and
        it has to become something before it reaches a wire.
        """
        filtered = _filtered(_sequence(undetected=range(100, 160)))
        overlay = build_overlay(
            filtered,
            video_path="/data/swing.mov",
            content_key=KEY,
            start_frame=100,
            end_frame=110,
        )

        values = [point.visibility for frame in overlay.frames for point in frame.points]
        assert values
        assert all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values)


class TestWhatIsSent:
    def test_every_landmark_by_default_in_ascending_order(self) -> None:
        overlay = _overlay()

        assert overlay.landmarks == list(Landmark)
        assert [point.landmark for point in overlay.frames[0].points] == list(Landmark)

    def test_a_subset_drops_the_edges_it_cannot_support(self) -> None:
        """An edge with an endpoint the caller has no position for is not sent.

        Otherwise a consumer drawing a subset receives a line to a point that is
        not in its own list, and has to discover the omission itself.
        """
        wanted = (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER, Landmark.LEFT_ELBOW)
        overlay = _overlay(landmarks=wanted)

        assert overlay.landmarks == list(wanted)
        assert all(set(pair) <= set(wanted) for pair in overlay.connections)
        assert (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER) in [
            tuple(pair) for pair in overlay.connections
        ]

    def test_connections_come_from_the_one_definition_of_a_skeleton(self) -> None:
        """Sent, not left to the consumer, so the app cannot form a second opinion
        about human anatomy. Phase 9's bone-length checks read the same tuple."""
        overlay = _overlay()

        assert [tuple(pair) for pair in overlay.connections] == list(POSE_CONNECTIONS)

    def test_timestamps_are_the_real_clock(self) -> None:
        """Slow motion is divided out before these times, matching every other
        duration in the system -- and therefore not where the frame sits in the
        file, which is what the player seeks by."""
        filtered = _filtered(slow_motion_factor=4.0)
        overlay = build_overlay(
            filtered,
            video_path="/data/swing.mov",
            content_key=KEY,
            start_frame=120,
            end_frame=121,
        )

        assert overlay.slow_motion_factor == 4.0
        assert overlay.frames[0].timestamp_s == pytest.approx(120 / FPS / 4.0)

    def test_an_unfiltered_landmark_is_refused_rather_than_skipped(self) -> None:
        """Asking for something that was not filtered is a caller error, and
        silently returning fewer points would surface as a skeleton missing a
        limb for no stated reason."""
        filtered = filter_sequence(
            _sequence(),
            FilterConfig(smoothing=SmoothingConfig(window_s=0.15)),
            space=LandmarkSpace.FRAME_WIDTHS,
            landmarks=(Landmark.LEFT_WRIST,),
        )

        with pytest.raises(OverlayError) as caught:
            build_overlay(
                filtered,
                video_path="/data/swing.mov",
                content_key=KEY,
                start_frame=0,
                end_frame=2,
            )

        assert "nose" in str(caught.value)


class TestClub:
    def test_nothing_looked_is_distinguished_from_nothing_found(self) -> None:
        """`club_tracked` answers "was the club asked for", not "was it found".

        A caller that asked and got a track where every frame was refused has
        evidence the club could not be seen; one that never asked has none. The
        overlay would collapse the two at the clip level if this flag followed
        the presence of a track.
        """
        assert _overlay().club_tracked is False
        assert _overlay(club_requested=True).club_tracked is True
        assert all(frame.shaft is None for frame in _overlay(club_requested=True).frames)

    def test_no_calibration_is_reported_as_no_calibration(self) -> None:
        """`undistorted` decides whether the skeleton is expected to sit on the
        pixels underneath it, which is a thing a reader can otherwise only
        discover by being confused."""
        assert _overlay().undistorted is False
