"""Reference frame tests: conversions, round trips, and what is refused.

The conversions in `analyzer/coordinates.py` are the ones with no visible
failure mode. A distance computed in the wrong frame is still a plausible
number, an angle still lands between 0 and 180, and nothing about the output
says the frame was not square or that y was pointing the other way. So they are
tested three ways:

* against **facts**, not tolerances -- equal pixel displacements must measure
  equal, and a true 45 degree line must read 45;
* by **round trip**, which catches a correction applied twice or in the wrong
  order where a one-way check cannot; and
* by what they **refuse**, since a frame this build cannot produce has to say so
  rather than return something.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from analyzer.contracts.pose import (
    STORED_SPACES,
    UNREACHABLE_SPACES,
    FrameGeometry,
    LandmarkSpace,
)
from analyzer.coordinates import (
    CoordinateError,
    convert,
    frame_widths_to_image,
    frame_widths_to_pixels,
    image_to_frame_widths,
    image_to_pixels,
    require_reachable,
)

SQUARE = FrameGeometry(width=1000, height=1000)
PORTRAIT = FrameGeometry(width=1080, height=1920)
LANDSCAPE = FrameGeometry(width=1920, height=1080)


def point(x: float, y: float, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


class TestImageToFrameWidths:
    """The two corrections, applied together and exactly once."""

    def test_y_increases_upward_after_conversion(self) -> None:
        """Image y grows downward; every rule above this is about high and low."""
        high_in_frame = image_to_frame_widths(point(0.5, 0.1), SQUARE)
        low_in_frame = image_to_frame_widths(point(0.5, 0.9), SQUARE)
        assert high_in_frame[1] > low_in_frame[1]

    def test_a_square_frame_still_flips_y(self) -> None:
        """The scale is the identity there; the flip is not."""
        converted = image_to_frame_widths(point(0.25, 0.25), SQUARE)
        assert converted[0] == pytest.approx(0.25)
        assert converted[1] == pytest.approx(0.75)

    def test_x_is_never_touched(self) -> None:
        for geometry in (SQUARE, PORTRAIT, LANDSCAPE):
            assert image_to_frame_widths(point(0.31, 0.62), geometry)[0] == pytest.approx(0.31)

    def test_equal_pixel_distances_measure_equal_on_a_portrait_frame(self) -> None:
        """The defect the conversion exists to remove.

        108 pixels across and 108 pixels down are the same distance. In raw
        normalised coordinates on a 1080x1920 frame they are 0.1 and 0.05625,
        and every length, speed and angle built from them inherits that.
        """
        across = image_to_frame_widths(point(0.1, 0.5), PORTRAIT) - image_to_frame_widths(
            point(0.0, 0.5), PORTRAIT
        )
        down = image_to_frame_widths(point(0.5, 108 / 1920), PORTRAIT) - image_to_frame_widths(
            point(0.5, 0.0), PORTRAIT
        )
        assert float(np.linalg.norm(across)) == pytest.approx(
            float(np.linalg.norm(down)), rel=1e-12
        )

    def test_a_true_45_degree_line_reads_as_45_degrees(self) -> None:
        """And reads as 29.4 without the correction, which is the whole problem."""
        start = image_to_frame_widths(point(0.4, 0.5), PORTRAIT)
        end = image_to_frame_widths(point(0.4 + 200 / 1080, 0.5 - 200 / 1920), PORTRAIT)
        delta = end - start
        assert math.degrees(math.atan2(delta[1], delta[0])) == pytest.approx(45.0, abs=1e-9)

        uncorrected = math.degrees(math.atan2(200 / 1920, 200 / 1080))
        assert uncorrected == pytest.approx(29.36, abs=0.01)

    def test_the_z_channel_passes_through_unchanged(self) -> None:
        """MediaPipe's z is already on the scale of x, so it is already frame widths."""
        assert image_to_frame_widths(point(0.5, 0.5, -0.17), PORTRAIT)[2] == pytest.approx(-0.17)

    def test_a_whole_trajectory_converts_at_once(self) -> None:
        trajectory = np.array([[0.1, 0.2, 0.0], [0.3, 0.4, 0.0], [0.5, 0.6, 0.0]])
        converted = image_to_frame_widths(trajectory, SQUARE)
        assert converted.shape == (3, 3)
        assert converted[1, 1] == pytest.approx(0.6)

    def test_nan_survives_the_conversion(self) -> None:
        """An undetected frame must not become a coordinate at the origin."""
        converted = image_to_frame_widths(point(np.nan, np.nan), PORTRAIT)
        assert np.all(np.isnan(converted[:2]))

    def test_the_input_is_not_modified(self) -> None:
        original = point(0.3, 0.7, 0.1)
        copy = original.copy()
        image_to_frame_widths(original, PORTRAIT)
        assert original == pytest.approx(copy)


class TestRoundTrips:
    """A correction applied twice, or in the wrong order, only shows up here."""

    @pytest.mark.parametrize("geometry", [SQUARE, PORTRAIT, LANDSCAPE])
    def test_image_survives_a_round_trip(self, geometry: FrameGeometry) -> None:
        original = np.array([[0.1, 0.2, 0.3], [0.9, 0.05, -0.4], [0.5, 0.5, 0.0]])
        there = image_to_frame_widths(original, geometry)
        back = frame_widths_to_image(there, geometry)
        assert back == pytest.approx(original, abs=1e-12)

    @pytest.mark.parametrize("geometry", [SQUARE, PORTRAIT, LANDSCAPE])
    def test_frame_widths_survives_a_round_trip(self, geometry: FrameGeometry) -> None:
        original = np.array([[0.25, 1.1, 0.0], [0.75, 0.2, 0.1]])
        back = image_to_frame_widths(frame_widths_to_image(original, geometry), geometry)
        assert back == pytest.approx(original, abs=1e-12)

    def test_convert_round_trips_through_the_dispatcher(self) -> None:
        original = np.array([[0.2, 0.8, 0.0]])
        there = convert(original, LandmarkSpace.IMAGE, LandmarkSpace.FRAME_WIDTHS, PORTRAIT)
        back = convert(there, LandmarkSpace.FRAME_WIDTHS, LandmarkSpace.IMAGE, PORTRAIT)
        assert back == pytest.approx(original, abs=1e-12)

    def test_converting_to_the_same_frame_changes_nothing(self) -> None:
        original = np.array([[0.2, 0.8, 0.0]])
        same = convert(original, LandmarkSpace.IMAGE, LandmarkSpace.IMAGE, PORTRAIT)
        assert same == pytest.approx(original)


class TestPixels:
    """Where the conversion becomes checkable by eye."""

    def test_both_axes_scale_by_the_width(self) -> None:
        """Which is what makes a frame-widths distance isotropic in pixels too."""
        origin = frame_widths_to_pixels(np.array([0.0, 0.0]), PORTRAIT)
        across = frame_widths_to_pixels(np.array([0.1, 0.0]), PORTRAIT)
        up = frame_widths_to_pixels(np.array([0.0, 0.1]), PORTRAIT)

        assert float(across[0] - origin[0]) == pytest.approx(108.0)
        assert float(origin[1] - up[1]) == pytest.approx(108.0)

    def test_the_bottom_left_of_the_frame_is_the_origin(self) -> None:
        pixels = frame_widths_to_pixels(np.array([0.0, 0.0]), PORTRAIT)
        assert pixels == pytest.approx([0.0, 1920.0])

    def test_a_landmark_lands_in_the_same_pixel_by_either_route(self) -> None:
        """The check the overlay script performs visually, made exact."""
        image = point(0.42, 0.61, 0.0)
        direct = image_to_pixels(image, PORTRAIT)
        through_frame_widths = frame_widths_to_pixels(
            image_to_frame_widths(image, PORTRAIT), PORTRAIT
        )
        assert through_frame_widths == pytest.approx(direct, abs=1e-9)


class TestRefusals:
    """Frames that are named so a later phase adds the capability, not the concept."""

    @pytest.mark.parametrize("space", sorted(UNREACHABLE_SPACES, key=lambda s: s.value))
    def test_an_unreachable_frame_is_refused(self, space: LandmarkSpace) -> None:
        with pytest.raises(CoordinateError, match=space.value):
            require_reachable(space)

    def test_the_refusal_names_what_supplies_the_frame(self) -> None:
        """Not a phase number: the thing that produces it, or the thing it lacks.

        CAMERA is produced by this build and is still refused here, because it is
        not a *conversion* of one clip's landmarks -- it is a measurement made
        from two of them. WORLD is refused because nothing measures a gravity
        direction or a target line, which is a fact about the capture rather than
        about a phase not having happened yet.
        """
        with pytest.raises(CoordinateError, match="triangulated from two calibrated views"):
            require_reachable(LandmarkSpace.CAMERA)
        with pytest.raises(CoordinateError, match="gravity direction and a target line"):
            require_reachable(LandmarkSpace.WORLD)

    def test_a_reachable_frame_passes(self) -> None:
        for space in (LandmarkSpace.IMAGE, LandmarkSpace.FRAME_WIDTHS, LandmarkSpace.HIP_LOCAL):
            require_reachable(space)

    def test_hip_local_is_not_reachable_from_a_picture(self) -> None:
        """It would mean recovering the depth the picture lost."""
        with pytest.raises(CoordinateError, match="depth the picture lost"):
            convert(
                np.array([[0.5, 0.5, 0.0]]),
                LandmarkSpace.IMAGE,
                LandmarkSpace.HIP_LOCAL,
                PORTRAIT,
            )

    def test_camera_space_is_a_measurement_and_never_a_conversion(self) -> None:
        """Even though this build produces CAMERA, no conversion reaches it.

        One picture cannot become three dimensions by arithmetic. It takes a
        second calibrated view of the same instant, which is a measurement, and
        `convert` is not where a measurement lives.
        """
        with pytest.raises(CoordinateError, match="triangulated from two calibrated views"):
            convert(
                np.array([[0.5, 0.5, 0.0]]),
                LandmarkSpace.IMAGE,
                LandmarkSpace.CAMERA,
                PORTRAIT,
            )


class TestVocabulary:
    def test_only_the_estimator_s_own_frames_are_stored(self) -> None:
        """FRAME_WIDTHS is derived on read, so the store never grows a column for it."""
        assert set(STORED_SPACES) == {LandmarkSpace.IMAGE, LandmarkSpace.HIP_LOCAL}

    def test_every_frame_is_either_stored_derivable_or_explained(self) -> None:
        """A new member cannot be added without deciding which of the three it is."""
        derived = {LandmarkSpace.FRAME_WIDTHS}
        accounted = set(STORED_SPACES) | derived | set(UNREACHABLE_SPACES)
        assert accounted == set(LandmarkSpace)
