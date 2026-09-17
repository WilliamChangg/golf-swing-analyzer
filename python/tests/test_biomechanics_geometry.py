"""Geometry primitive tests, against angles worked out by hand.

Every expected value in this file comes from a triangle whose angles are known
without running the code: a 3-4-5 right triangle, an equilateral triangle, a
45-degree diagonal. That is the point of testing primitives this way -- a
tolerance agreed with the implementation proves the implementation agrees with
itself, while 3-4-5 giving 90 degrees is a fact about geometry that the code has
to meet rather than define.

The aspect-ratio conversion gets the most attention here, because it is the
correction with no visible failure mode. A distance computed without it is still
a plausible number, an angle still lands between 0 and 180, and nothing about
the output says the frame was not square.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from analyzer.biomechanics.geometry import (
    angle_between_deg,
    distance,
    foreshortening_angle_deg,
    interior_angle_deg,
    line_tilt_deg,
    midpoint,
    plane_coordinates,
    project_onto,
    tilt_from_vertical_deg,
)
from analyzer.contracts.pose import FrameGeometry

SQUARE = FrameGeometry(width=1000, height=1000)
PORTRAIT = FrameGeometry(width=1080, height=1920)  # aspect 16:9 on its side


def point(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float64)


class TestPlaneCoordinates:
    """The two corrections that have no visible failure mode."""

    def test_y_increases_upward_after_conversion(self) -> None:
        """Image y grows downward; every metric here is about high and low."""
        high_in_frame = plane_coordinates(point(0.5, 0.1), SQUARE)
        low_in_frame = plane_coordinates(point(0.5, 0.9), SQUARE)
        assert high_in_frame[1] > low_in_frame[1]

    def test_a_square_frame_leaves_coordinates_alone(self) -> None:
        converted = plane_coordinates(point(0.25, 0.25), SQUARE)
        assert converted[0] == pytest.approx(0.25)
        assert converted[1] == pytest.approx(0.75)

    def test_equal_pixel_distances_measure_equal_on_a_portrait_frame(self) -> None:
        """The defect the conversion exists to remove.

        108 pixels across and 108 pixels down are the same distance. In raw
        normalised coordinates on a 1080x1920 frame they are 0.1 and 0.05625,
        and every length, speed and angle built from them inherits that.
        """
        horizontal = distance(
            plane_coordinates(point(0.0, 0.5), PORTRAIT),
            plane_coordinates(point(0.1, 0.5), PORTRAIT),  # 108 px across
        )
        vertical = distance(
            plane_coordinates(point(0.5, 0.0), PORTRAIT),
            plane_coordinates(point(0.5, 108 / 1920), PORTRAIT),  # 108 px down
        )
        assert float(horizontal) == pytest.approx(float(vertical), rel=1e-12)

    def test_a_true_45_degree_line_reads_as_45_degrees(self) -> None:
        """And reads as 29.4 without the correction, which is the whole problem."""
        # 200 pixels right and 200 pixels up on a 1080x1920 frame.
        start = plane_coordinates(point(0.4, 0.5), PORTRAIT)
        end = plane_coordinates(point(0.4 + 200 / 1080, 0.5 - 200 / 1920), PORTRAIT)
        assert float(line_tilt_deg(start, end)) == pytest.approx(45.0, abs=1e-9)

        uncorrected = math.degrees(math.atan2(200 / 1920, 200 / 1080))
        assert uncorrected == pytest.approx(29.36, abs=0.01)

    def test_the_z_channel_is_dropped_rather_than_carried(self) -> None:
        """It is a single-camera depth estimate on an unstated scale."""
        converted = plane_coordinates(np.array([0.5, 0.5, 0.9]), SQUARE)
        assert converted.shape == (2,)

    def test_a_whole_trajectory_converts_at_once(self) -> None:
        trajectory = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        converted = plane_coordinates(trajectory, SQUARE)
        assert converted.shape == (3, 2)
        assert converted[1, 1] == pytest.approx(0.6)


class TestAngles:
    def test_a_3_4_5_triangle_has_a_right_angle(self) -> None:
        assert float(
            interior_angle_deg(point(0.0, 3.0), point(0.0, 0.0), point(4.0, 0.0))
        ) == pytest.approx(90.0)

    def test_an_equilateral_triangle_has_60_degree_corners(self) -> None:
        apex = point(0.5, math.sqrt(3) / 2)
        assert float(interior_angle_deg(point(0.0, 0.0), apex, point(1.0, 0.0))) == pytest.approx(
            60.0
        )

    def test_a_straight_line_through_the_vertex_is_180_degrees(self) -> None:
        """A straight leg, in the form every joint angle here takes."""
        assert float(
            interior_angle_deg(point(0.0, 1.0), point(0.0, 0.0), point(0.0, -1.0))
        ) == pytest.approx(180.0)

    def test_angle_between_is_unsigned(self) -> None:
        up, right = point(0.0, 1.0), point(1.0, 0.0)
        assert float(angle_between_deg(up, right)) == pytest.approx(90.0)
        assert float(angle_between_deg(right, up)) == pytest.approx(90.0)

    def test_parallel_vectors_give_zero_rather_than_nan(self) -> None:
        """`arccos` of a cosine rounded to 1.0000000000000002 is NaN."""
        vector = point(0.3, 0.7)
        assert float(angle_between_deg(vector, vector * 2.0)) == pytest.approx(0.0)

    def test_a_zero_length_vector_has_no_angle(self) -> None:
        assert np.isnan(float(angle_between_deg(point(0.0, 0.0), point(1.0, 0.0))))

    def test_a_missing_landmark_propagates_as_nan(self) -> None:
        assert np.isnan(
            float(interior_angle_deg(point(np.nan, 1.0), point(0.0, 0.0), point(1.0, 0.0)))
        )


class TestTilt:
    def test_vertical_is_zero(self) -> None:
        assert float(tilt_from_vertical_deg(point(0.0, 0.0), point(0.0, 1.0))) == pytest.approx(0.0)

    def test_leaning_right_is_positive(self) -> None:
        assert float(tilt_from_vertical_deg(point(0.0, 0.0), point(1.0, 1.0))) == pytest.approx(
            45.0
        )

    def test_leaning_left_is_negative(self) -> None:
        assert float(tilt_from_vertical_deg(point(0.0, 0.0), point(-1.0, 1.0))) == pytest.approx(
            -45.0
        )


class TestLineTilt:
    """A line has an orientation, not a direction, and the sign must say which."""

    def test_level_is_zero(self) -> None:
        assert float(line_tilt_deg(point(0.0, 0.5), point(1.0, 0.5))) == pytest.approx(0.0)

    def test_positive_means_the_right_hand_end_in_the_frame_is_higher(self) -> None:
        assert float(line_tilt_deg(point(0.0, 0.0), point(1.0, 1.0))) == pytest.approx(45.0)

    def test_the_argument_order_does_not_change_the_answer(self) -> None:
        """The defect the x-ordering replaced.

        Folding a vector angle onto (-90, 90] silently reverses any leftward
        vector, so the sign came to mean "the end point is higher" only while
        the end point was on the right of the frame. A player facing the camera
        has their right shoulder on the frame's left, which is precisely the
        case this system reads.
        """
        low_left, high_right = point(0.0, 0.0), point(1.0, 1.0)
        assert float(line_tilt_deg(low_left, high_right)) == pytest.approx(
            float(line_tilt_deg(high_right, low_left))
        )

    def test_an_anatomically_right_landmark_on_the_frames_left_is_read_by_position(self) -> None:
        """The real arrangement on a face-on clip, worked through.

        The player faces the camera, so their right shoulder is at the smaller
        x. With the right shoulder higher, the endpoint further right in the
        frame -- the left shoulder -- is the lower one, so the tilt is negative.
        """
        right_shoulder = point(0.45, 1.02)
        left_shoulder = point(0.58, 0.99)
        assert float(line_tilt_deg(left_shoulder, right_shoulder)) < 0.0

    def test_stays_within_ninety_degrees(self) -> None:
        for angle in np.linspace(-179.0, 179.0, 61):
            end = point(math.cos(math.radians(angle)), math.sin(math.radians(angle)))
            assert -90.0 <= float(line_tilt_deg(point(0.0, 0.0), end)) <= 90.0

    def test_does_not_jump_when_a_near_level_line_crosses_horizontal(self) -> None:
        """The +-180 discontinuity that would wrap a shoulder signal every swing."""
        just_above = float(line_tilt_deg(point(0.0, 0.0), point(-1.0, 0.001)))
        just_below = float(line_tilt_deg(point(0.0, 0.0), point(-1.0, -0.001)))
        assert abs(just_above - just_below) < 1.0


class TestForeshortening:
    def test_an_unshortened_segment_has_not_turned(self) -> None:
        assert float(foreshortening_angle_deg(1.0, 1.0)) == pytest.approx(0.0)

    def test_half_width_is_sixty_degrees(self) -> None:
        """cos(60 deg) = 0.5, by hand."""
        assert float(foreshortening_angle_deg(0.5, 1.0)) == pytest.approx(60.0)

    def test_a_vanished_segment_is_a_right_angle(self) -> None:
        assert float(foreshortening_angle_deg(0.0, 1.0)) == pytest.approx(90.0)

    @pytest.mark.parametrize(("turn", "cosine"), [(30.0, 0.8660254), (45.0, 0.7071068)])
    def test_recovers_the_angle_a_span_was_built_from(self, turn: float, cosine: float) -> None:
        assert float(foreshortening_angle_deg(cosine, 1.0)) == pytest.approx(turn, abs=1e-4)

    def test_a_span_wider_than_its_reference_clamps_to_zero(self) -> None:
        """Landmark noise pushes one frame past the maximum; that is not a negative turn."""
        assert float(foreshortening_angle_deg(1.2, 1.0)) == pytest.approx(0.0)

    def test_an_absent_reference_gives_nan(self) -> None:
        assert np.isnan(float(foreshortening_angle_deg(0.5, 0.0)))
        assert np.isnan(float(foreshortening_angle_deg(0.5, float("nan"))))

    def test_works_across_a_whole_series(self) -> None:
        spans = np.array([1.0, 0.5, 0.0])
        assert foreshortening_angle_deg(spans, 1.0) == pytest.approx([0.0, 60.0, 90.0])


class TestProjection:
    def test_names_the_side_a_point_sits_on(self) -> None:
        """How the trail shoulder is identified from where the hands are."""
        shoulder_line = point(1.0, 0.0)  # pointing towards the right of the frame
        assert float(project_onto(point(0.3, 0.9), shoulder_line)) > 0
        assert float(project_onto(point(-0.3, 0.9), shoulder_line)) < 0

    def test_a_point_over_the_middle_projects_to_nothing(self) -> None:
        assert float(project_onto(point(0.0, 1.0), point(1.0, 0.0))) == pytest.approx(0.0)

    def test_projection_is_independent_of_the_direction_vectors_length(self) -> None:
        offset = point(0.3, 0.4)
        assert float(project_onto(offset, point(2.0, 0.0))) == pytest.approx(
            float(project_onto(offset, point(1.0, 0.0)))
        )


class TestHelpers:
    def test_midpoint_is_halfway(self) -> None:
        assert midpoint(point(0.0, 0.0), point(1.0, 3.0)) == pytest.approx([0.5, 1.5])

    def test_distance_is_euclidean(self) -> None:
        assert float(distance(point(0.0, 0.0), point(3.0, 4.0))) == pytest.approx(5.0)
