"""The conventions the club layer would otherwise get silently wrong.

Two of them, and both fail without raising anything:

* **the sign of an angle** when a frame's y flips, which turns a backswing into a
  downswing and leaves every number looking reasonable;
* **the wrap of an angle series** through +-180, which a swing crosses exactly
  once and always in the downswing.

`analyzer/club/extract.py` documents that it recomputes a direction from the
converted endpoints rather than negating the detector's own, and that the two are
equal. This is the file that makes that claim checkable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from analyzer.club.geometry import (
    angle_deg,
    angle_difference_deg,
    angle_to_frame_widths,
    point_segment_distance_px,
    to_frame_widths,
    unwrap_deg,
)
from analyzer.contracts.pose import FrameGeometry
from analyzer.coordinates import frame_widths_to_pixels, pixels_to_frame_widths

# Deliberately not square, so a conversion that confuses width with height fails
# here rather than in the one place nothing checks.
OBLONG = FrameGeometry(width=1920, height=1080)


class TestTheFrameChange:
    @pytest.mark.parametrize(
        "point", [(0.0, 0.0), (1920.0, 1080.0), (960.0, 540.0), (100.0, 900.0)]
    )
    def test_pixels_round_trip_through_frame_widths(self, point: tuple[float, float]) -> None:
        converted = pixels_to_frame_widths(np.array(point), OBLONG)
        back = frame_widths_to_pixels(converted, OBLONG)
        assert back[0] == pytest.approx(point[0])
        assert back[1] == pytest.approx(point[1])

    def test_both_axes_divide_by_the_width(self) -> None:
        """Which is what makes frame widths isotropic, and IMAGE space not.

        A displacement of 100 px measures the same whichever way it points. The
        same displacement in IMAGE space would come out 1.78x larger vertically
        on this frame, and nothing about the number would look wrong.
        """
        origin = pixels_to_frame_widths(np.array([500.0, 500.0]), OBLONG)
        across = pixels_to_frame_widths(np.array([600.0, 500.0]), OBLONG)
        down = pixels_to_frame_widths(np.array([500.0, 600.0]), OBLONG)
        assert abs(across[0] - origin[0]) == pytest.approx(abs(down[1] - origin[1]))

    def test_y_points_up(self) -> None:
        """A pixel lower on the screen has a smaller frame-widths y."""
        high = pixels_to_frame_widths(np.array([0.0, 100.0]), OBLONG)
        low = pixels_to_frame_widths(np.array([0.0, 900.0]), OBLONG)
        assert high[1] > low[1]


class TestTheAngleConvention:
    @pytest.mark.parametrize("pixel_angle", [0.0, 30.0, 90.0, 179.0, -45.0, -120.0])
    def test_negating_the_pixel_angle_is_converting_the_endpoints(self, pixel_angle: float) -> None:
        """**The claim `extract.py` makes**, pinned to floating-point precision.

        The reported direction is recomputed from the converted endpoints,
        because only one of the two routes can be the definition and that is the
        one that cannot drift from the coordinates actually reported. This says
        the other route agrees, so the shorthand in the detector's own docstring
        is not a second convention.
        """
        grip_px = (900.0, 500.0)
        radians = math.radians(pixel_angle)
        tip_px = (grip_px[0] + 300.0 * math.cos(radians), grip_px[1] + 300.0 * math.sin(radians))

        grip = to_frame_widths(grip_px, OBLONG)
        tip = to_frame_widths(tip_px, OBLONG)
        recomputed = angle_deg(tip[0] - grip[0], tip[1] - grip[1])

        assert angle_difference_deg(
            recomputed, angle_to_frame_widths(pixel_angle)
        ) == pytest.approx(0.0, abs=1e-9)

    def test_a_shaft_pointing_up_the_screen_reports_a_positive_angle(self) -> None:
        """The sign that inverts a swing, stated once as a fact rather than a rule."""
        assert angle_to_frame_widths(-90.0) == pytest.approx(90.0)

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [(10.0, 350.0, 20.0), (350.0, 10.0, -20.0), (0.0, 180.0, -180.0), (45.0, 45.0, 0.0)],
    )
    def test_the_difference_takes_the_shortest_way_round(
        self, a: float, b: float, expected: float
    ) -> None:
        # Note the third case: two exactly opposite directions come back as -180
        # rather than +180, which is the half-open end of the interval and is
        # harmless because every caller reads the magnitude. Pinned so that a
        # future caller reading the sign finds this rather than discovering it.
        assert angle_difference_deg(a, b) == pytest.approx(expected)


class TestUnwrappingVersusFolding:
    def test_a_shaft_is_unwrapped_where_a_shoulder_line_would_be_folded(self) -> None:
        """The two operations are opposites and both live in this codebase now.

        A shoulder line has no direction, so `phases.signals` folds its angle onto
        a half turn. A shaft's two ends *are* distinguishable, because one of them
        is the grip — so folding it would destroy the information the pose layer
        supplied, and the fix for the +-180 crossing is to unwrap along time.
        """
        turning = np.array([160.0, 175.0, -170.0, -155.0, -140.0])
        unwrapped = unwrap_deg(turning)
        assert np.all(np.diff(unwrapped) > 0.0), "a steadily turning shaft must stay monotonic"
        assert np.allclose(np.diff(unwrapped), 15.0)

    def test_the_wrapped_series_would_have_reported_a_reversal(self) -> None:
        """What the unwrap is for, said as the failure it prevents."""
        turning = np.array([160.0, 175.0, -170.0, -155.0])
        assert np.diff(turning).min() < -300.0


class TestTheGripTest:
    def test_distance_is_to_the_segment_and_not_to_its_line(self) -> None:
        """A fence post is collinear with the hands far more often than it is near them."""
        far_along_the_line = point_segment_distance_px((0.0, 0.0), (100.0, 0.0), (200.0, 0.0))
        assert far_along_the_line == pytest.approx(100.0)

    def test_a_segment_through_the_point_is_at_zero(self) -> None:
        assert point_segment_distance_px((50.0, 0.0), (0.0, 0.0), (100.0, 0.0)) == pytest.approx(
            0.0
        )

    def test_a_perpendicular_offset_is_the_offset(self) -> None:
        assert point_segment_distance_px((50.0, 7.0), (0.0, 0.0), (100.0, 0.0)) == pytest.approx(
            7.0
        )

    def test_a_degenerate_segment_is_a_point(self) -> None:
        assert point_segment_distance_px((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)) == pytest.approx(5.0)
