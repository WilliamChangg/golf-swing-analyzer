"""Angles, distances and the one frame change the club layer performs.

Everything here is small and everything here is a convention, which is why it is
one module rather than three lines repeated in four places. Two of the three
mistakes this phase could make silently are in this file: the sign of an angle
when a frame's y flips, and the wrap of an angle series that passes through 180
degrees.

## The frame change

Detection happens on the pixel grid, because a Hough transform has no other
frame. Everything above Phase 3 measures in FRAME_WIDTHS. Pixels are isotropic
and frame widths are isotropic, so the change between them is a scale and a flip
of y -- no aspect correction, because there is no anisotropy to correct.

The consequence for angles is exactly one sign. A direction measured in pixels
has y increasing downward, so a shaft pointing up the screen has a *negative*
pixel angle and a *positive* frame-widths one. `angle_to_frame_widths` is that
negation, it is applied once on the way out of the detector, and it is the reason
`ShaftCandidate.angle_deg` documents its convention in its own docstring.

## The wrap

A swing carries the shaft through more than a full turn: address to the top is
most of a half turn, the top to the finish is most of another, and the direction
passes through 180 degrees somewhere in the downswing on every clip. An angle
reported on (-180, 180] therefore jumps by 360 there, and anything that
differences it reads that jump as an angular rate of tens of thousands of degrees
per second -- past every bound, in the middle of the only phase that matters.

Phase 4 met the mirror image of this and fixed it the other way: a shoulder line
has no direction, so its angle is *folded* onto a half turn. A shaft does have a
direction, so folding it would be wrong, and the fix is to **unwrap** along time
instead. `unwrap_deg` does it once and the tracker uses nothing else.

Both operations exist in this codebase now, they are opposites, and a reader who
confuses them gets a plausible number either way -- which is why each says in its
own docstring which kind of object it is for.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.pose import FrameGeometry
from analyzer.coordinates import pixels_to_frame_widths


def angle_deg(dx: float, dy: float) -> float:
    """Direction of a vector, in degrees on (-180, 180].

    Frame-agnostic: it reports the angle of whatever vector it is given, in
    whatever frame that vector's components are in. The caller owns the meaning
    of the sign, which is the point of keeping this free of any y convention.
    """
    return math.degrees(math.atan2(dy, dx))


def angle_to_frame_widths(pixel_angle_deg: float) -> float:
    """A direction measured on the pixel grid, in the frame every measurement uses.

    One negation, because the two frames differ only in the sign of y. Applied on
    the way out of the detector and nowhere else, so no two call sites can
    disagree about which way a positive shaft angle points.
    """
    return -pixel_angle_deg


def angle_difference_deg(a: float, b: float) -> float:
    """The signed difference `a - b`, wrapped onto **[-180, 180)**.

    For comparing two directions measured independently -- a candidate against a
    prediction, say -- where neither belongs to a series and unwrapping is not
    available. The shortest way round is the right answer there, because two
    directions 350 degrees apart are 10 degrees apart.

    Note the half-open interval, which is the opposite end from `angle_deg`'s
    (-180, 180]: two exactly opposite directions come back as -180 rather than
    +180. Which sign that case takes is arbitrary, and it is harmless here
    because every caller reads the magnitude -- an exact reversal is the largest
    disagreement there is either way.
    """
    return (a - b + 180.0) % 360.0 - 180.0


def unwrap_deg(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove the 360-degree jumps from a *series* of directions, in place-safe form.

    For a shaft, which has a direction and turns continuously through a swing.
    **Not** for a shoulder or hip line, which has an orientation and is folded
    onto a half turn instead -- see `phases.signals._line_angle_deg`. The two
    treatments look similar and mean opposite things.

    NaNs are preserved and do not break the unwrap: each unbroken run of finite
    values is unwrapped on its own, because a gap in the series is a gap in the
    evidence and carrying a phase across it would invent a number of turns the
    clip never showed.
    """
    result = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(result)
    if not np.any(finite):
        return result

    start = 0
    size = result.size
    while start < size:
        if not finite[start]:
            start += 1
            continue
        end = start
        while end < size and finite[end]:
            end += 1
        if end - start > 1:
            result[start:end] = np.degrees(np.unwrap(np.radians(result[start:end])))
        start = end
    return result


def point_segment_distance_px(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    """Distance from a point to a line *segment*, in the units of its inputs.

    The segment rather than the infinite line it lies on, and the difference is
    the whole of the grip test. A fence post four metres behind the player is
    collinear with the hands in plenty of frames; its nearest *point* is nowhere
    near them. Measuring to the infinite line would accept it.
    """
    px, py = point
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    span = dx * dx + dy * dy
    if span <= 0.0:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / span))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def to_frame_widths(point_px: tuple[float, float], geometry: FrameGeometry) -> tuple[float, float]:
    """One pixel coordinate in frame widths, as a plain pair."""
    converted = pixels_to_frame_widths(np.asarray(point_px, dtype=np.float64), geometry)
    return float(converted[0]), float(converted[1])
